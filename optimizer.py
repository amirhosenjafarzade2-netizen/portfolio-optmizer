import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from joblib import Parallel, delayed

class PortfolioOptimizer:
    """Class to optimize portfolio weights using multiple methods to maximize Sharpe Ratio."""
    def __init__(self, returns, cov_matrix, risk_free_rate, transaction_costs, min_weights, max_weights, 
                 dividend_yields, tax_rates, inflation_rate, use_transaction_costs, use_weight_constraints, 
                 use_dividends, use_taxes, use_inflation, allow_short_selling, pop_size=50):
        """Initialize optimizer with financial parameters and configuration flags."""
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.transaction_costs = transaction_costs if use_transaction_costs else np.zeros_like(returns)
        self.min_weights = min_weights if use_weight_constraints else np.full_like(returns, -0.5 if allow_short_selling else 0.0)
        self.max_weights = max_weights if use_weight_constraints else np.full_like(returns, 1.5 if allow_short_selling else 1.0)
        self.dividend_yields = dividend_yields if use_dividends else np.zeros_like(returns)
        self.tax_rates = tax_rates if use_taxes else np.zeros_like(returns)
        self.inflation_rate = inflation_rate if use_inflation else 0.0
        self.allow_short_selling = allow_short_selling
        self.pop_size = pop_size
        self.population = [self._initialize_weights() for _ in range(pop_size)]
        self.best_sharpe = -float('inf')
        self.best_weights = None
        self.stagnation_count = 0
        self.use_transaction_costs = use_transaction_costs
        self.use_weight_constraints = use_weight_constraints
        self.use_dividends = use_dividends
        self.use_taxes = use_taxes
        self.use_inflation = use_inflation

    def _initialize_weights(self):
        """Generate initial weights respecting constraints."""
        weights = np.random.dirichlet(np.ones(len(self.returns)))
        weights = np.clip(weights, self.min_weights, self.max_weights)
        return weights / weights.sum()

    def fitness(self, weights):
        """Calculate Sharpe Ratio adjusted for dividends, taxes, transaction costs, and inflation."""
        portfolio_return = np.dot(weights, self.returns + self.dividend_yields)
        tax_impact = np.dot(weights, (self.returns + self.dividend_yields) * self.tax_rates)
        cost_penalty = np.sum(np.abs(weights) * self.transaction_costs)
        adjusted_return = portfolio_return - tax_impact - cost_penalty - self.inflation_rate
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return (adjusted_return - (self.risk_free_rate - self.inflation_rate)) / portfolio_vol if portfolio_vol > 0 else -float('inf')

    def calculate_var_cvar(self, weights, confidence_level=0.95):
        """Calculate Value-at-Risk and Conditional Value-at-Risk at 95% confidence."""
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        portfolio_return = np.dot(weights, self.returns + self.dividend_yields)
        z_score = norm.ppf(1 - confidence_level)
        var = portfolio_return + z_score * portfolio_vol
        cvar = portfolio_return + (norm.pdf(z_score) / (1 - confidence_level)) * portfolio_vol
        return var, cvar

    def select(self, fitnesses):
        """Tournament selection: pick best from random subsets."""
        selected = []
        for _ in range(self.pop_size):
            tournament_indices = np.random.choice(self.pop_size, size=5, replace=False)
            tournament_fitness = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx])
        return selected

    def crossover(self, parent1, parent2):
        """Blend crossover to create a child portfolio."""
        alpha = np.random.rand()
        child = alpha * parent1 + (1 - alpha) * parent2
        child = np.clip(child, self.min_weights, self.max_weights)
        return child / child.sum()

    def mutate(self):
        """Mutate population by adjusting weights with low probability."""
        for i in range(len(self.population)):
            if np.random.rand() < 0.1:
                idx = np.random.randint(len(self.population[i]))
                self.population[i][idx] += np.random.uniform(-0.1, 0.1)
                self.population[i] = np.clip(self.population[i], self.min_weights, self.max_weights)
                self.population[i] /= self.population[i].sum() or 1.0

    def optimize_ga(self, generations=100, progress_bar=None):
        """Run genetic algorithm with early stopping and parallel fitness evaluation."""
        for gen in range(generations):
            fitnesses = Parallel(n_jobs=2)(delayed(self.fitness)(w) for w in self.population)
            current_best = max(fitnesses)
            if current_best > self.best_sharpe:
                self.best_sharpe = current_best
                self.best_weights = self.population[np.argmax(fitnesses)].copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            if self.stagnation_count >= 20:
                break
            if progress_bar:
                progress_bar.progress((gen + 1) / generations)
            selected = self.select(fitnesses)
            self.population = [self.crossover(selected[i], selected[i+1]) for i in range(0, len(selected), 2)]
            self.mutate()
        var, cvar = self.calculate_var_cvar(self.best_weights)
        return self.best_weights, self.best_sharpe, var, cvar

    def optimize_slsqp(self):
        """Optimize using SLSQP to maximize Sharpe Ratio."""
        def objective(weights):
            return -self.fitness(weights)
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weights[i], self.max_weights[i]) for i in range(len(self.returns))]
        initial_guess = np.ones(len(self.returns)) / len(self.returns)
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        var, cvar = self.calculate_var_cvar(result.x)
        return result.x, -result.fun, var, cvar

    def optimize_monte_carlo(self, num_simulations=5000):
        """Optimize using Monte Carlo simulation with parallel evaluation."""
        weights_list = [self._initialize_weights() for _ in range(num_simulations)]
        fitnesses = Parallel(n_jobs=2)(delayed(self.fitness)(w) for w in weights_list)
        best_idx = np.argmax(fitnesses)
        best_weights = weights_list[best_idx]
        best_sharpe = fitnesses[best_idx]
        var, cvar = self.calculate_var_cvar(best_weights)
        return best_weights, best_sharpe, var, cvar

    @classmethod
    def suggest_ideal_metrics(cls, num_assets, time_horizon, risk_free_rate, optimize_returns, optimize_vols, optimize_corrs,
                             asset_classes=None, use_dividends=False, use_taxes=False, use_inflation=False,
                             inflation_rate=0.0, use_transaction_costs=False, use_weight_constraints=False,
                             allow_short_selling=False):
        """Suggest ideal realistic metrics for assets to maximize compounded wealth over time horizon."""
        class_ranges = {
            "Stocks": {"return": (0.06, 0.15), "vol": (0.15, 0.30)},
            "Bonds": {"return": (0.02, 0.05), "vol": (0.05, 0.10)},
            "REITs": {"return": (0.04, 0.10), "vol": (0.10, 0.20)},
            "Crypto": {"return": (0.10, 0.30), "vol": (0.40, 0.80)},
            "Gold": {"return": (0.02, 0.08), "vol": (0.10, 0.20)},
            "Commodities": {"return": (0.03, 0.09), "vol": (0.15, 0.25)},
            "Real Estate": {"return": (0.05, 0.10), "vol": (0.10, 0.15)}
        }
        asset_classes = asset_classes or ["Stocks"] * num_assets
        return_lows = [class_ranges[ac]["return"][0] if optimize_returns else 0.08 for ac in asset_classes]
        return_highs = [class_ranges[ac]["return"][1] if optimize_returns else 0.12 for ac in asset_classes]
        vol_lows = [class_ranges[ac]["vol"][0] if optimize_vols else 0.15 for ac in asset_classes]
        vol_highs = [class_ranges[ac]["vol"][1] if optimize_vols else 0.25 for ac in asset_classes]
        corr_low, corr_high = (-0.3, 0.8) if optimize_corrs else (0.0, 0.3)
        
        pop_size = 20
        generations = 20
        dim = num_assets
        corr_size = dim * (dim - 1) // 2
        
        def generate_metrics():
            rets = [np.random.uniform(r_low, r_high) for r_low, r_high in zip(return_lows, return_highs)]
            vols = [np.random.uniform(v_low, v_high) for v_low, v_high in zip(vol_lows, vol_highs)]
            corrs = np.eye(dim)
            for i in range(dim):
                for j in range(i+1, dim):
                    corrs[i,j] = corrs[j,i] = np.random.uniform(corr_low, corr_high)
            return np.array(rets), np.array(vols), corrs
        
        def metrics_fitness(rets, vols, corrs):
            cov = np.outer(vols, vols) * corrs
            transaction_costs = np.zeros(dim) if use_transaction_costs else np.zeros(dim)
            min_weights = np.full(dim, 0.0)
            max_weights = np.full(dim, 1.0)
            dividend_yields = np.zeros(dim) if use_dividends else np.zeros(dim)
            tax_rates = np.zeros(dim) if use_taxes else np.zeros(dim)
            
            opt = cls(rets, cov, risk_free_rate, transaction_costs, min_weights, max_weights,
                      dividend_yields, tax_rates, inflation_rate, use_transaction_costs, use_weight_constraints,
                      use_dividends, use_taxes, use_inflation, allow_short_selling)
            _, sharpe, _, _ = opt.optimize_ga(generations=20)
            adj_return = sharpe * np.sqrt(np.dot(opt.best_weights.T, np.dot(cov, opt.best_weights))) + (risk_free_rate - inflation_rate)
            wealth = (1 + adj_return) ** time_horizon
            return wealth
        
        population = [generate_metrics() for _ in range(pop_size)]
        best_wealth = -np.inf
        best_metrics = None
        
        for gen in range(generations):
            fitnesses = Parallel(n_jobs=2)(delayed(metrics_fitness)(*ind) for ind in population)
            best_idx = np.argmax(fitnesses)
            if fitnesses[best_idx] > best_wealth:
                best_wealth = fitnesses[best_idx]
                best_metrics = population[best_idx]
            
            selected = [population[np.random.choice(pop_size, p=fitnesses/np.sum(fitnesses))] for _ in range(pop_size)]
            new_pop = []
            for i in range(0, pop_size, 2):
                p1, p2 = selected[i], selected[i+1]
                alpha = np.random.rand()
                child_rets = alpha * p1[0] + (1 - alpha) * p2[0]
                child_vols = alpha * p1[1] + (1 - alpha) * p2[1]
                child_corrs = alpha * p1[2] + (1 - alpha) * p2[2]
                if np.random.rand() < 0.1:
                    child_rets += np.random.uniform(-0.01, 0.01, dim)
                    child_rets = np.clip(child_rets, return_lows, return_highs)
                    child_vols += np.random.uniform(-0.01, 0.01, dim)
                    child_vols = np.clip(child_vols, vol_lows, vol_highs)
                    for ii in range(dim):
                        for jj in range(ii+1, dim):
                            child_corrs[ii,jj] += np.random.uniform(-0.1, 0.1)
                            child_corrs[ii,jj] = np.clip(child_corrs[ii,jj], corr_low, corr_high)
                            child_corrs[jj,ii] = child_corrs[ii,jj]
                new_pop.append((child_rets, child_vols, child_corrs))
            population = new_pop
        
        rets, vols, corrs = best_metrics
        data = {
            "Asset": [f"{asset_classes[i]}" for i in range(num_assets)],
            "Return": rets,
            "Volatility": vols,
        }
        for j in range(num_assets):
            data[f"Corr_to_Asset{j+1}"] = corrs[:, j]
        return data
