# optimizer.py
import numpy as np
from scipy.optimize import minimize
import random

def optimize_portfolio(method, expected_returns, volatilities, correlations, dividend_yields, inflation, tax_rate, risk_free_rate, iterations, use_sharpe, use_inflation, use_tax_rate):
    num_assets = len(expected_returns)
    cov_matrix = np.diag(volatilities) @ correlations @ np.diag(volatilities)
    
    if method == "Monte Carlo":
        weights = monte_carlo_optimize(expected_returns, cov_matrix, iterations, num_assets)
    elif method == "Genetic Algorithm":
        weights = genetic_algorithm_optimize(expected_returns, cov_matrix, iterations, num_assets)
    elif method == "SciPy Optimize (Mean-Variance)":
        weights = scipy_optimize(expected_returns, cov_matrix, num_assets)
    else:
        raise ValueError("Unknown method")
    
    # Calculate metrics
    metrics = {}
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    portfolio_dividend = np.dot(weights, dividend_yields)
    
    metrics['Portfolio Return'] = portfolio_return
    metrics['Portfolio Volatility'] = portfolio_volatility
    metrics['Portfolio Dividend Yield'] = portfolio_dividend
    
    if use_inflation:
        metrics['Real Return'] = portfolio_return - inflation
    if use_tax_rate:
        after_tax_return = portfolio_return * (1 - tax_rate) + portfolio_dividend * tax_rate  # Simplified tax on gains, dividends taxed differently
        metrics['After-Tax Return'] = after_tax_return
    if use_sharpe:
        metrics['Sharpe Ratio'] = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Other metrics: Suggest adding Max Drawdown, but needs historical data; for now, skip.
    # Beta: needs market data, skip.
    
    return weights, metrics

def monte_carlo_optimize(returns, cov, iterations, num_assets):
    best_sharpe = -np.inf
    best_weights = None
    for _ in range(iterations):
        weights = np.random.dirichlet(np.ones(num_assets))
        port_ret = np.dot(weights, returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe = port_ret / port_vol  # Simplified, no rf
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
    return best_weights

def genetic_algorithm_optimize(returns, cov, iterations, num_assets):
    # Simple GA implementation
    population_size = 100
    population = [np.random.dirichlet(np.ones(num_assets)) for _ in range(population_size)]
    
    for _ in range(iterations // population_size):
        fitness = [np.dot(w, returns) / np.sqrt(np.dot(w.T, np.dot(cov, w))) for w in population]
        parents = [population[i] for i in np.argsort(fitness)[-population_size//2:]]
        
        new_population = []
        for _ in range(population_size):
            p1, p2 = random.choices(parents, k=2)
            child = (p1 + p2) / 2
            child /= child.sum()  # Normalize
            if random.random() < 0.01:  # Mutation
                child += np.random.normal(0, 0.01, num_assets)
                child = np.clip(child, 0, 1)
                child /= child.sum()
            new_population.append(child)
        population = new_population
    
    best_idx = np.argmax([np.dot(w, returns) / np.sqrt(np.dot(w.T, np.dot(cov, w))) for w in population])
    return population[best_idx]

def scipy_optimize(returns, cov, num_assets):
    def objective(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))  # Minimize volatility for max return, but here target max sharpe simplified
    
    def constraint_return(weights):
        return np.dot(weights, returns) - np.max(returns) * 0.8  # Arbitrary target
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(num_assets)]
    result = minimize(objective, np.ones(num_assets)/num_assets, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x
