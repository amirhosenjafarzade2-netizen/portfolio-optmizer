import numpy as np
import random

def validate_correlation_matrix(corr):
    if not np.allclose(corr, corr.T):
        raise ValueError("Correlation matrix must be symmetric")
    eigenvalues = np.linalg.eigvals(corr)
    if np.any(eigenvalues < -1e-10):
        raise ValueError("Correlation matrix must be positive semi-definite")

def optimize_portfolio(method, expected_returns, volatilities, correlations, dividend_yields, inflation, tax_rate, risk_free_rate, iterations, use_sharpe, use_inflation, use_tax_rate):
    num_assets = len(expected_returns)
    
    # Validate inputs
    if np.any(volatilities <= 0):
        raise ValueError("Volatilities must be positive")
    validate_correlation_matrix(correlations)
    
    cov_matrix = np.diag(volatilities) @ correlations @ np.diag(volatilities)
    
    if method == "Monte Carlo":
        weights = monte_carlo_optimize(expected_returns, cov_matrix, iterations, num_assets, risk_free_rate)
    elif method == "Genetic Algorithm":
        weights = genetic_algorithm_optimize(expected_returns, cov_matrix, iterations, num_assets, risk_free_rate)
    elif method == "Gradient Descent (Mean-Variance)":
        weights = gradient_descent_optimize(expected_returns, cov_matrix, num_assets, risk_free_rate)
    else:
        raise ValueError("Unknown method")
    
    # Calculate metrics
    metrics = {}
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) + 1e-10)
    portfolio_dividend = np.dot(weights, dividend_yields)
    
    metrics['Portfolio Return'] = portfolio_return
    metrics['Portfolio Volatility'] = portfolio_volatility
    metrics['Portfolio Dividend Yield'] = portfolio_dividend
    
    if use_inflation:
        metrics['Real Return'] = portfolio_return - inflation
    if use_tax_rate:
        after_tax_return = portfolio_return * (1 - tax_rate) + portfolio_dividend * tax_rate
        metrics['After-Tax Return'] = after_tax_return
    if use_sharpe:
        metrics['Sharpe Ratio'] = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
    
    return weights, metrics

def monte_carlo_optimize(returns, cov, iterations, num_assets, rf_rate):
    best_sharpe = -np.inf
    best_weights = None
    for _ in range(iterations):
        weights = np.random.dirichlet(np.ones(num_assets))
        port_ret = np.dot(weights, returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)) + 1e-10)
        sharpe = (port_ret - rf_rate) / port_vol if port_vol > 0 else -np.inf
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
    return best_weights

def genetic_algorithm_optimize(returns, cov, iterations, num_assets, rf_rate):
    population_size = 50  # Reduced from 100
    population = [np.random.dirichlet(np.ones(num_assets)) for _ in range(population_size)]
    
    def fitness(w):
        port_ret = np.dot(w, returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)) + 1e-10)
        return (port_ret - rf_rate) / port_vol if port_vol > 0 else -np.inf
    
    for _ in range(iterations // population_size):
        fitness_scores = [fitness(w) for w in population]
        parents_indices = np.argsort(fitness_scores)[-population_size//2:]
        parents = [population[i] for i in parents_indices]
        
        new_population = []
        for _ in range(population_size):
            p1, p2 = random.choices(parents, k=2)
            child = (p1 + p2) / 2
            child /= child.sum() + 1e-10
            if random.random() < 0.1:  # Mutation
                child += np.random.normal(0, 0.02, num_assets)
                child = np.clip(child, 0, 1)
                child /= child.sum() + 1e-10
            new_population.append(child)
        population = new_population
    
    best_idx = np.argmax([fitness(w) for w in population])
    return population[best_idx]

def gradient_descent_optimize(returns, cov, num_assets, rf_rate, max_iter=1000, lr=0.01):
    weights = np.ones(num_assets) / num_assets
    
    for iteration in range(max_iter):
        port_ret = np.dot(weights, returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)) + 1e-10)
        
        if port_vol < 1e-8:
            break
            
        sharpe = (port_ret - rf_rate) / port_vol
        grad_ret = returns
        grad_vol = np.dot(cov, weights) / port_vol
        grad_sharpe = (grad_ret * port_vol - (port_ret - rf_rate) * grad_vol) / (port_vol ** 2)
        
        weights = weights + lr * grad_sharpe
        weights = np.clip(weights, 0, None)
        weights = weights / (weights.sum() + 1e-10)
    
    return weights
