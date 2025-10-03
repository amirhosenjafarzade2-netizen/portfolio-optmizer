import numpy as np
from scipy.stats import norm

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes Greeks for a European option.
    Parameters:
        S: Current asset price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Implied volatility
        option_type: 'call' or 'put'
    Returns:
        Dictionary with Delta, Gamma, Theta, Vega, Rho
    """
    if T <= 0 or sigma <= 0:
        return {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        delta = norm.cdf(d1) - 1
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type.lower() == 'call' else -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta / 365,  # Convert to daily
        'Vega': vega / 100,    # Per 1% change in volatility
        'Rho': rho / 100       # Per 1% change in interest rate
    }

def calculate_portfolio_greeks(weights, asset_prices, strike_prices, times_to_maturity, implied_vols, risk_free_rate, option_types):
    """
    Calculate portfolio-level Greeks based on asset weights and option parameters.
    Parameters:
        weights: Array of portfolio weights
        asset_prices: Array of current asset prices
        strike_prices: Array of option strike prices
        times_to_maturity: Array of times to maturity (years)
        implied_vols: Array of implied volatilities
        risk_free_rate: Risk-free rate
        option_types: Array of option types ('call' or 'put')
    Returns:
        Dictionary with portfolio-level Delta, Gamma, Theta, Vega, Rho
    """
    portfolio_greeks = {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}
    for i, w in enumerate(weights):
        greeks = black_scholes_greeks(
            S=asset_prices[i],
            K=strike_prices[i],
            T=times_to_maturity[i],
            r=risk_free_rate,
            sigma=implied_vols[i],
            option_type=option_types[i]
        )
        for key in portfolio_greeks:
            portfolio_greeks[key] += w * greeks[key]
    
    return portfolio_greeks
