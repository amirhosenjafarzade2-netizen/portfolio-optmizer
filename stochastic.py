import numpy as np
from optimizer import calculate_metrics

def simulate_scenarios(weights, expected_returns, return_stds, volatilities, vol_stds, correlations, dividend_yields, div_stds, inflation, tax_rates, risk_free_rate, use_sharpe, use_inflation, use_tax_rate, use_advanced_metrics, num_simulations, df=5):
    """
    Simulate portfolio scenarios using t-distributions to capture fat tails.
    Parameters:
        weights: Array of portfolio weights
        expected_returns: Array of expected returns
        return_stds: Array of standard deviations for returns
        volatilities: Array of volatilities
        vol_stds: Array of standard deviations for volatilities
        correlations: Correlation matrix
        dividend_yields: Array of dividend yields
        div_stds: Array of standard deviations for dividend yields
        inflation: Inflation rate
        tax_rates: Array of per-asset tax rates
        risk_free_rate: Risk-free rate
        use_sharpe: Boolean to include Sharpe Ratio
        use_inflation: Boolean to include inflation
        use_tax_rate: Boolean to include tax rates
        use_advanced_metrics: Boolean to include VaR and Sortino
        num_simulations: Number of simulations
        df: Degrees of freedom for t-distribution (default 5 for moderate fat tails)
    Returns:
        Dictionary with metrics (mean, p5, p50, p95) for each scenario
    """
    # Initialize arrays to store simulated metrics
    portfolio_returns = np.zeros(num_simulations)
    portfolio_vols = np.zeros(num_simulations)
    portfolio_div_yields = np.zeros(num_simulations)
    real_returns = np.zeros(num_simulations)
    after_tax_returns = np.zeros(num_simulations)
    sharpe_ratios = np.zeros(num_simulations) if use_sharpe else None
    var_values = np.zeros(num_simulations) if use_advanced_metrics else None
    sortino_ratios = np.zeros(num_simulations) if use_advanced_metrics else None

    # Generate t-distributed random variables
    num_assets = len(weights)
    cov_matrix = np.diag(vol_stds) @ correlations @ np.diag(vol_stds)
    L = np.linalg.cholesky(cov_matrix)  # Cholesky decomposition for correlated volatilities

    for i in range(num_simulations):
        # Simulate returns, volatilities, and dividend yields using t-distribution
        t_samples = np.random.standard_t(df, size=(num_assets, 3))  # Shape: (num_assets, 3) for returns, vols, divs
        sim_returns = expected_returns + return_stds * t_samples[:, 0]
        sim_vols = volatilities + vol_stds * (t_samples[:, 1] @ L)  # Apply correlations to volatilities
        sim_div_yields = dividend_yields + div_stds * t_samples[:, 2]

        # Ensure volatilities are positive
        sim_vols = np.maximum(sim_vols, 0.01)

        # Calculate covariance matrix for this scenario
        sim_cov_matrix = np.diag(sim_vols) @ correlations @ np.diag(sim_vols)

        # Calculate metrics for this scenario
        metrics = calculate_metrics(
            weights=weights,
            expected_returns=sim_returns,
            volatilities=sim_vols,
            correlations=correlations,
            dividend_yields=sim_div_yields,
            inflation=inflation,
            tax_rate=tax_rates,
            risk_free_rate=risk_free_rate,
            use_sharpe=use_sharpe,
            use_inflation=use_inflation,
            use_tax_rate=use_tax_rate,
            use_advanced_metrics=use_advanced_metrics,
            cov_matrix=sim_cov_matrix
        )

        # Store results
        portfolio_returns[i] = metrics['Portfolio Return']
        portfolio_vols[i] = metrics['Portfolio Volatility']
        portfolio_div_yields[i] = metrics['Dividend Yield']
        real_returns[i] = metrics['Real Return'] if use_inflation else metrics['Portfolio Return']
        after_tax_returns[i] = metrics['After-Tax Return'] if use_tax_rate else metrics['Portfolio Return']
        if use_sharpe:
            sharpe_ratios[i] = metrics['Sharpe Ratio']
        if use_advanced_metrics:
            var_values[i] = metrics['VaR']
            sortino_ratios[i] = metrics['Sortino Ratio']

    # Compile results with percentiles
    results = {
        'Portfolio Return': {
            'mean': np.mean(portfolio_returns),
            'p5': np.percentile(portfolio_returns, 5),
            'p50': np.percentile(portfolio_returns, 50),
            'p95': np.percentile(portfolio_returns, 95),
            'samples': portfolio_returns
        },
        'Portfolio Volatility': {
            'mean': np.mean(portfolio_vols),
            'p5': np.percentile(portfolio_vols, 5),
            'p50': np.percentile(portfolio_vols, 50),
            'p95': np.percentile(portfolio_vols, 95),
            'samples': portfolio_vols
        },
        'Dividend Yield': {
            'mean': np.mean(portfolio_div_yields),
            'p5': np.percentile(portfolio_div_yields, 5),
            'p50': np.percentile(portfolio_div_yields, 50),
            'p95': np.percentile(portfolio_div_yields, 95),
            'samples': portfolio_div_yields
        }
    }
    if use_inflation:
        results['Real Return'] = {
            'mean': np.mean(real_returns),
            'p5': np.percentile(real_returns, 5),
            'p50': np.percentile(real_returns, 50),
            'p95': np.percentile(real_returns, 95),
            'samples': real_returns
        }
    if use_tax_rate:
        results['After-Tax Return'] = {
            'mean': np.mean(after_tax_returns),
            'p5': np.percentile(after_tax_returns, 5),
            'p50': np.percentile(after_tax_returns, 50),
            'p95': np.percentile(after_tax_returns, 95),
            'samples': after_tax_returns
        }
    if use_sharpe:
        results['Sharpe Ratio'] = {
            'mean': np.mean(sharpe_ratios),
            'p5': np.percentile(sharpe_ratios, 5),
            'p50': np.percentile(sharpe_ratios, 50),
            'p95': np.percentile(sharpe_ratios, 95),
            'samples': sharpe_ratios
        }
    if use_advanced_metrics:
        results['VaR'] = {
            'mean': np.mean(var_values),
            'p5': np.percentile(var_values, 5),
            'p50': np.percentile(var_values, 50),
            'p95': np.percentile(var_values, 95),
            'samples': var_values
        }
        results['Sortino Ratio'] = {
            'mean': np.mean(sortino_ratios),
            'p5': np.percentile(sortino_ratios, 5),
            'p50': np.percentile(sortino_ratios, 50),
            'p95': np.percentile(sortino_ratios, 95),
            'samples': sortino_ratios
        }

    return results
