import numpy as np
from optimizer import calculate_metrics

def simulate_scenarios(weights, expected_returns, return_stds, volatilities, vol_stds, correlations, dividend_yields, div_stds, inflation, tax_rate, risk_free_rate, use_sharpe, use_inflation, use_tax_rate, use_advanced_metrics, num_simulations):
    sim_metrics = {}

    for _ in range(num_simulations):
        sampled_returns = np.random.normal(expected_returns, return_stds)
        sampled_vols = np.abs(np.random.normal(volatilities, vol_stds))
        sampled_divs = np.abs(np.random.normal(dividend_yields, div_stds))
        sim_cov = np.diag(sampled_vols) @ correlations @ np.diag(sampled_vols)
        sim_metrics_temp = calculate_metrics(weights, sampled_returns, sim_cov, sampled_divs, inflation, tax_rate, risk_free_rate, use_sharpe, use_inflation, use_tax_rate, use_advanced_metrics)
        for k, v in sim_metrics_temp.items():
            if k not in sim_metrics:
                sim_metrics[k] = []
            sim_metrics[k].append(v)

    # Aggregate percentiles
    aggregated = {}
    for k, vals in sim_metrics.items():
        aggregated[k] = {
            'mean': np.mean(vals),
            'p5': np.percentile(vals, 5),
            'p50': np.median(vals),
            'p95': np.percentile(vals, 95),
            'samples': vals  # For plotting
        }

    return aggregated
