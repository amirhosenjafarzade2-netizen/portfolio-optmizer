import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import signal
from optimizer import PortfolioOptimizer

# Set maximum execution time for optimization (5 minutes)
def timeout_handler(signum, frame):
    raise TimeoutError("Optimization timed out")
signal.signal(signal.SIGALRM, timeout_handler)

# Page configuration
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimization Tool")

# Input: Number of assets
num_assets = st.number_input(
    "Number of Assets",
    min_value=2,
    value=2,
    step=1,
    help="Specify the number of assets in the portfolio (minimum 2)."
)

# Input: Asset names
st.subheader("Asset Names")
asset_names = []
cols = st.columns(2)
for i in range(num_assets):
    with cols[i % 2]:
        asset_names.append(st.text_input(f"Asset {i+1} Name", f"Asset {i+1}", help="Enter a unique name for the asset."))

# Input: Optimization method
optimization_method = st.selectbox(
    "Optimization Method",
    ["Genetic Algorithm", "Monte Carlo", "SLSQP"],
    help="Choose the optimization algorithm."
)

# Input: Iterations
if optimization_method in ["Genetic Algorithm", "Monte Carlo"]:
    ga_iterations = st.number_input(
        "Genetic Algorithm Iterations",
        min_value=50,
        max_value=500,
        value=100,
        step=10,
        help="Number of generations for Genetic Algorithm (used if selected)."
    ) if optimization_method == "Genetic Algorithm" else 100
    monte_carlo_iterations = st.number_input(
        "Monte Carlo Iterations",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000,
        help="Number of random portfolios for Monte Carlo (used if selected)."
    ) if optimization_method == "Monte Carlo" else 5000
else:
    ga_iterations, monte_carlo_iterations = 100, 5000  # Defaults for caching

# Input: Input format
input_format = st.radio(
    "Enter metrics as:",
    ("Decimal (e.g., 0.15 for 15%)", "Percentage (e.g., 15 for 15%)"),
    help="Choose whether to input values as decimals or percentages."
)

# Input: Metrics
st.subheader("Asset Metrics")
returns = []
volatilities = []
correlations = np.eye(num_assets)
with st.form("metrics_form"):
    # Core metrics
    for i in range(num_assets):
        st.write(f"#### {asset_names[i]}")
        cols = st.columns(2)
        with cols[0]:
            return_input = st.number_input(
                f"Expected Annual Return ({input_format})",
                value=10.0 if input_format.startswith("Percentage") else 0.10,
                step=0.01,
                min_value=0.0,
                max_value=100.0 if input_format.startswith("Percentage") else 1.0
            )
            returns.append(return_input / 100 if input_format.startswith("Percentage") else return_input)
        with cols[1]:
            vol_input = st.number_input(
                f"Annual Volatility ({input_format})",
                value=20.0 if input_format.startswith("Percentage") else 0.20,
                step=0.01,
                min_value=0.0,
                max_value=100.0 if input_format.startswith("Percentage") else 1.0
            )
            volatilities.append(vol_input / 100 if input_format.startswith("Percentage") else vol_input)

    # Correlation matrix
    st.write("#### Correlation Matrix")
    corr_cols = st.columns(num_assets)
    for row in range(num_assets):
        for col in range(row + 1, num_assets):
            with corr_cols[col]:
                correlations[row, col] = st.number_input(
                    f"Corr {asset_names[row]} & {asset_names[col]}",
                    value=0.3,
                    min_value=-1.0,
                    max_value=1.0,
                    step=0.1
                )
                correlations[col, row] = correlations[row, col]

    # Optional metrics
    st.write("#### Optional Metrics")
    use_transaction_costs = st.checkbox("Include Transaction Costs", value=False)
    transaction_costs = [0.0] * num_assets
    if use_transaction_costs:
        for i in range(num_assets):
            transaction_costs[i] = st.number_input(
                f"Transaction Cost for {asset_names[i]} (as decimal, e.g., 0.001 for 0.1%)",
                value=0.001,
                step=0.0001,
                min_value=0.0
            )

    use_dividends = st.checkbox("Include Dividend Yields", value=False)
    dividend_yields = [0.0] * num_assets
    if use_dividends:
        for i in range(num_assets):
            dividend_yields[i] = st.number_input(
                f"Dividend Yield for {asset_names[i]} (as decimal, e.g., 0.02 for 2%)",
                value=0.02,
                step=0.0001,
                min_value=0.0
            )

    use_taxes = st.checkbox("Include Tax Rates", value=False)
    tax_rates = [0.0] * num_assets
    if use_taxes:
        for i in range(num_assets):
            tax_rates[i] = st.number_input(
                f"Tax Rate for {asset_names[i]} (as decimal, e.g., 0.15 for 15%)",
                value=0.15,
                step=0.0001,
                min_value=0.0
            )

    use_inflation = st.checkbox("Include Inflation Rate", value=False)
    inflation_rate = st.number_input(
        "Inflation Rate (as decimal, e.g., 0.03 for 3%)",
        value=0.03,
        step=0.0001,
        min_value=0.0
    ) if use_inflation else 0.0

    risk_free_rate = st.number_input(
        "Risk-Free Rate (as decimal, e.g., 0.03 for 3%)",
        value=0.03,
        step=0.0001,
        min_value=0.0
    )

    submitted = st.form_submit_button("Submit Metrics")

# Optimization logic
@st.cache_data(ttl=3600)
def run_optimization(returns, cov_matrix, risk_free_rate, transaction_costs, dividend_yields, tax_rates,
                     inflation_rate, use_transaction_costs, use_dividends, use_taxes, use_inflation,
                     optimization_method, ga_iterations, monte_carlo_iterations):
    optimizer = PortfolioOptimizer(
        returns, cov_matrix, risk_free_rate, transaction_costs, dividend_yields, tax_rates,
        inflation_rate, use_transaction_costs, use_dividends, use_taxes, use_inflation
    )
    try:
        signal.alarm(300)  # 5-minute timeout
        if optimization_method == "Genetic Algorithm":
            weights, sharpe, var, cvar = optimizer.optimize_ga(generations=ga_iterations)
        elif optimization_method == "Monte Carlo":
            weights, sharpe, var, cvar = optimizer.optimize_monte_carlo(num_simulations=monte_carlo_iterations)
        else:  # SLSQP
            weights, sharpe, var, cvar = optimizer.optimize_slsqp()
        signal.alarm(0)  # Cancel alarm
        portfolio_return = np.dot(weights, returns + (dividend_yields if use_dividends else np.zeros.like(returns)))
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        cost_penalty = np.sum(np.abs(weights) * transaction_costs) if use_transaction_costs else 0.0
        tax_impact = np.dot(weights, (returns + (dividend_yields if use_dividends else np.zeros.like(returns))) * tax_rates) if use_taxes else 0.0
        return {
            "weights": weights,
            "sharpe": sharpe,
            "var": var,
            "cvar": cvar,
            "return": portfolio_return,
            "volatility": portfolio_vol,
            "cost_penalty": cost_penalty,
            "tax_impact": tax_impact
        }
    except TimeoutError:
        st.error("Optimization timed out. Try reducing iterations or using SLSQP.")
        return None

# Run optimization
if st.button("Optimize Portfolio"):
    with st.spinner("Running optimization..."):
        progress_bar = st.progress(0)
        returns = np.array(returns)
        volatilities = np.array(volatilities)
        cov_matrix = np.outer(volatilities, volatilities) * correlations
        transaction_costs = np.array(transaction_costs)
        dividend_yields = np.array(dividend_yields)
        tax_rates = np.array(tax_rates)

        # Ensure correlation matrix is positive semi-definite
        eigenvalues = np.linalg.eigvals(correlations)
        if np.any(eigenvalues < 0):
            correlations += np.eye(num_assets) * 1e-6
            cov_matrix = np.outer(volatilities, volatilities) * correlations

        results = run_optimization(
            returns, cov_matrix, risk_free_rate, transaction_costs, dividend_yields, tax_rates,
            inflation_rate, use_transaction_costs, use_dividends, use_taxes, use_inflation,
            optimization_method, ga_iterations, monte_carlo_iterations
        )

        if results:
            progress_bar.progress(1.0)
            st.subheader("Optimization Results")

            # Weights
            st.write("### Optimal Weights")
            weights_df = pd.DataFrame({
                "Asset": asset_names,
                "Weight (%)": [w * 100 for w in results["weights"]]
            })
            st.table(weights_df)
            fig_weights = go.Figure(data=[
                go.Bar(x=asset_names, y=[w * 100 for w in results["weights"]])
            ])
            fig_weights.update_layout(
                title=f"Portfolio Weights ({optimization_method})",
                xaxis_title="Asset",
                yaxis_title="Weight (%)"
            )
            st.plotly_chart(fig_weights)

            # Metrics
            st.write("### Performance Metrics")
            metrics_data = {
                "Metric": ["Expected Return", "Expected Volatility", "Sharpe Ratio", "VaR (95%)", "CVaR (95%)"],
                "Value": [
                    f"{results['return'] * 100:.2f}%",
                    f"{results['volatility'] * 100:.2f}%",
                    f"{results['sharpe']:.2f}",
                    f"{results['var'] * 100:.2f}%",
                    f"{results['cvar'] * 100:.2f}%"
                ]
            }
            if use_transaction_costs:
                metrics_data["Metric"].append("Transaction Cost Impact")
                metrics_data["Value"].append(f"{results['cost_penalty'] * 100:.2f}%")
            if use_taxes:
                metrics_data["Metric"].append("Tax Impact")
                metrics_data["Value"].append(f"{results['tax_impact'] * 100:.2f}%")
            if use_inflation:
                adjusted_return = results['return'] - results['tax_impact'] - results['cost_penalty'] - inflation_rate
                metrics_data["Metric"].append("Adjusted Return (after inflation)")
                metrics_data["Value"].append(f"{adjusted_return * 100:.2f}%")
            st.table(pd.DataFrame(metrics_data))

            # Efficient Frontier
            st.write("### Efficient Frontier")
            num_portfolios = 1000
            sim_weights = [np.random.dirichlet(np.ones(num_assets)) for _ in range(num_portfolios)]
            sim_returns = [np.dot(w, returns + (dividend_yields if use_dividends else np.zeros_like(returns))) for w in sim_weights]
            sim_vols = [np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) for w in sim_weights]
            sim_sharpes = [(r - (risk_free_rate - inflation_rate if use_inflation else 0)) / v if v > 0 else 0 for r, v in zip(sim_returns, sim_vols)]
            fig_frontier = go.Figure()
            fig_frontier.add_trace(go.Scatter(
                x=[v * 100 for v in sim_vols],
                y=[r * 100 for r in sim_returns],
                mode='markers',
                marker=dict(color=sim_sharpes, colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe Ratio")),
                name="Portfolios"
            ))
            fig_frontier.add_trace(go.Scatter(
                x=[results['volatility'] * 100],
                y=[results['return'] * 100],
                mode='markers',
                marker=dict(size=15, symbol='star'),
                name=optimization_method
            ))
            fig_frontier.update_layout(
                title="Efficient Frontier",
                xaxis_title="Volatility (%)",
                yaxis_title="Return (%)",
                showlegend=True
            )
            st.plotly_chart(fig_frontier)

            # CSV Export
            st.write("### Export Results")
            export_data = {
                "Asset": asset_names,
                "Weight (%)": [w * 100 for w in results["weights"]],
                "Return (%)": [r * 100 for r in returns],
                "Volatility (%)": [v * 100 for v in volatilities]
            }
            if use_dividends:
                export_data["Dividend Yield (%)"] = [d * 100 for d in dividend_yields]
            if use_taxes:
                export_data["Tax Rate (%)"] = [t * 100 for t in tax_rates]
            export_df = pd.DataFrame(export_data)
            export_df["Sharpe Ratio"] = results["sharpe"]
            export_df["Portfolio Return (%)"] = results["return"] * 100
            export_df["Portfolio Volatility (%)"] = results["volatility"] * 100
            export_df["VaR (95%)"] = results["var"] * 100
            export_df["CVaR (95%)"] = results["cvar"] * 100
            if use_transaction_costs:
                export_df["Transaction Cost Impact (%)"] = results["cost_penalty"] * 100
            if use_taxes:
                export_df["Tax Impact (%)"] = results["tax_impact"] * 100
            if use_inflation:
                export_df["Adjusted Return (%)"] = (results["return"] - results["tax_impact"] - results["cost_penalty"] - inflation_rate) * 100
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"portfolio_results_{optimization_method.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
