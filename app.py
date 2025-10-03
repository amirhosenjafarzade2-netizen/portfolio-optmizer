import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
import signal
from optimizer import PortfolioOptimizer
from session_manager import SessionStateManager
from pdf_generator import generate_pdf_report

# Set maximum execution time for optimization (5 minutes)
def timeout_handler(signum, frame):
    raise TimeoutError("Optimization timed out")
signal.signal(signal.SIGALRM, timeout_handler)

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimization Tool")

# Initialize session state
state_manager = SessionStateManager()

# Beginner mode toggle
beginner_mode = st.checkbox("Beginner Mode", value=False, help="Simplify inputs with defaults for new users.")
if beginner_mode:
    st.session_state.use_transaction_costs = False
    st.session_state.use_dividends = False
    st.session_state.use_taxes = False
    st.session_state.use_inflation = False
    st.session_state.use_weight_constraints = False
    st.session_state.allow_short_selling = False

# Asset Class Templates
template_options = {
    "None": None,
    "Balanced Portfolio": {
        'asset_names': ['Stocks', 'Bonds', 'REITs', 'Gold'],
        'returns': [0.10, 0.04, 0.07, 0.05],
        'volatilities': [0.20, 0.08, 0.15, 0.12],
        'correlations': np.array([
            [1.0, 0.2, 0.5, 0.1],
            [0.2, 1.0, 0.3, 0.0],
            [0.5, 0.3, 1.0, 0.2],
            [0.1, 0.0, 0.2, 1.0]
        ])
    },
    "Aggressive Growth": {
        'asset_names': ['Stocks', 'Crypto', 'Emerging Markets'],
        'returns': [0.10, 0.20, 0.12],
        'volatilities': [0.20, 0.50, 0.30],
        'correlations': np.array([
            [1.0, 0.4, 0.6],
            [0.4, 1.0, 0.3],
            [0.6, 0.3, 1.0]
        ])
    },
    "Diversified Income": {
        'asset_names': ['Bonds', 'REITs', 'Real Estate', 'Commodities'],
        'returns': [0.04, 0.07, 0.08, 0.06],
        'volatilities': [0.08, 0.15, 0.12, 0.18],
        'correlations': np.array([
            [1.0, 0.3, 0.4, 0.2],
            [0.3, 1.0, 0.5, 0.3],
            [0.4, 0.5, 1.0, 0.2],
            [0.2, 0.3, 0.2, 1.0]
        ])
    }
}
selected_template = st.selectbox("Load Asset Class Template", list(template_options.keys()), index=0)
if selected_template != "None" and st.button("Apply Template"):
    template = template_options[selected_template]
    num_assets = len(template['asset_names'])
    st.session_state.asset_names = template['asset_names']
    st.session_state.returns = template['returns']
    st.session_state.volatilities = template['volatilities']
    st.session_state.correlations = template['correlations']
    state_manager.update(num_assets)

# Input format selection
input_format = st.radio(
    "Enter returns, volatilities, and rates as:",
    ("Decimal (e.g., 0.15 for 15%)", "Percentage (e.g., 15 for 15%)"),
    help="Choose whether to input values as decimals or percentages."
)

# File uploader for CSV
st.write("### Data Input")
st.write("Upload a CSV file (optional) with columns: 'Asset', 'Return', 'Volatility', and optional 'Correlation_{AssetName}', 'Transaction_Cost', 'Min_Weight', 'Max_Weight', 'Dividend_Yield', 'Tax_Rate'")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV to prefill asset data.")

# Process CSV if uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.asset_names = df['Asset'].tolist()
    st.session_state.returns = df['Return'].tolist()
    st.session_state.volatilities = df['Volatility'].tolist()
    num_assets = len(st.session_state.asset_names)
    if all(f'Correlation_{name}' in df.columns for name in st.session_state.asset_names):
        st.session_state.correlations = np.array([[df[f'Correlation_{st.session_state.asset_names[j]}'][i] 
                                                  for j in range(num_assets)] for i in range(num_assets)])
    else:
        st.session_state.correlations = np.eye(num_assets)
    st.session_state.transaction_costs = df.get('Transaction_Cost', [0.0] * num_assets).tolist()
    st.session_state.min_weights = df.get('Min_Weight', [0.0] * num_assets).tolist()
    st.session_state.max_weights = df.get('Max_Weight', [1.0] * num_assets).tolist()
    st.session_state.dividend_yields = df.get('Dividend_Yield', [0.0] * num_assets).tolist()
    st.session_state.tax_rates = df.get('Tax_Rate', [0.0] * num_assets).tolist()
    st.session_state.use_transaction_costs = 'Transaction_Cost' in df.columns
    st.session_state.use_weight_constraints = 'Min_Weight' in df.columns and 'Max_Weight' in df.columns
    st.session_state.use_dividends = 'Dividend_Yield' in df.columns
    st.session_state.use_taxes = 'Tax_Rate' in df.columns

# Input for number of assets
num_assets = st.number_input(
    "Number of assets",
    min_value=2,
    value=max(2, len(st.session_state.asset_names)),
    step=1,
    help="Specify the number of assets in the portfolio (minimum 2)."
)
state_manager.update(num_assets, beginner_mode=beginner_mode)

# Form for asset details
with st.form("asset_details"):
    st.write("### Core Asset Details")
    cols = st.columns(2)
    for i in range(num_assets):
        with cols[i % 2]:
            st.session_state.asset_names[i] = st.text_input(
                f"Name of Asset {i+1}",
                st.session_state.asset_names[i],
                help="Enter a unique name for the asset (e.g., 'Apple', 'Gold')."
            )
            label_return = f"Expected Annual Return for {st.session_state.asset_names[i]} ({input_format})"
            return_input = st.number_input(
                label_return,
                value=st.session_state.returns[i] * (100 if input_format.startswith("Percentage") else 1),
                step=0.01,
                min_value=0.0,
                max_value=100.0 if input_format.startswith("Percentage") else 1.0,
                help="Annual expected return (e.g., 0.15 or 15 for 15% return)."
            )
            st.session_state.returns[i] = return_input / 100 if input_format.startswith("Percentage") else return_input
            label_vol = f"Annual Volatility for {st.session_state.asset_names[i]} ({input_format})"
            vol_input = st.number_input(
                label_vol,
                value=st.session_state.volatilities[i] * (100 if input_format.startswith("Percentage") else 1),
                step=0.01,
                min_value=0.0,
                max_value=100.0 if input_format.startswith("Percentage") else 1.0,
                help="Annual standard deviation of returns (e.g., 0.2 or 20 for 20% volatility)."
            )
            st.session_state.volatilities[i] = vol_input / 100 if input_format.startswith("Percentage") else vol_input

    if not beginner_mode:
        st.write("### Correlation Matrix")
        st.write("Enter correlations (upper triangle, values between -1 and 1, diagonal is 1)", help="Correlations measure how assets move together.")
        corr_cols = st.columns(num_assets)
        for row in range(num_assets):
            for col in range(row, num_assets):
                if row == col:
                    st.session_state.correlations[row, col] = 1.0
                else:
                    with corr_cols[col]:
                        st.session_state.correlations[row, col] = st.number_input(
                            f"Corr {st.session_state.asset_names[row]} & {st.session_state.asset_names[col]}",
                            value=float(st.session_state.correlations[row, col]),
                            min_value=-1.0,
                            max_value=1.0,
                            step=0.1
                        )
                        st.session_state.correlations[col, row] = st.session_state.correlations[row, col]

    # Normalize correlation matrix
    eigenvalues = np.linalg.eigvals(st.session_state.correlations)
    if np.any(eigenvalues < 0):
        st.session_state.correlations = np.corrcoef(st.session_state.correlations + np.eye(num_assets) * 1e-6)

    if not beginner_mode:
        st.write("### Optional Parameters")
        with st.expander("Advanced Settings"):
            st.session_state.use_transaction_costs = st.checkbox("Include Transaction Costs", value=st.session_state.use_transaction_costs)
            if st.session_state.use_transaction_costs:
                for i in range(num_assets):
                    st.session_state.transaction_costs[i] = st.number_input(
                        f"Transaction Cost for {st.session_state.asset_names[i]} (as decimal, e.g., 0.001 for 0.1%)",
                        value=st.session_state.transaction_costs[i],
                        step=0.0001,
                        min_value=0.0
                    )

            st.session_state.use_weight_constraints = st.checkbox("Include Weight Constraints", value=st.session_state.use_weight_constraints)
            st.session_state.allow_short_selling = st.checkbox("Allow Short Selling", value=st.session_state.allow_short_selling)
            if st.session_state.use_weight_constraints:
                for i in range(num_assets):
                    min_val = -0.5 if st.session_state.allow_short_selling else 0.0
                    max_val = 1.5 if st.session_state.allow_short_selling else 1.0
                    st.session_state.min_weights[i] = st.number_input(
                        f"Minimum Weight for {st.session_state.asset_names[i]}",
                        value=st.session_state.min_weights[i],
                        min_value=min_val,
                        max_value=max_val,
                        step=0.01
                    )
                    st.session_state.max_weights[i] = st.number_input(
                        f"Maximum Weight for {st.session_state.asset_names[i]}",
                        value=st.session_state.max_weights[i],
                        min_value=min_val,
                        max_value=max_val,
                        step=0.01
                    )

            st.session_state.use_dividends = st.checkbox("Include Dividend Yields", value=st.session_state.use_dividends)
            if st.session_state.use_dividends:
                for i in range(num_assets):
                    st.session_state.dividend_yields[i] = st.number_input(
                        f"Dividend Yield for {st.session_state.asset_names[i]} (as decimal, e.g., 0.02 for 2%)",
                        value=st.session_state.dividend_yields[i],
                        step=0.0001,
                        min_value=0.0
                    )

            st.session_state.use_taxes = st.checkbox("Include Tax Rates", value=st.session_state.use_taxes)
            if st.session_state.use_taxes:
                for i in range(num_assets):
                    st.session_state.tax_rates[i] = st.number_input(
                        f"Tax Rate for {st.session_state.asset_names[i]} (as decimal, e.g., 0.15 for 15%)",
                        value=st.session_state.tax_rates[i],
                        step=0.0001,
                        min_value=0.0
                    )

            st.session_state.use_inflation = st.checkbox("Include Inflation Rate", value=st.session_state.use_inflation)
            if st.session_state.use_inflation:
                st.session_state.inflation_rate = st.number_input(
                    "Inflation Rate (as decimal, e.g., 0.03 for 3%)",
                    value=st.session_state.inflation_rate,
                    step=0.0001,
                    min_value=0.0
                )

    submitted = st.form_submit_button("Submit Asset Details")
    if submitted:
        st.success("Asset details submitted successfully!")

# Optimization settings
st.write("### Optimization Settings")
risk_free_rate = st.number_input(
    "Risk-Free Rate (as decimal, e.g., 0.03 for 3%)",
    value=0.03,
    step=0.0001,
    min_value=0.0,
    help="The risk-free rate, typically based on government bonds."
)
optimization_method = st.selectbox(
    "Optimization Method",
    ["Genetic Algorithm", "SLSQP", "Monte Carlo"],
    help="Choose the optimization algorithm."
)
ga_iterations = st.number_input(
    "Genetic Algorithm Iterations",
    min_value=50,
    max_value=500,
    value=100,
    step=10,
    help="Number of generations for Genetic Algorithm."
)
monte_carlo_iterations = st.number_input(
    "Monte Carlo Iterations",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000,
    help="Number of random portfolios for Monte Carlo."
)

# Optimization logic
@st.cache_data(ttl=3600)
def run_optimization(returns, cov_matrix, risk_free_rate, transaction_costs, min_weights, max_weights,
                     dividend_yields, tax_rates, inflation_rate, use_transaction_costs, use_weight_constraints,
                     use_dividends, use_taxes, use_inflation, allow_short_selling, optimization_method,
                     ga_iterations, monte_carlo_iterations):
    optimizer = PortfolioOptimizer(
        returns, cov_matrix, risk_free_rate, transaction_costs, min_weights, max_weights,
        dividend_yields, tax_rates, inflation_rate, use_transaction_costs, use_weight_constraints,
        use_dividends, use_taxes, use_inflation, allow_short_selling
    )
    results = {}
    try:
        signal.alarm(300)  # 5-minute timeout
        if optimization_method == "Genetic Algorithm":
            weights, sharpe, var, cvar = optimizer.optimize_ga(generations=ga_iterations)
            results["Genetic Algorithm"] = {
                "weights": weights,
                "sharpe": sharpe,
                "var": var,
                "cvar": cvar,
                "return": np.dot(weights, returns + dividend_yields),
                "volatility": np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))),
                "cost_penalty": np.sum(np.abs(weights) * transaction_costs),
                "tax_impact": np.dot(weights, (returns + dividend_yields) * tax_rates)
            }
        elif optimization_method == "SLSQP":
            weights, sharpe, var, cvar = optimizer.optimize_slsqp()
            results["SLSQP"] = {
                "weights": weights,
                "sharpe": sharpe,
                "var": var,
                "cvar": cvar,
                "return": np.dot(weights, returns + dividend_yields),
                "volatility": np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))),
                "cost_penalty": np.sum(np.abs(weights) * transaction_costs),
                "tax_impact": np.dot(weights, (returns + dividend_yields) * tax_rates)
            }
        elif optimization_method == "Monte Carlo":
            weights, sharpe, var, cvar = optimizer.optimize_monte_carlo(num_simulations=monte_carlo_iterations)
            results["Monte Carlo"] = {
                "weights": weights,
                "sharpe": sharpe,
                "var": var,
                "cvar": cvar,
                "return": np.dot(weights, returns + dividend_yields),
                "volatility": np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))),
                "cost_penalty": np.sum(np.abs(weights) * transaction_costs),
                "tax_impact": np.dot(weights, (returns + dividend_yields) * tax_rates)
            }
        signal.alarm(0)  # Cancel alarm
        return results
    except TimeoutError:
        st.error("Optimization timed out. Try reducing iterations or using the SLSQP method.")
        return None

# Run optimization
if st.button("Run Optimization"):
    returns = np.array(st.session_state.returns)
    volatilities = np.array(st.session_state.volatilities)
    correlations = st.session_state.correlations
    cov_matrix = np.outer(volatilities, volatilities) * correlations
    transaction_costs = np.array(st.session_state.transaction_costs)
    min_weights = np.array(st.session_state.min_weights)
    max_weights = np.array(st.session_state.max_weights)
    dividend_yields = np.array(st.session_state.dividend_yields)
    tax_rates = np.array(st.session_state.tax_rates)
    inflation_rate = st.session_state.inflation_rate

    results = run_optimization(
        returns, cov_matrix, risk_free_rate, transaction_costs, min_weights, max_weights,
        dividend_yields, tax_rates, inflation_rate,
        st.session_state.use_transaction_costs, st.session_state.use_weight_constraints,
        st.session_state.use_dividends, st.session_state.use_taxes, st.session_state.use_inflation,
        st.session_state.allow_short_selling, optimization_method, ga_iterations, monte_carlo_iterations
    )

    if results:
        tabs = st.tabs(["Weights", "Metrics", "Efficient Frontier", "Wealth Projection", "Suggestions", "Sensitivity Analysis"])
        
        with tabs[0]:  # Weights
            st.write("### Optimal Weights")
            for method, result in results.items():
                weights_df = pd.DataFrame({
                    "Asset": st.session_state.asset_names,
                    "Weight (%)": [w * 100 for w in result['weights']]
                })
                st.write(f"#### {method}")
                st.table(weights_df)
                fig = go.Figure(data=[
                    go.Bar(x=st.session_state.asset_names, y=[w * 100 for w in result['weights']])
                ])
                fig.update_layout(title=f"{method} Portfolio Weights", xaxis_title="Asset", yaxis_title="Weight (%)")
                st.plotly_chart(fig)

        with tabs[1]:  # Metrics
            st.write("### Performance Metrics")
            for method, result in results.items():
                metrics_data = {
                    "Metric": ["Expected Return (Nominal)", "Expected Volatility", "Sharpe Ratio", "Value-at-Risk (95%)", "Conditional VaR (95%)"],
                    "Value": [
                        f"{result['return'] * 100:.2f}%",
                        f"{result['volatility'] * 100:.2f}%",
                        f"{result['sharpe']:.2f}",
                        f"{result['var'] * 100:.2f}%",
                        f"{result['cvar'] * 100:.2f}%"
                    ]
                }
                if st.session_state.use_transaction_costs:
                    metrics_data["Metric"].append("Transaction Cost Impact")
                    metrics_data["Value"].append(f"{result['cost_penalty'] * 100:.2f}%")
                if st.session_state.use_taxes:
                    metrics_data["Metric"].append("Tax Impact")
                    metrics_data["Value"].append(f"{result['tax_impact'] * 100:.2f}%")
                if st.session_state.use_inflation:
                    adjusted_return = result['return'] - result['tax_impact'] - result['cost_penalty'] - inflation_rate
                    metrics_data["Metric"].append("Adjusted Return (after inflation)")
                    metrics_data["Value"].append(f"{adjusted_return * 100:.2f}%")
                st.write(f"#### {method}")
                st.table(pd.DataFrame(metrics_data))

        with tabs[2]:  # Efficient Frontier
            st.write("### Efficient Frontier")
            num_portfolios = 1000
            sim_weights = [np.random.dirichlet(np.ones(num_assets)) for _ in range(num_portfolios)]
            sim_returns = [np.dot(w, returns + dividend_yields) for w in sim_weights]
            sim_vols = [np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) for w in sim_weights]
            sim_sharpes = [(r - (risk_free_rate - inflation_rate if st.session_state.use_inflation else 0)) / v if v > 0 else 0 for r, v in zip(sim_returns, sim_vols)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[v * 100 for v in sim_vols],
                y=[r * 100 for r in sim_returns],
                mode='markers',
                marker=dict(color=sim_sharpes, colorscale='Viridis', showscale=True, colorbar=dict(title="Sharpe Ratio")),
                name="Portfolios"
            ))
            for method, result in results.items():
                fig.add_trace(go.Scatter(
                    x=[result['volatility'] * 100],
                    y=[result['return'] * 100],
                    mode='markers',
                    marker=dict(size=15, symbol='star'),
                    name=method
                ))
            fig.update_layout(
                title="Efficient Frontier",
                xaxis_title="Volatility (%)",
                yaxis_title="Return (%)",
                showlegend=True
            )
            st.plotly_chart(fig)

        with tabs[3]:  # Wealth Projection
            st.write("### Wealth Projection")
            time_horizon = st.slider("Time Horizon (Years)", 1, 30, 10)
            for method, result in results.items():
                adjusted_return = result['return'] - result['tax_impact'] - result['cost_penalty'] - (inflation_rate if st.session_state.use_inflation else 0)
                wealth = [(1 + adjusted_return) ** t for t in range(time_horizon + 1)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(time_horizon + 1)),
                    y=wealth,
                    mode='lines+markers',
                    name=method
                ))
                fig.update_layout(
                    title=f"Wealth Projection ({method})",
                    xaxis_title="Years",
                    yaxis_title="Wealth (Multiple of Initial Investment)"
                )
                st.plotly_chart(fig)

        with tabs[4]:  # Suggestions
            st.write("### Portfolio Improvement Suggestions")
            primary_method = list(results.keys())[0]
            best_weights = results[primary_method]['weights']
            asset_names = st.session_state.asset_names
            effective_diversification = 1 / np.sum(best_weights ** 2)
            avg_corr = (np.sum(correlations) - num_assets) / (num_assets * (num_assets - 1)) if num_assets > 1 else 0
            
            st.write(f"Current effective diversification: {effective_diversification:.2f} (out of {num_assets} assets)")
            st.write(f"Average correlation between assets: {avg_corr:.2f}")
            
            if effective_diversification < num_assets / 2 or avg_corr > 0.7:
                st.warning("Portfolio may benefit from more diversification. Consider adding 1-2 assets with low correlations (e.g., <0.3) to existing ones, like Gold or Bonds if you have mostly Stocks.")
                st.info("Suggestion: Add an asset with return ~0.05, volatility ~0.10, and correlations <0.2 to all current assets.")
            else:
                st.success("Portfolio appears well-diversified.")
            
            st.write("#### Risk Contribution Analysis")
            risk_contributions = []
            portfolio_vol = results[primary_method]['volatility']
            for i in range(num_assets):
                marginal_contrib = np.dot(cov_matrix[i], best_weights) / portfolio_vol
                risk_contrib = best_weights[i] * marginal_contrib / portfolio_vol
                risk_contributions.append(risk_contrib * 100)
            contrib_data = {"Asset": asset_names, "Risk Contribution (%)": risk_contributions}
            st.table(pd.DataFrame(contrib_data))
            high_risk_asset = asset_names[np.argmax(risk_contributions)]
            if max(risk_contributions) > 100 / num_assets * 1.5:
                st.warning(f"{high_risk_asset} contributes {max(risk_contributions):.2f}% to portfolio risk, more than its fair share ({100/num_assets:.2f}%). Consider replacing with a less volatile or less correlated asset.")
            
            st.write("#### Asset Removal Suggestions")
            removal_improvements = {}
            for i in range(num_assets):
                with st.spinner(f"Simulating removal of {asset_names[i]}..."):
                    temp_returns = np.delete(returns, i)
                    temp_volatilities = np.delete(volatilities, i)
                    temp_correlations = np.delete(np.delete(correlations, i, axis=0), i, axis=1)
                    temp_cov = np.outer(temp_volatilities, temp_volatilities) * temp_correlations
                    temp_transaction_costs = np.delete(transaction_costs, i)
                    temp_min_weights = np.delete(min_weights, i)
                    temp_max_weights = np.delete(max_weights, i)
                    temp_dividend_yields = np.delete(dividend_yields, i)
                    temp_tax_rates = np.delete(tax_rates, i)
                    
                    temp_optimizer = PortfolioOptimizer(
                        temp_returns, temp_cov, risk_free_rate, temp_transaction_costs, temp_min_weights, temp_max_weights,
                        temp_dividend_yields, temp_tax_rates, inflation_rate,
                        st.session_state.use_transaction_costs, st.session_state.use_weight_constraints,
                        st.session_state.use_dividends, st.session_state.use_taxes, st.session_state.use_inflation,
                        st.session_state.allow_short_selling
                    )
                    _, temp_sharpe, _, _ = temp_optimizer.optimize_monte_carlo(1000)
                    improvement = (temp_sharpe - results[primary_method]['sharpe']) / results[primary_method]['sharpe'] * 100 if results[primary_method]['sharpe'] != 0 else 0
                    removal_improvements[asset_names[i]] = improvement
            
            if removal_improvements:
                best_removal = max(removal_improvements, key=removal_improvements.get)
                if removal_improvements[best_removal] > 5:
                    st.warning(f"Consider removing {best_removal} - could improve Sharpe by {removal_improvements[best_removal]:.2f}%")
                else:
                    st.success("No significant improvement from removing any single asset.")
                st.table(pd.DataFrame({"Asset": list(removal_improvements.keys()), "Sharpe Improvement (%)": list(removal_improvements.values())}))

        with tabs[5]:  # Sensitivity Analysis
            st.write("### Sensitivity Analysis")
            sensitivity_data = []
            for i in range(num_assets):
                for factor, label in [(1.1, "+10%"), (0.9, "-10%")]:
                    for metric, metric_label in [("returns", "Return"), ("volatilities", "Volatility")]:
                        temp_returns = returns.copy()
                        temp_volatilities = volatilities.copy()
                        if metric == "returns":
                            temp_returns[i] *= factor
                        else:
                            temp_volatilities[i] *= factor
                        temp_cov = np.outer(temp_volatilities, temp_volatilities) * correlations
                        temp_optimizer = PortfolioOptimizer(
                            temp_returns, temp_cov, risk_free_rate, transaction_costs, min_weights, max_weights,
                            dividend_yields, tax_rates, inflation_rate,
                            st.session_state.use_transaction_costs, st.session_state.use_weight_constraints,
                            st.session_state.use_dividends, st.session_state.use_taxes, st.session_state.use_inflation,
                            st.session_state.allow_short_selling
                        )
                        _, temp_sharpe, _, _ = temp_optimizer.optimize_monte_carlo(1000)
                        sensitivity_data.append({
                            "Asset": asset_names[i],
                            "Metric": metric_label,
                            "Change": label,
                            "Sharpe Ratio": temp_sharpe,
                            "Sharpe Change (%)": (temp_sharpe - results[primary_method]['sharpe']) / results[primary_method]['sharpe'] * 100 if results[primary_method]['sharpe'] != 0 else 0
                        })
            st.table(pd.DataFrame(sensitivity_data))

        # Export results as CSV
        for method, result in results.items():
            export_data = {
                "Asset": st.session_state.asset_names,
                "Weight (%)": [w * 100 for w in result['weights']],
                "Return (%)": [r * 100 for r in returns],
                "Volatility (%)": [v * 100 for v in volatilities]
            }
            if st.session_state.use_dividends:
                export_data["Dividend Yield (%)"] = [d * 100 for d in dividend_yields]
            if st.session_state.use_taxes:
                export_data["Tax Rate (%)"] = [t * 100 for t in tax_rates]
            export_df = pd.DataFrame(export_data)
            export_df["Method"] = method
            export_df["Sharpe Ratio"] = result['sharpe']
            export_df["Portfolio Return (%)"] = result['return'] * 100
            export_df["Portfolio Volatility (%)"] = result['volatility'] * 100
            export_df["VaR (95%)"] = result['var'] * 100
            export_df["CVaR (95%)"] = result['cvar'] * 100
            if st.session_state.use_transaction_costs:
                export_df["Transaction Cost Impact (%)"] = result['cost_penalty'] * 100
            if st.session_state.use_taxes:
                export_df["Tax Impact (%)"] = result['tax_impact'] * 100
            if st.session_state.use_inflation:
                export_df["Adjusted Return (%)"] = (result['return'] - result['tax_impact'] - result['cost_penalty'] - inflation_rate) * 100
            csv = export_df.to_csv(index=False)
            st.download_button(
                label=f"Download {method} Results as CSV",
                data=csv,
                file_name=f"portfolio_results_{method.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

        # PDF Export
        st.write("### Export Report as PDF")
        if st.button("Generate PDF Report"):
            pdf_buffer = generate_pdf_report(results, st.session_state.use_transaction_costs, st.session_state.use_taxes, st.session_state.use_inflation, inflation_rate)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="portfolio_report.pdf",
                mime="application/pdf"
            )
