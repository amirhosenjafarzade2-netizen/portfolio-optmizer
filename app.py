import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from optimizer import PortfolioOptimizer

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimization Tool")

# Session state manager
class SessionStateManager:
    """Manage Streamlit session state initialization and updates."""
    def __init__(self):
        defaults = {
            'asset_names': [],
            'returns': [],
            'volatilities': [],
            'correlations': np.array([]),
            'transaction_costs': [],
            'min_weights': [],
            'max_weights': [],
            'dividend_yields': [],
            'tax_rates': [],
            'use_transaction_costs': False,
            'use_weight_constraints': False,
            'use_dividends': False,
            'use_taxes': False,
            'use_inflation': False,
            'inflation_rate': 0.0,
            'allow_short_selling': False
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def update(self, num_assets):
        """Update session state for a given number of assets."""
        if len(st.session_state.asset_names) != num_assets:
            st.session_state.asset_names = [f"Asset {i+1}" for i in range(num_assets)] if not st.session_state.asset_names else st.session_state.asset_names[:num_assets]
            st.session_state.returns = [0.1] * num_assets if not st.session_state.returns else st.session_state.returns[:num_assets]
            st.session_state.volatilities = [0.2] * num_assets if not st.session_state.volatilities else st.session_state.volatilities[:num_assets]
            st.session_state.correlations = np.eye(num_assets) if st.session_state.correlations.size == 0 else st.session_state.correlations[:num_assets, :num_assets]
            st.session_state.transaction_costs = [0.0] * num_assets if not st.session_state.transaction_costs else st.session_state.transaction_costs[:num_assets]
            st.session_state.min_weights = [0.0] * num_assets if not st.session_state.min_weights else st.session_state.min_weights[:num_assets]
            st.session_state.max_weights = [1.0] * num_assets if not st.session_state.max_weights else st.session_state.max_weights[:num_assets]
            st.session_state.dividend_yields = [0.0] * num_assets if not st.session_state.dividend_yields else st.session_state.dividend_yields[:num_assets]
            st.session_state.tax_rates = [0.0] * num_assets if not st.session_state.tax_rates else st.session_state.tax_rates[:num_assets]

# Initialize session state
state_manager = SessionStateManager()

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
state_manager.update(num_assets)

# Form for asset details
with st.form("asset_details"):
    st.write("### Core Asset Details")
    cols = st.columns(2)
    for i in range(num_assets):
        with cols[i % 2]:
            st.session_state.asset_names[i] = st.text_input(
                f"Name of Asset {i+1}",
                st.session_state.asset_names[i],
                help="Enter a unique name for the asset (e.g., 'Apple', 'Stock1')."
            )
            label_return = f"Expected Annual Return for {st.session_state.asset_names[i]} ({input_format})"
            return_input = st.number_input(
                label_return,
                value=st.session_state.returns[i] * (100 if input_format.startswith("Percentage") else 1),
                step=0.01,
                help="Annual expected return (e.g., 0.15 or 15 for 15% return)."
            )
            st.session_state.returns[i] = return_input / 100 if input_format.startswith("Percentage") else return_input
            label_vol = f"Annual Volatility for {st.session_state.asset_names[i]} ({input_format})"
            vol_input = st.number_input(
                label_vol,
                value=st.session_state.volatilities[i] * (100 if input_format.startswith("Percentage") else 1),
                step=0.01,
                min_value=0.0,
                help="Annual standard deviation of returns (e.g., 0.2 or 20 for 20% volatility)."
            )
            st.session_state.volatilities[i] = vol_input / 100 if input_format.startswith("Percentage") else vol_input

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

    # Normalize correlation matrix to ensure positive semi-definiteness
    eigenvalues = np.linalg.eigvals(st.session_state.correlations)
    if np.any(eigenvalues < 0):
        st.session_state.correlations = np.corrcoef(st.session_state.correlations + np.eye(num_assets) * 1e-6)  # Small perturbation

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
                    f"Minimum Weight for {st.session_state.asset_names[i]} (as decimal, e.g., 0.1 for 10%)",
                    value=st.session_state.min_weights[i],
                    min_value=min_val,
                    max_value=max_val,
                    step=0.01
                )
                st.session_state.max_weights[i] = st.number_input(
                    f"Maximum Weight for {st.session_state.asset_names[i]} (as decimal, e.g., 0.5 for 50%)",
                    value=st.session_state.max_weights[i],
                    min_value=min_val,
                    max_value=max_val,
                    step=0.01
                )

        st.session_state.use_dividends = st.checkbox("Include Dividend Yields", value=st.session_state.use_dividends)
        if st.session_state.use_dividends:
            for i in range(num_assets):
                label_div = f"Dividend Yield for {st.session_state.asset_names[i]} ({input_format})"
                div_input = st.number_input(
                    label_div,
                    value=st.session_state.dividend_yields[i] * (100 if input_format.startswith("Percentage") else 1),
                    step=0.01,
                    min_value=0.0
                )
                st.session_state.dividend_yields[i] = div_input / 100 if input_format.startswith("Percentage") else div_input

        st.session_state.use_taxes = st.checkbox("Include Tax Rates", value=st.session_state.use_taxes)
        if st.session_state.use_taxes:
            for i in range(num_assets):
                label_tax = f"Tax Rate on Returns for {st.session_state.asset_names[i]} ({input_format})"
                tax_input = st.number_input(
                    label_tax,
                    value=st.session_state.tax_rates[i] * (100 if input_format.startswith("Percentage") else 1),
                    step=0.01,
                    min_value=0.0,
                    max_value=100.0 if input_format.startswith("Percentage") else 1.0
                )
                st.session_state.tax_rates[i] = tax_input / 100 if input_format.startswith("Percentage") else tax_input

        st.session_state.use_inflation = st.checkbox("Include Inflation Rate", value=st.session_state.use_inflation)
        if st.session_state.use_inflation:
            st.session_state.inflation_rate = st.number_input(
                f"Expected Inflation Rate ({input_format})",
                value=st.session_state.inflation_rate * (100 if input_format.startswith("Percentage") else 1),
                step=0.01,
                min_value=0.0
            )
            st.session_state.inflation_rate = st.session_state.inflation_rate / 100 if input_format.startswith("Percentage") else st.session_state.inflation_rate

    st.write("### Optimization Settings")
    optimization_method = st.selectbox(
        "Optimization Method",
        ("Genetic Algorithm", "SLSQP", "Monte Carlo", "Compare All"),
        help="Choose the method to optimize the portfolio or compare all methods."
    )
    ga_iterations = st.number_input(
        "Genetic Algorithm Iterations",
        min_value=50,
        max_value=1000,
        value=200,
        step=10,
        help="Number of generations for Genetic Algorithm."
    )
    monte_carlo_iterations = st.number_input(
        "Monte Carlo Iterations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        help="Number of random portfolios for Monte Carlo."
    )
    risk_free_rate = st.number_input(
        f"Risk-Free Rate ({input_format})",
        value=0.02 * (100 if input_format.startswith("Percentage") else 1),
        step=0.01
    )
    risk_free_rate = risk_free_rate / 100 if input_format.startswith("Percentage") else risk_free_rate
    submitted = st.form_submit_button("Optimize Portfolio")

# Process optimization and display results
if submitted:
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

    # Validate inputs
    if np.any(min_weights > max_weights):
        st.error("Minimum weights cannot exceed maximum weights.")
    elif np.sum(min_weights) > 1.0 and not st.session_state.allow_short_selling:
        st.error("Sum of minimum weights exceeds 100%.")
    elif np.linalg.matrix_rank(cov_matrix) < len(returns):
        st.error("Covariance matrix is singular. Adjust correlations to ensure linear independence.")
    else:
        optimizer = PortfolioOptimizer(
            returns, cov_matrix, risk_free_rate, transaction_costs, min_weights, max_weights,
            dividend_yields, tax_rates, inflation_rate,
            st.session_state.use_transaction_costs, st.session_state.use_weight_constraints,
            st.session_state.use_dividends, st.session_state.use_taxes, st.session_state.use_inflation,
            st.session_state.allow_short_selling
        )
        progress_bar = st.progress(0)
        results = {}

        methods = ["Genetic Algorithm", "SLSQP", "Monte Carlo"] if optimization_method == "Compare All" else [optimization_method]
        for method in methods:
            with st.spinner(f"Running {method}..."):
                if method == "Genetic Algorithm":
                    best_weights, best_sharpe, var, cvar = optimizer.optimize_ga(ga_iterations, progress_bar=progress_bar)
                elif method == "SLSQP":
                    best_weights, best_sharpe, var, cvar = optimizer.optimize_slsqp()
                    progress_bar.progress(100)
                else:  # Monte Carlo
                    best_weights, best_sharpe, var, cvar = optimizer.optimize_monte_carlo(monte_carlo_iterations)
                    progress_bar.progress(100)
                results[method] = {
                    'weights': best_weights,
                    'sharpe': best_sharpe,
                    'return': np.dot(best_weights, returns + (dividend_yields if st.session_state.use_dividends else np.zeros_like(returns))),
                    'volatility': np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights))),
                    'var': var,
                    'cvar': cvar,
                    'tax_impact': np.dot(best_weights, (returns + (dividend_yields if st.session_state.use_dividends else np.zeros_like(returns))) * tax_rates) if st.session_state.use_taxes else 0.0,
                    'cost_penalty': np.sum(np.abs(best_weights) * transaction_costs) if st.session_state.use_transaction_costs else 0.0
                }

        # Display results in tabs
        tabs = st.tabs(["Summary", "Weights", "Charts", "Settings"])
        with tabs[0]:  # Summary
            st.write("### Portfolio Summary")
            for method, result in results.items():
                st.write(f"#### {method}")
                summary_data = {
                    "Metric": ["Expected Return (Nominal)", "Expected Volatility", "Sharpe Ratio", "Value-at-Risk (95%)", "Conditional VaR (95%)"],
                    "Value": [f"{result['return'] * 100:.2f}%", f"{result['volatility'] * 100:.2f}%", f"{result['sharpe']:.2f}", 
                              f"{result['var'] * 100:.2f}%", f"{result['cvar'] * 100:.2f}%"]
                }
                if st.session_state.use_transaction_costs:
                    summary_data["Metric"].append("Transaction Cost Impact")
                    summary_data["Value"].append(f"{result['cost_penalty'] * 100:.2f}%")
                if st.session_state.use_taxes:
                    summary_data["Metric"].append("Tax Impact")
                    summary_data["Value"].append(f"{result['tax_impact'] * 100:.2f}%")
                if st.session_state.use_inflation:
                    adjusted_return = result['return'] - result['tax_impact'] - result['cost_penalty'] - (inflation_rate if st.session_state.use_inflation else 0.0)
                    summary_data["Metric"].append("Adjusted Return (after inflation)")
                    summary_data["Value"].append(f"{adjusted_return * 100:.2f}%")
                st.table(pd.DataFrame(summary_data))

        with tabs[1]:  # Weights
            st.write("### Portfolio Weights")
            for method, result in results.items():
                st.write(f"#### {method}")
                weights_data = {
                    "Asset": st.session_state.asset_names,
                    "Weight (%)": [f"{w * 100:.2f}" for w in result['weights']]
                }
                st.table(pd.DataFrame(weights_data))

        with tabs[2]:  # Charts
            st.write("### Efficient Frontier with Capital Allocation Line")
            num_simulations = 1000
            sim_weights = [optimizer._initialize_weights() for _ in range(num_simulations)]
            sim_returns = [np.dot(w, returns + (dividend_yields if st.session_state.use_dividends else np.zeros_like(returns))) for w in sim_weights]
            sim_vols = [np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) for w in sim_weights]
            fig_frontier = go.Figure()
            fig_frontier.add_trace(go.Scatter(
                x=[v * 100 for v in sim_vols],
                y=[r * 100 for r in sim_returns],
                mode='markers',
                name='Portfolios',
                marker=dict(color='blue', opacity=0.3)
            ))
            for method, result in results.items():
                fig_frontier.add_trace(go.Scatter(
                    x=[result['volatility'] * 100],
                    y=[result['return'] * 100],
                    mode='markers',
                    name=f"{method} Portfolio",
                    marker=dict(size=10)
                ))
                # Add CAL
                cal_vols = np.linspace(0, max(sim_vols) * 100 * 1.5, 100)
                cal_returns = [(risk_free_rate + (result['return'] - risk_free_rate) / (result['volatility'] * 100) * v) * 100 for v in cal_vols]
                fig_frontier.add_trace(go.Scatter(
                    x=cal_vols,
                    y=cal_returns,
                    mode='lines',
                    name=f"{method} CAL",
                    line=dict(dash='dash')
                ))
            fig_frontier.update_layout(
                title='Efficient Frontier with Capital Allocation Line',
                xaxis_title='Volatility (%)',
                yaxis_title='Return (%)',
                template='plotly_dark' if st.get_option("theme.base") == "dark" else 'plotly',
                width=800
            )
            st.plotly_chart(fig_frontier, use_container_width=True)

            # Download frontier plot
            buffer = io.BytesIO()
            fig_frontier.write_png(buffer)
            st.download_button(
                label="Download Efficient Frontier as PNG",
                data=buffer,
                file_name="efficient_frontier.png",
                mime="image/png"
            )

            st.write("### Portfolio Weights")
            for method, result in results.items():
                st.write(f"#### {method}")
                fig_bar = go.Figure(data=[
                    go.Bar(x=st.session_state.asset_names, y=[w * 100 for w in result['weights']])
                ])
                fig_bar.update_layout(
                    title=f"Weights - {method}",
                    xaxis_title="Assets",
                    yaxis_title="Weight (%)",
                    template='plotly_dark' if st.get_option("theme.base") == "dark" else 'plotly'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                buffer = io.BytesIO()
                fig_bar.write_png(buffer)
                st.download_button(
                    label=f"Download {method} Weights as PNG",
                    data=buffer,
                    file_name=f"weights_{method.lower().replace(' ', '_')}.png",
                    mime="image/png"
                )

        with tabs[3]:  # Settings
            st.write("### Input Settings")
            settings_data = {
                "Parameter": ["Number of Assets", "Input Format", "Optimization Method", "Allow Short Selling"],
                "Value": [num_assets, input_format, optimization_method, st.session_state.allow_short_selling]
            }
            if optimization_method in ["Genetic Algorithm", "Compare All"]:
                settings_data["Parameter"].append("GA Iterations")
                settings_data["Value"].append(ga_iterations)
            if optimization_method in ["Monte Carlo", "Compare All"]:
                settings_data["Parameter"].append("Monte Carlo Iterations")
                settings_data["Value"].append(monte_carlo_iterations)
            st.table(pd.DataFrame(settings_data))

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
