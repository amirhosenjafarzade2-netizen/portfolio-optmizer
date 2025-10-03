import streamlit as st
from optimizer import optimize_portfolio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from stochastic import simulate_scenarios
from multiperiod import multiperiod_simulation
from options_greeks import calculate_portfolio_greeks

# Set page config for better appearance
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# Initialize session state
if 'num_assets' not in st.session_state:
    st.session_state.num_assets = 2
if 'asset_names' not in st.session_state:
    st.session_state.asset_names = ["Asset 1", "Asset 2"]

# Title and description
st.title("Portfolio Optimizer")
st.markdown("Optimize your investment portfolio using various methods. Enter asset details and metrics below.")
st.warning("This is for educational purposes only. Not financial advice. Consult a professional for real investments.")

# Save/Load inputs
col1, col2 = st.columns(2)
with col1:
    # Filter session state to include only serializable keys
    serializable_state = {k: v for k, v in st.session_state.items() if k.startswith(('num_assets', 'asset_names', 'ret_', 'vol_', 'div_', 'tax_', 'corr_', 'price_', 'strike_', 'ttm_', 'ivol_', 'opt_type_'))}
    st.download_button("Save Inputs", data=json.dumps(serializable_state), file_name="portfolio_inputs.json")
with col2:
    uploaded_file = st.file_uploader("Load Inputs", type="json")
    if uploaded_file:
        try:
            data = json.load(uploaded_file)
            # Only update valid keys to avoid injecting unexpected data
            valid_keys = set(st.session_state.keys()) | set(['num_assets', 'asset_names'] + [f'ret_{i}' for i in range(10)] + [f'vol_{i}' for i in range(10)] + [f'div_{i}' for i in range(10)] + [f'tax_{i}' for i in range(10)] + [f'corr_{i}_{j}' for i in range(10) for j in range(i+1, 10)] + [f'price_{i}' for i in range(10)] + [f'strike_{i}' for i in range(10)] + [f'ttm_{i}' for i in range(10)] + [f'ivol_{i}' for i in range(10)] + [f'opt_type_{i}' for i in range(10)])
            for k, v in data.items():
                if k in valid_keys:
                    st.session_state[k] = v
            st.rerun()
        except Exception as e:
            st.error(f"Error loading inputs: {str(e)}")

# Reset button
if st.button("Reset All Inputs"):
    st.session_state.clear()
    st.session_state.num_assets = 2
    st.session_state.asset_names = ["Asset 1", "Asset 2"]
    st.rerun()

# Step 1: Asset Configuration
with st.expander("Asset Configuration", expanded=True):
    num_assets = st.number_input(
        "Enter number of assets",
        min_value=2,
        max_value=10,
        key="num_assets",
        help="Select the number of assets in your portfolio (2-10)."
    )

    # Update asset_names list if num_assets changes
    if num_assets != len(st.session_state.asset_names):
        st.session_state.asset_names = [f"Asset {i+1}" for i in range(num_assets)]

    # Asset names input
    st.subheader("Asset Names")
    cols = st.columns(2)
    asset_names = []
    for i in range(num_assets):
        with cols[i % 2]:
            name = st.text_input(
                f"Asset {i+1} name",
                value=st.session_state.asset_names[i],
                key=f"asset_{i}",
                help=f"Enter the name for asset {i+1}."
            )
            asset_names.append(name)
    st.session_state.asset_names = asset_names

# Step 2: Optimization method
with st.expander("Optimization Method", expanded=True):
    methods = ["Monte Carlo", "Genetic Algorithm", "Gradient Descent (Mean-Variance)", "SciPy (Constrained)"]
    method = st.selectbox(
        "Select optimization method",
        methods,
        help="Choose the algorithm to optimize your portfolio."
    )

# Constraints
with st.expander("Constraints", expanded=False):
    use_constraints = st.checkbox("Use weight constraints", value=False)
    if use_constraints:
        min_weight = st.number_input("Minimum weight per asset", min_value=0.0, max_value=0.5, value=0.0)
        max_weight = st.number_input("Maximum weight per asset", min_value=0.0, max_value=1.0, value=1.0)
    else:
        min_weight = max_weight = None

# Step 3: Metrics configuration
with st.expander("Metrics Configuration", expanded=True):
    st.write("Enable/disable metrics and provide values for global and per-asset parameters.")

    # Global metrics
    use_inflation = st.checkbox("Use Inflation", value=True)
    inflation = st.number_input(
        "Inflation rate (decimal)",
        value=0.03,
        min_value=0.0,
        max_value=0.5
    ) if use_inflation else 0.0

    use_tax_rate = st.checkbox("Use Tax Rate", value=True)

    use_sharpe = st.checkbox("Include Sharpe Ratio", value=True)
    if use_sharpe:
        risk_free_rate = st.number_input(
            "Risk-free rate (decimal)",
            value=0.02,
            min_value=0.0,
            max_value=0.5
        )
    else:
        risk_free_rate = 0.0

    use_advanced_metrics = st.checkbox("Include Advanced Metrics (VaR, Sortino)", value=False)

    use_stochastic = st.checkbox("Stochastic Mode", value=False, help="Simulate scenarios with random inputs.")
    if use_stochastic:
        num_simulations = st.number_input("Number of simulations", min_value=100, max_value=10000, value=1000)
        std_factor = st.number_input("Uncertainty factor for std (e.g., 0.2 means std = 0.2 * mean)", value=0.2, min_value=0.0, max_value=1.0)

    use_multiperiod = st.checkbox("Multi-Period Mode", value=False, help="Simulate over time horizon with rebalancing.")
    if use_multiperiod:
        horizon = st.number_input("Investment horizon (years)", min_value=1, max_value=50, value=5)
        rebalance_freq = st.selectbox("Rebalance frequency", ["Annual", "Quarterly", "Monthly"], index=0)
        num_mp_sim = st.number_input("Number of multi-period simulations", min_value=100, max_value=5000, value=1000)

# Per asset metrics
with st.expander("Per Asset Metrics", expanded=True):
    st.subheader("Asset Parameters")
    expected_returns = []
    volatilities = []
    dividend_yields = []
    tax_rates = []
    asset_prices = []
    strike_prices = []
    times_to_maturity = []
    implied_vols = []
    option_types = []
    cols = st.columns(2)
    for i, asset in enumerate(asset_names):
        with cols[i % 2]:
            st.markdown(f"**{asset}**")
            exp_ret = st.number_input(
                f"Expected return (decimal)",
                value=0.10,
                min_value=-1.0,
                max_value=1.0,
                key=f"ret_{i}"
            )
            vol = st.number_input(
                f"Volatility (decimal)",
                value=0.15,
                min_value=0.01,
                max_value=1.0,
                key=f"vol_{i}"
            )
            div_yield = st.number_input(
                f"Dividend yield (decimal)",
                value=0.02,
                min_value=0.0,
                max_value=0.5,
                key=f"div_{i}"
            )
            tax_rate = st.number_input(
                f"Tax rate (decimal)",
                value=0.20,
                min_value=0.0,
                max_value=1.0,
                key=f"tax_{i}"
            ) if use_tax_rate else 0.0
            use_options_greeks = st.checkbox("Use Options Greeks", value=False, help="Calculate Greeks for options on each asset.")
            if use_options_greeks:
                asset_price = st.number_input(
                    f"Current asset price",
                    value=100.0,
                    min_value=0.01,
                    key=f"price_{i}"
                )
                strike_price = st.number_input(
                    f"Option strike price",
                    value=100.0,
                    min_value=0.01,
                    key=f"strike_{i}"
                )
                time_to_maturity = st.number_input(
                    f"Time to maturity (years)",
                    value=1.0,
                    min_value=0.01,
                    max_value=10.0,
                    key=f"ttm_{i}"
                )
                implied_vol = st.number_input(
                    f"Implied volatility (decimal)",
                    value=vol,
                    min_value=0.01,
                    max_value=1.0,
                    key=f"ivol_{i}"
                )
                option_type = st.selectbox(
                    f"Option type",
                    ["call", "put"],
                    index=0,
                    key=f"opt_type_{i}"
                )
            else:
                asset_price = strike_price = time_to_maturity = implied_vol = None
                option_type = "call"
            expected_returns.append(exp_ret)
            volatilities.append(vol)
            dividend_yields.append(div_yield)
            tax_rates.append(tax_rate)
            asset_prices.append(asset_price)
            strike_prices.append(strike_price)
            times_to_maturity.append(time_to_maturity)
            implied_vols.append(implied_vol)
            option_types.append(option_type)

# Correlation Matrix
with st.expander("Correlation Matrix", expanded=False):
    use_correlations = st.checkbox("Use Correlations", value=True)
    correlations = np.eye(num_assets)
    if use_correlations:
        st.subheader("Correlation Matrix")
        default_corr = st.checkbox("Use default correlation (0.2 for all pairs)", value=False)
        if default_corr:
            correlations = np.ones((num_assets, num_assets)) * 0.2
            np.fill_diagonal(correlations, 1.0)
        else:
            corr_df = pd.DataFrame(np.eye(num_assets), index=asset_names, columns=asset_names)
            cols = st.columns(2)
            for i in range(num_assets):
                for j in range(i+1, num_assets):
                    with cols[(i+j) % 2]:
                        corr = st.number_input(
                            f"Correlation: {asset_names[i]} - {asset_names[j]}",
                            value=0.0,
                            min_value=-1.0,
                            max_value=1.0,
                            key=f"corr_{i}_{j}"
                        )
                        corr_df.iloc[i, j] = corr
                        corr_df.iloc[j, i] = corr
            correlations = corr_df.values

# Input format
with st.expander("Input Format", expanded=True):
    input_format = st.radio(
        "Input numbers as:",
        ("Decimal (e.g., 0.05)", "Percentage (e.g., 5)")
    )
    scale = 0.01 if input_format == "Percentage (e.g., 5)" else 1.0

# Apply scale
if scale == 0.01:
    expected_returns = [r * scale for r in expected_returns]
    volatilities = [v * scale for v in volatilities]
    dividend_yields = [d * scale for d in dividend_yields]
    tax_rates = [t * scale for t in tax_rates]
    inflation *= scale
    risk_free_rate *= scale
    implied_vols = [v * scale if v is not None else None for v in implied_vols]

# Number of iterations
iterations = st.number_input(
    "Number of iterations for optimization",
    min_value=100,
    max_value=5000,
    value=1000
)

# Cache plot functions
@st.cache_data
def plot_pie(weights, asset_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(weights * 100, labels=asset_names, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax.set_title("Portfolio Allocation")
    return fig

@st.cache_data
def plot_heatmap(correlations, asset_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', ax=ax, xticklabels=asset_names, yticklabels=asset_names, vmin=-1, vmax=1, center=0)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    return fig

@st.cache_data
def plot_efficient_frontier(portfolio_returns, portfolio_vols, opt_return, opt_vol, asset_names):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(portfolio_vols, portfolio_returns, c='blue', alpha=0.3, s=10, label='Random Portfolios')
    ax.scatter(opt_vol, opt_return, c='red', marker='*', s=300, label='Optimized Portfolio')
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

@st.cache_data
def plot_histogram(data, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=30, color='blue', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    return fig

# Optimize button
if st.button("Optimize Portfolio"):
    progress_bar = st.progress(0)
    with st.spinner("Optimizing..."):
        try:
            weights, metrics = optimize_portfolio(
                method=method,
                expected_returns=np.array(expected_returns),
                volatilities=np.array(volatilities),
                correlations=correlations,
                dividend_yields=np.array(dividend_yields),
                inflation=inflation,
                tax_rate=np.array(tax_rates),
                risk_free_rate=risk_free_rate,
                iterations=iterations,
                use_sharpe=use_sharpe,
                use_inflation=use_inflation,
                use_tax_rate=use_tax_rate,
                use_advanced_metrics=use_advanced_metrics,
                min_weight=min_weight if use_constraints else None,
                max_weight=max_weight if use_constraints else None
            )
            progress_bar.progress(0.5)

            if use_options_greeks and all(v is not None for v in asset_prices + strike_prices + times_to_maturity + implied_vols):
                greeks = calculate_portfolio_greeks(
                    weights, np.array(asset_prices), np.array(strike_prices),
                    np.array(times_to_maturity), np.array(implied_vols),
                    risk_free_rate, option_types
                )
                metrics.update(greeks)

            if use_stochastic:
                from stochastic import simulate_scenarios
                return_stds = np.array(expected_returns) * std_factor
                vol_stds = np.array(volatilities) * std_factor
                div_stds = np.array(dividend_yields) * std_factor
                stochastic_metrics = simulate_scenarios(
                    weights, np.array(expected_returns), return_stds, np.array(volatilities), vol_stds, correlations,
                    np.array(dividend_yields), div_stds, inflation, np.array(tax_rates), risk_free_rate,
                    use_sharpe, use_inflation, use_tax_rate, use_advanced_metrics, num_simulations
                )
                metrics.update(stochastic_metrics)

            if use_multiperiod:
                from multiperiod import multiperiod_simulation
                mp_metrics = multiperiod_simulation(
                    weights, np.array(expected_returns), np.array(volatilities), correlations,
                    horizon, rebalance_freq, num_mp_sim
                )
                metrics.update(mp_metrics)

            progress_bar.progress(1.0)
        except ValueError as ve:
            st.error(f"Input error: {str(ve)}")
            st.stop()
        except np.linalg.LinAlgError:
            st.error("Numerical error: Invalid correlation matrix.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.stop()

    st.success("Optimization Complete!")

    # Display weights
    with st.expander("Optimized Portfolio Allocation", expanded=True):
        weights_df = pd.DataFrame({"Asset": asset_names, "Weight (%)": [w * 100 for w in weights]})
        st.table(weights_df)

        fig_pie = plot_pie(weights, asset_names)
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    # Metrics display
    with st.expander("Portfolio Metrics", expanded=True):
        if not use_stochastic:
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")
            st.table(metrics_df)
        else:
            # Display stochastic as table with percentiles
            sto_df = pd.DataFrame(metrics).T  # rows metrics, columns mean p5 p50 p95
            st.table(sto_df)

    # Correlation heatmap
    if use_correlations:
        with st.expander("Correlation Heatmap", expanded=False):
            fig_corr = plot_heatmap(correlations, asset_names)
            st.pyplot(fig_corr)
            plt.close(fig_corr)

    # Efficient frontier
    if method in ["Monte Carlo", "Genetic Algorithm"]:
        with st.expander("Efficient Frontier Simulation", expanded=False):
            portfolio_returns = []
            portfolio_vols = []
            cov_matrix = np.diag(volatilities) @ correlations @ np.diag(volatilities)

            for i in range(500):
                rand_weights = np.random.dirichlet(np.ones(num_assets))
                port_ret = np.dot(rand_weights, expected_returns)
                port_vol = np.sqrt(np.dot(rand_weights.T, np.dot(cov_matrix, rand_weights)))
                portfolio_returns.append(port_ret)
                portfolio_vols.append(port_vol)

            fig_ef = plot_efficient_frontier(portfolio_returns, portfolio_vols, metrics.get('Portfolio Return', 0), metrics.get('Portfolio Volatility', 0), asset_names)
            st.pyplot(fig_ef)
            plt.close(fig_ef)

    # Stochastic histograms if enabled
    if use_stochastic:
        with st.expander("Stochastic Distributions", expanded=False):
            fig_ret = plot_histogram(metrics['Portfolio Return']['samples'], "Distribution of Portfolio Returns")
            st.pyplot(fig_ret)
            plt.close(fig_ret)

    # Multiperiod if enabled
    if use_multiperiod:
        with st.expander("Multi-Period Simulation", expanded=False):
            fig_wealth = plot_histogram(metrics['Final Wealth']['samples'], "Distribution of Final Portfolio Wealth")
            st.pyplot(fig_wealth)
            plt.close(fig_wealth)
