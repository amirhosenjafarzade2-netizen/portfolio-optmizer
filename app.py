import streamlit as st
from optimizer import optimize_portfolio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set page config for better appearance
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# Initialize session state
if 'num_assets' not in st.session_state:
    st.session_state.num_assets = 2
if 'asset_names' not in st.session_state:
    st.session_state.asset_names = ["Asset 1", "Asset 2"]

# Title and description
st.title("Portfolio Optimizer")
st.markdown("Optimize your investment portfolio using Monte Carlo, Genetic Algorithm, or Gradient Descent methods. Enter asset details and metrics below.")

# Reset button
if st.button("Reset All Inputs"):
    st.session_state.clear()
    st.session_state.num_assets = 2
    st.session_state.asset_names = ["Asset 1", "Asset 2"]
    st.rerun()

# Step 1: Number of assets
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
    methods = ["Monte Carlo", "Genetic Algorithm", "Gradient Descent (Mean-Variance)"]
    method = st.selectbox(
        "Select optimization method",
        methods,
        help="Choose the algorithm to optimize your portfolio."
    )

# Step 3: Metrics configuration
with st.expander("Metrics Configuration", expanded=True):
    st.write("Enable/disable metrics and provide values for global and per-asset parameters.")

    # Global metrics
    use_inflation = st.checkbox("Use Inflation", value=True, help="Include inflation adjustment in metrics.")
    inflation = st.number_input(
        "Inflation rate (decimal, e.g., 0.03 for 3%)",
        value=0.03,
        min_value=0.0,
        max_value=0.5,
        help="Annual inflation rate as a decimal."
    ) if use_inflation else 0.0

    use_tax_rate = st.checkbox("Use Tax Rate", value=True, help="Include tax rate in metrics.")

    use_sharpe = st.checkbox(
        "Include Sharpe Ratio in Results",
        value=True,
        help="Calculate the Sharpe Ratio to evaluate risk-adjusted return (requires risk-free rate)."
    )
    if use_sharpe:
        st.markdown("**Risk-Free Rate**: Enter the return of a risk-free investment (e.g., Treasury bill yield) used to calculate the Sharpe Ratio.")
        risk_free_rate = st.number_input(
            "Risk-free rate (decimal)",
            value=0.02,
            min_value=0.0,
            max_value=0.5,
            help="Risk-free rate as a decimal (e.g., 0.02 for 2%). Used in Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility."
        )
    else:
        risk_free_rate = 0.0

# Per asset metrics
with st.expander("Per Asset Metrics", expanded=True):
    st.subheader("Asset Parameters")
    expected_returns = []
    volatilities = []
    dividend_yields = []
    tax_rates = []
    cols = st.columns(2)
    for i, asset in enumerate(asset_names):
        with cols[i % 2]:
            st.markdown(f"**{asset}**")
            exp_ret = st.number_input(
                f"Expected return (decimal)",
                value=0.10,
                min_value=-1.0,
                max_value=1.0,
                key=f"ret_{i}",
                help=f"Expected annual return for {asset} as a decimal."
            )
            vol = st.number_input(
                f"Volatility (decimal)",
                value=0.15,
                min_value=0.01,
                max_value=1.0,
                key=f"vol_{i}",
                help=f"Annual volatility for {asset} as a decimal."
            )
            div_yield = st.number_input(
                f"Dividend yield (decimal)",
                value=0.02,
                min_value=0.0,
                max_value=0.5,
                key=f"div_{i}",
                help=f"Dividend yield for {asset} as a decimal."
            )
            tax_rate = st.number_input(
                f"Tax rate (decimal, e.g., 0.20 for 20%)",
                value=0.20,
                min_value=0.0,
                max_value=1.0,
                key=f"tax_{i}",
                help=f"Tax rate for {asset} as a decimal."
            ) if use_tax_rate else 0.0
            expected_returns.append(exp_ret)
            volatilities.append(vol)
            dividend_yields.append(div_yield)
            tax_rates.append(tax_rate)

# Pairwise correlations
with st.expander("Correlation Matrix", expanded=False):
    use_correlations = st.checkbox("Use Correlations", value=True, help="Include asset correlations in optimization.")
    correlations = np.eye(num_assets)
    if use_correlations:
        st.subheader("Correlation Matrix")
        default_corr = st.checkbox("Use default correlation (0.2 for all pairs)", value=False, help="Set all pairwise correlations to 0.2.")
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
                            key=f"corr_{i}_{j}",
                            help=f"Correlation coefficient between {asset_names[i]} and {asset_names[j]}."
                        )
                        corr_df.iloc[i, j] = corr
                        corr_df.iloc[j, i] = corr
            correlations = corr_df.values

# Input format
with st.expander("Input Format", expanded=True):
    input_format = st.radio(
        "Input numbers as:",
        ("Decimal (e.g., 0.05)", "Percentage (e.g., 5)"),
        help="Choose whether inputs are in decimal (e.g., 0.05 for 5%) or percentage (e.g., 5 for 5%)."
    )
    scale = 0.01 if input_format == "Percentage (e.g., 5)" else 1.0

# Apply scale if needed
if scale == 0.01:
    expected_returns = [r * scale for r in expected_returns]
    volatilities = [v * scale for v in volatilities]
    dividend_yields = [d * scale for d in dividend_yields]
    tax_rates = [t * scale for t in tax_rates]
    inflation *= scale
    risk_free_rate *= scale

# Number of iterations
iterations = st.number_input(
    "Number of iterations for optimization",
    min_value=100,
    max_value=5000,
    value=1000,
    help="Number of iterations for the optimization algorithm (higher values increase computation time)."
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
    ax.scatter(opt_vol, opt_return, c='red', marker='*', s=300, label='Optimized Portfolio', edgecolors='black', linewidth=1.5)
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Efficient Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

# Optimize button
if st.button("Optimize Portfolio", help="Run the portfolio optimization with the specified parameters."):
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
                use_tax_rate=use_tax_rate
            )
            progress_bar.progress(1.0)
        except ValueError as ve:
            st.error(f"Input error: {str(ve)}")
            st.stop()
        except np.linalg.LinAlgError:
            st.error("Numerical error: Invalid correlation matrix or singular covariance matrix.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.stop()
    
    st.success("Optimization Complete!")
    
    # Display weights
    with st.expander("Optimized Portfolio Allocation", expanded=True):
        weights_df = pd.DataFrame({"Asset": asset_names, "Weight (%)": [w * 100 for w in weights]})
        st.table(weights_df)
    
        # Pie chart
        fig_pie = plot_pie(weights, asset_names)
        st.pyplot(fig_pie)
        plt.close(fig_pie)
    
    # Metrics display
    with st.expander("Portfolio Metrics", expanded=True):
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")
        st.table(metrics_df)
    
    # Correlation heatmap if used
    if use_correlations:
        with st.expander("Correlation Heatmap", expanded=False):
            fig_corr = plot_heatmap(correlations, asset_names)
            st.pyplot(fig_corr)
            plt.close(fig_corr)
    
    # Efficient frontier simulation
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
                progress_bar.progress((i + 1) / 500)
            
            fig_ef = plot_efficient_frontier(portfolio_returns, portfolio_vols, metrics['Portfolio Return'], metrics['Portfolio Volatility'], asset_names)
            st.pyplot(fig_ef)
            plt.close(fig_ef)
