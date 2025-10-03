# main.py
import streamlit as st
from optimizer import optimize_portfolio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.title("Portfolio Optimizer")

# Step 1: Number of assets and names
num_assets = st.number_input("Enter number of assets", min_value=2, max_value=20, value=2)
asset_names = []
for i in range(num_assets):
    asset_names.append(st.text_input(f"Asset {i+1} name", value=f"Asset {i+1}"))

# Step 2: Optimization method
methods = ["Monte Carlo", "Genetic Algorithm", "SciPy Optimize (Mean-Variance)"]
method = st.selectbox("Select optimization method", methods)

# Step 3: Metrics selection and input
st.header("Metrics Configuration")
st.write("Enable/disable metrics and provide values. Some metrics are per asset, others global or pairwise.")

# Global metrics
use_inflation = st.checkbox("Use Inflation", value=True)
inflation = st.number_input("Inflation rate (as decimal, e.g., 0.03 for 3%)", value=0.03) if use_inflation else 0.0

use_tax_rate = st.checkbox("Use Tax Rate", value=True)
tax_rate = st.number_input("Tax rate (as decimal, e.g., 0.20 for 20%)", value=0.20) if use_tax_rate else 0.0

# Per asset metrics
expected_returns = []
volatilities = []
dividend_yields = []

st.header("Per Asset Metrics")
for i, asset in enumerate(asset_names):
    st.subheader(asset)
    exp_ret = st.number_input(f"Expected yearly return for {asset} (decimal)", value=0.10)
    vol = st.number_input(f"Volatility for {asset} (decimal)", value=0.15)
    div_yield = st.number_input(f"Dividend yield for {asset} (decimal)", value=0.02)
    
    expected_returns.append(exp_ret)
    volatilities.append(vol)
    dividend_yields.append(div_yield)

# Pairwise correlations
use_correlations = st.checkbox("Use Correlations", value=True)
correlations = np.eye(num_assets)  # Default to identity matrix (no correlation)
if use_correlations:
    st.header("Correlation Matrix")
    corr_df = pd.DataFrame(np.eye(num_assets), index=asset_names, columns=asset_names)
    for i in range(num_assets):
        for j in range(i+1, num_assets):
            corr = st.number_input(f"Correlation between {asset_names[i]} and {asset_names[j]}", value=0.0, min_value=-1.0, max_value=1.0)
            corr_df.iloc[i, j] = corr
            corr_df.iloc[j, i] = corr
    correlations = corr_df.values

# Other important metrics I suggest adding: Sharpe Ratio (calculated), Beta (if market provided), but for now, we'll calculate Sharpe post-optimization.
# Assuming risk-free rate for Sharpe
use_sharpe = st.checkbox("Calculate Sharpe Ratio", value=True)
risk_free_rate = st.number_input("Risk-free rate (decimal)", value=0.02) if use_sharpe else 0.0

# Input format: percentage or decimal
input_format = st.radio("Input numbers as:", ("Decimal (e.g., 0.05)", "Percentage (e.g., 5)"))
if input_format == "Percentage (e.g., 5)":
    scale = 0.01
else:
    scale = 1.0

# Apply scale if needed
if scale == 0.01:
    expected_returns = [r * scale for r in expected_returns]
    volatilities = [v * scale for v in volatilities]
    dividend_yields = [d * scale for d in dividend_yields]
    inflation *= scale
    tax_rate *= scale
    risk_free_rate *= scale

# Number of iterations
iterations = st.number_input("Number of iterations for optimization", min_value=100, max_value=100000, value=10000)

# Optimize button
if st.button("Optimize Portfolio"):
    with st.spinner("Optimizing..."):
        # Call optimizer
        weights, metrics = optimize_portfolio(
            method=method,
            expected_returns=np.array(expected_returns),
            volatilities=np.array(volatilities),
            correlations=correlations,
            dividend_yields=np.array(dividend_yields),
            inflation=inflation,
            tax_rate=tax_rate,
            risk_free_rate=risk_free_rate,
            iterations=iterations,
            use_sharpe=use_sharpe,
            use_inflation=use_inflation,
            use_tax_rate=use_tax_rate
        )
    
    st.success("Optimization Complete!")
    
    # Display weights
    st.header("Optimized Portfolio Allocation")
    weights_df = pd.DataFrame({"Asset": asset_names, "Weight (%)": [w * 100 for w in weights]})
    st.table(weights_df)
    
    # Pie chart
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(weights * 100, labels=asset_names, autopct='%1.1f%%')
    ax_pie.set_title("Portfolio Allocation")
    st.pyplot(fig_pie)
    
    # Metrics display
    st.header("Portfolio Metrics")
    for key, value in metrics.items():
        st.write(f"{key}: {value:.4f}")
    
    # Correlation heatmap if used
    if use_correlations:
        st.header("Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(correlations, annot=True, cmap='coolwarm', ax=ax_corr)
        ax_corr.set_xticklabels(asset_names, rotation=45)
        ax_corr.set_yticklabels(asset_names, rotation=0)
        st.pyplot(fig_corr)
    
    # Efficient frontier or simulation plot for Monte Carlo / GA
    if method in ["Monte Carlo", "Genetic Algorithm"]:
        # Simulate some portfolios for frontier
        st.header("Efficient Frontier Simulation")
        portfolio_returns = []
        portfolio_vols = []
        for _ in range(1000):
            rand_weights = np.random.dirichlet(np.ones(num_assets))
            port_ret = np.dot(rand_weights, expected_returns)
            port_vol = np.sqrt(np.dot(rand_weights.T, np.dot(np.cov(np.array([volatilities] * num_assets) * correlations), rand_weights)))
            portfolio_returns.append(port_ret)
            portfolio_vols.append(port_vol)
        
        fig_ef, ax_ef = plt.subplots()
        ax_ef.scatter(portfolio_vols, portfolio_returns, c='blue', alpha=0.5)
        ax_ef.scatter(metrics['Portfolio Volatility'], metrics['Portfolio Return'], c='red', marker='*', s=200, label='Optimized')
        ax_ef.set_xlabel("Volatility")
        ax_ef.set_ylabel("Return")
        ax_ef.legend()
        st.pyplot(fig_ef)
