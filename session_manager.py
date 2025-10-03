import streamlit as st
import numpy as np

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

    def update(self, num_assets, beginner_mode=False):
        """Update session state for a given number of assets."""
        if len(st.session_state.asset_names) != num_assets:
            st.session_state.asset_names = [f"Asset {i+1}" for i in range(num_assets)] if not st.session_state.asset_names else st.session_state.asset_names[:num_assets]
            st.session_state.returns = [0.1] * num_assets if not st.session_state.returns or beginner_mode else st.session_state.returns[:num_assets]
            st.session_state.volatilities = [0.2] * num_assets if not st.session_state.volatilities or beginner_mode else st.session_state.volatilities[:num_assets]
            st.session_state.correlations = np.full((num_assets, num_assets), 0.3) if not st.session_state.correlations.size or beginner_mode else st.session_state.correlations[:num_assets, :num_assets]
            if beginner_mode:
                np.fill_diagonal(st.session_state.correlations, 1.0)
            st.session_state.transaction_costs = [0.0] * num_assets if not st.session_state.transaction_costs else st.session_state.transaction_costs[:num_assets]
            st.session_state.min_weights = [0.0] * num_assets if not st.session_state.min_weights else st.session_state.min_weights[:num_assets]
            st.session_state.max_weights = [1.0] * num_assets if not st.session_state.max_weights else st.session_state.max_weights[:num_assets]
            st.session_state.dividend_yields = [0.0] * num_assets if not st.session_state.dividend_yields else st.session_state.dividend_yields[:num_assets]
            st.session_state.tax_rates = [0.0] * num_assets if not st.session_state.tax_rates else st.session_state.tax_rates[:num_assets]
