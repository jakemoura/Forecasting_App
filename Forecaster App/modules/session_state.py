"""
Session state management functions for the Streamlit application.

Contains functions for storing, retrieving, and managing forecast results
and application state across Streamlit sessions.
"""

import streamlit as st
import numpy as np


def store_forecast_results(results, avg_mapes, avg_smapes, avg_rmses, avg_mases, 
                          product_mapes, product_smapes, product_mases, product_rmses,
                          best_models_per_product, best_mapes_per_product, 
                          sarima_params, diagnostic_messages, uploaded_filename,
                          business_config, yearly_renewals_applied, 
                          enable_business_aware_selection, hybrid_df):
    """
    Store all forecast results and configurations in session state.
    
    Args:
        results: Dictionary of model results
        avg_mapes, avg_smapes, avg_rmses, avg_mases: Average metrics
        product_mapes, product_smapes, product_mases, product_rmses: Per-product metrics
        best_models_per_product: Dict of best model per product
        best_mapes_per_product: Dict of best MAPE per product
        sarima_params: SARIMA parameters dict
        diagnostic_messages: List of diagnostic messages
        uploaded_filename: Name of uploaded file
        business_config: Business adjustment configuration
        yearly_renewals_applied: Whether yearly renewals were applied
        enable_business_aware_selection: Business-aware selection setting
        hybrid_df: Hybrid model results DataFrame
    """
    # Store main results
    st.session_state.forecast_results = results
    st.session_state.forecast_mapes = avg_mapes
    st.session_state.forecast_smapes = avg_smapes
    st.session_state.forecast_mases = avg_mases
    st.session_state.forecast_rmses = avg_rmses
    
    # Store per-product metrics
    st.session_state.product_mapes = product_mapes
    st.session_state.product_smapes = product_smapes
    st.session_state.product_mases = product_mases
    st.session_state.product_rmses = product_rmses
    
    # Store best model selections
    st.session_state.best_models_per_product = best_models_per_product
    st.session_state.best_mapes_per_product = best_mapes_per_product
    # Also persist dual mapping variants if pipeline provided them earlier
    if 'best_models_per_product_standard' in st.session_state:
        st.session_state.best_models_per_product_standard = st.session_state.get('best_models_per_product_standard')
    if 'best_models_per_product_backtesting' in st.session_state:
        st.session_state.best_models_per_product_backtesting = st.session_state.get('best_models_per_product_backtesting')
    
    # Store model parameters and diagnostics
    st.session_state.forecast_sarima_params = sarima_params
    st.session_state.diagnostic_messages = diagnostic_messages
    st.session_state.uploaded_filename = uploaded_filename
    
    # Store business configuration / forecast adjustment (post-refactor)
    # Legacy keys (apply_business_adjustments, business_growth_assumption, market_conditions) were removed
    # when forecast_conservatism slider replaced them. We defensively fallback for backward compatibility.
    st.session_state.forecast_conservatism_used = business_config.get('forecast_conservatism', 100)
    st.session_state.business_adjustments_applied = business_config.get('apply_business_adjustments', False)
    st.session_state.business_growth_used = business_config.get('business_growth_assumption', 0)
    st.session_state.market_conditions_used = business_config.get('market_conditions', 'N/A')
    st.session_state.business_aware_selection_used = enable_business_aware_selection
    
    # Store additional flags
    st.session_state.yearly_renewals_applied = yearly_renewals_applied
    
    # Add hybrid model to results if available
    if hybrid_df is not None:
        # Calculate average MAPE for hybrid model
        hybrid_avg_mape = np.mean(list(best_mapes_per_product.values()))
        st.session_state.forecast_mapes["Best per Product"] = hybrid_avg_mape
        st.session_state.forecast_results["Best per Product"] = hybrid_df

    # Capture baseline (unadjusted) results for dynamic conservatism slider
    # If the original run used a conservatism factor != 100, reconstruct baseline by dividing forecast rows
    try:
        factor_used = st.session_state.get('forecast_conservatism_used', 100)
        if 'baseline_forecast_results' not in st.session_state or st.session_state.get('baseline_forecast_results') is None:
            if isinstance(st.session_state.forecast_results, dict):
                baseline = {}
                for model_name, model_df in st.session_state.forecast_results.items():
                    # Handle DataFrame or list of DataFrames; store pass-through for unknown types
                    if factor_used != 100 and hasattr(model_df, 'copy') and hasattr(model_df, 'columns'):
                        df_copy = model_df.copy()
                        if 'Type' in df_copy.columns:
                            mask = df_copy['Type'] == 'forecast'
                            if mask.any():
                                df_copy.loc[mask, 'ACR'] = df_copy.loc[mask, 'ACR'] / (factor_used/100.0)
                        baseline[model_name] = df_copy
                    elif factor_used != 100 and isinstance(model_df, list):
                        adj_list = []
                        for sub in model_df:
                            if hasattr(sub, 'copy') and hasattr(sub, 'columns') and 'Type' in sub.columns:
                                sub_c = sub.copy()
                                mask = sub_c['Type'] == 'forecast'
                                if mask.any():
                                    sub_c.loc[mask,'ACR'] = sub_c.loc[mask,'ACR'] / (factor_used/100.0)
                                adj_list.append(sub_c)
                            else:
                                adj_list.append(sub)
                        baseline[model_name] = adj_list
                    else:
                        # factor 100 or unknown type – store as-is
                        baseline[model_name] = model_df
                st.session_state.baseline_forecast_results = baseline
            else:
                st.session_state.baseline_forecast_results = None
    except Exception:
        # Fail silently; dynamic adjustment will fallback to current adjusted results
        st.session_state.baseline_forecast_results = None


def clear_session_state_for_new_forecast():
    """Clear all forecast-related session state for new forecasts."""
    keys_to_clear = [
        'forecast_results', 'forecast_mapes', 'product_mapes', 
        'best_models_per_product', 'best_mapes_per_product', 
        'forecast_sarima_params', 'diagnostic_messages', 'uploaded_filename', 
        'business_adjustments_applied', 'business_growth_used', 
        'market_conditions_used', 'adjusted_forecast_results', 
        'product_adjustments_applied', 'yearly_renewals_applied',
        'business_aware_selection_used', 'forecast_smapes', 
        'forecast_mases', 'forecast_rmses', 'product_smapes',
        'product_mases', 'product_rmses',
        'best_models_per_product_standard', 'best_models_per_product_backtesting',
        'data_context',  # Clear data context for fresh analysis
        'baseline_forecast_results', 'forecast_conservatism_used'  # Clear baseline to rebuild with new forecast timing
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def initialize_session_state_variables():
    """Initialize all session state variables if they don't exist."""
    session_vars = [
        'forecast_results', 'forecast_mapes', 'forecast_smapes', 
        'forecast_mases', 'forecast_rmses', 'forecast_sarima_params', 
        'diagnostic_messages', 'uploaded_filename', 'business_aware_selection_used',
        'product_mapes', 'product_smapes', 'product_mases', 'product_rmses',
        'best_models_per_product', 'best_mapes_per_product',
        'business_adjustments_applied', 'business_growth_used', 
        'market_conditions_used', 'yearly_renewals_applied',
        'forecast_conservatism_used'
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None


def has_existing_results():
    """Check if there are existing forecast results in session state."""
    return st.session_state.get('forecast_results') is not None


def get_forecast_configuration():
    """
    Get the current forecast configuration from session state.
    
    Returns:
        dict: Configuration dictionary with all forecast settings
    """
    return {
        'business_adjustments_applied': st.session_state.get('business_adjustments_applied', False),
        'business_growth_used': st.session_state.get('business_growth_used', 0),
        'market_conditions_used': st.session_state.get('market_conditions_used', 'Stable'),
        'forecast_conservatism_used': st.session_state.get('forecast_conservatism_used', 100),
        'business_aware_selection_used': st.session_state.get('business_aware_selection_used', False),
        'yearly_renewals_applied': st.session_state.get('yearly_renewals_applied', False),
        'uploaded_filename': st.session_state.get('uploaded_filename', 'Unknown')
    }


def get_model_performance_data():
    """
    Get model performance data from session state.
    
    Returns:
        dict: Performance data including all metrics
    """
    return {
        'forecast_mapes': st.session_state.get('forecast_mapes', {}),
        'forecast_smapes': st.session_state.get('forecast_smapes', {}),
        'forecast_mases': st.session_state.get('forecast_mases', {}),
        'forecast_rmses': st.session_state.get('forecast_rmses', {}),
        'product_mapes': st.session_state.get('product_mapes', {}),
        'product_smapes': st.session_state.get('product_smapes', {}),
        'product_mases': st.session_state.get('product_mases', {}),
        'product_rmses': st.session_state.get('product_rmses', {}),
        'best_models_per_product': st.session_state.get('best_models_per_product', {}),
        'best_mapes_per_product': st.session_state.get('best_mapes_per_product', {}),
        'forecast_sarima_params': st.session_state.get('forecast_sarima_params', {}),
        'diagnostic_messages': st.session_state.get('diagnostic_messages', [])
    }
