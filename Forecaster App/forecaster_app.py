"""
Multimodel Time-Series Forecaster with Statistical Rigor

Mathematically grounded approach focusing on:
1. PROPER MODEL SELECTION: Enhanced parameter search with statistical validation
2. STATIONARITY ENFORCEMENT: Ensuring models meet statistical assumptions  
3. MINIMAL BUSINESS CONSTRAINTS: Only prevent clearly unrealistic outcomes (negatives)
4. DIAGNOSTIC VALIDATION: Comprehensive analysis and model adequacy checks

Philosophy: Let the statistics drive the forecasts, apply minimal business logic only where mathematically justified.

Author: Jake Moura
"""

import io
import warnings
import streamlit as st

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Too few observations")

# Import our modular components
from modules.ui_config import setup_page_config, create_sidebar_controls
from modules.tab_content import render_forecast_tab, render_example_data_tab, render_model_guide_tab, render_footer
from modules.data_validation import validate_data_format, prepare_data, analyze_data_quality, display_data_analysis_results, display_date_format_error, get_valid_products
from modules.forecasting_pipeline import run_forecasting_pipeline
from modules.business_logic import process_yearly_renewals, calculate_model_rankings, find_best_models_per_product, create_hybrid_best_model
from modules.session_state import store_forecast_results, initialize_session_state_variables
from modules.utils import read_any_excel
from modules.models import HAVE_PMDARIMA, HAVE_PROPHET, HAVE_LGBM
from modules.ui_components import display_backtesting_results, display_data_context_summary


def main():
    """Main application entry point."""
    # Setup page configuration
    setup_page_config()
    
    # Initialize session state
    initialize_session_state_variables()
    
    # Create sidebar controls and get configuration
    controls_config = create_sidebar_controls()
    
    # Create main tabs
    tab_forecast, tab_example, tab_glossary = st.tabs([
        "🎯 **Forecast**", 
        "📊 **Example Data**", 
        "📚 **Model Guide**"
    ])
    
    # Render forecast tab (main functionality)
    with tab_forecast:
        uploaded, run_btn = render_forecast_tab(controls_config)
        
        # Process forecast if file uploaded and run button pressed
        if run_btn and uploaded:
            process_forecast(uploaded, controls_config)
    
    # Render other tabs
    with tab_example:
        render_example_data_tab()
    
    with tab_glossary:
        render_model_guide_tab()
    
    # Render footer
    render_footer()


def process_forecast(uploaded, config):
    """
    Process the uploaded file and run forecasting pipeline.
    
    Args:
        uploaded: Streamlit uploaded file object
        config: Configuration dictionary from sidebar controls
    """
    try:
        # Check if data analysis was already completed during upload
        if st.session_state.get('data_analysis_complete') and 'processed_data' in st.session_state:
            # Use the already processed data
            raw = st.session_state['processed_data']
            st.success("✅ **Using previously analyzed data**")
        else:
            # Fallback: Read and validate data (shouldn't happen normally)
            st.warning("⚠️ **Re-analyzing data** - this shouldn't happen normally")
            raw = read_any_excel(io.BytesIO(uploaded.read()))
            validate_data_format(raw)
            raw = prepare_data(raw)
        
        st.markdown("---")

        # Get valid products for forecasting
        diagnostic_messages = []
        diagnostic_messages.append(
            f"🎯 Smart Forecasting: Testing {len(config['models_selected'])} models on data products for optimal product-by-product selection"
        )
        valid_products = get_valid_products(raw, diagnostic_messages)
        products = raw["Product"].unique()

        # Get backtesting settings from sidebar config
        enable_backtesting = bool(config.get('enable_backtesting', True))

        # Run the forecasting pipeline
        pipeline_results = run_forecasting_pipeline(
            raw_data=raw,
            models_selected=config['models_selected'],
            horizon=config['horizon'],
            enable_statistical_validation=config['enable_statistical_validation'],
            apply_business_adjustments=config['apply_business_adjustments'],
            business_growth_assumption=config['business_growth_assumption'],
            market_multiplier=config['market_multiplier'],
            market_conditions=config['market_conditions'],
            enable_business_aware_selection=config.get('enable_business_aware_selection', False),
            enable_prophet_holidays=config['enable_prophet_holidays'],
            enable_backtesting=enable_backtesting,
            use_backtesting_selection=True,
            backtest_months=int(config.get('backtest_months', 15)),  # Changed default from 12 to 15
            backtest_gap=int(config.get('backtest_gap', 0)),
            validation_horizon=int(config.get('validation_horizon', 3)),  # default quarterly
            fiscal_year_start_month=int(config.get('fiscal_year_start_month', 1)),
            # Enhanced rolling validation parameters (configurable)
            enable_enhanced_rolling=bool(config.get('enable_enhanced_rolling', True)),
            min_train_size=int(config.get('enhanced_min_train_size', 12)),
            max_train_size=int(config.get('enhanced_max_train_size', 18)),
            recency_alpha=float(config.get('enhanced_recency_alpha', 0.6))
            ,enable_expanding_cv=bool(config.get('enable_expanding_cv', False))
        )

        # Unpack pipeline results (now includes backtesting results)
        (
            results,
            avg_mapes,
            sarima_params,
            pipeline_diagnostics,
            additional_metrics,
            best_models_per_product,
            best_mapes_per_product,
            backtesting_results,
        ) = pipeline_results

        # Merge diagnostic messages
        diagnostic_messages.extend(pipeline_diagnostics)

        # Process yearly renewals if uploaded
        yearly_renewals_applied = process_yearly_renewals(results, diagnostic_messages)

        # Extract additional metrics
        avg_smapes = additional_metrics['smapes']
        avg_mases = additional_metrics['mases']
        avg_rmses = additional_metrics['rmses']

        # Get per-product metrics from session state (set by pipeline)
        product_mapes = st.session_state.get('product_mapes', {})
        product_smapes = st.session_state.get('product_smapes', {})
        product_mases = st.session_state.get('product_mases', {})
        product_rmses = st.session_state.get('product_rmses', {})

        # Safety check: ensure all metrics dictionaries are not None
        if not product_mapes:
            product_mapes = {}
        if not product_smapes:
            product_smapes = {}
        if not product_mases:
            product_mases = {}
        if not product_rmses:
            product_rmses = {}

        # Calculate model rankings across multiple metrics (using session state data)
        model_names = list(results.keys())
        metric_ranks, avg_ranks, best_model_by_rank = calculate_model_rankings(
            product_mapes, product_smapes, product_mases, product_rmses, model_names, products
        )

        # Store ranking data in session state for downstream use
        st.session_state.model_avg_ranks = avg_ranks
        st.session_state.best_model_by_rank = best_model_by_rank

        # Create hybrid "Best per Product" model
        hybrid_df = create_hybrid_best_model(
            results, best_models_per_product, best_mapes_per_product, products
        )

        # Store all results in session state
        store_forecast_results(
            results=results,
            avg_mapes=avg_mapes,
            avg_smapes=avg_smapes,
            avg_rmses=avg_rmses,
            avg_mases=avg_mases,
            product_mapes=product_mapes,
            product_smapes=product_smapes,
            product_mases=product_mases,
            product_rmses=product_rmses,
            best_models_per_product=best_models_per_product,
            best_mapes_per_product=best_mapes_per_product,
            sarima_params=sarima_params,
            diagnostic_messages=diagnostic_messages,
            uploaded_filename=uploaded.name,
            business_config=config,
            yearly_renewals_applied=yearly_renewals_applied,
            enable_business_aware_selection=config['enable_business_aware_selection'],
            hybrid_df=hybrid_df,
        )

        # Persist the full sidebar configuration for downstream UI hints/meta
        try:
            st.session_state.business_config = config
        except Exception:
            pass

        # Persist backtesting results for display after rerun
        st.session_state.backtesting_results = backtesting_results if backtesting_results else {}

        # Refresh page to show results
        st.rerun()

    except Exception as e:
        st.error(f"❌ **Error processing forecast:** {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
