"""
Streamlit UI configuration and layout components.

Contains functions for setting up the page configuration, sidebar controls,
and main interface layout elements.
"""

import streamlit as st
import pandas as pd
from .models import HAVE_PMDARIMA, HAVE_PROPHET, HAVE_LGBM


def setup_page_config():
    """Configure the Streamlit page settings and styling."""
    st.set_page_config(
        page_title="Multimodel Forecaster", 
        layout="wide",
        page_icon="🎯",
        initial_sidebar_state="expanded"
    )
    
    # Configure Altair for better chart rendering
    import altair as alt
    alt.data_transformers.enable('json')
    alt.theme.enable('default')
    
    # Suppress warnings for cleaner output
    import warnings
    warnings.filterwarnings('ignore')


def create_sidebar_controls():
    """Create and return all sidebar control values."""
    st.sidebar.markdown("---")
    
    # Forecast settings
    with st.sidebar.container():
        st.markdown("### 📈 **Forecast Settings**")
        horizon = st.number_input("📅 Months to forecast", 6, 120, 12, 3)
        
        st.markdown("### 🤖 **Select Models**")
        available_models = ["SARIMA", "ETS", "Poly-2", "Poly-3"]
        if HAVE_PROPHET:
            available_models.append("Prophet")
        if HAVE_PMDARIMA:
            available_models.append("Auto-ARIMA")
        if HAVE_LGBM:
            available_models.append("LightGBM")
        
        models_selected = st.multiselect(
            "Choose forecasting models",
            available_models,
            default=available_models,
            help="Select which models to run. More models = better comparison but longer runtime."
        )
        
        # Prophet-specific settings
        if HAVE_PROPHET and "Prophet" in models_selected:
            st.markdown("#### 🎯 **Prophet Settings**")
            enable_prophet_holidays = st.checkbox(
                "🎉 Include holiday effects",
                value=False,
                help="Add US and worldwide holiday seasonality to Prophet model for better accuracy around holidays"
            )
        else:
            enable_prophet_holidays = False

    st.sidebar.markdown("---")
    
    # Non-Compliant Upfront RevRec upload section
    yearly_renewals_file = create_renewals_upload_section()
    
    st.sidebar.markdown("---")
    
    # Business adjustments
    business_config = create_business_adjustments_section()

    # Accuracy & Validation (separate, auto-opened)
    accuracy_validation_config = create_accuracy_validation_section()
    
    # Advanced controls
    advanced_config = create_advanced_options_section()
    
    # Package status
    create_package_status_section()
    
    return {
        'horizon': horizon,
        'models_selected': models_selected,
        'enable_prophet_holidays': enable_prophet_holidays,
        'yearly_renewals_file': yearly_renewals_file,
        **business_config,
        **accuracy_validation_config,
        **advanced_config
    }


def create_renewals_upload_section():
    """Create the non-compliant upfront RevRec upload section."""
    with st.sidebar.expander("📋 **Non-Compliant Upfront RevRec Upload**", expanded=False):
        st.markdown("**Non-Compliant Revenue Recognition**")
        st.caption("Upload Excel file with non-compliant Upfront RevRec actuals to add as separate line items in historical data")
        
        yearly_renewals_file = st.file_uploader(
            "Upload Non-Compliant Upfront RevRec Excel",
            type=["xls", "xlsx", "xlsb"],
            key="yearly_renewals_uploader",
            help="Excel file with Date, Product, and ACR columns (same format as main data)"
        )
        
        if yearly_renewals_file:
            st.success(f"✅ Uploaded: {yearly_renewals_file.name}")
            st.session_state['yearly_renewals_file'] = yearly_renewals_file
            st.session_state['yearly_renewals_filename'] = yearly_renewals_file.name
        else:
            # Clear session state if no file is uploaded
            if 'yearly_renewals_file' in st.session_state:
                del st.session_state['yearly_renewals_file']
            if 'yearly_renewals_filename' in st.session_state:
                del st.session_state['yearly_renewals_filename']
    
    return yearly_renewals_file


def create_business_adjustments_section():
    """Create business adjustments controls and return configuration."""
    with st.sidebar.expander("📊 **Business Adjustments**", expanded=False):
        apply_business_adjustments = st.checkbox("Enable business adjustments", False)
        
        if apply_business_adjustments:
            business_growth_assumption = st.slider(
                "📈 Annual growth %", -50, 100, 0, 5,
                help="Expected annual growth rate to apply to forecasts"
            )
            market_conditions = st.selectbox(
                "🌍 Market conditions", 
                ["Stable", "Growth", "Contraction", "Uncertain"],
                help="Overall market outlook affecting forecasts"
            )
            
            # Market condition multipliers
            market_multipliers = {
                "Stable": 1.0,
                "Growth": 1.05,  # 5% boost
                "Contraction": 0.95,  # 5% reduction
                "Uncertain": 0.98  # 2% reduction for uncertainty
            }
            market_multiplier = market_multipliers[market_conditions]
        else:
            business_growth_assumption = 0
            market_conditions = "Stable"
            market_multiplier = 1.0

        # Divider remains for clarity, but accuracy section is separate now
        st.markdown("---")

    return {
        'apply_business_adjustments': apply_business_adjustments,
        'business_growth_assumption': business_growth_assumption,
        'market_conditions': market_conditions,
        'market_multiplier': market_multiplier,
    }


def create_accuracy_validation_section():
    """Create Accuracy & Validation controls in a separate expander and return configuration."""
    with st.sidebar.expander("🎯 **Accuracy & Validation**", expanded=True):
        st.caption("Advanced validation is always applied to double‑check accuracy with past data. This takes longer but improves confidence.")
        improve_accuracy = True  # Always enabled

        method_label = st.selectbox(
            "Validation method",
            ["Automatic (recommended)", "Walk‑forward", "Cross‑validation"],
            index=0,
            help=(
                "Automatic picks the most reliable signal from walk‑forward and cross‑validation. "
                "Walk‑forward simulates forecasting month‑by‑month (real‑world behavior). "
                "Cross‑validation checks multiple time splits for stability across eras."
            )
        )

        with st.expander("Backtesting settings (optional)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                backtest_gap = st.number_input(
                    "Leakage gap (months)", min_value=0, max_value=6, value=1, step=1,
                    help=(
                        "Small buffer between training end and test start to avoid peeking ahead. "
                        "Use 0–1 for most cases; use 2 if you have reporting lag or end‑of‑month spikes."
                    )
                )
            with c2:
                validation_horizon = st.number_input(
                    "Validation horizon (months)", min_value=1, max_value=24, value=12, step=1,
                    help=(
                        "How far ahead each backtest predicts. 12 = full seasonality (default). "
                        "Use 6 for near‑term focus or shorter history; 3 for very near‑term; 18–24 only with lots of history."
                    )
                )

            # Quick recommendations
            st.caption(
                "• Most users: horizon = 12, gap = 1.  • Short history: horizon = 6 (or 3), gap = 0.  "
                "• Heavy seasonality or long history: horizon = 12–18, gap = 1–2."
            )

        # Fiscal calendar configuration (used for seasonal diagnostics)
        months = [
            (1, "January"), (2, "February"), (3, "March"), (4, "April"), (5, "May"), (6, "June"),
            (7, "July"), (8, "August"), (9, "September"), (10, "October"), (11, "November"), (12, "December")
        ]
        month_labels = [m[1] for m in months]
        default_index = 6  # July
        selected_label = st.selectbox(
            "Fiscal year start month",
            options=month_labels,
            index=default_index,
            help="For seasonal diagnostics: sets which month is counted as fiscal month 1."
        )
        fiscal_year_start_month = next(m for m, label in months if label == selected_label)

    # Derive flags used by the pipeline
    enable_advanced_validation = True
    if method_label.startswith("Automatic"):
        enable_walk_forward = bool(enable_advanced_validation)
        enable_cross_validation = bool(enable_advanced_validation)
    elif method_label.startswith("Walk"):
        enable_walk_forward = True
        enable_cross_validation = False
    else:
        enable_walk_forward = False
        enable_cross_validation = True

    return {
        'enable_advanced_validation': enable_advanced_validation,
        'enable_walk_forward': enable_walk_forward,
        'enable_cross_validation': enable_cross_validation,
    'backtest_gap': int(backtest_gap),
    'validation_horizon': int(validation_horizon),
    'fiscal_year_start_month': int(fiscal_year_start_month)
    }


def create_advanced_options_section():
    """Create advanced options controls and return configuration."""
    with st.sidebar.expander("⚙️ **Advanced Options**", expanded=False):
        confidence_intervals = st.checkbox("Show confidence intervals", True)
        enable_statistical_validation = st.checkbox(
            "Statistical validation", True,
            help="Apply statistical bounds to prevent extreme outliers"
        )
        
        enable_business_aware_selection = st.checkbox(
            "Business-aware model selection", True,
            help="Prioritize business-appropriate models over pure MAPE optimization. Recommended for consumptive business revenue forecasting."
        )

    if not enable_statistical_validation:
        st.caption("⚠️ Raw model outputs (no bounds)")
    
    if not enable_business_aware_selection:
        st.caption("📊 Pure MAPE optimization (may favor polynomial models)")
    else:
        st.caption("🏢 Considers business context (deprioritizes polynomial for revenue)")

    return {
            'confidence_intervals': confidence_intervals,
            'enable_statistical_validation': enable_statistical_validation,
            'enable_business_aware_selection': enable_business_aware_selection
        }


def create_package_status_section():
    """Display package availability status."""
    with st.sidebar.expander("📦 **Available Models**", expanded=False):
        st.markdown("**Core Models:** ✅ Always available")
        st.caption("• SARIMA • ETS • Poly-2 • Poly-3")
        
        st.markdown("**Advanced Models:**")
        if HAVE_PMDARIMA:
            st.markdown("✅ Auto-ARIMA")
        else:
            st.markdown("❌ Auto-ARIMA")
            
        if HAVE_PROPHET:
            st.markdown("✅ Prophet (with optional holidays)")
        else:
            st.markdown("❌ Prophet")
            
        if HAVE_LGBM:
            st.markdown("✅ LightGBM")
        else:
            st.markdown("❌ LightGBM")

        if not (HAVE_PMDARIMA and HAVE_PROPHET and HAVE_LGBM):
            st.caption("💡 Run SETUP.bat to install missing packages")
            if st.button("🔄 Refresh Package Detection", help="Click if you just installed packages"):
                st.rerun()


def create_main_tabs():
    """Create and return the main interface tabs."""
    return st.tabs([
        "🎯 **Forecast**", 
        "📊 **Example Data**", 
        "📚 **Model Guide**"
    ])


def display_data_requirements():
    """Display the data requirements section."""
    st.markdown("#### 📊 **Data Requirements**")
    
    # Compact requirement cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background-color: #f0f8f0; padding: 0.5rem; border-radius: 6px; border-left: 3px solid #28a745;">
        <h5 style="color: #28a745; margin: 0; font-size: 0.9rem;">🟢 Excellent (36+ months)</h5>
        <p style="margin: 0.3rem 0; font-size: 0.8rem;"><strong>MAPE:</strong> 5-15%</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background-color: #fff8e1; padding: 0.5rem; border-radius: 6px; border-left: 3px solid #ffc107;">
        <h5 style="color: #e67e00; margin: 0; font-size: 0.9rem;">🟡 Good (24-36 months)</h5>
        <p style="margin: 0.3rem 0; font-size: 0.8rem;"><strong>MAPE:</strong> 10-20%</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div style="background-color: #ffeaa7; padding: 0.5rem; border-radius: 6px; border-left: 3px solid #fdcb6e;">
        <h5 style="color: #e17055; margin: 0; font-size: 0.9rem;">🟠 Limited (18-24 months)</h5>
        <p style="margin: 0.3rem 0; font-size: 0.8rem;"><strong>MAPE:</strong> 15-30%</p>
        </div>
        """, unsafe_allow_html=True)


def display_advanced_requirements():
    """Display the expandable advanced requirements section."""
    with st.expander("🔍 **Advanced Requirements & Model Details**", expanded=False):
        st.markdown("""
        **Model-Specific Data Needs:**
        - **SARIMA & Auto-ARIMA**: 36+ months optimal for seasonal patterns
        - **Prophet**: 24+ months minimum, 36+ months recommended  
        - **ETS**: 24+ months for seasonal decomposition
        - **Polynomial Models**: 12-18 months minimum (less data sensitive)
        - **LightGBM**: 18+ months for feature engineering
        
        **💡 Pro Tip:** With limited data, Polynomial models are most reliable while seasonal models may struggle.
        
        **⚠️ Business Context Warning:** For consumptive business revenue forecasting, polynomial models may produce misleading forecasts despite low MAPE. Enable "Business-Aware Model Selection" in Advanced Options to automatically deprioritize these models for revenue products.
        """)


def display_upload_section():
    """Display the file upload section and return uploaded file and run button status."""
    st.markdown("### 📁 **Upload Your Data**")
    
    # Streamlined Purview warning
    st.info("""
    **📋 File Requirements:**
    • Excel file with columns: **"Date", "Product", "ACR"**
    • Must be labeled as **"General"** (not Confidential/Highly Confidential)
    • Check the top of Excel ribbon for Purview classification labels
    """)
    
    uploaded = st.file_uploader(
        "Choose your Excel file", 
        type=["xls", "xlsx", "xlsb"],
        help="Upload Excel with Date, Product, ACR columns"
    )
    
    # Clean run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_btn = st.button("🚀 **Run Forecasts**", type="primary", use_container_width=True)
    
    return uploaded, run_btn


def initialize_session_state():
    """Initialize all session state variables."""
    session_vars = [
        'forecast_results', 'forecast_mapes', 'forecast_sarima_params', 
        'diagnostic_messages', 'uploaded_filename', 'business_aware_selection_used',
        'product_mapes', 'best_models_per_product', 'best_mapes_per_product',
        'business_adjustments_applied', 'business_growth_used', 
        'market_conditions_used', 'yearly_renewals_applied'
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None


def clear_session_state():
    """Clear all forecast-related session state for new forecasts."""
    keys_to_clear = [
        'forecast_results', 'forecast_mapes', 'product_mapes', 
        'best_models_per_product', 'best_mapes_per_product', 
        'forecast_sarima_params', 'diagnostic_messages', 'uploaded_filename', 
        'business_adjustments_applied', 'business_growth_used', 
        'market_conditions_used', 'adjusted_forecast_results', 
        'product_adjustments_applied', 'yearly_renewals_applied',
        'business_aware_selection_used'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
