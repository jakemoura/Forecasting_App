"""
Streamlit UI configuration and layout components.

Contains functions for setting up the page configuration, sidebar controls,
and main interface layout elements.
"""

import streamlit as st
import pandas as pd
import io
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
    # Check if sidebar should update
    sidebar_update_trigger = st.session_state.get('sidebar_update_trigger', False)
    analysis_complete = st.session_state.get('data_analysis_complete', False)
    data_context = st.session_state.get('data_context', {})
    
    # Use a unique key that changes when sidebar should update
    # Force re-render when data context changes
    data_context_str = str(data_context.get('min_months', 0)) + str(data_context.get('max_months', 0)) + str(analysis_complete)
    sidebar_key = f"sidebar_{data_context_str}_{analysis_complete}"
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", key="debug_mode")
    if debug_mode:
        st.sidebar.caption("Debug mode enabled - check for detailed info")
    
    st.sidebar.markdown("---")
    
    # Forecast settings
    with st.sidebar.container(key=sidebar_key):
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
    accuracy_validation_config = create_accuracy_validation_section(debug_mode)
    
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


def create_accuracy_validation_section(debug_mode=False):
    """Create Accuracy & Validation controls with smart backtesting recommendations."""
    with st.sidebar.expander("🎯 **Accuracy & Validation**", expanded=True):
        st.caption("Smart backtesting validates models against historical data for better accuracy. More months = more robust validation.")
        
        # Get enhanced data context for smart recommendations
        data_context = st.session_state.get('data_context', {})
        recommendations = data_context.get('backtesting_recommendations', {})
        data_quality = data_context.get('data_quality_score', {})
        
        # Check if data analysis is complete
        analysis_complete = st.session_state.get('data_analysis_complete', False)
        
        # Debug: Show what we're getting
        if debug_mode:
            st.caption(f"🔍 Debug: data_context keys: {list(data_context.keys()) if data_context else 'None'}")
            st.caption(f"🔍 Debug: recommendations: {recommendations}")
            st.caption(f"🔍 Debug: analysis_complete: {analysis_complete}")
        
        # Smart backtesting slider with data-driven recommendations
        st.markdown("#### 📊 **Backtesting Period**")
        st.markdown("")  # Add spacing
        
        if analysis_complete and recommendations:
            # Use data-driven recommendations
            min_value = recommendations.get('min_value', 6)
            max_value = recommendations.get('max_value', 24)
            default_value = recommendations.get('default_value', 12)
            
            # Clean, compact data quality status
            status_icon = recommendations.get('icon', '📊')
            status_title = recommendations.get('title', 'Data Analysis')
            status_desc = recommendations.get('description', 'Analyzing data...')
            
            if recommendations.get('status') == 'limited':
                st.warning(f"{status_icon} **{status_title}**: {status_desc}")
            elif recommendations.get('status') == 'moderate':
                st.info(f"{status_icon} **{status_title}**: {status_desc}")
            elif recommendations.get('status') == 'good':
                st.success(f"{status_icon} **{status_title}**: {status_desc}")
            else:
                st.success(f"{status_icon} **{status_title}**: {status_desc}")
            
            # Compact recommendation
            st.caption(f"💡 {recommendations.get('message', 'Use data-driven backtesting')}")
            
            # Simplified metrics in one row
            if data_quality and 'score' in data_quality:
                score = data_quality['score']
                grade = data_quality.get('grade', 'N/A')
                consistency = data_quality.get('consistency_ratio', 0)
                
                st.caption(f"**Quality**: {grade} ({score}/100) | **Consistency**: {consistency:.0%} | **Products**: {data_context.get('total_products', 0)}")
        elif analysis_complete:
            # Analysis complete but no recommendations - show status
            st.success("✅ **Data Analysis Complete!**")
            st.caption("Processing recommendations...")
            
            # Use calculated recommendations if available, otherwise fallback
            if recommendations:
                min_value = recommendations.get('min_value', 6)
                max_value = recommendations.get('max_value', 24)
                default_value = recommendations.get('default_value', 12)
            else:
                # Fallback values
                min_value = 6
                max_value = 24
                default_value = 12
        else:
            # No data uploaded yet - show waiting state
            st.info("📤 **Please upload data for recommendations**")
            
            # Disable slider until data is uploaded
            min_value = 6
            max_value = 24
            default_value = 12
        
        # Only enable slider if analysis is complete
        slider_disabled = not analysis_complete
        
        st.markdown("")  # Add spacing before slider
        backtest_months = st.slider(
            "Backtest last X months",
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=1,
            disabled=slider_disabled,
            help=f"Smart recommendation: {recommendations.get('recommended_range', '6-24 months') if analysis_complete else 'Upload data first'} based on your data"
        )
        
        # Simple backtesting - no advanced settings needed
        # The main slider controls everything for simple validation

        # Fiscal calendar configuration (used for seasonal diagnostics)
        st.markdown("#### 📅 **Fiscal Calendar**")
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

    # Simple backtesting flag - always enabled for reliable validation
    enable_backtesting = True

    return {
        'enable_backtesting': enable_backtesting,
        'backtest_months': int(backtest_months),
        'backtest_gap': 1,  # Fixed at 1 month for simple validation
        'validation_horizon': 6,  # Fixed at 6 months for reasonable validation
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
    
    # AUTO-ANALYSIS: Run immediately after upload
    if uploaded is not None:
        try:
            # Import here to avoid circular imports
            from .utils import read_any_excel
            from .data_validation import validate_data_format, prepare_data, analyze_data_quality, display_data_analysis_results
            from .ui_components import display_data_context_summary
            
            # Read and validate data
            raw = read_any_excel(io.BytesIO(uploaded.read()))
            validate_data_format(raw)
            
            # Prepare data
            try:
                raw = prepare_data(raw)
                
                # Run data analysis immediately
                st.markdown("### 🔍 **Data Analysis & Backtesting Recommendations**")
                with st.spinner("Analyzing your data for optimal backtesting recommendations..."):
                    data_analysis, overall_status = analyze_data_quality(raw)
                    display_data_analysis_results(data_analysis, overall_status)
                    # Get debug mode from session state
                    current_debug_mode = st.session_state.get('debug_mode', False)
                    display_data_context_summary(current_debug_mode)
                
                # Show ready message
                st.success("✅ **Data Analysis Complete!** Check the sidebar for smart backtesting recommendations.")
                
                # Store uploaded file info for debugging
                st.session_state['last_uploaded_file'] = uploaded.name
                st.session_state['last_uploaded_size'] = uploaded.size
                
                # Store the processed data for the forecast pipeline
                st.session_state['processed_data'] = raw
                st.session_state['data_analysis_complete'] = True
                st.session_state['sidebar_update_trigger'] = True
                
            except ValueError as e:
                st.error(f"❌ **Data Error**: {str(e)}")
                return uploaded, None
                
        except Exception as e:
            st.error(f"❌ **Upload Error**: {str(e)}")
            return uploaded, None
    
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
