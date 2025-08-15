"""
Multimodel Time-Series Forecaster with Statistical Rigor

Mathematically grounded approach focusing on:
1. PROPER MODEL SELECTION: Enhanced parameter search with statistical validation
2. STATIONARITY ENFORCEMENT: Ensuring models meet statistical assumptions  
3. MINIMAL BUSINESS CONSTRAINTS: Only prevent clearly unrealistic outcomes (negatives)
4. DIAGNOSTIC VALIDATION: Comprehensive analysis and model adequacy checks

Philosophy: Let the statistics drive the forecasts, apply minimal business logic only where mathematically justified.

Author: Jake Moura (jakemoura@microsoft.com)
"""

# Type: ignore - Suppress type checker warnings for data science libraries
# mypy: disable-error-code="import-untyped,attr-defined,arg-type,assignment"
# pyright: reportOptionalMemberAccess=false, reportGeneralTypeIssues=false

import io
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Import sklearn for remaining polynomial features in main file
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning

# Import statsmodels for ETS
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Optional deps status from modules
from modules.models import HAVE_PMDARIMA, HAVE_PROPHET, HAVE_LGBM

# Import utility functions
from modules.utils import read_any_excel, coerce_month_start, debug_chart_data

# Import UI components
from modules.ui_components import (
    display_product_forecast, normalized_monthly_by_days, display_forecast_results,
    create_download_excel, display_model_comparison_table,
    create_adjustment_controls, display_diagnostic_messages, fy
)

# Import forecasting pipeline
from modules.forecasting_pipeline import run_forecasting_pipeline

# Import metrics functions
from modules.metrics import calculate_validation_metrics

# Import models and statistical functions
from modules.models import (
    apply_statistical_validation, apply_business_adjustments_to_forecast
)

# ============ Beautiful Page Setup ============
st.set_page_config(
    page_title="Multimodel Forecaster", 
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# Configure Altair for better chart rendering
alt.data_transformers.enable('json')
alt.theme.enable('default')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============ CLEAN SIDEBAR UI ============

st.sidebar.markdown("---")

# Import modeling functions
from modules.models import (
    detect_seasonality_strength, get_seasonality_aware_split, 
    apply_trend_aware_forecasting, fit_final_sarima_model,
    fit_best_sarima, fit_best_lightgbm, select_business_aware_best_model
)

# If you want to force re-check packages, uncomment this:
# if st.sidebar.button("🔄 Refresh Package Detection"):
#     st.experimental_rerun()

# ============ CLEAN SIDEBAR UI ============
# ─────────────────────── SIDEBAR CONTROLS ─────────────────────
st.sidebar.markdown("---")

# Forecast settings in a clean container
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
        # Store the uploaded file in session state for processing
        st.session_state['yearly_renewals_file'] = yearly_renewals_file
        st.session_state['yearly_renewals_filename'] = yearly_renewals_file.name
    else:
        # Clear session state if no file is uploaded
        if 'yearly_renewals_file' in st.session_state:
            del st.session_state['yearly_renewals_file']
        if 'yearly_renewals_filename' in st.session_state:
            del st.session_state['yearly_renewals_filename']

st.sidebar.markdown("---")

# Business adjustments in a collapsible section
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

# Advanced controls in a collapsible section
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

st.sidebar.markdown("---")

# Package status in a clean, collapsible format
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

# ─────────────────────── MAIN INTERFACE ─────────────────────
# Clean, modern tab layout
tab_forecast, tab_example, tab_glossary = st.tabs([
    "🎯 **Forecast**", 
    "📊 **Example Data**", 
    "📚 **Model Guide**"
])

# --- Clean Forecast Tab ---
with tab_forecast:
    # Clean, modern header
    st.markdown("# 🎯 **Multimodel Time‑Series Forecaster**")
    st.markdown("---")
    
    # Check if we have existing results to show
    has_results = st.session_state.get('forecast_results') is not None
    
    if has_results:
        # Clean results header - only show results and new forecast button
        st.success("🎉 **Forecast Complete!** Your results are ready below.")
        
        # Show controls for existing results in a clean layout
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown("### 📊 **Results Dashboard**")
            st.caption("Use the sections below to explore your forecasts, download data, and view model comparisons.")
        with col2:
            if st.button("🔄 **New Forecast**", type="secondary", use_container_width=True):
                # Clear existing results
                for key in ['forecast_results', 'forecast_mapes', 'product_mapes', 'best_models_per_product', 'best_mapes_per_product', 'forecast_sarima_params', 'diagnostic_messages', 'uploaded_filename', 'business_adjustments_applied', 'business_growth_used', 'market_conditions_used', 'adjusted_forecast_results', 'product_adjustments_applied', 'yearly_renewals_applied']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
                
        st.markdown("---")
        # Display the existing results
        display_forecast_results()
        
    else:        # Show upload interface for new forecasts
        # Compact data requirements overview
        with st.container():
            st.markdown("#### 📊 **Data Requirements**")
            
            # Compact requirement cards with smaller padding and font
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
        
        # Compact expandable details
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
        
        # Clean file upload section
        with st.container():
            st.markdown("### 📁 **Upload Your Data**")
            
            # Streamlined Purview warning in a cleaner format
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
        
        # Initialize session state for results
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = None
        if 'forecast_mapes' not in st.session_state:
            st.session_state.forecast_mapes = None
        if 'forecast_sarima_params' not in st.session_state:
            st.session_state.forecast_sarima_params = None
        if 'diagnostic_messages' not in st.session_state:
            st.session_state.diagnostic_messages = None
        if 'uploaded_filename' not in st.session_state:
            st.session_state.uploaded_filename = None
        if 'business_aware_selection_used' not in st.session_state:
            st.session_state.business_aware_selection_used = None

        if run_btn and not uploaded:
            st.warning("Please upload a workbook first.")
            st.stop()

        if run_btn and uploaded:
            # Clear previous results when running new forecasts
            st.session_state.forecast_results = None
            st.session_state.forecast_mapes = None
            st.session_state.product_mapes = None
            st.session_state.best_models_per_product = None
            st.session_state.best_mapes_per_product = None
            st.session_state.forecast_sarima_params = None
            st.session_state.diagnostic_messages = None
            st.session_state.uploaded_filename = None
            st.session_state.business_adjustments_applied = False
            st.session_state.business_growth_used = 0
            st.session_state.market_conditions_used = "Stable"
            st.session_state.business_aware_selection_used = None
            st.session_state.yearly_renewals_applied = False
            
            raw = read_any_excel(io.BytesIO(uploaded.read()))
            required = {"Date", "Product", "ACR"}
            if not required.issubset(raw.columns):
                st.error(f"Workbook must contain columns: {required}"); st.stop()
            
            # === CLEAN DATA VALIDATION SECTION ===
            st.markdown("---")
            st.markdown("### 📊 **Data Analysis Results**")
            
            # Prepare data first for validation
            try:
                raw["Date"] = coerce_month_start(raw["Date"])
                raw.sort_values("Date", inplace=True)
            except ValueError as e:
                st.error(f"❌ **Date Format Error**")
                st.markdown(f"**Issue:** {str(e)}")
                
                with st.expander("📖 **Supported Date Formats**", expanded=True):
                    st.markdown("""
                    **Examples of supported formats:**
                    - `Apr-22`, `May-22` (interpreted as 2022)
                    - `Apr-2022`, `May-2022`  
                    - `April-22`, `May-22`
                    - `04/2022`, `05/2022`
                    - Standard Excel date formats
                    """)
                st.stop()
            
            # Analyze data by product with beautiful status indicators
            data_analysis = []
            overall_status = "good"
            
            for product, grp in raw.groupby("Product"):
                try:
                    series = grp.set_index("Date")["ACR"].astype(float)
                    if not isinstance(series.index, pd.DatetimeIndex):
                        series.index = pd.to_datetime(series.index)
                    series.index = series.index.to_period("M").to_timestamp(how="start")
                    
                    months_count = len(series)
                    date_range = f"{series.index.min().strftime('%b %Y')} to {series.index.max().strftime('%b %Y')}"
                    
                    # Determine status with beautiful indicators
                    if months_count >= 36:
                        status_icon = "🟢"
                        status_text = "Excellent"
                        accuracy = "5-15%"
                        recommendation = "All models optimal"
                        status_color = "#28a745"
                    elif months_count >= 24:
                        status_icon = "🟡"
                        status_text = "Good" 
                        accuracy = "10-20%"
                        recommendation = "Most models work well"
                        status_color = "#ffc107"
                        if overall_status == "good":
                            overall_status = "moderate"
                    elif months_count >= 18:
                        status_icon = "🟠"
                        status_text = "Moderate"
                        accuracy = "15-30%"
                        recommendation = "Polynomial models recommended"
                        status_color = "#fd7e14"
                        overall_status = "limited"
                    else:
                        status_icon = "🔴"
                        status_text = "Insufficient"
                        accuracy = "20%+"
                        recommendation = "Need more data"
                        status_color = "#dc3545"
                        overall_status = "insufficient"
                    
                    data_analysis.append({
                        "Product": product,
                        "📊 Months": f"{months_count} months",
                        "📅 Date Range": date_range,
                        "🎯 Status": f"{status_icon} {status_text}",
                        "📈 Expected MAPE": accuracy,
                        "💡 Recommendation": recommendation
                    })
                    
                except Exception as e:
                    data_analysis.append({
                        "Product": product,
                        "📊 Months": "Error",
                        "📅 Date Range": "Invalid data",
                        "🎯 Status": "🔴 Error",
                        "📈 Expected MAPE": "N/A",
                        "💡 Recommendation": f"Data error: {str(e)[:30]}..."
                    })
                    overall_status = "insufficient"
            
            # Beautiful results table
            if data_analysis:
                analysis_df = pd.DataFrame(data_analysis)
                st.dataframe(
                    analysis_df, 
                    hide_index=True, 
                    use_container_width=True,
                    column_config={
                        "🎯 Status": st.column_config.TextColumn(width="medium"),
                        "📈 Expected MAPE": st.column_config.TextColumn(width="small"),
                        "💡 Recommendation": st.column_config.TextColumn(width="large")
                    }
                )
            
            # Clean overall recommendation with beautiful styling
            st.markdown("<br>", unsafe_allow_html=True)
            
            if overall_status == "good":
                st.success("🎉 **Excellent Data Quality** - Ready for high-accuracy forecasting!")
            elif overall_status == "moderate":
                st.info("✅ **Good Data Quality** - Should produce reliable forecasts with most models")
            elif overall_status == "limited":
                st.warning("⚠️ **Limited Data** - Basic forecasting possible. Consider Polynomial models for best results")
            else:
                st.error("❌ **Insufficient Data** - Need at least 18 months for reliable forecasting")
                st.stop()
            
            st.markdown("---")
            
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", message="Too few observations")
            
            # Build forecasts per model & pillar, and track SARIMA AIC params
            results, mapes, sarima_params = {}, {}, {}
            smapes, mases, rmses = {}, {}, {}  # Additional metric dictionaries
            diagnostic_messages = []  # Collect all diagnostic messages
            products = raw["Product"].unique()
            
            # Add diagnostic message about product-by-product approach
            diagnostic_messages.append(f"🎯 Smart Forecasting: Testing {len(models_selected)} models on {len(products)} data products for optimal product-by-product selection")
                            # Count valid lines first to get accurate total
            valid_products = []
            for product, grp in raw.groupby("Product"):
                try:
                    series = grp.set_index("Date")["ACR"].astype(float)
                    # Ensure index is DatetimeIndex before applying period operations
                    if not isinstance(series.index, pd.DatetimeIndex):
                        try:
                            series.index = pd.to_datetime(series.index)
                        except (ValueError, pd.errors.OutOfBoundsDatetime) as e:
                            diagnostic_messages.append(f"❌ Product {product}: Date parsing error - {str(e)[:100]}. Skipping.")
                            continue
                    series.index = series.index.to_period("M").to_timestamp(how="start")
                except Exception as e:
                    diagnostic_messages.append(f"❌ Product {product}: Data processing error - {str(e)[:100]}. Skipping.")
                    continue
                
                # Check if product has sufficient data (same check as main loop)
                if len(series) < 12:
                    continue
                
                # Check seasonality-aware split (same check as main loop)
                seasonality_strength = detect_seasonality_strength(series)
                split_result = get_seasonality_aware_split(series, seasonal_period=12, diagnostic_messages=None)
                if split_result[0] is not None:
                    valid_products.append(product)
            
            total = len(models_selected) * len(valid_products)
            prog = st.progress(0.0, text="Running models…")
            done = 0
            processed_products = 0  # Track how many products we've actually processed
            for product, grp in raw.groupby("Product"):
                try:
                    series = grp.set_index("Date")["ACR"].astype(float)
                    # Ensure index is DatetimeIndex before applying period operations
                    if not isinstance(series.index, pd.DatetimeIndex):
                        try:
                            series.index = pd.to_datetime(series.index)
                        except (ValueError, pd.errors.OutOfBoundsDatetime) as e:
                            diagnostic_messages.append(f"❌ Product {product}: Date parsing error - {str(e)[:100]}. Skipping.")
                            continue
                    series.index = series.index.to_period("M").to_timestamp(how="start")
                except Exception as e:
                    diagnostic_messages.append(f"❌ Product {product}: Data processing error - {str(e)[:100]}. Skipping.")
                    continue
                
                # Skip products that aren't in our valid_products list
                if product not in valid_products:
                    continue
                
                processed_products += 1
                
                # Data quality checks (redundant but kept for diagnostic messages)
                if len(series) < 12:
                    diagnostic_messages.append(f"❌ Product {product}: Insufficient data ({len(series)} months). Skipping.")
                    continue
                    
                # Remove any potential outliers (values > 5 standard deviations from mean)
                z_scores = np.abs((series - series.mean()) / series.std())
                if (z_scores > 5).any():
                    outlier_count = (z_scores > 5).sum()
                    diagnostic_messages.append(f"Product {product}: Detected {outlier_count} extreme outliers, capping them.")
                    series = series.clip(series.quantile(0.01), series.quantile(0.99))
                
                # Seasonality-aware split for strong seasonal patterns
                seasonality_strength = detect_seasonality_strength(series)
                if seasonality_strength > 0.6:
                    diagnostic_messages.append(f"📈 Product {product}: Strong seasonality detected (strength: {seasonality_strength:.2f})")
                elif seasonality_strength > 0.3:
                    diagnostic_messages.append(f"📊 Product {product}: Moderate seasonality detected (strength: {seasonality_strength:.2f})")
                else:
                    diagnostic_messages.append(f"📉 Product {product}: Weak seasonality detected (strength: {seasonality_strength:.2f})")
                
                split_result = get_seasonality_aware_split(series, seasonal_period=12, diagnostic_messages=diagnostic_messages)
                if split_result[0] is None:
                    # Skip this product if insufficient data
                    continue
                train, val = split_result
                
                future_idx = pd.date_range(
                    pd.Timestamp(series.index[-1]) + pd.DateOffset(months=1), periods=horizon, freq="MS")  # type: ignore
                act_df = pd.DataFrame({"Product": product, "Date": series.index, "ACR": series.values, "Type": "actual"})
                
                # Initialize model results for this product
                for m in models_selected:
                    results.setdefault(m, [])
                    mapes.setdefault(m, [])
                    smapes.setdefault(m, [])
                    mases.setdefault(m, [])
                    rmses.setdefault(m, [])

                # SARIMA with AIC/BIC dual selection
                if "SARIMA" in models_selected:
                    # Step 1: Find best parameters using train/validation split
                    best_model, selection_criterion, criterion_value, best_validation_mape, best_validation_smape, best_validation_mase, best_validation_rmse = fit_best_sarima(train, val, seasonality_strength)
                    
                    if best_model:
                        # Record parameters - access order and seasonal_order from fitted model
                        # For SARIMAX models, these are stored in the specification
                        try:
                            order = best_model.specification['order']  # type: ignore
                            seasonal_order = best_model.specification['seasonal_order']  # type: ignore
                            best_params = (order, seasonal_order)
                        except (AttributeError, KeyError):
                            # Fallback - try direct access
                            order = getattr(best_model, 'order', (1, 1, 1))
                            seasonal_order = getattr(best_model, 'seasonal_order', (0, 0, 0, 12))
                            best_params = (order, seasonal_order)
                        
                        sarima_params[product] = (
                            order,
                            seasonal_order,
                            criterion_value,
                            selection_criterion  # Add which criterion was used
                        )
                        mapes["SARIMA"].append(best_validation_mape)
                        smapes["SARIMA"].append(best_validation_smape)
                        mases["SARIMA"].append(best_validation_mase)
                        rmses["SARIMA"].append(best_validation_rmse)
                        
                        # Step 2: Retrain on FULL series for final forecasting
                        final_model = fit_final_sarima_model(series, best_params, seasonality_strength)
                        
                        if final_model:
                            # Generate forecasts from the end of the full series
                            forecast_result = final_model.get_forecast(horizon)  # type: ignore
                            pf = forecast_result.predicted_mean  # type: ignore
                            
                            # Apply trend-aware forecasting to eliminate cliff effects
                            pf_trend_adjusted = apply_trend_aware_forecasting(pf, series, len(train), "SARIMA", diagnostic_messages)
                            
                            # Apply minimal statistical validation (if enabled)
                            if enable_statistical_validation:
                                pf_validated = apply_statistical_validation(pf_trend_adjusted, series, "SARIMA")
                            else:
                                pf_validated = pf_trend_adjusted
                            
                            # Apply business adjustments if enabled
                            if apply_business_adjustments:
                                pf_final = apply_business_adjustments_to_forecast(
                                    pf_validated, business_growth_assumption, market_multiplier)
                                diagnostic_messages.append(f"📈 SARIMA Product {product}: Applied business adjustments (Growth: {business_growth_assumption}%, Market: {market_conditions})")
                            else:
                                pf_final = pf_validated
                            

                            df_f = pd.DataFrame({"Product": product, "Date": future_idx,
                                               "ACR": pf_final, "Type": "forecast"})
                            results["SARIMA"].append(pd.concat([act_df, df_f]))
                            
                            diagnostic_messages.append(f"✅ SARIMA Product {product}: Order {order}, Seasonal {seasonal_order}, {selection_criterion}: {criterion_value:.1f}, MAPE {best_validation_mape:.1%}")
                        else:
                            diagnostic_messages.append(f"❌ SARIMA Product {product}: Final model training failed")
                            mapes["SARIMA"].append(1.0)
                            smapes["SARIMA"].append(1.0)
                            mases["SARIMA"].append(np.nan)
                            rmses["SARIMA"].append(np.nan)
                    else:
                        diagnostic_messages.append(f"❌ SARIMA Product {product}: Failed to find suitable model")
                        mapes["SARIMA"].append(1.0)
                        smapes["SARIMA"].append(1.0)
                        mases["SARIMA"].append(np.nan)
                        rmses["SARIMA"].append(np.nan)
                    
                    done += 1
                    prog.progress(min(done / total, 1.0), text=f"Running SARIMA on Product {product} ({done}/{total})")                # ETS with trend-aware forecasting
                if "ETS" in models_selected:
                    try:
                        # Find best ETS configuration using train/val split
                        ets = ExponentialSmoothing(train, trend="add", 
                                                   seasonal="mul", seasonal_periods=12).fit()
                        pv = ets.forecast(len(val))
                        
                        # Validate ETS performance and calculate all metrics
                        val_mape, val_smape, val_mase, val_rmse = calculate_validation_metrics(val, pv, train)
                        best_ets_config = ("mul", val_mape, val_smape, val_mase, val_rmse)
                        
                        if val_mape > 1.0:  # If validation MAPE > 100%, try different ETS configuration
                            try:
                                ets_alt = ExponentialSmoothing(train, trend="add", 
                                                             seasonal="add", seasonal_periods=12).fit()
                                pv_alt = ets_alt.forecast(len(val))
                                val_mape_alt, val_smape_alt, val_mase_alt, val_rmse_alt = calculate_validation_metrics(val, pv_alt, train)
                                if val_mape_alt < val_mape:
                                    best_ets_config = ("add", val_mape_alt, val_smape_alt, val_mase_alt, val_rmse_alt)
                                    val_mape = val_mape_alt
                            except:
                                pass
                        
                        # Retrain on full series with best configuration
                        seasonal_type = best_ets_config[0]
                        ets_final = ExponentialSmoothing(series, trend="add", 
                                                       seasonal=seasonal_type, seasonal_periods=12).fit()
                        
                        # Generate forecasts
                        pf = ets_final.forecast(horizon)
                        mapes["ETS"].append(best_ets_config[1])  # MAPE
                        smapes["ETS"].append(best_ets_config[2])  # SMAPE
                        mases["ETS"].append(best_ets_config[3])  # MASE
                        rmses["ETS"].append(best_ets_config[4])  # RMSE
                        
                        # Apply trend-aware forecasting
                        pf_trend_adjusted = apply_trend_aware_forecasting(pf, series, len(train), "ETS", diagnostic_messages)
                        
                        # Apply statistical validation if enabled
                        if enable_statistical_validation:
                            pf_validated = apply_statistical_validation(pf_trend_adjusted, series, "ETS")
                        else:
                            pf_validated = pf_trend_adjusted
                        # Apply business adjustments if enabled
                        if apply_business_adjustments:
                            pf_final = apply_business_adjustments_to_forecast(pf_validated, business_growth_assumption, market_multiplier)
                        else:
                            pf_final = pf_validated
                        
                        fore_df = pd.DataFrame({"Product": product, "Date": future_idx, "ACR": pf_final, "Type": "forecast"})
                        results["ETS"].append(pd.concat([act_df, fore_df], ignore_index=True))
                        
                        diagnostic_messages.append(f"✅ ETS Product {product}: {seasonal_type} seasonal, MAPE {val_mape:.1%}")
                        
                    except Exception as e:
                        diagnostic_messages.append(f"❌ ETS Product {product}: {str(e)[:50]}")
                        mapes["ETS"].append(1.0)  # High MAPE for failed models
                        smapes["ETS"].append(1.0)
                        mases["ETS"].append(np.nan)
                        rmses["ETS"].append(np.nan)
                    
                    done += 1
                    prog.progress(min(done / total, 1.0), text=f"Running ETS on Product {product} ({done}/{total})")                # Polynomial models with minimal intervention to preserve curve shape
                if "Poly-2" in models_selected:
                    try:
                        # For validation: Use training data indices
                        X_train = np.arange(len(train)).reshape(-1, 1)
                        X_val = np.arange(len(train), len(train) + len(val)).reshape(-1, 1)
                        
                        poly_features = PolynomialFeatures(degree=2)
                        X_train_poly = poly_features.fit_transform(X_train)
                        X_val_poly = poly_features.transform(X_val)
                        
                        # Fit polynomial regression on training data
                        poly_model = LinearRegression()
                        poly_model.fit(X_train_poly, train.values)
                        
                        # Validate and calculate all metrics
                        pv = poly_model.predict(X_val_poly)
                        val_mape, val_smape, val_mase, val_rmse = calculate_validation_metrics(val, pv, train)
                        
                        # Only proceed if validation MAPE is reasonable
                        if val_mape < 2.0:  # Less than 200% MAPE
                            mapes["Poly-2"].append(val_mape)
                            smapes["Poly-2"].append(val_smape)
                            mases["Poly-2"].append(val_mase)
                            rmses["Poly-2"].append(val_rmse)
                            
                            # For final forecasting: Refit on FULL series to ensure smooth continuation
                            X_full = np.arange(len(series)).reshape(-1, 1)
                            X_full_poly = poly_features.fit_transform(X_full)
                            
                            final_poly_model = LinearRegression()
                            final_poly_model.fit(X_full_poly, np.asarray(series.values))  # type: ignore
                            
                            # Generate forecasts starting from where the full series ends
                            X_forecast = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
                            X_forecast_poly = poly_features.transform(X_forecast)
                            pf = final_poly_model.predict(X_forecast_poly)
                            
                            # Only apply non-negativity constraint for polynomial models
                            # Preserve the polynomial curve shape as much as possible
                            if series.min() >= 0:  # Only if all historical data is non-negative
                                pf = np.maximum(pf, 0)
                            
                            # Skip trend-aware forecasting and heavy statistical validation for polynomials
                            # Apply minimal business adjustments if enabled
                            if apply_business_adjustments:
                                pf_final = apply_business_adjustments_to_forecast(pf, business_growth_assumption, market_multiplier)
                            else:
                                pf_final = pf
                            

                            fore_df = pd.DataFrame({"Product": product, "Date": future_idx, "ACR": pf_final, "Type": "forecast"})
                            results["Poly-2"].append(pd.concat([act_df, fore_df], ignore_index=True))
                            
                            diagnostic_messages.append(f"✅ Poly-2 Product {product}: MAPE {val_mape:.1%} (pure polynomial)")
                        else:
                            diagnostic_messages.append(f"❌ Poly-2 Product {product}: Poor fit (MAPE {val_mape:.1%}), skipping")
                            mapes["Poly-2"].append(1.0)
                            smapes["Poly-2"].append(1.0)
                            mases["Poly-2"].append(np.nan)
                            rmses["Poly-2"].append(np.nan)
                        
                    except Exception as e:
                        diagnostic_messages.append(f"❌ Poly-2 Product {product}: {str(e)[:50]}")
                        mapes["Poly-2"].append(1.0)
                        smapes["Poly-2"].append(1.0)
                        mases["Poly-2"].append(np.nan)
                        rmses["Poly-2"].append(np.nan)
                    done += 1
                    prog.progress(min(done / total, 1.0), text=f"Running Poly-2 on Product {product} ({done}/{total})")                # Prophet model with trend-aware forecasting
                if "Prophet" in models_selected and HAVE_PROPHET:
                    try:
                        # Import Prophet here to ensure it's defined
                        from prophet import Prophet
                        # Create holidays dataframe if enabled
                        holidays_df = None
                        if enable_prophet_holidays:
                            try:
                                from prophet.make_holidays import make_holidays_df
                                # Create holidays for a reasonable date range (2010-2030)
                                years = list(range(2010, 2031))
                                
                                # Combine US and worldwide holidays
                                us_holidays = make_holidays_df(year_list=years, country='US')
                                worldwide_holidays = make_holidays_df(year_list=years, country=None)  # Global holidays
                                
                                # Combine and remove duplicates
                                holidays_df = pd.concat([us_holidays, worldwide_holidays]).drop_duplicates(subset=['ds']).reset_index(drop=True)
                                holidays_df['holiday'] = holidays_df['holiday'].astype(str)
                                
                                diagnostic_messages.append(f"📅 Prophet Product {product}: Added {len(holidays_df)} holiday effects")
                                
                            except Exception as holiday_error:
                                diagnostic_messages.append(f"⚠️ Prophet Product {product}: Could not load holidays, proceeding without them")
                                holidays_df = None
                        
                        # Prepare data for Prophet (validation first)
                        prophet_df = pd.DataFrame({
                            'ds': train.index,
                            'y': train.values
                        })
                        
                        # Fit Prophet model for validation
                        prophet_model = Prophet(
                            yearly_seasonality=True,  # type: ignore
                            weekly_seasonality=False,  # type: ignore
                            daily_seasonality=False,  # type: ignore
                            changepoint_prior_scale=0.05,
                            holidays=holidays_df
                        )
                        prophet_model.fit(prophet_df)
                        
                        # Validate on validation set and calculate all metrics
                        val_future = pd.DataFrame({'ds': val.index})
                        val_forecast = prophet_model.predict(val_future)
                        pv = val_forecast['yhat'].values
                        
                        # Calculate all validation metrics
                        val_mape, val_smape, val_mase, val_rmse = calculate_validation_metrics(val, pv, train)
                        mapes["Prophet"].append(val_mape)
                        smapes["Prophet"].append(val_smape)
                        mases["Prophet"].append(val_mase)
                        rmses["Prophet"].append(val_rmse)
                        
                        # Create new Prophet model for final forecasting on full series
                        full_prophet_df = pd.DataFrame({
                            'ds': series.index,
                            'y': series.values
                        })
                        final_prophet_model = Prophet(
                            yearly_seasonality=True,  # type: ignore
                            weekly_seasonality=False,  # type: ignore
                            daily_seasonality=False,  # type: ignore
                            changepoint_prior_scale=0.05,
                            holidays=holidays_df
                        )
                        final_prophet_model.fit(full_prophet_df)
                        # Create future dataframe for forecasting
                        future_df = pd.DataFrame({'ds': future_idx})
                        forecast = final_prophet_model.predict(future_df)
                        pf = forecast['yhat'].values
                        
                        # Apply trend-aware forecasting
                        pf_trend_adjusted = apply_trend_aware_forecasting(pf, series, len(train), "Prophet", diagnostic_messages)
                        
                        # Apply statistical validation if enabled
                        if enable_statistical_validation:
                            pf_validated = apply_statistical_validation(pf_trend_adjusted, series, "Prophet")
                        else:
                            pf_validated = pf_trend_adjusted
                        
                        # Apply business adjustments if enabled
                        if apply_business_adjustments:
                            pf_final = apply_business_adjustments_to_forecast(pf_validated, business_growth_assumption, market_multiplier)
                        else:
                            pf_final = pf_validated
                        
                        fore_df = pd.DataFrame({"Product": product, "Date": future_idx, "ACR": pf_final, "Type": "forecast"})
                        results["Prophet"].append(pd.concat([act_df, fore_df], ignore_index=True))
                        
                        # Enhanced diagnostic message
                        holiday_status = "with holidays" if enable_prophet_holidays and holidays_df is not None else "no holidays"
                        diagnostic_messages.append(f"✅ Prophet Product {product}: MAPE {val_mape:.1%} ({holiday_status})")
                        
                    except Exception as e:
                        diagnostic_messages.append(f"❌ Prophet Product {product}: {str(e)[:50]}")
                        mapes["Prophet"].append(1.0)  # High MAPE for failed models
                        smapes["Prophet"].append(1.0)
                        mases["Prophet"].append(np.nan)
                        rmses["Prophet"].append(np.nan)
                    
                    done += 1
                    prog.progress(min(done / total, 1.0), text=f"Running Prophet on Product {product} ({done}/{total})")

                # Auto-ARIMA model with enhanced parameter search
                if "Auto-ARIMA" in models_selected and HAVE_PMDARIMA:
                    try:
                        # Ensure auto_arima is available
                        auto_arima_func = None
                        try:
                            from pmdarima.arima import auto_arima as auto_arima_func
                        except ImportError:
                            try:
                                from pmdarima import auto_arima as auto_arima_func
                            except ImportError:
                                auto_arima_func = None
                                
                        if auto_arima_func is None:
                            diagnostic_messages.append("❌ Auto-ARIMA: Could not import auto_arima from pmdarima")
                            mapes["Auto-ARIMA"].append(1.0)
                            smapes["Auto-ARIMA"].append(1.0)
                            mases["Auto-ARIMA"].append(np.nan)
                            rmses["Auto-ARIMA"].append(np.nan)
                            continue
        
                        # Fit Auto-ARIMA on training data with seasonality awareness
                        auto_model = auto_arima_func(
                            train,
                            seasonal=True,
                            m=12,  # Monthly seasonality
                            max_p=3, max_d=2, max_q=3,
                            max_P=2, max_D=1, max_Q=2,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            trace=False
                        )
                        
                        # Validate on validation set and calculate all metrics
                        pv = auto_model.predict(len(val))
                        val_mape, val_smape, val_mase, val_rmse = calculate_validation_metrics(val, pv, train)
                        mapes["Auto-ARIMA"].append(val_mape)
                        smapes["Auto-ARIMA"].append(val_smape)
                        mases["Auto-ARIMA"].append(val_mase)
                        rmses["Auto-ARIMA"].append(val_rmse)
                        
                        # Refit on full series for final forecasting
                        auto_model_full = auto_arima_func(
                            series,
                            seasonal=True,
                            m=12,
                            max_p=3, max_d=2, max_q=3,
                            max_P=2, max_D=1, max_Q=2,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            trace=False
                        )
                        
                        # Generate forecasts
                        pf = auto_model_full.predict(horizon)
                        
                        # Apply trend-aware forecasting
                        pf_trend_adjusted = apply_trend_aware_forecasting(pf, series, len(train), "Auto-ARIMA", diagnostic_messages)
                        
                        # Apply statistical validation if enabled
                        if enable_statistical_validation:
                            pf_validated = apply_statistical_validation(pf_trend_adjusted, series, "Auto-ARIMA")
                        else:
                            pf_validated = pf_trend_adjusted
                        
                        # Apply business adjustments if enabled
                        if apply_business_adjustments:
                            pf_final = apply_business_adjustments_to_forecast(pf_validated, business_growth_assumption, market_multiplier)
                        else:
                            pf_final = pf_validated
                        
                        fore_df = pd.DataFrame({"Product": product, "Date": future_idx, "ACR": pf_final, "Type": "forecast"})
                        results["Auto-ARIMA"].append(pd.concat([act_df, fore_df], ignore_index=True))
                        
                        # Get model details for diagnostics
                        model_order = auto_model_full.order
                        seasonal_order = auto_model_full.seasonal_order
                        diagnostic_messages.append(f"✅ Auto-ARIMA Product {product}: Order {model_order}, Seasonal {seasonal_order}, MAPE {val_mape:.1%}")
                        
                    except Exception as e:
                        diagnostic_messages.append(f"❌ Auto-ARIMA Product {product}: {str(e)[:50]}")
                        mapes["Auto-ARIMA"].append(1.0)  # High MAPE for failed models
                        smapes["Auto-ARIMA"].append(1.0)
                        mases["Auto-ARIMA"].append(np.nan)
                        rmses["Auto-ARIMA"].append(np.nan)
                    
                    done += 1
                    prog.progress(min(done / total, 1.0), text=f"Running Auto-ARIMA on Product {product} ({done}/{total})")

                # LightGBM model with feature engineering
                if "LightGBM" in models_selected and HAVE_LGBM:
                    try:
                        # Create features for LightGBM
                        def create_lgbm_features(ts_data, name_prefix=""):
                            """Create time series features for LightGBM"""
                            df = pd.DataFrame(index=ts_data.index)
                            df['ACR'] = ts_data.values
                            
                            # Lag features
                            for lag in [1, 2, 3, 6, 12]:
                                df[f'lag_{lag}'] = df['ACR'].shift(lag)
                            
                            # Rolling statistics
                            for window in [3, 6, 12]:
                                df[f'rolling_mean_{window}'] = df['ACR'].rolling(window=window).mean()
                                df[f'rolling_std_{window}'] = df['ACR'].rolling(window=window).std()
                            
                            # Time-based features
                            df['month'] = df.index.month  # type: ignore
                            df['quarter'] = df.index.quarter  # type: ignore
                            df['year'] = df.index.year  # type: ignore
                            
                            # Trend features
                            df['trend'] = range(len(df))
                            
                            return df.dropna()
                        
                        # Create feature datasets
                        train_features = create_lgbm_features(train)
                        val_features = create_lgbm_features(val)
                        
                        # Validate feature datasets
                        if len(train_features) < 12 or len(val_features) < 1:
                            diagnostic_messages.append(f"❌ LightGBM Product {product}: Insufficient data after feature creation (train: {len(train_features)}, val: {len(val_features)})")
                            mapes["LightGBM"].append(1.0)
                        else:
                            feature_cols = [col for col in train_features.columns if col != 'ACR']
                            
                            # Additional validation for feature matrices
                            if len(feature_cols) == 0:
                                diagnostic_messages.append(f"❌ LightGBM Product {product}: No valid features after processing")
                                mapes["LightGBM"].append(1.0)
                                smapes["LightGBM"].append(1.0)
                                mases["LightGBM"].append(np.nan)
                                rmses["LightGBM"].append(np.nan)
                            elif (isinstance(train_features[feature_cols], pd.DataFrame) and train_features[feature_cols].isnull().values.all()) or (isinstance(val_features[feature_cols], pd.DataFrame) and val_features[feature_cols].isnull().values.all()):
                                diagnostic_messages.append(f"❌ LightGBM Product {product}: All features are null after processing")
                                mapes["LightGBM"].append(1.0)
                                smapes["LightGBM"].append(1.0)
                                mases["LightGBM"].append(np.nan)
                                rmses["LightGBM"].append(np.nan)
                            else:
                                # Remove any remaining NaN values
                                train_features = train_features.bfill().ffill()
                                val_features = val_features.bfill().ffill()
                                
                                # Find best LightGBM parameters
                                best_lgbm, best_params, best_mape, best_smape, best_mase, best_rmse = fit_best_lightgbm(
                                    train_features, val_features, feature_cols, diagnostic_messages
                                )
                                
                                if best_lgbm is not None:
                                    mapes["LightGBM"].append(best_mape)
                                    smapes["LightGBM"].append(best_smape)
                                    mases["LightGBM"].append(best_mase)
                                    rmses["LightGBM"].append(best_rmse)
                                
                                    # Create full series features for final forecasting
                                    full_features = create_lgbm_features(series)
                                    
                                    # Retrain on full dataset with explicit parameter extraction
                                    from lightgbm import LGBMRegressor
                                    final_lgbm = LGBMRegressor(
                                        n_estimators=int(best_params.get("n_estimators", 100)),
                                        learning_rate=float(best_params.get("learning_rate", 0.1)),
                                        num_leaves=int(best_params.get("num_leaves", 31)),
                                        max_depth=int(best_params.get("max_depth", 6)),
                                        random_state=42,
                                        verbose=-1,
                                        force_col_wise=True
                                    )
                                    final_lgbm.fit(full_features[feature_cols], full_features['ACR'])
                                
                                # Generate iterative forecasts
                                forecast_values = []
                                current_series = series.copy()
                                
                                for step in range(horizon):
                                    # Create features for current step
                                    step_features = create_lgbm_features(current_series)
                                    if len(step_features) == 0:
                                        break
                                    
                                    # Predict next value
                                    step_X = step_features[feature_cols].iloc[-1:].values
                                    next_pred = final_lgbm.predict(step_X)[0]  # type: ignore
                                    forecast_values.append(next_pred)
                                    
                                    # Add prediction to series for next iteration
                                    next_date = pd.Timestamp(current_series.index[-1]) + pd.DateOffset(months=1)  # type: ignore
                                    current_series = pd.concat([
                                        current_series, 
                                        pd.Series([next_pred], index=[next_date])
                                    ])
                                
                                if len(forecast_values) > 0:
                                    pf = np.array(forecast_values)
                                    
                                    # Apply trend-aware forecasting
                                    pf_trend_adjusted = apply_trend_aware_forecasting(pf, series, len(train), "LightGBM", diagnostic_messages)
                                    
                                    # Apply statistical validation if enabled
                                    if enable_statistical_validation:
                                        pf_validated = apply_statistical_validation(pf_trend_adjusted, series, "LightGBM")
                                    else:
                                        pf_validated = pf_trend_adjusted
                                    
                                    # Apply business adjustments if enabled
                                    if apply_business_adjustments:
                                        pf_final = apply_business_adjustments_to_forecast(pf_validated, business_growth_assumption, market_multiplier)
                                    else:
                                        pf_final = pf_validated
                                    
                                    fore_df = pd.DataFrame({"Product": product, "Date": future_idx[:len(pf_final)], "ACR": pf_final, "Type": "forecast"})
                                    results["LightGBM"].append(pd.concat([act_df, fore_df], ignore_index=True))
                                else:
                                    # LightGBM failed - add default values
                                    mapes["LightGBM"].append(1.0)
                                    smapes["LightGBM"].append(1.0)
                                    mases["LightGBM"].append(np.nan)
                                    rmses["LightGBM"].append(np.nan)
                                    
                    except Exception as e:
                        diagnostic_messages.append(f"❌ LightGBM Product {product}: {str(e)[:50]}")
                        mapes["LightGBM"].append(1.0)
                        smapes["LightGBM"].append(1.0)
                        mases["LightGBM"].append(np.nan)
                        rmses["LightGBM"].append(np.nan)
                    
                    done += 1
                    prog.progress(min(done / total, 1.0), text=f"Running LightGBM on Product {product} ({done}/{total})")                # ──────────── Polynomial regression: Poly-3 with minimal intervention ────────────
                if "Poly-3" in models_selected:
                    try:
                        # For validation: Use training data indices
                        X_train = np.arange(len(train)).reshape(-1, 1)
                        X_val = np.arange(len(train), len(train) + len(val)).reshape(-1, 1)
                        
                        poly_features = PolynomialFeatures(degree=3)
                        X_train_poly = poly_features.fit_transform(X_train)
                        X_val_poly = poly_features.transform(X_val)
                        
                        # Fit polynomial regression on training data
                        poly_model = LinearRegression()
                        poly_model.fit(X_train_poly, train.values)
                        
                        # Validate and calculate all metrics
                        pv = poly_model.predict(X_val_poly)
                        val_mape, val_smape, val_mase, val_rmse = calculate_validation_metrics(val, pv, train)
                        
                        # Only proceed if validation MAPE is reasonable
                        if val_mape < 2.0:  # Less than 200% MAPE
                            mapes["Poly-3"].append(val_mape)
                            smapes["Poly-3"].append(val_smape)
                            mases["Poly-3"].append(val_mase)
                            rmses["Poly-3"].append(val_rmse)
                            
                            # For final forecasting: Refit on FULL series to ensure smooth continuation
                            X_full = np.arange(len(series)).reshape(-1, 1)
                            X_full_poly = poly_features.fit_transform(X_full)
                            
                            final_poly_model = LinearRegression()
                            final_poly_model.fit(X_full_poly, np.asarray(series.values))  # type: ignore
                            
                            # Generate forecasts starting from where the full series ends
                            X_forecast = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
                            X_forecast_poly = poly_features.transform(X_forecast)
                            pf = final_poly_model.predict(X_forecast_poly)
                            
                            # Only apply non-negativity constraint for polynomial models
                            # Preserve the polynomial curve shape as much as possible
                            if series.min() >= 0:  # Only if all historical data is non-negative
                                pf = np.maximum(pf, 0)
                            
                            # For Poly-3, add minimal smoothing only for extreme cases
                            # Check for unrealistic jumps (>500% month-over-month growth)
                            smoothed_needed = False
                            for i in range(1, len(pf)):
                                if pf[i] > pf[i-1] * 5:  # More than 500% growth
                                    smoothed_needed = True
                                    break
                            
                            if smoothed_needed:
                                # Apply very light smoothing only to extreme outliers
                                pf_smoothed = pf.copy()
                                for i in range(1, len(pf_smoothed)):
                                    if pf_smoothed[i] > pf_smoothed[i-1] * 5:  # Cap at 500% growth
                                        pf_smoothed[i] = pf_smoothed[i-1] * 3  # Limit to 300% growth
                                pf = pf_smoothed
                                diagnostic_messages.append(f"📊 Poly-3 Product {product}: Applied minimal smoothing to extreme growth")
                            
                            # Skip trend-aware forecasting and heavy statistical validation for polynomials
                            # Apply minimal business adjustments if enabled
                            if apply_business_adjustments:
                                pf_final = apply_business_adjustments_to_forecast(pf, business_growth_assumption, market_multiplier)
                            else:
                                pf_final = pf
                            

                            fore_df = pd.DataFrame({"Product": product, "Date": future_idx, "ACR": pf_final, "Type": "forecast"})
                            results["Poly-3"].append(pd.concat([act_df, fore_df], ignore_index=True))
                            
                            diagnostic_messages.append(f"✅ Poly-3 Product {product}: MAPE {val_mape:.1%} (pure polynomial)")
                        else:
                            diagnostic_messages.append(f"❌ Poly-3 Product {product}: Poor fit (MAPE {val_mape:.1%}), skipping")
                            mapes["Poly-3"].append(1.0)
                            smapes["Poly-3"].append(1.0)
                            mases["Poly-3"].append(np.nan)
                            rmses["Poly-3"].append(np.nan)
                        
                    except Exception as e:
                        diagnostic_messages.append(f"❌ Poly-3 Product {product}: {str(e)[:50]}")
                        mapes["Poly-3"].append(1.0)
                        smapes["Poly-3"].append(1.0)
                        mases["Poly-3"].append(np.nan)
                        rmses["Poly-3"].append(np.nan)
                    done += 1
                    prog.progress(min(done / total, 1.0), text=f"Running Poly-3 on Product {product} ({done}/{total})")

            prog.empty()

            # Combine & plot – only keep models that actually produced DataFrames
            for key in list(results.keys()):
                if results[key]:
                    df_m = pd.concat(results[key])
                    df_m["FiscalYear"] = df_m["Date"].apply(fy)
                    results[key] = df_m
                else:
                    # drop empty models (e.g. Poly-2)
                    results.pop(key)

            # Store per-product metrics from validation-based calculations
            product_mapes = {}  # {model_name: {product_name: mape_value}}
            product_smapes = {}  # {model_name: {product_name: smape_value}}
            product_mases = {}   # {model_name: {product_name: mase_value}}
            product_rmses = {}   # {model_name: {product_name: rmse_value}}
            
            for model_name in results.keys():
                product_mapes[model_name] = {}
                product_smapes[model_name] = {}
                product_mases[model_name] = {}
                product_rmses[model_name] = {}
                
                # Only populate if we have metrics for this model
                if model_name in mapes:
                    for i, product in enumerate(products):
                        # MAPE - always available
                        if i < len(mapes[model_name]):
                            product_mapes[model_name][product] = mapes[model_name][i]
                        else:
                            product_mapes[model_name][product] = 1.0
                        
                        # SMAPE - check bounds separately
                        if model_name in smapes and i < len(smapes[model_name]):
                            product_smapes[model_name][product] = smapes[model_name][i]
                        else:
                            product_smapes[model_name][product] = 1.0
                        
                        # MASE - check bounds separately
                        if model_name in mases and i < len(mases[model_name]):
                            product_mases[model_name][product] = mases[model_name][i]
                        else:
                            product_mases[model_name][product] = np.nan
                        
                        # RMSE - check bounds separately
                        if model_name in rmses and i < len(rmses[model_name]):
                            product_rmses[model_name][product] = rmses[model_name][i]
                        else:
                            product_rmses[model_name][product] = np.nan

            # --- Model Ranking by Metric ---
            # For each metric, rank models (lower is better)
            model_names = list(results.keys())
            metric_ranks = {m: {model: 0 for model in model_names} for m in ["MAPE", "SMAPE", "MASE", "RMSE"]}
            # For each product, rank models for each metric
            for product in products:
                # MAPE
                mape_vals = {model: product_mapes[model].get(product, np.nan) for model in model_names}
                mape_sorted = sorted((v, k) for k, v in mape_vals.items() if not np.isnan(v))
                for rank, (v, k) in enumerate(mape_sorted):
                    metric_ranks["MAPE"][k] += rank + 1
                # SMAPE
                smape_vals = {model: product_smapes[model].get(product, np.nan) for model in model_names}
                smape_sorted = sorted((v, k) for k, v in smape_vals.items() if not np.isnan(v))
                for rank, (v, k) in enumerate(smape_sorted):
                    metric_ranks["SMAPE"][k] += rank + 1
                # MASE
                mase_vals = {model: product_mases[model].get(product, np.nan) for model in model_names}
                mase_sorted = sorted((v, k) for k, v in mase_vals.items() if not np.isnan(v))
                for rank, (v, k) in enumerate(mase_sorted):
                    metric_ranks["MASE"][k] += rank + 1
                # RMSE
                rmse_vals = {model: product_rmses[model].get(product, np.nan) for model in model_names}
                rmse_sorted = sorted((v, k) for k, v in rmse_vals.items() if not np.isnan(v))
                for rank, (v, k) in enumerate(rmse_sorted):
                    metric_ranks["RMSE"][k] += rank + 1
            # Compute average rank for each model
            avg_ranks = {model: np.mean([metric_ranks[m][model] for m in metric_ranks]) for model in model_names}
            # Select best model by average rank
            best_model_by_rank = min(avg_ranks.keys(), key=lambda k: avg_ranks[k]) if avg_ranks else None
            # Store in session state for downstream use if needed
            st.session_state.model_avg_ranks = avg_ranks
            st.session_state.best_model_by_rank = best_model_by_rank
            st.session_state.product_smapes = product_smapes
            st.session_state.product_mases = product_mases
            st.session_state.product_rmses = product_rmses

            # Find best model for each product individually using multi-metric ranking
            best_models_per_product = {}  # {product_name: model_name}
            best_mapes_per_product = {}   # {product_name: mape_value}
            
            for product in products:
                # Calculate per-product rankings across all 4 metrics
                product_model_metrics = {}
                product_model_ranks = {model: 0 for model in model_names}
                
                for model_name in results.keys():
                    if product in product_mapes[model_name]:
                        product_model_metrics[model_name] = {
                            'MAPE': product_mapes[model_name][product],
                            'SMAPE': product_smapes[model_name].get(product, np.nan),
                            'MASE': product_mases[model_name].get(product, np.nan),
                            'RMSE': product_rmses[model_name].get(product, np.nan)
                        }
                
                if product_model_metrics:
                    # Calculate rankings for this specific product across all metrics
                    for metric in ['MAPE', 'SMAPE', 'MASE', 'RMSE']:
                        metric_vals = {model: metrics[metric] for model, metrics in product_model_metrics.items() 
                                     if not np.isnan(metrics[metric])}
                        if metric_vals:
                            metric_sorted = sorted(metric_vals.items(), key=lambda x: x[1])
                            for rank, (model, _) in enumerate(metric_sorted):
                                product_model_ranks[model] += rank + 1
                    
                    # Select model with best average rank for this product
                    if enable_business_aware_selection:
                        # Apply business-aware filtering to ranked candidates
                        product_model_mapes = {model: product_mapes[model][product] 
                                             for model in product_model_metrics.keys()}
                        # Pass ranking information for smarter business-aware selection
                        valid_ranks = {model: rank/4 for model, rank in product_model_ranks.items() if rank > 0}  # Average rank
                        best_model_for_product, best_mape_for_product = select_business_aware_best_model(
                            product_model_mapes, product, diagnostic_messages, valid_ranks
                        )
                    else:
                        # Pure multi-metric ranking selection
                        valid_ranks = {model: rank for model, rank in product_model_ranks.items() if rank > 0}
                        if valid_ranks:
                            best_model_for_product = min(valid_ranks.keys(), key=lambda k: valid_ranks[k])
                            best_mape_for_product = product_mapes[best_model_for_product][product]
                            
                            if diagnostic_messages:
                                avg_rank = valid_ranks[best_model_for_product] / 4  # Average across 4 metrics
                                diagnostic_messages.append(
                                    f"📊 Product {product}: Multi-metric ranking selected {best_model_for_product} "
                                    f"(Avg Rank: {avg_rank:.1f}, MAPE: {best_mape_for_product:.1%})"
                                )
                        else:
                            # Fallback to MAPE if ranking fails
                            product_model_mapes = {model: product_mapes[model][product] 
                                                 for model in product_model_metrics.keys()}
                            best_model_for_product = min(product_model_mapes.keys(), key=lambda k: product_model_mapes[k])
                            best_mape_for_product = product_model_mapes[best_model_for_product]
                        
                    best_models_per_product[product] = best_model_for_product
                    best_mapes_per_product[product] = best_mape_for_product
                else:
                    # Fallback to first available model
                    best_models_per_product[product] = list(results.keys())[0]
                    best_mapes_per_product[product] = 1.0
            
            # Create hybrid "Best per Product" model combining best forecasts for each product
            hybrid_results = []
            for product in products:
                best_model_name = best_models_per_product[product]
                if best_model_name in results:
                    product_data = results[best_model_name]
                    product_specific_data = product_data[product_data["Product"] == product].copy()
                    if not product_specific_data.empty:
                        product_specific_data["BestModel"] = best_model_name  # Add metadata
                        hybrid_results.append(product_specific_data)
            
            if hybrid_results:
                hybrid_df = pd.concat(hybrid_results, ignore_index=True)
                hybrid_df["FiscalYear"] = hybrid_df["Date"].apply(fy)
                results["Best per Product"] = hybrid_df
            
                # Calculate average MAPE for display (keeping original logic for comparison)
            avg_mapes = {m: np.mean(mapes[m]) for m in mapes if mapes[m]}
            # Calculate average metrics for all models
            avg_smapes = {m: np.mean(smapes[m]) for m in smapes if smapes[m]}
            avg_mases = {m: np.mean(mases[m]) for m in mases if mases[m] and not np.isnan(mases[m]).all()}
            avg_rmses = {m: np.mean(rmses[m]) for m in rmses if rmses[m] and not np.isnan(rmses[m]).all()}
            
            # Calculate average MAPE for hybrid model
            if hybrid_results:
                hybrid_avg_mape = np.mean(list(best_mapes_per_product.values()))
                avg_mapes["Best per Product"] = hybrid_avg_mape
            
            best_model = min(avg_mapes.keys(), key=lambda k: avg_mapes[k])  # type: ignore
            best_mape = avg_mapes[best_model] * 100
            
            # Process yearly renewals overlay if file is uploaded
            yearly_renewals_applied = False
            if 'yearly_renewals_file' in st.session_state and st.session_state['yearly_renewals_file'] is not None:
                try:
                    yearly_file = st.session_state['yearly_renewals_file']
                    yearly_renewals_data = read_any_excel(io.BytesIO(yearly_file.read()))
                    
                    # Validate yearly renewals data structure
                    required_yearly_cols = {"Date", "Product", "ACR"}  # Assuming same structure as main data
                    if required_yearly_cols.issubset(yearly_renewals_data.columns):
                        # Process yearly renewals data
                        yearly_renewals_data["Date"] = coerce_month_start(yearly_renewals_data["Date"])
                        yearly_renewals_data.sort_values("Date", inplace=True)
                        
                        # Track all renewal data for debugging
                        all_historical_entries = 0
                        all_future_entries = 0
                        all_future_details = []
                        
                        # Apply yearly renewals overlay to all models
                        for model_name in results.keys():
                            if model_name in results and results[model_name] is not None:
                                df_model = results[model_name].copy()
                                
                                # Add yearly renewals as additional rows to the actuals (historical)
                                yearly_renewal_rows = []
                                for _, renewal_row in yearly_renewals_data.iterrows():
                                    renewal_date = renewal_row["Date"]
                                    renewal_product = renewal_row["Product"]
                                    renewal_amount = renewal_row["ACR"]
                                    
                                    # Create new row for historical non-compliant RevRec
                                    new_row = {
                                        "Product": renewal_product,
                                        "Date": renewal_date,
                                        "ACR": renewal_amount,
                                        "Type": "non-compliant",  # New type for non-compliant RevRec
                                        "FiscalYear": fy(pd.Timestamp(renewal_date))
                                    }
                                    yearly_renewal_rows.append(new_row)
                                
                                # Project renewal patterns into future forecast periods at 100% probability
                                # Get the forecast dates for this model
                                forecast_dates = df_model[df_model["Type"] == "forecast"]["Date"].unique()
                                
                                # For each product in renewals data, project the pattern forward
                                for product in yearly_renewals_data["Product"].unique():
                                    product_renewals = yearly_renewals_data[yearly_renewals_data["Product"] == product].copy()
                                    
                                    if len(product_renewals) > 0:
                                        # Project EVERY historical renewal forward (not just the latest)
                                        forecast_start = pd.Timestamp(forecast_dates.min()) if len(forecast_dates) > 0 else pd.Timestamp.now()
                                        forecast_end = pd.Timestamp(forecast_dates.max()) if len(forecast_dates) > 0 else pd.Timestamp.now()
                                        
                                        # For each historical renewal, project it forward yearly
                                        for _, historical_renewal in product_renewals.iterrows():
                                            renewal_amount = historical_renewal["ACR"]
                                            historical_date = pd.Timestamp(historical_renewal["Date"])
                                            
                                            # Project this specific renewal forward yearly covering the entire forecast period
                                            projection_year = historical_date.year + 1  # Start from the year after this historical renewal
                                            max_projection_year = forecast_end.year + 1  # Project up to the end of forecast period
                                            
                                            while projection_year <= max_projection_year:
                                                # Create renewal date for this year (same month/day as historical renewal)
                                                try:
                                                    next_renewal_date = historical_date.replace(year=projection_year)
                                                except ValueError:
                                                    # Handle leap year edge case (Feb 29 -> Feb 28)
                                                    next_renewal_date = historical_date.replace(year=projection_year, day=28)
                                                
                                                # Check if this renewal falls within the forecast period
                                                if next_renewal_date >= forecast_start and next_renewal_date <= forecast_end:
                                                    # Add future renewal at 100% probability
                                                    future_renewal_row = {
                                                        "Product": product,
                                                        "Date": next_renewal_date,
                                                        "ACR": renewal_amount,
                                                        "Type": "non-compliant-forecast",  # Future non-compliant RevRec
                                                        "FiscalYear": fy(next_renewal_date)
                                                    }
                                                    yearly_renewal_rows.append(future_renewal_row)
                                                    
                                                    # Track for debugging (only for first model to avoid duplicates)
                                                    if model_name == list(results.keys())[0]:
                                                        all_future_entries += 1
                                                        all_future_details.append(f"{product} on {next_renewal_date.strftime('%Y-%m')} = ${renewal_amount:,.0f}")
                                                
                                                projection_year += 1  # Move to next year
                                
                                if yearly_renewal_rows:
                                    # Add all renewal rows (historical + future) to the dataframe
                                    yearly_df = pd.DataFrame(yearly_renewal_rows)
                                    df_model = pd.concat([df_model, yearly_df], ignore_index=True)
                                    df_model = df_model.sort_values(['Product', 'Date']).reset_index(drop=True)
                                
                                # Update the results with non-compliant RevRec added
                                results[model_name] = df_model
                        
                        yearly_renewals_applied = True
                        historical_count = len(yearly_renewals_data)
                        
                        diagnostic_messages.append(f"✅ Non-Compliant Upfront RevRec: Added {historical_count} historical entries + projected {all_future_entries} future renewals at 100% probability (each historical renewal projected forward yearly)")
                        
                        # Add debugging info about future projections
                        if all_future_entries > 0 and all_future_details:
                            unique_months = len(set(detail.split(' on ')[1].split(' =')[0] for detail in all_future_details))
                            diagnostic_messages.append(f"🔮 Future Non-Compliant Upfront RevRec Renewals: {all_future_entries} total projections across {unique_months} monthly patterns - {', '.join(all_future_details[:5])}{'...' if len(all_future_details) > 5 else ''}")
                        else:
                            # Additional debugging for troubleshooting
                            # Get forecast period from any model for debugging
                            sample_model_data = next(iter(results.values()))
                            sample_forecast_dates = sample_model_data[sample_model_data["Type"] == "forecast"]["Date"].unique()
                            if len(sample_forecast_dates) > 0:
                                forecast_period = f"{pd.Timestamp(sample_forecast_dates.min()).strftime('%Y-%m')} to {pd.Timestamp(sample_forecast_dates.max()).strftime('%Y-%m')}"
                                diagnostic_messages.append(f"⚠️ No future non-compliant renewals projected (forecast period: {forecast_period})")
                            else:
                                diagnostic_messages.append(f"⚠️ No future non-compliant renewals projected - no forecast data found")
                    else:
                        diagnostic_messages.append(f"❌ Yearly Renewals Error: File must contain columns {required_yearly_cols}")
                except Exception as e:
                    diagnostic_messages.append(f"❌ Yearly Renewals Error: {str(e)[:100]}")
            
            # Store results in session state for persistence
            st.session_state.forecast_results = results
            st.session_state.forecast_mapes = avg_mapes
            st.session_state.forecast_smapes = avg_smapes  # Store average SMAPE data
            st.session_state.forecast_mases = avg_mases    # Store average MASE data
            st.session_state.forecast_rmses = avg_rmses    # Store average RMSE data
            st.session_state.product_mapes = product_mapes  # Store per-product MAPE data
            st.session_state.product_smapes = product_smapes  # Store per-product SMAPE data
            st.session_state.product_mases = product_mases    # Store per-product MASE data
            st.session_state.product_rmses = product_rmses    # Store per-product RMSE data
            st.session_state.best_models_per_product = best_models_per_product  # Store best model for each product
            st.session_state.best_mapes_per_product = best_mapes_per_product    # Store best MAPE for each product
            st.session_state.forecast_sarima_params = sarima_params
            st.session_state.diagnostic_messages = diagnostic_messages
            st.session_state.uploaded_filename = uploaded.name
            st.session_state.business_adjustments_applied = apply_business_adjustments
            st.session_state.business_growth_used = business_growth_assumption
            st.session_state.market_conditions_used = market_conditions
            st.session_state.business_aware_selection_used = enable_business_aware_selection  # Store business-aware selection status
            st.session_state.yearly_renewals_applied = yearly_renewals_applied  # Store whether yearly renewals were applied
            
            # Refresh page to show clean results view (hide upload interface)
            st.rerun()

# Credit
st.markdown("**Created by: Jake Moura (jakemoura@microsoft.com)**")

# --- Example Data Tab ---
with tab_example:
    st.subheader("Example Data Format & Template")
    st.markdown(
        "Upload your data as an Excel file with three columns: `Date`, `Product`, and `ACR`.\n"
        "- `Date`: Month‑start dates (e.g., 2021-01-01).\n"
        "- `Product`: Sub‑strategic pillar names (you can include multiple pillars).\n"
        "- `ACR`: Actual Cost Revenue values."
    )
    sample_df = pd.DataFrame({
        "Date": ["2021-01-01", "2021-02-01", "2021-03-01"],
        "Product": ["Pillar A", "Pillar B", "Pillar A"],
        "ACR": [100.0, 120.5, 110.3]
    })
    st.dataframe(sample_df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        sample_df.to_excel(writer, index=False, sheet_name="Template")
    buf.seek(0)
    st.download_button(
        "Download Example Template",
        data=buf,
        file_name="example_data_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --- Glossary Tab ---
with tab_glossary:
    st.subheader("📝 Glossary of Forecasting Methods & Metrics")
    
    # Key Metrics Section
    st.markdown("### 🎯 **Key Performance Metrics**")
    st.markdown("""
    **MAPE (Mean Absolute Percentage Error)** - The primary accuracy measure:
    - **What it means:** Average percentage difference between forecast and actual values
    - **Why lower is better:** Shows how close forecasts are to reality
    - **Example:** 10% MAPE = forecasts typically within 10% of actual values
    - **Interpretation:** 
      - 0-10% = Excellent accuracy 🎯
      - 10-20% = Good accuracy 📈  
      - 20-50% = Moderate accuracy ⚠️
      - 50%+ = Lower accuracy 🔴
    
    **AIC (Akaike Information Criterion)** - Statistical model quality:
    - **What it means:** Balances model fit quality with complexity
    - **Why lower is better:** Better statistical fit without overfitting
    
    **BIC (Bayesian Information Criterion)** - Alternative statistical model quality:
    - **What it means:** Similar to AIC but penalizes complexity more heavily
    - **Why it matters:** Favors simpler, more generalizable models
    - **Usage:** Compared to AIC in SARIMA selection, especially good for smaller datasets
    - **Our approach:** Test both and choose the one that performs better on validation data
    """)
    st.markdown("""
    **Smart Product-by-Product Selection (Recommended):**
    1. **Run all models** on each data product individually
    2. **Test accuracy** using validation for each model-product combination
    3. **Calculate all metrics** (MAPE, SMAPE, MASE, RMSE) for each model's predictions per product
    4. **Rank models** across all four metrics for each specific product
    5. **Winner per product:** Model with the **best average ranking across all metrics** for that specific product is selected
    6. **Hybrid approach:** Combine the best models for each product into one optimized forecast
    
    **Traditional Overall Selection (Alternative):**
    1. **Run all models** on your historical data
    2. **Test accuracy** using validation (unseen historical data)
    3. **Calculate average MAPE** across all products for each model
    4. **Winner:** Model with **lowest average MAPE** across all products
    5. **Why this works:** The most accurate on historical data is likely most accurate for future predictions
    
    **Why Product-by-Product is Better:**
    - Different products may have different patterns (seasonal vs. trending)
    - One model might excel at growth patterns, another at seasonal patterns
    - Optimizes accuracy for each specific data pattern
    - Often achieves lower overall error than any single model
    """)
    
    # Models Section
    st.markdown("### 🔮 **Forecasting Models Explained**")
    st.markdown(
        "**Best per Product ⭐**: Smart hybrid approach that uses the most accurate model for each data product individually. Combines multiple models for optimal results. With business-aware selection enabled, prioritizes seasonally-aware models over polynomial fits for revenue forecasting.\n\n"
        "**SARIMA**: Advanced statistical time series model with seasonal patterns. Uses both AIC and BIC criteria for optimal parameter selection, then chooses the best performer on validation data. Excellent for data with clear seasonal trends and sufficient history.\n\n"
        "**ETS (Exponential Smoothing)**: Decomposes data into Error, Trend, and Seasonality. Automatically adapts to your data patterns. Great for business data with growth trends.\n\n"
        "**Prophet**: Facebook's business-focused model with optional holiday effects and growth assumptions. Can include US and worldwide holidays for better accuracy around holiday periods. Designed specifically for business forecasting scenarios.\n\n"
        "**Auto-ARIMA**: Automated statistical modeling that finds the best ARIMA configuration. Smart parameter selection with business validation.\n\n"
        "**LightGBM**: Machine learning model using gradient boosting. Captures complex non-linear patterns using historical lags and features.\n\n"
        "**Polynomial (Poly-2/3)**: Pure mathematical trend fitting using 2nd or 3rd degree curves. ⚠️ **Business Note**: While these may show better MAPE on historical data, they can be problematic for consumptive businesses where revenue recognition depends on daily patterns and business cycles. Use with caution for revenue forecasting.\n\n"
        "**Interactive Adjustments**: Apply custom growth/haircut percentages to any product starting from any future month. Perfect for management overrides and scenario planning."
    )
