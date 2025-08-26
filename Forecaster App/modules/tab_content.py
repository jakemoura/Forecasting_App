"""
Tab content functions for the main Streamlit interface.

Contains functions for rendering the content of different tabs including
the main forecast tab, example data tab, and model guide tab.
"""

import io
import pandas as pd
import streamlit as st
from .ui_components import display_forecast_results


def render_forecast_tab(controls_config):
    """
    Render the main forecast tab content.
    
    Args:
        controls_config: Configuration dictionary from sidebar controls
        
    Returns:
        tuple: (uploaded_file, run_button_pressed) or (None, None) if showing results
    """
    from .session_state import has_existing_results, clear_session_state_for_new_forecast
    from .ui_config import display_data_requirements, display_advanced_requirements, display_upload_section
    
    # Clean, modern header
    st.markdown("# 🎯 **Multimodel Time‑Series Forecaster**")
    st.markdown("---")
    
    # Check if we have existing results to show
    if has_existing_results():
        return _render_results_view()
    else:
        return _render_upload_view(controls_config)


def _render_results_view():
    """Render the results view when forecasts are complete."""
    # Clean results header
    st.success("🎉 **Forecast Complete!** Your results are ready below.")
    # Sidebar new forecast action (moved from main layout)
    with st.sidebar:
        st.markdown("---")
        if st.button("🔄 **New Forecast**", type="secondary", use_container_width=True):
            from .session_state import clear_session_state_for_new_forecast
            clear_session_state_for_new_forecast()
            st.rerun()
        st.markdown("---")

    # Directly show results starting at Results Summary (removed intermediate 'Results Dashboard' section)
    # Display the existing results
    display_forecast_results()
    
    return None, None


def _render_upload_view(controls_config):
    """Render the upload view for new forecasts."""
    from .ui_config import display_data_requirements, display_advanced_requirements, display_upload_section, initialize_session_state
    
    # Compact data requirements overview
    with st.container():
        display_data_requirements()
    
    # Compact expandable details
    display_advanced_requirements()
    
    # Clean file upload section
    with st.container():
        uploaded, run_btn = display_upload_section()
    
    # Initialize session state for results
    initialize_session_state()
    
    # Validate inputs
    if run_btn and not uploaded:
        st.warning("Please upload a workbook first.")
        return None, None
    
    return uploaded, run_btn


def render_example_data_tab():
    """Render the example data tab content."""
    st.subheader("Example Data Format & Template")
    st.markdown(
        "Upload your data as an Excel file with three columns: `Date`, `Product`, and `ACR`.\n"
        "- `Date`: Month‑start dates (e.g., 2021-01-01).\n"
        "- `Product`: Sub‑strategic pillar names (you can include multiple pillars).\n"
        "- `ACR`: Actual Cost Revenue values."
    )
    
    # Create sample data
    sample_df = pd.DataFrame({
        "Date": ["2021-01-01", "2021-02-01", "2021-03-01"],
        "Product": ["Pillar A", "Pillar B", "Pillar A"],
        "ACR": [100.0, 120.5, 110.3]
    })
    st.dataframe(sample_df)
    
    # Create download button for template
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


def render_model_guide_tab():
    """Render the model guide/glossary tab content."""
    st.subheader("📝 Comprehensive Guide to Forecasting Methods & Metrics")
    
    # Key Metrics Section
    st.markdown("### 🎯 **Primary Performance Metric: WAPE**")
    st.markdown("""
    **WAPE (Weighted Absolute Percentage Error)** - Our primary accuracy measure:
    
    **🔢 Formula:** `sum(|Actual - Forecast|) / sum(|Actual|)`
    
    **💡 Why WAPE instead of MAPE?**
    - **Revenue-aligned:** Weights errors by actual dollar amounts, not percentages
    - **Robust:** Handles zero/small values better than traditional MAPE
    - **Business-relevant:** Higher revenue products have more impact on the error
    - **Interpretable:** 15% WAPE = forecasts are typically within 15% of actual revenue
    
    **📊 Accuracy Interpretation:**
    - **0-10%** = Excellent accuracy 🎯 (Professional-grade forecasting)
    - **10-20%** = Good accuracy 📈 (Acceptable for most business decisions)  
    - **20-30%** = Moderate accuracy ⚠️ (Use with caution for critical decisions)
    - **30%+** = Lower accuracy 🔴 (Consider manual adjustments or business overrides)
    
    **⚙️ Secondary Metrics for Model Selection:**
    - **SMAPE:** Symmetric MAPE, useful for comparing models on volatile data
    - **MASE:** Mean Absolute Scaled Error, compares performance to seasonal naive
    - **RMSE:** Root Mean Square Error, penalizes larger errors more heavily
    - **AIC/BIC:** Statistical model quality measures (lower is better)
    
    **🎲 Multi-Metric Ranking:** Models are ranked across ALL metrics for robustness, then the best average rank wins per product.
    """)
    
    # Model Selection Approaches
    st.markdown("### 🔄 **Model Selection Approaches**")
    st.markdown("""
    **🏆 Best per Product (Backtesting) - Recommended:**
    - **How it works:** Each product uses its individually best-performing model based on rigorous backtesting
    - **Selection criteria:** Pure backtesting WAPE with strict eligibility requirements
    - **Tie-breaking:** Mean WAPE → p75 WAPE → MASE → recent worst-month error
    - **Business-aware:** Polynomial models deprioritized for revenue forecasting
    - **Result:** Hybrid forecast combining the optimal model for each product
    
    **📊 Best per Product (Standard) - Statistical:**
    - **How it works:** Multi-metric ranking approach across all validation metrics
    - **Selection criteria:** Best average rank across WAPE, SMAPE, MASE, and RMSE
    - **Advantages:** More robust to outliers, considers multiple performance dimensions
    - **Use case:** When backtesting history is insufficient or for exploratory analysis
    
    **🎯 Individual Model Selection:**
    - **Single model:** Choose one model (SARIMA, ETS, Prophet, etc.) for all products
    - **Use case:** When you want consistency across products or have domain expertise
    - **Advantage:** Simpler interpretation and deployment
    """)

    # Backtesting methodology
    st.markdown("### 🧪 **Rigorous Backtesting Methodology**")
    
    st.markdown("""
    **🔄 Walk-Forward Validation Process:**
    1. **Expanding windows:** Train on increasing historical data, predict forward
    2. **Step size:** 6-month steps to capture multiple seasonal cycles  
    3. **Gap handling:** 0-month gap by default (configurable for autocorrelation)
    4. **Validation horizon:** 6-month forward predictions (mimics real forecasting)
    5. **Multiple folds:** Creates multiple out-of-sample validation windows
    
    **✅ Strict Eligibility Criteria:**
    - **History requirement:** ≥24 months of data for reliable backtesting
    - **Minimum folds:** ≥2 backtesting windows for statistical significance
    - **Stability check:** p95 WAPE ≤ 2× mean WAPE (excludes unstable models)
    - **Baseline beating:** Must perform ≥5% better WAPE than Seasonal-Naive
    - **MASE requirement:** MASE < 1.0 (better than seasonal baseline)
    
    **🎯 Selection Scoring (Backtesting Mode):**
    1. **Primary:** Mean WAPE across all backtesting folds
    2. **Tie-break 1:** p75 WAPE (75th percentile performance)
    3. **Tie-break 2:** MASE (scaled error vs seasonal naive)
    4. **Tie-break 3:** Recent worst-month error (recency bias)
    
    **🛡️ Business-Aware Safeguards:**
    - **Polynomial deprioritization:** Poly-2/Poly-3 models used only if no alternatives
    - **Revenue focus:** Optimized for consumptive business revenue patterns
    - **Fallback logic:** Seasonal-Naive or ETS[A,A,A] when eligibility fails
    """)
    
    # Individual Models Section
    st.markdown("### 🔮 **Individual Forecasting Models**")
    
    st.markdown("""
    **🏆 Best per Product (Backtesting)**
    - **What it is:** Intelligent hybrid combining the best-performing model per product
    - **Selection method:** Rigorous walk-forward backtesting with strict eligibility
    - **Advantages:** Optimizes for each product's unique patterns, highest accuracy
    - **Business focus:** Deprioritizes polynomial models for revenue forecasting
    - **WAPE display:** Average of selected models' backtesting performance
    
    **📊 Best per Product (Standard)**  
    - **What it is:** Multi-metric statistical selection across all validation measures
    - **Selection method:** Best average rank across WAPE, SMAPE, MASE, RMSE
    - **Advantages:** More robust to outliers, works with shorter history
    - **Use case:** When backtesting eligibility is insufficient
    
    **📈 SARIMA (Seasonal AutoRegressive Integrated Moving Average)**
    - **Strengths:** Excellent for seasonal business data with clear patterns
    - **Our approach:** Dual AIC/BIC optimization, then validation-based final selection
    - **Best for:** Revenue with strong seasonal cycles, sufficient history (24+ months)
    - **Limitations:** Requires stable seasonal patterns, computationally intensive
    
    **⚡ ETS (Exponential Smoothing)**
    - **Strengths:** Automatically adapts to Error, Trend, Seasonality components
    - **Our approach:** Auto-optimization of smoothing parameters
    - **Best for:** Growing businesses with evolving seasonal patterns
    - **Limitations:** May struggle with abrupt changes or complex seasonality
    
    **🚀 Prophet (Facebook's Business Forecaster)**
    - **Strengths:** Built for business data, handles holidays and growth changes
    - **Our approach:** Optional holiday effects (US/worldwide), growth assumptions
    - **Best for:** Revenue affected by holidays, businesses with growth inflections
    - **Limitations:** Can be overconfident in trend extrapolation
    
    **🤖 Auto-ARIMA**
    - **Strengths:** Automated statistical modeling with parameter optimization
    - **Our approach:** Smart parameter search with business validation
    - **Best for:** Stationary data with complex autocorrelation structures
    - **Limitations:** May miss seasonal patterns in shorter series
    
    **🌳 LightGBM (Gradient Boosting)**
    - **Strengths:** Captures complex non-linear patterns, handles multiple features
    - **Our approach:** Engineered lag features, calendar effects, hyperparameter tuning
    - **Best for:** Complex businesses with multiple driving factors
    - **Limitations:** Black box, requires sufficient training data
    
    **📐 Polynomial Models (Poly-2/Poly-3)**
    - **What they are:** Pure mathematical trend fitting (quadratic/cubic curves)
    - **⚠️ Business warning:** Can create unrealistic growth projections
    - **Our safeguards:** Deprioritized in business-aware selection
    - **Use case:** Only when all other models fail eligibility requirements
    
    **🔧 Interactive Adjustments**
    - **Purpose:** Management overrides and scenario planning
    - **How it works:** Apply growth/haircut percentages starting from any future month
    - **Use cases:** Market condition changes, business strategy shifts, conservative planning
    """)


def render_footer():
    """Render the application footer."""
    st.markdown("**Created by: Jake Moura**")
