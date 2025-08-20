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
    st.subheader("📝 Glossary of Forecasting Methods & Metrics")
    
    # Key Metrics Section
    st.markdown("### 🎯 **Key Performance Metrics**")
    st.markdown("""
    **WAPE (Weighted Absolute Percentage Error)** - Primary accuracy measure:
    - **What it means:** Dollar-weighted error: sum(|A−F|) / sum(|A|)
    - **Why lower is better:** Aligns with revenue impact; robust to small denominators
    - **Example:** 10% WAPE = dollar-weighted forecasts typically within 10% of actual values
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
    2. **Test accuracy** using backtesting for each model-product combination
    3. **Calculate all metrics** (WAPE, SMAPE, MASE, RMSE) for each model's predictions per product
    4. **Rank models** across all four metrics for each specific product
    5. **Winner per product:** Model with the **best average ranking across all metrics** for that specific product is selected
    6. **Hybrid approach:** Combine the best models for each product into one optimized forecast
    
    **Traditional Overall Selection (Alternative):**
    1. **Run all models** on your historical data
    2. **Test accuracy** using validation (unseen historical data)
    3. **Calculate average WAPE** across all products for each model
    4. **Winner:** Model with **lowest average WAPE** across all products
    5. **Why this works:** The most accurate on historical data is likely most accurate for future predictions
    
    **Why Product-by-Product is Better:**
    - Different products may have different patterns (seasonal vs. trending)
    - One model might excel at growth patterns, another at seasonal patterns
    - Optimizes accuracy for each specific data pattern
    - Often achieves lower overall error than any single model
    """)

    # Selection modes and backtesting guide
    st.markdown("### 🧭 **Smart Backtesting & Model Selection**")
    
    st.markdown("""
    **Backtesting‑Only Selection (WAPE‑first):**
    - Models are selected per product strictly by cross‑validated backtesting WAPE
    - Scoring: mean WAPE → p75 WAPE → MASE; unstable models are excluded (p95 > 2× mean)
    - Eligibility: ≥24 months history and ≥2 folds (h=6); must beat Seasonal‑Naive by ≥5% WAPE and have MASE < 1.0
    - If data is too short, we provisionally use Seasonal‑Naive (or ETS[A,A,A])
    """)
    
    st.markdown("### 🔎 **Backtesting Details**")
    
    st.markdown("""
    **How It Works:**
    1. Expanding‑window backtests with step = 6 months, gap = 0, last 12–18 months
    2. Compute WAPE/SMAPE/MASE/RMSE; selection uses WAPE only (others for tie‑breaks/diagnostics)
    3. Cache results for speed and determinism; single‑threaded fits
    4. Fallback to Seasonal‑Naive when CV cannot be run
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


def render_footer():
    """Render the application footer."""
    st.markdown("**Created by: Jake Moura (jakemoura@microsoft.com)**")
