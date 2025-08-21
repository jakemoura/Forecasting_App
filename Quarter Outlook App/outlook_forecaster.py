"""
Quarterly Outlook Forecaster - Daily Data Edition (Modular Version)

Specialized forecaster for quarterly business outlook based on partial quarter data.
Uses fiscal year calendar (July-June) where Q1 starts in July.

Key Features:
1. FISCAL QUARTER DETECTION: Automatically detects current fiscal quarter and progress
2. DAILY DATA PROCESSING: Handles daily business data with weekday/weekend awareness
3. QUARTERLY FORECASTING: Projects full quarter performance from partial data
4. BUSINESS CALENDAR: Respects fiscal year timing (Q1: Jul-Sep, Q2: Oct-Dec, Q3: Jan-Mar, Q4: Apr-Jun)

Philosophy: Quick, actionable quarterly outlook for business decision-making.

Author: Jake Moura (jakemoura@microsoft.com)
"""

import io
import warnings
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Import modular components
from modules.fiscal_calendar import get_fiscal_quarter_info
from modules.data_processing import read_any_excel, coerce_daily_dates, analyze_daily_data
from modules.quarterly_forecasting import forecast_quarter_completion, apply_capacity_adjustment_to_forecast, apply_conservatism_adjustment_to_forecast
from modules.ui_components import (
    create_forecast_summary_table, create_forecast_visualization, create_progress_indicator,
    display_model_comparison, create_excel_download, display_spike_analysis,
    display_capacity_adjustment_impact, display_backtesting_details
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============ Outlook Page Setup ============
st.set_page_config(
    page_title="Quarterly Outlook Forecaster", 
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

st.title("📈 Quarterly Outlook Forecaster")
st.markdown("**Daily Data Edition** - Project quarter performance from partial data using fiscal year calendar (July-June)")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Analysis date info (will be auto-set from data)
    st.subheader("Analysis Date")
    if 'outlook_last_data_date' in st.session_state:
        analysis_date = st.session_state.outlook_last_data_date
        st.success(f"**From your data:** {analysis_date.strftime('%B %d, %Y')}")
        st.caption("✅ Analysis date set from last date in uploaded file")
        
        # Display quarter info based on uploaded data
        current_quarter_info = get_fiscal_quarter_info(analysis_date)
        st.subheader("📊 Quarter Being Forecasted")
        st.info(f"**{current_quarter_info['quarter_name']}**")
        st.info(f"**Period:** {current_quarter_info['quarter_start'].strftime('%b %d')} - {current_quarter_info['quarter_end'].strftime('%b %d, %Y')}")
        
        # Show data range if available
        if 'outlook_data_range' in st.session_state:
            data_start, data_end = st.session_state.outlook_data_range
            st.caption(f"📅 Your data: {data_start.strftime('%b %d')} - {data_end.strftime('%b %d, %Y')}")
            
            # Show fiscal year explanation
            st.caption("💡 **Fiscal Year Calendar:** Q1 (Jul-Sep), Q2 (Oct-Dec), Q3 (Jan-Mar), Q4 (Apr-Jun)")
    else:
        analysis_date = datetime.now()
        st.info("**Default:** Using today's date")
        st.caption("⏳ Upload data to set analysis date and quarter")
        
        # Display default quarter info
        current_quarter_info = get_fiscal_quarter_info(analysis_date)
        st.subheader("📊 Current Quarter (Default)")
        st.info(f"**{current_quarter_info['quarter_name']}**")
        st.info(f"**Period:** {current_quarter_info['quarter_start'].strftime('%b %d')} - {current_quarter_info['quarter_end'].strftime('%b %d, %Y')}")
        st.caption("💡 **Fiscal Year Calendar:** Q1 (Jul-Sep), Q2 (Oct-Dec), Q3 (Jan-Mar), Q4 (Apr-Jun)")
    
    # Add forecast conservatism adjustment
    st.subheader("Forecast Adjustment")
    forecast_conservatism = st.slider(
        "Forecast Conservatism", 
        90, 110, 97, 1, 
        help="Adjust forecasts to account for systematic bias. 97% = 3% more conservative based on validation testing. 100% = no adjustment.",
        format="%d%%"
    )
    st.caption(f"🎯 Applying {forecast_conservatism}% factor - {'more conservative' if forecast_conservatism < 100 else 'more optimistic' if forecast_conservatism > 100 else 'no adjustment'}")
    
    st.subheader("Capacity Constraints")
    apply_capacity_adjustment = st.checkbox("Apply capacity constraints", value=False, help="Adjust forecasts for operational limitations that prevent achieving projected growth")
    
    # Store in session state for use in results tab
    st.session_state['apply_capacity_adjustment'] = apply_capacity_adjustment
    
    capacity_adjustment = 1.0  # Default value (no adjustment)
    if apply_capacity_adjustment:
        # Quick capacity calculator
        with st.expander("💡 Capacity Calculator Helper", expanded=False):
            st.markdown("**Calculate capacity factor based on weekly revenue loss:**")
            
            # Auto-populate values from current forecast results
            default_unconstrained = 325.0  # Fallback default in millions
            default_weeks_remaining = 3.4
            
            # Get actual forecast total from processed results in this session
            if 'outlook_forecasts' in st.session_state and st.session_state.outlook_forecasts:
                try:
                    # Calculate the actual unconstrained total from best models with conservatism adjustment
                    actual_total = 0
                    conservatism_factor = forecast_conservatism / 100.0
                    for product, data in st.session_state.outlook_forecasts.items():
                        if 'forecast' in data and 'summary' in data['forecast'] and 'quarter_total' in data['forecast']['summary']:
                            forecast_result = data['forecast']
                            original_quarter_total = forecast_result['summary']['quarter_total']
                            actual_to_date = forecast_result['actual_to_date']
                            
                            # Apply conservatism adjustment to the forecast portion only
                            forecast_portion = original_quarter_total - actual_to_date
                            adjusted_forecast_portion = forecast_portion * conservatism_factor
                            adjusted_quarter_total = actual_to_date + adjusted_forecast_portion
                            actual_total += adjusted_quarter_total
                    
                    if actual_total > 0:
                        default_unconstrained = actual_total / 1000000  # Convert to millions
                    
                    # Estimate weeks remaining from days (use first forecast for timing)
                    first_forecast = list(st.session_state.outlook_forecasts.values())[0]
                    if 'forecast' in first_forecast and 'quarter_progress' in first_forecast['forecast']:
                        progress = first_forecast['forecast']['quarter_progress']
                        if 'days_remaining' in progress:
                            days_remaining = progress['days_remaining']
                            default_weeks_remaining = max(0.1, days_remaining / 7)
                except (KeyError, IndexError, TypeError, ZeroDivisionError):
                    # If we can't access forecast data, use defaults
                    pass
            
            col1, col2, col3 = st.columns(3)
            with col1:
                unconstrained_forecast = st.number_input("Unconstrained forecast ($M)", value=default_unconstrained, min_value=0.0, step=1.0)
            with col2:
                weekly_revenue_loss = st.number_input("Weekly revenue loss ($M)", value=0.0, min_value=0.0, step=0.5)
            with col3:
                weeks_remaining = st.number_input("Weeks remaining", value=default_weeks_remaining, min_value=0.1, step=0.1)
            
            if unconstrained_forecast > 0 and weekly_revenue_loss > 0:
                total_loss = weekly_revenue_loss * weeks_remaining
                capacity_factor_calculated = max(0.5, 1 - (total_loss / unconstrained_forecast))
                st.info(f"💡 **Suggested factor:** {capacity_factor_calculated:.3f}")
                st.caption(f"📊 **Calculation:** {total_loss:.1f}M total loss / {unconstrained_forecast:.1f}M forecast = {(total_loss/unconstrained_forecast)*100:.1f}% reduction")
                
                if st.button("📋 Use Calculated Factor"):
                    capacity_adjustment = capacity_factor_calculated
        
        capacity_adjustment = st.slider(
            "Capacity limitation factor", 
            0.5, 1.0, capacity_adjustment, 0.001, 
            help="Multiplier to reduce forecasts due to capacity constraints. 0.85 = 15% reduction from capacity limits",
            format="%.3f"
        )
        st.caption(f"🔧 Applying {(1-capacity_adjustment)*100:.1f}% reduction to account for capacity constraints")
        
        # Show dollar impact if forecast data is available
        if 'outlook_forecasts' in st.session_state and st.session_state.outlook_forecasts:
            try:
                total_unconstrained = sum([f['forecast']['quarter_total'] for f in st.session_state.outlook_forecasts.values() if 'forecast' in f and 'quarter_total' in f['forecast']])
                if total_unconstrained > 0:
                    revenue_reduction = total_unconstrained * (1 - capacity_adjustment)
                    st.warning(f"⚠️ **Impact:** ${revenue_reduction:,.0f} reduction from ${total_unconstrained:,.0f} forecast (capacity constraints)")
            except (KeyError, TypeError) as e:
                st.warning("⚠️ **Note:** This reduces all forecast models to account for operational limitations that prevent achieving projected growth rates.")
        else:
            st.warning("⚠️ **Note:** This reduces all forecast models to account for operational limitations that prevent achieving projected growth rates.")
    
    # Store capacity adjustment in session state for use in results tab
    st.session_state['capacity_adjustment'] = capacity_adjustment
    
    st.subheader("Monthly Renewal Detection")
    spike_detection = st.checkbox("Detect monthly renewals", value=True, help="Automatically detect and forecast monthly subscription renewals (non-consumptive SKUs with specific revenue recognition dates)")
    spike_threshold = 2.0  # Default value
    spike_intensity = 0.85  # Default value
    if spike_detection:
        spike_threshold = st.slider("Spike sensitivity", 1.5, 4.0, 2.0, 0.1, help="Multiplier above baseline to consider a renewal spike. Higher = less sensitive")
        spike_intensity = st.slider("Spike intensity", 0.5, 1.0, 0.85, 0.05, help="Reduce spike impact to avoid over-forecasting. 0.85 = 15% reduction from historical spikes")
        st.caption("💡 For non-consumptive renewals, try 2.0-3.0 sensitivity. Lower intensity (0.7-0.9) for more conservative forecasts.")

# Main content
tab_upload, tab_results = st.tabs(["📁 Data Upload", "📊 Outlook Results"])

with tab_upload:
    st.markdown("### 📁 Upload Daily Data")
    st.markdown("Upload an Excel file with daily business data. Required columns: **Date**, **Product**, **ACR**")
    
    # File requirements and purview warning
    st.info("""
    **📋 File Requirements:**
    • Excel file with columns: **"Date", "Product", "ACR"**
    • Must be labeled as **"General"** (not Confidential/Highly Confidential)
    • Check the top of Excel ribbon for Purview classification labels
    """)
    
    # Example template section
    with st.expander("📋 Example Data Format & Download Template", expanded=False):
        st.markdown("""
        **Required Columns:**
        - `Date`: Daily dates (e.g., 2025-01-01, 1/1/2025, etc.)
        - `Product`: Business product names
        - `ACR`: Daily values to forecast (revenue, sales, etc.)
        """)
        
        # Create sample data with fiscal year context
        sample_df = pd.DataFrame({
            "Date": [
                "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-05",
                "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-05",
                "2025-06-27", "2025-06-28", "2025-06-29", "2025-06-30"
            ],
            "Product": [
                "Product A", "Product A", "Product A", "Product A", "Product A",
                "Product B", "Product B", "Product B", "Product B", "Product B", 
                "Product A", "Product A", "Product A", "Product A"
            ],
            "ACR": [
                1250.0, 1180.0, 1320.0, 1400.0, 980.0,
                850.0, 920.0, 875.0, 950.0, 760.0,
                1380.0, 1290.0, 1450.0, 1520.0
            ]
        })
        
        st.markdown("**Example Data Preview:**")
        st.dataframe(sample_df, use_container_width=True)
        
        # Create downloadable Excel template
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            sample_df.to_excel(writer, index=False, sheet_name="Daily_Data_Template")
            
            # Add a second sheet with instructions
            instructions_df = pd.DataFrame({
                "Instructions": [
                    "This template is for the Quarterly Outlook Forecaster",
                    "Uses fiscal year calendar: Q1 (Jul-Sep), Q2 (Oct-Dec), Q3 (Jan-Mar), Q4 (Apr-Jun)",
                    "Upload daily data to get quarterly projections",
                    "",
                    "Column Requirements:",
                    "• Date: Daily dates in any standard format",
                    "• Product: Business product names", 
                    "• ACR: Daily values (revenue, sales, units, etc.)",
                    "",
                    "Tips for Best Results:",
                    "• Include at least 5-10 days of current quarter data",
                    "• Use consistent daily data (avoid large gaps)",
                    "• Consider business calendars (holidays, weekends)",
                    "• Multiple products can be analyzed together"
                ]
            })
            instructions_df.to_excel(writer, index=False, sheet_name="Instructions")
        
        buf.seek(0)
        st.download_button(
            "📥 Download Daily Data Template",
            data=buf,
            file_name="daily_outlook_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download Excel template with sample daily data and instructions"
        )
    
    uploaded = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'])
    
    if uploaded:
        try:
            # Read and validate data
            raw = read_any_excel(io.BytesIO(uploaded.read()))
            required = {"Date", "Product", "ACR"}
            
            if not required.issubset(raw.columns):
                st.error(f"❌ Missing required columns. Found: {list(raw.columns)}. Required: {required}")
                st.stop()
            
            # Process dates
            try:
                raw["Date"] = coerce_daily_dates(raw["Date"])
            except ValueError as e:
                st.error(f"❌ Date parsing error: {e}")
                st.stop()
            
            # Auto-set analysis date from the last date in the data
            last_data_date = raw["Date"].max()
            first_data_date = raw["Date"].min()
            analysis_date = last_data_date
            
            # Store in session state for sidebar display
            st.session_state.outlook_last_data_date = analysis_date
            st.session_state.outlook_data_range = (first_data_date, last_data_date)
            
            # Update quarter info based on actual data date
            current_quarter_info = get_fiscal_quarter_info(analysis_date)
            
            st.success(f"📅 **Analysis Date Auto-Set:** {analysis_date.strftime('%B %d, %Y')} (last date in your data)")
            st.info(f"🎯 **Forecasting Quarter:** {current_quarter_info['quarter_name']} ({current_quarter_info['quarter_start'].strftime('%b %d')} - {current_quarter_info['quarter_end'].strftime('%b %d, %Y')})")
            
            # Add debugging info to help understand the date logic
            with st.expander("📋 Date Analysis Details", expanded=False):
                st.write(f"**Data Date Range:** {first_data_date.strftime('%B %d, %Y')} to {last_data_date.strftime('%B %d, %Y')}")
                st.write(f"**Last Date Month:** {last_data_date.strftime('%B')} ({last_data_date.month})")
                st.write(f"**Fiscal Year Logic:** Month {last_data_date.month} → {current_quarter_info['quarter_name']}")
                
                # Show fiscal year mapping
                st.write("**Fiscal Quarter Mapping:**")
                st.write("• Apr-Jun = Q4 of current fiscal year")
                st.write("• Jul-Sep = Q1 of next fiscal year") 
                st.write("• Oct-Dec = Q2 of next fiscal year")
                st.write("• Jan-Mar = Q3 of current fiscal year")
                
                if last_data_date.month >= 4 and last_data_date.month <= 6:
                    expected_q = f"FY{last_data_date.year % 100:02d} Q4"
                elif last_data_date.month >= 7 and last_data_date.month <= 9:
                    expected_q = f"FY{(last_data_date.year + 1) % 100:02d} Q1"
                elif last_data_date.month >= 10 and last_data_date.month <= 12:
                    expected_q = f"FY{(last_data_date.year + 1) % 100:02d} Q2"
                else:  # Jan-Mar
                    expected_q = f"FY{last_data_date.year % 100:02d} Q3"
                
                st.write(f"**Expected Quarter for {last_data_date.strftime('%B %Y')}:** {expected_q}")
                
                if expected_q != current_quarter_info['quarter_name']:
                    st.warning(f"⚠️ **Mismatch detected!** Expected {expected_q} but got {current_quarter_info['quarter_name']}")
                else:
                    st.success(f"✅ **Quarter calculation correct:** {current_quarter_info['quarter_name']}")

            # Show data summary
            st.subheader("📊 Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(raw))
            with col2:
                st.metric("Products", raw["Product"].nunique())
            with col3:
                st.metric("Date Range", f"{(last_data_date - first_data_date).days + 1} days")
            with col4:
                total_value = raw["ACR"].sum()
                st.metric("Total Value", f"${total_value:,.0f}")
            
            # Process by product
            forecasts_by_product = {}
            
            for product in raw["Product"].unique():
                product_data = raw[raw["Product"] == product].copy()
                product_series = product_data.set_index("Date")["ACR"].sort_index()
                
                # Generate forecast for this product
                result = forecast_quarter_completion(
                    product_series, 
                    current_date=analysis_date,
                    detect_spikes=spike_detection,
                    spike_threshold=spike_threshold,
                    spike_intensity=spike_intensity
                )
                
                if 'error' not in result:
                    forecasts_by_product[product] = {
                        'forecast': result,
                        'data': product_series
                    }
            
            # Store results in session state
            st.session_state.outlook_forecasts = forecasts_by_product
            st.session_state.outlook_analysis_date = analysis_date
            
            if forecasts_by_product:
                st.success(f"✅ **Successfully processed {len(forecasts_by_product)} products!** Switch to the 'Outlook Results' tab to view forecasts.")
            else:
                st.warning("⚠️ No valid forecasts could be generated. Check your data and date ranges.")
                
        except Exception as e:
            st.error(f"❌ **Error processing file:** {str(e)}")
            st.write("Please check your file format and data structure.")

with tab_results:
    if 'outlook_forecasts' in st.session_state:
        forecasts_data = st.session_state.outlook_forecasts
        analysis_date = st.session_state.get('outlook_analysis_date', datetime.now())
        
        # ============ TOP SECTION - MAIN RESULTS ============
        st.markdown("### 📊 Quarterly Outlook Results")
        # Compute average WAPEs across products to auto-pick mode
        std_wapes, bt_wapes = [], []
        for _, pdata in forecasts_data.items():
            f = pdata.get('forecast', {})
            std_best = f.get('best_model')
            std_wape = f.get('wape_scores', {}).get(std_best, None)
            if std_wape is not None and std_wape != float('inf'):
                std_wapes.append(float(std_wape))
            bt = f.get('backtesting', {})
            bt_best = bt.get('best_model')
            bt_wape = bt.get('per_model_wape', {}).get(bt_best, None) if bt_best else None
            if bt_wape is not None and bt_wape != float('inf'):
                bt_wapes.append(float(bt_wape))
        std_avg = sum(std_wapes)/len(std_wapes) if std_wapes else float('inf')
        bt_avg = sum(bt_wapes)/len(bt_wapes) if bt_wapes else float('inf')
        # Validation Strategy: Always prefer backtesting for overfitting protection
        st.info(
            "✅ **Enhanced Validation Strategy**\n\n"
            "• **Primary Method**: Backtesting validation (walk-forward with recent weighting)\n"
            "• **Why Backtesting**: Tests models on unknown future data to prevent overfitting\n"
            "• **Automatic Fallback**: Uses standard validation only when insufficient data\n"
            "• **Business Benefit**: More reliable forecasts for quarterly planning decisions"
        )
        
        # Always use backtesting mode for optimal validation
        selection_mode = "Backtesting"

        with st.expander("Why Use Backtesting Validation?", expanded=False):
            st.markdown(
                """
                ## 🛡️ **Overfitting Protection**
                • **Backtesting** validates models on unknown future data, preventing overfitting to historical patterns
                • **Standard validation** uses in-sample data which can create false confidence in overfit models
                
                ## 📊 **Daily Quarterly Optimization**
                • **Recent focus**: Validates ONLY on last 2-3 weeks (not ancient history)
                • **Short horizons**: 2-day validation windows appropriate for daily business volatility  
                • **Recent weighting**: Latest validation folds weighted 4x more heavily
                • **Limited scope**: Max 7 validation folds clustered at end of time series
                
                ## 💼 **Business Benefits**
                • **Reliable forecasts**: Models proven to work on unseen data
                • **WAPE metric**: Dollar-weighted accuracy aligned with revenue impact
                • **Automatic fallback**: Uses standard validation when data is insufficient
                
                ## ⚙️ **Technical Details**
                • **Walk-forward validation**: Expanding training windows with future prediction
                • **Exponential weighting**: Recent validation folds weighted 4x more heavily
                • **Stability checks**: Filters out unstable or unreasonable model predictions
                """
            )
        
        # Overall summary - calculate total from chosen models with adjustments
        total_forecast = 0
        total_actual = 0

        # Get adjustment factors
        conservatism_factor = forecast_conservatism / 100.0
        capacity_factor = st.session_state.get('capacity_adjustment', 1.0) if st.session_state.get('apply_capacity_adjustment', False) else 1.0
        
        # Calculate totals using chosen strategy (per product)
        standard_total_forecast = 0.0

        for product, data in forecasts_data.items():
            forecast_result = data['forecast']
            actual_to_date = forecast_result['actual_to_date']

            # Standard model and WAPE
            std_model = forecast_result.get('best_model')
            std_wape = None
            if std_model and 'wape_scores' in forecast_result:
                try:
                    std_wape = float(forecast_result['wape_scores'].get(std_model, float('inf')))
                except Exception:
                    std_wape = float('inf')

            # Backtesting model and WAPE
            bt = forecast_result.get('backtesting', {})
            bt_model = bt.get('best_model')
            bt_wape = None
            if bt_model:
                try:
                    bt_wape = float(bt.get('per_model_wape', {}).get(bt_model, float('inf')))
                except Exception:
                    bt_wape = float('inf')

            # Decide chosen model
            chosen_model = std_model
            if selection_mode == 'Backtesting' and bt_model and bt_model in forecast_result.get('forecasts', {}):
                chosen_model = bt_model

            # Quarter total for chosen
            if chosen_model and chosen_model in forecast_result.get('forecasts', {}):
                original_quarter_total = forecast_result['forecasts'][chosen_model].get('quarter_total', forecast_result['summary']['quarter_total'])
            else:
                original_quarter_total = forecast_result['summary']['quarter_total']
            
            # Apply conservatism adjustment to the forecast portion only
            forecast_portion = original_quarter_total - actual_to_date
            adjusted_forecast_portion = forecast_portion * conservatism_factor
            
            # Apply capacity adjustment to the forecast portion as well
            capacity_adjusted_forecast_portion = adjusted_forecast_portion * capacity_factor
            
            adjusted_quarter_total = actual_to_date + capacity_adjusted_forecast_portion
            
            total_forecast += adjusted_quarter_total
            total_actual += actual_to_date

            # Baseline Standard adjusted total for impact calc
            if std_model and std_model in forecast_result.get('forecasts', {}):
                std_total = forecast_result['forecasts'][std_model].get('quarter_total', forecast_result['summary']['quarter_total'])
            else:
                std_total = forecast_result['summary']['quarter_total']
            std_portion = std_total - actual_to_date
            std_adj_total = actual_to_date + (std_portion * conservatism_factor * capacity_factor)
            standard_total_forecast += std_adj_total
        
        total_remaining = total_forecast - total_actual

        # Top blue box: summary + show both Standard and Backtesting average WAPE
        delta_total = total_forecast - standard_total_forecast
        std_avg_text = f"{std_avg:.1%}" if std_avg != float('inf') else "n/a"
        bt_avg_text = f"{bt_avg:.1%}" if bt_avg != float('inf') else "n/a"

        # Count validation methods actually used
        bt_count = 0
        std_count = 0
        total_folds = 0
        
        for product, data in forecasts_data.items():
            forecast_result = data['forecast']
            bt = forecast_result.get('backtesting', {})
            bt_model = bt.get('best_model')
            
            if bt_model and bt_model in forecast_result.get('forecasts', {}):
                bt_count += 1
                # Count validation folds for this product
                validation_details = bt.get('validation_details', {})
                if bt_model in validation_details:
                    model_details = validation_details[bt_model]
                    if isinstance(model_details, dict) and 'iterations' in model_details:
                        total_folds += model_details['iterations']
            else:
                std_count += 1
        
        # Create validation summary
        validation_info = f"✅ **Validation Results ({len(forecasts_data)} products):**\n\n"
        
        if bt_count > 0:
            avg_folds = total_folds / bt_count if bt_count > 0 else 0
            validation_info += f"• **{bt_count} products**: Backtesting validation (avg {avg_folds:.1f} folds, recent-weighted)\n"
        
        if std_count > 0:
            validation_info += f"• **{std_count} products**: Standard validation (insufficient data for backtesting)\n"
        
        validation_info += f"\n**Performance**: Standard WAPE: {std_avg_text} | Backtesting WAPE: {bt_avg_text}\n"
        validation_info += f"**Impact vs all-Standard**: {('+' if delta_total>=0 else '')}${delta_total:,.0f}"
        
        st.success(validation_info)
        
        # Main metrics at the top
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Actual to Date", f"${total_actual:,.0f}")
        with col2:
            st.metric("Forecast Remaining", f"${total_remaining:,.0f}")
        with col3:
            st.metric("Quarter Total", f"${total_forecast:,.0f}")
        with col4:
            # Quarter progress from first product
            first_product_data = list(forecasts_data.values())[0]
            if 'quarter_progress' in first_product_data['forecast']:
                progress = first_product_data['forecast']['quarter_progress']
                completion_pct = progress['completion_pct']
                st.metric("Quarter Progress", f"{completion_pct:.1f}%")
        
        # ============ MONTHLY BREAKDOWN ============
        st.markdown("### 📅 Monthly Breakdown")
        
        from modules.ui_components import display_monthly_splits
        
        conservatism_factor = forecast_conservatism / 100.0
        capacity_factor = st.session_state.get('capacity_adjustment', 1.0) if st.session_state.get('apply_capacity_adjustment', False) else 1.0
        
        display_monthly_splits(forecasts_data, analysis_date, conservatism_factor, capacity_factor, show_header=False)
        
        # ============ CHART SECTION ============
        st.markdown("### 📈 Forecast Visualization")
        
        # Show chart for first product (or allow selection if multiple products)
        if len(forecasts_data) == 1:
            # Single product - show chart directly
            product_name = list(forecasts_data.keys())[0]
            product_data = list(forecasts_data.values())[0]
            
            forecast_result = product_data['forecast'].copy()
            product_series = product_data['data']
            
            # Apply adjustments for chart
            if conservatism_factor != 1.0:
                forecast_result = apply_conservatism_adjustment_to_forecast(forecast_result, conservatism_factor)
            if st.session_state.get('apply_capacity_adjustment', False):
                capacity_factor = st.session_state.get('capacity_adjustment', 1.0)
                forecast_result = apply_capacity_adjustment_to_forecast(forecast_result, capacity_factor)
            
            # Decide best model for chart based on selection
            best_model = forecast_result.get('best_model', None)
            bt_model = forecast_result.get('backtesting', {}).get('best_model')
            std_wape = forecast_result.get('wape_scores', {}).get(best_model, float('inf'))
            bt_wape = None
            if bt_model:
                bt_wape = forecast_result.get('backtesting', {}).get('per_model_wape', {}).get(bt_model, float('inf'))
            if selection_mode == 'Backtesting' and bt_model and bt_model in forecast_result.get('forecasts', {}):
                best_model = bt_model

            # Create chart
            if 'forecasts' in forecast_result:
                from modules.ui_components import create_forecast_visualization
                chart, selected_model = create_forecast_visualization(
                    product_series, 
                    forecast_result['forecasts'], 
                    forecast_result['quarter_info'], 
                    best_model=best_model,
                    backtesting_results=forecast_result.get('backtesting', {}),
                    wape_scores=forecast_result.get('wape_scores', {})
                )
                if chart:
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Add chart legend for backtesting indicators
                    validation_details = forecast_result.get('backtesting', {}).get('validation_details', {}).get(selected_model, {})
                    has_validation = validation_details.get('iterations', 0) > 0
                    has_periods = len(validation_details.get('validation_periods', [])) > 0
                    
                    if has_validation:
                        if has_periods:
                            st.caption("📊 **Chart Legend:** Blue line = Historical data | Orange dashed = Forecast | 🔴 Red circles = Detected spikes | 🔺 Green triangles = Validation start points | Purple dotted = Backtesting predictions")
                        else:
                            st.caption("📊 **Chart Legend:** Blue line = Historical data | Orange dashed = Forecast | 🔴 Red circles = Detected spikes | 🔺 Green triangles = Validation points (recent periods)")
            # Rationale for selection
            if 'model_evaluation' in forecast_result:
                standard_model = forecast_result.get('best_model')
                wape_scores = forecast_result.get('wape_scores', {})
                std_wape = wape_scores.get(standard_model, float('inf'))
                bt = forecast_result.get('backtesting', {})
                bt_model = bt.get('best_model')
                bt_wape = None
                if bt_model:
                    bt_wape = bt.get('per_model_wape', {}).get(bt_model, None)
                # Determine mode used for rationale display
                chosen_model = standard_model
                mode_used = 'Standard'
                if selection_mode == 'Backtesting' and bt_model:
                    chosen_model = bt_model
                    mode_used = 'Backtesting'
                chosen_total = forecast_result['forecasts'].get(chosen_model, {}).get('quarter_total', forecast_result['summary']['quarter_total'])
                alt_total = forecast_result['forecasts'].get(standard_model if chosen_model != standard_model else (bt_model or standard_model), {}).get('quarter_total', chosen_total)
                delta_total = chosen_total - alt_total
                # Compose rationale text in blue info box
                rationale = f"Using {mode_used}: {chosen_model}. "
                details = []
                if std_wape != float('inf'):
                    details.append(f"Standard WAPE: {std_wape:.1%}")
                if bt_wape is not None and bt_wape != float('inf'):
                    details.append(f"Backtesting WAPE: {bt_wape:.1%}")
                if details:
                    rationale += " | ".join(details)
                impact = f"Impact vs alternate: {('+' if delta_total>=0 else '')}${delta_total:,.0f} on quarter total"
                st.info(f"{rationale}\n\n{impact}")
            
            # Add backtesting breakdown dropdown
            if forecast_result.get('backtesting', {}).get('validation_details'):
                with st.expander("🔄 Backtesting Validation Breakdown", expanded=False):
                    display_backtesting_details(forecast_result['backtesting'])
        else:
            # Multiple products - show selection
            selected_product = st.selectbox(
                "Select product to visualize:",
                options=list(forecasts_data.keys()),
                help="Choose which product to display in the forecast chart"
            )
            
            if selected_product:
                product_data = forecasts_data[selected_product]
                forecast_result = product_data['forecast'].copy()
                product_series = product_data['data']
                
                # Apply adjustments for chart
                if conservatism_factor != 1.0:
                    forecast_result = apply_conservatism_adjustment_to_forecast(forecast_result, conservatism_factor)
                if st.session_state.get('apply_capacity_adjustment', False):
                    capacity_factor = st.session_state.get('capacity_adjustment', 1.0)
                    forecast_result = apply_capacity_adjustment_to_forecast(forecast_result, capacity_factor)
                
                # Decide best model for chart based on selection
                best_model = forecast_result.get('best_model', None)
                bt_model = forecast_result.get('backtesting', {}).get('best_model')
                std_wape = forecast_result.get('wape_scores', {}).get(best_model, float('inf'))
                bt_wape = None
                if bt_model:
                    bt_wape = forecast_result.get('backtesting', {}).get('per_model_wape', {}).get(bt_model, float('inf'))
                if selection_mode == 'Backtesting' and bt_model and bt_model in forecast_result.get('forecasts', {}):
                    best_model = bt_model
                
                # Create chart
                if 'forecasts' in forecast_result:
                    from modules.ui_components import create_forecast_visualization
                    chart, selected_model = create_forecast_visualization(
                        product_series, 
                        forecast_result['forecasts'], 
                        forecast_result['quarter_info'], 
                        best_model=best_model,
                        backtesting_results=forecast_result.get('backtesting', {}),
                        wape_scores=forecast_result.get('wape_scores', {})
                    )
                    if chart:
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Add chart legend for backtesting indicators
                        validation_details = forecast_result.get('backtesting', {}).get('validation_details', {}).get(selected_model, {})
                        has_validation = validation_details.get('iterations', 0) > 0
                        has_periods = len(validation_details.get('validation_periods', [])) > 0
                        
                        if has_validation:
                            if has_periods:
                                st.caption("📊 **Chart Legend:** Blue line = Historical data | Orange dashed = Forecast | 🔴 Red circles = Detected spikes | 🔺 Green triangles = Validation start points | Purple dotted = Backtesting predictions")
                            else:
                                st.caption("📊 **Chart Legend:** Blue line = Historical data | Orange dashed = Forecast | 🔴 Red circles = Detected spikes | 🔺 Green triangles = Validation points (recent periods)")
                # Rationale for selection
                if 'model_evaluation' in forecast_result:
                    standard_model = forecast_result.get('best_model')
                    wape_scores = forecast_result.get('wape_scores', {})
                    std_wape = wape_scores.get(standard_model, float('inf'))
                    bt = forecast_result.get('backtesting', {})
                    bt_model = bt.get('best_model')
                    bt_wape = None
                    if bt_model:
                        bt_wape = bt.get('per_model_wape', {}).get(bt_model, None)
                    chosen_model = standard_model
                    mode_used = 'Standard'
                    if selection_mode == 'Backtesting' and bt_model:
                        chosen_model = bt_model
                        mode_used = 'Backtesting'
                    chosen_total = forecast_result['forecasts'].get(chosen_model, {}).get('quarter_total', forecast_result['summary']['quarter_total'])
                    alt_total = forecast_result['forecasts'].get(standard_model if chosen_model != standard_model else (bt_model or standard_model), {}).get('quarter_total', chosen_total)
                    delta_total = chosen_total - alt_total
                    # Compose rationale text in blue info box
                    rationale = f"Using {mode_used}: {chosen_model}. "
                    details = []
                    if std_wape != float('inf'):
                        details.append(f"Standard WAPE: {std_wape:.1%}")
                    if bt_wape is not None and bt_wape != float('inf'):
                        details.append(f"Backtesting WAPE: {bt_wape:.1%}")
                    if details:
                        rationale += " | ".join(details)
                    impact = f"Impact vs alternate: {('+' if delta_total>=0 else '')}${delta_total:,.0f} on quarter total"
                    st.info(f"{rationale}\n\n{impact}")
                
                # Add backtesting breakdown dropdown
                if forecast_result.get('backtesting', {}).get('validation_details'):
                    with st.expander("🔄 Backtesting Validation Breakdown", expanded=False):
                        display_backtesting_details(forecast_result['backtesting'])
        
        # ============ DOWNLOAD SECTION ============
        st.markdown("### 📥 Export Results")
        
        col_download, col_info = st.columns([1, 2])
        
        with col_download:
            excel_data = None
            if st.button("📊 Generate Excel Report", use_container_width=True):
                # Get adjustment factors from session state and sidebar
                conservatism_factor = forecast_conservatism / 100.0
                capacity_factor = st.session_state.get('capacity_adjustment', 1.0) if st.session_state.get('apply_capacity_adjustment', False) else 1.0
                excel_data = create_excel_download(
                    forecasts_data,
                    analysis_date,
                    "quarterly_outlook_forecast.xlsx",
                    conservatism_factor=conservatism_factor,
                    capacity_factor=capacity_factor
                )
            if excel_data:
                st.download_button(
                    f"💾 Download Report — {selection_mode}",
                    data=excel_data,
                    file_name=f"quarterly_outlook_forecast_{analysis_date.strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            elif excel_data is not None:
                st.error("❌ Could not generate Excel report.")
        
        with col_info:
            st.info("📋 **Excel report includes:** Summary, Monthly Breakdown, Individual Models, and Daily Forecasts with dates and actuals")
        
        # ============ COLLAPSIBLE ADVANCED SECTIONS ============
        
        # Key Insights (Collapsible)
        with st.expander("💡 Key Insights & Model Selection", expanded=False):
            st.success(f"📊 Analyzing **{len(forecasts_data)} products** with best-performing models")
            
            # Show best models for each product
            model_info = []
            for product, data in forecasts_data.items():
                best_model = data['forecast'].get('best_model', 'Unknown')
                model_info.append(f"**{product}**: {best_model}")
            
            if model_info:
                st.info(f"🎯 **Best Models Selected:**\n" + "\n".join([f"• {info}" for info in model_info]))
            
            # Show applied adjustments
            adjustments = []
            if forecast_conservatism != 100:
                adj_desc = "more conservative" if forecast_conservatism < 100 else "more optimistic"
                adjustments.append(f"**{forecast_conservatism}%** forecast adjustment ({adj_desc})")
            
            if st.session_state.get('apply_capacity_adjustment', False):
                capacity_factor = st.session_state.get('capacity_adjustment', 1.0)
                reduction_pct = (1 - capacity_factor) * 100
                adjustments.append(f"**{reduction_pct:.1f}%** capacity constraint reduction")
            
            if adjustments:
                st.warning(f"⚙️ **Applied Adjustments:** {', '.join(adjustments)}")
        
        # Detailed Product Analysis (Collapsible)
        with st.expander("🔍 Detailed Product Analysis", expanded=False):
            for product, data in forecasts_data.items():
                forecast_result = data['forecast'].copy()
                product_series = data['data']
                
                st.markdown(f"#### 📦 {product}")
                
                # Apply forecast conservatism adjustment first
                conservatism_factor = forecast_conservatism / 100.0
                if conservatism_factor != 1.0:
                    forecast_result = apply_conservatism_adjustment_to_forecast(forecast_result, conservatism_factor)
                
                # Apply capacity adjustment if requested (after conservatism)
                original_result = forecast_result.copy()  # Capture after conservatism adjustment
                if st.session_state.get('apply_capacity_adjustment', False):
                    capacity_factor = st.session_state.get('capacity_adjustment', 1.0)
                    forecast_result = apply_capacity_adjustment_to_forecast(forecast_result, capacity_factor)
                    
                    # Show capacity impact
                    display_capacity_adjustment_impact(original_result, forecast_result)
                
                # Progress indicator
                if 'quarter_progress' in forecast_result:
                    create_progress_indicator(forecast_result['quarter_progress'])
                
                # Forecast summary table
                if 'forecasts' in forecast_result:
                    st.subheader("📈 Enhanced Forecast Models Summary")
                    summary_table = create_forecast_summary_table(
                        forecast_result['forecasts'], 
                        forecast_result.get('model_evaluation', {}),
                        forecast_result.get('wape_scores', {}),
                        forecast_result.get('backtesting', {})
                    )
                    st.dataframe(summary_table, use_container_width=True)
                    
                    # Enhanced model comparison with WAPE and backtesting
                    if 'model_evaluation' in forecast_result:
                        display_model_comparison(
                            forecast_result['forecasts'], 
                            forecast_result['model_evaluation'],
                            forecast_result.get('wape_scores', {}),
                            forecast_result.get('backtesting', {})
                        )
                    
                    # Display backtesting details if available
                    if forecast_result.get('backtesting', {}).get('validation_details'):
                        display_backtesting_details(forecast_result['backtesting'])
                
                # Spike analysis if available
                renewal_forecast = forecast_result.get('forecasts', {}).get('Monthly Renewals')
                if renewal_forecast and 'spike_analysis' in renewal_forecast:
                    display_spike_analysis(renewal_forecast['spike_analysis'])
                
                st.markdown("---")
        
    else:
        st.info("👈 **Please upload data first** in the 'Data Upload' tab to see forecast results.")
        
        # Show example of what results will look like
        st.markdown("### 📋 What You'll See Here")
        st.markdown("""
        After uploading your daily business data, this tab will show:
        
        **📊 Interactive Visualizations**
        - Daily data trends with forecast projection
        - Progress indicators for quarter completion
        - Model performance comparisons
        
        **📈 Multiple Forecast Models**
        - Linear trend analysis
        - Moving averages
        - Prophet time series (if available)
        - LightGBM machine learning (if available)
        - XGBoost machine learning (if available)
        - ARIMA time series (if available)
        - Monthly renewal detection
        
        **💡 Business Insights**
        - Best performing forecast model
        - Quarter completion percentage
        - Daily run rates and projections
        - Confidence intervals for uncertainty
        
        **⚙️ Advanced Features**
        - Capacity constraint adjustments
        - Monthly subscription renewal detection
        - Excel export of all results
        """)
