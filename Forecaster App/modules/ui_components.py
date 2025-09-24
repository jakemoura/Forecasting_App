"""
Streamlit UI components for forecast display and results visualization.

Contains functions for displaying forecast results, charts, and interactive
components in the Streamlit interface.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from pathlib import Path
import streamlit as st
import altair as alt
from datetime import datetime
from pathlib import Path


def fy(ts):
    """Convert timestamp to fiscal year string (Jul-Jun)."""
    return f"FY{ts.year + (1 if ts.month >= 7 else 0)}"


def fiscal_period_display(date, fiscal_year_start_month=7):
    """
    Convert date to fiscal period display format like 'P01 - July (FY2025)'.
    Uses zero-padded periods (P01, P02, ..., P12) for proper Excel sorting.
    
    Args:
        date: pandas Timestamp or datetime
        fiscal_year_start_month: Month that starts the fiscal year (default: 7 for July)
    
    Returns:
        String in format 'P01 - July (FY2025)'
    """
    if pd.isna(date):
        return ""
    
    date = pd.Timestamp(date)
    
    # Calculate fiscal year and period
    if date.month >= fiscal_year_start_month:
        # Current calendar year fiscal year
        fiscal_period = date.month - fiscal_year_start_month + 1
        fiscal_year = date.year + 1  # FY starts in current year, named for next year
    else:
        # Next calendar year fiscal year (months 1-6 for July start)
        fiscal_period = date.month + (12 - fiscal_year_start_month) + 1
        fiscal_year = date.year  # FY started in previous year, named for current year
    
    # Get month name
    month_name = date.strftime('%B')
    
    # Use zero-padded period number for proper Excel sorting (P01, P02, ..., P12)
    return f"P{fiscal_period:02d} - {month_name} (FY{fiscal_year})"


def display_advanced_validation_settings():
    """
    Display advanced validation settings in the sidebar using a simple master toggle
    with optional customization, collapsed by default to keep the UI clean.

    Returns:
        Tuple of (enable_advanced_validation, enable_walk_forward, enable_cross_validation)
    """
    with st.sidebar.expander("🔍 Advanced (optional)", expanded=False):
        master_toggle = st.checkbox(
            "Improve accuracy (slower)",
            value=False,
            help="Runs deeper validation and compares a few proven models; takes longer but can improve accuracy."
        )

        # Default behavior: when master is on, enable all advanced methods.
        enable_advanced_validation = bool(master_toggle)
        enable_walk_forward = bool(master_toggle)
        enable_cross_validation = bool(master_toggle)

        # Optional fine-tuning for power users without cluttering the default UI
        customize = st.checkbox(
            "Customize methods",
            value=False,
            disabled=not master_toggle,
            help="Optionally choose which validation methods to run."
        )

        if customize and master_toggle:
            col1, col2, col3 = st.columns(3)
            with col1:
                enable_advanced_validation = st.checkbox(
                    "Enhanced MAPE",
                    value=True,
                    help="Adds confidence intervals, bias, and seasonal patterns."
                )
            with col2:
                enable_walk_forward = st.checkbox(
                    "Walk-forward",
                    value=True,
                    help="Rolling windows validation for real-world behavior."
                )
            with col3:
                enable_cross_validation = st.checkbox(
                    "Cross-validation",
                    value=True,
                    help="Multiple splits to assess stability across time."
                )
            # (Removed stray misplaced UI elements)

    return enable_advanced_validation, enable_walk_forward, enable_cross_validation


def display_enhanced_mape_analysis(backtesting_results):
    """Clean enhanced WAPE summary (corruption removed)."""
    st.markdown("### Enhanced WAPE Breakdown")
    enhanced_rows = []
    for product, models in (backtesting_results or {}).items():
        for model, res in (models or {}).items():
            if isinstance(res, dict):
                enh = res.get('enhanced_analysis')
                if enh:
                    enhanced_rows.append({
                        'Product': product,
                        'Model': model,
                        'WAPE': f"{enh.get('mape', 0):.1%}",
                        'WAPE_Std': f"{enh.get('mape_std', 0):.1%}",
                        'CI_Lower': f"{enh.get('mape_ci_lower', 0):.1%}",
                        'CI_Upper': f"{enh.get('mape_ci_upper', 0):.1%}",
                        'Bias': f"{enh.get('bias', 0):+.1%}",
                        'Outliers': enh.get('outlier_count', 0),
                        'Worst_Month': f"{enh.get('worst_month_error', 0):.1%}",
                        'Best_Month': f"{enh.get('best_month_error', 0):.1%}"
                    })
    if not enhanced_rows:
        st.info("No enhanced analysis available.")
        return
    df = pd.DataFrame(enhanced_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    try:
        biases = [float(r['Bias'].strip('%+')) for r in enhanced_rows]
        avg_bias = np.mean(biases)
        if abs(avg_bias) < 2:
            st.success(f"Low aggregate bias ({avg_bias:+.1f}%)")
        else:
            st.warning(f"Noticeable bias ({avg_bias:+.1f}%)")
    except Exception:
        pass


def display_cross_validation_results(backtesting_results):
    """Display cross-validation results."""
    st.markdown("### Time Series Cross-Validation Results")
    st.markdown("Model stability across different time periods.")
    
    cv_results = []
    
    for product, models in backtesting_results.items():
        for model, results in models.items():
            if 'cross_validation' in results and results['cross_validation']:
                cv = results['cross_validation']
                cv_results.append({
                    'Product': product,
                    'Model': model,
                    'Folds': cv.get('folds_completed', 0),
                    'Mean_MAPE': f"{cv.get('mean_mape', 0):.1%}",
                    'Std_MAPE': f"{cv.get('std_mape', 0):.1%}",
                    'Stability': "Stable" if cv.get('std_mape', 1) < 0.05 else "Moderate" if cv.get('std_mape', 1) < 0.1 else "Unstable"
                })
    
    if cv_results:
        df = pd.DataFrame(cv_results)
        st.dataframe(
            df,
            column_config={
                'Folds': st.column_config.NumberColumn('Folds', help='Number of cross-validation folds completed'),
                'Mean_MAPE': st.column_config.TextColumn('Mean WAPE', help='Average WAPE across all folds'),
                'Std_MAPE': st.column_config.TextColumn('WAPE Std', help='Standard deviation showing stability'),
                'Stability': st.column_config.TextColumn('Stability', help='Stable/Moderate/Unstable based on variance')
            },
            use_container_width=True
        )
        
        # Display insights
        stable_models = len([r for r in cv_results if r['Stability'] == 'Stable'])
        total_models = len(cv_results)
        
        if stable_models / total_models > 0.7:
            st.success(f"✅ {stable_models}/{total_models} models show stable performance across time periods")
        else:
            st.warning(f"⚠️ Only {stable_models}/{total_models} models show stable performance - results may vary")
    else:
        st.info("No cross-validation results available.")


def display_seasonal_performance_analysis(backtesting_results):
    """Display seasonal performance analysis."""
    st.markdown("### Seasonal Performance Analysis")
    st.markdown("Model accuracy patterns by month and quarter.")
    
    seasonal_results = []
    
    for product, models in backtesting_results.items():
        for model, results in models.items():
            if 'seasonal_analysis' in results and results['seasonal_analysis']:
                seasonal = results['seasonal_analysis']
                if 'error' not in seasonal:
                    seasonal_results.append({
                        'Product': product,
                        'Model': model,
                        'Best_Month': seasonal.get('best_month_name', 'N/A'),
                        'Best_Month_WAPE': f"{seasonal.get('best_month_mape', 0):.1%}",
                        'Worst_Month': seasonal.get('worst_month_name', 'N/A'),
                        'Worst_Month_WAPE': f"{seasonal.get('worst_month_mape', 0):.1%}",
                        'Best_Quarter': seasonal.get('best_quarter_name', 'N/A'),
                        'Worst_Quarter': seasonal.get('worst_quarter_name', 'N/A'),
                        'Total_Periods': seasonal.get('total_periods', 0)
                    })
    
    if seasonal_results:
        df = pd.DataFrame(seasonal_results)
        st.dataframe(
            df,
            column_config={
                'Best_Month': st.column_config.TextColumn('Best Month', help='Month with lowest average WAPE'),
                'Best_Month_WAPE': st.column_config.TextColumn('Best WAPE', help='WAPE for best performing month'),
                'Worst_Month': st.column_config.TextColumn('Worst Month', help='Month with highest average MAPE'),
                'Worst_Month_WAPE': st.column_config.TextColumn('Worst WAPE', help='WAPE for worst performing month'),
                'Best_Quarter': st.column_config.TextColumn('Best Quarter', help='Quarter with lowest average WAPE'),
                'Worst_Quarter': st.column_config.TextColumn('Worst Quarter', help='Quarter with highest average WAPE'),
                'Total_Periods': st.column_config.NumberColumn('Periods', help='Total periods analyzed')
            },
            use_container_width=True
        )
        
        # Display insights
        st.markdown("**📅 Seasonal Insights:**")
        
        # Most common best/worst months
        best_months = [r['Best_Month'] for r in seasonal_results if r['Best_Month'] != 'N/A']
        worst_months = [r['Worst_Month'] for r in seasonal_results if r['Worst_Month'] != 'N/A']
        
        if best_months:
            best_month_counts = pd.Series(best_months).value_counts()
            most_accurate_month = best_month_counts.index[0]
            st.success(f"✅ Most accurate forecasting month: **{most_accurate_month}** ({best_month_counts.iloc[0]} products)")
        
        if worst_months:
            worst_month_counts = pd.Series(worst_months).value_counts()
            least_accurate_month = worst_month_counts.index[0]
            st.warning(f"⚠️ Most challenging forecasting month: **{least_accurate_month}** ({worst_month_counts.iloc[0]} products)")
            st.info("💡 Consider additional data sources or model adjustments for challenging months")
    else:
        st.info("No seasonal performance analysis results available.")


def display_data_context_summary():
    """Display enhanced data context summary with backtesting recommendations."""
    data_context = st.session_state.get('data_context', {})
    if not data_context:
        return
    
    # Display data context summary prominently (no expander needed)
    st.markdown("#### 📊 **Data Analysis Summary**")
    recommendations = data_context.get('backtesting_recommendations', {})
    data_quality = data_context.get('data_quality_score', {})
    
    if recommendations:
        # Display data quality status
        col1, col2 = st.columns([2, 1])
        
        with col1:
            status_icon = recommendations.get('icon', '📊')
            status_title = recommendations.get('title', 'Data Analysis')
            status_desc = recommendations.get('description', 'Analyzing data...')
            
            if recommendations.get('status') == 'limited':
                st.warning(f"{status_icon} **{status_title}**: {status_desc}")
            elif recommendations.get('status') == 'moderate':
                st.info(f"{status_icon} **{status_title}**: {status_desc}")
            else:
                st.success(f"{status_icon} **{status_title}**: {status_desc}")
            
            # Show recommendation message
            st.markdown(f"💡 **Recommendation**: {recommendations.get('message', 'Use data-driven backtesting')}")
            
            # Show recommended backtesting range
            recommended_range = recommendations.get('recommended_range', '6-24 months')
            default_value = recommendations.get('default_value', 12)
            st.info(f"🎯 **Recommended Backtesting**: {recommended_range} (default: {default_value} months)")
        
        with col2:
            # Show data quality metrics
            if data_quality and 'score' in data_quality:
                score = data_quality['score']
                grade = data_quality.get('grade', 'N/A')
                
                st.metric("Data Quality", f"{score}/100", f"Grade: {grade}")
                
                # Show consistency info
                consistency = data_quality.get('consistency_ratio', 0)
                if consistency > 0.8:
                    st.success("✅ High Consistency")
                elif consistency > 0.6:
                    st.info("📊 Good Consistency")
                else:
                    st.warning("⚠️ Variable Consistency")
    
    # Show data summary table
    st.markdown("#### 📈 **Data Summary**")
    
    summary_data = {
        'Metric': [
            'Total Products',
            'Available Months (Min)',
            'Available Months (Max)',
            'Available Months (Avg)',
            'Data Quality Score',
            'Consistency Ratio'
        ],
        'Value': [
            str(data_context.get('total_products', 0)),
            str(data_context.get('min_months', 0)),
            str(data_context.get('max_months', 0)),
            str(data_context.get('avg_months', 0)),
            f"{data_quality.get('score', 'N/A')}/100" if data_quality else 'N/A',
            f"{data_quality.get('consistency_ratio', 0):.1%}" if data_quality else 'N/A'
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, hide_index=True, use_container_width=True)
    
    # Show backtesting strategy
    if recommendations:
        st.markdown("#### 🎯 **Backtesting Strategy**")
        
        # Show the actual calculated recommendation
        status = recommendations.get('status', 'unknown')
        icon = recommendations.get('icon', '📊')
        title = recommendations.get('title', 'Data Analysis')
        description = recommendations.get('description', '')
        message = recommendations.get('message', '')
        
        # Highlight the current recommendation
        st.success(f"{icon} **{title}**: {description}")
        st.info(f"💡 **Strategy**: {message}")
        
        # Show the calculated backtesting range
        recommended_range = recommendations.get('recommended_range', '6-24 months')
        default_value = recommendations.get('default_value', 12)
        min_value = recommendations.get('min_value', 6)
        max_value = recommendations.get('max_value', 24)
        
        st.markdown(f"""
        **🎯 Calculated Backtesting Range:**
        - **Recommended**: {recommended_range}
        - **Default Value**: {default_value} months
        - **Slider Range**: {min_value} to {max_value} months
        """)
        
        # Add refresh button after analysis results
        st.markdown("---")
        if st.button("🔄 **Refresh Sidebar**", key="refresh_sidebar_after_analysis", help="Click to update the sidebar with new recommendations"):
            st.rerun()
        


def display_backtesting_results(backtesting_results):
    """Display comprehensive backtesting validation results with explanations."""
    if not backtesting_results:
        return
        # Explain what backtesting is and how it works
        st.markdown("#### 📚 **What is Backtesting?**")
        st.info("""
        **Backtesting** validates forecasting models by testing them on historical data they haven't seen during training. 
        This gives you confidence that the models will perform well on future data.
        
        **How it works:**
        1. **Training Period**: Models learn from older data
        2. **Validation Period**: Models predict on recent historical data
        3. **Performance Measurement**: Compare predictions to actual values using WAPE
        4. **Model Selection**: Choose the best performing model per product
        """)
        
        # Show backtesting configuration
        st.markdown("#### ⚙️ **Backtesting Configuration**")
        config_info = {
            "Validation Method": "Simple Train/Test Split",
            "Training Data": "Historical data excluding last X months",
            "Test Period": "Last X months (as specified in sidebar)",
            "Performance Metric": "WAPE (Weighted Absolute Percentage Error)",
            "Selection Strategy": "Best model per product based on backtesting WAPE"
        }
        
        for key, value in config_info.items():
            st.caption(f"**{key}**: {value}")
        
        # Summary table: backtesting MAPE per model/product
        st.markdown("#### 📊 **Backtesting Performance Results (WAPE)**")
        
        rows = []
        total_models = 0
        successful_backtests = 0
        
        for product, models in backtesting_results.items():
            for model, res in models.items():
                if not isinstance(res, dict):
                    continue
                
                total_models += 1
                
                # Get backtesting results
                backtesting = res.get('backtesting_validation')
                if backtesting and isinstance(backtesting, dict) and backtesting.get('success'):
                    successful_backtests += 1
                    mape = backtesting.get('mape', 0)
                    backtest_months = backtesting.get('backtest_period', 0)
                    test_months = backtesting.get('test_months', 0)
                    
                    rows.append({
                        'Product': product, 
                        'Model': model, 
                        'Backtest MAPE%': round(float(mape)*100, 2),
                        'Backtest Period': f"{backtest_months} months",
                        'Test Period': f"{test_months} months",
                        'Status': '✅ Success'
                    })
                else:
                    # Show failed backtesting attempts
                    rows.append({
                        'Product': product, 
                        'Model': model, 
                        'Backtest MAPE%': 'N/A',
                        'Backtest Period': 'N/A',
                        'Test Period': 'N/A',
                        'Status': '❌ Failed'
                    })
        
        if rows:
            # Sort by Product, then by Status (Success first), then by WAPE
            df = pd.DataFrame(rows)
            df['sort_key'] = df.apply(lambda x: (x['Product'], x['Status'] != '✅ Success', 
                                                float(x.get('Backtest WAPE%', x.get('Backtest MAPE%'))) if x.get('Backtest WAPE%', x.get('Backtest MAPE%')) != 'N/A' else 999), axis=1)
            df = df.sort_values('sort_key').drop('sort_key', axis=1)
            
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            # Performance interpretation
            st.markdown("#### 🎯 **Performance Interpretation**")
            st.caption("""
            **WAPE Guidelines:**
            - **< 10%**: Excellent accuracy
            - **10-20%**: Good accuracy  
            - **20-30%**: Moderate accuracy
            - **> 30%**: Poor accuracy (consider different model)
            """)
            
            # Show backtesting insights
            st.markdown("#### 📈 **Backtesting Insights**")
            
            # Calculate success rate
            success_rate = (successful_backtests / total_models) * 100 if total_models > 0 else 0
            
            if success_rate >= 80:
                st.success(f"✅ **High Success Rate**: {success_rate:.1f}% of models successfully backtested ({successful_backtests}/{total_models})")
            elif success_rate >= 60:
                st.info(f"📊 **Good Success Rate**: {success_rate:.1f}% of models successfully backtested ({successful_backtests}/{total_models})")
            else:
                st.warning(f"⚠️ **Low Success Rate**: {success_rate:.1f}% of models successfully backtested ({successful_backtests}/{total_models})")
            
            # Model selection explanation
            st.markdown("#### 🏆 **Model Selection Process**")
            st.info("""
            **How models are selected:**
            1. **Backtesting Performance**: Models ranked by WAPE on validation data
            2. **Product-Specific Selection**: Best model chosen per product (not overall)
            3. **Fallback Strategy**: If backtesting fails, falls back to basic WAPE rankings
            4. **Hybrid Approach**: Combines backtesting results with WAPE rankings for robustness
            """)
            
            # Show detailed results if user wants
            if st.checkbox("Show detailed backtesting analysis", value=False):
                tab_labels = []
                sections = []
                
                # Enhanced MAPE analysis
                tab_labels.append("Enhanced MAPE")
                sections.append('enhanced')
                
                # Seasonal analysis
                if any('seasonal_analysis' in (res or {}) for models in backtesting_results.values() for res in models.values() if isinstance(res, dict)):
                    tab_labels.append("Seasonal")
                    sections.append('seasonal')
                
                tabs = st.tabs(tab_labels)
                t_index = 0
                
                if 'enhanced' in sections:
                    with tabs[t_index]:
                        display_enhanced_mape_analysis(backtesting_results)
                    t_index += 1
                
                if 'seasonal' in sections:
                    with tabs[t_index]:
                        display_seasonal_performance_analysis(backtesting_results)
                    t_index += 1
        else:
            st.warning("⚠️ **No successful backtesting results available**")
            st.info("""
            **Possible reasons:**
            - Insufficient historical data for validation
            - Models failed to converge during backtesting
            - Data quality issues prevented validation
            
            **What happens next:**
            The system automatically falls back to MAPE rankings from basic validation to ensure you still get model recommendations.
            """)
            
            # Show fallback strategy
            st.markdown("#### 🔄 **Fallback Strategy**")
            st.success("""
            **Automatic Fallback Activated:**
            When backtesting fails, the system uses MAPE rankings from basic model validation. 
            This ensures you always get reliable model recommendations, even with limited data.
            """)


def display_product_forecast(data, product, model_name, best_models_per_product=None, best_mapes_per_product=None):
    """
    Display forecast visualization for a specific product.
    
    Args:
        data: DataFrame with forecast data
        product: Product name to display
        model_name: Model name being displayed
        best_models_per_product: Dict of best model per product
        best_mapes_per_product: Dict of best MAPE per product
    """
    # Filter data for this product
    product_data = data[data["Product"] == product].copy()
    # Robust coercion to avoid silent Altair drops
    if not product_data.empty:
        product_data['Date'] = pd.to_datetime(product_data['Date'], errors='coerce')
        product_data['ACR'] = pd.to_numeric(product_data['ACR'], errors='coerce')
        product_data = product_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Date','ACR'])
    
    if product_data.empty:
        st.warning(f"No data available for product: {product}")
        # Diagnostic dump for investigation
        st.write("Raw head:")
        try:
            st.dataframe(data[data["Product"]==product].head(5))
        except Exception:
            pass
        return
    
    # Sort by date to ensure proper order
    product_data = product_data.sort_values("Date")
    
    # Display model info for this product
    col1, col2 = st.columns([3, 1])
    # Track the actual per-product model used (for backtest overlay and chart title)
    actual_model = model_name
    with col1:
        # Prefer explicit BestModel column if present (hybrid views like Standard / Backtesting / Raw)
        if 'BestModel' in product_data.columns and not product_data['BestModel'].isna().all():
            actual_model = str(product_data['BestModel'].iloc[0])
            # Attach selection rationale if available
            reason = None
            try:
                reasons = st.session_state.get('best_model_reasons_backtesting', {}) or {}
                reason = reasons.get(product)
            except Exception:
                reason = None
            if reason:
                st.caption(f"📊 Using **{actual_model}** model for this product — {reason}")
            else:
                st.caption(f"📊 Using **{actual_model}** model for this product")
        elif model_name == "Best per Product" and best_models_per_product and product in best_models_per_product:
            actual_model = best_models_per_product[product]
            st.caption(f"📊 Using **{actual_model}** model for this product")
        else:
            st.caption(f"📊 Using **{model_name}** model")
    
    with col2:
        # Prefer per‑model per‑product WAPE from session, fallback to best_mapes_per_product
        wape_pct = None
        try:
            product_mapes = st.session_state.get('product_mapes', {}) or {}
            if model_name in product_mapes and product in product_mapes[model_name]:
                wape_pct = float(product_mapes[model_name][product]) * 100.0
        except Exception:
            wape_pct = None
        if wape_pct is None and best_mapes_per_product and product in best_mapes_per_product:
            try:
                wape_pct = float(best_mapes_per_product[product]) * 100.0
            except Exception:
                wape_pct = None
        if wape_pct is not None and np.isfinite(wape_pct):
            if wape_pct <= 15:
                st.success(f"🎯 {wape_pct:.1f}% WAPE")
            elif wape_pct <= 25:
                st.info(f"📈 {wape_pct:.1f}% WAPE")
            else:
                st.warning(f"⚠️ {wape_pct:.1f}% WAPE")
    
    # Create the main forecast chart with optional backtesting overlay
    try:
        # Prepare backtesting data using the winning per‑product model
        backtesting_results = st.session_state.get('backtesting_results', {})
        # default overlay: the selected/winning model for this product
        overlay_model = actual_model
        backtesting_chart_data = prepare_backtesting_chart_data(backtesting_results, product, overlay_model)
        
        # Debug: show head and unique type values for this product slice
        with st.expander("🔎 Debug: product data slice (first 10 rows)", expanded=False):
            try:
                st.write({
                    'rows': len(product_data),
                    'date_dtype': str(product_data['Date'].dtype) if 'Date' in product_data.columns else 'missing',
                    'acr_dtype': str(product_data['ACR'].dtype) if 'ACR' in product_data.columns else 'missing',
                    'type_unique': product_data['Type'].astype(str).unique().tolist() if 'Type' in product_data.columns else ['missing']
                })
                st.dataframe(product_data.head(10), use_container_width=True)
            except Exception:
                pass

        # Create chart with backtesting overlay
        chart = create_forecast_chart(product_data, product, actual_model, backtesting_chart_data)
        st.altair_chart(chart, use_container_width=True)
        
        # Show legend for chart elements
        legend_text = """
            **📊 Chart Legend:**
            - 🔵 **Blue Line**: Historical actual data
            - 🟠 **Orange Dashed**: Future forecast
"""
        # Add non-compliant legend entries if they exist
        has_noncompliant_historical = any(product_data.get("Type", "").astype(str).str.contains("non-compliant", case=False, na=False))
        has_noncompliant_forecast = any(product_data.get("Type", "").astype(str).str.contains("non-compliant-forecast", case=False, na=False))
        
        if has_noncompliant_historical:
            legend_text += "            - � **Red Dotted**: Historical non-compliant rev rec renewals\n"
        if has_noncompliant_forecast:
            legend_text += "            - � **Dark Red Dashed**: Future non-compliant rev rec renewals\n"
            
        if backtesting_chart_data is not None and not backtesting_chart_data.empty:
            legend_text += """            - 🟢 **Green Dashed**: Backtesting predictions (what the model predicted during validation)
            - 🟢 **Green Solid**: Backtesting actuals (real values during validation period)
"""
        
        st.info(legend_text)
            
    except Exception as e:
        st.error(f"❌ Chart generation failed: {str(e)}")
        # Fallback: show data table
        st.dataframe(product_data[["Date", "ACR", "Type"]].head(10))
    
    # (Removed per-product Standard vs Backtesting rationale)
    
    # Show forecast summary stats
    forecast_data = product_data[product_data["Type"] == "forecast"]
    if not forecast_data.empty:
        total_forecast = forecast_data["ACR"].sum()
        avg_monthly = forecast_data["ACR"].mean()
        
        # Calculate fiscal year forecast for this product
        # Get fiscal year start month from config, session state, or default to July
        fiscal_year_start_month = st.session_state.get('fiscal_year_start_month', 7)
        if 'config' in st.session_state and st.session_state.config:
            fiscal_year_start_month = int(st.session_state.config.get('fiscal_year_start_month', fiscal_year_start_month))
        
        # Calculate current fiscal year dates with auto-flip logic
        now = datetime.now()
        if now.month >= fiscal_year_start_month:
            calendar_fy_year = now.year + 1
        else:
            calendar_fy_year = now.year
        
        # Check if current calendar FY has complete data (all 12 months)
        calendar_fy_start_date = pd.Timestamp(year=calendar_fy_year - 1, month=fiscal_year_start_month, day=1)
        calendar_fy_end_date = pd.Timestamp(year=calendar_fy_year, month=fiscal_year_start_month, day=1) - pd.DateOffset(days=1)
        
        calendar_fy_product_data = product_data[
            (pd.to_datetime(product_data['Date']) >= calendar_fy_start_date) & 
            (pd.to_datetime(product_data['Date']) <= calendar_fy_end_date)
        ].copy()
        
        # Check months elapsed in calendar FY
        calendar_months_elapsed = 0
        if not calendar_fy_product_data.empty:
            calendar_actual_rows = calendar_fy_product_data[calendar_fy_product_data['Type'].isin(['actual', 'history', 'historical'])].copy()
            if not calendar_actual_rows.empty:
                calendar_actual_rows['Date'] = pd.to_datetime(calendar_actual_rows['Date'])
                calendar_actual_months = calendar_actual_rows['Date'].dt.to_period('M').unique()
                calendar_months_elapsed = int(len(calendar_actual_months))
        calendar_months_elapsed = min(12, max(0, calendar_months_elapsed))
        
        # If current FY has 0 months remaining (all 12 months have actual data), flip to next FY
        if calendar_months_elapsed >= 12:
            current_fy_year = calendar_fy_year + 1
            fy_start_date = pd.Timestamp(year=current_fy_year - 1, month=fiscal_year_start_month, day=1)
            fy_end_date = pd.Timestamp(year=current_fy_year, month=fiscal_year_start_month, day=1) - pd.DateOffset(days=1)
        else:
            current_fy_year = calendar_fy_year
            fy_start_date = calendar_fy_start_date
            fy_end_date = calendar_fy_end_date
        
        # Filter product data for target fiscal year
        fy_product_data = product_data[
            (pd.to_datetime(product_data['Date']) >= fy_start_date) & 
            (pd.to_datetime(product_data['Date']) <= fy_end_date)
        ].copy()
        
        fy_total = 0
        if not fy_product_data.empty:
            fy_actuals = fy_product_data[fy_product_data['Type'].isin(['actual', 'history', 'historical'])]['ACR'].sum()
            fy_forecast = fy_product_data[fy_product_data['Type'] == 'forecast']['ACR'].sum()
            fy_noncompliant = fy_product_data[fy_product_data['Type'].isin(['non-compliant', 'non-compliant-forecast'])]['ACR'].sum()
            fy_total = fy_actuals + fy_forecast + fy_noncompliant
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Total Forecast", f"${total_forecast/1e6:.1f}M")
        with col2:
            st.metric("📈 Avg Monthly", f"${avg_monthly/1e6:.1f}M")
        with col3:
            months_count = len(forecast_data)
            st.metric("📅 Months", f"{months_count}")
        with col4:
            if fy_total > 0:
                st.metric(f"🗓️ FY{current_fy_year}", f"${fy_total/1e6:.1f}M", 
                         help=f"Complete fiscal year projection (actuals + forecast)")
            else:
                st.metric(f"🗓️ FY{current_fy_year}", "No Data", 
                         help="No data available for current fiscal year")


def prepare_backtesting_chart_data(backtesting_results, product_name, model_name):
    """
    Prepare backtesting data for chart overlay.
    
    Args:
        backtesting_results: Dictionary containing backtesting results
        product_name: Name of the product
        model_name: Name of the model
    
    Returns:
        DataFrame with backtesting data formatted for charting, or None if no data
    """
    if not backtesting_results or product_name not in backtesting_results:
        return None
    
    product_results = backtesting_results[product_name]
    if not product_results or model_name not in product_results:
        return None
    
    model_result = product_results[model_name]
    if not model_result or not isinstance(model_result, dict):
        return None
    
    backtesting_validation = model_result.get('backtesting_validation')
    if not backtesting_validation or not isinstance(backtesting_validation, dict):
        return None
    
    # If enhanced rolling provided per‑fold series, assemble all folds; else show most recent only
    chart_rows = []
    folds = backtesting_validation.get('validation_results') or []
    if isinstance(folds, list) and len(folds) > 0 and isinstance(folds[0], dict) and 'y_true' in folds[0] and 'y_pred' in folds[0] and 'val_dates' in folds[0]:
        for f in folds:
            fold_id = f.get('fold')
            dates = f.get('val_dates') or []
            y_true = f.get('y_true') or []
            y_pred = f.get('y_pred') or []
            for d, yt in zip(dates, y_true):
                chart_rows.append({'Date': pd.to_datetime(d), 'ACR': float(yt), 'Type': 'backtest-actual', 'Fold': int(fold_id)})
            for d, yp in zip(dates, y_pred):
                chart_rows.append({'Date': pd.to_datetime(d), 'ACR': float(yp), 'Type': 'backtest-prediction', 'Fold': int(fold_id)})
    else:
        # Fallback to single most-recent fold data
        train_data = backtesting_validation.get('train_data')
        test_data = backtesting_validation.get('test_data')
        predictions = backtesting_validation.get('predictions')
        if train_data is None or test_data is None or predictions is None:
            return None
        if len(test_data) == len(predictions):
            for date, pred in zip(test_data.index, predictions):
                chart_rows.append({'Date': date, 'ACR': pred, 'Type': 'backtest-prediction', 'Fold': 1})
        for date, actual in test_data.items():
            chart_rows.append({'Date': date, 'ACR': actual, 'Type': 'backtest-actual', 'Fold': 1})
    if not chart_rows:
        return None
    return pd.DataFrame(chart_rows)


def create_forecast_chart(data, product_name, model_name, backtesting_data=None):
    """
    Create an Altair chart for forecast visualization with optional backtesting overlay.
    
    Args:
        data: DataFrame with forecast data
        product_name: Name of the product
        model_name: Name of the model
        backtesting_data: Optional DataFrame with backtesting results
    
    Returns:
        Altair chart object
    """
    # Prepare data for charting (coerce types and drop invalid rows)
    chart_data = data.copy()
    # Coerce Date to datetime and drop rows that fail conversion
    chart_data['Date'] = pd.to_datetime(chart_data['Date'], errors='coerce')
    chart_data = chart_data.dropna(subset=['Date'])
    # Ensure numeric ACR and drop NaNs/Infs which can blank the chart
    chart_data['ACR'] = pd.to_numeric(chart_data['ACR'], errors='coerce')
    chart_data = chart_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['ACR'])
    # Ensure Type exists; if missing, infer historical vs forecast by Date cutoff
    if 'Type' not in chart_data.columns or chart_data['Type'].isna().all():
        cutoff = chart_data['Date'].max()
        chart_data['Type'] = np.where(chart_data['Date'] <= cutoff, 'actual', 'forecast')
    
    # Create base chart
    # Normalize/derive robust type flags so filters don't silently drop all rows
    type_norm = None
    if 'Type' in chart_data.columns:
        try:
            type_norm = chart_data['Type'].astype(str).str.strip().str.lower()
        except Exception:
            type_norm = None
    actual_mask = pd.Series(False, index=chart_data.index)
    forecast_mask = pd.Series(False, index=chart_data.index)
    noncomp_mask = pd.Series(False, index=chart_data.index)
    noncomp_historical_mask = pd.Series(False, index=chart_data.index)
    if type_norm is not None:
        actual_mask |= type_norm.isin(['actual','history','historical'])
        noncomp_mask |= type_norm.isin(['non-compliant-forecast','noncompliant-forecast'])
        noncomp_historical_mask |= type_norm.isin(['non-compliant','noncompliant'])
        forecast_mask |= type_norm.isin(['forecast','future','fcst','prediction','predicted'])
    if not actual_mask.any() and not forecast_mask.any():
        # As a last resort, treat all rows as actual to avoid a blank chart
        actual_mask = pd.Series(True, index=chart_data.index)
    elif not actual_mask.any() and forecast_mask.any():
        boundary = chart_data.loc[forecast_mask, 'Date'].min()
        actual_mask = chart_data['Date'] < boundary
    elif actual_mask.any() and not forecast_mask.any():
        boundary = chart_data.loc[actual_mask, 'Date'].max()
        forecast_mask = chart_data['Date'] > boundary
    chart_data['is_actual'] = actual_mask
    chart_data['is_forecast'] = forecast_mask
    chart_data['is_noncompliant'] = noncomp_mask
    chart_data['is_noncompliant_historical'] = noncomp_historical_mask

    base = alt.Chart(chart_data)
    
    # Historical data line
    historical = base.transform_filter(
        alt.datum.is_actual
    ).mark_line(
        point=True,
        color='steelblue',
        strokeWidth=3
    ).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('ACR:Q', title='ACR ($)', scale=alt.Scale(zero=False)),
        tooltip=['Date:T', 'ACR:Q', 'Type:N']
    )
    
    # Forecast data line
    forecast = base.transform_filter(
        alt.datum.is_forecast
    ).mark_line(
        point=True,
        color='orange',
        strokeWidth=3,
        strokeDash=[5, 5]
    ).encode(
        x='Date:T',
        y='ACR:Q',
        tooltip=['Date:T', 'ACR:Q', 'Type:N']
    )
    
    # Non-compliant forecast line (future projected renewals)
    noncompliant = base.transform_filter(
        alt.datum.is_noncompliant
    ).mark_line(
        point=True,
        color='darkred',
        strokeWidth=2,
        strokeDash=[3, 3]
    ).encode(
        x='Date:T',
        y='ACR:Q',
        tooltip=['Date:T', 'ACR:Q', 'Type:N']
    )
    
    # Non-compliant historical line (historical renewals)
    noncompliant_historical = base.transform_filter(
        alt.datum.is_noncompliant_historical
    ).mark_line(
        point=True,
        color='red',
        strokeWidth=2,
        strokeDash=[1, 1]
    ).encode(
        x='Date:T',
        y='ACR:Q',
        tooltip=['Date:T', 'ACR:Q', 'Type:N']
    )
    
    # Initialize chart layers
    chart_layers = [historical, forecast, noncompliant, noncompliant_historical]
    
    # Add backtesting overlay if available (use a separate data source)
    if backtesting_data is not None and not backtesting_data.empty:
        base_bt = alt.Chart(backtesting_data)
        # Backtesting predictions (what the model predicted during validation)
        backtest_predictions = base_bt.transform_filter(
            alt.datum.Type == 'backtest-prediction'
        ).mark_line(
            point=True,
            color='green',
            strokeWidth=2,
            strokeDash=[8, 4]
        ).encode(
            x='Date:T',
            y='ACR:Q',
            tooltip=['Date:T', 'ACR:Q', 'Type:N', 'Fold:N'],
            detail='Fold:N'
        )
        # Backtesting actuals (real values during validation period)
        backtest_actuals = base_bt.transform_filter(
            alt.datum.Type == 'backtest-actual'
        ).mark_line(
            point=True,
            color='darkgreen',
            strokeWidth=3
        ).encode(
            x='Date:T',
            y='ACR:Q',
            tooltip=['Date:T', 'ACR:Q', 'Type:N', 'Fold:N'],
            detail='Fold:N'
        )
        chart_layers.extend([backtest_predictions, backtest_actuals])
    
    # If every layer ends up empty, fall back to a single line to avoid a blank panel
    if not (chart_data['is_actual'].any() or chart_data['is_forecast'].any() or chart_data['is_noncompliant'].any() or chart_data['is_noncompliant_historical'].any()):
        fallback = base.mark_line(point=True, color='steelblue', strokeWidth=3).encode(
            x='Date:T', y='ACR:Q', tooltip=['Date:T','ACR:Q']
        )
        chart_layers = [fallback]

    # Combine all layers
    chart = alt.layer(*chart_layers).resolve_scale(
        color='independent'
    ).properties(
        title=f'{product_name} - {model_name} Forecast{" + Backtesting" if backtesting_data is not None and not backtesting_data.empty else ""}',
        height=400,
        width='container'
    )
    
    return chart


def normalized_monthly_by_days(df, value_col='value', base_days=None):
    """
    Normalize monthly values by the number of days in each month.
    
    Args:
        df: DataFrame with DatetimeIndex and value column
        value_col: Name of the value column to normalize
        base_days: Base number of days (default: 30)
    
    Returns:
        Series with normalized values
    """
    df = df.copy()
    # Ensure index is DatetimeIndex before applying period operations
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df['month'] = df.index.to_period('M')

    # (1) sum and (2) count days in each month
    monthly_sum   = df.groupby('month')[value_col].sum()
    monthly_count = df.groupby('month').size()

    # (3) average per day
    avg_per_day = monthly_sum / monthly_count

    # (optional) scale back to a fixed-length month
    if base_days is not None:
        norm = avg_per_day * base_days
    else:
        norm = avg_per_day    # for plotting
    norm.index = norm.index.to_timestamp()
    return norm


def create_download_excel(results_dict, selected_model, filename_base="forecast_results"):
    """
    Create Excel buffer for downloading forecast results.
    
    Args:
        results_dict: Dictionary of model results
        selected_model: Model to include in download
        filename_base: Base filename for the download
    
    Returns:
        Tuple of (buffer, suggested_filename)
    """
    if selected_model not in results_dict:
        raise ValueError(f"Model {selected_model} not found in results")
    
    # Get the data to download
    download_data = results_dict[selected_model].copy()
    
    # Add fiscal period column if Date column exists
    if 'Date' in download_data.columns:
        download_data['Fiscal_Period'] = download_data['Date'].apply(
            lambda x: fiscal_period_display(x, fiscal_year_start_month=7)
        )
        
        # Reorder columns to put Fiscal_Period after Date
        cols = list(download_data.columns)
        if 'Fiscal_Period' in cols:
            cols.remove('Fiscal_Period')
            date_idx = cols.index('Date') if 'Date' in cols else 0
            cols.insert(date_idx + 1, 'Fiscal_Period')
            download_data = download_data[cols]
    
    # Create Excel buffer
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        download_data.to_excel(writer, index=False, sheet_name="Forecast_Results")
    
    buffer.seek(0)
    
    # Generate filename with timestamp
    today = datetime.today().strftime("%Y%m%d")
    filename = f"{filename_base}_{selected_model}_{today}.xlsx"
    
    return buffer, filename


def display_model_comparison_table(avg_mapes, avg_smapes=None, avg_mases=None, avg_rmses=None, model_ranks=None):
    """
    Display a comparison table of model performance metrics.
    
    Args:
        avg_mapes: Dictionary of average MAPE by model
        avg_smapes: Dictionary of average SMAPE by model (optional)
        avg_mases: Dictionary of average MASE by model (optional)
        avg_rmses: Dictionary of average RMSE by model (optional)
        model_ranks: Dictionary of average ranks by model (optional)
    """
    # Prepare performance data
    performance_data = []
    
    for model_name, mape_value in avg_mapes.items():
        row_data = {
            "Model": model_name,
            "WAPE": f"{mape_value:.1%}",
            "MAPE_Raw": mape_value  # For sorting (key kept for compatibility)
        }
        
        # Add additional metrics if available
        if avg_smapes and model_name in avg_smapes:
            row_data["SMAPE"] = f"{avg_smapes[model_name]:.1%}"
        
        if avg_mases and model_name in avg_mases:
            mase_val = avg_mases[model_name]
            if not np.isnan(mase_val):
                row_data["MASE"] = f"{mase_val:.3f}"
            else:
                row_data["MASE"] = "N/A"
        
        if avg_rmses and model_name in avg_rmses:
            rmse_val = avg_rmses[model_name]
            if not np.isnan(rmse_val):
                row_data["RMSE"] = f"{rmse_val:,.0f}"
            else:
                row_data["RMSE"] = "N/A"
        
        if model_ranks and model_name in model_ranks:
            rank_val = model_ranks[model_name]
            row_data["Avg Rank"] = f"{rank_val:.1f}"
            row_data["Rank_Raw"] = rank_val  # For sorting
        
        performance_data.append(row_data)
    
    # Create DataFrame and sort by best performance
    performance_df = pd.DataFrame(performance_data)
    
    # Sort by rank if available, otherwise by MAPE
    if "Rank_Raw" in performance_df.columns:
        performance_df = performance_df.sort_values("Rank_Raw")
    else:
        performance_df = performance_df.sort_values("MAPE_Raw")
    
    # Remove raw sorting columns before display
    display_cols = [col for col in performance_df.columns if not col.endswith("_Raw")]
    performance_df = performance_df[display_cols]
    
    # Display the table
    st.dataframe(performance_df, hide_index=True, use_container_width=True)


def create_adjustment_controls(unique_products, forecast_dates):
    """
    Create interactive adjustment controls for forecast modification.
    
    Args:
        unique_products: List of product names
        forecast_dates: List of forecast dates
    
    Returns:
        Dictionary of product adjustments
    """
    st.markdown("Adjust individual product forecasts with custom growth/haircut percentages:")
    
    # Create month-year options for dropdown
    month_year_options = []
    for date in forecast_dates:
        date_obj = pd.to_datetime(date)
        month_year = date_obj.strftime("%B %Y")  # e.g., "January 2024"
        month_year_options.append((month_year, date))
    
    # Create adjustment controls in columns
    adjustment_cols = st.columns(min(3, len(unique_products)))
    product_adjustments = {}
    
    for i, product in enumerate(unique_products):
        with adjustment_cols[i % len(adjustment_cols)]:
            st.markdown(f"**{product}**")
            
            # Percentage adjustment
            adj_pct = st.slider(
                f"Adjustment %",
                min_value=-50, max_value=100, value=0, step=5,
                key=f"adj_pct_{product}",
                help="Positive = growth, Negative = haircut"
            )
            
            # Start month selection
            start_month = st.selectbox(
                f"Start Month",
                options=month_year_options,
                format_func=lambda x: x[0],  # Display month-year string
                key=f"start_month_{product}",
                help="Month to begin applying adjustment"
            )
            
            # Only add adjustment if start_month is selected
            if start_month is not None:
                product_adjustments[product] = {
                    'percentage': adj_pct,
                    'start_date': start_month[1],  # Store the actual date
                    'start_month_display': start_month[0]  # Store display string
                }
            else:
                product_adjustments[product] = {
                    'percentage': adj_pct,
                    'start_date': None,
                    'start_month_display': 'Not selected'
                }
    
    return product_adjustments


def create_multi_fiscal_year_adjustment_controls(chart_model_data, fiscal_year_start_month=7):
    """
    Create multi-fiscal year YoY growth adjustment controls.
    Detects how many fiscal years are being forecasted and allows setting
    YoY growth targets for each fiscal year.
    
    Args:
        chart_model_data: DataFrame with forecast data
        fiscal_year_start_month: Fiscal year start month (default: 7 for July)
    
    Returns:
        Dictionary of fiscal year adjustments per product
    """
    st.markdown("### 🎯 **Multi-Fiscal Year Growth Targets**")
    
    # Add clear button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Set YoY growth targets for each forecasted fiscal year.")
    with col2:
        if st.button("🗑️ Clear All", key="clear_fy_adjustments", help="Clear all fiscal year adjustments"):
            # Clear all fiscal year adjustment-related session state
            keys_to_clear = [
                'fiscal_year_adjustments_applied',
                'fiscal_year_adjustment_summary',
                'adjusted_forecast_results',
                'adjustment_type'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Get unique products
    unique_products = chart_model_data["Product"].unique()
    
    # Analyze fiscal year coverage for each product
    fiscal_year_analysis = analyze_fiscal_year_coverage(chart_model_data, fiscal_year_start_month)
    
    if not fiscal_year_analysis:
        st.warning("⚠️ Unable to analyze fiscal year coverage from forecast data.")
        return {}
    
    # Show fiscal year coverage summary
    st.markdown("#### 📊 **Fiscal Year Coverage Analysis**")
    
    max_fy_years = max(len(product_info['fiscal_years']) for product_info in fiscal_year_analysis.values())
    
    summary_data = []
    for product, info in fiscal_year_analysis.items():
        fy_list = ", ".join([f"FY{fy}" for fy in sorted(info['fiscal_years'])])
        summary_data.append({
            'Product': product,
            'Fiscal Years Covered': fy_list,
            'Total FY Count': len(info['fiscal_years']),
            'Current YoY Growth': f"{info.get('current_yoy_growth', 0):.1f}%" if info.get('current_yoy_growth') else "N/A"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.info(f"📈 **Detected {max_fy_years} fiscal year(s)** in forecast period. Set growth targets for each year below.")
    
    # Debug information (can be removed later)
    if st.checkbox("🔍 Show Debug Info", value=False, key="debug_fy_adjustments"):
        st.write("**Current Session State:**")
        stored_adjustments = st.session_state.get('fiscal_year_adjustments_applied', {})
        st.write(f"Stored adjustments: {len(stored_adjustments)} products")
        for prod, adj in stored_adjustments.items():
            enabled_fys = [fy for fy, info in adj.items() if info.get('enabled', False)]
            st.write(f"- {prod}: {enabled_fys} fiscal years enabled")
        
        st.write(f"**Current Form State:**")
        current_enabled = []
        for product in unique_products:
            product_info = fiscal_year_analysis.get(product, {})
            fiscal_years = sorted(product_info.get('fiscal_years', []))
            for fy_year in fiscal_years:
                key = f"enable_fy_{product}_{fy_year}"
                if key in st.session_state and st.session_state[key]:
                    current_enabled.append(f"{product}-FY{fy_year}")
        st.write(f"Currently enabled: {current_enabled}")
    
    # Create adjustment controls for each product and fiscal year
    st.markdown("#### 🎛️ **YoY Growth Target Controls**")
    
    fiscal_year_adjustments = {}
    
    for product in unique_products:
        with st.expander(f"📊 {product} - Fiscal Year Growth Targets", expanded=False):
            product_info = fiscal_year_analysis.get(product, {})
            fiscal_years = sorted(product_info.get('fiscal_years', []))
            
            if not fiscal_years:
                st.warning(f"⚠️ No fiscal year data available for {product}")
                continue
            
            # Show current YoY growth if available
            current_yoy = product_info.get('current_yoy_growth')
            if current_yoy is not None:
                st.metric("Current Projected YoY Growth", f"{current_yoy:.1f}%")
            
            product_fy_adjustments = {}
            
            # Create controls for each fiscal year
            fy_cols = st.columns(min(3, len(fiscal_years)))
            
            for i, fy_year in enumerate(fiscal_years):
                with fy_cols[i % len(fy_cols)]:
                    st.markdown(f"**FY{fy_year}**")
                    
                    # Get current YoY for this specific FY if available
                    fy_current_yoy = product_info.get('fy_specific_yoy', {}).get(fy_year, current_yoy)
                    if fy_current_yoy is not None:
                        st.caption(f"Current: {fy_current_yoy:.1f}%")
                    
                    # Get stored adjustments for persistence (defined once)
                    stored_adjustments = st.session_state.get('fiscal_year_adjustments_applied', {})
                    
                    # Get previously stored target value if available
                    stored_target = None
                    if (product in stored_adjustments and
                        fy_year in stored_adjustments[product] and
                        stored_adjustments[product][fy_year].get('enabled', False)):
                        stored_target = stored_adjustments[product][fy_year].get('target_yoy')
                    
                    # Target YoY growth input
                    default_value = stored_target if stored_target is not None else (float(fy_current_yoy) if fy_current_yoy is not None else 10.0)
                    target_yoy = st.number_input(
                        f"Target YoY Growth (%)",
                        min_value=-50.0,
                        max_value=200.0,
                        value=default_value,
                        step=1.0,
                        key=f"fy_target_{product}_{fy_year}",
                        help=f"Target YoY growth for FY{fy_year}"
                    )
                    
                    # Check if this adjustment was previously enabled (for persistence)
                    was_previously_enabled = (
                        product in stored_adjustments and
                        fy_year in stored_adjustments[product] and
                        stored_adjustments[product][fy_year].get('enabled', False)
                    )
                    
                    # Enable toggle with persistence
                    enable_fy_adjustment = st.checkbox(
                        f"Apply",
                        value=was_previously_enabled,
                        key=f"enable_fy_{product}_{fy_year}",
                        help=f"Enable YoY growth adjustment for FY{fy_year}"
                    )
                    
                    if enable_fy_adjustment:
                        # Get previously stored distribution method if available
                        stored_distribution = None
                        if (product in stored_adjustments and
                            fy_year in stored_adjustments[product] and
                            stored_adjustments[product][fy_year].get('enabled', False)):
                            stored_distribution = stored_adjustments[product][fy_year].get('distribution_method', 'Smooth')
                        
                        # Growth distribution method
                        distribution_options = ["Smooth", "Linear Ramp", "Exponential", "Front-loaded", "Back-loaded"]
                        default_index = 0
                        if stored_distribution and stored_distribution in distribution_options:
                            default_index = distribution_options.index(stored_distribution)
                        
                        distribution_method = st.selectbox(
                            "Distribution",
                            options=distribution_options,
                            index=default_index,
                            key=f"dist_{product}_{fy_year}",
                            help="How to distribute growth across the fiscal year"
                        )
                        
                        product_fy_adjustments[fy_year] = {
                            'target_yoy': target_yoy,
                            'current_yoy': fy_current_yoy,
                            'distribution_method': distribution_method,
                            'enabled': True
                        }
                        
                        # Show adjustment summary with better error handling
                        try:
                            if fy_current_yoy is not None and not pd.isna(fy_current_yoy):
                                adjustment_needed = target_yoy - fy_current_yoy
                                if adjustment_needed > 0:
                                    st.success(f"📈 +{adjustment_needed:.1f}% adjustment needed")
                                elif adjustment_needed < -1:
                                    st.error(f"📉 {adjustment_needed:+.1f}% adjustment needed")
                                else:
                                    st.info(f"📊 {adjustment_needed:+.1f}% adjustment needed")
                            else:
                                st.info(f"🎯 Target: {target_yoy:.1f}% YoY")
                        except Exception as e:
                            # Fallback to simple target display if there's any calculation error
                            st.info(f"🎯 Target: {target_yoy:.1f}% YoY")
                    else:
                        product_fy_adjustments[fy_year] = {'enabled': False}
            
            fiscal_year_adjustments[product] = product_fy_adjustments
    
    return fiscal_year_adjustments


def analyze_fiscal_year_coverage(chart_model_data, fiscal_year_start_month=7):
    """
    Analyze which fiscal years are covered in the forecast data.
    
    Args:
        chart_model_data: DataFrame with forecast data
        fiscal_year_start_month: Fiscal year start month
    
    Returns:
        Dictionary with fiscal year analysis per product
    """
    analysis = {}
    
    for product in chart_model_data["Product"].unique():
        product_data = chart_model_data[chart_model_data["Product"] == product].copy()
        
        if product_data.empty:
            continue
        
        # Convert dates
        product_data['Date'] = pd.to_datetime(product_data['Date'])
        
        # Separate actual and forecast data
        actual_data = product_data[product_data['Type'].isin(['actual', 'history', 'historical'])].copy()
        forecast_data = product_data[product_data['Type'] == 'forecast'].copy()
        
        # Determine fiscal years covered
        fiscal_years_covered = set()
        
        # Add fiscal years from forecast data
        if not forecast_data.empty:
            for _, row in forecast_data.iterrows():
                date = row['Date']
                fy = calculate_fiscal_year(date, fiscal_year_start_month)
                fiscal_years_covered.add(fy)
        
        # Calculate current YoY growth if possible
        current_yoy_growth = None
        if not actual_data.empty and not forecast_data.empty:
            try:
                # Get last 12 months of actual data
                actual_monthly = actual_data.groupby(actual_data['Date'].dt.to_period('M'))['ACR'].sum()
                last_12_actual = actual_monthly.tail(12).sum() if len(actual_monthly) >= 12 else None
                
                # Get first 12 months of forecast data
                forecast_monthly = forecast_data.groupby(forecast_data['Date'].dt.to_period('M'))['ACR'].sum()
                first_12_forecast = forecast_monthly.head(12).sum() if len(forecast_monthly) >= 12 else None
                
                if last_12_actual and first_12_forecast and last_12_actual > 0:
                    current_yoy_growth = ((first_12_forecast / last_12_actual) - 1.0) * 100.0
            except Exception:
                pass
        
        analysis[product] = {
            'fiscal_years': list(fiscal_years_covered),
            'current_yoy_growth': current_yoy_growth,
            'actual_data_available': not actual_data.empty,
            'forecast_data_available': not forecast_data.empty
        }
    
    return analysis


def calculate_fiscal_year(date, fiscal_year_start_month=7):
    """
    Calculate fiscal year for a given date.
    
    Args:
        date: pandas Timestamp or datetime
        fiscal_year_start_month: Fiscal year start month
    
    Returns:
        int: Fiscal year
    """
    date = pd.Timestamp(date)
    
    if date.month >= fiscal_year_start_month:
        return date.year + 1
    else:
        return date.year


def apply_multi_fiscal_year_adjustments(chart_model_data, fiscal_year_adjustments, fiscal_year_start_month=7):
    """
    Apply multi-fiscal year YoY growth adjustments to forecast data.
    
    Args:
        chart_model_data: DataFrame with forecast data
        fiscal_year_adjustments: Dictionary of fiscal year adjustments per product
        fiscal_year_start_month: Fiscal year start month
    
    Returns:
        DataFrame: Adjusted forecast data
    """
    adjusted_data = chart_model_data.copy()
    adjustment_summary = {}
    
    for product, fy_adjustments in fiscal_year_adjustments.items():
        product_mask = adjusted_data['Product'] == product
        product_data = adjusted_data[product_mask].copy()
        
        if product_data.empty:
            continue
        
        # Process fiscal years in chronological order to ensure proper YoY compounding
        sorted_fy_years = sorted([fy for fy, adj in fy_adjustments.items() if adj.get('enabled', False)])
        
        for fy_year in sorted_fy_years:
            adjustment_info = fy_adjustments[fy_year]
            
            target_yoy = adjustment_info['target_yoy']
            current_yoy = adjustment_info.get('current_yoy')
            distribution_method = adjustment_info.get('distribution_method', 'Smooth')
            
            # Define fiscal year date range
            fy_start = pd.Timestamp(year=fy_year - 1, month=fiscal_year_start_month, day=1)
            fy_end = pd.Timestamp(year=fy_year, month=fiscal_year_start_month, day=1) - pd.DateOffset(days=1)
            
            # Get forecast data within this fiscal year (from the adjusted data, not original)
            forecast_mask = (
                (adjusted_data['Product'] == product) &
                (adjusted_data['Type'] == 'forecast') &
                (pd.to_datetime(adjusted_data['Date']) >= fy_start) &
                (pd.to_datetime(adjusted_data['Date']) <= fy_end)
            )
            
            fy_forecast_data = adjusted_data[forecast_mask].copy()
            
            if fy_forecast_data.empty:
                continue
            
            # For sequential processing, we need to calculate YoY relative to the previous FY
            # Get the previous fiscal year's total for comparison
            prev_fy_year = fy_year - 1
            prev_fy_start = pd.Timestamp(year=prev_fy_year - 1, month=fiscal_year_start_month, day=1)
            prev_fy_end = pd.Timestamp(year=prev_fy_year, month=fiscal_year_start_month, day=1) - pd.DateOffset(days=1)
            
            # Get previous FY data from adjusted data (to account for previous adjustments)
            prev_fy_mask = (
                (adjusted_data['Product'] == product) &
                (pd.to_datetime(adjusted_data['Date']) >= prev_fy_start) &
                (pd.to_datetime(adjusted_data['Date']) <= prev_fy_end)
            )
            
            prev_fy_data = adjusted_data[prev_fy_mask]
            
            # Calculate the target multiplier for this fiscal year
            target_multiplier = 1 + (target_yoy / 100)
            
            if not prev_fy_data.empty:
                # Calculate current YoY based on current adjusted values
                current_fy_total = fy_forecast_data['ACR'].sum()
                prev_fy_total = prev_fy_data['ACR'].sum()
                
                if prev_fy_total > 0:
                    current_yoy_actual = ((current_fy_total / prev_fy_total) - 1) * 100
                    adjustment_ratio = target_multiplier / (1 + current_yoy_actual / 100)
                else:
                    adjustment_ratio = target_multiplier
            else:
                # First fiscal year or no previous data, use target as absolute multiplier
                if current_yoy is not None:
                    adjustment_ratio = target_multiplier / (1 + current_yoy / 100)
                else:
                    adjustment_ratio = target_multiplier
            
            # Apply distribution method
            num_months = len(fy_forecast_data)
            if num_months == 0:
                continue  # Skip if no data in this fiscal year
                
            if distribution_method == "Linear Ramp":
                factors = np.linspace(adjustment_ratio * 0.8, adjustment_ratio * 1.2, num_months)
            elif distribution_method == "Exponential":
                x = np.linspace(0.5, 2.0, num_months)
                factors = adjustment_ratio * (x / np.mean(x))
            elif distribution_method == "Front-loaded":
                weights = np.linspace(1.5, 0.5, num_months)
                factors = adjustment_ratio * (weights / np.mean(weights))
            elif distribution_method == "Back-loaded":
                weights = np.linspace(0.5, 1.5, num_months)
                factors = adjustment_ratio * (weights / np.mean(weights))
            else:  # Smooth
                factors = np.full(num_months, adjustment_ratio)
            
            # Apply adjustments
            fy_forecast_indices = fy_forecast_data.index
            original_values = adjusted_data.loc[fy_forecast_indices, 'ACR'].values
            
            # Ensure factors and values have matching dimensions
            if len(original_values) != len(factors):
                # Adjust factors to match actual data length
                if len(factors) > len(original_values):
                    # Trim factors to match data length
                    factors = factors[:len(original_values)]
                else:
                    # Repeat the last factor value to match data length
                    last_factor = factors[-1] if len(factors) > 0 else adjustment_ratio
                    factors = np.concatenate([factors, np.full(len(original_values) - len(factors), last_factor)])
            
            # Apply the adjustments
            try:
                adjusted_values = original_values * factors
                adjusted_data.loc[fy_forecast_indices, 'ACR'] = adjusted_values
            except Exception as e:
                # Skip this adjustment if there's still a dimension mismatch
                continue
            
            # Track adjustment summary
            if product not in adjustment_summary:
                adjustment_summary[product] = {}
            
            # Calculate total impact (only if adjustment was successful)
            try:
                total_impact = adjusted_values.sum() - original_values.sum()
            except:
                total_impact = 0
            
            adjustment_summary[product][f"FY{fy_year}"] = {
                'target_yoy': target_yoy,
                'current_yoy': current_yoy,
                'adjustment_ratio': adjustment_ratio,
                'distribution_method': distribution_method,
                'months_adjusted': num_months,
                'total_impact': total_impact
            }
    
    return adjusted_data, adjustment_summary


def display_diagnostic_messages(messages, max_messages=10):
    """
    Display diagnostic messages in an organized format.
    
    Args:
        messages: List of diagnostic message strings
        max_messages: Maximum number of messages to display
    """
    if not messages:
        st.info("No diagnostic messages available.")
        return
    
    # Group messages by type
    success_msgs = [msg for msg in messages if msg.startswith("✅")]
    warning_msgs = [msg for msg in messages if msg.startswith("⚠️")]
    error_msgs = [msg for msg in messages if msg.startswith("❌")]
    info_msgs = [msg for msg in messages if not any(msg.startswith(prefix) for prefix in ["✅", "⚠️", "❌"])]
    
    # Display each category
    if success_msgs:
        with st.expander(f"✅ Success Messages ({len(success_msgs)})", expanded=False):
            for msg in success_msgs[:max_messages]:
                st.success(msg)
            if len(success_msgs) > max_messages:
                st.caption(f"... and {len(success_msgs) - max_messages} more")
    
    if warning_msgs:
        with st.expander(f"⚠️ Warnings ({len(warning_msgs)})", expanded=False):
            for msg in warning_msgs[:max_messages]:
                st.warning(msg)
            if len(warning_msgs) > max_messages:
                st.caption(f"... and {len(warning_msgs) - max_messages} more")
    
    if error_msgs:
        with st.expander(f"❌ Errors ({len(error_msgs)})", expanded=True):
            for msg in error_msgs[:max_messages]:
                st.error(msg)
            if len(error_msgs) > max_messages:
                st.caption(f"... and {len(error_msgs) - max_messages} more")
    
    if info_msgs:
        with st.expander(f"ℹ️ Information ({len(info_msgs)})", expanded=False):
            for msg in info_msgs[:max_messages]:
                st.info(msg)
            if len(info_msgs) > max_messages:
                st.caption(f"... and {len(info_msgs) - max_messages} more")


def display_forecast_results():
    """Display forecast results from session state with improved UX"""
    results = st.session_state.forecast_results
    avg_mapes = st.session_state.forecast_mapes
    product_mapes = st.session_state.get('product_mapes', {})
    best_models_per_product = st.session_state.get('best_models_per_product', {})
    best_mapes_per_product = st.session_state.get('best_mapes_per_product', {})
    sarima_params = st.session_state.forecast_sarima_params
    diagnostic_messages = st.session_state.diagnostic_messages
    uploaded_filename = st.session_state.uploaded_filename

    # Defensive check
    if not results or not avg_mapes:
        st.error("⚠️ Session state data is incomplete. Please run a new forecast.")
        return

    # Backward compatibility: rename legacy Raw key to Mix if present (results + metrics)
    try:
        if "Best per Product (Raw)" in results and "Best per Product (Mix)" not in results:
            results["Best per Product (Mix)"] = results.pop("Best per Product (Raw)")
            st.session_state.forecast_results = results
        if "Best per Product (Raw)" in avg_mapes and "Best per Product (Mix)" not in avg_mapes:
            avg_mapes["Best per Product (Mix)"] = avg_mapes["Best per Product (Raw)"]
    except Exception:
        pass

    # Determine active best metrics aligned with backtesting-driven selection
    # Prefer selected per-product backtesting WAPEs when available to avoid confusion with non-eligible minima
    best_model = min(avg_mapes, key=avg_mapes.get)
    if best_model == "Best per Product (Raw)" and "Best per Product (Mix)" in results:
        best_model = "Best per Product (Mix)"
    best_model_display = best_model if best_model != "Best per Product (Raw)" else "Best per Product (Mix)"
    try:
        # Use mean of selected per‑product WAPEs (backtesting view) if available
        if best_mapes_per_product and isinstance(best_mapes_per_product, dict) and len(best_mapes_per_product) > 0:
            vals = [float(v) for v in best_mapes_per_product.values() if v is not None and np.isfinite(v)]
            best_mape = (np.mean(vals) * 100.0) if vals else (avg_mapes.get(best_model, 1.0) * 100.0)
        else:
            best_mape = avg_mapes.get(best_model, 1.0) * 100.0
    except Exception:
        best_mape = avg_mapes.get(best_model, 1.0) * 100.0

    # === COMBINED RESULTS SUMMARY (Summary + Key Metrics will render after active view resolved) ===
    st.markdown("## 📈 Results Summary")
    total_products = len({product for model_results in results.values() for product in model_results["Product"].unique()}) if results else 0
    confidence_level = "High" if best_mape <= 15 else ("Medium" if best_mape <= 25 else "Low")
    
    # === BACKTESTING RESULTS SECTION ===
    backtesting_results = st.session_state.get('backtesting_results', {})
    
    if backtesting_results:
        st.markdown("## 🔍 **Backtesting Results & Validation**")
        st.info("""
        **Backtesting** validates forecasting models by testing them on historical data they haven't seen during training. 
        This gives you confidence that the models will perform well on future data.
        """)
        
        # Display backtesting results
        display_backtesting_results(backtesting_results)
        
        st.markdown("---")
    else:
        st.info("💡 **Backtesting**: Enable backtesting in the sidebar to see model validation results.")
        st.markdown("---")

    # Short model label mapper
    def _short_model_label(name: str) -> str:
        mapping = {
            "Best per Product (Backtesting)": "Backtesting"
        }
        return mapping.get(name, name)

    # Metrics will be displayed later once forecast selection & totals computed

    # Optional detailed comparison hidden by default
    # (Removed Standard vs Backtesting comparison expander)

    # (Executive Summary CTA download removed by request; download moved next to selection toggles below)

    # Restore model view toggles (exclude internal diagnostic variants to reduce confusion)
    composite_keys = [k for k in results.keys() if k.startswith("Best per Product (")]  # exclude raw base key
    label_explanations = {"Backtesting": "Deeper walk‑forward / CV driven per‑product picks (more robust)."}
    def _label_for(k: str) -> str:
        if k.endswith("(Backtesting)"): return "Backtesting"
        return k
    # Only surface Backtesting; remove view selector
    active_key = None
    for pref in ["Best per Product (Backtesting)"]:
        if pref in results:
            active_key = pref
            break
    if not active_key:
        active_key = best_model  # ultimate fallback

    # (Download CSV moved to bottom inside a collapsible pane)

    # Simple info about average WAPE for the active model key
    try:
        active_mape = st.session_state.forecast_mapes.get(active_key, np.nan)
        if np.isfinite(active_mape):
            st.caption(f"Average WAPE (active view): {active_mape*100:.1f}%")
    except Exception:
        pass

    # === KEY PERFORMANCE INDICATORS ===
    # (Removed standalone Key Metrics header; unified grid shown below)
    
    def _coerce_model_data(model_data):
        """Return a unified DataFrame for a model's data.

        Handles cases where model_data is already a DataFrame or a list of per‑product DataFrames.
        Falls back to empty DataFrame if structure unrecognized.
        """
        try:
            if isinstance(model_data, list):
                dfs = [d for d in model_data if hasattr(d, 'columns') and 'Product' in d.columns]
                if dfs:
                    return pd.concat(dfs, ignore_index=True)
                # If list but no valid DataFrames, return empty
                return pd.DataFrame()
            if hasattr(model_data, 'columns'):
                return model_data
        except Exception:
            pass
        return pd.DataFrame()

    df_active = _coerce_model_data(results.get(active_key, pd.DataFrame()))
    # Helper slices
    if not df_active.empty and {'Product', 'Date', 'ACR'}.issubset(df_active.columns):
        if 'Type' in df_active.columns:
            type_col = df_active['Type'].fillna('')
            is_fore = type_col.isin(['forecast','non-compliant-forecast'])
            is_hist = type_col.eq('actual')
        else:
            is_fore = pd.Series([True]*len(df_active))
            is_hist = pd.Series([False]*len(df_active))
        df_fore = df_active[is_fore].copy()
        df_hist = df_active[is_hist].copy()
    else:
        df_fore = pd.DataFrame()
        df_hist = pd.DataFrame()

    # Debug info
    if df_active.empty:
        st.warning("⚠️ No data available for selected model")
    elif df_fore.empty:
        st.warning("⚠️ No forecast data found in selected model")
        st.write(f"Data types available: {df_active.get('Type', pd.Series()).unique().tolist()}")
        
    # KPI 1: Total forecast (selected mode)
    total_forecast = float(df_fore['ACR'].sum()) if not df_fore.empty else 0.0

    # (Removed YoY Growth KPI per user request; keep helper functions for any downstream per-product use.)
    def _last_12_actual_sum(dh: pd.DataFrame) -> float:
        if dh.empty:
            return 0.0
        d = dh.copy(); d['Date'] = pd.to_datetime(d['Date']); d = d.sort_values('Date')
        months = d['Date'].dt.to_period('M')
        uniq = months.drop_duplicates()
        last12 = uniq.tail(12)
        if len(last12) < 12:
            return 0.0
        # ensure contiguous monthly sequence
        try:
            if any((last12[i] - last12[i-1]) != 1 for i in range(1, len(last12))):
                return 0.0
        except Exception:
            return 0.0
        mask = months.isin(last12)
        return float(d.loc[mask,'ACR'].sum())
    
    def _first_12_fore_sum(df_f: pd.DataFrame) -> float:
        if df_f.empty:
            return 0.0
        d = df_f.copy(); d['Date'] = pd.to_datetime(d['Date']); d = d.sort_values('Date')
        months = d['Date'].dt.to_period('M')
        uniq = months.drop_duplicates()
        first12 = uniq.head(12)
        if len(first12) < 12:
            return 0.0
        try:
            if any((first12[i] - first12[i-1]) != 1 for i in range(1, len(first12))):
                return 0.0
        except Exception:
            return 0.0
        mask = months.isin(first12)
        return float(d.loc[mask,'ACR'].sum())
    
    hist_12 = _last_12_actual_sum(df_hist)
    fore_12 = _first_12_fore_sum(df_fore)
    MIN_BASE = 1e6
    yoy = ((fore_12 / hist_12) - 1.0) * 100.0 if hist_12 > MIN_BASE and fore_12 > 0 else np.nan

    # KPI 2: Avg MoM growth over forecast horizon
    try:
        agg = df_fore.groupby('Date')['ACR'].sum().sort_index()
        mom = agg.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        avg_mom = float(mom.mean() * 100.0) if not mom.empty else np.nan
    except Exception:
        avg_mom = np.nan

    # KPI 3: Confidence (reuse computed)
    conf = confidence_level

    # KPI 4: Method mix (share by BestModel per product)
    mix_label = ""
    try:
        if 'BestModel' in df_fore.columns:
            first_by_product = df_fore.groupby('Product')['BestModel'].first()
            counts = first_by_product.value_counts(normalize=True).sort_values(ascending=False)
            items = list(counts.items())
            full_mix = ", ".join([f"{k} {v*100:.0f}%" for k, v in items])
            # More generous compact display for better readability
            if len(items) >= 3:
                mix_label = f"{items[0][0]} {items[0][1]*100:.0f}%, {items[1][0]} {items[1][1]*100:.0f}%" + (f" (+{len(items)-2})" if len(items) > 2 else "")
            elif len(items) == 2:
                mix_label = f"{items[0][0]} {items[0][1]*100:.0f}%, {items[1][0]} {items[1][1]*100:.0f}%"
            elif len(items) == 1:
                mix_label = f"{items[0][0]} {items[0][1]*100:.0f}%"
            else:
                mix_label = ""
            mix_help = full_mix
        else:
            mix_help = ""
    except Exception:
        mix_label = ""
        mix_help = ""

    # Show KPI cards (using 4 columns for better spacing)
    # === Unified Metrics Grid (Summary + Forecast KPIs) ===
    metrics_cols = st.columns(6)
    with metrics_cols[0]:
        st.metric("Products", total_products)
    with metrics_cols[1]:
        st.metric("Best WAPE", f"{best_mape:.1f}%")
    with metrics_cols[2]:
        st.metric("Confidence", confidence_level)
    with metrics_cols[3]:
        # Replace single-approach KPI with composite label and method mix
        label = "Best per Product (Backtesting)"
        mix = st.session_state.get('best_models_per_product_backtesting_mix', None)
        st.metric(label, mix or "Backtesting")
    with metrics_cols[4]:
        st.metric("Total Forecast", f"${total_forecast/1e6:.1f}M")
    with metrics_cols[5]:
        st.metric("Avg MoM", f"{avg_mom:.1f}%" if np.isfinite(avg_mom) else "—", help="Average month-over-month growth")

    if mix_label:
        st.caption(f"Model Mix: {mix_label}")
    # Concise rationale line
    st.caption("Selection: lowest mean WAPE on backtesting (ties → p75 WAPE → MASE).")
    st.markdown("---")

    # Product summary table (collapsible)
    with st.expander("📄 Products (selected mode)", expanded=False):
        rows = []
        if not df_active.empty:
            for product, grp in df_active[df_active['Type'].eq('forecast') if 'Type' in df_active.columns else df_active.index.isin(df_active.index)].groupby('Product'):
                total_p = float(grp['ACR'].sum())
                model_p = grp['BestModel'].iloc[0] if 'BestModel' in grp.columns and not grp['BestModel'].isna().all() else ''
                # per product avg MoM
                try:
                    agg_p = grp.groupby('Date')['ACR'].sum().sort_index()
                    mom_p = agg_p.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                    avg_mom_p = float(mom_p.mean() * 100.0) if not mom_p.empty else np.nan
                except Exception:
                    avg_mom_p = np.nan
                trend = "↑" if np.isfinite(avg_mom_p) and avg_mom_p > 0.5 else ("↓" if np.isfinite(avg_mom_p) and avg_mom_p < -0.5 else "→")
                # per product WAPE (prefer backtesting per‑product mapes)
                wape_prod = None
                try:
                    if best_mapes_per_product and product in best_mapes_per_product and np.isfinite(best_mapes_per_product[product]):
                        wape_prod = float(best_mapes_per_product[product]) * 100.0
                    elif isinstance(product_mapes, dict) and model_p in product_mapes and product in product_mapes[model_p]:
                        wape_prod = float(product_mapes[model_p][product]) * 100.0
                except Exception:
                    wape_prod = None
                # Drift badge (if drift applied to this product's active model forecast)
                drift_products = st.session_state.get('drift_applied_products', [])
                drift_badge = '🪄' if product in drift_products else ''
                rows.append({
                    'Product': product,
                    'Model': (model_p + (' ' + drift_badge if drift_badge else '')),
                    'ForecastTotal': total_p / 1e6,  # Convert to millions for better display
                    'WAPE%': wape_prod,
                    'AvgMoM%': avg_mom_p,
                    'Trend': trend
                })
        if rows:
            df_sum = pd.DataFrame(rows).sort_values(by='ForecastTotal', ascending=False)
            st.dataframe(
                df_sum,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'ForecastTotal': st.column_config.NumberColumn('Forecast ($M)', format="%.1f", help='Sum of forecast horizon in millions'),
                    'WAPE%': st.column_config.NumberColumn('WAPE %', format="%.1f%%", help='Per‑product validation WAPE (lower is better)'),
                    'AvgMoM%': st.column_config.NumberColumn('MoM %', format="%.1f%%", help='Average month-over-month growth'),
                    'Model': st.column_config.TextColumn('Model', help='Chosen model per product. 🪄 = drift applied to flat forecast.')
                }
            )

    # (Backtesting sections moved to bottom to declutter the primary view)

    # (Forecast impact section moved to bottom)

    # Key insights
    if best_models_per_product:
        # Check if multi-metric ranking was used
        if st.session_state.get('model_avg_ranks'):
            st.success("✅ **Backtesting-Only Selection:** Each product uses the model with the lowest WAPE on cross‑validated backtests (ties → p75 WAPE → MASE).")
            with st.expander("📊 How Selection Works (Backtesting WAPE)", expanded=False):
                st.markdown("""
                **Per‑product selection, scored strictly on backtesting WAPE:**
                - **Primary:** Recency‑weighted mean WAPE across folds (falls back to mean WAPE)
                - **Tie‑breaks:** p75 WAPE → MASE → trend‑improvement check
                - **Eligibility:** Enough history for ≥4 folds (current config ≈ 30 months), MASE < 1.0, ≥10% better WAPE than Seasonal‑Naive, and stability p95 WAPE ≤ 2.25× mean (≤2.5× with high fold consistency; stricter thresholds for LightGBM).
                - **Fallback:** If backtesting is insufficient, the app falls back to **Best per Product (Standard)** multi‑metric selection. Seasonal‑Naive remains the baseline and may be chosen when it is the only eligible option.
                """)
        else:
            st.success("✅ **Backtesting Applied:** Models are chosen per product by lowest WAPE.")
    
    # Business warnings (if any)
    poly_warnings = []
    if best_models_per_product:
        poly_products = [product for product, model in best_models_per_product.items() 
                       if model in ["Poly-2", "Poly-3"]]
        if poly_products:
            poly_warnings = poly_products
    elif best_model in ["Poly-2", "Poly-3"]:
        poly_warnings = ["All products"]
    
    if poly_warnings:
        business_aware_status = "enabled" if st.session_state.get('business_aware_selection_used', False) else "disabled"
        st.warning(f"""
        ⚠️ **Business Alert:** Polynomial models detected for: {', '.join(poly_warnings[:3])}{'...' if len(poly_warnings) > 3 else ''}. 
        These may create unrealistic growth for revenue forecasting. Business-Aware Selection is **{business_aware_status}**.
        """)
    
    # Yearly renewals overlay indicator
    if st.session_state.get('yearly_renewals_applied', False):
        st.info("📋 **Non-Compliant Upfront RevRec Applied:** Historical non-compliant revenue recognition data has been added, and future renewals projected at 100% probability. Red lines = historical, Dark red lines = projected future renewals.")
    
    st.markdown("---")        
    # === FORECAST VISUALIZATIONS (HIGH PRIORITY) ===
    st.markdown("## 📈 **Your Forecast Results**")
    
    # Model selection for viewing charts
    col1, col2 = st.columns([3, 1])
    with col1:
        raw_keys = list(results.keys())
        # Keep base individual models (no parentheses) and the three sanctioned composite variants
        allowed_composites = {
            "Best per Product (Standard)",
            "Best per Product (Backtesting)",
            "Best per Product (Mix)"
        }
        cleaned = []
        seen = set()
        for k in raw_keys:
            if k == "Best per Product":
                # Hide legacy generic aggregate (redundant now)
                continue
            if k.startswith("Best per Product (") and k not in allowed_composites:
                # Skip any unexpected composite flavor
                continue
            label = k.strip()
            if label in seen:
                continue
            seen.add(label)
            cleaned.append(label)
        # Ensure composites appear grouped at end in preferred order
        ordered = [m for m in cleaned if m not in allowed_composites]
        for comp in ["Best per Product (Standard)", "Best per Product (Backtesting)", "Best per Product (Mix)"]:
            if comp in cleaned:
                ordered.append(comp)
        # Prioritize "Best per Product (Backtesting)" as the default, then fallback to other options
        if "Best per Product (Backtesting)" in ordered:
            default_key = "Best per Product (Backtesting)"
        elif "Best per Product (Mix)" in ordered:
            default_key = "Best per Product (Mix)"
        elif "Best per Product (Standard)" in ordered:
            default_key = "Best per Product (Standard)"
        elif best_model in ordered:
            default_key = best_model
        else:
            default_key = ordered[0]
        default_idx = ordered.index(default_key) if default_key in ordered else 0
        chart_model = st.selectbox(
            "📊 Select model to view",
            options=ordered,
            index=default_idx,
            key="model_choice_charts",
            help="🏆 Best per Product (Backtesting): Rigorous walk-forward validation (recommended). 📊 Best per Product (Standard): Multi-metric ranking fallback for insufficient history. 🎯 Individual models: Single model consistency."
        )
    with col2:
        chart_mape = avg_mapes.get(chart_model, np.nan) * 100
        if not np.isnan(chart_mape) and np.isfinite(chart_mape):
            if chart_model == "Best per Product (Backtesting)":
                st.success(f"🏆 {chart_mape:.1f}% WAPE")
            elif chart_model == "Best per Product (Standard)":
                st.warning(f"📊 {chart_mape:.1f}% WAPE")
            elif chart_model in ["Best per Product (Mix)"]:
                st.success(f"🏆 {chart_mape:.1f}% WAPE")
            elif chart_model == best_model:
                st.success(f"🏆 {chart_mape:.1f}% WAPE")
            else:
                st.info(f"📈 WAPE: {chart_mape:.1f}%")
        else:
            # Fallback for models without WAPE data or infinite WAPE
            if chart_model and chart_model.startswith("Best per Product"):
                st.success("🏆 Best Model (Composite)")
            else:
                st.info("📈 Model Selected")
    
    # Add guidance when Standard is selected
    if chart_model == "Best per Product (Standard)":
        st.info("ℹ️ **Standard Selection**: Uses multi-metric ranking. Recommended only when insufficient data for backtesting (<24 months history).")
    
    # Use adjusted data if available, otherwise use original  
    if st.session_state.get('adjusted_forecast_results') is not None:
        chart_model_data = _coerce_model_data(st.session_state.adjusted_forecast_results[chart_model])
        
        # Show adjustment info based on type
        adjustment_type = st.session_state.get('adjustment_type', 'unknown')
        
        if adjustment_type == "manual":
            manual_adjustments = st.session_state.get('product_adjustments_applied', {})
            if any(adj.get('percentage', 0) != 0 for adj in manual_adjustments.values()):
                st.info("📊 **Manual Adjustments Applied**: Showing forecasts with your custom percentage adjustments")
        elif adjustment_type == "fiscal_year_growth":
            fy_adjustments = st.session_state.get('fiscal_year_adjustments_applied', {})
            enabled_products = []
            for product, fy_adj in fy_adjustments.items():
                if any(adj.get('enabled', False) for adj in fy_adj.values()):
                    enabled_products.append(product)
            if enabled_products:
                st.success(f"🎯 **Fiscal Year Growth Targets Applied**: Showing smoothed YoY growth for {len(enabled_products)} product(s)")
    else:
        chart_model_data = _coerce_model_data(results[chart_model])
        
        # Business warnings for polynomial models
        if chart_model in ["Poly-2", "Poly-3"]:
            st.warning("⚠️ **Business Warning**: Polynomial models may create unrealistic growth curves for revenue forecasting.")
    # === DYNAMIC FORECAST TOTALS SECTION (now full width) ===
    st.markdown("### 💰 **Forecast Totals**")

    # Calculate totals for the selected model (include both forecast types)
    forecast_data = chart_model_data[chart_model_data['Type'] == 'forecast']
    noncompliant_forecast_data = chart_model_data[chart_model_data['Type'] == 'non-compliant-forecast']
    all_forecast_data = pd.concat([forecast_data, noncompliant_forecast_data], ignore_index=True)
    unique_products = chart_model_data["Product"].unique()

    if len(all_forecast_data) > 0:
        # Calculate totals per product (including non-compliant forecasts)
        product_totals = {}
        product_noncompliant_totals = {}
        grand_total = 0
        grand_noncompliant_total = 0
        
        for product in unique_products:
            # Regular forecast total
            product_forecast = forecast_data[forecast_data['Product'] == product]
            product_total = product_forecast['ACR'].sum()
            product_totals[product] = product_total
            grand_total += product_total
            
            # Non-compliant forecast total
            product_noncompliant_forecast = noncompliant_forecast_data[noncompliant_forecast_data['Product'] == product]
            product_noncompliant_total = product_noncompliant_forecast['ACR'].sum()
            product_noncompliant_totals[product] = product_noncompliant_total
            grand_noncompliant_total += product_noncompliant_total
        
        # Display in columns
        if len(unique_products) > 1:
            # Multiple products - show individual totals + grand total
            cols = st.columns(len(unique_products) + 1)  # +1 for grand total
            
            for i, (product, total) in enumerate(product_totals.items()):
                noncompliant_total = product_noncompliant_totals.get(product, 0)
                combined_total = total + noncompliant_total
                
                with cols[i]:
                    if noncompliant_total > 0:
                        # Show breakdown when non-compliant exists
                        st.metric(
                            label=f"📊 {product}",
                            value=f"${combined_total/1e6:.1f}M",
                            delta=f"+${noncompliant_total/1e6:.1f}M Upfront RevRec",
                            help=f"Total: ${combined_total/1e6:.1f}M (Forecast: ${total/1e6:.1f}M + Non-Compliant: ${noncompliant_total/1e6:.1f}M)"
                        )
                    else:
                        st.metric(
                            label=f"📊 {product}",
                            value=f"${total/1e6:.1f}M",
                            help=f"Total forecast revenue for {product}"
                        )
            
            # Grand total in the last column
            combined_grand_total = grand_total + grand_noncompliant_total
            with cols[-1]:
                if grand_noncompliant_total > 0:
                    st.metric(
                        label="🎯 **Total All Products**",
                        value=f"${combined_grand_total/1e6:.1f}M",
                        delta=f"+${grand_noncompliant_total/1e6:.1f}M Upfront RevRec",
                        help=f"Combined Total: ${combined_grand_total/1e6:.1f}M (Forecast: ${grand_total/1e6:.1f}M + Non-Compliant: ${grand_noncompliant_total/1e6:.1f}M)"
                    )
                else:
                    st.metric(
                        label="🎯 **Total All Products**",
                        value=f"${grand_total/1e6:.1f}M",
                        help="Sum of all product forecasts"
                    )
        else:
            # Single product - show just the total
            product = unique_products[0]
            noncompliant_total = product_noncompliant_totals.get(product, 0)
            combined_total = product_totals.get(product, 0) + noncompliant_total
            
            col1c, col2c, col3c = st.columns([1, 2, 1])
            with col2c:
                if noncompliant_total > 0:
                    st.metric(
                        label=f"🎯 **Total Forecast - {product}**",
                        value=f"${combined_total/1e6:.1f}M",
                        delta=f"+${noncompliant_total/1e6:.1f}M Upfront RevRec",
                        help=f"Combined Total: ${combined_total/1e6:.1f}M (Forecast: ${product_totals.get(product, 0)/1e6:.1f}M + Non-Compliant: ${noncompliant_total/1e6:.1f}M)"
                    )
                else:
                    st.metric(
                        label=f"🎯 **Total Forecast - {product}**",
                        value=f"${combined_total/1e6:.1f}M",
                        help=f"Total forecast revenue for {product}"
                    )
        
        # Additional context
        forecast_months = len(forecast_data) // len(unique_products) if len(unique_products) > 0 else 0
        if forecast_months > 0:
            avg_monthly = grand_total / forecast_months
            st.caption(f"📅 Forecast covers {forecast_months} months | 📈 Average monthly: ${avg_monthly/1e6:.1f}M")
        
        # Show adjustment indicators if applicable
        if st.session_state.get('adjusted_forecast_results') is not None:
            product_adjustments = st.session_state.get('product_adjustments_applied', {})
            if any(adj.get('percentage', 0) != 0 for adj in product_adjustments.values()):
                adjustments_summary = []
                for product, adj_info in product_adjustments.items():
                    if isinstance(adj_info, dict) and adj_info.get('percentage', 0) != 0:
                        pct = adj_info['percentage']
                        adjustments_summary.append(f"{product}: {'+' if pct > 0 else ''}{pct}%")

                if adjustments_summary:
                    st.info(f"🎛️ **Adjustments Applied:** {', '.join(adjustments_summary)}")
    else:
        st.warning("⚠️ No forecast data available for the selected model")

    # === FISCAL YEAR FORECAST SECTION ===
    st.markdown("### 📅 **Current Fiscal Year Forecast**")
    
    # Get fiscal year start month from config, session state, or default to July
    fiscal_year_start_month = st.session_state.get('fiscal_year_start_month', 7)
    if 'config' in st.session_state and st.session_state.config:
        fiscal_year_start_month = int(st.session_state.config.get('fiscal_year_start_month', fiscal_year_start_month))
    
    # Calculate current fiscal year based on calendar date
    now = datetime.now()
    if now.month >= fiscal_year_start_month:
        calendar_fy_year = now.year + 1
    else:
        calendar_fy_year = now.year
    
    # Determine fiscal year date range for calendar-based FY
    calendar_fy_start_date = pd.Timestamp(year=calendar_fy_year - 1, month=fiscal_year_start_month, day=1)
    calendar_fy_end_date = pd.Timestamp(year=calendar_fy_year, month=fiscal_year_start_month, day=1) - pd.DateOffset(days=1)
    
    # Check if we have data for the full calendar-based fiscal year
    calendar_fy_data = chart_model_data[
        (pd.to_datetime(chart_model_data['Date']) >= calendar_fy_start_date) & 
        (pd.to_datetime(chart_model_data['Date']) <= calendar_fy_end_date)
    ].copy()
    
    # Calculate months elapsed in calendar-based FY to check if we should flip to next year
    calendar_fy_actual_rows = calendar_fy_data[calendar_fy_data['Type'].isin(['actual', 'history', 'historical'])].copy()
    calendar_months_elapsed = 0
    if not calendar_fy_actual_rows.empty:
        calendar_fy_actual_rows['Date'] = pd.to_datetime(calendar_fy_actual_rows['Date'])
        calendar_actual_months = calendar_fy_actual_rows['Date'].dt.to_period('M').unique()
        calendar_months_elapsed = int(len(calendar_actual_months))
    calendar_months_elapsed = min(12, max(0, calendar_months_elapsed))
    calendar_months_remaining = 12 - calendar_months_elapsed
    
    # If current FY has 0 months remaining (all 12 months have actual data), flip to next FY
    if calendar_months_remaining == 0:
        current_fy_year = calendar_fy_year + 1
        fy_start_date = pd.Timestamp(year=current_fy_year - 1, month=fiscal_year_start_month, day=1)
        fy_end_date = pd.Timestamp(year=current_fy_year, month=fiscal_year_start_month, day=1) - pd.DateOffset(days=1)
    else:
        current_fy_year = calendar_fy_year
        fy_start_date = calendar_fy_start_date
        fy_end_date = calendar_fy_end_date
    
    # Calculate fiscal year totals combining actuals + forecasts for the target FY
    fy_data = chart_model_data[
        (pd.to_datetime(chart_model_data['Date']) >= fy_start_date) & 
        (pd.to_datetime(chart_model_data['Date']) <= fy_end_date)
    ].copy()
    
    if not fy_data.empty:
        # Calculate totals by type for current fiscal year
        fy_actuals = fy_data[fy_data['Type'].isin(['actual', 'history', 'historical'])]['ACR'].sum()
        fy_forecast = fy_data[fy_data['Type'] == 'forecast']['ACR'].sum()
        fy_noncompliant = fy_data[fy_data['Type'].isin(['non-compliant', 'non-compliant-forecast'])]['ACR'].sum()
        fy_total = fy_actuals + fy_forecast + fy_noncompliant
        
        # Show fiscal year metrics
        fy_col1, fy_col2, fy_col3, fy_col4 = st.columns(4)
        
        with fy_col1:
            st.metric(
                label=f"📈 FY{current_fy_year} Total",
                value=f"${fy_total/1e6:.1f}M",
                help=f"Complete fiscal year projection including actuals through current date"
            )
        
        with fy_col2:
            st.metric(
                label="✅ Actuals YTD",
                value=f"${fy_actuals/1e6:.1f}M",
                help="Actual revenue captured so far this fiscal year"
            )
        
        with fy_col3:
            st.metric(
                label="🔮 Remaining Forecast",
                value=f"${fy_forecast/1e6:.1f}M",
                help="Forecasted revenue for remaining months of fiscal year"
            )
        
        with fy_col4:
            if fy_noncompliant > 0:
                st.metric(
                    label="⚡ Non-Compliant",
                    value=f"${fy_noncompliant/1e6:.1f}M",
                    help="Non-compliant revenue recognition (historical + future)"
                )
            else:
                st.metric(
                    label="📊 Completion",
                    value=f"{(fy_actuals/fy_total*100):.0f}%" if fy_total > 0 else "0%",
                    help="Percentage of fiscal year completed based on actuals"
                )
        
        # Show fiscal year period info
        # Revised logic: elapsed months are determined by presence of ACTUAL data within the fiscal year,
        # not by today's calendar date. If no actuals yet in FY, elapsed = 0 (so remaining = 12).
        fy_actual_rows = fy_data[fy_data['Type'].isin(['actual', 'history', 'historical'])].copy()
        if not fy_actual_rows.empty:
            fy_actual_rows['Date'] = pd.to_datetime(fy_actual_rows['Date'])
            actual_months = fy_actual_rows['Date'].dt.to_period('M').unique()
            months_elapsed = int(len(actual_months))
        else:
            months_elapsed = 0
        months_elapsed = min(12, max(0, months_elapsed))
        months_remaining = 12 - months_elapsed
        st.caption(
            f"📅 FY{current_fy_year}: {fy_start_date.strftime('%b %Y')} - {fy_end_date.strftime('%b %Y')} | "
            f"⏰ {months_elapsed} months elapsed, {months_remaining} months remaining"
        )
        
        # Show auto-flip indicator if we've switched to next FY due to complete data
        if current_fy_year > calendar_fy_year:
            st.info(f"🔄 **Auto-flipped to FY{current_fy_year}** - Previous fiscal year (FY{calendar_fy_year}) has complete data with 0 months remaining")
    else:
        st.info(f"ℹ️ No data available for current fiscal year (FY{current_fy_year})")

    st.markdown("---")

    # Quick product selector (render only one chart at a time for stability/performance)
    unique_products = list(chart_model_data["Product"].unique()) if not chart_model_data.empty else []
    if len(unique_products) == 0:
        st.warning("No products found in results")
        return
    product = st.selectbox("Select product", options=unique_products, index=0, key="chart_product_choice")
    st.markdown(f"### 📊 **{product}**")
    display_product_forecast(chart_model_data, product, chart_model, best_models_per_product, best_mapes_per_product)

    # === GROWTH ANALYSIS (PROMINENT DISPLAY) ===
    st.markdown("### 📈 **Month-over-Month Growth Analysis**")
    st.markdown(f"Growth trends for **{chart_model}** model")

    # Show adjustment info if applied
    adjustment_type = st.session_state.get('adjustment_type')
    if adjustment_type == "manual" and st.session_state.get('product_adjustments_applied'):
        manual_adjustments = st.session_state.get('product_adjustments_applied', {})
        if any(adj.get('percentage', 0) != 0 for adj in manual_adjustments.values()):
            st.info("📊 Growth charts reflect your custom manual adjustments")
    elif adjustment_type == "fiscal_year_growth" and st.session_state.get('fiscal_year_adjustments_applied'):
        fy_adjustments = st.session_state.get('fiscal_year_adjustments_applied', {})
        enabled_products = []
        for product, fy_adj in fy_adjustments.items():
            if any(adj.get('enabled', False) for adj in fy_adj.values()):
                enabled_products.append(product)
        if enabled_products:
            st.info(f"🎯 Growth charts reflect smoothed fiscal year growth targets for {len(enabled_products)} product(s)")

    # Use the same selected product for growth analysis to avoid rendering many charts at once
    product_mom = product
    df_product_mom = chart_model_data[chart_model_data["Product"] == product_mom].copy()
    dfm_growth = None  # Initialize to handle potential errors

    st.markdown(f"**📊 {product_mom}**")

    # Show specific adjustment info for this product
    if adjustment_type == "manual" and st.session_state.get('product_adjustments_applied') and product_mom in st.session_state.product_adjustments_applied:
        adj_info = st.session_state.product_adjustments_applied[product_mom]
        if isinstance(adj_info, dict):
            adj_pct = adj_info.get('percentage', 0)
            start_display = adj_info.get('start_display', 'Unknown')
            if adj_pct != 0:
                if adj_pct > 0:
                    st.markdown(f"*📈 +{adj_pct}% manual growth from {start_display}*")
                else:
                    st.markdown(f"*📉 {adj_pct}% manual haircut from {start_display}*")
    elif adjustment_type == "fiscal_year_growth" and st.session_state.get('fiscal_year_adjustments_applied') and product_mom in st.session_state.fiscal_year_adjustments_applied:
        fy_info = st.session_state.fiscal_year_adjustments_applied[product_mom]
        enabled_fy_adjustments = []
        for fy_year, adj_info in fy_info.items():
            if adj_info.get('enabled', False):
                target_yoy = adj_info['target_yoy']
                distribution = adj_info.get('distribution_method', 'Smooth')
                enabled_fy_adjustments.append(f"FY{fy_year}: {target_yoy:.1f}% ({distribution})")
        
        if enabled_fy_adjustments:
            adjustments_text = ", ".join(enabled_fy_adjustments)
            st.markdown(f"*🎯 Fiscal Year Growth Targets: {adjustments_text}*")

    try:
        df_product_mom = df_product_mom.set_index("Date")
        # Build normalized MoM series including forecast
        norm_mom = normalized_monthly_by_days(df_product_mom, "ACR")
        mom_growth = norm_mom.pct_change().dropna()

        # Validate growth data
        if len(mom_growth) == 0:
            st.warning(f"⚠️ Insufficient data for growth analysis: {product_mom}")
            dfm_growth = None
            raise ValueError("No MoM data")

        dfm_growth = (
            mom_growth
            .rename("growth")              # series → column name
            .reset_index()                 # ['month','growth']
            .rename(columns={"month": "Date"})
        )

        # Validate growth dataframe
        if len(dfm_growth) == 0 or dfm_growth['growth'].isna().all():
            st.warning(f"⚠️ No valid growth data for {product_mom}")
            dfm_growth = None
            raise ValueError("No valid growth data")

        dfm_growth['growth_formatted'] = dfm_growth['growth'].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )

        # Base chart
        base_mom = alt.Chart(dfm_growth)

        # Line chart with improved styling
        chart_mom_line = base_mom.mark_line(point=True, strokeWidth=2, stroke='#1f77b4').encode(
            x=alt.X("Date:T", title="Month"),
            y=alt.Y("growth:Q", axis=alt.Axis(format="%"), title="MoM Growth"),
            tooltip=["Date:T", alt.Tooltip("growth_formatted:N", title="Growth")]
        )

        # Data labels for key points (show every other point to avoid clutter)
        labels_mom = base_mom.transform_window(
            row_number='row_number()'
        ).transform_filter(
            'datum.row_number % 2 == 1'  # Show every other point
        ).mark_text(
            align='center',
            baseline='bottom',
            dy=-10,
            fontSize=9,
            fontWeight='bold',
            color='#1f77b4'
        ).encode(
            x="Date:T",
            y="growth:Q",
            text=alt.Text("growth_formatted:N")
        )
        chart_mom = (chart_mom_line + labels_mom).properties(
            height=250,
            title=f"Month-over-Month Growth - {product_mom} ({chart_model} Model)"
        )
        st.altair_chart(chart_mom, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Growth chart failed for {product_mom}: {str(e)}")
        st.info("📈 Showing growth data table instead:")

        # Fallback: show growth data table if it was created
        try:
            # Check if dfm_growth was successfully created before the exception
            if dfm_growth is not None and len(dfm_growth) > 0:
                st.dataframe(dfm_growth[['Date', 'growth_formatted']].tail(12), use_container_width=True)
            else:
                st.warning("No growth data available")
        except (NameError, UnboundLocalError, AttributeError):
            st.warning("No growth data available")

    st.markdown("---")

    # === DOWNLOAD SECTION (HIGH PRIORITY) ===
    st.markdown("## 📥 **Download Results**")
    col1, col2 = st.columns([2, 1])
    with col1:
        # Default to Best per Product (Backtesting) and hide legacy generic 'Best per Product'
        download_options = [k for k in results.keys() if k != "Best per Product"]
        default_choice = "Best per Product (Backtesting)" if "Best per Product (Backtesting)" in download_options else best_model
        # Auto-select; no need to expose a confusing selector if only backtesting is supported
        choice = st.selectbox(
            "Choose model for download",
            download_options,
            index=download_options.index(default_choice) if default_choice in download_options else 0,
            key="download_model_choice"
        )

        # Use adjusted results if available, otherwise use current forecast results (which include live conservatism)
        if st.session_state.get('adjusted_forecast_results') is not None:
            download_data = st.session_state.adjusted_forecast_results[choice]
            adjustment_type = st.session_state.get('adjustment_type')
            if adjustment_type == "manual" and st.session_state.get('product_adjustments_applied'):
                st.success("📊 Download includes your custom manual adjustments!")
            elif adjustment_type == "fiscal_year_growth" and st.session_state.get('fiscal_year_adjustments_applied'):
                st.success("🎯 Download includes your fiscal year growth targets!")
        else:
            # Use session state forecast_results which includes live conservatism adjustments
            download_data = st.session_state.forecast_results[choice]
            
        # Show conservatism factor if not 100%
        conservatism_factor = st.session_state.get('forecast_conservatism_used', 100)
        if conservatism_factor != 100:
            st.success(f"🎛️ Download includes {conservatism_factor}% conservatism adjustment!")

        # Show yearly renewals overlay status
        if st.session_state.get('yearly_renewals_applied', False):
            st.success("📋 Download includes non-compliant Upfront RevRec data as separate line items!")

        # Prepare download data with fiscal period column
        download_data_with_fiscal = download_data.copy()
        if 'Date' in download_data_with_fiscal.columns:
            # Add fiscal period column
            download_data_with_fiscal['Fiscal_Period'] = download_data_with_fiscal['Date'].apply(
                lambda x: fiscal_period_display(x, fiscal_year_start_month=7)
            )
            
            # Reorder columns to put Fiscal_Period after Date
            cols = list(download_data_with_fiscal.columns)
            if 'Fiscal_Period' in cols:
                cols.remove('Fiscal_Period')
                date_idx = cols.index('Date') if 'Date' in cols else 0
                cols.insert(date_idx + 1, 'Fiscal_Period')
                download_data_with_fiscal = download_data_with_fiscal[cols]

        buf2 = io.BytesIO()
        with pd.ExcelWriter(buf2, engine="openpyxl") as writer:
            download_data_with_fiscal.to_excel(writer, index=False, sheet_name="Actuals_Forecast")
        buf2.seek(0)
        today = datetime.today().strftime("%Y%m%d")

        # Add adjustment indicators to filename
        filename_suffix = ""
        adjustment_type = st.session_state.get('adjustment_type')
        
        if st.session_state.get('adjusted_forecast_results') is not None:
            if adjustment_type == "manual":
                product_adjustments = st.session_state.get('product_adjustments_applied', {})
                adjustment_percentages = []
                for product, adj_info in product_adjustments.items():
                    if isinstance(adj_info, dict):
                        adj_pct = adj_info.get('percentage', 0)
                        if adj_pct != 0 and adj_pct is not None:
                            # Handle both int and float values with error handling
                            try:
                                if isinstance(adj_pct, (int, float)):
                                    adj_str = f"{int(adj_pct):+d}"
                                else:
                                    adj_float = float(adj_pct)
                                    adj_str = f"{int(adj_float):+d}"
                                adjustment_percentages.append(f"{adj_str}%")
                            except (ValueError, TypeError):
                                continue
                    else:
                        if adj_info != 0 and adj_info is not None:
                            # Handle both int and float values with error handling
                            try:
                                if isinstance(adj_info, (int, float)):
                                    adj_str = f"{int(adj_info):+d}"
                                else:
                                    adj_float = float(adj_info)
                                    adj_str = f"{int(adj_float):+d}"
                                adjustment_percentages.append(f"{adj_str}%")
                            except (ValueError, TypeError):
                                continue

                if adjustment_percentages:
                    filename_suffix += f"_adj{'-'.join(adjustment_percentages)}"
                    
            elif adjustment_type == "fiscal_year_growth":
                fy_adjustments = st.session_state.get('fiscal_year_adjustments_applied', {})
                fy_percentages = []
                for product, fy_info in fy_adjustments.items():
                    if isinstance(fy_info, dict):
                        for fy, fy_data in fy_info.items():
                            # fy_data should be a dict with 'target_yoy', 'enabled', etc.
                            if isinstance(fy_data, dict) and fy_data.get('enabled', False):
                                target_yoy = fy_data.get('target_yoy', 0)
                                if target_yoy != 0 and target_yoy is not None:
                                    try:
                                        if isinstance(target_yoy, (int, float)):
                                            growth_str = f"{int(target_yoy):+d}"
                                        else:
                                            target_float = float(target_yoy)
                                            growth_str = f"{int(target_float):+d}"
                                        fy_percentages.append(f"FY{fy}_{growth_str}%")
                                    except (ValueError, TypeError):
                                        # Skip invalid values
                                        continue
                
                if fy_percentages:
                    filename_suffix += f"_fyGrowth_{'-'.join(fy_percentages)}"

        if st.session_state.get('business_adjustments_applied', False):
            growth = st.session_state.get('business_growth_used', 0)
            if growth != 0 and growth is not None:
                # Handle both int and float values with error handling
                try:
                    if isinstance(growth, (int, float)):
                        growth_str = f"{int(growth):+d}"
                    else:
                        growth_float = float(growth)
                        growth_str = f"{int(growth_float):+d}"
                    filename_suffix += f"_biz{growth_str}%"
                except (ValueError, TypeError):
                    # Skip invalid growth values
                    pass

        if st.session_state.get('yearly_renewals_applied', False):
            filename_suffix += "_YearlyRenewals"
            
        # Add conservatism factor to filename if not 100%
        conservatism_factor = st.session_state.get('forecast_conservatism_used', 100)
        if conservatism_factor != 100:
            filename_suffix += f"_conservatism{conservatism_factor}pct"

        with col2:
            st.download_button(
                "📊 **Download Excel**", data=buf2,
                file_name=f"{Path(uploaded_filename).stem}_{choice}{filename_suffix}_{today}.xlsx",
                type="primary",
                use_container_width=True
            )

        st.markdown("---")

        # === INTERACTIVE ADJUSTMENTS (MEDIUM PRIORITY) ===
        with st.expander("🎛️ **Interactive Forecast Adjustments**", expanded=False):
            
            # Create tabs for different adjustment types
            tab1, tab2 = st.tabs(["📊 Manual Adjustments", "🎯 Fiscal Year Growth Targets"])
            
            with tab1:
                st.markdown("Adjust individual product forecasts with custom growth/haircut percentages:")

                # Get unique products from the best model results
                best_model_data = results[best_model]
                unique_products = best_model_data["Product"].unique()

                # Get forecast date range for start month options
                forecast_dates = best_model_data[best_model_data["Type"] == "forecast"]["Date"].unique()
                forecast_dates = sorted(forecast_dates)

                # Create month-year options for dropdown
                month_year_options = []
                for date in forecast_dates:
                    date_obj = pd.to_datetime(date)
                    month_year = date_obj.strftime("%B %Y")  # e.g., "January 2024"
                    month_year_options.append((month_year, date))

                # Create adjustment controls in columns
                adjustment_cols = st.columns(min(3, len(unique_products)))
                product_adjustments = {}

                for i, product in enumerate(unique_products):
                    with adjustment_cols[i % len(adjustment_cols)]:
                        st.subheader(f"📊 {product}")

                        # Start month dropdown
                        start_month_display = st.selectbox(
                            "Start Month",
                            options=[option[0] for option in month_year_options],
                            index=0,
                            key=f"start_{product}",
                            help="Select when the adjustment should begin"
                        )

                        # Get the actual date for the selected month
                        start_date = next(option[1] for option in month_year_options if option[0] == start_month_display)

                        adjustment_pct = st.slider(
                            f"Adjustment %",
                            min_value=-50,
                            max_value=200,
                            value=0,
                            step=5,
                            key=f"adj_{product}",
                            help=f"Positive values = growth, negative values = haircut"
                        )

                        product_adjustments[product] = {
                            'percentage': adjustment_pct,
                            'start_date': start_date,
                            'start_display': start_month_display
                        }

                        # Show the adjustment impact
                        if adjustment_pct > 0:
                            st.success(f"📈 +{adjustment_pct}% growth from {start_month_display}")
                        elif adjustment_pct < 0:
                            st.error(f"📉 {adjustment_pct}% haircut from {start_month_display}")
                        else:
                            st.info("🔄 No adjustment")

                # Apply adjustments if any are non-zero
                any_manual_adjustments = any(adj['percentage'] != 0 for adj in product_adjustments.values())

                if any_manual_adjustments:
                    st.info("🔧 Manual forecast adjustments are active. Download will include adjusted values.")

                    # Create adjusted results for download using the original logic
                    adjusted_results = {}
                    for model_name, model_data in results.items():
                        adjusted_model_data = model_data.copy()

                        for product, adjustment_info in product_adjustments.items():
                            adjustment_pct = adjustment_info['percentage']
                            start_date = adjustment_info['start_date']

                            if adjustment_pct != 0:
                                # Apply adjustment only to forecast rows for this product starting from the specified date
                                mask = (
                                    (adjusted_model_data["Product"] == product) &
                                    (adjusted_model_data["Type"] == "forecast") &
                                    (adjusted_model_data["Date"] >= start_date)
                                )
                                if mask.any():
                                    multiplier = 1 + (adjustment_pct / 100)
                                    adjusted_model_data.loc[mask, "ACR"] = adjusted_model_data.loc[mask, "ACR"] * multiplier

                        adjusted_results[model_name] = adjusted_model_data

                    # Store adjusted results for download
                    st.session_state.adjusted_forecast_results = adjusted_results
                    st.session_state.product_adjustments_applied = product_adjustments
                    st.session_state.adjustment_type = "manual"
            
            with tab2:
                # Multi-Fiscal Year Growth Target Controls
                fiscal_year_start_month = st.session_state.get('fiscal_year_start_month', 7)
                if 'config' in st.session_state and st.session_state.config:
                    fiscal_year_start_month = int(st.session_state.config.get('fiscal_year_start_month', fiscal_year_start_month))
                
                # Use original data for YoY calculations, not adjusted data
                original_chart_data = _coerce_model_data(results[chart_model])
                
                fiscal_year_adjustments = create_multi_fiscal_year_adjustment_controls(
                    original_chart_data, fiscal_year_start_month
                )
                
                # Apply fiscal year adjustments if any are enabled
                any_fy_adjustments = any(
                    any(fy_adj.get('enabled', False) for fy_adj in product_fy_adj.values())
                    for product_fy_adj in fiscal_year_adjustments.values()
                )
                
                # Check if current fiscal year adjustments differ from stored ones
                stored_fy_adjustments = st.session_state.get('fiscal_year_adjustments_applied', {})
                fy_adjustments_changed = (fiscal_year_adjustments != stored_fy_adjustments)
                
                if any_fy_adjustments:
                    st.info("🎯 Fiscal year growth target adjustments are active. Download will include adjusted values.")
                    
                    # Only recalculate and rerun if adjustments have changed
                    if fy_adjustments_changed or 'adjusted_forecast_results' not in st.session_state:
                        # Apply fiscal year adjustments to all models
                        fy_adjusted_results = {}
                        adjustment_summary = {}
                        
                        for model_name, model_data in results.items():
                            fy_adjusted_data, adj_summary = apply_multi_fiscal_year_adjustments(
                                model_data, fiscal_year_adjustments, fiscal_year_start_month
                            )
                            fy_adjusted_results[model_name] = fy_adjusted_data
                            if adj_summary:
                                adjustment_summary.update(adj_summary)
                        
                        # Store fiscal year adjusted results
                        st.session_state.adjusted_forecast_results = fy_adjusted_results
                        st.session_state.fiscal_year_adjustments_applied = fiscal_year_adjustments
                        st.session_state.fiscal_year_adjustment_summary = adjustment_summary
                        st.session_state.adjustment_type = "fiscal_year_growth"
                        
                        # Clear any manual adjustments to avoid conflicts
                        if 'product_adjustments_applied' in st.session_state:
                            del st.session_state['product_adjustments_applied']
                        
                        # Force rerun to update KPIs and charts
                        st.rerun()
                    
                    # Show summary of applied adjustments
                    adjustment_summary = st.session_state.get('fiscal_year_adjustment_summary', {})
                    if adjustment_summary:
                        st.markdown("#### 📊 **Applied Fiscal Year Growth Targets:**")
                        for product, fy_adjustments in adjustment_summary.items():
                            with st.expander(f"📈 {product} Adjustments", expanded=False):
                                for fy_label, adj_info in fy_adjustments.items():
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(f"{fy_label} Target", f"{adj_info['target_yoy']:.1f}%")
                                    with col2:
                                        current = adj_info.get('current_yoy')
                                        if current is not None:
                                            st.metric("Previous", f"{current:.1f}%")
                                        else:
                                            st.metric("Previous", "N/A")
                                    with col3:
                                        impact = adj_info['total_impact']
                                        st.metric("Impact", f"${impact/1e6:+.1f}M")
                                    
                                    st.caption(f"Distribution: {adj_info['distribution_method']} | Months: {adj_info['months_adjusted']}")
                else:
                    # Only clear fiscal year adjustments if there were previously some applied
                    needs_rerun = False
                    if st.session_state.get('adjustment_type') == "fiscal_year_growth":
                        if 'fiscal_year_adjustments_applied' in st.session_state:
                            del st.session_state['fiscal_year_adjustments_applied']
                            needs_rerun = True
                        if 'fiscal_year_adjustment_summary' in st.session_state:
                            del st.session_state['fiscal_year_adjustment_summary']
                            needs_rerun = True
                        
                        # If no manual adjustments either, clear all adjustments
                        if not any_manual_adjustments:
                            if 'adjusted_forecast_results' in st.session_state:
                                del st.session_state['adjusted_forecast_results']
                                needs_rerun = True
                            if 'adjustment_type' in st.session_state:
                                del st.session_state['adjustment_type']
                                needs_rerun = True
                        
                        # Force rerun to update KPIs when clearing adjustments
                        if needs_rerun:
                            st.rerun()

        # === TECHNICAL DETAILS (LOW PRIORITY - EXPANDABLE) ===

        # Model Performance Comparison
        with st.expander("📊 **Model Performance Comparison**", expanded=False):
            st.markdown("### Multi-Metric Performance Comparison")
            st.caption("📊 Models ranked using MAPE, SMAPE, MASE, and RMSE metrics")

            performance_data = []
            model_avg_ranks = st.session_state.get('model_avg_ranks', {})

            for model_name, mape_value in avg_mapes.items():
                mape_pct = mape_value * 100

                if model_name == "Best per Product":
                    model_display = f"{model_name} ⭐"
                    accuracy_note = "🎯 Optimized per product"
                    rank_info = "N/A"
                else:
                    model_display = model_name
                    accuracy_note = "🥇 Best" if model_name == best_model else f"📈 {(mape_pct - best_mape):+.2f}% vs best"

                    # Add ranking information if available
                    if model_avg_ranks and model_name in model_avg_ranks:
                        avg_rank = model_avg_ranks[model_name]
                        rank_info = f"{avg_rank:.1f}"
                    else:
                        rank_info = "N/A"

                performance_data.append({
                    "Model": model_display,
                    "Avg WAPE": f"{mape_pct:.2f}%",
                    "Multi-Metric Rank": rank_info,
                    "Status": accuracy_note
                })

            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, hide_index=True, use_container_width=True)

            # Show detailed metrics if available
            if st.session_state.get('product_smapes') or st.session_state.get('product_mases'):
                st.markdown("### 📈 All Metrics Summary (WAPE, MASE, SMAPE, RMSE)")
                st.caption("Lower values are better for all metrics")

                # Create summary of all metrics across all products
                all_metrics_data = []
                for model_name in avg_mapes.keys():
                    if model_name == "Best per Product":
                        continue

                    # Calculate averages across all products for each metric
                    avg_smape = "N/A"
                    avg_mase = "N/A"
                    avg_rmse = "N/A"

                    if st.session_state.get('product_smapes') and model_name in st.session_state.product_smapes:
                        smape_values = list(st.session_state.product_smapes[model_name].values())
                        if smape_values:
                            avg_smape = f"{np.mean(smape_values):.1%}"

                    if st.session_state.get('product_mases') and model_name in st.session_state.product_mases:
                        mase_values = list(st.session_state.product_mases[model_name].values())
                        if mase_values:
                            avg_mase = f"{np.mean(mase_values):.2f}"

                    if st.session_state.get('product_rmses') and model_name in st.session_state.product_rmses:
                        rmse_values = list(st.session_state.product_rmses[model_name].values())
                        if rmse_values:
                            avg_rmse = f"${np.mean(rmse_values)/1e6:.1f}M"

                    all_metrics_data.append({
                        "Model": model_name,
                        "WAPE": f"{avg_mapes[model_name]*100:.2f}%",
                        "SMAPE": avg_smape,
                        "MASE": avg_mase,
                        "RMSE": avg_rmse
                    })

                if all_metrics_data:
                    all_metrics_df = pd.DataFrame(all_metrics_data)
                    st.dataframe(all_metrics_df, hide_index=True, use_container_width=True)

            # Per-product model selection details
            if best_models_per_product:
                st.markdown("### Per-Product Model Selection")
                st.caption("📊 Models selected per product by backtesting WAPE (ties → p75 WAPE → MASE). Other metrics shown for diagnostics.")
                product_performance_data = []

                # Get metric data from session state
                product_smapes = st.session_state.get('product_smapes', {})
                product_mases = st.session_state.get('product_mases', {})
                product_rmses = st.session_state.get('product_rmses', {})

                for product, best_model_for_product in best_models_per_product.items():
                    best_mape_for_product = best_mapes_per_product.get(product, np.nan)
                    mape_pct = best_mape_for_product * 100 if not np.isnan(best_mape_for_product) else "N/A"

                    # Get other metrics for the selected model
                    smape_val = "N/A"
                    mase_val = "N/A"
                    rmse_val = "N/A"

                    if best_model_for_product in product_smapes and product in product_smapes[best_model_for_product]:
                        smape_raw = product_smapes[best_model_for_product][product]
                        smape_val = f"{smape_raw*100:.2f}%" if not np.isnan(smape_raw) else "N/A"

                    if best_model_for_product in product_mases and product in product_mases[best_model_for_product]:
                        mase_raw = product_mases[best_model_for_product][product]
                        mase_val = f"{mase_raw:.2f}" if not np.isnan(mase_raw) else "N/A"

                    if best_model_for_product in product_rmses and product in product_rmses[best_model_for_product]:
                        rmse_raw = product_rmses[best_model_for_product][product]
                        if not np.isnan(rmse_raw):
                            if rmse_raw >= 1e6:
                                rmse_val = f"${rmse_raw/1e6:.1f}M"
                            elif rmse_raw >= 1e3:
                                rmse_val = f"${rmse_raw/1e3:.1f}K"
                            else:
                                rmse_val = f"${rmse_raw:.0f}"

                    if isinstance(mape_pct, (int, float)):
                        if mape_pct <= 10:
                            accuracy_icon = "🎯 Excellent"
                        elif mape_pct <= 20:
                            accuracy_icon = "📈 Good"
                        elif mape_pct <= 50:
                            accuracy_icon = "⚠️ Moderate"
                        else:
                            accuracy_icon = "🔴 Lower"
                    else:
                        accuracy_icon = "❓ Unknown"

                    product_performance_data.append({
                        "Product": product,
                        "Best Model": best_model_for_product,
                        "WAPE": f"{mape_pct:.2f}%" if isinstance(mape_pct, (int, float)) else mape_pct,
                        "SMAPE": smape_val,
                        "MASE": mase_val,
                        "RMSE": rmse_val,
                        "Accuracy": accuracy_icon
                    })

                product_performance_df = pd.DataFrame(product_performance_data)
                st.dataframe(product_performance_df, hide_index=True, use_container_width=True)

                st.info("""
                **How Product-by-Product Selection Works (Backtesting):**
                - Each model is walk‑forward backtested per product (gap=0, h=6)
                - We compute WAPE, SMAPE, MASE, and RMSE; selection uses WAPE
                - Tie‑breaks: p75 WAPE → MASE → recent worst‑month error
                - Ineligible models (too few folds, MASE ≥ 1.0, unstable p95, or <5% better than Seasonal‑Naive) are excluded
                - The table shows diagnostic metrics for the selected model per product
                """)

        # === ADVANCED VALIDATION (if available) — placed after Download Results ===
        adv = st.session_state.get('advanced_validation_results')
        if adv:
            st.markdown("---")
            # TODO: Implement display_advanced_validation_results function
            st.info("Advanced validation results available but display function not implemented yet.")

        # === BACKTESTING DETAILS (product/model table) ===
        bt_all = st.session_state.get('backtesting_results', {}) or {}
        if bt_all:
            st.markdown("---")
            st.markdown("## 🧪 Backtesting Details")
            products_bt = sorted(list(bt_all.keys()))
            product_choice = st.selectbox("Select product", options=["All"] + products_bt, index=0, key="bt_details_product")

            rows = []
            for prod, models_map in bt_all.items():
                if product_choice != "All" and prod != product_choice:
                    continue
                if not isinstance(models_map, dict):
                    continue
                for model_name, res in models_map.items():
                    if not isinstance(res, dict):
                        continue
                    bt = res.get('backtesting_validation') or {}
                    if not isinstance(bt, dict):
                        continue
                    bt_err = None
                    try:
                        bt_err = bt.get('error')
                    except Exception:
                        bt_err = None
                    rows.append({
                        "Product": prod,
                        "Model": model_name,
                        "WAPE": f"{bt.get('recent_weighted_wape', bt.get('wape', bt.get('mape', float('nan')))):.3f}" if isinstance(bt.get('recent_weighted_wape', bt.get('wape', bt.get('mape', None))), (int, float)) else "N/A",
                        "SMAPE": f"{bt.get('smape', float('nan')):.3f}" if isinstance(bt.get('smape', None), (int, float)) else "N/A",
                        "MASE": f"{bt.get('mase', float('nan')):.3f}" if isinstance(bt.get('mase', None), (int, float)) else "N/A",
                        "RMSE": f"{bt.get('rmse', float('nan')):.0f}" if isinstance(bt.get('rmse', None), (int, float)) else "N/A",
                        "p75_WAPE": f"{bt.get('p75_wape', bt.get('p75_mape', float('nan'))):.3f}" if isinstance(bt.get('p75_wape', bt.get('p75_mape', None)), (int, float)) else "N/A",
                        "p95_WAPE": f"{bt.get('p95_wape', bt.get('p95_mape', float('nan'))):.3f}" if isinstance(bt.get('p95_wape', bt.get('p95_mape', None)), (int, float)) else "N/A",
                        "Folds": bt.get('iterations', bt.get('folds', 'N/A')),
                        "BacktestMonths": bt.get('backtest_period', 'N/A'),
                        "ValHorizon": bt.get('validation_horizon', 'N/A'),
                        "Why_NA/Failed": bt_err if isinstance(bt_err, str) else ("N/A" if bt.get('success', True) else "Failed without error text")
                    })

            if rows:
                df_bt = pd.DataFrame(rows)
                
                # Find the best (lowest) weighted WAPE for highlighting
                numeric_wapes = []
                for row in rows:
                    wape_str = row["WAPE"]
                    try:
                        if wape_str != "N/A":
                            numeric_wapes.append(float(wape_str))
                    except (ValueError, TypeError):
                        pass
                
                if numeric_wapes:
                    min_wape = min(numeric_wapes)
                    st.info(f"🏆 **Best Weighted WAPE**: {min_wape:.1%} across all models")
                
                st.dataframe(df_bt, use_container_width=True, hide_index=True)
            else:
                st.info("No backtesting rows available for the selected product.")

        # === DOWNLOAD RESULTS (moved to bottom) ===
        with st.expander("⬇️ Download Results (Best per Product • Backtesting)", expanded=False):
            try:
                # Use session state forecast_results which includes live conservatism adjustments
                df_dl = st.session_state.forecast_results.get("Best per Product (Backtesting)", pd.DataFrame())
                if not df_dl.empty and {'Product','Date','ACR'}.issubset(df_dl.columns):
                    if 'Type' in df_dl.columns:
                        mask = df_dl['Type'].isin(['forecast', 'non-compliant-forecast'])
                        df_dl = df_dl.loc[mask].copy()
                    base_cols = [c for c in ['Product','Date','Type','ACR','BestModel'] if c in df_dl.columns]
                    other_cols = [c for c in df_dl.columns if c not in base_cols]
                    df_dl = df_dl[base_cols + other_cols]
                    
                    # Generate filename with conservatism factor if not 100%
                    filename = "forecast_backtesting_best_per_product"
                    conservatism_factor = st.session_state.get('forecast_conservatism_used', 100)
                    if conservatism_factor != 100:
                        filename += f"_conservatism{conservatism_factor}pct"
                    filename += ".csv"
                    
                    csv_bytes = df_dl.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv_bytes, file_name=filename, mime="text/csv")
                else:
                    st.caption("No downloadable forecast found for Backtesting view.")
            except Exception:
                st.caption("Download unavailable.")

        # Technical diagnostics
        with st.expander("🔍 **Technical Diagnostics**", expanded=False):
            if diagnostic_messages:
                st.markdown("**Processing Summary:**")
                total_products = len(set(product for model_results in results.values()
                                    for product in model_results["Product"].unique())) if results else 0
                st.markdown(f"- Processed {total_products} data products")
                st.markdown(f"- Generated {len(diagnostic_messages)} diagnostic events")
                if best_models_per_product:
                    st.markdown(f"- Applied product-by-product model selection for {len(best_models_per_product)} products")

                # Categorize diagnostic messages by type
                st.markdown("**Detailed Diagnostics by Category:**")
                
                # Define categories and their icons
                categories = {
                    "🚀 **Pipeline & Setup**": ["🎯 Smart Forecasting:", "🔁 Using backtesting", "Processed", "Generated"],
                    "📊 **Data Processing**": ["Product", "Data processing", "Insufficient data", "outliers", "seasonality"],
                    "🔧 **Model Training**": ["✅", "❌", "MAPE", "Order", "Seasonal", "criterion"],
                    "🧪 **Backtesting & Validation**": ["backtesting", "validation", "insufficient data", "Backtesting failed"],
                    "⚙️ **Business Logic**": ["business adjustments", "growth", "market", "drift", "renewals"],
                    "⚠️ **Warnings & Errors**": ["⚠️", "❌", "Error", "Failed", "skipping", "not available"]
                }
                
                # Group messages by category
                categorized_messages = {cat: [] for cat in categories.keys()}
                uncategorized = []
                
                for msg in diagnostic_messages:
                    categorized = False
                    for cat, keywords in categories.items():
                        if any(keyword.lower() in msg.lower() for keyword in keywords):
                            categorized_messages[cat].append(msg)
                            categorized = True
                            break
                    if not categorized:
                        uncategorized.append(msg)
                
                # Display categorized messages
                for category, messages in categorized_messages.items():
                    if messages:
                        with st.expander(category, expanded=False):
                            for msg in messages:
                                # Determine message type and styling
                                if "✅" in msg or "Success" in msg:
                                    st.success(msg)
                                elif "❌" in msg or "Error" in msg or "Failed" in msg:
                                    st.error(msg)
                                elif "⚠️" in msg or "Warning" in msg:
                                    st.warning(msg)
                                elif "📈" in msg or "📊" in msg or "📉" in msg:
                                    st.info(msg)
                                elif "🔧" in msg or "⚙️" in msg:
                                    st.info(msg)
                                else:
                                    st.info(msg)
                
                # Show uncategorized messages if any
                if uncategorized:
                    with st.expander("📝 **Other Messages**", expanded=False):
                        for msg in uncategorized:
                            st.info(msg)
                
                # Add enhanced logging summary
                st.markdown("---")
                st.markdown("**📈 Enhanced Logging Summary:**")
                
                # Count by message type
                success_count = sum(1 for msg in diagnostic_messages if "✅" in msg)
                error_count = sum(1 for msg in diagnostic_messages if "❌" in msg)
                warning_count = sum(1 for msg in diagnostic_messages if "⚠️" in msg)
                info_count = sum(1 for msg in diagnostic_messages if "📊" in msg or "📈" in msg or "📉" in msg)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("✅ Success", success_count)
                with col2:
                    st.metric("❌ Errors", error_count)
                with col3:
                    st.metric("⚠️ Warnings", warning_count)
                with col4:
                    st.metric("📊 Info", info_count)
                
                # Show processing efficiency
                if total_products > 0:
                    efficiency = (success_count / (success_count + error_count)) * 100 if (success_count + error_count) > 0 else 0
                    st.metric("🎯 Processing Efficiency", f"{efficiency:.1f}%")

            # SARIMA technical details
            if "SARIMA" in results and sarima_params:
                st.markdown("**SARIMA Model Parameters:**")
                for product, param_data in sarima_params.items():
                    if len(param_data) == 4:
                        order, seas, criterion_value, criterion_type = param_data
                    else:
                        order, seas, criterion_value = param_data[:3]
                        criterion_type = "AIC"

                    if order is not None:
                        st.markdown(f"- **{product}**: Order {order}, Seasonal {seas}, {criterion_type}: {criterion_value:.2f}")

    # (Removed backtesting overview & forecast impact sections)
