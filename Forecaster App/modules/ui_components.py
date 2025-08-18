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
    """Clean enhanced MAPE summary (corruption removed)."""
    st.markdown("### Enhanced MAPE Breakdown")
    enhanced_rows = []
    for product, models in (backtesting_results or {}).items():
        for model, res in (models or {}).items():
            if isinstance(res, dict):
                enh = res.get('enhanced_analysis')
                if enh:
                    enhanced_rows.append({
                        'Product': product,
                        'Model': model,
                        'MAPE': f"{enh.get('mape', 0):.1%}",
                        'MAPE_Std': f"{enh.get('mape_std', 0):.1%}",
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
                'Mean_MAPE': st.column_config.TextColumn('Mean MAPE', help='Average MAPE across all folds'),
                'Std_MAPE': st.column_config.TextColumn('MAPE Std', help='Standard deviation showing stability'),
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


def display_seasonal_performance_analysis(advanced_validation_results):
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
                        'Best_Month_MAPE': f"{seasonal.get('best_month_mape', 0):.1%}",
                        'Worst_Month': seasonal.get('worst_month_name', 'N/A'),
                        'Worst_Month_MAPE': f"{seasonal.get('worst_month_mape', 0):.1%}",
                        'Best_Quarter': seasonal.get('best_quarter_name', 'N/A'),
                        'Worst_Quarter': seasonal.get('worst_quarter_name', 'N/A'),
                        'Total_Periods': seasonal.get('total_periods', 0)
                    })
    
    if seasonal_results:
        df = pd.DataFrame(seasonal_results)
        st.dataframe(
            df,
            column_config={
                'Best_Month': st.column_config.TextColumn('Best Month', help='Month with lowest average MAPE'),
                'Best_Month_MAPE': st.column_config.TextColumn('Best MAPE', help='MAPE for best performing month'),
                'Worst_Month': st.column_config.TextColumn('Worst Month', help='Month with highest average MAPE'),
                'Worst_Month_MAPE': st.column_config.TextColumn('Worst MAPE', help='MAPE for worst performing month'),
                'Best_Quarter': st.column_config.TextColumn('Best Quarter', help='Quarter with lowest average MAPE'),
                'Worst_Quarter': st.column_config.TextColumn('Worst Quarter', help='Quarter with highest average MAPE'),
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


def display_data_context_summary(debug_mode=False):
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
            elif recommendations.get('status') == 'good':
                st.success(f"{status_icon} **{status_title}**: {status_desc}")
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
        
        # Debug information (only show in debug mode)
        if debug_mode:
            st.markdown("---")
            st.markdown("#### 🐛 **Debug Information**")
            st.json({
                "data_context": data_context,
                "recommendations": recommendations,
                "data_quality": data_quality
            })
        


def display_backtesting_results(backtesting_results):
    """Display comprehensive backtesting validation results with explanations."""
    if not backtesting_results:
        return
    
    with st.expander("🔍 **Backtesting Results & Validation**", expanded=False):
        # Explain what backtesting is and how it works
        st.markdown("#### 📚 **What is Backtesting?**")
        st.info("""
        **Backtesting** validates forecasting models by testing them on historical data they haven't seen during training. 
        This gives you confidence that the models will perform well on future data.
        
        **How it works:**
        1. **Training Period**: Models learn from older data
        2. **Validation Period**: Models predict on recent historical data
        3. **Performance Measurement**: Compare predictions to actual values using MAPE
        4. **Model Selection**: Choose the best performing model per product
        """)
        
        # Show backtesting configuration
        st.markdown("#### ⚙️ **Backtesting Configuration**")
        config_info = {
            "Validation Method": "Simple Train/Test Split",
            "Training Data": "Historical data excluding last X months",
            "Test Period": "Last X months (as specified in sidebar)",
            "Performance Metric": "MAPE (Mean Absolute Percentage Error)",
            "Selection Strategy": "Best model per product based on backtesting MAPE"
        }
        
        for key, value in config_info.items():
            st.caption(f"**{key}**: {value}")
        
        # Summary table: backtesting MAPE per model/product
        st.markdown("#### 📊 **Backtesting Performance Results**")
        
        rows = []
        for product, models in backtesting_results.items():
            for model, res in models.items():
                if not isinstance(res, dict):
                    continue
                
                # Get backtesting results
                backtesting = res.get('backtesting_validation')
                if backtesting and backtesting.get('success'):
                    mape = backtesting.get('mape', 0)
                    backtest_months = backtesting.get('backtest_period', 0)
                    test_months = backtesting.get('test_months', 0)
                    
                    rows.append({
                        'Product': product, 
                        'Model': model, 
                        'Backtest MAPE%': round(float(mape)*100, 2),
                        'Backtest Period': f"{backtest_months} months",
                        'Test Period': f"{test_months} months"
                    })
        
        if rows:
            # Sort by MAPE (best performance first)
            df = pd.DataFrame(rows).sort_values(['Product', 'Backtest MAPE%'])
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            # Performance interpretation
            st.markdown("#### 🎯 **Performance Interpretation**")
            st.caption("""
            **MAPE Guidelines:**
            - **< 10%**: Excellent accuracy
            - **10-20%**: Good accuracy  
            - **20-30%**: Moderate accuracy
            - **> 30%**: Poor accuracy (consider different model)
            """)
            
            # Show backtesting insights
            st.markdown("#### 📈 **Backtesting Insights**")
            
            # Calculate success rate
            total_models = sum(len(models) for models in backtesting_results.values())
            successful_backtests = len(rows)
            success_rate = (successful_backtests / total_models) * 100 if total_models > 0 else 0
            
            if success_rate >= 80:
                st.success(f"✅ **High Success Rate**: {success_rate:.1f}% of models successfully backtested")
            elif success_rate >= 60:
                st.info(f"📊 **Good Success Rate**: {success_rate:.1f}% of models successfully backtested")
            else:
                st.warning(f"⚠️ **Low Success Rate**: {success_rate:.1f}% of models successfully backtested")
            
            # Model selection explanation
            st.markdown("#### 🏆 **Model Selection Process**")
            st.info("""
            **How models are selected:**
            1. **Backtesting Performance**: Models ranked by MAPE on validation data
            2. **Product-Specific Selection**: Best model chosen per product (not overall)
            3. **Fallback Strategy**: If backtesting fails, falls back to basic MAPE rankings
            4. **Hybrid Approach**: Combines backtesting results with MAPE rankings for robustness
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
    
    if product_data.empty:
        st.warning(f"No data available for product: {product}")
        return
    
    # Sort by date to ensure proper order
    product_data = product_data.sort_values("Date")
    
    # Display model info for this product
    col1, col2 = st.columns([3, 1])
    with col1:
    # Prefer explicit BestModel column if present (hybrid views like Standard / Backtesting / Raw)
        if 'BestModel' in product_data.columns and not product_data['BestModel'].isna().all():
            actual_model = str(product_data['BestModel'].iloc[0])
            st.caption(f"📊 Using **{actual_model}** model for this product")
        elif model_name == "Best per Product" and best_models_per_product and product in best_models_per_product:
            actual_model = best_models_per_product[product]
            st.caption(f"📊 Using **{actual_model}** model for this product")
        else:
            st.caption(f"📊 Using **{model_name}** model")
    
    with col2:
        if best_mapes_per_product and product in best_mapes_per_product:
            mape_val = best_mapes_per_product[product] * 100
            if mape_val <= 15:
                st.success(f"🎯 {mape_val:.1f}% MAPE")
            elif mape_val <= 25:
                st.info(f"📈 {mape_val:.1f}% MAPE")
            else:
                st.warning(f"⚠️ {mape_val:.1f}% MAPE")
    
    # Create the main forecast chart
    try:
        chart = create_forecast_chart(product_data, product, model_name)
        st.altair_chart(chart, use_container_width=True)
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
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Total Forecast", f"${total_forecast/1e6:.1f}M")
        with col2:
            st.metric("📈 Avg Monthly", f"${avg_monthly/1e6:.1f}M")
        with col3:
            months_count = len(forecast_data)
            st.metric("📅 Months", f"{months_count}")


def create_forecast_chart(data, product_name, model_name):
    """
    Create an Altair chart for forecast visualization.
    
    Args:
        data: DataFrame with forecast data
        product_name: Name of the product
        model_name: Name of the model
    
    Returns:
        Altair chart object
    """
    # Prepare data for charting
    chart_data = data.copy()
    chart_data['Date'] = pd.to_datetime(chart_data['Date'])
    
    # Create base chart
    base = alt.Chart(chart_data)
    
    # Historical data line
    historical = base.transform_filter(
        alt.datum.Type == 'actual'
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
        alt.datum.Type == 'forecast'
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
    
    # Non-compliant forecast line (if exists)
    noncompliant = base.transform_filter(
        alt.datum.Type == 'non-compliant-forecast'
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
    
    # Combine all layers
    chart = (historical + forecast + noncompliant).resolve_scale(
        color='independent'
    ).properties(
        title=f'{product_name} - {model_name} Forecast',
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
    download_data = results_dict[selected_model]
    
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
            "MAPE": f"{mape_value:.1%}",
            "MAPE_Raw": mape_value  # For sorting
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

    # Determine best model (after possible key rename) and normalize any legacy reference
    best_model = min(avg_mapes, key=avg_mapes.get)
    if best_model == "Best per Product (Raw)" and "Best per Product (Mix)" in results:
        best_model = "Best per Product (Mix)"
    best_model_display = best_model if best_model != "Best per Product (Raw)" else "Best per Product (Mix)"
    best_mape = avg_mapes[best_model] * 100

    # === COMBINED RESULTS SUMMARY (Summary + Key Metrics will render after active view resolved) ===
    st.markdown("## 📈 Results Summary")
    total_products = len({product for model_results in results.values() for product in model_results["Product"].unique()}) if results else 0
    confidence_level = "High" if best_mape <= 15 else ("Medium" if best_mape <= 25 else "Low")

    # Short model label mapper (Raw or Mix both -> Mix)
    def _short_model_label(name: str) -> str:
        mapping = {
            "Best per Product (Backtesting)": "Backtesting",
            "Best per Product (Standard)": "Standard",
            "Best per Product (Raw)": "Mix",
            "Best per Product (Mix)": "Mix"
        }
        return mapping.get(name, name)

    # Metrics will be displayed later once forecast selection & totals computed

    # Optional detailed comparison hidden by default
    # (Removed Standard vs Backtesting comparison expander)

    # (Executive Summary CTA download removed by request; download moved next to selection toggles below)

    # Restore model view toggles (exclude internal diagnostic variants to reduce confusion)
    composite_keys = [k for k in results.keys() if k.startswith("Best per Product (")]  # exclude raw base key
    label_explanations = {
        "Standard": "Fast basic validation (single split).",
        "Backtesting": "Deeper walk‑forward / CV driven per‑product picks (more robust).",
        "Mix": "Per‑product blend: chooses lower MAPE of Standard vs Backtesting for each product."
    }
    def _label_for(k: str) -> str:
        if k.endswith("(Standard)"): return "Standard"
        if k.endswith("(Backtesting)"): return "Backtesting"
        if k.endswith("(Mix)"): return "Mix"
        return k
    # Only surface composite variants except raw base
    # Include legacy Raw as fallback ordering if still present
    order_pref = ["Best per Product (Standard)", "Best per Product (Backtesting)", "Best per Product (Mix)", "Best per Product (Raw)"]
    composite_keys_sorted = [k for k in order_pref if k in composite_keys]
    active_key = None
    if composite_keys_sorted:
        default_key = best_model if best_model in composite_keys_sorted else composite_keys_sorted[0]
        labels = [_label_for(k) for k in composite_keys_sorted]
        selected_label = st.radio(
            "View mode",
            options=labels,
            index=labels.index(_label_for(default_key)),
            horizontal=True,
            help="Switch Standard (fast), Backtesting (robust), or Mix (per‑product blend)." 
        )
        key_map = { _label_for(k): k for k in composite_keys_sorted }
        active_key = key_map.get(selected_label, default_key)
        with st.expander("What do these modes mean?", expanded=False):
            for lbl in labels:
                if lbl in label_explanations:
                    st.markdown(f"**{lbl}:** {label_explanations[lbl]}")
    if not active_key:
        for pref in ["Best per Product (Mix)", "Best per Product (Backtesting)", "Best per Product (Standard)"]:
            if pref in results:
                active_key = pref
                break
        if not active_key:
            active_key = best_model  # ultimate fallback

    # Download button (always executed once active_key resolved)
    try:
        df_dl = results.get(active_key, pd.DataFrame())
        if not df_dl.empty and {'Product','Date','ACR'}.issubset(df_dl.columns):
            if 'Type' in df_dl.columns:
                mask = df_dl['Type'].isin(['forecast', 'non-compliant-forecast'])
                df_dl = df_dl.loc[mask].copy()
            base_cols = [c for c in ['Product','Date','Type','ACR','BestModel'] if c in df_dl.columns]
            other_cols = [c for c in df_dl.columns if c not in base_cols]
            df_dl = df_dl[base_cols + other_cols]
            csv_bytes = df_dl.to_csv(index=False).encode('utf-8')
            st.download_button(f"⬇️ Download CSV — {active_key}", csv_bytes, file_name="forecast.csv", mime="text/csv")
    except Exception:
        pass

    # Simple info about average MAPE for the active model key
    try:
        active_mape = st.session_state.forecast_mapes.get(active_key, np.nan)
        if np.isfinite(active_mape):
            st.caption(f"Average MAPE (active view): {active_mape*100:.1f}%")
    except Exception:
        pass

    # === KEY PERFORMANCE INDICATORS ===
    # (Removed standalone Key Metrics header; unified grid shown below)
    
    df_active = results.get(active_key, pd.DataFrame())
    # Helper slices
    if not df_active.empty and {'Product', 'Date', 'ACR'}.issubset(df_active.columns):
        is_fore = df_active.get('Type').isin(['forecast','non-compliant-forecast']) if 'Type' in df_active.columns else pd.Series([True]*len(df_active))
        is_hist = df_active.get('Type').eq('actual') if 'Type' in df_active.columns else pd.Series([False]*len(df_active))
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
        st.metric("Best MAPE", f"{best_mape:.1f}%")
    with metrics_cols[2]:
        st.metric("Confidence", confidence_level)
    with metrics_cols[3]:
        st.metric("Approach", _short_model_label(best_model_display))
    with metrics_cols[4]:
        st.metric("Total Forecast", f"${total_forecast/1e6:.1f}M")
    with metrics_cols[5]:
        st.metric("Avg MoM", f"{avg_mom:.1f}%" if np.isfinite(avg_mom) else "—", help="Average month-over-month growth")

    if mix_label:
        st.caption(f"Model Mix: {mix_label}")
    # Concise rationale line
    if st.session_state.get('model_avg_ranks'):
        best_rank = st.session_state.model_avg_ranks.get(best_model, float('inf'))
        st.caption(f"Selection: multi-metric rank score {best_rank:.1f}.")
    else:
        st.caption("Selection: lowest average MAPE.")
    st.markdown("---")

    # Product summary table (collapsible)
    with st.expander("📄 Products (selected mode)", expanded=False):
        rows = []
        if not df_active.empty:
            for product, grp in df_active[df_active['Type'].eq('forecast') if 'Type' in df_active.columns else df_active.index.isin(df_active.index)].groupby('Product'):
                total_p = float(grp['ACR'].sum())
                model_p = grp['BestModel'].iloc[0] if 'BestModel' in grp.columns and not grp['BestModel'].isna().all() else ''
                # per product YoY
                hist_p = df_hist[df_hist['Product'] == product] if not df_hist.empty else pd.DataFrame()
                hist12_p = _last_12_actual_sum(hist_p) if not hist_p.empty else 0.0
                fore_p = df_fore[df_fore['Product']==product]
                fore_sum12_p = _first_12_fore_sum(fore_p)
                MIN_BASE_P = 5e5
                yoy_p = ((fore_sum12_p / hist12_p) - 1.0) * 100.0 if hist12_p > MIN_BASE_P and fore_sum12_p > 0 else np.nan
                # per product avg MoM
                try:
                    agg_p = grp.groupby('Date')['ACR'].sum().sort_index()
                    mom_p = agg_p.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                    avg_mom_p = float(mom_p.mean() * 100.0) if not mom_p.empty else np.nan
                except Exception:
                    avg_mom_p = np.nan
                trend = "↑" if np.isfinite(avg_mom_p) and avg_mom_p > 0.5 else ("↓" if np.isfinite(avg_mom_p) and avg_mom_p < -0.5 else "→")
                # Drift badge (if drift applied to this product's active model forecast)
                drift_products = st.session_state.get('drift_applied_products', [])
                drift_badge = '🪄' if product in drift_products else ''
                rows.append({
                    'Product': product,
                    'Model': (model_p + (' ' + drift_badge if drift_badge else '')),
                    'ForecastTotal': total_p / 1e6,  # Convert to millions for better display
                    'YoY%': yoy_p,
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
                    'YoY%': st.column_config.NumberColumn('YoY %', format="%.1f%%", help='Vs last 12 months actual'),
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
            st.success("✅ **Multi-Metric Selection Applied:** Each product uses its best performing model across MAPE, SMAPE, MASE, and RMSE!")
            with st.expander("📊 How Multi-Metric Ranking Works", expanded=False):
                st.markdown("""
                **For each product, we evaluate models using 4 key metrics:**
                - **MAPE** (Mean Absolute Percentage Error): Overall accuracy as a percentage
                - **SMAPE** (Symmetric MAPE): Balanced accuracy that treats over/under-forecasts equally  
                - **MASE** (Mean Absolute Scaled Error): Accuracy relative to a naive seasonal forecast
                - **RMSE** (Root Mean Squared Error): Penalizes larger forecast errors more heavily
                
                **Selection Process:**
                1. Rank all models by each metric for each product
                2. Calculate average rank across all 4 metrics 
                3. Select the model with the best average ranking
                4. This gives more robust selection than any single metric alone
                """)
        else:
            st.success("✅ **Smart Selection Applied:** Each product uses its most accurate model!")
    
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
        default_key = best_model if best_model in ordered else ("Best per Product (Mix)" if "Best per Product (Mix)" in ordered else ordered[0])
        default_idx = ordered.index(default_key) if default_key in ordered else 0
        chart_model = st.selectbox(
            "📊 Select model to view",
            options=ordered,
            index=default_idx,
            key="model_choice_charts",
            help="View individual model outputs or one of the composite per‑product selections (Standard, Backtesting, Mix)."
        )
    with col2:
        if chart_model == best_model:
            if st.session_state.get('model_avg_ranks'):
                best_rank = st.session_state.model_avg_ranks.get(best_model, float('inf'))
                st.success(f"🏆 Best Model (Multi-Metric Rank: {best_rank:.1f})")
            else:
                st.success("🏆 Best Model")
        else:
            chart_mape = avg_mapes.get(chart_model, np.nan) * 100
            if not np.isnan(chart_mape):
                    # Show ranking info if available
                    if st.session_state.get('model_avg_ranks') and chart_model in st.session_state.model_avg_ranks:
                        chart_rank = st.session_state.model_avg_ranks[chart_model]
                        st.info(f"📈 MAPE: {chart_mape:.1f}% (Multi-Metric Rank: {chart_rank:.1f})")
                    else:
                        st.info(f"📈 MAPE: {chart_mape:.1f}%")
        
        # Use adjusted data if available, otherwise use original  
        if st.session_state.get('adjusted_forecast_results') is not None:
            chart_model_data = st.session_state.adjusted_forecast_results[chart_model]
            if any(adj != 0 for adj in st.session_state.get('product_adjustments_applied', {}).values()):
                st.info("📊 Showing adjusted forecasts based on your custom settings")
        else:
            chart_model_data = results[chart_model]
        
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

    st.markdown("---")

    # Quick product tabs for easy navigation
    unique_products = chart_model_data["Product"].unique()
    if len(unique_products) > 1:
        product_tabs = st.tabs([f"📊 {product}" for product in unique_products])

        for tab_obj, product in zip(product_tabs, unique_products):
            with tab_obj:
                display_product_forecast(chart_model_data, product, chart_model, best_models_per_product, best_mapes_per_product)
    else:
        # Single product - no tabs needed
        product = unique_products[0]
        st.markdown(f"### 📊 **{product}**")
        display_product_forecast(chart_model_data, product, chart_model, best_models_per_product, best_mapes_per_product)

    # === GROWTH ANALYSIS (OPTIONAL) ===
    with st.expander("📈 **Month-over-Month Growth Analysis**", expanded=False):
        st.markdown(f"Growth trends for **{chart_model}** model")

        # Show adjustment info if applied
        if st.session_state.get('product_adjustments_applied') and any(adj != 0 for adj in st.session_state.get('product_adjustments_applied', {}).values()):
            st.info("📊 Growth charts reflect your custom product adjustments")

        for product_mom in unique_products:
            df_product_mom = chart_model_data[chart_model_data["Product"] == product_mom].copy()
            dfm_growth = None  # Initialize to handle potential errors

            st.markdown(f"**📊 {product_mom}**")

            # Show adjustment info for this product if applied
            if st.session_state.get('product_adjustments_applied') and product_mom in st.session_state.product_adjustments_applied:
                adj_info = st.session_state.product_adjustments_applied[product_mom]
                if isinstance(adj_info, dict):
                    adj_pct = adj_info.get('percentage', 0)
                    start_display = adj_info.get('start_display', 'Unknown')
                    if adj_pct != 0:
                        if adj_pct > 0:
                            st.markdown(f"*📈 +{adj_pct}% growth from {start_display}*")
                        else:
                            st.markdown(f"*📉 {adj_pct}% haircut from {start_display}*")

            try:
                df_product_mom = df_product_mom.set_index("Date")
                # Build normalized MoM series including forecast
                norm_mom = normalized_monthly_by_days(df_product_mom, "ACR")
                mom_growth = norm_mom.pct_change().dropna()

                # Validate growth data
                if len(mom_growth) == 0:
                    st.warning(f"⚠️ Insufficient data for growth analysis: {product_mom}")
                    continue

                dfm_growth = (
                    mom_growth
                    .rename("growth")              # series → column name
                    .reset_index()                 # ['month','growth']
                    .rename(columns={"month": "Date"})
                )

                # Validate growth dataframe
                if len(dfm_growth) == 0 or dfm_growth['growth'].isna().all():
                    st.warning(f"⚠️ No valid growth data for {product_mom}")
                    continue

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
        choice = st.selectbox("Choose model for download", list(results.keys()),
                             index=list(results.keys()).index(best_model), key="download_model_choice")

        # Use adjusted results if available, otherwise use original
        if st.session_state.get('adjusted_forecast_results') is not None:
            download_data = st.session_state.adjusted_forecast_results[choice]
            if st.session_state.get('product_adjustments_applied'):
                st.success("📊 Download includes your custom adjustments!")
        else:
            download_data = results[choice]

        # Show yearly renewals overlay status
        if st.session_state.get('yearly_renewals_applied', False):
            st.success("📋 Download includes non-compliant Upfront RevRec data as separate line items!")

        buf2 = io.BytesIO()
        with pd.ExcelWriter(buf2, engine="openpyxl") as writer:
            download_data.to_excel(writer, index=False, sheet_name="Actuals_Forecast")
        buf2.seek(0)
        today = datetime.today().strftime("%Y%m%d")

        # Add adjustment indicators to filename
        filename_suffix = ""
        if st.session_state.get('adjusted_forecast_results') is not None:
            product_adjustments = st.session_state.get('product_adjustments_applied', {})
            adjustment_percentages = []
            for product, adj_info in product_adjustments.items():
                if isinstance(adj_info, dict):
                    adj_pct = adj_info.get('percentage', 0)
                    if adj_pct != 0:
                        adjustment_percentages.append(f"{adj_pct:+d}%")
                else:
                    if adj_info != 0:
                        adjustment_percentages.append(f"{adj_info:+d}%")

            if adjustment_percentages:
                filename_suffix += f"_adj{'-'.join(adjustment_percentages)}"

        if st.session_state.get('business_adjustments_applied', False):
            growth = st.session_state.get('business_growth_used', 0)
            if growth != 0:
                filename_suffix += f"_biz{growth:+d}%"

        if st.session_state.get('yearly_renewals_applied', False):
            filename_suffix += "_YearlyRenewals"

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
            any_adjustments = any(adj['percentage'] != 0 for adj in product_adjustments.values())

            if any_adjustments:
                st.info("🔧 Forecast adjustments are active. Download will include adjusted values.")

                # Create adjusted results for download
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
            else:
                # No adjustments, use original results
                if 'adjusted_forecast_results' in st.session_state:
                    del st.session_state['adjusted_forecast_results']
                if 'product_adjustments_applied' in st.session_state:
                    del st.session_state['product_adjustments_applied']

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
                    "Avg MAPE": f"{mape_pct:.2f}%",
                    "Multi-Metric Rank": rank_info,
                    "Status": accuracy_note
                })

            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, hide_index=True, use_container_width=True)

            # Show detailed metrics if available
            if st.session_state.get('product_smapes') or st.session_state.get('product_mases'):
                st.markdown("### 📈 All Metrics Summary")
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
                        "MAPE": f"{avg_mapes[model_name]*100:.2f}%",
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
                st.caption("📊 Models selected using **multi-metric ranking** across MAPE, SMAPE, MASE, and RMSE (lower values = better performance)")
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
                        "MAPE": f"{mape_pct:.2f}%" if isinstance(mape_pct, (int, float)) else mape_pct,
                        "SMAPE": smape_val,
                        "MASE": mase_val,
                        "RMSE": rmse_val,
                        "Accuracy": accuracy_icon
                    })

                product_performance_df = pd.DataFrame(product_performance_data)
                st.dataframe(product_performance_df, hide_index=True, use_container_width=True)

                st.info("""
                **How Product-by-Product Selection Works:**
                - Each model is tested on each data product individually
                - We calculate **all four metrics** (MAPE, SMAPE, MASE, RMSE) for each model-product combination  
                - Models are **ranked across all four metrics** for each specific product
                - The model with the **best average ranking across all metrics** for each product is selected
                - The table above shows **all calculated metrics** for the selected model per product
                - This multi-metric approach provides more robust model selection than MAPE alone
                """)

        # === ADVANCED VALIDATION (if available) — placed after Download Results ===
        adv = st.session_state.get('advanced_validation_results')
        if adv:
            st.markdown("---")
            display_advanced_validation_results(adv)

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

                st.markdown("**Detailed Diagnostics:**")
                for msg in diagnostic_messages:
                    st.info(msg)

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
