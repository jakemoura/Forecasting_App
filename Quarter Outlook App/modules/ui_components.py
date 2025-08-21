"""
UI Components for Outlook Forecaster

Streamlit UI components for displaying forecasts, charts, and analysis
for the Quarterly Outlook Forecaster application.
"""

import io
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
from datetime import datetime


def create_forecast_summary_table(forecasts, model_evaluation=None, wape_scores=None, backtesting_results=None):
    """
    Create a summary table of all forecast models with enhanced metrics.
    
    Args:
        forecasts: dict of forecast results
        model_evaluation: dict of enhanced scores (optional)
        wape_scores: dict of WAPE scores (optional)
        backtesting_results: dict with backtesting information (optional)
        
    Returns:
        pd.DataFrame: Summary table for display
    """
    summary_data = []
    
    for model_name, forecast_data in forecasts.items():
        row = {
            'Model': model_name,
            'Quarter Total': f"${forecast_data.get('quarter_total', 0):,.0f}",
            'Remaining Days Forecast': f"${forecast_data.get('remaining_total', 0):,.0f}",
            'Daily Average': f"${forecast_data.get('daily_avg', 0):,.0f}"
        }
        
        # Add WAPE score if available
        if wape_scores and model_name in wape_scores:
            wape = wape_scores[model_name]
            if wape == float('inf'):
                row['WAPE'] = 'N/A'
            else:
                row['WAPE'] = f"{wape:.1%}"  # Display as percentage
        elif model_name == 'Monthly Renewals':
            # Special case for Monthly Renewals - not a competing forecast model
            row['WAPE'] = 'Special Purpose'
        else:
            row['WAPE'] = 'N/A'
        
        # Add backtesting WAPE if available
        if backtesting_results and 'per_model_wape' in backtesting_results:
            bt_wape = backtesting_results['per_model_wape'].get(model_name)
            if bt_wape is not None and bt_wape != float('inf'):
                row['Backtest WAPE'] = f"{bt_wape:.1%}"
            else:
                row['Backtest WAPE'] = 'N/A'
        else:
            row['Backtest WAPE'] = 'N/A'
        
        # Add enhanced score if available
        if model_evaluation and model_name in model_evaluation:
            score = model_evaluation[model_name]
            if score == float('inf'):
                row['Enhanced Score'] = 'N/A'
            else:
                row['Enhanced Score'] = f"{score:.1f}"
        else:
            row['Enhanced Score'] = 'N/A'
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def create_forecast_visualization(quarter_data, forecasts, quarter_info, selected_model=None, best_model=None, 
                                backtesting_results=None, wape_scores=None):
    """
    Create an interactive forecast visualization using Altair with model selection and backtesting results.
    
    Args:
        quarter_data: pandas Series with historical data
        forecasts: dict of forecast results
        quarter_info: quarter information dict
        selected_model: str, specific model to display (if None, shows dropdown)
        best_model: str, the best performing model to auto-select
        backtesting_results: dict with backtesting validation information
        wape_scores: dict with WAPE scores for models
        
    Returns:
        tuple: (alt.Chart, model_selector_widget) - Chart object and model selector
    """
    import streamlit as st
    import altair as alt
    
    if not forecasts:
        return None, None
    
    # Model selector dropdown
    model_names = list(forecasts.keys())
    
    # Auto-select the best model if provided
    if selected_model is None:
        if best_model and best_model in model_names:
            default_index = model_names.index(best_model)
        else:
            default_index = 0
            
        selected_model = st.selectbox(
            "📊 Select Forecast Model to Display:",
            options=model_names,
            index=default_index,
            help=f"Choose which forecasting model to visualize. Best model: {best_model}" if best_model else "Choose which forecasting model to visualize"
        )
    
    # Get the selected forecast
    selected_forecast = forecasts.get(selected_model, list(forecasts.values())[0])
    
    # Prepare historical data
    hist_df = pd.DataFrame({
        'Date': quarter_data.index,
        'Value': quarter_data.values,
        'Type': 'Historical',
        'Model': 'Actual Data'
    })
    
    # Prepare forecast data
    forecast_values = selected_forecast.get('forecast', [])
    if len(forecast_values) > 0:
        # Create future dates
        last_date = quarter_data.index.max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=len(forecast_values),
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Value': forecast_values,
            'Type': 'Forecast',
            'Model': selected_model
        })
        
        # Combine historical and forecast data
        combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
    else:
        combined_df = hist_df
    
    # Identify spike days if available
    spike_days_df = pd.DataFrame()
    if 'spike_analysis' in selected_forecast:
        spike_analysis = selected_forecast['spike_analysis']
        if spike_analysis.get('has_spikes', False):
            spike_dates = spike_analysis.get('spike_days', [])
            if len(spike_dates) > 0:
                # Convert spike dates to ensure compatibility
                try:
                    # Ensure we have datetime objects
                    if isinstance(spike_dates[0], str):
                        spike_dates = pd.to_datetime(spike_dates)
                    
                    # Filter spike dates that are in our data and get values
                    historical_spike_dates = []
                    spike_values = []
                    
                    for spike_date in spike_dates:
                        try:
                            if spike_date in quarter_data.index:
                                historical_spike_dates.append(spike_date)
                                spike_values.append(quarter_data.loc[spike_date])
                        except (KeyError, TypeError):
                            # Skip dates that can't be found or compared
                            continue
                    
                    if historical_spike_dates:
                        spike_days_df = pd.DataFrame({
                            'Date': historical_spike_dates,
                            'Value': spike_values,
                            'Type': 'Spike',
                            'Model': 'Detected Spikes'
                        })
                except Exception as e:
                    # If spike processing fails, continue without spikes
                    spike_days_df = pd.DataFrame()
    
    # Create the base chart
    base = alt.Chart(combined_df).add_selection(
        alt.selection_interval(bind='scales')
    )
    
    # Historical data line
    historical_line = base.mark_line(
        color='steelblue',
        strokeWidth=2
    ).transform_filter(
        alt.datum.Type == 'Historical'
    ).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Value:Q', title='Daily Value ($)', scale=alt.Scale(zero=False)),
        tooltip=['Date:T', 'Value:Q', 'Model:N']
    )
    
    # Forecast data line
    forecast_line = base.mark_line(
        color='orange',
        strokeWidth=2,
        strokeDash=[5, 5]
    ).transform_filter(
        alt.datum.Type == 'Forecast'
    ).encode(
        x=alt.X('Date:T'),
        y=alt.Y('Value:Q'),
        tooltip=['Date:T', 'Value:Q', 'Model:N']
    )
    
    # Spike markers (if any detected)
    chart_layers = [historical_line, forecast_line]
    if not spike_days_df.empty:
        # Large red circles for spikes
        spike_circles = alt.Chart(spike_days_df).mark_circle(
            size=200,
            color='red',
            stroke='darkred',
            strokeWidth=3,
            opacity=0.9
        ).encode(
            x=alt.X('Date:T'),
            y=alt.Y('Value:Q'),
            tooltip=['Date:T', 'Value:Q', alt.Tooltip('Model:N', title='Type')]
        )
        
        # Add star markers on top for extra visibility
        spike_stars = alt.Chart(spike_days_df).mark_point(
            shape='cross',
            size=150,
            color='yellow',
            stroke='black',
            strokeWidth=2
        ).encode(
            x=alt.X('Date:T'),
            y=alt.Y('Value:Q'),
            tooltip=['Date:T', 'Value:Q', alt.Tooltip('Model:N', title='Type')]
        )
        
        chart_layers.extend([spike_circles, spike_stars])
    # Prepare backtesting validation markers and trends (if available)
    if backtesting_results and backtesting_results.get('validation_details'):
        validation_details = backtesting_results['validation_details'].get(selected_model, {})
        
        # Direct access to quarterly validation_results without conversion
        if 'validation_results' in validation_details:
            quarterly_results = validation_details['validation_results']
            
            val_markers_data = []
            prediction_segments = []
            
            for i, result in enumerate(quarterly_results):
                try:
                    # Get validation period info
                    val_start_date = pd.to_datetime(result.get('val_start'))
                    val_end_date = pd.to_datetime(result.get('val_end'))
                    forecast_values = result.get('forecast_values', [])
                    fold_num = result.get('fold', i + 1)
                    
                    # Add validation start marker (green triangle)
                    if val_start_date in quarter_data.index:
                        val_markers_data.append({
                            'Date': val_start_date,
                            'Value': quarter_data.loc[val_start_date],
                            'Type': 'Validation Start',
                            'Model': f'Fold {fold_num} ({len(forecast_values)}d prediction)'
                        })
                    
                    # Add prediction trends (purple dotted lines)
                    if forecast_values and len(forecast_values) > 0:
                        # Create dates for forecast period
                        forecast_dates = pd.date_range(val_start_date, periods=len(forecast_values), freq='D')
                        
                        for j, (pred_date, pred_value) in enumerate(zip(forecast_dates, forecast_values)):
                            try:
                                pred_value = float(pred_value)
                                if not pd.isna(pred_value) and pred_value != float('inf'):
                                    prediction_segments.append({
                                        'Date': pred_date,
                                        'Value': pred_value,
                                        'Type': 'Backtesting Prediction',
                                        'Model': f'Fold {fold_num} Prediction',
                                        'Fold': str(fold_num)
                                    })
                            except (ValueError, TypeError):
                                continue
                                
                except (KeyError, ValueError, TypeError) as e:
                    continue
            
            # Add validation markers (green triangles)
            if val_markers_data:
                validation_df = pd.DataFrame(val_markers_data)
                validation_markers = alt.Chart(validation_df).mark_point(
                    shape='triangle-up',
                    size=120,
                    color='green',
                    stroke='darkgreen',
                    strokeWidth=2,
                    opacity=0.8
                ).encode(
                    x=alt.X('Date:T'),
                    y=alt.Y('Value:Q'),
                    tooltip=['Date:T', 'Value:Q', alt.Tooltip('Model:N', title='Validation Fold')]
                )
                chart_layers.append(validation_markers)
            
            # Add prediction lines (purple dotted)
            if prediction_segments:
                predictions_df = pd.DataFrame(prediction_segments)
                prediction_lines = alt.Chart(predictions_df).mark_line(
                    color='purple',
                    strokeWidth=4,
                    strokeDash=[8, 4],
                    opacity=1.0
                ).encode(
                    x=alt.X('Date:T'),
                    y=alt.Y('Value:Q'),
                    detail=alt.Detail('Fold:N'),
                    tooltip=['Date:T', 'Value:Q', alt.Tooltip('Model:N', title='Validation Prediction')]
                )
                chart_layers.append(prediction_lines)
        
        # Handle enhanced/simple backtesting format (validation_periods)
        elif 'validation_periods' in validation_details:
            validation_periods = validation_details['validation_periods']
            
            val_markers_data = []
            prediction_segments = []
            
            for period in validation_periods:
                try:
                    val_start = pd.to_datetime(period['val_start'])
                    forecast_values = period.get('forecast_values', [])
                    fold_num = period.get('iteration', 1)
                    
                    # Add validation marker
                    if val_start in quarter_data.index:
                        val_markers_data.append({
                            'Date': val_start,
                            'Value': quarter_data.loc[val_start],
                            'Type': 'Validation Start',
                            'Model': f'Fold {fold_num}'
                        })
                    
                    # Add predictions
                    if forecast_values and len(forecast_values) > 0:
                        val_end = pd.to_datetime(period['val_end'])
                        pred_dates = pd.date_range(val_start, val_end, freq='D')[:len(forecast_values)]
                        
                        for pred_date, pred_value in zip(pred_dates, forecast_values):
                            try:
                                prediction_segments.append({
                                    'Date': pred_date,
                                    'Value': float(pred_value),
                                    'Type': 'Backtesting Prediction',
                                    'Model': f'Fold {fold_num} Prediction',
                                    'Fold': str(fold_num)
                                })
                            except (ValueError, TypeError):
                                continue
                                
                except (KeyError, ValueError, TypeError):
                    continue
            
            # Add charts
            if val_markers_data:
                validation_df = pd.DataFrame(val_markers_data)
                validation_markers = alt.Chart(validation_df).mark_point(
                    shape='triangle-up', size=120, color='green', opacity=0.8
                ).encode(x=alt.X('Date:T'), y=alt.Y('Value:Q'), tooltip=['Date:T', 'Value:Q'])
                chart_layers.append(validation_markers)
            
            if prediction_segments:
                predictions_df = pd.DataFrame(prediction_segments)
                prediction_lines = alt.Chart(predictions_df).mark_line(
                    color='purple', strokeWidth=4, strokeDash=[8, 4], opacity=1.0
                ).encode(
                    x=alt.X('Date:T'), y=alt.Y('Value:Q'), detail=alt.Detail('Fold:N'),
                    tooltip=['Date:T', 'Value:Q']
                )
                chart_layers.append(prediction_lines)
    
    # Build title with backtesting information
    base_title = f"Daily Data & Forecast - {quarter_info['quarter_name']} ({selected_model})"
    
    # Add performance information to title
    if wape_scores and selected_model in wape_scores:
        standard_wape = wape_scores[selected_model]
        if standard_wape != float('inf'):
            base_title += f" | Standard WAPE: {standard_wape:.1%}"
    
    if backtesting_results and backtesting_results.get('per_model_wape'):
        bt_wape = backtesting_results['per_model_wape'].get(selected_model)
        if bt_wape is not None and bt_wape != float('inf'):
            base_title += f" | Backtesting WAPE: {bt_wape:.1%}"
            
            # Add validation details and method info
            validation_details = backtesting_results.get('validation_details', {}).get(selected_model, {})
            iterations = validation_details.get('iterations', 0)
            method_used = backtesting_results.get('method_used', '')
            
            if iterations > 0:
                if 'quarterly' in method_used.lower():
                    base_title += f" | ✓ {iterations} Quarterly Validation Folds"
                else:
                    base_title += f" | ✓ {iterations} Recent Validation Folds"
    
    # Combine all chart layers
    chart = alt.layer(*chart_layers).resolve_scale(
        color='independent'
    ).properties(
        width=700,
        height=350,
        title=base_title
    )
    
    # Add updated legend explanation below chart
    st.caption(
        "📊 **Chart Legend:** "
        "Blue line = Historical data | "
        "Orange dashed = Forecast | "
        "Red circles = Monthly renewal spikes | "
        "Green triangles = Backtesting validation points | "
        "Purple dotted = Backtesting prediction trends"
    )
    
    return chart, selected_model


def create_progress_indicator(quarter_progress):
    """
    Create a visual progress indicator for the quarter.
    
    Args:
        quarter_progress: dict with days_completed, days_remaining, total_days
    """
    completion_pct = quarter_progress['completion_pct']
    
    # Create columns for layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Days Completed",
            quarter_progress['days_completed'],
            delta=None
        )
    
    with col2:
        st.metric(
            "Days Remaining", 
            quarter_progress['days_remaining'],
            delta=None
        )
    
    with col3:
        st.metric(
            "Quarter Progress",
            f"{completion_pct:.1f}%",
            delta=None
        )
    
    # Progress bar
    st.progress(completion_pct / 100.0)


def display_model_comparison(forecasts, model_evaluation, wape_scores=None, backtesting_results=None):
    """
    Display a detailed comparison of forecast models with enhanced metrics.
    
    Args:
        forecasts: dict of forecast results
        model_evaluation: dict of enhanced scores
        wape_scores: dict of WAPE scores (optional)
        backtesting_results: dict with backtesting information (optional)
    """
    st.subheader("🔍 Enhanced Model Performance Comparison")
    
    if not forecasts:
        st.warning("No forecasts available for comparison.")
        return
    
    # Create detailed comparison table
    comparison_data = []
    for model_name, forecast_data in forecasts.items():
        # Enhanced score
        enhanced_score = model_evaluation.get(model_name, float('inf'))
        
        # WAPE score
        wape = wape_scores.get(model_name, float('inf')) if wape_scores else float('inf')
        
        # Backtesting WAPE
        bt_wape = None
        if backtesting_results and 'per_model_wape' in backtesting_results:
            bt_wape = backtesting_results['per_model_wape'].get(model_name)
        
        if model_name == 'Monthly Renewals':
            # Special case for Monthly Renewals - not a competing forecast model
            wape_str = 'Special Purpose'
            bt_wape_str = 'Special Purpose'
            score_str = 'Special Purpose'
            sort_score = 999  # Put it at the bottom of sort order
        else:
            # WAPE string
            if wape != float('inf'):
                wape_str = f"{wape:.1%}"
            else:
                wape_str = 'N/A'
            
            # Backtesting WAPE string
            if bt_wape is not None and bt_wape != float('inf'):
                bt_wape_str = f"{bt_wape:.1%}"
            else:
                bt_wape_str = 'N/A'
            
            # Enhanced score string
            if enhanced_score != float('inf'):
                score_str = f"{enhanced_score:.1f}"
                sort_score = enhanced_score
            else:
                score_str = 'N/A'
                sort_score = 999
        
        comparison_data.append({
            'Model': model_name,
            'Quarter Total': forecast_data.get('quarter_total', 0),
            'Daily Average': forecast_data.get('daily_avg', 0),
            'WAPE': wape_str,
            'Backtest WAPE': bt_wape_str,
            'Enhanced Score': score_str,
            'Sort Score': sort_score
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by Enhanced Score (lower is better)
    df_sorted = df.sort_values('Sort Score')
    
    # Display table
    display_df = df_sorted[['Model', 'Quarter Total', 'Daily Average', 'WAPE', 'Backtest WAPE', 'Enhanced Score']].copy()
    display_df['Quarter Total'] = display_df['Quarter Total'].apply(lambda x: f"${x:,.0f}")
    display_df['Daily Average'] = display_df['Daily Average'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Highlight best model
    if len(df_sorted) > 0:
        best_model = df_sorted.iloc[0]['Model']
        best_score = df_sorted.iloc[0]['Enhanced Score']
        best_wape = df_sorted.iloc[0]['WAPE']
        st.success(f"🏆 **Best Model:** {best_model} (WAPE: {best_wape}, Enhanced Score: {best_score})")
        
        # Show backtesting information if available
        if backtesting_results and backtesting_results.get('best_model'):
            bt_best = backtesting_results['best_model']
            bt_method = backtesting_results.get('method_used', 'enhanced-walk-forward')
            if bt_best != best_model:
                st.info(f"🔄 **Backtesting Best:** {bt_best} (Method: {bt_method})")
            else:
                st.info(f"✅ **Confirmed by Backtesting:** {bt_best} (Method: {bt_method})")


def display_backtesting_details(backtesting_results):
    """
    Display detailed backtesting validation results.
    
    Args:
        backtesting_results: dict with backtesting information
    """
    if not backtesting_results or not backtesting_results.get('validation_details'):
        return
    
    st.subheader("🔄 Backtesting Validation Details")
    
    validation_details = backtesting_results['validation_details']
    method_used = backtesting_results.get('method_used', 'unknown')
    
    st.info(f"**Validation Method:** {method_used}")
    
    # Create summary table of validation results
    validation_data = []
    for model_name, details in validation_details.items():
        if isinstance(details, dict) and 'error' not in details:
            # Handle different validation method key naming
            if 'weighted_remaining_wape' in details:  # Quarterly backtesting
                mean_wape = details.get('mean_wape', 0)
                p75_wape = details.get('p75_wape', 0) 
                recent_weighted = details.get('weighted_remaining_wape', 0)
            else:  # Enhanced/simple methods
                mean_wape = details.get('mean_wape', 0)
                p75_wape = details.get('p75_wape', 0)
                recent_weighted = details.get('recent_weighted_wape', 0)
                
            row = {
                'Model': model_name,
                'Mean WAPE': f"{mean_wape:.1%}",
                'P75 WAPE': f"{p75_wape:.1%}",
                'Recent Weighted WAPE': f"{recent_weighted:.1%}",
                'Iterations': details.get('iterations', 0)
            }
            validation_data.append(row)
    
    if validation_data:
        validation_df = pd.DataFrame(validation_data)
        st.dataframe(validation_df, use_container_width=True)
        
        st.caption("📊 **Metrics Explanation:**")
        st.caption("• **Mean WAPE**: Average error across all validation folds")
        st.caption("• **P75 WAPE**: 75th percentile error (robustness indicator)")
        st.caption("• **Recent Weighted WAPE**: Recent folds weighted more heavily")
        st.caption("• **Iterations**: Number of successful validation folds")
    else:
        st.warning("⚠️ No detailed validation results available.")


def create_confidence_intervals_chart(forecasts, quarter_info):
    """
    Create a chart showing confidence intervals across models.
    
    Args:
        forecasts: dict of forecast results
        quarter_info: quarter information dict
        
    Returns:
        alt.Chart: Confidence intervals visualization
    """
    if not forecasts:
        return None
    
    # Extract quarter totals from all models
    quarter_totals = [f.get('quarter_total', 0) for f in forecasts.values()]
    model_names = list(forecasts.keys())
    
    # Calculate statistics
    median_forecast = np.median(quarter_totals)
    q25 = np.percentile(quarter_totals, 25)
    q75 = np.percentile(quarter_totals, 75)
    min_forecast = min(quarter_totals)
    max_forecast = max(quarter_totals)
    
    # Create data for chart
    chart_data = pd.DataFrame({
        'Model': model_names,
        'Quarter_Total': quarter_totals
    })
    
    # Box plot showing distribution
    box_plot = alt.Chart(chart_data).mark_boxplot().encode(
        y=alt.Y('Quarter_Total:Q', title='Quarter Total Forecast ($)'),
        x=alt.value(50)  # Center the box plot
    ).properties(
        width=200,
        height=300,
        title=f"Forecast Distribution - {quarter_info['quarter_name']}"
    )
    
    return box_plot


def create_excel_download(forecasts_data, analysis_date, filename, conservatism_factor=1.0, capacity_factor=1.0):
    """
    Create Excel file with all forecast results for download.
    
    Args:
        forecasts_data: dict with forecast results by product
        analysis_date: datetime of analysis
        filename: str, filename for download
        conservatism_factor: float, conservatism adjustment factor (default 1.0 = no adjustment)
        capacity_factor: float, capacity constraint factor (default 1.0 = no adjustment)
        
    Returns:
        bytes: Excel file content or None if creation fails
    """
    try:
        output = io.BytesIO()
        
        # Import adjustment functions for applying to forecast data
        from .quarterly_forecasting import apply_conservatism_adjustment_to_forecast, apply_capacity_adjustment_to_forecast
        from .fiscal_calendar import get_fiscal_quarter_info
        from datetime import datetime, timedelta
        
        # Prepare summary data with adjustments applied
        summary_data = []
        
        for product, data in forecasts_data.items():
            if 'forecast' in data:
                forecast_result = data['forecast'].copy()  # Work with a copy
                
                # Apply adjustments in the same order as the main app
                # 1. Apply conservatism adjustment first
                if conservatism_factor != 1.0:
                    forecast_result = apply_conservatism_adjustment_to_forecast(forecast_result, conservatism_factor)
                
                # 2. Apply capacity adjustment second (after conservatism)
                if capacity_factor != 1.0:
                    forecast_result = apply_capacity_adjustment_to_forecast(forecast_result, capacity_factor)
                
                # Get best model and its adjusted forecast
                best_model = forecast_result.get('best_model', 'Unknown')
                quarter_total = forecast_result.get('summary', {}).get('quarter_total', 0)
                actual_to_date = forecast_result.get('actual_to_date', 0)
                remaining_forecast = quarter_total - actual_to_date
                daily_avg = forecast_result.get('summary', {}).get('daily_run_rate', 0)
                
                # Quarter progress info
                progress = forecast_result.get('quarter_progress', {})
                days_completed = progress.get('days_completed', 0)
                total_days = progress.get('total_days', 0)
                completion_pct = progress.get('completion_pct', 0)
                
                summary_data.append({
                    'Product': product,
                    'Best_Model': best_model,
                    'Quarter_Total': quarter_total,
                    'Actual_to_Date': actual_to_date,
                    'Remaining_Forecast': remaining_forecast,
                    'Daily_Run_Rate': daily_avg,
                    'Days_Completed': days_completed,
                    'Total_Days': total_days,
                    'Completion_Percent': completion_pct,
                    'Analysis_Date': analysis_date.strftime('%Y-%m-%d') if analysis_date else 'Unknown',
                    'Conservatism_Factor': conservatism_factor,
                    'Capacity_Factor': capacity_factor
                })
        
        if not summary_data:
            return None
            
        # Create Excel file with openpyxl
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Monthly breakdown sheet - combine all products
            monthly_data = []
            
            for product, data in forecasts_data.items():
                if 'forecast' in data and 'data' in data:
                    forecast_result = data['forecast'].copy()
                    historical_data = data['data']  # This is the actual historical data
                    
                    # Apply adjustments
                    if conservatism_factor != 1.0:
                        forecast_result = apply_conservatism_adjustment_to_forecast(forecast_result, conservatism_factor)
                    if capacity_factor != 1.0:
                        forecast_result = apply_capacity_adjustment_to_forecast(forecast_result, capacity_factor)
                    
                    # Get quarter info
                    quarter_info = forecast_result.get('quarter_info', get_fiscal_quarter_info(analysis_date))
                    quarter_start = quarter_info['quarter_start']
                    quarter_end = quarter_info['quarter_end']
                    
                    # Filter historical data to current quarter
                    quarter_mask = (historical_data.index >= quarter_start) & (historical_data.index <= quarter_end)
                    quarter_historical = historical_data[quarter_mask]
                    
                    # Group by month and calculate actuals
                    monthly_actuals = quarter_historical.groupby(quarter_historical.index.to_period('M')).sum()
                    
                    # Get the best forecast and forecasts dict
                    best_model = forecast_result.get('best_model', 'Unknown')
                    forecasts = forecast_result.get('forecasts', {})
                    
                    # Calculate monthly forecast splits
                    quarter_progress = forecast_result.get('quarter_progress', {})
                    days_remaining = quarter_progress.get('days_remaining', 0)
                    
                    if days_remaining > 0 and best_model in forecasts:
                        # Get the forecast values for the best model
                        best_forecast = forecasts[best_model]
                        remaining_total = best_forecast.get('remaining_total', 0)
                        
                        # Calculate how to split remaining forecast across months
                        data_end_date = quarter_historical.index.max() if len(quarter_historical) > 0 else analysis_date
                        
                        # Create future dates for remaining period
                        future_dates = pd.date_range(
                            start=data_end_date + timedelta(days=1),
                            end=quarter_end,
                            freq='D'
                        )
                        
                        # Group future dates by month and calculate proportional splits
                        if len(future_dates) > 0:
                            future_monthly = future_dates.to_series().groupby(future_dates.to_period('M')).count()
                            
                            # Split forecast proportionally by business days in each month
                            for month_period, days_in_month in future_monthly.items():
                                proportion = days_in_month / len(future_dates)
                                monthly_forecast = remaining_total * proportion
                                
                                # Check if we have actuals for this month
                                month_actual = monthly_actuals.get(month_period, 0)
                                month_total = month_actual + monthly_forecast
                                
                                # Convert period to datetime for formatting
                                month_datetime = pd.to_datetime(str(month_period))
                                
                                monthly_data.append({
                                    'Product': product,
                                    'Month': str(month_period),
                                    'Month_Name': month_datetime.strftime('%B %Y'),
                                    'Actuals': month_actual,
                                    'Forecast': monthly_forecast,
                                    'Total': month_total,
                                    'Best_Model': best_model,
                                    'Days_Actual': len(quarter_historical[quarter_historical.index.to_period('M') == month_period]),
                                    'Days_Forecast': days_in_month
                                })
                    
                    # Add months that have only actuals (no forecast needed)
                    for month_period, month_actual in monthly_actuals.items():
                        # Check if we already added this month above
                        existing_row = next((row for row in monthly_data 
                                           if row['Product'] == product and row['Month'] == str(month_period)), None)
                        if not existing_row:
                            # Convert period to datetime for formatting
                            month_datetime = pd.to_datetime(str(month_period))
                            
                            monthly_data.append({
                                'Product': product,
                                'Month': str(month_period),
                                'Month_Name': month_datetime.strftime('%B %Y'),
                                'Actuals': month_actual,
                                'Forecast': 0,
                                'Total': month_actual,
                                'Best_Model': best_model,
                                'Days_Actual': len(quarter_historical[quarter_historical.index.to_period('M') == month_period]),
                                'Days_Forecast': 0
                            })
            
            # Create monthly breakdown sheet
            if monthly_data:
                monthly_df = pd.DataFrame(monthly_data)
                monthly_df = monthly_df.sort_values(['Product', 'Month'])
                monthly_df.to_excel(writer, sheet_name='Monthly_Breakdown', index=False)
            else:
                # Create an empty sheet as placeholder
                empty_df = pd.DataFrame(columns=['Product', 'Month', 'Month_Name', 'Actuals', 'Forecast', 'Total', 'Best_Model', 'Days_Actual', 'Days_Forecast'])
                empty_df.to_excel(writer, sheet_name='Monthly_Breakdown', index=False)
            
            # Individual product details with adjustments applied
            for product, data in forecasts_data.items():
                if 'forecast' in data and 'forecasts' in data['forecast']:
                    forecast_result = data['forecast'].copy()  # Work with a copy
                    
                    # Apply adjustments in the same order as the main app
                    # 1. Apply conservatism adjustment first
                    if conservatism_factor != 1.0:
                        forecast_result = apply_conservatism_adjustment_to_forecast(forecast_result, conservatism_factor)
                    
                    # 2. Apply capacity adjustment second (after conservatism)
                    if capacity_factor != 1.0:
                        forecast_result = apply_capacity_adjustment_to_forecast(forecast_result, capacity_factor)
                    
                    forecasts = forecast_result['forecasts']
                    model_evaluation = forecast_result.get('model_evaluation', {})
                    quarter_progress = forecast_result.get('quarter_progress', {})
                    data_period = forecast_result.get('data_period', {})
                    
                    model_data = []
                    for model_name, forecast in forecasts.items():
                        mape_score = model_evaluation.get(model_name, 'N/A')
                        
                        model_data.append({
                            'Model': model_name,
                            'Quarter_Total': forecast.get('quarter_total', 0),
                            'Daily_Average': forecast.get('daily_avg', 0),
                            'Remaining_Total': forecast.get('remaining_total', 0),
                            'MAPE_Score': mape_score
                        })
                    
                    if model_data:
                        model_df = pd.DataFrame(model_data)
                        # Clean sheet name for Excel (max 31 chars, no special chars)
                        clean_name = ''.join(c for c in product if c.isalnum() or c in (' ', '-', '_'))[:31]
                        model_df.to_excel(writer, sheet_name=clean_name, index=False)
                        
                        # Add daily numbers for each model (adjusted values) with dates and actuals
                        daily_sheet_name = f"{clean_name}_Daily"[:31]
                        
                        # Get historical data for this product
                        if 'data' in data:
                            historical_data = data['data']
                            
                            # Get quarter info
                            quarter_info = forecast_result.get('quarter_info', get_fiscal_quarter_info(analysis_date))
                            quarter_start = quarter_info['quarter_start']
                            quarter_end = quarter_info['quarter_end']
                            
                            # Filter historical data to current quarter
                            quarter_mask = (historical_data.index >= quarter_start) & (historical_data.index <= quarter_end)
                            quarter_historical = historical_data[quarter_mask]
                            
                            # Prepare daily forecast data with dates and actuals
                            days_remaining = quarter_progress.get('days_remaining', 0)
                            if days_remaining > 0:
                                daily_data = {}
                                
                                # Get the last date with actual data
                                last_actual_date = quarter_historical.index.max() if len(quarter_historical) > 0 else analysis_date
                                
                                # Create future dates for forecast period
                                future_dates = pd.date_range(
                                    start=last_actual_date + timedelta(days=1),
                                    end=quarter_end,
                                    freq='D'
                                )
                                
                                # Limit to the expected remaining days
                                future_dates = future_dates[:days_remaining]
                                
                                daily_data['Date'] = [date.strftime('%Y-%m-%d') for date in future_dates]
                                daily_data['Actuals'] = [0] * len(future_dates)  # Future dates have no actuals
                                
                                # Add adjusted forecast values for each model
                                for model_name, forecast in forecasts.items():
                                    forecast_values = forecast.get('forecast', [])
                                    if hasattr(forecast_values, '__len__') and len(forecast_values) == days_remaining:
                                        # Convert numpy array to list if needed
                                        if hasattr(forecast_values, 'tolist'):
                                            forecast_values = forecast_values.tolist()
                                        daily_data[f"{model_name}_Forecast"] = forecast_values[:len(future_dates)]
                                    else:
                                        # Fallback to adjusted daily average for consistent days
                                        daily_avg = forecast.get('daily_avg', 0)
                                        daily_data[f"{model_name}_Forecast"] = [daily_avg] * len(future_dates)
                                
                                # Also add historical data with actuals
                                if len(quarter_historical) > 0:
                                    # Create historical data rows
                                    historical_rows = []
                                    for date, actual_value in quarter_historical.items():
                                        historical_rows.append({
                                            'Date': date.strftime('%Y-%m-%d'),
                                            'Actuals': actual_value,
                                            **{f"{model_name}_Forecast": 0 for model_name in forecasts.keys()}  # Historical days have no forecast
                                        })
                                    
                                    # Create forecast data rows
                                    forecast_rows = []
                                    for i, date_str in enumerate(daily_data['Date']):
                                        forecast_row = {'Date': date_str, 'Actuals': 0}
                                        for model_name in forecasts.keys():
                                            forecast_row[f"{model_name}_Forecast"] = daily_data[f"{model_name}_Forecast"][i]
                                        forecast_rows.append(forecast_row)
                                    
                                    # Combine historical and forecast data
                                    all_daily_data = historical_rows + forecast_rows
                                    
                                    if all_daily_data:
                                        daily_df = pd.DataFrame(all_daily_data)
                                        daily_df = daily_df.sort_values('Date')  # Sort by date
                                        daily_df.to_excel(writer, sheet_name=daily_sheet_name, index=False)
                                elif daily_data and len(daily_data) > 1:  # More than just 'Date' column
                                    daily_df = pd.DataFrame(daily_data)
                                    daily_df.to_excel(writer, sheet_name=daily_sheet_name, index=False)
        
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        # Log the error for debugging
        print(f"Excel generation error: {str(e)}")
        return None


def display_spike_analysis(spike_analysis):
    """
    Display spike analysis results in the UI.
    
    Args:
        spike_analysis: dict with spike detection results
    """
    st.subheader("📈 Monthly Renewal Analysis")
    
    if not spike_analysis.get('has_spikes', False):
        st.info("📊 No significant monthly renewal spikes detected in the data.")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Spike Days Found",
            len(spike_analysis.get('spike_days', [])),
            help="Number of days identified as renewal spikes"
        )
    
    with col2:
        baseline = spike_analysis.get('baseline_avg', 0)
        st.metric(
            "Baseline Daily Avg",
            f"${baseline:,.0f}",
            help="Average daily value excluding spike days"
        )
    
    with col3:
        spike_avg = spike_analysis.get('spike_avg', 0)
        st.metric(
            "Spike Average",
            f"${spike_avg:,.0f}",
            help="Average value on renewal spike days"
        )
    
    with col4:
        contribution = spike_analysis.get('spike_contribution', 0) * 100
        st.metric(
            "Spike Revenue %",
            f"{contribution:.1f}%",
            help="Percentage of total revenue from spike days"
        )
    
    # Spike pattern
    spike_pattern = spike_analysis.get('spike_pattern', [])
    if spike_pattern:
        st.write("**Most Common Renewal Days:**")
        for day, count in spike_pattern[:3]:
            st.write(f"• Day {day} of month ({count} occurrences)")
        
        # Additional insights
        multiplier = spike_analysis.get('spike_multiplier', 1)
        st.info(f"📊 **Spike Analysis:** Renewal days are {multiplier:.1f}x higher than baseline on average")


def display_capacity_adjustment_impact(original_result, adjusted_result):
    """
    Display the impact of capacity adjustments on forecasts.
    
    Args:
        original_result: dict with original forecast results
        adjusted_result: dict with capacity-adjusted forecast results
    """
    if 'capacity_adjustment' not in adjusted_result:
        return
    
    capacity_info = adjusted_result['capacity_adjustment']
    reduction_pct = capacity_info['reduction_pct']
    
    st.subheader("⚙️ Capacity Constraint Impact")
    
    # Calculate impact
    original_total = original_result.get('summary', {}).get('quarter_total', 0)
    adjusted_total = adjusted_result.get('summary', {}).get('quarter_total', 0)
    dollar_impact = original_total - adjusted_total
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Original Forecast",
            f"${original_total:,.0f}",
            help="Unconstrained forecast"
        )
    
    with col2:
        st.metric(
            "Adjusted Forecast",
            f"${adjusted_total:,.0f}",
            delta=f"-${dollar_impact:,.0f}",
            help="Capacity-constrained forecast"
        )
    
    with col3:
        st.metric(
            "Reduction Applied",
            f"{reduction_pct:.1f}%",
            help="Capacity constraint factor applied"
        )
    
    if dollar_impact > 0:
        st.warning(f"💡 **Capacity constraints reduce forecast by ${dollar_impact:,.0f}** due to operational limitations.")
    else:
        st.info("ℹ️ No capacity constraints applied to this forecast.")

def display_monthly_splits(forecasts_data, analysis_date, conservatism_factor=1.0, capacity_factor=1.0, show_header=True):
    """
    Display monthly breakdown of actuals and forecasts in the UI.
    
    Args:
        forecasts_data: dict with forecast results by product
        analysis_date: datetime of analysis
        conservatism_factor: float, conservatism adjustment factor
        capacity_factor: float, capacity constraint factor
        show_header: bool, whether to show the header (default True)
    """
    if show_header:
        st.subheader("📅 Monthly Breakdown")
    
    # Import necessary functions
    from .quarterly_forecasting import apply_conservatism_adjustment_to_forecast, apply_capacity_adjustment_to_forecast
    from .fiscal_calendar import get_fiscal_quarter_info
    from datetime import datetime, timedelta
    import pandas as pd
    
    monthly_data = []
    
    for product, data in forecasts_data.items():
        if 'forecast' in data and 'data' in data:
            forecast_result = data['forecast'].copy()
            historical_data = data['data']  # This is the actual historical data
            
            # Apply adjustments
            if conservatism_factor != 1.0:
                forecast_result = apply_conservatism_adjustment_to_forecast(forecast_result, conservatism_factor)
            if capacity_factor != 1.0:
                forecast_result = apply_capacity_adjustment_to_forecast(forecast_result, capacity_factor)
            
            # Get quarter info
            quarter_info = forecast_result.get('quarter_info', get_fiscal_quarter_info(analysis_date))
            quarter_start = quarter_info['quarter_start']
            quarter_end = quarter_info['quarter_end']
            
            # Filter historical data to current quarter
            quarter_mask = (historical_data.index >= quarter_start) & (historical_data.index <= quarter_end)
            quarter_historical = historical_data[quarter_mask]
            
            # Group by month and calculate actuals
            monthly_actuals = quarter_historical.groupby(quarter_historical.index.to_period('M')).sum()
            
            # Get the best forecast and forecasts dict
            best_model = forecast_result.get('best_model', 'Unknown')
            forecasts = forecast_result.get('forecasts', {})
            
            # Calculate monthly forecast splits
            quarter_progress = forecast_result.get('quarter_progress', {})
            days_remaining = quarter_progress.get('days_remaining', 0)
            
            if days_remaining > 0 and best_model in forecasts:
                # Get the forecast values for the best model
                best_forecast = forecasts[best_model]
                remaining_total = best_forecast.get('remaining_total', 0)
                
                # Calculate how to split remaining forecast across months
                data_end_date = quarter_historical.index.max() if len(quarter_historical) > 0 else analysis_date
                
                # Create future dates for remaining period
                future_dates = pd.date_range(
                    start=data_end_date + timedelta(days=1),
                    end=quarter_end,
                    freq='D'
                )
                
                # Group future dates by month and calculate proportional splits
                if len(future_dates) > 0:
                    future_monthly = future_dates.to_series().groupby(future_dates.to_period('M')).count()                            # Split forecast proportionally by business days in each month
                    for month_period, days_in_month in future_monthly.items():
                        proportion = days_in_month / len(future_dates)
                        monthly_forecast = remaining_total * proportion
                        
                        # Check if we have actuals for this month
                        month_actual = monthly_actuals.get(month_period, 0)
                        month_total = month_actual + monthly_forecast
                        
                        # Convert period to datetime for formatting
                        month_datetime = pd.to_datetime(str(month_period))
                        
                        monthly_data.append({
                            'Product': product,
                            'Month': str(month_period),
                            'Month_Name': month_datetime.strftime('%B %Y'),
                            'Actuals': month_actual,
                            'Forecast': monthly_forecast,
                            'Total': month_total,
                            'Best_Model': best_model,
                            'Days_Actual': len(quarter_historical[quarter_historical.index.to_period('M') == month_period]),
                            'Days_Forecast': days_in_month
                        })
            
            # Add months that have only actuals (no forecast needed)
            for month_period, month_actual in monthly_actuals.items():
                # Check if we already added this month above
                existing_row = next((row for row in monthly_data 
                                   if row['Product'] == product and row['Month'] == str(month_period)), None)
                if not existing_row:
                    # Convert period to datetime for formatting
                    month_datetime = pd.to_datetime(str(month_period))
                    
                    monthly_data.append({
                        'Product': product,
                        'Month': str(month_period),
                        'Month_Name': month_datetime.strftime('%B %Y'),
                        'Actuals': month_actual,
                        'Forecast': 0,
                        'Total': month_actual,
                        'Best_Model': best_model,
                        'Days_Actual': len(quarter_historical[quarter_historical.index.to_period('M') == month_period]),
                        'Days_Forecast': 0
                    })
    
    if monthly_data:
        # Create DataFrame and display
        monthly_df = pd.DataFrame(monthly_data)
        
        # Sort by Product first, then by the actual datetime value for proper chronological order
        monthly_df['Month_Sort'] = pd.to_datetime(monthly_df['Month']).dt.to_period('M')
        monthly_df = monthly_df.sort_values(['Product', 'Month_Sort'])
        monthly_df = monthly_df.drop('Month_Sort', axis=1)  # Remove the sort helper column
        
        # Format currency columns for display
        display_df = monthly_df.copy()
        display_df['Actuals'] = display_df['Actuals'].apply(lambda x: f"${x:,.0f}")
        display_df['Forecast'] = display_df['Forecast'].apply(lambda x: f"${x:,.0f}")
        display_df['Total'] = display_df['Total'].apply(lambda x: f"${x:,.0f}")
        
        # Select and reorder columns for display
        display_columns = ['Product', 'Month_Name', 'Actuals', 'Forecast', 'Total', 'Best_Model', 'Days_Actual', 'Days_Forecast']
        display_df = display_df[display_columns]
        
        # Rename columns for better presentation
        display_df.columns = ['Product', 'Month', 'Actuals', 'Forecast', 'Total', 'Model', 'Days w/ Actuals', 'Days Forecasted']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Show summary by month - also sort chronologically
        st.subheader("📊 Monthly Summary (All Products)")
        monthly_summary = monthly_df.groupby('Month_Name').agg({
            'Actuals': 'sum',
            'Forecast': 'sum',
            'Total': 'sum'
        }).reset_index()
        
        # Sort the summary by chronological order as well
        monthly_summary['Month_Sort'] = pd.to_datetime(monthly_summary['Month_Name'], format='%B %Y')
        monthly_summary = monthly_summary.sort_values('Month_Sort')
        monthly_summary = monthly_summary.drop('Month_Sort', axis=1)
        
        # Format currency for summary
        monthly_summary['Actuals'] = monthly_summary['Actuals'].apply(lambda x: f"${x:,.0f}")
        monthly_summary['Forecast'] = monthly_summary['Forecast'].apply(lambda x: f"${x:,.0f}")
        monthly_summary['Total'] = monthly_summary['Total'].apply(lambda x: f"${x:,.0f}")
        
        monthly_summary.columns = ['Month', 'Total Actuals', 'Total Forecast', 'Total Expected']
        st.dataframe(monthly_summary, use_container_width=True)
        
    else:
        st.info("📋 No monthly data available. Upload data to see monthly breakdowns.")
