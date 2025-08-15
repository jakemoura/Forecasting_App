"""
Quarterly Forecasting Engine for Outlook Forecaster

Main forecasting engine that combines multiple models to predict
quarterly performance from partial daily data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .fiscal_calendar import get_fiscal_quarter_info, get_business_days_in_period
from .forecasting_models import *
from .spike_detection import detect_monthly_spikes, create_monthly_renewal_forecast, overlay_spikes_on_forecast
from .model_evaluation import (
    evaluate_individual_forecasts,
    select_best_model_weighted,
    smart_backtesting_select_model,
)


def forecast_quarter_completion(series, current_date=None, detect_spikes=True, spike_threshold=2.0, spike_intensity=0.85):
    """
    Main forecasting function - predicts full quarter performance from partial data.
    
    Args:
        series: pandas Series with datetime index containing daily data
        current_date: datetime, analysis date (defaults to last date in series)
        detect_spikes: bool, whether to detect and forecast monthly spikes
        spike_threshold: float, sensitivity for spike detection
        spike_intensity: float, intensity factor for spike overlay (0.85 = 15% reduction)
        
    Returns:
        dict: Comprehensive forecast results with multiple models and analysis
    """
    if len(series) == 0:
        return {'error': 'No data provided'}
    
    if current_date is None:
        current_date = series.index.max()
    
    # Get fiscal quarter information
    quarter_info = get_fiscal_quarter_info(current_date)
    quarter_start = quarter_info['quarter_start']
    quarter_end = quarter_info['quarter_end']
    
    # Filter data to current quarter only
    quarter_mask = (series.index >= quarter_start) & (series.index <= quarter_end)
    quarter_data = series[quarter_mask].sort_index()
    
    if len(quarter_data) == 0:
        return {'error': f'No data found for {quarter_info["quarter_name"]}'}
    
    # Calculate days completed and remaining in quarter
    total_quarter_days = get_business_days_in_period(quarter_start, quarter_end)
    data_end_date = quarter_data.index.max()
    days_completed = get_business_days_in_period(quarter_start, data_end_date)
    days_remaining = total_quarter_days - days_completed
    
    if days_remaining <= 0:
        # Quarter is complete
        actual_total = quarter_data.sum()
        return {
            'quarter_info': quarter_info,
            'actual_total': actual_total,
            'days_completed': days_completed,
            'days_remaining': 0,
            'quarter_complete': True,
            'forecasts': {},
            'summary': {
                'quarter_total': actual_total,
                'confidence_interval': (actual_total, actual_total)
            }
        }
    
    # Fit multiple forecasting models
    forecasts = {}
    
    # 1. Linear Trend Model
    linear_result = fit_linear_trend_model(quarter_data)
    if linear_result['model_type'] != 'fallback_mean':
        # Project linear trend forward
        future_x = np.arange(len(quarter_data), len(quarter_data) + days_remaining).reshape(-1, 1)
        linear_forecast = linear_result['model'].predict(future_x)
        forecasts['Linear Trend'] = {
            'forecast': linear_forecast,
            'daily_avg': np.mean(linear_forecast),
            'remaining_total': np.sum(linear_forecast),
            'quarter_total': quarter_data.sum() + np.sum(linear_forecast)
        }
    
    # 2. Moving Average Model
    ma_result = fit_moving_average_model(quarter_data, window=7)
    if 'forecast_value' in ma_result:
        ma_forecast = np.full(days_remaining, ma_result['forecast_value'])
        forecasts['Moving Average'] = {
            'forecast': ma_forecast,
            'daily_avg': ma_result['forecast_value'],
            'remaining_total': np.sum(ma_forecast),
            'quarter_total': quarter_data.sum() + np.sum(ma_forecast)
        }
    
    # 3. Prophet Model (if available) - Use full historical data for training
    prophet_result = fit_prophet_daily_model(series)  # Use full series instead of just quarter_data
    if prophet_result['model_type'] == 'prophet' and 'model' in prophet_result:
        try:
            # Create future dates for forecast
            future_dates = pd.date_range(
                start=data_end_date + timedelta(days=1),
                periods=days_remaining,
                freq='D'
            )
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Generate forecast
            prophet_forecast = prophet_result['model'].predict(future_df)
            prophet_values = prophet_forecast['yhat'].values
            prophet_values = np.maximum(prophet_values, 0)  # Ensure non-negative
            
            forecasts['Prophet'] = {
                'forecast': prophet_values,
                'daily_avg': np.mean(prophet_values),
                'remaining_total': np.sum(prophet_values),
                'quarter_total': quarter_data.sum() + np.sum(prophet_values)
            }
        except Exception:
            pass
    
    # 4. Run Rate Model (simple daily average)
    daily_avg = quarter_data.mean()
    run_rate_forecast = np.full(days_remaining, daily_avg)
    forecasts['Run Rate'] = {
        'forecast': run_rate_forecast,
        'daily_avg': daily_avg,
        'remaining_total': np.sum(run_rate_forecast),
        'quarter_total': quarter_data.sum() + np.sum(run_rate_forecast)
    }
    
    # 5. Time Series Models (ARIMA, Exponential Smoothing) - Use full historical data
    if len(series) >= 14:  # Use full series length instead of quarter_data
        # ARIMA
        arima_result = fit_arima_model(series)  # Use full series instead of quarter_data
        if arima_result['model_type'] == 'arima' and 'model' in arima_result:
            try:
                arima_forecast = arima_result['model'].forecast(steps=days_remaining)
                arima_values = np.maximum(arima_forecast, 0)  # Ensure non-negative
                forecasts['ARIMA'] = {
                    'forecast': arima_values,
                    'daily_avg': np.mean(arima_values),
                    'remaining_total': np.sum(arima_values),
                    'quarter_total': quarter_data.sum() + np.sum(arima_values)
                }
            except Exception:
                pass
        
        # Exponential Smoothing
        exp_result = fit_exponential_smoothing_model(series)  # Use full series instead of quarter_data
        if exp_result['model_type'] == 'exponential_smoothing' and 'model' in exp_result:
            try:
                exp_forecast = exp_result['model'].forecast(steps=days_remaining)
                exp_values = np.maximum(exp_forecast, 0)  # Ensure non-negative
                forecasts['Exponential Smoothing'] = {
                    'forecast': exp_values,
                    'daily_avg': np.mean(exp_values),
                    'remaining_total': np.sum(exp_values),
                    'quarter_total': quarter_data.sum() + np.sum(exp_values)
                }
            except Exception:
                pass
    
    # 6. Machine Learning Models (if sufficient data) - Use full historical data
    if len(series) >= 21:  # Use full series length instead of quarter_data
        # XGBoost
        xgb_model, xgb_features = fit_xgboost_model(series)  # Use full series instead of quarter_data
        if xgb_model is not None:
            try:
                # For now, use a reasonable prediction based on recent trends
                recent_avg = series.iloc[-min(14, len(series)):].mean()  # Last 14 days or available
                xgb_forecast = np.full(days_remaining, recent_avg)  # Use recent average as prediction
                
                forecasts['XGBoost'] = {
                    'forecast': xgb_forecast,
                    'daily_avg': np.mean(xgb_forecast),
                    'remaining_total': np.sum(xgb_forecast),
                    'quarter_total': quarter_data.sum() + np.sum(xgb_forecast)
                }
            except Exception:
                pass
        
        # LightGBM
        lgbm_result = fit_lightgbm_daily_model(series)  # Use full series instead of quarter_data
        if lgbm_result['model_type'] == 'lightgbm' and 'model' in lgbm_result:
            try:
                # Use the actual forecast value from LightGBM if available
                if 'forecast_value' in lgbm_result:
                    lgbm_forecast = np.full(days_remaining, lgbm_result['forecast_value'])
                else:
                    # Fallback to recent average
                    recent_avg = series.iloc[-min(14, len(series)):].mean()
                    lgbm_forecast = np.full(days_remaining, recent_avg)
                
                forecasts['LightGBM'] = {
                    'forecast': lgbm_forecast,
                    'daily_avg': np.mean(lgbm_forecast),
                    'remaining_total': np.sum(lgbm_forecast),
                    'quarter_total': quarter_data.sum() + np.sum(lgbm_forecast)
                }
            except Exception:
                pass
    
    # 7. Spike Detection and Analysis - Use full historical data for better pattern detection
    spike_analysis = None
    if detect_spikes:
        # Try spike detection on full series first for better pattern recognition
        if len(series) >= 30:  # Need at least 30 days for reliable spike detection
            spike_analysis = detect_monthly_spikes(series, spike_threshold)
            # Debug: Print spike detection results
            if spike_analysis and spike_analysis.get('debug_info'):
                debug = spike_analysis['debug_info']
                print(f"[DEBUG] Spike Detection on Full Series ({len(series)} days):")
                print(f"  - Has spikes: {spike_analysis['has_spikes']}")
                print(f"  - Spike count: {debug.get('final_spikes_count', 0)}")
                print(f"  - Baseline: {spike_analysis.get('baseline_avg', 0):.2f}")
                print(f"  - Spike avg: {spike_analysis.get('spike_avg', 0):.2f}")
                print(f"  - Spike pattern (top days): {spike_analysis.get('spike_pattern', [])}")
                if spike_analysis.get('spike_days') is not None and len(spike_analysis['spike_days']) > 0:
                    recent_spikes = spike_analysis['spike_days'][-5:]  # Last 5 spike dates
                    print(f"  - Recent spike dates: {[d.strftime('%Y-%m-%d') for d in recent_spikes]}")
        else:
            # Fallback to quarter data if full series is too short
            spike_analysis = detect_monthly_spikes(quarter_data, spike_threshold)
    
    # 8. Monthly Renewal Model (if spikes detected or forced detection)
    if spike_analysis and spike_analysis['has_spikes']:
        renewal_forecast = create_monthly_renewal_forecast(
            quarter_data, days_remaining, spike_analysis
        )
        forecasts['Monthly Renewals'] = {
            'forecast': renewal_forecast,
            'daily_avg': np.mean(renewal_forecast),
            'remaining_total': np.sum(renewal_forecast),
            'quarter_total': quarter_data.sum() + np.sum(renewal_forecast),
            'spike_analysis': spike_analysis
        }
    elif detect_spikes and len(series) >= 30:
        # Force create a renewal model even if no spikes in current quarter
        # Use patterns from full historical data
        historical_spike_analysis = detect_monthly_spikes(series, spike_threshold * 0.8)  # Lower threshold for historical
        if historical_spike_analysis['has_spikes']:
            renewal_forecast = create_monthly_renewal_forecast(
                series, days_remaining, historical_spike_analysis
            )
            forecasts['Monthly Renewals'] = {
                'forecast': renewal_forecast,
                'daily_avg': np.mean(renewal_forecast),
                'remaining_total': np.sum(renewal_forecast),
                'quarter_total': quarter_data.sum() + np.sum(renewal_forecast),
                'spike_analysis': historical_spike_analysis
            }
    
    # Apply spike overlay to ALL existing forecasts if spikes detected (from current or historical data)
    active_spike_analysis = None
    
    # Use current quarter spike analysis if available
    if spike_analysis and spike_analysis['has_spikes']:
        active_spike_analysis = spike_analysis
        print(f"[DEBUG] Using current quarter spike analysis for overlay")
    # Otherwise, try to use historical spike analysis if we have sufficient data
    elif detect_spikes and len(series) >= 30:
        historical_spike_analysis = detect_monthly_spikes(series, spike_threshold * 0.8)  # Lower threshold for historical
        if historical_spike_analysis['has_spikes']:
            active_spike_analysis = historical_spike_analysis
            print(f"[DEBUG] Using historical spike analysis for overlay")
    
    # Apply spike overlay to all models if we have any spike analysis
    if active_spike_analysis and active_spike_analysis['has_spikes']:
        print(f"[DEBUG] Applying spike overlay to {len([m for m in forecasts.keys() if m != 'Monthly Renewals'])} models")
        for model_name, forecast_data in forecasts.items():
            if model_name != 'Monthly Renewals':  # Skip Monthly Renewals as it's already spike-aware
                # Get the base forecast
                base_forecast = forecast_data['forecast']
                original_total = forecast_data['quarter_total']
                
                # Overlay spikes onto the base forecast
                enhanced_forecast = overlay_spikes_on_forecast(
                    base_forecast, quarter_data, active_spike_analysis, days_remaining, spike_intensity
                )
                
                # Update forecast data with enhanced values
                forecast_data['forecast'] = enhanced_forecast
                forecast_data['daily_avg'] = np.mean(enhanced_forecast)
                forecast_data['remaining_total'] = np.sum(enhanced_forecast)
                forecast_data['quarter_total'] = quarter_data.sum() + np.sum(enhanced_forecast)
                forecast_data['spike_analysis'] = active_spike_analysis
                
                print(f"[DEBUG] {model_name}: ${original_total:,.0f} → ${forecast_data['quarter_total']:,.0f}")
    else:
        print(f"[DEBUG] No spike analysis available for overlay")
    
    # Evaluate models and select best performers
    model_scores = evaluate_individual_forecasts(series, forecasts)  # Use full series for evaluation
    
    # Add forecast reasonableness checks and weighted scoring
    weighted_scores = {}
    historical_avg = series.mean()
    quarter_avg = quarter_data.mean()
    
    for model_name, mape_score in model_scores.items():
        if model_name in forecasts:
            forecast_data = forecasts[model_name]
            
            # Calculate weighted score considering MAPE, forecast reasonableness, and stability
            base_score = mape_score if mape_score != float('inf') else 100.0
            
            # Reasonableness penalty: forecast shouldn't be more than 3x historical average
            reasonableness_penalty = 0
            forecast_avg = forecast_data.get('daily_avg', historical_avg)
            if forecast_avg > historical_avg * 3:
                reasonableness_penalty = 20.0  # Heavy penalty for unrealistic forecasts
            elif forecast_avg > historical_avg * 2:
                reasonableness_penalty = 10.0  # Moderate penalty
            
            # Stability bonus: prefer models with consistent daily forecasts
            stability_bonus = 0
            if 'forecast' in forecast_data and len(forecast_data['forecast']) > 1:
                forecast_std = np.std(forecast_data['forecast'])
                forecast_mean = np.mean(forecast_data['forecast'])
                if forecast_mean > 0:
                    cv = forecast_std / forecast_mean  # Coefficient of variation
                    if cv < 0.3:  # Low variability is good
                        stability_bonus = -2.0
            
            # Calculate final weighted score (lower is better)
            weighted_score = base_score + reasonableness_penalty + stability_bonus
            weighted_scores[model_name] = weighted_score
            
            print(f"[DEBUG] {model_name}: MAPE={mape_score:.1f}%, Reasonableness={reasonableness_penalty:.1f}, Final Score={weighted_score:.1f}")
    
    # Use weighted scores for model selection (Standard selection)
    final_model_scores = weighted_scores if weighted_scores else model_scores
    
    # Create ensemble forecast from top performing models
    top_models = sorted(final_model_scores.items(), key=lambda x: x[1])[:3]  # Top 3 by score
    ensemble_forecast = fit_ensemble_model(quarter_data, {
        name: forecasts[name] for name, score in top_models if name in forecasts
    })
    
    if ensemble_forecast['model_type'] != 'fallback_mean':
        ensemble_values = np.full(days_remaining, ensemble_forecast['forecast'])
        forecasts['Ensemble'] = {
            'forecast': ensemble_values,
            'daily_avg': ensemble_forecast['forecast'],
            'remaining_total': np.sum(ensemble_values),
            'quarter_total': quarter_data.sum() + np.sum(ensemble_values)
        }
        
        # Apply spike overlay to ensemble if we have any spike analysis
        if active_spike_analysis and active_spike_analysis['has_spikes']:
            enhanced_ensemble = overlay_spikes_on_forecast(
                ensemble_values, quarter_data, active_spike_analysis, days_remaining, spike_intensity
            )
            forecasts['Ensemble']['forecast'] = enhanced_ensemble
            forecasts['Ensemble']['daily_avg'] = np.mean(enhanced_ensemble)
            forecasts['Ensemble']['remaining_total'] = np.sum(enhanced_ensemble)
            forecasts['Ensemble']['quarter_total'] = quarter_data.sum() + np.sum(enhanced_ensemble)
            forecasts['Ensemble']['spike_analysis'] = active_spike_analysis
    
    # Calculate summary statistics
    # Smart backtesting-driven selection (parallel path)
    bt_result = smart_backtesting_select_model(series, forecasts, method='auto', horizon=min(7, days_remaining), gap=7, folds=3)
    backtesting_best_model_name = bt_result.get('best_model', None)
    backtesting_mapes = bt_result.get('per_model_mape', {})

    if forecasts:
        quarter_totals = [f['quarter_total'] for f in forecasts.values()]
        median_forecast = np.median(quarter_totals)
        confidence_low = np.percentile(quarter_totals, 10)
        confidence_high = np.percentile(quarter_totals, 90)
        
        # Select best model (Standard) - use weighted scoring instead of just MAPE
        if final_model_scores:
            best_model_name = min(final_model_scores.items(), key=lambda x: x[1])[0]
        else:
            best_model_name = 'Run Rate'
        best_forecast = forecasts.get(best_model_name, forecasts.get('Run Rate', forecasts[list(forecasts.keys())[0]]))
    else:
        median_forecast = quarter_data.sum()
        confidence_low = confidence_high = median_forecast
        best_forecast = {'quarter_total': median_forecast}
        best_model_name = 'Run Rate'
    
    # Add spike analysis to all models if spikes were detected (current or historical)
    final_spike_analysis = active_spike_analysis if 'active_spike_analysis' in locals() else spike_analysis
    if final_spike_analysis is not None and final_spike_analysis.get('has_spikes', False):
        for model_name, model_data in forecasts.items():
            if 'spike_analysis' not in model_data:
                model_data['spike_analysis'] = final_spike_analysis

    result = {
        'quarter_info': quarter_info,
        'data_period': {
            'start': quarter_data.index.min(),
            'end': quarter_data.index.max(),
            'days_with_data': len(quarter_data)
        },
        'quarter_progress': {
            'days_completed': days_completed,
            'days_remaining': days_remaining,
            'total_days': total_quarter_days,
            'completion_pct': (days_completed / total_quarter_days) * 100
        },
        'actual_to_date': quarter_data.sum(),
        'forecasts': forecasts,
    'model_evaluation': final_model_scores,
    'mape_scores': model_scores,
        'best_model': best_model_name,
        'summary': {
            'quarter_total': best_forecast['quarter_total'],
            'median_forecast': median_forecast,
            'confidence_interval': (confidence_low, confidence_high),
            'daily_run_rate': quarter_data.mean()
        },
        'quarter_complete': False
    }

    # Attach backtesting selection results and an alternate best model if present
    if backtesting_best_model_name and backtesting_best_model_name in forecasts:
        result['backtesting'] = {
            'best_model': backtesting_best_model_name,
            'per_model_mape': backtesting_mapes,
            'method_used': bt_result.get('method_used'),
            'quarter_total': forecasts[backtesting_best_model_name]['quarter_total']
        }
    else:
        result['backtesting'] = {
            'best_model': None,
            'per_model_mape': backtesting_mapes,
            'method_used': bt_result.get('method_used')
        }

    return result


def apply_capacity_adjustment_to_forecast(forecast_result, capacity_factor):
    """
    Apply capacity constraints to forecast results.
    
    Args:
        forecast_result: dict, output from forecast_quarter_completion
        capacity_factor: float, multiplier to reduce forecasts (0.0-1.0)
        
    Returns:
        dict: Adjusted forecast results
    """
    if 'forecasts' not in forecast_result:
        return forecast_result
    
    # Apply capacity adjustment to all forecasts
    adjusted_forecasts = {}
    for model_name, forecast_data in forecast_result['forecasts'].items():
        adjusted_data = forecast_data.copy()
        
        # Adjust forecast values
        if 'forecast' in adjusted_data:
            adjusted_data['forecast'] = adjusted_data['forecast'] * capacity_factor
        if 'daily_avg' in adjusted_data:
            adjusted_data['daily_avg'] = adjusted_data['daily_avg'] * capacity_factor
        if 'remaining_total' in adjusted_data:
            adjusted_data['remaining_total'] = adjusted_data['remaining_total'] * capacity_factor
        if 'quarter_total' in adjusted_data:
            # Adjust only the forecasted portion, not the actual data
            actual_to_date = forecast_result.get('actual_to_date', 0)
            forecasted_portion = adjusted_data['quarter_total'] - actual_to_date
            adjusted_data['quarter_total'] = actual_to_date + (forecasted_portion * capacity_factor)
        
        adjusted_forecasts[model_name] = adjusted_data
    
    # Update summary with adjusted values
    adjusted_result = forecast_result.copy()
    adjusted_result['forecasts'] = adjusted_forecasts
    
    if 'summary' in adjusted_result:
        summary = adjusted_result['summary'].copy()
        actual_to_date = adjusted_result.get('actual_to_date', 0)
        
        # Adjust summary forecasts
        original_forecast = summary['quarter_total'] - actual_to_date
        summary['quarter_total'] = actual_to_date + (original_forecast * capacity_factor)
        
        original_median = summary['median_forecast'] - actual_to_date
        summary['median_forecast'] = actual_to_date + (original_median * capacity_factor)
        
        # Adjust confidence intervals
        conf_low, conf_high = summary['confidence_interval']
        original_low = conf_low - actual_to_date
        original_high = conf_high - actual_to_date
        summary['confidence_interval'] = (
            actual_to_date + (original_low * capacity_factor),
            actual_to_date + (original_high * capacity_factor)
        )
        
        adjusted_result['summary'] = summary
    
    # Add capacity adjustment info
    adjusted_result['capacity_adjustment'] = {
        'applied': True,
        'factor': capacity_factor,
        'reduction_pct': (1 - capacity_factor) * 100
    }
    
    return adjusted_result


def apply_conservatism_adjustment_to_forecast(forecast_result, conservatism_factor):
    """
    Apply conservatism adjustment to forecast results to reduce systematic bias.
    
    Args:
        forecast_result: dict, output from forecast_quarter_completion
        conservatism_factor: float, multiplier to adjust forecasts (0.97 = 3% more conservative)
        
    Returns:
        dict: Adjusted forecast results
    """
    if 'forecasts' not in forecast_result:
        return forecast_result
    
    # Apply conservatism adjustment to all forecasts
    adjusted_forecasts = {}
    for model_name, forecast_data in forecast_result['forecasts'].items():
        adjusted_data = forecast_data.copy()
        
        # Adjust forecast values (only future predictions, not historical data)
        if 'forecast' in adjusted_data:
            adjusted_data['forecast'] = adjusted_data['forecast'] * conservatism_factor
        if 'daily_avg' in adjusted_data:
            adjusted_data['daily_avg'] = adjusted_data['daily_avg'] * conservatism_factor
        if 'remaining_total' in adjusted_data:
            adjusted_data['remaining_total'] = adjusted_data['remaining_total'] * conservatism_factor
        if 'quarter_total' in adjusted_data:
            # Adjust only the forecasted portion, not the actual data
            actual_to_date = forecast_result.get('actual_to_date', 0)
            forecasted_portion = adjusted_data['quarter_total'] - actual_to_date
            adjusted_data['quarter_total'] = actual_to_date + (forecasted_portion * conservatism_factor)
        
        adjusted_forecasts[model_name] = adjusted_data
    
    # Update summary with adjusted values
    adjusted_result = forecast_result.copy()
    adjusted_result['forecasts'] = adjusted_forecasts
    
    if 'summary' in adjusted_result:
        summary = adjusted_result['summary'].copy()
        actual_to_date = adjusted_result.get('actual_to_date', 0)
        
        # Adjust summary forecasts
        original_forecast = summary['quarter_total'] - actual_to_date
        summary['quarter_total'] = actual_to_date + (original_forecast * conservatism_factor)
        
        original_median = summary['median_forecast'] - actual_to_date
        summary['median_forecast'] = actual_to_date + (original_median * conservatism_factor)
        
        # Adjust confidence intervals
        conf_low, conf_high = summary['confidence_interval']
        original_low = conf_low - actual_to_date
        original_high = conf_high - actual_to_date
        summary['confidence_interval'] = (
            actual_to_date + (original_low * conservatism_factor),
            actual_to_date + (original_high * conservatism_factor)
        )
        
        adjusted_result['summary'] = summary
    
    # Add conservatism adjustment info
    adjusted_result['conservatism_adjustment'] = {
        'applied': True,
        'factor': conservatism_factor,
        'adjustment_pct': (conservatism_factor - 1) * 100
    }
    
    return adjusted_result
