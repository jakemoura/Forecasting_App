"""
Monthly Spike Detection for Outlook Forecaster

Detects monthly subscription renewals and recurring revenue spikes
in daily business data for better forecasting accuracy.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from collections import Counter


def detect_monthly_spikes(series, spike_threshold=2.0):
    """
    Detect monthly subscription renewal spikes in daily business data.
    
    Args:
        series: pandas Series with datetime index containing daily data
        spike_threshold: float, multiplier above baseline to consider a spike
        
    Returns:
        dict: Spike analysis results including spike days, patterns, and statistics
    """
    if len(series) < 10:
        return {
            'has_spikes': False,
            'spike_days': [],
            'spike_pattern': [],
            'baseline_avg': series.mean() if len(series) > 0 else 0,
            'spike_avg': 0,
            'spike_contribution': 0,
            'spike_frequency': 0,
            'spike_multiplier': 1,
            'debug_info': {'reason': 'insufficient_data', 'data_length': len(series)}
        }
    
    # Use median as a more robust baseline (less affected by outliers/spikes)
    baseline_median = series.median()
    
    # Also calculate mean for comparison
    baseline_mean = series.mean()
    
    # Use the lower of median and mean as baseline (more conservative)
    initial_baseline = min(baseline_median, baseline_mean)
    
    # Identify potential spikes using the threshold
    potential_spikes = series > (initial_baseline * spike_threshold)
    
    # Calculate refined baseline from non-spike days
    non_spike_data = series[~potential_spikes]
    
    if len(non_spike_data) < 3:
        # If too many days are spikes, use a more conservative baseline
        baseline_avg = series.quantile(0.25)  # 25th percentile
    else:
        baseline_avg = non_spike_data.mean()
    
    # Final spike detection with refined baseline
    final_spike_mask = series > (baseline_avg * spike_threshold)
    spike_days = series[final_spike_mask]
    
    # Debug information
    debug_info = {
        'data_length': len(series),
        'initial_baseline': initial_baseline,
        'refined_baseline': baseline_avg,
        'threshold_value': baseline_avg * spike_threshold,
        'potential_spikes_count': potential_spikes.sum(),
        'final_spikes_count': final_spike_mask.sum(),
        'series_max': series.max(),
        'series_min': series.min(),
        'spike_threshold': spike_threshold
    }
    
    if len(spike_days) == 0:
        debug_info['reason'] = 'no_spikes_detected'
        return {
            'has_spikes': False,
            'spike_days': [],
            'spike_pattern': [],
            'baseline_avg': baseline_avg,
            'spike_avg': 0,
            'spike_contribution': 0,
            'spike_frequency': 0,
            'spike_multiplier': 1,
            'debug_info': debug_info
        }
    
    # Analyze spike patterns by day of month
    spike_dates = spike_days.index
    spike_days_of_month = [d.day for d in spike_dates]
    
    # Find most common spike days - but require multiple occurrences for true patterns
    day_counts = Counter(spike_days_of_month)
    
    # Filter to only include days that occur multiple times (true recurring patterns)
    # Require at least 2 occurrences, or if we have >6 months of data, require at least 3
    # But if we have less than 3 months of data, allow single occurrences
    if len(series) < 90:  # Less than 3 months
        min_occurrences = 1
    elif len(series) > 180:  # More than 6 months
        min_occurrences = 3
    else:  # 3-6 months
        min_occurrences = 2
        
    recurring_days = [(day, count) for day, count in day_counts.items() if count >= min_occurrences]
    
    # Sort by frequency to get the most consistent patterns
    most_common_days = sorted(recurring_days, key=lambda x: x[1], reverse=True)[:3]
    
    # Debug info for pattern analysis
    debug_info['all_spike_days'] = dict(day_counts.most_common())
    debug_info['recurring_days'] = dict(recurring_days)
    debug_info['min_occurrences_required'] = min_occurrences
    
    print(f"[DEBUG] Spike day analysis:")
    print(f"  - All spike days found: {dict(day_counts.most_common())}")
    print(f"  - Recurring patterns (≥{min_occurrences}x): {dict(recurring_days)}")
    print(f"  - Selected for forecasting: {[day for day, count in most_common_days]}")
    
    # Calculate spike contribution to total revenue
    spike_contribution = spike_days.sum() / series.sum()
    
    # Additional validation: ensure spikes are significantly higher than baseline
    avg_spike_multiplier = spike_days.mean() / baseline_avg if baseline_avg > 0 else 1
    
    # Add spike dates and values to debug info
    debug_info['spike_dates'] = spike_dates.strftime('%Y-%m-%d').tolist()
    debug_info['spike_values'] = spike_days.values.tolist()
    debug_info['spike_multiplier'] = avg_spike_multiplier
    debug_info['reason'] = 'spikes_detected'
    
    return {
        'has_spikes': True,
        'spike_days': spike_dates,  # Keep as datetime objects
        'spike_pattern': most_common_days,
        'baseline_avg': baseline_avg,
        'spike_avg': spike_days.mean(),
        'spike_contribution': spike_contribution,
        'spike_frequency': len(spike_days) / len(series) * 30,
        'spike_multiplier': avg_spike_multiplier,
        'debug_info': debug_info
    }


def create_monthly_renewal_forecast(series, remaining_days, spike_analysis):
    """
    Create forecast that includes expected monthly renewal spikes.
    
    Args:
        series: Historical daily data
        remaining_days: Number of days to forecast (all days, including weekends)
        spike_analysis: Results from detect_monthly_spikes
    
    Returns:
        np.ndarray: Array of daily forecasts including expected spikes
    """
    if not spike_analysis['has_spikes']:
        baseline_daily = series.mean()
        return np.full(remaining_days, baseline_daily)
    
    # Create forecast array
    forecast = []
    baseline_daily = spike_analysis['baseline_avg']
    spike_avg = spike_analysis['spike_avg']
    
    # Get the last data date to determine when to expect next spikes
    last_date = series.index.max()
    
    # Generate calendar dates for forecast period (all days, including weekends)
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=remaining_days,
        freq='D'
    )
    
    # Extract spike days for easy comparison - only use truly recurring patterns
    common_spike_days = []
    if spike_analysis['spike_pattern']:
        # Only use patterns that passed the minimum occurrence test
        common_spike_days = [day for day, count in spike_analysis['spike_pattern'][:2]]
    
    print(f"[DEBUG] Monthly Renewals forecast will use spike days: {common_spike_days}")
    
    spike_days_found = 0
    # Generate forecast for each day (including weekends)
    for i, future_date in enumerate(future_dates):
        day_of_month = future_date.day
        
        # Check if this day matches common spike pattern
        if common_spike_days and day_of_month in common_spike_days:
            forecast.append(spike_avg)
            spike_days_found += 1
        else:
            forecast.append(baseline_daily)
    
    print(f"[DEBUG] Monthly Renewals forecast: {spike_days_found} spike days found, spike avg: ${spike_avg:,.0f}, baseline: ${baseline_daily:,.0f}")
    
    return np.array(forecast[:remaining_days])


def overlay_spikes_on_forecast(base_forecast, historical_data, spike_analysis, forecast_days, spike_intensity=0.85):
    """
    Overlay expected monthly spikes onto a base forecast with reduced intensity.
    
    Args:
        base_forecast: Base forecast array
        historical_data: Historical daily data
        spike_analysis: Spike analysis results
        forecast_days: Number of days being forecasted
        spike_intensity: float, factor to reduce spike intensity (0.85 = 15% reduction)
    
    Returns:
        np.ndarray: Enhanced forecast with spikes overlaid
    """
    if not spike_analysis['has_spikes']:
        return base_forecast
    
    # Normalize forecast to a NumPy array to ensure safe positional indexing
    # Some models return pandas Series with non-zero/DateTime indices; using
    # raw [i] indexing on a Series can raise KeyError. Converting to ndarray
    # guarantees 0..N-1 positional access.
    if isinstance(base_forecast, pd.Series):
        enhanced_forecast = base_forecast.to_numpy(copy=True)
    elif isinstance(base_forecast, (list, tuple, np.ndarray)):
        enhanced_forecast = np.array(base_forecast, dtype=float).copy()
    else:
        try:
            enhanced_forecast = np.array(base_forecast, dtype=float).copy()
        except Exception:
            # Fallback to a flat array at baseline if conversion fails
            enhanced_forecast = np.full(int(forecast_days), float(spike_analysis.get('baseline_avg', 0.0)))
    last_date = historical_data.index.max()
    
    # Generate calendar dates for forecast period (all days, includes weekends)
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=min(len(enhanced_forecast), int(forecast_days)),
        freq='D'
    )
    
    # Calculate conservative spike value (reduced intensity to avoid over-forecasting)
    baseline_avg = spike_analysis['baseline_avg']
    spike_avg = spike_analysis['spike_avg']
    
    # Use a blend of spike and baseline rather than full spike value
    conservative_spike_value = baseline_avg + ((spike_avg - baseline_avg) * spike_intensity)
    
    # Calculate the spike boost amount
    spike_boost = (conservative_spike_value - baseline_avg)
    
    # Extract common spike days - only use truly recurring patterns
    common_spike_days = []
    if spike_analysis['spike_pattern']:
        # Only use the top recurring patterns (those that passed the minimum occurrence test)
        common_spike_days = [day for day, count in spike_analysis['spike_pattern'][:2]]  # Top 2 recurring patterns
        
    print(f"[DEBUG] Spike overlay will apply to days: {common_spike_days}")
    
    # Apply spikes more aggressively - always add spike boost on renewal days
    spike_applications = 0
    for i, future_date in enumerate(future_dates):
        if i >= len(enhanced_forecast):
            break
            
        day_of_month = future_date.day
        
        # Apply spike boost on expected spike days
        if common_spike_days and day_of_month in common_spike_days:
            # Always add the spike boost, regardless of base forecast level
            enhanced_forecast[i] = enhanced_forecast[i] + spike_boost
            spike_applications += 1
            
    print(f"[DEBUG] Spike overlay applied to {spike_applications} days with boost of ${spike_boost:,.0f}")
    print(f"[DEBUG] Days receiving spike overlay: {[d.day for d in future_dates if d.day in common_spike_days]}")
    
    return enhanced_forecast
