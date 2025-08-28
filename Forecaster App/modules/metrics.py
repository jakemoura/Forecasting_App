"""
Metrics and evaluation functions for forecasting models.

Contains implementations of WAPE (weighted APE), SMAPE, MASE, RMSE and other
evaluation metrics for time series forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta


def wape(actual, forecast):
    """Weighted Absolute Percentage Error (WAPE).

    Returns a scale-sensitive error aligned to revenue (dollar-weighted):
        sum(|A - F|) / sum(|A|)

    For all-zero actuals, returns 0.0 if forecasts are all zero else +inf.
    """
    import numpy as np  # Local import to avoid circulars in edge tooling
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.sum(np.abs(actual))
    if not np.isfinite(denom) or denom == 0:
        return 0.0 if np.allclose(actual, forecast, atol=1e-12) else np.inf
    return float(np.sum(np.abs(actual - forecast)) / denom)


def smape(actual, forecast):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Args:
        actual: Array of actual values
        forecast: Array of forecasted values
    
    Returns:
        SMAPE value (0 to 1, lower is better)
    """
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    # Handle zeros and avoid division by zero
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    
    # Avoid division by zero
    mask = denominator != 0
    if not mask.any():
        return 0.0
    
    smape_val = np.mean(np.abs(actual[mask] - forecast[mask]) / denominator[mask])
    return smape_val


def mase(actual, forecast, insample, seasonal_period: int = 12):
    """Compute Mean Absolute Scaled Error (MASE) using the standard Hyndman definition.

    MASE = MAE(forecast) / MAE(naive seasonal differences from insample)
    where denominator = mean_{t=m+1..n} |Y_t - Y_{t-m}|.

    Args:
        actual: out-of-sample actual values
        forecast: corresponding forecast values
        insample: in-sample (training) historical series used for scaling
        seasonal_period: seasonality (m). Defaults to 12 for monthly data.

    Returns:
        float MASE value (np.nan if cannot be computed). 1.0 means equal to seasonal naive.
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    insample = np.asarray(insample)
    m = int(max(1, seasonal_period))
    if insample.size <= m:
        return np.nan
    scale_diffs = np.abs(insample[m:] - insample[:-m])
    denom = scale_diffs.mean() if scale_diffs.size else np.nan
    if denom is None or not np.isfinite(denom) or denom == 0:
        return 0.0 if np.allclose(actual, forecast) else np.inf
    mae_forecast = np.mean(np.abs(actual - forecast))
    return mae_forecast / denom


def rmse(actual, forecast):
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        actual: Array of actual values
        forecast: Array of forecasted values
    
    Returns:
        RMSE value (lower is better)
    """
    return np.sqrt(mean_squared_error(actual, forecast))


def _robust_mape(actual, forecast):
    """Robust MAPE that avoids explosion on zeros.

    Uses |A - F| / max(|A|, epsilon) with epsilon derived from non-zero actual scale.
    Returns value in 0..inf (lower better)."""
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    nz = np.abs(actual[actual != 0])
    if nz.size == 0:
        # All zeros: treat any forecast error relative to 1 to avoid div-by-zero explosion
        epsilon = 1.0
    else:
        epsilon = max(1e-6, 0.1 * np.median(nz))
    denom = np.where(np.abs(actual) < epsilon, epsilon, np.abs(actual))
    return float(np.mean(np.abs(actual - forecast) / denom))


def calculate_validation_metrics(actual, forecast, train_data, seasonal_period: int = 12):
    """
    Calculate all validation metrics (WAPE, SMAPE, MASE, RMSE).
    
    Args:
        actual: Actual validation values
        forecast: Forecasted validation values
        train_data: Training data for seasonal naive calculation
    
    Returns:
        Tuple of (wape, smape, mase, rmse) — return arity/order unchanged (slot 0 now WAPE)
    """
    # PRIMARY (WAPE)
    try:
        wape_val = wape(actual, forecast)
    except Exception:
        wape_val = np.inf

    # Keep the others for diagnostics
    try:
        smape_val = smape(actual, forecast)
    except Exception:
        smape_val = 1.0

    try:
        mase_val = mase(actual, forecast, train_data, seasonal_period=seasonal_period)
    except Exception:
        mase_val = np.nan

    try:
        rmse_val = rmse(actual, forecast)
    except Exception:
        rmse_val = np.nan

    return wape_val, smape_val, mase_val, rmse_val


def calculate_forecast_accuracy_summary(results_dict):
    """
    Calculate summary accuracy metrics across all models and products.
    
    Args:
        results_dict: Dictionary of {model_name: DataFrame} with forecast results
    
    Returns:
        Dictionary with accuracy summary statistics
    """
    summary = {}
    
    for model_name, df in results_dict.items():
        # Extract actual vs forecast data
        actual_data = df[df['Type'] == 'actual']
        forecast_data = df[df['Type'] == 'forecast']
        
        if len(actual_data) == 0 or len(forecast_data) == 0:
            continue
        
        model_metrics = []
        
        # Calculate metrics per product
        for product in df['Product'].unique():
            product_actual = actual_data[actual_data['Product'] == product]['ACR']
            product_forecast = forecast_data[forecast_data['Product'] == product]['ACR']
            
            if len(product_actual) > 0 and len(product_forecast) > 0:
                # For accuracy calculation, we need overlapping periods
                # This is a simplified version - in practice, you'd align dates
                min_len = min(len(product_actual), len(product_forecast))
                if min_len > 0:
                    mape_val = _robust_mape(
                        product_actual.iloc[-min_len:], 
                        product_forecast.iloc[:min_len]
                    )
                    model_metrics.append(mape_val)
        
        if model_metrics:
            summary[model_name] = {
                'mean_mape': np.mean(model_metrics),
                'median_mape': np.median(model_metrics),
                'std_mape': np.std(model_metrics),
                'min_mape': np.min(model_metrics),
                'max_mape': np.max(model_metrics),
                'product_count': len(model_metrics)
            }
    
    return summary


def walk_forward_validation(series, model_fitting_func, window_size=24, step_size=1, horizon=12,
                           model_params=None, diagnostic_messages=None, gap: int = 0):
    """
    Expanding-window cross-validation for robust WAPE calculation.

    - Uses expanding train window starting at len == max(window_size, 24)
    - Advances by step_size each fold; fixed horizon per fold
    - Returns fold-level WAPE list and summary stats (mean, p75, p95), plus MASE per fold

    Args:
        series: Pandas Series of monthly values
        model_fitting_func: Callable(train_data, **params) -> fitted model with forecast()/predict()
        window_size: minimum training size (months), default 24
        step_size: fold increment (months), set equal to validation_horizon for policy
        horizon: validation horizon per fold (months)
        gap: gap between train end and validation start
        model_params: optional dict of params for fitting function
        diagnostic_messages: list for status logs

    Returns:
        dict with keys: mean_mape, p75_mape, p95_mape, mean_mase, iterations, mape_scores, mase_scores, validation_results
    """
    if model_params is None:
        model_params = {}
    
    # Per-fold WAPE list (internally called mape historically)
    mape_scores = []  # WAPE per fold
    smape_scores = []
    rmse_scores = []
    mase_scores = []
    validation_results = []
    
    # Ensure we have enough data for walk-forward validation
    min_train = max(24, int(window_size))
    min_required = min_train + max(0, gap) + horizon
    if len(series) < min_required:
        if diagnostic_messages:
            diagnostic_messages.append(f"⚠️ Walk-forward validation: Need {min_required} months, have {len(series)}. Using single split.")
        return None

    # Expanding window: train_end grows by step_size; start at 0
    max_iterations = max(0, (len(series) - min_train - max(0, gap) - horizon) // step_size + 1)
    max_reasonable_iterations = min(50, len(series) // max(1, step_size))
    max_iterations = min(max_reasonable_iterations, max_iterations)

    for i in range(max_iterations):
        train_end_idx = min_train + i * step_size
        val_start_idx = train_end_idx + max(0, gap)
        val_end_idx = val_start_idx + horizon

        if val_end_idx > len(series):
            break

        train_data = series.iloc[:train_end_idx]
        actual_data = series.iloc[val_start_idx:val_end_idx]

        try:
            # Fit model and generate forecast
            fitted_model = model_fitting_func(train_data, **model_params)
            if hasattr(fitted_model, 'forecast'):
                forecast = fitted_model.forecast(len(actual_data))
            elif hasattr(fitted_model, 'predict'):
                forecast = fitted_model.predict(start=len(train_data), end=len(train_data) + len(actual_data) - 1)
            else:
                continue
                
            # Calculate metrics (slot 0 now WAPE)
            mape = wape(actual_data, forecast)
            smape_val = smape(actual_data, forecast)
            rmse_val = rmse(actual_data, forecast)
            mase_val = mase(actual_data, forecast, train_data, seasonal_period=12)
            
            mape_scores.append(mape)
            smape_scores.append(smape_val)
            rmse_scores.append(rmse_val)
            mase_scores.append(mase_val)
            
            validation_results.append({
                'iteration': len(mape_scores),
                'train_start': series.index[0],
                'train_end': series.index[train_end_idx - 1],
                'val_start': series.index[val_start_idx],
                'val_end': series.index[val_end_idx - 1],
                'mape': mape,
                'smape': smape_val,
                'rmse': rmse_val,
                'mase': mase_val
            })

        except Exception as e:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Walk-forward iteration {len(mape_scores) + 1} failed: {str(e)[:50]}")
            continue
    
    if not mape_scores:
        return None
    
    mape_arr = np.array(mape_scores)
    # Recency‑weighted mean WAPE (double weight on the most recent fold)
    try:
        n_folds = len(mape_arr)
        if n_folds > 0:
            # Exponential weights from oldest->newest with ratio 2
            exponents = np.linspace(0, 1, n_folds)
            weights = np.power(2.0, exponents)
            rw_mape = float(np.sum(weights * mape_arr) / np.sum(weights))
        else:
            rw_mape = float(np.mean(mape_arr)) if mape_arr.size else np.inf
    except Exception:
        rw_mape = float(np.mean(mape_arr)) if mape_arr.size else np.inf
    # Recent p75 computed over the most recent half of folds (>=2)
    try:
        recent_k = max(2, int(max(1, len(mape_arr)) * 0.5))
        recent_slice = mape_arr[-recent_k:]
        recent_p75 = float(np.percentile(recent_slice, 75)) if recent_slice.size else float(np.percentile(mape_arr, 75))
    except Exception:
        recent_p75 = float(np.percentile(mape_arr, 75)) if mape_arr.size else np.inf
    # Standardized WAPE keys (new) while preserving legacy keys for compatibility
    results = {
        # New standardized keys
        'mean_wape': float(np.mean(mape_arr)),
        'p75_wape': float(np.percentile(mape_arr, 75)),
        'p95_wape': float(np.percentile(mape_arr, 95)),
        'wapes_by_fold': mape_scores,
        'folds': len(mape_scores),
        'recent_weighted_wape': rw_mape,
        # Legacy keys maintained to avoid breaking older readers
        'mean_mape': float(np.mean(mape_arr)),
        'std_mape': float(np.std(mape_arr)),
        'median_mape': float(np.median(mape_arr)),
        'min_mape': float(np.min(mape_arr)),
        'max_mape': float(np.max(mape_arr)),
        'recent_weighted_mape': rw_mape,
        'recent_p75_mape': recent_p75,
        'p75_mape': float(np.percentile(mape_arr, 75)),
        'p95_mape': float(np.percentile(mape_arr, 95)),
        'mean_smape': float(np.mean(smape_scores)),
        'mean_rmse': float(np.mean(rmse_scores)),
        'mean_mase': float(np.nanmean(mase_scores)) if mase_scores else np.nan,
        'iterations': len(mape_scores),
        'validation_results': validation_results,
        'mape_scores': mape_scores,
        'mase_scores': mase_scores
    }
    
    if diagnostic_messages:
        diagnostic_messages.append(
            f"📊 Walk-forward CV: {len(mape_scores)} folds (gap={max(0, gap)}), "
            f"Mean WAPE: {np.mean(mape_scores):.1%} (p75 {np.percentile(mape_arr,75):.1%}, p95 {np.percentile(mape_arr,95):.1%})"
        )
    
    return results


def time_series_cross_validation(series, model_fitting_func, n_splits=5, horizon=12,
                                 model_params=None, diagnostic_messages=None, gap: int = 0):
    """
    Perform time series cross-validation for robust model evaluation.
    
    Args:
        series: Time series data
        model_fitting_func: Function that takes (train_data, **model_params) and returns fitted model
        n_splits: Number of cross-validation splits
    horizon: Forecast horizon for each split (must be >= 1 and fixed)
    gap: Optional number of periods between the end of train and start of validation to reduce leakage
        model_params: Dictionary of parameters to pass to model fitting function
        diagnostic_messages: List to append diagnostic messages
    
    Returns:
        Dictionary with cross-validation results
    """
    if model_params is None:
        model_params = {}
    
    min_train_size = 24  # Minimum 2 years for training
    total_required = min_train_size + max(0, gap) + (n_splits * horizon)
    
    if len(series) < total_required:
        if diagnostic_messages:
            diagnostic_messages.append(f"⚠️ Cross-validation: Need {total_required} months, have {len(series)}. Reducing splits.")
    n_splits = max(1, (len(series) - min_train_size - max(0, gap)) // horizon)
    
    mape_scores = []
    validation_results = []
    
    # Calculate split points
    available_data = len(series) - min_train_size - max(0, gap)
    split_increment = max(1, available_data // n_splits)
    
    for i in range(n_splits):
        train_end_idx = min_train_size + (i * split_increment)
        val_start_idx = train_end_idx + max(0, gap)
        val_end_idx = val_start_idx + horizon

        if val_end_idx > len(series):
            break

        train_data = series.iloc[:train_end_idx]
        actual_data = series.iloc[val_start_idx:val_end_idx]

        # Enforce fixed horizon; skip if not enough room
        if len(actual_data) < horizon:
            continue

        try:
            # Fit model and generate forecast
            fitted_model = model_fitting_func(train_data, **model_params)
            if hasattr(fitted_model, 'forecast'):
                forecast = fitted_model.forecast(len(actual_data))
            elif hasattr(fitted_model, 'predict'):
                forecast = fitted_model.predict(start=len(train_data), end=len(train_data) + len(actual_data) - 1)
            else:
                continue
                
            # Calculate metrics (slot 0 now WAPE)
            mape = wape(actual_data, forecast)
            mape_scores.append(mape)
            
            validation_results.append({
                'fold': i + 1,
                'train_size': len(train_data),
                'val_size': len(actual_data),
                'mape': mape,
                'train_end': series.index[train_end_idx - 1],
                'val_start': series.index[val_start_idx],
                'val_end': series.index[val_end_idx - 1]
            })

        except Exception as e:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Cross-validation fold {i + 1} failed: {str(e)[:50]}")
            continue
    
    if not mape_scores:
        return None
    
    results = {
        'mean_mape': np.mean(mape_scores),
        'std_mape': np.std(mape_scores),
        'folds_completed': len(mape_scores),
        'validation_results': validation_results,
        'mape_scores': mape_scores
    }
    
    if diagnostic_messages:
        diagnostic_messages.append(
            f"🔄 Cross-validation: {len(mape_scores)} folds (gap={max(0, gap)}), "
            f"Mean WAPE: {np.mean(mape_scores):.1%} ± {np.std(mape_scores):.1%}"
        )
    
    return results


def enhanced_mape_analysis(actual, forecast, dates=None, product_name=""):
    """
    Provide detailed MAPE analysis including confidence intervals and error distribution.
    
    Args:
        actual: Array of actual values
        forecast: Array of forecasted values
        dates: Array of dates corresponding to the values
        product_name: Name of the product for reporting
    
    Returns:
        Dictionary with detailed MAPE analysis
    """
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    # Calculate basic WAPE (stored under legacy 'mape' key for compatibility)
    mape = wape(actual, forecast)
    
    # Calculate period-by-period percentage errors
    mask = actual != 0  # Avoid division by zero
    percentage_errors = np.zeros_like(actual, dtype=float)
    eps = 1e-6 if np.sum(mask)==0 else max(1e-6, 0.1 * np.median(np.abs(actual[mask])))
    denom = np.where(np.abs(actual) < eps, eps, np.abs(actual))
    percentage_errors = np.abs(actual - forecast) / denom * 100
    
    # Calculate confidence intervals
    mape_ci_lower = np.percentile(percentage_errors[mask], 25)
    mape_ci_upper = np.percentile(percentage_errors[mask], 75)
    
    # Identify outlier periods
    outlier_threshold = np.percentile(percentage_errors[mask], 95)
    outlier_periods = np.where(percentage_errors > outlier_threshold)[0]
    
    # Calculate directional bias
    if mask.any():
        signed_errors = ((forecast - actual) / denom) * 100
    else:
        signed_errors = np.zeros_like(actual, dtype=float)
    bias = np.mean(signed_errors)
    
    analysis = {
        'mape': mape,
        'mape_std': np.std(percentage_errors[mask]),
        'mape_ci_lower': mape_ci_lower,
        'mape_ci_upper': mape_ci_upper,
        'bias': bias,  # Positive = over-forecasting, Negative = under-forecasting
        'outlier_periods': outlier_periods.tolist(),
        'outlier_count': len(outlier_periods),
        'worst_month_error': np.max(percentage_errors[mask]) if mask.any() else 0,
        'best_month_error': np.min(percentage_errors[mask]) if mask.any() else 0,
        'periods_analyzed': np.sum(mask)
    }
    
    # Add date information if available
    if dates is not None and len(dates) == len(actual):
        dates = pd.to_datetime(dates)
        if len(outlier_periods) > 0:
            analysis['outlier_dates'] = [dates[i].strftime('%Y-%m') for i in outlier_periods]
        
        # Find worst and best performing months
        if mask.any():
            worst_idx = np.argmax(percentage_errors)
            best_idx = np.argmin(percentage_errors[mask])
            analysis['worst_month_date'] = dates[worst_idx].strftime('%Y-%m')
            analysis['best_month_date'] = dates[np.where(mask)[0][best_idx]].strftime('%Y-%m')
    
    return analysis


def seasonal_mape_analysis(actual, forecast, dates, product_name="", fiscal_year_start_month: int = 1):
    """
    Analyze MAPE performance by season/month for seasonal patterns.
    
    Args:
        actual: Array of actual values
        forecast: Array of forecasted values
        dates: Array of dates corresponding to the values
        product_name: Name of the product for reporting
    
    Returns:
        Dictionary with seasonal MAPE analysis
    """
    df = pd.DataFrame({
        'actual': actual,
        'forecast': forecast,
        'date': pd.to_datetime(dates)
    })
    
    # Remove zero actual values to avoid division errors
    df = df[df['actual'] != 0].copy()
    
    if len(df) == 0:
        return {'error': 'No valid data points for seasonal analysis'}
    
    # Calendar month/quarter
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['ape'] = np.abs((df['actual'] - df['forecast']) / df['actual']) * 100
    
    # Monthly analysis
    monthly_stats = df.groupby('month')['ape'].agg(['mean', 'std', 'count']).round(1)
    monthly_mape = monthly_stats['mean'].to_dict()
    monthly_std = monthly_stats['std'].fillna(0).to_dict()
    monthly_counts = monthly_stats['count'].to_dict()
    
    # Quarterly analysis
    quarterly_stats = df.groupby('quarter')['ape'].agg(['mean', 'std', 'count']).round(1)
    quarterly_mape = quarterly_stats['mean'].to_dict()
    quarterly_std = quarterly_stats['std'].fillna(0).to_dict()
    quarterly_counts = quarterly_stats['count'].to_dict()
    
    # Seasonal patterns (calendar)
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                  7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    quarter_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
    
    analysis = {
        'monthly_mape': monthly_mape,
        'monthly_std': monthly_std,
        'monthly_counts': monthly_counts,
        'quarterly_mape': quarterly_mape,
        'quarterly_std': quarterly_std,
        'quarterly_counts': quarterly_counts,
        'worst_performing_month': monthly_stats['mean'].idxmax() if len(monthly_stats) > 0 else None,
        'best_performing_month': monthly_stats['mean'].idxmin() if len(monthly_stats) > 0 else None,
        'worst_performing_quarter': quarterly_stats['mean'].idxmax() if len(quarterly_stats) > 0 else None,
        'best_performing_quarter': quarterly_stats['mean'].idxmin() if len(quarterly_stats) > 0 else None,
        'total_periods': len(df)
    }
    
    # Add readable names
    if analysis['worst_performing_month']:
        analysis['worst_month_name'] = month_names.get(analysis['worst_performing_month'], 'Unknown')
        analysis['worst_month_mape'] = monthly_mape.get(analysis['worst_performing_month'], 0)
    
    if analysis['best_performing_month']:
        analysis['best_month_name'] = month_names.get(analysis['best_performing_month'], 'Unknown')
        analysis['best_month_mape'] = monthly_mape.get(analysis['best_performing_month'], 0)
    
    if analysis['worst_performing_quarter']:
        analysis['worst_quarter_name'] = quarter_names.get(analysis['worst_performing_quarter'], 'Unknown')
        analysis['worst_quarter_mape'] = quarterly_mape.get(analysis['worst_performing_quarter'], 0)
    
    if analysis['best_performing_quarter']:
        analysis['best_quarter_name'] = quarter_names.get(analysis['best_performing_quarter'], 'Unknown')
        analysis['best_quarter_mape'] = quarterly_mape.get(analysis['best_performing_quarter'], 0)
    
    # If a fiscal year start is provided (other than January), compute fiscal-season views
    try:
        fy_start = int(fiscal_year_start_month) if fiscal_year_start_month is not None else 1
    except Exception:
        fy_start = 1

    if fy_start and 1 <= fy_start <= 12 and fy_start != 1:
        # Map to fiscal month (1..12 where 1 == fiscal start month)
        # fm = ((calendar_month - fy_start + 12) % 12) + 1
        df['fiscal_month'] = ((df['month'] - fy_start + 12) % 12) + 1
        df['fiscal_quarter'] = ((df['fiscal_month'] - 1) // 3) + 1

        fiscal_monthly_stats = df.groupby('fiscal_month')['ape'].agg(['mean', 'std', 'count']).round(1)
        fiscal_quarterly_stats = df.groupby('fiscal_quarter')['ape'].agg(['mean', 'std', 'count']).round(1)

        fiscal_monthly_mape = fiscal_monthly_stats['mean'].to_dict()
        fiscal_monthly_std = fiscal_monthly_stats['std'].fillna(0).to_dict()
        fiscal_monthly_counts = fiscal_monthly_stats['count'].to_dict()

        fiscal_quarterly_mape = fiscal_quarterly_stats['mean'].to_dict()
        fiscal_quarterly_std = fiscal_quarterly_stats['std'].fillna(0).to_dict()
        fiscal_quarterly_counts = fiscal_quarterly_stats['count'].to_dict()

        # Build fiscal month name mapping rotated to start at fy_start
        base_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        offset = (fy_start - 1) % 12
        rotated = base_months[offset:] + base_months[:offset]
        fiscal_month_names = {i + 1: rotated[i] for i in range(12)}
        fiscal_quarter_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}

        analysis.update({
            'fiscal_year_start_month': fy_start,
            'fiscal_monthly_mape': fiscal_monthly_mape,
            'fiscal_monthly_std': fiscal_monthly_std,
            'fiscal_monthly_counts': fiscal_monthly_counts,
            'fiscal_quarterly_mape': fiscal_quarterly_mape,
            'fiscal_quarterly_std': fiscal_quarterly_std,
            'fiscal_quarterly_counts': fiscal_quarterly_counts,
            'worst_performing_fiscal_month': fiscal_monthly_stats['mean'].idxmax() if len(fiscal_monthly_stats) > 0 else None,
            'best_performing_fiscal_month': fiscal_monthly_stats['mean'].idxmin() if len(fiscal_monthly_stats) > 0 else None,
            'worst_performing_fiscal_quarter': fiscal_quarterly_stats['mean'].idxmax() if len(fiscal_quarterly_stats) > 0 else None,
            'best_performing_fiscal_quarter': fiscal_quarterly_stats['mean'].idxmin() if len(fiscal_quarterly_stats) > 0 else None,
            'fiscal_month_names': fiscal_month_names,
            'fiscal_quarter_names': fiscal_quarter_names,
        })

        # Add readable fiscal names and values
        if analysis.get('worst_performing_fiscal_month'):
            m = analysis['worst_performing_fiscal_month']
            analysis['worst_fiscal_month_name'] = fiscal_month_names.get(m, 'Unknown')
            analysis['worst_fiscal_month_mape'] = fiscal_monthly_mape.get(m, 0)
        if analysis.get('best_performing_fiscal_month'):
            m = analysis['best_performing_fiscal_month']
            analysis['best_fiscal_month_name'] = fiscal_month_names.get(m, 'Unknown')
            analysis['best_fiscal_month_mape'] = fiscal_monthly_mape.get(m, 0)
        if analysis.get('worst_performing_fiscal_quarter'):
            q = analysis['worst_performing_fiscal_quarter']
            analysis['worst_fiscal_quarter_name'] = fiscal_quarter_names.get(q, 'Unknown')
            analysis['worst_fiscal_quarter_mape'] = fiscal_quarterly_mape.get(q, 0)
        if analysis.get('best_performing_fiscal_quarter'):
            q = analysis['best_performing_fiscal_quarter']
            analysis['best_fiscal_quarter_name'] = fiscal_quarter_names.get(q, 'Unknown')
            analysis['best_fiscal_quarter_mape'] = fiscal_quarterly_mape.get(q, 0)

    return analysis


def simple_backtesting_validation(series, model_fitting_func, backtest_months=12, backtest_gap=1, 
                                 validation_horizon=12, model_params=None, diagnostic_messages=None):
    """
    Simple backtesting validation using a single train/test split.
    
    Args:
        series: Full time series data
        model_fitting_func: Function for fitting models
        backtest_months: Number of months to use for backtesting
        backtest_gap: Gap between training and test data (months)
        validation_horizon: How far ahead to predict in backtesting
        model_params: Parameters for model fitting
        diagnostic_messages: List to append diagnostic messages
    
    Returns:
        Dictionary with backtesting results or None if validation fails
    """
    try:
        # Calculate minimum data requirements
        min_training_data = 12  # Minimum months needed for training
        min_test_data = 3       # Minimum months needed for testing
        total_required = backtest_months + validation_horizon + backtest_gap + min_training_data
        
        if len(series) < total_required:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Insufficient data for {backtest_months} month backtesting. Need at least {total_required} months (have {len(series)}).")
            return None
        
        # Split data: train on everything except last (backtest_months + gap) months
        split_point = len(series) - backtest_months - backtest_gap
        train_data = series.iloc[:split_point]
        test_data = series.iloc[split_point:split_point + validation_horizon]
        
        if len(train_data) < 12 or len(test_data) < 3:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Split resulted in insufficient train/test data: {len(train_data)} train, {len(test_data)} test months.")
            return None
        
        # Fit model on training data
        try:
            fitted_model = model_fitting_func(train_data, **model_params) if model_params else model_fitting_func(train_data)
            
            # Generate predictions for test period
            try:
                if hasattr(fitted_model, 'forecast'):
                    forecast = fitted_model.forecast(steps=len(test_data))
                elif hasattr(fitted_model, 'predict'):
                    # For models that need future dates
                    future_dates = pd.date_range(
                        start=test_data.index[0], 
                        periods=len(test_data), 
                        freq='MS'
                    )
                    forecast = fitted_model.predict(future_dates)
                else:
                    if diagnostic_messages:
                        diagnostic_messages.append(f"⚠️ Model doesn't support forecasting")
                    return None
            except Exception as forecast_error:
                if diagnostic_messages:
                    diagnostic_messages.append(f"⚠️ Model forecasting failed: {str(forecast_error)[:50]}")
                return None
            
            # Calculate validation metrics
            mape, smape_val, mase_val, rmse_val = calculate_validation_metrics(
                test_data.values, forecast, train_data.values
            )
            
            # Enhanced MAPE analysis for the backtesting period
            enhanced = enhanced_mape_analysis(
                test_data.values, forecast, test_data.index, ""
            )
            
            return {
                'mape': mape,
                'wape': wape(test_data.values, forecast),  # Add WAPE for consistency
                'smape': smape_val,
                'mase': mase_val,
                'rmse': rmse_val,
                'train_months': len(train_data),
                'test_months': len(test_data),
                'backtest_period': backtest_months,
                'gap': backtest_gap,
                'validation_horizon': validation_horizon,
                'enhanced_analysis': enhanced,
                'method': 'simple_backtesting',
                'validation_type': 'simple',  # For UI display
                'aggregation_method': 'single_fold_mape',  # For UI display
                'validation_completed': True,  # For UI display
                'num_folds': 1,  # For UI display
                'training_window_range': f"{len(train_data)} months",  # For UI display
                'success': True,
                # Add data for chart overlay
                'train_data': train_data,
                'test_data': test_data,
                'predictions': forecast
            }
            
        except Exception as e:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Model fitting failed during backtesting: {str(e)[:50]}")
            return None
            
    except Exception as e:
        if diagnostic_messages:
            diagnostic_messages.append(f"⚠️ Backtesting validation failed: {str(e)[:50]}")
        return None


def enhanced_rolling_validation(series, model_fitting_func, min_train_size=12, max_train_size=18, 
                              validation_horizon=3, backtest_months=15, recency_alpha=0.6,
                              model_params=None, diagnostic_messages=None):
    """
    Enhanced rolling validation with 12-18 month training windows and quarterly validation.
    
    Key improvements:
    1. Training windows of 12-18 months (not fixed 24)
    2. Validation horizon = 3 months (quarterly)
    3. Rolling folds until backtest_months is covered
    4. Aggregate fold errors using WAPE, not MAPE
    5. Recency-weighted WAPE with exponential decay (alpha=0.6)
    
    Args:
        series: Pandas Series of monthly values
        model_fitting_func: Callable(train_data, **params) -> fitted model with forecast()/predict()
        min_train_size: minimum training window in months (default 12)
        max_train_size: maximum training window in months (default 18) 
        validation_horizon: validation period per fold in months (default 3)
        backtest_months: total backtesting period in months (default 15)
        recency_alpha: decay factor for older folds (default 0.6)
        model_params: optional dict of params for fitting function
        diagnostic_messages: list for status logs
    
    Returns:
        dict with enhanced metrics including recency-weighted WAPE
    """
    if model_params is None:
        model_params = {}
    
    fold_wapes = []
    fold_smapes = []
    fold_mases = []
    validation_results = []
    
    # Initialize chart data variables (will store most recent fold data)
    chart_train_data = None
    chart_test_data = None
    chart_predictions = None
    
    # Ensure we have enough data
    min_required = max_train_size + backtest_months
    if len(series) < min_required:
        if diagnostic_messages:
            diagnostic_messages.append(f"⚠️ Enhanced validation: Need {min_required} months, have {len(series)}.")
        return None

    # Calculate rolling fold positions working backwards from the end
    series_end = len(series)
    backtest_start = series_end - backtest_months
    
    # Generate folds covering the backtest period
    fold_idx = 0
    current_val_end = series_end
    
    while current_val_end > backtest_start + validation_horizon:
        # Validation window for this fold
        val_start_idx = current_val_end - validation_horizon
        val_end_idx = current_val_end
        
        # Skip if validation window goes beyond backtest start
        if val_start_idx < backtest_start:
            break
            
        # Dynamic training window size - grow with more data available
        available_train_data = val_start_idx
        train_size = min(max_train_size, max(min_train_size, available_train_data))
        train_start_idx = max(0, val_start_idx - train_size)
        
        # Skip if insufficient training data
        if val_start_idx - train_start_idx < min_train_size:
            break
            
        # Extract data for this fold
        train_data = series.iloc[train_start_idx:val_start_idx]
        actual_data = series.iloc[val_start_idx:val_end_idx]
        
        # Skip if insufficient data
        if len(train_data) < min_train_size or len(actual_data) < validation_horizon:
            current_val_end -= validation_horizon  # Move to next fold
            continue

        try:
            # Fit model and generate forecast
            fitted_model = model_fitting_func(train_data, **model_params)
            if hasattr(fitted_model, 'forecast'):
                forecast = fitted_model.forecast(len(actual_data))
            elif hasattr(fitted_model, 'predict'):
                forecast = fitted_model.predict(start=len(train_data), end=len(train_data) + len(actual_data) - 1)
            else:
                current_val_end -= validation_horizon
                continue
                
            # Calculate WAPE (not MAPE) for each fold
            wape_val = wape(actual_data, forecast)
            smape_val = smape(actual_data, forecast)
            mase_val = mase(actual_data, forecast, train_data, seasonal_period=12)
            
            fold_wapes.append(wape_val)
            fold_smapes.append(smape_val)
            fold_mases.append(mase_val)
            
            validation_results.append({
                'fold': fold_idx + 1,
                'train_start': series.index[train_start_idx],
                'train_end': series.index[val_start_idx - 1], 
                'val_start': series.index[val_start_idx],
                'val_end': series.index[val_end_idx - 1],
                'train_size': len(train_data),
                'val_size': len(actual_data),
                'wape': wape_val,
                'smape': smape_val,
                'mase': mase_val,
                'is_most_recent': fold_idx == 0,
                # Include per‑fold series for visualization of all folds
                'val_dates': list(actual_data.index),
                'y_true': list(np.asarray(actual_data, dtype=float)),
                'y_pred': list(np.asarray(forecast, dtype=float))
            })
            
            # Store chart data from most recent fold (fold_idx == 0) for chart overlay
            if fold_idx == 0:
                chart_train_data = train_data.copy()
                chart_test_data = actual_data.copy()  # Use actual test data with proper index
                chart_predictions = forecast
            
            fold_idx += 1

        except Exception as e:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Enhanced validation fold {fold_idx + 1} failed: {str(e)[:50]}")
        
        # Move to next fold (step back by validation_horizon)
        current_val_end -= validation_horizon
    
    if not fold_wapes or len(fold_wapes) < 2:
        if diagnostic_messages:
            diagnostic_messages.append(f"⚠️ Enhanced validation: Only {len(fold_wapes)} successful folds, need at least 2.")
        return None
    
    # Convert to numpy arrays for calculations
    wape_arr = np.array(fold_wapes)
    
    # Calculate recency weights with exponential decay (newest folds get highest weight)
    # weights = alpha ** np.arange(len(fold_wapes)-1, -1, -1)
    weights = recency_alpha ** np.arange(len(fold_wapes) - 1, -1, -1)
    weights_normalized = weights / np.sum(weights)
    
    # Calculate weighted WAPE (newest folds weighted more heavily)
    recent_weighted_wape = float(np.dot(wape_arr, weights_normalized))
    
    # Calculate standard metrics
    mean_wape = float(np.mean(wape_arr))
    p75_wape = float(np.percentile(wape_arr, 75))
    p95_wape = float(np.percentile(wape_arr, 95))
    
    # Calculate other standard metrics
    mean_smape = float(np.mean(fold_smapes))
    mean_mase = float(np.nanmean(fold_mases)) if fold_mases else np.nan
    
    # Trend analysis - are recent folds performing better?
    trend_improving = False
    trend_slope = 0.0
    if len(wape_arr) >= 3:
        # Simple linear trend: negative slope = improving (lower WAPE over time)
        fold_indices = np.arange(len(wape_arr))
        trend_slope = np.polyfit(fold_indices, wape_arr, 1)[0]
        trend_improving = trend_slope < 0
    
    # Stability metrics
    std_wape = float(np.std(wape_arr))
    fold_consistency = 1.0 - (std_wape / mean_wape) if mean_wape > 0 else 0.0
    
    results = {
        # Primary metrics (using WAPE instead of MAPE)
        'mape': recent_weighted_wape,  # Primary metric for compatibility
        'wape': recent_weighted_wape,  # Explicit WAPE
        'mean_wape': mean_wape,
        'p75_wape': p75_wape,
        'p95_wape': p95_wape,
        'recent_weighted_wape': recent_weighted_wape,
        
        # Other metrics
        'smape': mean_smape,
        'mase': mean_mase,
        'mean_mase': mean_mase,
        
        # Stability and trend
        'std_wape': std_wape,
        'fold_consistency': float(fold_consistency),
        'trend_slope': float(trend_slope),
        'trend_improving': bool(trend_improving),
        
        # Metadata
        'folds': len(fold_wapes),
        'iterations': len(fold_wapes),  # Compatibility alias
        'validation_horizon': validation_horizon,
        'min_train_size': min_train_size,
        'max_train_size': max_train_size,
        'backtest_months': backtest_months,
        'recency_alpha': recency_alpha,
        'method': 'enhanced_rolling',
        'validation_type': 'enhanced_rolling',  # For UI display
        'aggregation_method': 'recency_weighted_wape',  # For UI display
        'validation_completed': True,  # For UI display
        'num_folds': len(fold_wapes),  # For UI display
        'training_window_range': f"{min_train_size}-{max_train_size} months",  # For UI display
        'success': True,
        
        # Raw data
        'validation_results': validation_results,
        'fold_wapes': fold_wapes,
        'fold_smapes': fold_smapes,
        'fold_mases': fold_mases,
        'weights': weights_normalized.tolist(),
        
        # For backtesting compatibility
        'train_months': validation_results[-1]['train_size'] if validation_results else 0,
        'test_months': validation_horizon,
        'backtest_period': backtest_months,
        'gap': 0,
        
        # Chart data from most recent fold for backtesting overlay
        'train_data': chart_train_data,
        'test_data': chart_test_data,
        'predictions': chart_predictions
    }
    
    if diagnostic_messages:
        diagnostic_messages.append(
            f"📊 Enhanced rolling validation: {len(fold_wapes)} folds, "
            f"Weighted WAPE: {recent_weighted_wape:.1%}, "
            f"Mean WAPE: {mean_wape:.1%}, "
            f"P75: {p75_wape:.1%}, P95: {p95_wape:.1%}, "
            f"α={recency_alpha}, trend={'↗' if trend_improving else '↘'}"
        )
    
    return results


def comprehensive_validation_suite(actual, forecast, dates=None, product_name="",
                                  enable_walk_forward=False, enable_cross_validation=False,
                                  series=None, model_fitting_func=None, model_params=None,
                                  diagnostic_messages=None,
                                  backtest_months=15, backtest_gap=0,
                                  validation_horizon=3, fiscal_year_start_month=1,
                                  enable_enhanced_rolling=True, min_train_size=12, max_train_size=18, recency_alpha=0.6):
    """
    Enhanced validation suite with improved rolling backtesting for hyperscale businesses.
    
    Args:
        actual: Array of actual values for basic validation
        forecast: Array of forecasted values for basic validation
        dates: Array of dates corresponding to the values
        product_name: Name of the product for reporting
        enable_walk_forward: Ignored (kept for compatibility)
        enable_cross_validation: Ignored (kept for compatibility)
        series: Full time series for backtesting
        model_fitting_func: Function for fitting models in backtesting
        model_params: Parameters for model fitting
        diagnostic_messages: List to append diagnostic messages
        backtest_months: Number of months to use for backtesting (default 15)
        backtest_gap: Gap between training and test data (default 0 for faster feedback)
        validation_horizon: How far ahead to predict in backtesting (default 3 for quarterly)
        fiscal_year_start_month: Fiscal year start month
        enable_enhanced_rolling: Use enhanced rolling validation with WAPE and recency weighting
        min_train_size: Minimum training window in months (default 12)
        max_train_size: Maximum training window in months (default 18)
        recency_alpha: Decay factor for recency weighting (default 0.6)
    
    Returns:
        Dictionary with enhanced validation results
    """
    results = {
        'product_name': product_name,
        'basic_validation': {},
        'enhanced_analysis': {},
        'seasonal_analysis': {},
        'backtesting_validation': None,
        'method_recommendation': None
    }
    
    # Basic validation metrics
    train_data = series.iloc[:-len(actual)] if series is not None else []
    mape, smape_val, mase_val, rmse_val = calculate_validation_metrics(actual, forecast, train_data)
    
    results['basic_validation'] = {
        'mape': mape,
        'smape': smape_val,
        'mase': mase_val,
        'rmse': rmse_val
    }
    
    # Enhanced MAPE analysis
    try:
        results['enhanced_analysis'] = enhanced_mape_analysis(actual, forecast, dates, product_name)
    except Exception as e:
        if diagnostic_messages:
            diagnostic_messages.append(f"⚠️ Enhanced MAPE analysis failed for {product_name}: {str(e)[:50]}")
    
    # Seasonal analysis
    if dates is not None and len(dates) >= 12:
        try:
            results['seasonal_analysis'] = seasonal_mape_analysis(
                actual, forecast, dates, product_name, fiscal_year_start_month=fiscal_year_start_month
            )
        except Exception as e:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Seasonal MAPE analysis failed for {product_name}: {str(e)[:50]}")
    
    # Enhanced rolling backtesting validation
    if series is not None and model_fitting_func is not None:
        try:
            if enable_enhanced_rolling:
                # Use enhanced rolling validation with WAPE and recency weighting
                results['backtesting_validation'] = enhanced_rolling_validation(
                    series=series,
                    model_fitting_func=model_fitting_func,
                    min_train_size=min_train_size,
                    max_train_size=max_train_size,
                    validation_horizon=validation_horizon,
                    backtest_months=backtest_months,
                    recency_alpha=recency_alpha,
                    model_params=model_params,
                    diagnostic_messages=diagnostic_messages
                )
            else:
                # Fallback to simple backtesting
                results['backtesting_validation'] = simple_backtesting_validation(
                    series, model_fitting_func, backtest_months, backtest_gap, 
                    validation_horizon, model_params, diagnostic_messages
                )
        except Exception as e:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Backtesting validation failed for {product_name}: {str(e)[:50]}")
            # Fallback to simple backtesting if enhanced fails
            try:
                results['backtesting_validation'] = simple_backtesting_validation(
                    series, model_fitting_func, backtest_months, backtest_gap, 
                    validation_horizon, model_params, diagnostic_messages
                )
            except Exception as e2:
                if diagnostic_messages:
                    diagnostic_messages.append(f"⚠️ Fallback backtesting also failed for {product_name}: {str(e2)[:50]}")
    
    # Method recommendation with enhanced criteria
    try:
        bt = results.get('backtesting_validation')
        if bt and bt.get('success'):
            method_used = bt.get('method', 'simple')
            folds = bt.get('folds', 0)
            
            if method_used == 'enhanced_rolling' and folds >= 4:
                # Enhanced rolling validation with good fold count
                wape_val = bt.get('recent_weighted_wape', bt.get('wape', 0))
                consistency = bt.get('fold_consistency', 0)
                trend_text = "improving" if bt.get('trend_improving') else "stable"
                quality = "high" if folds >= 5 and consistency > 0.7 else "good"
                
                results['method_recommendation'] = {
                    'recommended': 'enhanced_rolling',
                    'reason': f"Enhanced rolling with {folds} folds, {quality} quality, {trend_text} trend",
                    'backtest_wape': wape_val,
                    'backtest_months': bt.get('backtest_months'),
                    'quality': quality,
                    'folds': folds,
                    'recency_weighted': True,
                    'uses_wape': True
                }
            else:
                # Simple backtesting or insufficient folds for enhanced
                mape_val = bt.get('mape', 0)
                results['method_recommendation'] = {
                    'recommended': 'simple_backtesting',
                    'reason': f"Simple backtesting with {folds} folds, WAPE {mape_val:.1%}",
                    'backtest_mape': mape_val,
                    'backtest_months': bt.get('backtest_months'),
                    'fallback_reason': 'insufficient_folds_for_enhanced' if method_used == 'enhanced_rolling' else 'simple_method'
                }
        else:
            results['method_recommendation'] = {
                'recommended': 'basic_only',
                'reason': 'Backtesting failed or insufficient data - using basic validation only',
                'fallback': True
            }
    except Exception:
        # Non-fatal if recommendation fails
        pass

    return results
