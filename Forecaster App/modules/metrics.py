"""
Metrics and evaluation functions for forecasting models.

Contains implementations of MAPE, SMAPE, MASE, RMSE and other
evaluation metrics for time series forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta


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
    Calculate all validation metrics (MAPE, SMAPE, MASE, RMSE).
    
    Args:
        actual: Actual validation values
        forecast: Forecasted validation values
        train_data: Training data for seasonal naive calculation
    
    Returns:
        Tuple of (mape, smape, mase, rmse)
    """
    # Convert to numpy arrays
    actual = np.array(actual)
    forecast = np.array(forecast)
    train_data = np.array(train_data)
    
    # Robust MAPE ignoring zero explosions
    try:
        mape_val = _robust_mape(actual, forecast)
    except Exception:
        mape_val = np.inf
    
    # Calculate SMAPE
    try:
        smape_val = smape(actual, forecast)
    except Exception:
        smape_val = 1.0
    
    # Calculate correct MASE scaling
    try:
        mase_val = mase(actual, forecast, train_data, seasonal_period=seasonal_period)
    except Exception:
        mase_val = np.nan
    
    # Calculate RMSE
    try:
        rmse_val = rmse(actual, forecast)
    except Exception:
        rmse_val = np.nan
    
    return mape_val, smape_val, mase_val, rmse_val


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
    Perform walk-forward validation for more robust MAPE calculation.
    
    Args:
        series: Time series data
        model_fitting_func: Function that takes (train_data, **model_params) and returns fitted model
    window_size: Training window size in months
    step_size: How many months to advance window each iteration
    horizon: Forecast horizon for each iteration (must be >= 1)
    gap: Optional number of periods between the end of train and start of validation to reduce leakage
        model_params: Dictionary of parameters to pass to model fitting function
        diagnostic_messages: List to append diagnostic messages
    
    Returns:
        Dictionary with validation results and metrics
    """
    if model_params is None:
        model_params = {}
    
    mape_scores = []
    smape_scores = []
    rmse_scores = []
    validation_results = []
    
    # Ensure we have enough data for walk-forward validation
    min_required = window_size + max(0, gap) + horizon
    if len(series) < min_required:
        if diagnostic_messages:
            diagnostic_messages.append(f"⚠️ Walk-forward validation: Need {min_required} months, have {len(series)}. Using single split.")
        return None
    
    max_iterations = max(0, (len(series) - window_size - max(0, gap) - horizon) // step_size + 1)
    # ROBUSTNESS FIX: Remove arbitrary 10-iteration cap to allow more thorough validation
    # Cap at reasonable maximum based on data length to prevent excessive computation
    max_reasonable_iterations = min(50, len(series) // 6)  # At most 50 iterations or series_length/6
    max_iterations = min(max_reasonable_iterations, max_iterations)
    
    for i in range(0, max_iterations * step_size, step_size):
        start_idx = i
        train_end_idx = start_idx + window_size
        val_start_idx = train_end_idx + max(0, gap)
        val_end_idx = val_start_idx + horizon

        if val_end_idx > len(series):
            break

        train_data = series.iloc[start_idx:train_end_idx]
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
                
            # Calculate metrics
            mape = _robust_mape(actual_data, forecast)
            smape_val = smape(actual_data, forecast)
            rmse_val = rmse(actual_data, forecast)
            
            mape_scores.append(mape)
            smape_scores.append(smape_val)
            rmse_scores.append(rmse_val)
            
            validation_results.append({
                'iteration': len(mape_scores),
                'train_start': series.index[start_idx],
                'train_end': series.index[train_end_idx - 1],
                'val_start': series.index[val_start_idx],
                'val_end': series.index[val_end_idx - 1],
                'mape': mape,
                'smape': smape_val,
                'rmse': rmse_val
            })

        except Exception as e:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Walk-forward iteration {len(mape_scores) + 1} failed: {str(e)[:50]}")
            continue
    
    if not mape_scores:
        return None
    
    results = {
        'mean_mape': np.mean(mape_scores),
        'std_mape': np.std(mape_scores),
        'median_mape': np.median(mape_scores),
        'min_mape': np.min(mape_scores),
        'max_mape': np.max(mape_scores),
        'mean_smape': np.mean(smape_scores),
        'mean_rmse': np.mean(rmse_scores),
        'iterations': len(mape_scores),
        'validation_results': validation_results,
        'mape_scores': mape_scores
    }
    
    if diagnostic_messages:
        diagnostic_messages.append(
            f"📊 Walk-forward validation: {len(mape_scores)} iterations (gap={max(0, gap)}), "
            f"Mean MAPE: {np.mean(mape_scores):.1%} ± {np.std(mape_scores):.1%}"
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
                
            # Calculate metrics
            mape = _robust_mape(actual_data, forecast)
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
            f"Mean MAPE: {np.mean(mape_scores):.1%} ± {np.std(mape_scores):.1%}"
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
    
    # Calculate basic MAPE
    mape = _robust_mape(actual, forecast)
    
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
                'smape': smape_val,
                'mase': mase_val,
                'rmse': rmse_val,
                'train_months': len(train_data),
                'test_months': len(test_data),
                'backtest_period': backtest_months,
                'gap': backtest_gap,
                'validation_horizon': validation_horizon,
                'enhanced_analysis': enhanced,
                'success': True
            }
            
        except Exception as e:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Model fitting failed during backtesting: {str(e)[:50]}")
            return None
            
    except Exception as e:
        if diagnostic_messages:
            diagnostic_messages.append(f"⚠️ Backtesting validation failed: {str(e)[:50]}")
        return None


def comprehensive_validation_suite(actual, forecast, dates=None, product_name="",
                                  enable_walk_forward=False, enable_cross_validation=False,
                                  series=None, model_fitting_func=None, model_params=None,
                                  diagnostic_messages=None,
                                  backtest_months=12, backtest_gap=1,
                                  validation_horizon=12, fiscal_year_start_month=1):
    """
    Simplified validation suite that focuses on basic backtesting.
    
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
        backtest_months: Number of months to use for backtesting
        backtest_gap: Gap between training and test data
        validation_horizon: How far ahead to predict in backtesting
        fiscal_year_start_month: Fiscal year start month
    
    Returns:
        Dictionary with validation results
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
    
    # Simple backtesting validation
    if series is not None and model_fitting_func is not None:
        try:
            results['backtesting_validation'] = simple_backtesting_validation(
                series, model_fitting_func, backtest_months, backtest_gap, 
                validation_horizon, model_params, diagnostic_messages
            )
        except Exception as e:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Backtesting validation failed for {product_name}: {str(e)[:50]}")
    
    # Method recommendation
    try:
        bt = results.get('backtesting_validation')
        if bt and bt.get('success'):
            results['method_recommendation'] = {
                'recommended': 'backtesting',
                'reason': f"Backtesting successful with {bt.get('mape', 0):.1%} MAPE over {bt.get('test_months', 0)} months",
                'backtest_mape': bt.get('mape'),
                'backtest_months': bt.get('backtest_period')
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
