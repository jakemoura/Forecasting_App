"""
Model Evaluation and Enhanced Metrics for Outlook Forecaster

Evaluates forecasting models using WAPE, SMAPE, MASE, RMSE and other metrics
to determine the best performing models for quarterly forecasting.
Enhanced with backtesting and walk-forward validation capabilities.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from datetime import datetime, timedelta

# Optional dependencies with graceful fallbacks
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except (ImportError, ValueError):
    HAVE_PROPHET = False
    Prophet = None

try:
    import scipy.sparse
    HAVE_SCIPY_SPARSE = True
except ImportError:
    HAVE_SCIPY_SPARSE = False
    scipy = None

try:
    import xgboost as xgb
    HAVE_XGBOOST = True
except (ImportError, ValueError):
    HAVE_XGBOOST = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAVE_STATSMODELS = True
except (ImportError, ValueError):
    HAVE_STATSMODELS = False
    ARIMA = None
    ExponentialSmoothing = None

from .spike_detection import detect_monthly_spikes
from .forecasting_models import fit_xgboost_model
from .forecasting_models import (
    fit_linear_trend_model,
    fit_moving_average_model,
    fit_prophet_daily_model,
    fit_arima_model,
    fit_exponential_smoothing_model,
    fit_lightgbm_daily_model,
)


# ============================================================================
# Enhanced Metrics Suite (WAPE, SMAPE, MASE, RMSE)
# ============================================================================

def wape(actual, forecast):
    """Weighted Absolute Percentage Error (WAPE).

    Returns a scale-sensitive error aligned to revenue (dollar-weighted):
        sum(|A - F|) / sum(|A|)

    For all-zero actuals, returns 0.0 if forecasts are all zero else +inf.
    """
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


def mase(actual, forecast, insample, seasonal_period: int = 7):
    """Compute Mean Absolute Scaled Error (MASE) using the standard Hyndman definition.

    MASE = MAE(forecast) / MAE(naive seasonal differences from insample)
    where denominator = mean_{t=m+1..n} |Y_t - Y_{t-m}|.

    Args:
        actual: out-of-sample actual values
        forecast: corresponding forecast values
        insample: in-sample (training) historical series used for scaling
        seasonal_period: seasonality (m). Defaults to 7 for daily data.

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


def calculate_validation_metrics(actual, forecast, train_data, seasonal_period: int = 7):
    """
    Calculate all validation metrics (WAPE, SMAPE, MASE, RMSE).
    
    Args:
        actual: Actual validation values
        forecast: Forecasted validation values
        train_data: Training data for seasonal naive calculation
        seasonal_period: Seasonality for MASE (7 for daily data)
    
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


def calculate_mape(actual_values, predicted_values):
    """
    Calculate WAPE (Weighted Absolute Percentage Error) with robust error handling.
    
    Note: This function has been updated to use WAPE instead of MAPE for better 
    performance with revenue forecasting. WAPE is dollar-weighted and more suitable
    for business forecasting where larger values should have more influence.
    
    Args:
        actual_values: array-like, actual values
        predicted_values: array-like, predicted values
        
    Returns:
        float: WAPE value (0-inf, lower is better)
    """
    if len(actual_values) == 0 or len(predicted_values) == 0:
        return float('inf')
    
    # Use WAPE instead of MAPE for better business forecasting
    return wape(actual_values, predicted_values)


# ============================================================================
# Walk-Forward Validation and Backtesting
# ============================================================================

def daily_backtesting_validation(series, model_fitting_func, window_size=7, step_size=1, horizon=2,
                           model_params=None, diagnostic_messages=None, gap: int = 0):
    """
    Daily data backtesting validation optimized for quarterly forecasting.
    
    Uses shorter horizons and heavily weights recent performance suitable for daily business data.

    Args:
        series: Pandas Series of daily values
        model_fitting_func: Callable(train_data, **params) -> fitted model with forecast()/predict()
        window_size: minimum training size (days), default 7 for daily data
        step_size: fold increment (days), typically 1 for daily
        horizon: validation horizon per fold (days), default 2 for daily forecasting
        gap: gap between train end and validation start
        model_params: optional dict of params for fitting function
        diagnostic_messages: list for status logs

    Returns:
        dict with keys: mean_mape, recent_weighted_mape, iterations, validation_results
    """
    if model_params is None:
        model_params = {}
    
    mape_scores = []  # WAPE per fold
    validation_results = []
    
    # Ensure we have enough data for daily backtesting (minimum 2 weeks for quarterly forecasting)
    min_train = max(7, int(window_size))
    min_required = min_train + max(0, gap) + horizon
    if len(series) < min_required:
        if diagnostic_messages:
            diagnostic_messages.append(f"⚠️ Daily backtesting: Need {min_required} days, have {len(series)}. Insufficient data.")
        return None

    # For DAILY QUARTERLY forecasting: focus validation on RECENT periods only
    # We only care about how models perform on the last 2-3 weeks, not ancient history
    
    # Limit validation to the most recent 20-30 days of data
    recent_validation_window = min(30, max(14, len(series) // 2))  # Last 2-4 weeks
    
    # Start validation from near the end of the available data
    validation_start_point = max(min_train, len(series) - recent_validation_window)
    
    # Calculate how many validation folds we can fit in the recent window
    available_validation_days = len(series) - validation_start_point - max(0, gap) - horizon
    max_iterations = min(7, max(1, available_validation_days // step_size))  # Max 7 recent folds
    
    if max_iterations <= 0:
        return None
    
    # Setup validation parameters for RECENT daily data only
    for i in range(max_iterations):
        # Start validation from recent periods, not from the beginning
        train_end_idx = validation_start_point + i * step_size
        
        # Ensure we don't go beyond available data
        if train_end_idx >= len(series) - max(0, gap) - horizon:
            break
            
        val_start_idx = train_end_idx + max(0, gap)
        val_end_idx = val_start_idx + horizon

        if val_end_idx > len(series):
            break

        train_data = series.iloc[:train_end_idx]
        actual_data = series.iloc[val_start_idx:val_end_idx]

        try:
            # Fit model and generate forecast
            fitted_model = model_fitting_func(train_data, **model_params)
            
            # Handle different model types more robustly
            forecast = None
            if isinstance(fitted_model, dict):
                # Most daily models return dicts with forecast info
                if 'forecast_value' in fitted_model:
                    # Simple single-value forecast (Run Rate, etc.)
                    forecast = [fitted_model['forecast_value']] * len(actual_data)
                elif 'model' in fitted_model and hasattr(fitted_model['model'], 'predict'):
                    # Linear models with scikit-learn interface
                    try:
                        # Project forward from end of training data
                        future_x = np.arange(len(train_data), len(train_data) + len(actual_data)).reshape(-1, 1)
                        forecast = fitted_model['model'].predict(future_x)
                    except Exception as e:
                        if diagnostic_messages:
                            diagnostic_messages.append(f"⚠️ Linear model prediction failed: {str(e)}")
                        forecast = [fitted_model.get('forecast_value', train_data.mean())] * len(actual_data)
                elif 'forecast' in fitted_model:
                    # Model already contains forecast
                    base_forecast = fitted_model['forecast']
                    if hasattr(base_forecast, '__len__') and len(base_forecast) >= len(actual_data):
                        forecast = base_forecast[:len(actual_data)]
                    else:
                        # Extend forecast if needed
                        forecast_val = base_forecast[0] if hasattr(base_forecast, '__len__') else base_forecast
                        forecast = [forecast_val] * len(actual_data)
                else:
                    # Fallback to mean
                    forecast = [train_data.mean()] * len(actual_data)
            elif hasattr(fitted_model, 'forecast') and callable(fitted_model.forecast):
                # Prophet-like models with forecast method
                forecast = fitted_model.forecast(len(actual_data))
            elif hasattr(fitted_model, 'predict') and callable(fitted_model.predict):
                # Time series models with predict method
                forecast = fitted_model.predict(start=len(train_data), end=len(train_data) + len(actual_data) - 1)
            else:
                if diagnostic_messages:
                    diagnostic_messages.append(f"⚠️ Unknown model type: {type(fitted_model)}")
                continue
            
            # Ensure forecast is a list/array
            if hasattr(forecast, 'values'):
                forecast = forecast.values
            elif not isinstance(forecast, (list, np.ndarray)):
                forecast = [forecast] * len(actual_data)
            
            # Calculate WAPE
            mape = wape(actual_data.values, forecast)
            
            if not np.isfinite(mape):
                if diagnostic_messages:
                    diagnostic_messages.append(f"⚠️ WAPE calculation returned {mape} for iteration {len(mape_scores) + 1}")
                continue
                
            mape_scores.append(mape)
            
            validation_results.append({
                'iteration': len(mape_scores),
                'train_start': series.index[0],
                'train_end': series.index[train_end_idx - 1],
                'val_start': series.index[val_start_idx],
                'val_end': series.index[val_end_idx - 1],
                'mape': mape,
                'train_days': len(train_data),
                'val_days': len(actual_data),
                'actual_values': actual_data.values.tolist(),
                'forecast_values': forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast)
            })
            
            # Successful iteration - continue to next
            pass

        except Exception as e:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Daily backtesting iteration {len(mape_scores) + 1} failed: {str(e)}")
            continue
    
    if not mape_scores:
        return None
    
    mape_arr = np.array(mape_scores)
    
    # HEAVILY weight recent performance for daily forecasting (exponential decay)
    try:
        n_folds = len(mape_arr)
        if n_folds > 0:
            # Very aggressive weighting - recent performance gets 4x weight
            exponents = np.linspace(0, 2, n_folds)  # More aggressive than monthly (was 0 to 1)
            weights = np.power(4.0, exponents)  # Much higher weight ratio (was 2.0)
            rw_mape = float(np.sum(weights * mape_arr) / np.sum(weights))
        else:
            rw_mape = float(np.mean(mape_arr)) if mape_arr.size else np.inf
    except Exception:
        rw_mape = float(np.mean(mape_arr)) if mape_arr.size else np.inf
    
    # Focus on most recent performance
    try:
        recent_slice = mape_arr[-2:] if len(mape_arr) >= 2 else mape_arr  # Last 2 folds
        recent_p75 = float(np.percentile(recent_slice, 75)) if recent_slice.size else np.inf
    except Exception:
        recent_p75 = np.inf
    
    results = {
        'mean_mape': float(np.mean(mape_arr)),
        'std_mape': float(np.std(mape_arr)),
        'recent_weighted_mape': rw_mape,  # This is the key metric for daily data
        'recent_p75_mape': recent_p75,
        'p75_mape': float(np.percentile(mape_arr, 75)),
        'iterations': len(mape_scores),
        'validation_results': validation_results,
        'mape_scores': mape_scores
    }
    
    if diagnostic_messages:
        diagnostic_messages.append(
            f"📊 Daily backtesting: {len(mape_scores)} folds, "
            f"Recent-weighted WAPE: {rw_mape:.1%}, Mean WAPE: {np.mean(mape_scores):.1%}"
        )
    
    return results


def walk_forward_validation(series, model_fitting_func, window_size=14, step_size=1, horizon=7,
                           model_params=None, diagnostic_messages=None, gap: int = 0):
    """
    Expanding-window cross-validation for robust WAPE calculation.

    - Uses expanding train window starting at len == max(window_size, 14)
    - Advances by step_size each fold; fixed horizon per fold
    - Returns fold-level WAPE list and summary stats (mean, p75, p95), plus MASE per fold

    Args:
        series: Pandas Series of daily values
        model_fitting_func: Callable(train_data, **params) -> fitted model with forecast()/predict()
        window_size: minimum training size (days), default 14
        step_size: fold increment (days), set equal to validation_horizon for policy
        horizon: validation horizon per fold (days)
        gap: gap between train end and validation start
        model_params: optional dict of params for fitting function
        diagnostic_messages: list for status logs

    Returns:
        dict with keys: mean_mape, p75_mape, p95_mape, mean_mase, iterations, mape_scores, mase_scores, validation_results
    """
    if model_params is None:
        model_params = {}
    
    mape_scores = []  # WAPE per fold
    smape_scores = []
    rmse_scores = []
    mase_scores = []
    validation_results = []
    
    # Ensure we have enough data for walk-forward validation
    min_train = max(14, int(window_size))
    min_required = min_train + max(0, gap) + horizon
    if len(series) < min_required:
        if diagnostic_messages:
            diagnostic_messages.append(f"⚠️ Walk-forward validation: Need {min_required} days, have {len(series)}. Using single split.")
        return None

    # Expanding window: train_end grows by step_size; start at 0
    max_iterations = max(0, (len(series) - min_train - max(0, gap) - horizon) // step_size + 1)
    max_reasonable_iterations = min(20, len(series) // max(1, step_size))
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
            mase_val = mase(actual_data, forecast, train_data, seasonal_period=7)
            
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
    results = {
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


def simple_backtesting_validation(series, model_fitting_func, backtest_days=7, backtest_gap=1, 
                                 validation_horizon=7, model_params=None, diagnostic_messages=None):
    """
    Simple backtesting validation using a single train/test split for daily data.
    
    Args:
        series: Full time series data
        model_fitting_func: Function for fitting models
        backtest_days: Number of days to use for backtesting
        backtest_gap: Gap between training and test data (days)
        validation_horizon: How far ahead to predict in backtesting
        model_params: Parameters for model fitting
        diagnostic_messages: List to append diagnostic messages
    
    Returns:
        Dictionary with backtesting results or None if validation fails
    """
    try:
        # Calculate minimum data requirements
        min_training_data = 7   # Minimum days needed for training
        min_test_data = 2       # Minimum days needed for testing
        total_required = backtest_days + validation_horizon + backtest_gap + min_training_data
        
        if len(series) < total_required:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Insufficient data for {backtest_days} day backtesting. Need at least {total_required} days (have {len(series)}).")
            return None
        
        # Split data: train on everything except last (backtest_days + gap) days
        split_point = len(series) - backtest_days - backtest_gap
        train_data = series.iloc[:split_point]
        test_data = series.iloc[split_point:split_point + validation_horizon]
        
        if len(train_data) < 7 or len(test_data) < 2:
            if diagnostic_messages:
                diagnostic_messages.append(f"⚠️ Split resulted in insufficient train/test data: {len(train_data)} train, {len(test_data)} test days.")
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
                        freq='D'
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
                test_data.values, forecast, train_data.values, seasonal_period=7
            )
            
            return {
                'mape': mape,
                'smape': smape_val,
                'mase': mase_val,
                'rmse': rmse_val,
                'train_days': len(train_data),
                'test_days': len(test_data),
                'backtest_period': backtest_days,
                'gap': backtest_gap,
                'validation_horizon': validation_horizon,
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


def evaluate_individual_forecasts(full_series, forecasts):
    """
    Evaluate each forecast model using WAPE on recent historical data.
    Uses the last 14-30 days of data for better model validation.
    
    Args:
        full_series: pandas Series with full historical daily data
        forecasts: dict of forecast results from different models
        
    Returns:
        dict: WAPE scores for each model (lower is better)
    """
    wape_scores = {}
    
    # Use recent historical data for evaluation (last 14-30 days)
    if len(full_series) < 7:
        return wape_scores
    
    # Use the most recent 14-30 days, or 80% of available data, whichever is smaller
    eval_size = min(30, max(7, int(len(full_series) * 0.8)))
    eval_data = full_series.iloc[-eval_size:]  # Use most recent data
    
    for model_name, forecast_info in forecasts.items():
        if model_name == 'Linear Trend':
            wape_scores[model_name] = evaluate_linear_trend_wape(eval_data)
        elif model_name == 'Moving Average':
            wape_scores[model_name] = evaluate_moving_average_wape(eval_data)
        elif model_name == 'Prophet':
            wape_scores[model_name] = evaluate_prophet_wape(eval_data)
        elif model_name == 'Run Rate':
            wape_scores[model_name] = evaluate_run_rate_wape(eval_data)
        elif model_name == 'Monthly Renewals':
            # Skip WAPE evaluation for Monthly Renewals - it's a special purpose model
            # for detecting subscription renewals, not a general forecasting model
            continue
        elif model_name == 'ARIMA':
            wape_scores[model_name] = evaluate_time_series_model_wape(eval_data, 'ARIMA')
        elif model_name == 'Exponential Smoothing':
            wape_scores[model_name] = evaluate_time_series_model_wape(eval_data, 'Exponential Smoothing')
        elif model_name == 'LightGBM':
            wape_scores[model_name] = evaluate_ml_model_wape(eval_data, full_series, 'LightGBM')
        elif model_name == 'XGBoost':
            wape_scores[model_name] = evaluate_ml_model_wape(eval_data, full_series, 'XGBoost')
        else:
            # Default evaluation for unknown models (conservative WAPE)
            wape_scores[model_name] = 0.25
    
    return wape_scores


def evaluate_linear_trend_wape(data):
    """Evaluate linear trend model using in-sample predictions."""
    if len(data) < 3:
        return float('inf')
    
    try:
        from sklearn.linear_model import LinearRegression
        
        # Create time index for regression
        x = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        
        # Split for validation
        train_size = max(2, int(len(data) * 0.7))
        x_train, x_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        if len(x_test) == 0:
            return float('inf')
        
        # Fit model and predict
        model = LinearRegression()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        
        return wape(y_test, predictions)
    except:
        return float('inf')


def evaluate_moving_average_wape(data):
    """Evaluate moving average model."""
    if len(data) < 8:
        return float('inf')
    
    try:
        window = min(7, len(data) // 2)
        predictions = []
        actual = []
        
        for i in range(window, len(data)):
            ma_pred = data.iloc[i-window:i].mean()
            predictions.append(ma_pred)
            actual.append(data.iloc[i])
        
        if len(predictions) == 0:
            return float('inf')
            
        return wape(np.array(actual), np.array(predictions))
    except:
        return float('inf')


def evaluate_prophet_wape(data):
    """Evaluate Prophet model using cross-validation."""
    if not HAVE_PROPHET or Prophet is None or len(data) < 14:
        return float('inf')
    
    try:
        # Split data for validation
        train_size = max(10, int(len(data) * 0.7))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        if len(test_data) < 2:
            return float('inf')
        
        # Prepare training data
        train_df = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        
        test_df = pd.DataFrame({
            'ds': test_data.index,
            'y': test_data.values
        })
        
        # Suppress Prophet logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        
        model = Prophet(daily_seasonality='auto', weekly_seasonality='auto', yearly_seasonality='auto')
        model.fit(train_df)
        
        forecast = model.predict(test_df[['ds']])
        predictions = np.array(forecast['yhat'].values)
        actual = np.array(test_df['y'].values)
        
        return wape(actual, predictions)
    except:
        return float('inf')


def evaluate_run_rate_wape(data):
    """Evaluate run rate model."""
    if len(data) < 3:
        return float('inf')
    
    try:
        # Simple run rate: predict each day using average of previous days
        predictions = []
        actual = []
        
        for i in range(2, len(data)):
            run_rate_pred = data.iloc[:i].mean()
            predictions.append(run_rate_pred)
            actual.append(data.iloc[i])
        
        if len(predictions) == 0:
            return float('inf')
            
        return wape(np.array(actual), np.array(predictions))
    except:
        return float('inf')


def evaluate_renewal_model_mape(eval_data, full_data):
    """
    DEPRECATED: Evaluate renewal model by checking spike prediction accuracy.
    
    This function is no longer used since Monthly Renewals model is excluded
    from MAPE competition as it's a special-purpose model for subscription renewals.
    """
    # Function kept for backward compatibility but always returns inf
    # since Monthly Renewals should not compete with forecasting models
    return float('inf')


def evaluate_time_series_model_wape(data, model_name):
    """Evaluate time series models (ARIMA, Exponential Smoothing, etc.)."""
    if not HAVE_STATSMODELS or len(data) < 10:
        return float('inf')
    
    try:
        # Split data for validation
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        if len(train_data) < 7 or len(test_data) < 2:
            return float('inf')
        
        if model_name == 'ARIMA' and ARIMA is not None:
            model = ARIMA(train_data, order=(1, 1, 1))
            fitted_model = model.fit()
            predictions = fitted_model.forecast(steps=len(test_data))
        elif model_name == 'Exponential Smoothing' and ExponentialSmoothing is not None:
            model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
            fitted_model = model.fit()
            predictions = fitted_model.forecast(steps=len(test_data))
        else:
            return float('inf')
        
        return wape(test_data.values, predictions)
    except:
        return float('inf')


def evaluate_ml_model_wape(eval_data, full_series, model_name):
    """
    Evaluate ML models using proper cross-validation on recent historical data.
    
    Args:
        eval_data: pandas Series with recent data for evaluation
        full_series: pandas Series with full historical data
        model_name: str, name of the model ('XGBoost' or 'LightGBM')
    """
    if len(eval_data) < 14:
        return float('inf')
    
    try:
        if model_name == 'XGBoost':
            return evaluate_xgboost_wape_proper(eval_data, full_series)
        elif model_name == 'LightGBM':
            return evaluate_lightgbm_wape_proper(eval_data, full_series)
        else:
            return float('inf')
    except:
        return float('inf')


def evaluate_xgboost_wape_proper(eval_data, full_series):
    """Evaluate XGBoost model using actual cross-validation."""
    try:
        if not HAVE_XGBOOST:
            return float('inf')
        
        # Use time-series cross-validation on evaluation data
        if len(eval_data) < 10:
            return float('inf')
        
        # Split evaluation data for validation
        train_size = int(len(eval_data) * 0.7)
        train_data = eval_data.iloc[:train_size]
        test_data = eval_data.iloc[train_size:]
        
        if len(train_data) < 7 or len(test_data) < 2:
            return float('inf')
        
        # Create features for XGBoost (day of week, day of month, etc.)
        def create_features(data):
            features = []
            for i, date in enumerate(data.index):
                features.append([
                    date.dayofweek,
                    date.day,
                    date.month,
                    i,  # trend component
                    data.iloc[max(0, i-7):i].mean() if i >= 7 else data.iloc[:i+1].mean()  # 7-day MA
                ])
            return np.array(features)
        
        X_train = create_features(train_data)
        y_train = train_data.values
        X_test = create_features(test_data)
        y_test = test_data.values
        
        # Train XGBoost model
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict and calculate WAPE
        predictions = model.predict(X_test)
        
        # Check for reasonable predictions (not more than 3x historical average)
        historical_avg = full_series.mean()
        if np.mean(predictions) > historical_avg * 3:
            return float('inf')  # Penalize unrealistic forecasts
        
        return wape(y_test, predictions)
    except Exception:
        return 0.30  # Conservative fallback


def evaluate_lightgbm_wape_proper(eval_data, full_series):
    """Evaluate LightGBM model using actual cross-validation."""
    try:
        # Import LightGBM check
        try:
            from lightgbm import LGBMRegressor
            HAVE_LGBM = True
        except ImportError:
            return float('inf')
        
        if not HAVE_LGBM or len(eval_data) < 10:
            return float('inf')
        
        # Split evaluation data for validation
        train_size = int(len(eval_data) * 0.7)
        train_data = eval_data.iloc[:train_size]
        test_data = eval_data.iloc[train_size:]
        
        if len(train_data) < 7 or len(test_data) < 2:
            return float('inf')
        
        # Create features for LightGBM
        def create_features(data):
            features = []
            for i, date in enumerate(data.index):
                features.append([
                    date.dayofweek,
                    date.day,
                    date.month,
                    i,  # trend component
                    data.iloc[max(0, i-7):i].mean() if i >= 7 else data.iloc[:i+1].mean(),  # 7-day MA
                    data.iloc[max(0, i-3):i].mean() if i >= 3 else data.iloc[:i+1].mean()   # 3-day MA
                ])
            return np.array(features)
        
        X_train = create_features(train_data)
        y_train = train_data.values
        X_test = create_features(test_data)
        y_test = test_data.values
        
        # Train LightGBM model
        model = LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        # Predict and calculate WAPE
        predictions = model.predict(X_test)
        
        # Check for reasonable predictions (not more than 3x historical average)
        historical_avg = full_series.mean()
        
        # Convert predictions to numpy array to handle sparse matrices
        try:
            # Check if predictions is a sparse matrix
            if hasattr(predictions, 'toarray'):
                predictions_array = predictions.toarray().flatten()  # type: ignore
            elif isinstance(predictions, list) and len(predictions) > 0:
                # List of sparse matrices or arrays
                if hasattr(predictions[0], 'toarray'):
                    predictions_array = np.concatenate([p.toarray().flatten() for p in predictions])  # type: ignore
                else:
                    predictions_array = np.concatenate([np.asarray(p).flatten() for p in predictions])
            else:
                # Regular array or scalar
                predictions_array = np.asarray(predictions).flatten()
        except (AttributeError, TypeError):
            # Fallback: convert to numpy array
            predictions_array = np.asarray(predictions).flatten()
        
        if np.mean(predictions_array) > historical_avg * 3:
            return float('inf')  # Penalize unrealistic forecasts
        
        return wape(y_test, predictions)
    except Exception:
        return 0.25  # Conservative fallback


def calculate_weighted_model_score(model_name, wape_score, forecast_data, quarter_data):
    """
    Calculate weighted score for model selection considering WAPE, stability, and bias.
    Lower scores are better.
    
    Args:
        model_name: str, name of the model
        wape_score: float, WAPE value (0-inf)
        forecast_data: dict, forecast results for this model
        quarter_data: pandas Series, historical data
        
    Returns:
        float: Weighted score (lower is better)
    """
    if wape_score == float('inf'):
        return float('inf')
    
    # Base score from WAPE (70% weight) - scale by 100 for easier interpretation
    base_score = wape_score * 100 * 0.7
    
    # Stability penalty based on forecast volatility (15% weight)
    stability_penalty = 0
    if 'forecast' in forecast_data and len(forecast_data['forecast']) > 1:
        forecast_values = forecast_data['forecast']
        forecast_std = np.std(forecast_values)
        forecast_mean = np.mean(forecast_values)
        if forecast_mean > 0:
            volatility = forecast_std / forecast_mean
            stability_penalty = min(volatility * 5, 10)  # Cap at 10 points
    
    # Bias penalty for models that historically over-forecast (15% weight)
    bias_penalty = 0
    quarter_mean = quarter_data.mean()
    forecast_mean = forecast_data.get('daily_avg', quarter_mean)
    if quarter_mean > 0:
        bias_ratio = forecast_mean / quarter_mean
        if bias_ratio > 1.2:  # Over-forecasting by more than 20%
            bias_penalty = (bias_ratio - 1) * 10  # Progressive penalty
        elif bias_ratio < 0.8:  # Under-forecasting by more than 20%
            bias_penalty = (1 - bias_ratio) * 5  # Smaller penalty for conservatism
    
    # Model-specific adjustments based on known characteristics
    model_adjustment = 0
    if model_name == 'Prophet' and wape_score < 0.20:
        # Prophet tends to be optimistic, but only small penalty if WAPE is very good
        model_adjustment = 0.3 if wape_score < 0.15 else 0.1
    elif model_name == 'Run Rate':
        # Run rate is conservative but reliable, small bonus
        model_adjustment = -0.5
    elif model_name == 'Moving Average':
        # Moving average is stable, small bonus
        model_adjustment = -0.3
    
    total_score = base_score + (stability_penalty * 0.15) + (bias_penalty * 0.15) + model_adjustment
    return max(0, total_score)


def select_best_model_weighted(quarter_data, forecasts, model_scores):
    """
    Select best model using weighted scoring that considers WAPE, stability, and bias.
    
    Args:
        quarter_data: pandas Series with daily data for the quarter
        forecasts: dict of forecast results from different models
        model_scores: dict of WAPE scores for each model
        
    Returns:
        str: Name of the best model
    """
    if not model_scores:
        return 'Run Rate'  # Fallback
    
    # First check if there's a clear WAPE winner (>2% relative difference)
    sorted_by_wape = sorted(model_scores.items(), key=lambda x: x[1])
    if len(sorted_by_wape) >= 2:
        best_wape_model, best_wape = sorted_by_wape[0]
        second_best_wape_model, second_best_wape = sorted_by_wape[1]
        
        # If best WAPE is significantly better (>2% relative difference), use it
        if best_wape < float('inf') and second_best_wape > 0 and (second_best_wape - best_wape) / second_best_wape > 0.02:
            if best_wape_model in forecasts:
                return best_wape_model
    
    # Otherwise use weighted scoring
    weighted_scores = {}
    for model_name, wape_score in model_scores.items():
        if model_name in forecasts:
            weighted_scores[model_name] = calculate_weighted_model_score(
                model_name, wape_score, forecasts[model_name], quarter_data
            )
    
    if not weighted_scores:
        return 'Run Rate'  # Fallback
    
    # Select model with lowest weighted score
    best_model = min(weighted_scores.items(), key=lambda x: x[1])[0]
    return best_model


# ============================================================================
# Enhanced Smart Backtesting with Walk-Forward Validation
# ============================================================================

def _forecast_horizon_for_model(model_name: str, train_series: pd.Series, horizon: int) -> np.ndarray:
    """Produce horizon-step forecast for a given model using train_series.
    Returns an array of length=horizon. Falls back to constant mean if not possible.
    """
    try:
        if model_name == 'Linear Trend':
            res = fit_linear_trend_model(train_series)
            if 'model' in res:
                x_future = np.arange(len(train_series), len(train_series) + horizon).reshape(-1, 1)
                return np.asarray(res['model'].predict(x_future)).flatten()
        elif model_name == 'Moving Average':
            res = fit_moving_average_model(train_series, window=min(7, max(3, len(train_series)//3)))
            fv = float(res.get('forecast_value', train_series.mean()))
            return np.full(horizon, fv)
        elif model_name == 'Run Rate':
            return np.full(horizon, float(train_series.mean()))
        elif model_name == 'ARIMA':
            res = fit_arima_model(train_series)
            if res.get('model_type') == 'arima' and 'model' in res:
                fc = res['model'].forecast(steps=horizon)
                return np.asarray(fc).flatten()
        elif model_name == 'Exponential Smoothing':
            res = fit_exponential_smoothing_model(train_series)
            if res.get('model_type') == 'exponential_smoothing' and 'model' in res:
                fc = res['model'].forecast(steps=horizon)
                return np.asarray(fc).flatten()
        elif model_name == 'Prophet':
            res = fit_prophet_daily_model(train_series)
            if res.get('model_type') == 'prophet' and 'model' in res:
                future_dates = pd.date_range(
                    start=train_series.index.max() + pd.Timedelta(days=1),
                    periods=horizon, freq='D'
                )
                future_df = pd.DataFrame({'ds': future_dates})
                pred = res['model'].predict(future_df)
                return np.asarray(pred['yhat'].values).flatten()
        elif model_name == 'LightGBM':
            # We only have a 1-step forecast value; approximate with repeat
            res = fit_lightgbm_daily_model(train_series)
            fv = float(res.get('forecast_value', train_series.mean()))
            return np.full(horizon, fv)
        # Skip XGBoost/Monthly Renewals for backtesting competition
    except Exception:
        pass
    # Fallback: constant mean
    return np.full(horizon, float(train_series.mean()) if len(train_series) else 0.0)


def smart_backtesting_select_model(
    full_series: pd.Series,
    forecasts: dict,
    method: str = 'enhanced',
    horizon: int = 7,
    gap: int = 1,
    folds: int = 3,
) -> dict:
    """Enhanced backtesting model selection with walk-forward validation and WAPE.

    Args:
        full_series: pandas Series with daily values
        forecasts: dict of model_name -> forecast payload (from engine)
        method: 'enhanced' for new validation, 'simple' for legacy
        horizon: days to forecast in each fold
        gap: gap between train end and test start
        folds: number of folds from the tail of the series

    Returns:
        dict with keys: best_model, per_model_wape, method_used, validation_details
    """
    # Candidate models from available forecasts, excluding special-purpose
    candidate_models = [
        m for m in forecasts.keys()
        if m not in ('Monthly Renewals', 'Ensemble')
    ]
    
    # Adjust minimum data requirement for daily backtesting
    min_data_required = 10 if method == 'enhanced' else (horizon + gap + 7)
    if len(candidate_models) == 0 or len(full_series) < min_data_required:
        return {
            'best_model': min(forecasts.keys(), key=lambda k: forecasts[k].get('quarter_total', float('inf')))
            if forecasts else 'Run Rate',
            'per_model_wape': {},
            'method_used': 'insufficient-data',
            'validation_details': {}
        }

    per_model_wapes: dict[str, float] = {}
    validation_details = {}
    
    # Enhanced method: use walk-forward validation
    if method == 'enhanced':
        # Create fitting functions for each model
        def create_model_fitting_func(model_name):
            def fitting_func(series):
                if model_name == 'Linear Trend':
                    return fit_linear_trend_model(series)
                elif model_name == 'Moving Average':
                    return fit_moving_average_model(series, window=min(7, max(3, len(series)//3)))
                elif model_name == 'Run Rate':
                    return {'forecast_value': series.mean()}
                elif model_name == 'Exponential Smoothing':
                    return fit_exponential_smoothing_model(series)
                elif model_name == 'Monthly Renewals':
                    return {'forecast_value': series.mean()}  # Simplified for backtesting
                # COMMENTED OUT: Focus on core models for daily quarterly forecasting
                # elif model_name == 'ARIMA':
                #     return fit_arima_model(series)
                # elif model_name == 'Prophet':
                #     return fit_prophet_daily_model(series)
                # elif model_name == 'LightGBM':
                #     return fit_lightgbm_daily_model(series)
                # elif model_name == 'XGBoost':
                #     return fit_xgboost_model(series)[0] if fit_xgboost_model(series)[0] is not None else {'forecast_value': series.mean()}
                else:
                    return {'forecast_value': series.mean()}
            return fitting_func
        
        # Run daily backtesting validation for each model (optimized for quarterly forecasting)
        diagnostic_msgs = []
        for model_name in candidate_models:
            try:
                model_msgs = []
                fitting_func = create_model_fitting_func(model_name)
                cv_results = daily_backtesting_validation(
                    series=full_series,
                    model_fitting_func=fitting_func,
                    window_size=7,  # 1 week minimum training for daily data
                    step_size=1,    # Daily steps
                    horizon=2,      # 2-day forecast horizon (shorter for daily)
                    gap=gap,
                    diagnostic_messages=model_msgs
                )
                
                # Collect diagnostic messages for analysis if needed
                diagnostic_msgs.extend([f"{model_name}: {msg}" for msg in model_msgs])
                
                if cv_results and cv_results['iterations'] > 0:
                    # Use recent weighted WAPE for better selection
                    per_model_wapes[model_name] = cv_results['recent_weighted_mape']
                    validation_details[model_name] = {
                        'mean_wape': cv_results['mean_mape'],
                        'p75_wape': cv_results['p75_mape'],
                        'iterations': cv_results['iterations'],
                        'recent_weighted_wape': cv_results['recent_weighted_mape'],
                        'validation_periods': cv_results.get('validation_results', [])  # Include actual validation periods
                    }
                    # Model successfully validated
                    pass
                else:
                    per_model_wapes[model_name] = float('inf')
                    validation_details[model_name] = {'error': 'validation_failed', 'details': 'No successful iterations'}
            except Exception as e:
                per_model_wapes[model_name] = float('inf')
                validation_details[model_name] = {'error': 'exception', 'details': str(e)}
        
        method_used = f'enhanced-walk-forward({horizon}h, gap={gap})'
        
    else:
        # Simple method: basic fold validation (legacy)
        n = len(full_series)
        max_folds = 0
        for model in candidate_models:
            errors = []
            used_folds = 0
            for f in range(folds, 0, -1):
                test_end = n - (f - 1) * horizon
                test_start = test_end - horizon
                train_end = test_start - gap
                if train_end <= 3 or test_start < 0 or test_end > n:
                    continue
                train_series = full_series.iloc[:train_end]
                test_series = full_series.iloc[test_start:test_end]
                if len(train_series) < 5 or len(test_series) < 1:
                    continue
                preds = _forecast_horizon_for_model(model, train_series, horizon=len(test_series))
                wape_score = wape(test_series.values, preds)
                if wape_score != float('inf'):
                    errors.append(wape_score)
                    used_folds += 1
            if errors:
                per_model_wapes[model] = float(np.mean(errors))
                max_folds = max(max_folds, used_folds)
                validation_details[model] = {'folds': used_folds, 'mean_wape': float(np.mean(errors))}
            else:
                per_model_wapes[model] = float('inf')
                validation_details[model] = {'error': 'no_valid_folds'}
        
        method_used = f'simple-walk-forward({max_folds} folds, h={horizon}, gap={gap})'

    # Choose best by lowest WAPE
    best_model = min(per_model_wapes.items(), key=lambda x: x[1])[0] if per_model_wapes else 'Run Rate'
    
    return {
        'best_model': best_model,
        'per_model_wape': per_model_wapes,
        'method_used': method_used,
        'validation_details': validation_details
    }
