"""
Model Evaluation and MAPE Calculation for Outlook Forecaster

Evaluates forecasting models using MAPE and other metrics
to determine the best performing models for quarterly forecasting.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_percentage_error

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


def calculate_mape(actual_values, predicted_values):
    """
    Calculate Mean Absolute Percentage Error (MAPE) with robust error handling.
    
    Args:
        actual_values: array-like, actual values
        predicted_values: array-like, predicted values
        
    Returns:
        float: MAPE percentage (0-100)
    """
    if len(actual_values) == 0 or len(predicted_values) == 0:
        return float('inf')
    
    # Convert to numpy arrays
    actual = np.array(actual_values)
    predicted = np.array(predicted_values)
    
    # Ensure same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    # Avoid division by zero - use small epsilon for very small values
    epsilon = 1e-8
    actual_safe = np.where(np.abs(actual) < epsilon, epsilon, actual)
    
    # Calculate MAPE with more robust formula
    percentage_errors = np.abs((actual - predicted) / actual_safe) * 100
    
    # Remove any infinite or NaN values
    percentage_errors = percentage_errors[np.isfinite(percentage_errors)]
    
    if len(percentage_errors) == 0:
        return float('inf')
    
    # Cap extreme percentage errors at 200% to avoid model explosion
    percentage_errors = np.clip(percentage_errors, 0, 200)
    
    mape = np.mean(percentage_errors)
    return float(mape)


def evaluate_individual_forecasts(full_series, forecasts):
    """
    Evaluate each forecast model using MAPE on recent historical data.
    Uses the last 30-60 days of data for better model validation.
    
    Args:
        full_series: pandas Series with full historical daily data
        forecasts: dict of forecast results from different models
        
    Returns:
        dict: MAPE scores for each model
    """
    mape_scores = {}
    
    # Use recent historical data for evaluation (last 30-60 days)
    if len(full_series) < 14:
        return mape_scores
    
    # Use the most recent 30-60 days, or 80% of available data, whichever is smaller
    eval_size = min(60, max(14, int(len(full_series) * 0.8)))
    eval_data = full_series.iloc[-eval_size:]  # Use most recent data
    
    for model_name, forecast_info in forecasts.items():
        if model_name == 'Linear Trend':
            mape_scores[model_name] = evaluate_linear_trend_mape(eval_data)
        elif model_name == 'Moving Average':
            mape_scores[model_name] = evaluate_moving_average_mape(eval_data)
        elif model_name == 'Prophet':
            mape_scores[model_name] = evaluate_prophet_mape(eval_data)
        elif model_name == 'Run Rate':
            mape_scores[model_name] = evaluate_run_rate_mape(eval_data)
        elif model_name == 'Monthly Renewals':
            # Skip MAPE evaluation for Monthly Renewals - it's a special purpose model
            # for detecting subscription renewals, not a general forecasting model
            continue
        elif model_name == 'ARIMA':
            mape_scores[model_name] = evaluate_time_series_model_mape(eval_data, 'ARIMA')
        elif model_name == 'Exponential Smoothing':
            mape_scores[model_name] = evaluate_time_series_model_mape(eval_data, 'Exponential Smoothing')
        elif model_name == 'LightGBM':
            mape_scores[model_name] = evaluate_ml_model_mape(eval_data, full_series, 'LightGBM')
        elif model_name == 'XGBoost':
            mape_scores[model_name] = evaluate_ml_model_mape(eval_data, full_series, 'XGBoost')
        else:
            # Default evaluation for unknown models
            mape_scores[model_name] = 25.0
    
    return mape_scores


def evaluate_linear_trend_mape(data):
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
        
        return calculate_mape(y_test, predictions)
    except:
        return float('inf')


def evaluate_moving_average_mape(data):
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
            
        return calculate_mape(np.array(actual), np.array(predictions))
    except:
        return float('inf')


def evaluate_prophet_mape(data):
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
        
        return calculate_mape(actual, predictions)
    except:
        return float('inf')


def evaluate_run_rate_mape(data):
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
            
        return calculate_mape(np.array(actual), np.array(predictions))
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


def evaluate_time_series_model_mape(data, model_name):
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
        
        return calculate_mape(test_data.values, predictions)
    except:
        return float('inf')


def evaluate_ml_model_mape(eval_data, full_series, model_name):
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
            return evaluate_xgboost_mape_proper(eval_data, full_series)
        elif model_name == 'LightGBM':
            return evaluate_lightgbm_mape_proper(eval_data, full_series)
        else:
            return float('inf')
    except:
        return float('inf')


def evaluate_xgboost_mape_proper(eval_data, full_series):
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
        
        # Predict and calculate MAPE
        predictions = model.predict(X_test)
        
        # Check for reasonable predictions (not more than 3x historical average)
        historical_avg = full_series.mean()
        if np.mean(predictions) > historical_avg * 3:
            return float('inf')  # Penalize unrealistic forecasts
        
        return calculate_mape(y_test, predictions)
    except Exception:
        return 30.0  # Conservative fallback


def evaluate_lightgbm_mape_proper(eval_data, full_series):
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
        
        # Predict and calculate MAPE
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
        
        return calculate_mape(y_test, predictions)
    except Exception:
        return 25.0  # Conservative fallback


def calculate_weighted_model_score(model_name, mape_score, forecast_data, quarter_data):
    """
    Calculate weighted score for model selection considering MAPE, stability, and bias.
    Lower scores are better.
    
    Args:
        model_name: str, name of the model
        mape_score: float, MAPE percentage
        forecast_data: dict, forecast results for this model
        quarter_data: pandas Series, historical data
        
    Returns:
        float: Weighted score (lower is better)
    """
    if mape_score == float('inf'):
        return float('inf')
    
    # Base score from MAPE (70% weight)
    base_score = mape_score * 0.7
    
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
    if model_name == 'Prophet' and mape_score < 20:
        # Prophet tends to be optimistic, but only small penalty if MAPE is very good
        model_adjustment = 0.3 if mape_score < 15 else 0.1
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
    Select best model using weighted scoring that considers MAPE, stability, and bias.
    
    Args:
        quarter_data: pandas Series with daily data for the quarter
        forecasts: dict of forecast results from different models
        model_scores: dict of MAPE scores for each model
        
    Returns:
        str: Name of the best model
    """
    if not model_scores:
        return 'Run Rate'  # Fallback
    
    # First check if there's a clear MAPE winner (>2% better than others)
    sorted_by_mape = sorted(model_scores.items(), key=lambda x: x[1])
    if len(sorted_by_mape) >= 2:
        best_mape_model, best_mape = sorted_by_mape[0]
        second_best_mape_model, second_best_mape = sorted_by_mape[1]
        
        # If best MAPE is significantly better (>2% difference), use it
        if best_mape < float('inf') and (second_best_mape - best_mape) > 2.0:
            if best_mape_model in forecasts:
                return best_mape_model
    
    # Otherwise use weighted scoring
    weighted_scores = {}
    for model_name, mape_score in model_scores.items():
        if model_name in forecasts:
            weighted_scores[model_name] = calculate_weighted_model_score(
                model_name, mape_score, forecasts[model_name], quarter_data
            )
    
    if not weighted_scores:
        return 'Run Rate'  # Fallback
    
    # Select model with lowest weighted score
    best_model = min(weighted_scores.items(), key=lambda x: x[1])[0]
    return best_model


# ========================= Smart Backtesting (Walk-forward / CV) ========================= #
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
    method: str = 'auto',
    horizon: int = 7,
    gap: int = 7,
    folds: int = 3,
) -> dict:
    """Select best model via simple walk-forward backtesting with leakage gap.

    Args:
        full_series: pandas Series with daily values
        forecasts: dict of model_name -> forecast payload (from engine)
        method: currently informational; kept for parity
        horizon: days to forecast in each fold
        gap: gap between train end and test start
        folds: number of folds from the tail of the series

    Returns:
        dict with keys: best_model, per_model_mape, method_used
    """
    # Candidate models from available forecasts, excluding special-purpose
    candidate_models = [
        m for m in forecasts.keys()
        if m not in ('Monthly Renewals', 'Ensemble')
    ]
    if len(candidate_models) == 0 or len(full_series) < (horizon + gap + 10):
        return {
            'best_model': min(forecasts.keys(), key=lambda k: forecasts[k].get('quarter_total', float('inf')))
            if forecasts else 'Run Rate',
            'per_model_mape': {},
            'method_used': 'insufficient-data'
        }

    per_model_mapes: dict[str, float] = {}
    n = len(full_series)
    # Define fold boundaries from the end
    # Ensure indices are valid
    max_folds = 0
    for model in candidate_models:
        errors = []
        used_folds = 0
        for f in range(folds, 0, -1):
            test_end = n - (f - 1) * horizon
            test_start = test_end - horizon
            train_end = test_start - gap
            if train_end <= 5 or test_start < 0 or test_end > n:
                continue
            train_series = full_series.iloc[:train_end]
            test_series = full_series.iloc[test_start:test_end]
            if len(train_series) < 10 or len(test_series) < 2:
                continue
            preds = _forecast_horizon_for_model(model, train_series, horizon=len(test_series))
            mape = calculate_mape(test_series.values, preds)
            if mape != float('inf'):
                errors.append(mape)
                used_folds += 1
        if errors:
            per_model_mapes[model] = float(np.mean(errors))
            max_folds = max(max_folds, used_folds)
        else:
            per_model_mapes[model] = float('inf')

    # Choose best by lowest MAPE
    best_model = min(per_model_mapes.items(), key=lambda x: x[1])[0] if per_model_mapes else 'Run Rate'
    return {
        'best_model': best_model,
        'per_model_mape': per_model_mapes,
        'method_used': f'walk-forward({max_folds} folds, h={horizon}, gap={gap})'
    }
