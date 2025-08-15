"""
Forecasting Models for Outlook Forecaster

Various forecasting models optimized for daily business data
and quarterly outlook projections.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Optional dependencies with graceful fallbacks
try:
    from prophet import Prophet
    HAVE_PROPHET = True
    print("Prophet successfully imported")
except (ImportError, ValueError) as e:
    HAVE_PROPHET = False
    Prophet = None
    print(f"Prophet not available: {e}")

try:
    from lightgbm import LGBMRegressor
    HAVE_LGBM = True
    print("LightGBM successfully imported")
except (ImportError, ValueError) as e:
    HAVE_LGBM = False
    LGBMRegressor = None
    print(f"LightGBM not available: {e}")

try:
    import xgboost as xgb
    HAVE_XGBOOST = True
    print("XGBoost successfully imported")
except (ImportError, ValueError) as e:
    HAVE_XGBOOST = False
    xgb = None
    print(f"XGBoost not available: {e}")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAVE_STATSMODELS = True
    print("Statsmodels successfully imported")
except (ImportError, ValueError) as e:
    HAVE_STATSMODELS = False
    ARIMA = None
    ExponentialSmoothing = None
    seasonal_decompose = None
    print(f"Statsmodels not available: {e}")

# Print dependency status
print(f"Advanced Models Status:")
print(f"  Prophet: {'✓' if HAVE_PROPHET else '✗'}")
print(f"  LightGBM: {'✓' if HAVE_LGBM else '✗'}")
print(f"  XGBoost: {'✓' if HAVE_XGBOOST else '✗'}")
print(f"  Statsmodels: {'✓' if HAVE_STATSMODELS else '✗'}")


def fit_linear_trend_model(series, business_days_only=False):
    """
    Fit simple linear trend model optimized for daily consumptive business data.
    
    Args:
        series: pandas Series with datetime index
        business_days_only: bool, whether to use only business days for fitting
        
    Returns:
        dict: Model results with fitted values and parameters
    """
    if business_days_only and len(series) > 10:
        # Use only business days for trend fitting (disabled for daily consumptive business)
        business_data = series[series.index.weekday < 5]
        if len(business_data) >= 5:
            series = business_data
    
    if len(series) < 3:
        return {'forecast': np.full(1, series.mean() if len(series) > 0 else 0), 'model_type': 'fallback_mean'}
    
    # Create time index for regression
    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(x, y)
    
    # Calculate fitted values
    fitted = model.predict(x)
    
    # Generate forecast for next time period
    next_x = np.array([[len(series)]])
    forecast_value = model.predict(next_x)[0]
    
    return {
        'model': model,
        'fitted_values': fitted,
        'forecast': np.array([forecast_value]),
        'forecast_value': forecast_value,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'model_type': 'linear_trend'
    }


def fit_moving_average_model(series, window=7):
    """
    Fit moving average model with trend adjustment.
    
    Args:
        series: pandas Series with datetime index
        window: int, rolling window size for moving average
        
    Returns:
        dict: Model results with forecast values
    """
    if len(series) < window:
        return {'forecast': np.full(1, series.mean() if len(series) > 0 else 0), 'model_type': 'fallback_mean'}
    
    # Calculate rolling mean
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    
    # Simple trend adjustment
    recent_trend = (rolling_mean.iloc[-3:].mean() - rolling_mean.iloc[-window:-3].mean()) if len(rolling_mean) >= window + 3 else 0
    
    forecast_value = rolling_mean.iloc[-1] + recent_trend
    
    return {
        'forecast': np.array([forecast_value]),
        'forecast_value': forecast_value,
        'rolling_mean': rolling_mean,
        'trend_adjustment': recent_trend,
        'model_type': 'moving_average'
    }


def fit_prophet_daily_model(series):
    """
    Fit Prophet model for daily data with automatic seasonality detection.
    
    Args:
        series: pandas Series with datetime index
        
    Returns:
        dict: Model results with forecast or fallback if Prophet unavailable
    """
    if not HAVE_PROPHET or len(series) < 14:
        return {'forecast': np.full(1, series.mean() if len(series) > 0 else 0), 'model_type': 'fallback_mean'}
    
    try:
        # Suppress Prophet logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        
        # Create and fit model
        model = None
        if HAVE_PROPHET and Prophet is not None:
            model = Prophet(
                daily_seasonality='auto',
                weekly_seasonality='auto',
                yearly_seasonality='auto',
                changepoint_prior_scale=0.05
            )
        
        if model is None:
            return {'forecast': np.full(1, series.mean() if len(series) > 0 else 0), 'model_type': 'fallback_mean'}
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        
        # Make forecast for next period
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        
        # Get the forecast value for the next period
        forecast_value = forecast['yhat'].iloc[-1]
        
        return {
            'model': model,
            'training_data': df,
            'forecast': np.array([forecast_value]),
            'forecast_value': forecast_value,
            'model_type': 'prophet'
        }
        
    except Exception as e:
        print(f"Prophet model failed: {e}")
        return {'forecast': np.full(1, series.mean() if len(series) > 0 else 0), 'model_type': 'fallback_mean'}


def fit_lightgbm_daily_model(series):
    """
    Fit LightGBM model with engineered features for daily data.
    
    Args:
        series: pandas Series with datetime index
        
    Returns:
        dict: Model results or fallback if LightGBM unavailable
    """
    if not HAVE_LGBM or len(series) < 14:
        return {'forecast': np.full(1, series.mean() if len(series) > 0 else 0), 'model_type': 'fallback_mean'}
    
    try:
        # Create features
        df = pd.DataFrame({'value': series.values}, index=series.index)
        
        # Add time-based features
        df['day_of_week'] = df.index.to_series().dt.dayofweek
        df['day_of_month'] = df.index.to_series().dt.day
        df['month'] = df.index.to_series().dt.month
        df['quarter'] = df.index.to_series().dt.quarter
        
        # Add lag features
        for lag in [1, 7, 14]:
            if len(df) > lag:
                df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Add rolling statistics
        for window in [3, 7]:
            if len(df) >= window:
                df[f'rolling_mean_{window}'] = df['value'].rolling(window).mean()
                df[f'rolling_std_{window}'] = df['value'].rolling(window).std()
        
        # Drop rows with NaN values (from lags and rolling)
        df_clean = df.dropna()
        
        if len(df_clean) < 7:
            return {'forecast': np.full(1, series.mean()), 'model_type': 'fallback_mean'}
        
        # Split features and target
        feature_cols = [col for col in df_clean.columns if col != 'value']
        X = df_clean[feature_cols]
        y = df_clean['value']
        
        # Train model
        model = None
        if HAVE_LGBM and LGBMRegressor is not None:
            model = LGBMRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        if model is None:
            return {'forecast': np.full(1, series.mean()), 'model_type': 'fallback_mean'}
        
        model.fit(X, y)
        
        # Generate forecast for next period
        # Create feature vector for next time step
        last_date = df.index[-1]
        next_date = last_date + pd.Timedelta(days=1)
        
        # Create next period features
        next_features = {}
        next_features['day_of_week'] = next_date.dayofweek
        next_features['day_of_month'] = next_date.day
        next_features['month'] = next_date.month
        next_features['quarter'] = next_date.quarter
        
        # Add lag features from most recent data
        for lag in [1, 7, 14]:
            if len(df) >= lag:
                next_features[f'lag_{lag}'] = df['value'].iloc[-lag]
        
        # Add rolling statistics from most recent data
        for window in [3, 7]:
            if len(df) >= window:
                next_features[f'rolling_mean_{window}'] = df['value'].iloc[-window:].mean()
                next_features[f'rolling_std_{window}'] = df['value'].iloc[-window:].std()
        
        # Create feature vector in correct order
        feature_vector = []
        for col in feature_cols:
            if col in next_features:
                feature_vector.append(next_features[col])
            else:
                feature_vector.append(0)  # Fallback for missing features
        
        # Make forecast
        try:
            prediction = model.predict([feature_vector])
            # Convert prediction to a numeric value safely
            if hasattr(prediction, 'toarray'):  # Sparse matrix to dense
                prediction = np.asarray(prediction.toarray()).flatten()  # type: ignore
            elif hasattr(prediction, 'flatten'):  # NumPy array
                prediction = np.asarray(prediction).flatten()
            else:  # Already a scalar or array-like
                prediction = np.asarray(prediction).flatten()
            
            # Extract first value
            forecast_value = float(prediction[0]) if len(prediction) > 0 else float(series.mean())
        except Exception:
            # Fallback to series mean if prediction fails
            forecast_value = float(series.mean())
        
        return {
            'model': model,
            'feature_columns': feature_cols,
            'training_data': df_clean,
            'forecast': np.array([forecast_value]),
            'forecast_value': forecast_value,
            'model_type': 'lightgbm'
        }
        
    except Exception:
        return {'forecast': np.full(1, series.mean()), 'model_type': 'fallback_mean'}


def fit_xgboost_model(series):
    """
    Fit XGBoost model with feature engineering for time series forecasting.
    
    Args:
        series: pandas Series with datetime index
        
    Returns:
        tuple: (model, feature_columns) or (None, None) if fitting fails
    """
    if not HAVE_XGBOOST or len(series) < 14:
        return None, None
    
    try:
        # Create DataFrame with time-based features
        df = pd.DataFrame({'target': series.values}, index=series.index)
        
        # Time-based features
        df['dayofweek'] = df.index.to_series().dt.dayofweek
        df['day'] = df.index.to_series().dt.day
        df['month'] = df.index.to_series().dt.month
        df['quarter'] = df.index.to_series().dt.quarter
        
        # Lag features
        for lag in [1, 7]:
            if len(df) > lag:
                df[f'lag_{lag}'] = df['target'].shift(lag)
        
        # Rolling features
        for window in [3, 7]:
            if len(df) >= window:
                df[f'rolling_mean_{window}'] = df['target'].rolling(window).mean()
        
        # Clean data
        df_clean = df.dropna()
        
        if len(df_clean) < 7:
            return None, None
        
        # Prepare features
        feature_cols = [col for col in df_clean.columns if col != 'target']
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        # Train XGBoost model
        model = None
        if HAVE_XGBOOST and xgb is not None:
            model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        
        if model is None:
            return None, None
        
        model.fit(X, y)
        return model, feature_cols
        
    except Exception:
        return None, None


def fit_arima_model(series):
    """
    Fit ARIMA model with automatic parameter selection.
    
    Args:
        series: pandas Series with datetime index
        
    Returns:
        dict: Model results or fallback if statsmodels unavailable
    """
    if not HAVE_STATSMODELS or len(series) < 10:
        return {'forecast': np.full(1, series.mean() if len(series) > 0 else 0), 'model_type': 'fallback_mean'}
    
    try:
        # Try simple ARIMA(1,1,1) model
        if HAVE_STATSMODELS and ARIMA is not None:
            model = ARIMA(series, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=1)
            forecast_value = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
            
            return {
                'model': fitted_model,
                'forecast': np.array([forecast_value]),
                'forecast_value': forecast_value,
                'model_type': 'arima'
            }
        else:
            return {'forecast': np.full(1, series.mean()), 'model_type': 'fallback_mean'}
        
    except Exception as e:
        print(f"ARIMA model failed: {e}")
        return {'forecast': np.full(1, series.mean()), 'model_type': 'fallback_mean'}


def fit_exponential_smoothing_model(series):
    """
    Fit Exponential Smoothing model.
    
    Args:
        series: pandas Series with datetime index
        
    Returns:
        dict: Model results or fallback if statsmodels unavailable
    """
    if not HAVE_STATSMODELS or len(series) < 10:
        return {'forecast': np.full(1, series.mean() if len(series) > 0 else 0), 'model_type': 'fallback_mean'}
    
    try:
        # Simple exponential smoothing
        if HAVE_STATSMODELS and ExponentialSmoothing is not None:
            model = ExponentialSmoothing(series, trend='add', seasonal=None)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=1)
            forecast_value = forecast.iloc[0] if hasattr(forecast, 'iloc') else forecast[0]
            
            return {
                'model': fitted_model,
                'forecast': np.array([forecast_value]),
                'forecast_value': forecast_value,
                'model_type': 'exponential_smoothing'
            }
        else:
            return {'forecast': np.full(1, series.mean()), 'model_type': 'fallback_mean'}
        
    except Exception as e:
        print(f"Exponential Smoothing model failed: {e}")
        return {'forecast': np.full(1, series.mean()), 'model_type': 'fallback_mean'}


def fit_seasonal_decompose_model(series):
    """
    Fit model using seasonal decomposition.
    
    Args:
        series: pandas Series with datetime index
        
    Returns:
        dict: Model results or fallback if statsmodels unavailable
    """
    if not HAVE_STATSMODELS or len(series) < 14:
        return {'forecast': np.full(1, series.mean() if len(series) > 0 else 0), 'model_type': 'fallback_mean'}
    
    try:
        # Perform seasonal decomposition
        if HAVE_STATSMODELS and seasonal_decompose is not None:
            decomposition = seasonal_decompose(series, model='additive', period=7)
            
            # Simple forecast based on trend and seasonal components
            trend_forecast = decomposition.trend.dropna().iloc[-1]
            seasonal_forecast = decomposition.seasonal.iloc[-7:].mean()  # Average weekly seasonal
            
            forecast_value = trend_forecast + seasonal_forecast
            
            return {
                'decomposition': decomposition,
                'forecast': np.array([forecast_value]),
                'forecast_value': forecast_value,
                'model_type': 'seasonal_decompose'
            }
        else:
            return {'forecast': np.full(1, series.mean()), 'model_type': 'fallback_mean'}
        
    except Exception as e:
        print(f"Seasonal decompose model failed: {e}")
        return {'forecast': np.full(1, series.mean()), 'model_type': 'fallback_mean'}


def fit_ensemble_model(series, individual_forecasts):
    """
    Create ensemble forecast by combining multiple models.
    
    Args:
        series: pandas Series with datetime index
        individual_forecasts: dict of individual model forecasts
        
    Returns:
        dict: Ensemble forecast results
    """
    if not individual_forecasts:
        return {'forecast': np.full(1, series.mean() if len(series) > 0 else 0), 'model_type': 'fallback_mean'}
    
    # Simple equal-weight ensemble
    valid_forecasts = []
    model_names = []
    
    for model_name, forecast_data in individual_forecasts.items():
        if 'forecast' in forecast_data and forecast_data['forecast'] is not None:
            if isinstance(forecast_data['forecast'], (list, np.ndarray)) and len(forecast_data['forecast']) > 0:
                valid_forecasts.append(forecast_data['forecast'][0] if len(forecast_data['forecast']) > 0 else 0)
                model_names.append(model_name)
            elif isinstance(forecast_data['forecast'], (int, float)):
                valid_forecasts.append(forecast_data['forecast'])
                model_names.append(model_name)
    
    if not valid_forecasts:
        return {'forecast': np.full(1, series.mean()), 'model_type': 'fallback_mean'}
    
    # Calculate ensemble forecast
    ensemble_forecast = np.mean(valid_forecasts)
    
    return {
        'forecast': ensemble_forecast,
        'component_forecasts': dict(zip(model_names, valid_forecasts)),
        'model_type': 'ensemble'
    }


def check_model_availability(series=None):
    """
    Quick diagnostic function to check which models are available and why.
    
    Args:
        series: optional pandas Series to check data requirements
        
    Returns:
        dict: Status of each model
    """
    status = {
        'prophet': {
            'installed': HAVE_PROPHET,
            'available': HAVE_PROPHET and (series is None or len(series) >= 14),
            'data_length': len(series) if series is not None else 'N/A',
            'min_required': 14
        },
        'lightgbm': {
            'installed': HAVE_LGBM,
            'available': HAVE_LGBM and (series is None or len(series) >= 14),
            'data_length': len(series) if series is not None else 'N/A',
            'min_required': 14
        },
        'xgboost': {
            'installed': HAVE_XGBOOST,
            'available': HAVE_XGBOOST and (series is None or len(series) >= 14),
            'data_length': len(series) if series is not None else 'N/A',
            'min_required': 14
        },
        'statsmodels': {
            'installed': HAVE_STATSMODELS,
            'available': HAVE_STATSMODELS and (series is None or len(series) >= 10),
            'data_length': len(series) if series is not None else 'N/A',
            'min_required': 10
        }
    }
    
    return status


def print_model_diagnostics(series=None):
    """Print a formatted diagnostic report of model availability."""
    status = check_model_availability(series)
    
    print("Model Availability Diagnostic Report")
    print("=" * 50)
    
    for model_name, info in status.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Installed: {'✓' if info['installed'] else '✗'}")
        print(f"  Available: {'✓' if info['available'] else '✗'}")
        print(f"  Data Length: {info['data_length']}")
        print(f"  Min Required: {info['min_required']}")
        
        if info['installed'] and not info['available'] and series is not None:
            print(f"  Issue: Data length ({info['data_length']}) < minimum required ({info['min_required']})")
    
    print("\n" + "=" * 50)
    return status
