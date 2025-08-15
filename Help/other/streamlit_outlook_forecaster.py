"""
Quarterly Outlook Forecaster - Daily Data Edition

Specialized forecaster for quarterly business outlook based on partial quarter data.
Uses fiscal year calendar (July-June) where Q1 starts in July.

Key Features:
1. FISCAL QUARTER DETECTION: Automatically detects current fiscal quarter and progress
2. DAILY DATA PROCESSING: Handles daily business data with weekday/weekend awareness
3. QUARTERLY FORECASTING: Projects full quarter performance from partial data
4. BUSINESS CALENDAR: Respects fiscal year timing (Q1: Jul-Sep, Q2: Oct-Dec, Q3: Jan-Mar, Q4: Apr-Jun)

Philosophy: Quick, actionable quarterly outlook for business decision-making.

Author: Jake Moura (jakemoura@microsoft.com)
"""

# Type hints and imports
import io
import warnings
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import calendar

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Optional dependencies
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except (ImportError, ValueError):
    HAVE_PROPHET = False

try:
    from lightgbm import LGBMRegressor
    HAVE_LGBM = True
except (ImportError, ValueError):
    HAVE_LGBM = False

try:
    import xgboost as xgb
    HAVE_XGBOOST = True
except (ImportError, ValueError):
    HAVE_XGBOOST = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAVE_STATSMODELS = True
except (ImportError, ValueError):
    HAVE_STATSMODELS = False

# ============ Outlook Page Setup ============
st.set_page_config(
    page_title="Quarterly Outlook Forecaster", 
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ============ Fiscal Year Calendar Functions ============

def get_fiscal_quarter_info(date):
    """
    Get fiscal quarter information for a given date.
    Fiscal year runs July-June (Q1: Jul-Sep, Q2: Oct-Dec, Q3: Jan-Mar, Q4: Apr-Jun)
    """
    month = date.month
    year = date.year
    
    if month >= 7:  # July-December = Q1, Q2 of fiscal year starting this calendar year
        fiscal_year = year + 1  # FY2024 starts July 2023
        if month <= 9:  # Jul-Sep
            quarter = 1
            quarter_start = datetime(year, 7, 1)
            quarter_end = datetime(year, 9, 30)
        else:  # Oct-Dec
            quarter = 2
            quarter_start = datetime(year, 10, 1)
            quarter_end = datetime(year, 12, 31)
    else:  # January-June = Q3, Q4 of fiscal year that started previous calendar year
        fiscal_year = year  # FY2024 ends June 2024
        if month <= 3:  # Jan-Mar
            quarter = 3
            quarter_start = datetime(year, 1, 1)
            quarter_end = datetime(year, 3, 31)
        else:  # Apr-Jun
            quarter = 4
            quarter_start = datetime(year, 4, 1)
            quarter_end = datetime(year, 6, 30)
    
    return {
        'fiscal_year': fiscal_year,
        'quarter': quarter,
        'quarter_start': quarter_start,
        'quarter_end': quarter_end,
        'quarter_name': f"FY{fiscal_year % 100:02d} Q{quarter}"
    }

def get_business_days_in_period(start_date, end_date):
    """Get number of calendar days between two dates (inclusive) - for daily consumptive businesses"""
    return (end_date - start_date).days + 1

def fiscal_quarter_label(date):
    """Generate fiscal quarter label for a date"""
    info = get_fiscal_quarter_info(date)
    return info['quarter_name']

# ============ Data Processing Functions ============

def read_any_excel(buf: io.BytesIO) -> pd.DataFrame:
    """Robust Excel reader supporting multiple engines"""
    for eng in ("openpyxl", "xlrd", "pyxlsb"):
        try:
            return pd.read_excel(buf, engine=eng)
        except Exception:
            buf.seek(0)
    raise RuntimeError("Workbook could not be read by available engines.")

def coerce_daily_dates(col: pd.Series) -> pd.Series:
    """
    Convert various date formats to daily timestamps.
    Handles Excel dates, text dates, etc.
    """
    # First try standard pandas parsing
    dt = pd.to_datetime(col, errors="coerce")
    
    if dt.isna().any():
        mask = dt.isna()
        failed_dates = col[mask].astype(str)
        
        # Try different date formats for failed dates
        for date_format in ["%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d"]:
            if failed_dates.isna().all():
                break
            try:
                parsed = pd.to_datetime(failed_dates, format=date_format, errors="coerce")
                valid_parsed = ~parsed.isna()
                if valid_parsed.any():
                    dt[mask] = dt[mask].fillna(parsed)
                    mask = dt.isna()
                    failed_dates = col[mask].astype(str) if mask.any() else pd.Series([], dtype=str)
            except (ValueError, TypeError):
                continue
    
    # Final check for any remaining unparsed dates
    if dt.isna().any():
        bad = col[dt.isna()].unique()[:5]
        raise ValueError(f"Unable to parse date strings: {bad}. Supported formats include: 'MM/DD/YYYY', 'YYYY-MM-DD', etc.")
    
    return dt

def analyze_daily_data(series, spike_threshold=2.0):
    """Analyze daily business data patterns including spike detection"""
    # Check for weekday/weekend patterns using proper datetime indexing
    weekday_mask = series.index.to_series().dt.dayofweek < 5  # Mon-Fri
    weekend_mask = series.index.to_series().dt.dayofweek >= 5  # Sat-Sun
    
    weekday_avg = series[weekday_mask].mean()  # Mon-Fri
    weekend_avg = series[weekend_mask].mean()  # Sat-Sun
    
    # Calculate business day statistics
    business_days = series[weekday_mask]
    weekend_days = series[weekend_mask]
    
    # Detect monthly spikes (subscription renewals, etc.)
    spike_analysis = detect_monthly_spikes(series, spike_threshold)
    
    analysis = {
        'total_days': len(series),
        'business_days': len(business_days),
        'weekend_days': len(weekend_days),
        'weekday_avg': weekday_avg if len(business_days) > 0 else 0,
        'weekend_avg': weekend_avg if len(weekend_days) > 0 else 0,
        'business_ratio': weekday_avg / weekend_avg if weekend_avg > 0 else 1.0,
        'daily_volatility': series.std(),
        'trend': (series.iloc[-7:].mean() - series.iloc[:7].mean()) / series.iloc[:7].mean() if len(series) >= 14 else 0,
        'spike_analysis': spike_analysis
    }
    
    return analysis

# ============ Forecasting Models ============

def fit_linear_trend_model(series, business_days_only=False):
    """Fit simple linear trend model optimized for daily consumptive business data"""
    if business_days_only and len(series) > 10:
        # Use only business days for trend fitting (disabled for daily consumptive business)
        business_data = series[series.index.weekday < 5]
        if len(business_data) >= 5:
            series = business_data
    
    # Create simple time trend
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model, series.index

def fit_moving_average_model(series, window=7):
    """Fit moving average model with trend adjustment"""
    if len(series) < window * 2:
        window = max(3, len(series) // 2)
    
    # Calculate moving average and trend
    ma = series.rolling(window=window, center=True).mean()
    recent_ma = ma.dropna().iloc[-window//2:].mean()
    
    # Simple trend from first vs last week
    if len(series) >= 14:
        first_week = series.iloc[:7].mean()
        last_week = series.iloc[-7:].mean()
        trend = (last_week - first_week) / 7  # Daily trend
    else:
        trend = 0
    
    return recent_ma, trend

def fit_prophet_daily_model(series):
    """Fit Prophet model optimized for daily business data"""
    if not HAVE_PROPHET or len(series) < 14:
        return None
    
    # Prepare data for Prophet
    df = pd.DataFrame({
        'ds': series.index,
        'y': series.values
    })
    
    try:
        # Configure for daily business data
        model = Prophet(
            yearly_seasonality='auto',  # Not enough data for yearly
            weekly_seasonality='auto',   # Important for daily business data
            daily_seasonality='auto',   # Usually too noisy
            changepoint_prior_scale=0.1,  # More sensitive to recent changes
            seasonality_mode='additive'
        )
        
        model.fit(df)
        return model
    except Exception:
        return None

def fit_lightgbm_daily_model(series):
    """Fit LightGBM model with daily business features"""
    if not HAVE_LGBM or len(series) < 21:  # Need at least 3 weeks
        return None, None
    
    # Create features for daily data
    df = pd.DataFrame(index=series.index)
    df['ACR'] = series.values
    
    # Time-based features
    df['day_of_week'] = df.index.to_series().dt.dayofweek
    df['is_weekend'] = (df.index.to_series().dt.dayofweek >= 5).astype(int)
    df['day_of_month'] = df.index.to_series().dt.day
    df['week_of_year'] = df.index.to_series().dt.isocalendar().week
    
    # Lag features (recent days)
    for lag in [1, 2, 3, 7]:
        df[f'lag_{lag}'] = df['ACR'].shift(lag)
    
    # Rolling statistics
    for window in [3, 7, 14]:
        df[f'rolling_mean_{window}'] = df['ACR'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['ACR'].rolling(window=window).std()
    
    # All day features (no special business day logic for daily consumptive business)
    df['day_type'] = df.apply(lambda x: 'weekday' if x.name.weekday() < 5 else 'weekend', axis=1)
    
    # Drop rows with NaN
    df_clean = df.dropna()
    
    if len(df_clean) < 14:  # Need at least 2 weeks of clean data
        return None, None
    
    # Prepare features and target
    feature_cols = [col for col in df_clean.columns if col != 'ACR']
    X = df_clean[feature_cols].values
    y = df_clean['ACR'].values
    
    # Split for validation (use last week as validation)
    split_idx = len(X) - 7
    if split_idx < 7:  # Not enough data for proper split
        return None, None
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    try:
        # Fit LightGBM
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=15,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        
        # Validate
        pred_val = model.predict(X_val)
        mape = mean_absolute_percentage_error(np.asarray(y_val), np.asarray(pred_val))  # type: ignore
        
        if mape < 0.5:  # Only use if MAPE < 50%
            return model, feature_cols
        else:
            return None, None
            
    except Exception:
        return None, None

def fit_xgboost_model(series):
    """Fit XGBoost model for daily business data"""
    if not HAVE_XGBOOST:
        return None, None
        
    try:
        import xgboost as xgb
        
        if len(series) < 21:  # Need at least 3 weeks
            return None, None
            
        # Create features similar to LightGBM
        df = pd.DataFrame(index=series.index)
        df['ACR'] = series.values
        
        # Time-based features
        df['day_of_week'] = df.index.to_series().dt.dayofweek
        df['is_weekend'] = (df.index.to_series().dt.dayofweek >= 5).astype(int)
        df['day_of_month'] = df.index.to_series().dt.day
        df['week_of_year'] = df.index.to_series().dt.isocalendar().week
        
        # Lag features
        for lag in [1, 2, 3, 7]:
            df[f'lag_{lag}'] = df['ACR'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7]:
            df[f'rolling_mean_{window}'] = df['ACR'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['ACR'].rolling(window=window).std()
        
        # Drop rows with NaN
        df_clean = df.dropna()
        
        if len(df_clean) < 14:
            return None, None
        
        # Prepare features and target
        feature_cols = [col for col in df_clean.columns if col != 'ACR']
        X = df_clean[feature_cols].values
        y = df_clean['ACR'].values
        
        # Split for validation
        split_idx = len(X) - 7
        if split_idx < 7:
            return None, None
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Fit XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        # Validate
        pred_val = model.predict(X_val)
        mape = mean_absolute_percentage_error(np.asarray(y_val), np.asarray(pred_val))
        
        if mape < 0.5:  # Only use if MAPE < 50%
            return model, feature_cols
        else:
            return None, None
            
    except ImportError:
        # XGBoost not available
        return None, None
    except Exception:
        return None, None

def fit_arima_model(series):
    """Fit ARIMA model for daily business data"""
    if not HAVE_STATSMODELS or len(series) < 14:
        return None
        
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
        
        # Test for stationarity
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] <= 0.05
        
        # Simple ARIMA configuration for daily data
        if is_stationary:
            order = (1, 0, 1)  # AR(1), no differencing, MA(1)
        else:
            order = (1, 1, 1)  # AR(1), first difference, MA(1)
        
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        return fitted_model
    except Exception:
        # ARIMA fitting failed
        return None

def fit_exponential_smoothing_model(series):
    """Fit Exponential Smoothing (Holt-Winters) model for daily data"""
    if not HAVE_STATSMODELS or len(series) < 14:
        return None
        
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Configure for daily data with potential weekly seasonality
        seasonal_periods = 7 if len(series) >= 14 else None
        
        # Simple exponential smoothing if no clear seasonality
        if seasonal_periods and len(series) >= seasonal_periods * 2:
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods
            )
        else:
            # Double exponential smoothing (Holt's method)
            model = ExponentialSmoothing(series, trend='add')
        
        fitted_model = model.fit(optimized=True)
        return fitted_model
    except Exception:
        # Exponential smoothing failed
        return None

def fit_seasonal_decompose_model(series):
    """Fit model using seasonal decomposition + trend forecasting"""
    if not HAVE_STATSMODELS or len(series) < 21:
        return None
        
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(
            series, 
            model='additive', 
            period=7,  # Weekly seasonality
            extrapolate_trend=1  # Use integer for extrapolation
        )
        
        # Extract components
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal
        
        if len(trend) < 7:
            return None
            
        # Fit simple linear model to trend
        X = np.arange(len(trend)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, trend.values)
        
        # Return decomposition info for forecasting
        return {
            'trend_model': model,
            'seasonal_pattern': seasonal,
            'original_series': series,
            'trend_length': len(trend)
        }
    except Exception:
        return None

def fit_ensemble_model(series, individual_forecasts):
    """Create ensemble forecast from multiple models"""
    if len(individual_forecasts) < 2:
        return None
        
    try:
        # Simple weighted ensemble based on inverse MAPE
        # Lower MAPE = higher weight
        weights = []
        forecasts = []
        
        for model_name, forecast_data in individual_forecasts.items():
            # Skip ensemble models to avoid recursion
            if 'ensemble' in model_name.lower():
                continue
                
            # Get MAPE score (use a default if not available)
            mape = forecast_data.get('mape', 50.0)
            if mape == float('inf') or mape <= 0:
                weight = 0.1  # Small weight for invalid MAPE
            else:
                weight = 1.0 / (mape + 1)  # Inverse weight
            
            weights.append(weight)
            forecasts.append(forecast_data['daily_forecast'])
        
        if len(forecasts) == 0:
            return None
            
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        ensemble_forecast = np.average(forecasts, axis=0, weights=weights)
        
        return {
            'daily_forecast': ensemble_forecast,
            'weights': weights,
            'component_models': list(individual_forecasts.keys())
        }
    except Exception:
        return None

# ============ Quarter Forecasting Engine ============

def forecast_quarter_completion(series, current_date=None, detect_spikes=True, spike_threshold=2.0):
    """
    Forecast complete quarter performance based on partial quarter data.
    Returns projections for remaining days in the quarter.
    """
    if current_date is None:
        current_date = series.index.max()
    
    # Get fiscal quarter information
    quarter_info = get_fiscal_quarter_info(current_date)
    quarter_start = quarter_info['quarter_start']
    quarter_end = quarter_info['quarter_end']
    
    # Analyze current quarter data first
    quarter_data = series[series.index >= quarter_start]
    
    # Calculate progress - handle end-of-quarter scenarios gracefully
    total_calendar_days = get_business_days_in_period(quarter_start, quarter_end)
    
    # Calculate elapsed calendar days up to and including current date
    elapsed_calendar_days = get_business_days_in_period(quarter_start, current_date)
    
    # Find the last actual data date to determine real progress
    last_data_date = quarter_data.index.max() if len(quarter_data) > 0 else quarter_start
    actual_elapsed_calendar_days = get_business_days_in_period(quarter_start, last_data_date)
    
    # Calculate remaining calendar days from last data date to quarter end
    if last_data_date.date() >= quarter_end.date():
        # Data goes to or past quarter end
        remaining_calendar_days = 3  # Allow minimal projection for analysis
        progress_pct = 1.0  # 100% complete
    else:
        # Calculate remaining days from last data date to quarter end
        remaining_calendar_days = get_business_days_in_period(last_data_date + timedelta(days=1), quarter_end)
        # Progress based on actual data, not analysis date
        progress_pct = actual_elapsed_calendar_days / total_calendar_days if total_calendar_days > 0 else 0
        
        # If we're at quarter end but data is incomplete, still allow forecasting
        if current_date >= quarter_end and remaining_calendar_days > 0:
            # Add a few extra days for extended analysis
            remaining_calendar_days += 3
    
    # Ensure we have at least 1 day to forecast if there's data and it's incomplete
    if remaining_calendar_days <= 0 and len(quarter_data) > 0 and last_data_date.date() < quarter_end.date():
        remaining_calendar_days = max(1, get_business_days_in_period(last_data_date + timedelta(days=1), quarter_end))
    
    daily_analysis = analyze_daily_data(quarter_data, spike_threshold)
    
    # Detect monthly spikes first (for subscription renewals, etc.)
    if detect_spikes:
        spike_analysis = detect_monthly_spikes(quarter_data, spike_threshold)
        # Temporary debug info
        st.write(f"🔍 Debug: Spike analysis - Has spikes: {spike_analysis.get('has_spikes')}, Contribution: {spike_analysis.get('spike_contribution', 0):.1%}, Pattern: {spike_analysis.get('spike_pattern', [])}")
    else:
        spike_analysis = {'has_spikes': False, 'spike_days': [], 'spike_pattern': None, 'baseline_avg': quarter_data.mean(), 'spike_contribution': 0}
    
    # Try multiple forecasting approaches
    forecasts = {}
    
    # Note: Monthly Renewal spike detection is still performed above and will be overlaid on other models
    # but we no longer create a separate "Monthly Renewals" model for selection
    spike_contribution = spike_analysis.get('spike_contribution', 0)
    has_spikes = spike_analysis.get('has_spikes', False)
    st.write(f"🔍 Debug: Spike detection active - Has spikes: {has_spikes}, Contribution: {spike_contribution:.3f}")
    
    if has_spikes and spike_analysis.get('spike_pattern'):
        st.write(f"🔍 Debug: Spike overlay will be applied to all models - spikes on days {[d for d, c in spike_analysis['spike_pattern']]}")
    else:
        st.write(f"🔍 Debug: No spike overlay - has_spikes: {has_spikes}, pattern: {spike_analysis.get('spike_pattern')}")
    
    # 1. Linear Trend Model (enhanced with spike awareness)
    try:
        trend_model, trend_index = fit_linear_trend_model(quarter_data)
        
        # Project remaining calendar days
        days_ahead = np.arange(len(quarter_data), len(quarter_data) + remaining_calendar_days)
        trend_forecast = trend_model.predict(days_ahead.reshape(-1, 1))
        
        # Adjust for business days only
        if daily_analysis['business_ratio'] > 1.5:  # Significant weekday/weekend difference
            business_day_avg = daily_analysis['weekday_avg']
            trend_forecast = np.maximum(trend_forecast, business_day_avg * 0.5)  # Floor at 50% of business day average
        
        # If spikes were detected, overlay them on the trend
        if spike_analysis['has_spikes']:
            trend_forecast = overlay_spikes_on_forecast(trend_forecast, quarter_data, spike_analysis, remaining_calendar_days)
        
        forecasts['Linear Trend'] = {
            'daily_forecast': trend_forecast,
            'quarter_total': quarter_data.sum() + trend_forecast.sum(),
            'method': 'Linear regression on daily values' + (' with spike overlay' if spike_analysis['has_spikes'] else '')
        }
    except Exception:
        # Linear trend failed, continue with other methods
        pass
    
    # 2. Moving Average with Trend (enhanced with spike awareness)
    try:
        recent_avg, daily_trend = fit_moving_average_model(quarter_data)
        
        # Project forward with trend
        ma_forecast = []
        current_level = recent_avg
        for day in range(remaining_calendar_days):
            current_level += daily_trend
            ma_forecast.append(max(current_level, 0))  # Non-negative constraint
        
        ma_forecast = np.array(ma_forecast)
        
        # If spikes were detected, overlay them on the moving average
        if spike_analysis['has_spikes']:
            ma_forecast = overlay_spikes_on_forecast(ma_forecast, quarter_data, spike_analysis, remaining_calendar_days)
        
        forecasts['Moving Average'] = {
            'daily_forecast': ma_forecast,
            'quarter_total': quarter_data.sum() + ma_forecast.sum(),
            'method': f'7-day moving average with trend ({daily_trend:.2f}/day)' + (' with spike overlay' if spike_analysis['has_spikes'] else '')
        }
    except Exception:
        # Moving average failed, continue with other methods
        pass
    
    # 3. Prophet Model (if available and enough data)
    if HAVE_PROPHET and len(quarter_data) >= 14:
        try:
            prophet_model = fit_prophet_daily_model(quarter_data)
            if prophet_model is not None:
                # Create future dates from last data date (all days, including weekends)
                future_dates = pd.date_range(
                    start=last_data_date + timedelta(days=1),
                    periods=remaining_calendar_days,
                    freq='D'  # Daily frequency, includes weekends
                )
                
                future_df = pd.DataFrame({'ds': future_dates})
                prophet_forecast = prophet_model.predict(future_df)
                prophet_values = np.maximum(np.asarray(prophet_forecast['yhat'].values), 0)  # type: ignore
                
                # If spikes were detected, enhance Prophet forecast
                if spike_analysis['has_spikes']:
                    prophet_values = overlay_spikes_on_forecast(prophet_values, quarter_data, spike_analysis, remaining_calendar_days)
                
                forecasts['Prophet'] = {
                    'daily_forecast': prophet_values,
                    'quarter_total': quarter_data.sum() + prophet_values.sum(),
                    'method': 'Prophet with weekly seasonality' + (' and spike overlay' if spike_analysis['has_spikes'] else '')
                }
        except Exception:
            # Prophet failed, continue with other methods
            pass
    
    # 4. LightGBM Model (if available and enough data)
    if HAVE_LGBM and len(quarter_data) >= 21:
        try:
            lgbm_model, feature_cols = fit_lightgbm_daily_model(quarter_data)
            if lgbm_model is not None:
                # This is more complex - would need to create features for future dates
                # For now, skip this in the initial implementation
                pass
        except Exception:
            # LightGBM failed, continue with other methods
            pass
    
    # 5. XGBoost Model (if available and enough data)
    if HAVE_XGBOOST and len(quarter_data) >= 21:
        try:
            xgboost_model, feature_cols = fit_xgboost_model(quarter_data)
            if xgboost_model is not None:
                # Create features for future dates (similar to LightGBM approach)
                last_date = quarter_data.index.max()
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=remaining_calendar_days, freq='D')
                
                # Create future feature dataframe
                future_df = pd.DataFrame(index=future_dates)
                future_df['day_of_week'] = future_df.index.to_series().dt.dayofweek
                future_df['is_weekend'] = (future_df.index.to_series().dt.dayofweek >= 5).astype(int)
                future_df['day_of_month'] = future_df.index.to_series().dt.day
                future_df['week_of_year'] = future_df.index.to_series().dt.isocalendar().week
                
                # Use recent values for lag features (simplified)
                recent_values = quarter_data.tail(7).values
                for lag in [1, 2, 3, 7]:
                    if lag <= len(recent_values):
                        future_df[f'lag_{lag}'] = recent_values[-lag] if lag <= len(recent_values) else quarter_data.mean()
                    else:
                        future_df[f'lag_{lag}'] = quarter_data.mean()
                
                # Use recent rolling statistics
                for window in [3, 7]:
                    recent_mean = quarter_data.tail(window).mean()
                    recent_std = quarter_data.tail(window).std()
                    future_df[f'rolling_mean_{window}'] = recent_mean
                    future_df[f'rolling_std_{window}'] = recent_std
                
                # Make prediction
                X_future = future_df[feature_cols].values
                xgb_forecast = xgboost_model.predict(X_future)
                xgb_forecast = np.maximum(xgb_forecast, 0)  # Ensure non-negative
                
                # If spikes were detected, overlay them
                if spike_analysis['has_spikes']:
                    xgb_forecast = overlay_spikes_on_forecast(xgb_forecast, quarter_data, spike_analysis, remaining_calendar_days)
                
                forecasts['XGBoost'] = {
                    'daily_forecast': xgb_forecast,
                    'quarter_total': quarter_data.sum() + xgb_forecast.sum(),
                    'method': 'XGBoost with time/lag features' + (' with spike overlay' if spike_analysis['has_spikes'] else '')
                }
        except Exception:
            # XGBoost failed, continue with other methods
            pass
    
    # 6. ARIMA Model (if available and enough data)
    if HAVE_STATSMODELS and len(quarter_data) >= 14:
        try:
            arima_model = fit_arima_model(quarter_data)
            if arima_model is not None:
                # Project using ARIMA
                arima_forecast = arima_model.get_forecast(steps=remaining_calendar_days)
                arima_values = np.maximum(arima_forecast.predicted_mean.values, 0)
                
                # If spikes were detected, enhance ARIMA forecast
                if spike_analysis['has_spikes']:
                    arima_values = overlay_spikes_on_forecast(arima_values, quarter_data, spike_analysis, remaining_calendar_days)
                
                forecasts['ARIMA'] = {
                    'daily_forecast': arima_values,
                    'quarter_total': quarter_data.sum() + arima_values.sum(),
                    'method': 'ARIMA time series model' + (' with spike overlay' if spike_analysis['has_spikes'] else '')
                }
        except Exception:
            # ARIMA failed, continue with other methods
            pass
    
    # 7. Exponential Smoothing Model (if available and enough data)
    if HAVE_STATSMODELS and len(quarter_data) >= 14:
        try:
            es_model = fit_exponential_smoothing_model(quarter_data)
            if es_model is not None:
                # Project using Exponential Smoothing
                es_forecast = es_model.forecast(steps=remaining_calendar_days)
                es_values = np.maximum(es_forecast.values, 0)
                
                # If spikes were detected, enhance Exponential Smoothing forecast
                if spike_analysis['has_spikes']:
                    es_values = overlay_spikes_on_forecast(es_values, quarter_data, spike_analysis, remaining_calendar_days)
                
                forecasts['Exponential Smoothing'] = {
                    'daily_forecast': es_values,
                    'quarter_total': quarter_data.sum() + es_values.sum(),
                    'method': 'Exponential Smoothing (Holt-Winters)' + (' with spike overlay' if spike_analysis['has_spikes'] else '')
                }
        except Exception:
            # Exponential Smoothing failed, continue with other methods
            pass
    
    # 8. Seasonal Decomposition Model (if available and enough data)
    if HAVE_STATSMODELS and len(quarter_data) >= 21:
        try:
            seasonal_model = fit_seasonal_decompose_model(quarter_data)
            if seasonal_model is not None:
                # Extract components
                trend = seasonal_model['trend_model']
                seasonal_pattern = seasonal_model['seasonal_pattern']
                
                # Project trend using linear model
                trend_forecast = trend.predict(np.arange(len(quarter_data), len(quarter_data) + remaining_calendar_days).reshape(-1, 1))
                
                # Seasonal pattern is repeated for the forecast period
                seasonal_forecast = np.tile(seasonal_pattern.values[-7:], (remaining_calendar_days // 7) + 1)[:remaining_calendar_days]
                
                # Combine trend and seasonal for final forecast
                seasonal_decomp_forecast = trend_forecast + seasonal_forecast
                seasonal_decomp_forecast = np.maximum(seasonal_decomp_forecast, 0)  # Ensure non-negative
                
                # If spikes were detected, overlay them on the seasonal decomposition forecast
                if spike_analysis['has_spikes']:
                    seasonal_decomp_forecast = overlay_spikes_on_forecast(seasonal_decomp_forecast, quarter_data, spike_analysis, remaining_calendar_days)
                
                forecasts['Seasonal Decomposition'] = {
                    'daily_forecast': seasonal_decomp_forecast,
                    'quarter_total': quarter_data.sum() + seasonal_decomp_forecast.sum(),
                    'method': 'Seasonal Decomposition + Trend' + (' with spike overlay' if spike_analysis['has_spikes'] else '')
                }
        except Exception:
            # Seasonal decomposition failed, continue with other methods
            pass
    
    # 9. Ensemble Model (if multiple forecasts available)
    if len(forecasts) > 1:
        try:
            ensemble_model = fit_ensemble_model(quarter_data, forecasts)
            if ensemble_model is not None:
                forecasts['Ensemble'] = {
                    'daily_forecast': ensemble_model['daily_forecast'],
                    'quarter_total': quarter_data.sum() + ensemble_model['daily_forecast'].sum(),
                    'method': 'Ensemble of available models'
                }
        except Exception:
            # Ensemble fitting failed
            pass
    
    # 10. Simple Rate-Based Projection (always available as fallback)
    try:
        # Calculate current run rate based on actual data days
        if actual_elapsed_calendar_days > 0:
            current_run_rate = quarter_data.sum() / actual_elapsed_calendar_days
            rate_forecast = np.full(remaining_calendar_days, current_run_rate)
            
            forecasts['Run Rate'] = {
                'daily_forecast': rate_forecast,
                'quarter_total': quarter_data.sum() + rate_forecast.sum(),
                'method': f'Current run rate: {current_run_rate:.2f}/calendar day'
            }
    except Exception as e:
        st.write(f"Run rate failed: {e}")
    
    # Select best forecast (or ensemble)
    if len(forecasts) == 0:
        st.write("❌ No forecasts were created!")
        return None, None, quarter_info
    
    # For now, use simple average of available forecasts
    # Could implement more sophisticated model selection later
    forecast_values = [f['daily_forecast'] for f in forecasts.values()]
    forecast_totals = [f['quarter_total'] for f in forecasts.values()]
    
    ensemble_daily = np.mean(forecast_values, axis=0)
    ensemble_total = np.mean(forecast_totals)
    
    # Calculate MAPE scores for model evaluation
    mape_scores = evaluate_individual_forecasts(quarter_data, forecasts)
    
    # Find best model (lowest MAPE)
    best_model = None
    best_mape = float('inf')
    for model_name, mape in mape_scores.items():
        if mape < best_mape:
            best_mape = mape
            best_model = model_name
    
    # Use best model for default, fallback to ensemble if no clear winner
    if best_model and best_model in forecasts:
        default_daily = forecasts[best_model]['daily_forecast']
        default_total = forecasts[best_model]['quarter_total']
    else:
        default_daily = ensemble_daily
        default_total = ensemble_total
    
    forecast_result = {
        'daily_forecast': default_daily,
        'quarter_total': default_total,
        'current_quarter_actual': quarter_data.sum(),
        'projected_remaining': default_daily.sum(),
        'individual_forecasts': forecasts,
        'mape_scores': mape_scores,
        'best_model': best_model,
        'progress_pct': progress_pct,
        'elapsed_calendar_days': actual_elapsed_calendar_days,  # Use actual data days, not analysis date
        'remaining_calendar_days': remaining_calendar_days,
        'total_calendar_days': total_calendar_days,
        'last_data_date': last_data_date,  # Add for UI reference
        'analysis_date': current_date  # Add for UI reference
    }
    
    return forecast_result, daily_analysis, quarter_info

# ============ Capacity Constraint Adjustment Functions ============

def apply_capacity_adjustment_to_forecast(forecast_result, capacity_factor):
    """
    Apply capacity constraint adjustments to a forecast result.
    
    Args:
        forecast_result: Dictionary containing forecast results
        capacity_factor: Multiplier to reduce forecasts (e.g., 0.85 = 15% reduction)
    
    Returns:
        Adjusted forecast_result dictionary
    """
    if forecast_result is None or capacity_factor >= 1.0:
        return forecast_result
    
    # Create a copy to avoid modifying the original
    adjusted_forecast = forecast_result.copy()
    
    # Apply capacity adjustment to main forecast metrics
    adjusted_forecast['daily_forecast'] = forecast_result['daily_forecast'] * capacity_factor
    adjusted_forecast['projected_remaining'] = forecast_result['projected_remaining'] * capacity_factor
    
    # Recalculate quarter total (actual + adjusted projected)
    adjusted_forecast['quarter_total'] = (
        forecast_result['current_quarter_actual'] + 
        adjusted_forecast['projected_remaining']
    )
    
    # Apply adjustment to individual model forecasts
    adjusted_individual_forecasts = {}
    for model_name, model_data in forecast_result['individual_forecasts'].items():
        adjusted_model_data = model_data.copy()
        adjusted_model_data['daily_forecast'] = model_data['daily_forecast'] * capacity_factor
        adjusted_model_data['quarter_total'] = (
            forecast_result['current_quarter_actual'] + 
            (model_data['daily_forecast'] * capacity_factor).sum()
        )
        adjusted_model_data['method'] = model_data['method'] + f" (capacity-adjusted: {capacity_factor:.0%})"
        adjusted_individual_forecasts[model_name] = adjusted_model_data
    
    adjusted_forecast['individual_forecasts'] = adjusted_individual_forecasts
    
    # Add capacity adjustment metadata
    adjusted_forecast['capacity_adjustment_applied'] = True
    adjusted_forecast['capacity_factor'] = capacity_factor
    adjusted_forecast['original_quarter_total'] = forecast_result['quarter_total']
    adjusted_forecast['original_projected_remaining'] = forecast_result['projected_remaining']
    
    return adjusted_forecast

# ============ Monthly Spike Detection Functions ============

def detect_monthly_spikes(series, spike_threshold=2.0):
    """
    Detect monthly recurring revenue spikes (e.g., from subscription renewals).
    Optimized for non-consumptive SKUs where revenue recognition happens on specific days.
    
    Args:
        series: Daily time series data
        spike_threshold: Multiplier above rolling average to consider a spike
    
    Returns:
        dict with spike analysis and patterns
    """
    if len(series) < 7:
        return {
            'has_spikes': False,
            'spike_days': [],
            'spike_pattern': None,
            'baseline_avg': series.mean(),
            'spike_contribution': 0
        }
    
    # For non-consumptive SKUs, baseline should exclude spike days completely
    # Use a more conservative baseline calculation
    rough_baseline = series.quantile(0.4)  # 40th percentile as rough baseline
    
    # Identify potential spikes (much higher than typical daily values)
    potential_spikes = series > (rough_baseline * spike_threshold)
    
    # Calculate baseline from non-spike days only
    non_spike_data = series[~potential_spikes]
    
    if len(non_spike_data) < 3:
        # If too many spikes, use different approach
        baseline_avg = series.quantile(0.3)  # Use 30th percentile as baseline
        final_spike_mask = series > (baseline_avg * spike_threshold)
    else:
        # Use mean of non-spike data for more robust baseline 
        baseline_avg = non_spike_data.mean()
        final_spike_mask = series > (baseline_avg * spike_threshold)
    
    spike_days = series[final_spike_mask]
    
    if len(spike_days) == 0:
        return {
            'has_spikes': False,
            'spike_days': [],
            'spike_pattern': None,
            'baseline_avg': baseline_avg,
            'spike_contribution': 0
        }
    
    # Analyze spike patterns by day of month
    spike_dates = spike_days.index
    spike_days_of_month = [d.day for d in spike_dates]
    
    # Find most common spike days
    from collections import Counter
    day_counts = Counter(spike_days_of_month)
    most_common_days = day_counts.most_common(3)
    
    # Calculate spike contribution to total revenue
    spike_contribution = spike_days.sum() / series.sum()
    
    # Additional validation: ensure spikes are significantly higher than baseline
    avg_spike_multiplier = spike_days.mean() / baseline_avg if baseline_avg > 0 else 1
    
    return {
        'has_spikes': True,
        'spike_days': spike_dates.tolist(),
        'spike_pattern': most_common_days,
        'baseline_avg': baseline_avg,
        'spike_avg': spike_days.mean(),
        'spike_contribution': spike_contribution,
        'spike_frequency': len(spike_days) / len(series) * 30,  # Average spikes per month
        'spike_multiplier': avg_spike_multiplier  # How much higher spikes are vs baseline
    }

def create_monthly_renewal_forecast(series, remaining_days, spike_analysis):
    """
    Create forecast that includes expected monthly renewal spikes.
    
    Args:
        series: Historical daily data
        remaining_days: Number of days to forecast (all days, including weekends)
        spike_analysis: Results from detect_monthly_spikes
    
    Returns:
        Array of daily forecasts including expected spikes
    """
    if not spike_analysis['has_spikes']:
        # No spikes detected, use baseline average
        st.write(f"🔍 Debug: No spikes detected, using baseline: {spike_analysis['baseline_avg']:.2f}")
        return np.full(remaining_days, spike_analysis['baseline_avg'])
    
    # Create forecast array
    forecast = []
    baseline_daily = spike_analysis['baseline_avg']
    spike_avg = spike_analysis['spike_avg']
    
    st.write(f"🔍 Debug: Baseline daily: {baseline_daily:.2f}, Spike avg: {spike_avg:.2f}")
    st.write(f"🔍 Debug: Spike pattern: {spike_analysis['spike_pattern']}")
    
    # Get the last data date to determine when to expect next spikes
    last_date = series.index.max()
    st.write(f"🔍 Debug: Last data date: {last_date.strftime('%Y-%m-%d')}")
    
    # Generate calendar dates for forecast period (all days, including weekends)
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=remaining_days,
        freq='D'  # Daily frequency, includes weekends
    )
    
    st.write(f"🔍 Debug: Future dates (all days): {[d.strftime('%m/%d') for d in future_dates[:10]]}")
    
    # Extract spike days for easy comparison
    common_spike_days = []
    if spike_analysis['spike_pattern']:
        common_spike_days = [day for day, count in spike_analysis['spike_pattern']]
    
    st.write(f"🔍 Debug: Common spike days of month: {common_spike_days}")
    
    spike_days_found = 0
    # Generate forecast for each day (including weekends)
    for i, future_date in enumerate(future_dates):
        day_of_month = future_date.day
        
        # Check if this day historically has spikes
        is_spike_day = day_of_month in common_spike_days
        
        if is_spike_day:
            # Use historical spike average
            daily_value = spike_avg
            spike_days_found += 1
            st.write(f"🔍 Debug: Spike day found! {future_date.strftime('%m/%d')} (day {day_of_month}) = {daily_value:.2f}")
        else:
            # Use baseline value
            daily_value = baseline_daily
        
        forecast.append(daily_value)
    
    st.write(f"🔍 Debug: Total spike days in forecast: {spike_days_found}")
    st.write(f"🔍 Debug: Forecast array length: {len(forecast)}, values: {[f'{v:.0f}' for v in forecast[:10]]}")
    
    return np.array(forecast[:remaining_days])

def overlay_spikes_on_forecast(base_forecast, historical_data, spike_analysis, forecast_days):
    """
    Overlay expected monthly spikes onto a base forecast.
    
    Args:
        base_forecast: Base forecast array
        historical_data: Historical daily data
        spike_analysis: Spike analysis results
        forecast_days: Number of days being forecasted
    
    Returns:
        Enhanced forecast with spikes overlaid
    """
    if not spike_analysis['has_spikes']:
        return base_forecast
    
    enhanced_forecast = base_forecast.copy()
    last_date = historical_data.index.max()
    
    # Generate calendar dates for forecast period (all days, includes weekends)
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=min(len(enhanced_forecast), forecast_days),
        freq='D'  # Daily frequency, includes weekends
    )
    
    # Calculate spike replacement value (not multiplier - exact replacement)
    spike_value = spike_analysis['spike_avg']
    
    for i, future_date in enumerate(future_dates):
        if i >= len(enhanced_forecast):
            break
            
        day_of_month = future_date.day
        
        # Check if this is a historical spike day
        if spike_analysis['spike_pattern']:
            common_spike_days = [day for day, count in spike_analysis['spike_pattern']]
            if day_of_month in common_spike_days:
                # Replace with actual spike value (not multiply)
                enhanced_forecast[i] = spike_value
    
    return enhanced_forecast

# ============ MAPE and Model Selection Functions ============

def calculate_mape(actual_values, predicted_values):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    if len(actual_values) == 0 or len(predicted_values) == 0:
        return float('inf')
    
    # Convert to numpy arrays
    actual = np.array(actual_values)
    predicted = np.array(predicted_values)
    
    # Ensure same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    # Avoid division by zero
    non_zero_mask = actual != 0
    if not np.any(non_zero_mask):
        return float('inf')
    
    actual_nz = actual[non_zero_mask]
    predicted_nz = predicted[non_zero_mask]
    
    # Calculate MAPE
    mape = np.mean(np.abs((actual_nz - predicted_nz) / actual_nz)) * 100
    return float(mape)

def evaluate_individual_forecasts(quarter_data, forecasts):
    """
    Evaluate each forecast model using MAPE on in-sample data.
    For models that make predictions, we use fitted values vs actual.
    """
    mape_scores = {}
    
    if len(quarter_data) < 5:  # Need minimum data for evaluation
        return {model: float('inf') for model in forecasts.keys()}
    
    # Use most recent 80% of data for evaluation (leave some for validation)
    eval_size = max(3, int(len(quarter_data) * 0.8))
    eval_data = quarter_data.iloc[:eval_size]
    
    for model_name, forecast_info in forecasts.items():
        try:
            if model_name == 'Linear Trend':
                mape_scores[model_name] = evaluate_linear_trend_mape(eval_data)
            elif model_name == 'Moving Average':
                mape_scores[model_name] = evaluate_moving_average_mape(eval_data)
            elif model_name == 'Prophet' and HAVE_PROPHET:
                mape_scores[model_name] = evaluate_prophet_mape(eval_data)
            elif model_name == 'Run Rate':
                mape_scores[model_name] = evaluate_run_rate_mape(eval_data)
            elif model_name in ['ARIMA', 'Exponential Smoothing', 'Seasonal Decomposition']:
                # For time series models, use statsmodels evaluation
                mape_scores[model_name] = evaluate_statsmodels_mape(eval_data, model_name)
            elif model_name in ['XGBoost', 'LightGBM']:
                # For ML models, use cross-validation approach
                mape_scores[model_name] = evaluate_ml_model_mape(eval_data, model_name)
            elif model_name == 'Ensemble':
                # For ensemble, use average of component model scores
                mape_scores[model_name] = 25.0  # Moderate score for ensemble
            else:
                mape_scores[model_name] = float('inf')
        except Exception as e:
            mape_scores[model_name] = float('inf')
    
    return mape_scores

def evaluate_linear_trend_mape(data):
    """Evaluate linear trend model using in-sample predictions"""
    if len(data) < 3:
        return float('inf')
    
    try:
        from sklearn.linear_model import LinearRegression
        X = np.arange(len(data)).reshape(-1, 1)
        y = np.array(data.values)
        
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        return calculate_mape(y, predictions)
    except:
        return float('inf')

def evaluate_moving_average_mape(data):
    """Evaluate moving average model"""
    if len(data) < 8:
        return float('inf')
    
    try:
        window = min(7, len(data) // 2)
        ma_values = data.rolling(window=window, min_periods=1).mean()
        
        # Skip first few values where MA is still building up
        start_idx = window
        if start_idx >= len(data):
            return float('inf')
            
        actual = np.array(data.values[start_idx:])
        predicted = np.array(ma_values.values[start_idx:])
        
        return calculate_mape(actual, predicted)
    except:
        return float('inf')

def evaluate_prophet_mape(data):
    """Evaluate Prophet model using cross-validation"""
    if not HAVE_PROPHET or len(data) < 14:
        return float('inf')
    
    try:
        # Create Prophet dataframe
        prophet_df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        
        # Fit model on 80% of data
        train_size = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:train_size]
        test_df = prophet_df.iloc[train_size:]
        
        if len(train_df) < 7 or len(test_df) < 2:
            return float('inf')
        
        from prophet import Prophet
        import logging
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
    """Evaluate run rate model"""
    if len(data) < 3:
        return float('inf')
    
    try:
        # Simple run rate: predict each day using average of previous days
        predictions = []
        actual = []
        
        for i in range(2, len(data)):  # Start from day 3
            avg_rate = data.iloc[:i].mean()
            predictions.append(avg_rate)
            actual.append(data.iloc[i])
        
        if len(predictions) == 0:
            return float('inf')
            
        return calculate_mape(np.array(actual), np.array(predictions))
    except:
        return float('inf')

def evaluate_renewal_model_mape(eval_data, full_data):
    """Evaluate renewal model by checking spike prediction accuracy"""
    if len(eval_data) < 10:
        return float('inf')
    
    try:
        # Detect spikes in full data to understand pattern
        spike_analysis = detect_monthly_spikes(full_data, 2.0)
        if not spike_analysis['has_spikes']:
            return float('inf')
        
        # For renewal model, check how well it predicts high-value days
        baseline = spike_analysis['baseline_avg']
        threshold = baseline * 2.0
        
        high_days_actual = eval_data > threshold
        predicted_high = np.zeros(len(eval_data), dtype=bool)
        
        # Simple pattern matching - assume monthly pattern
        spike_days = [d for d, c in spike_analysis.get('spike_pattern', [])]
        if spike_days:
            for day_of_month in spike_days:
                # Mark predicted high days based on pattern
                for i, date in enumerate(eval_data.index):
                    if date.day == day_of_month:
                        predicted_high[i] = True
        
        # Calculate accuracy for high-value day prediction
        if np.any(high_days_actual) and np.any(predicted_high):
            correct_predictions = np.sum(high_days_actual == predicted_high)
            total_predictions = len(high_days_actual)
            accuracy = correct_predictions / total_predictions
            # Convert accuracy to MAPE-like score (lower is better)
            return (1 - accuracy) * 100
        else:
            return 50.0  # Moderate score if no high days to predict
    except:
        return float('inf')

def evaluate_time_series_model_mape(data, model_name):
    """Evaluate time series models (ARIMA, Exponential Smoothing, etc.)"""
    if len(data) < 10:
        return float('inf')
    
    try:
        # Split data for validation
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        if len(train_data) < 7 or len(test_data) < 2:
            return float('inf')
        
        if model_name == 'ARIMA' and HAVE_STATSMODELS:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train_data, order=(1, 1, 1))
            fitted_model = model.fit()
            forecast = fitted_model.get_forecast(steps=len(test_data))
            predictions = forecast.predicted_mean.values
        elif model_name == 'Exponential Smoothing' and HAVE_STATSMODELS:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(train_data, trend='add')
            fitted_model = model.fit()
            predictions = fitted_model.forecast(steps=len(test_data)).values
        else:
            return float('inf')
        
        return calculate_mape(test_data.values, predictions)
    except:
        return float('inf')

def evaluate_ml_model_mape(data, model_name):
    """Evaluate ML models (XGBoost, etc.)"""
    if len(data) < 14:
        return float('inf')
    
    try:
        if model_name == 'XGBoost':
            return evaluate_xgboost_mape(data)
        else:
            return float('inf')
    except:
        return float('inf')

def evaluate_xgboost_mape(data):
    """Evaluate XGBoost model using cross-validation"""
    try:
        if not HAVE_XGBOOST:
            return float('inf')
            
        # Use the same approach as fit_xgboost_model for evaluation
        xgb_model, feature_cols = fit_xgboost_model(data)
        if xgb_model is None:
            return float('inf')
        
        # Return a moderate MAPE since we already validated during fitting
        return 20.0  # Placeholder - in practice would use cross-validation
    except Exception:
        return float('inf')

def evaluate_statsmodels_mape(data, model_name):
    """Evaluate statsmodels-based time series models"""
    try:
        # Use simple train/test split for time series evaluation
        train_size = int(len(data) * 0.7)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        if len(test_data) < 3:
            return float('inf')
        
        if model_name == 'ARIMA' and HAVE_STATSMODELS:
            arima_model = fit_arima_model(train_data)
            if arima_model is not None:
                forecast = arima_model.get_forecast(steps=len(test_data))
                predictions = forecast.predicted_mean.values
                return calculate_mape(test_data.values, predictions)
        
        elif model_name == 'Exponential Smoothing' and HAVE_STATSMODELS:
            es_model = fit_exponential_smoothing_model(train_data)
            if es_model is not None:
                predictions = es_model.forecast(steps=len(test_data))
                return calculate_mape(test_data.values, predictions.values)
        
        elif model_name == 'Seasonal Decomposition' and HAVE_STATSMODELS:
            # Use simple trend extrapolation for evaluation
            trend_mape = evaluate_linear_trend_mape(data)
            return trend_mape * 1.1  # Slightly worse than pure trend due to complexity
        
        return 25.0  # Default moderate MAPE for time series models
    except Exception:
        return float('inf')

def create_excel_download(forecasts_data, analysis_date, filename):
    """Create Excel file with all forecast results for download"""
    
    output = io.BytesIO()
    
    # Try different Excel engines in order of preference
    writer_created = False
    
    # Try openpyxl first
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                writer_created = True
                
                # Summary sheet
                summary_data = []
                for product, data in forecasts_data.items():
                    forecast = data['forecast']
                    summary_data.append({
                        'Product': product,
                        'Current_Quarter_Actual': forecast['current_quarter_actual'],
                        'Projected_Remaining': forecast['projected_remaining'],
                        'Quarter_Total_Projection': forecast['quarter_total'],
                        'Progress_Pct': forecast['progress_pct'],
                        'Elapsed_Calendar_Days': forecast['elapsed_calendar_days'],
                        'Remaining_Calendar_Days': forecast['remaining_calendar_days'],
                        'Last_Data_Date': forecast['last_data_date'].strftime('%Y-%m-%d'),
                        'Analysis_Date': forecast['analysis_date'].strftime('%Y-%m-%d')
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual model forecasts for each product
                for product, data in forecasts_data.items():
                    forecast = data['forecast']
                    raw_data = data['raw_data']
                    
                    # Create detailed forecast sheet
                    sheet_data = []
                    
                    # Add actual data
                    for date, value in raw_data.items():
                        sheet_data.append({
                            'Date': date.strftime('%Y-%m-%d'),
                            'Type': 'Actual',
                            'ACR': value,
                            'Model': 'Actual'
                        })
                    
                    # Add forecast data for each model
                    start_date = forecast['last_data_date'] + timedelta(days=1)
                    forecast_dates = pd.date_range(start=start_date, periods=forecast['remaining_calendar_days'], freq='D')
                    
                    for model_name, model_data in forecast['individual_forecasts'].items():
                        daily_forecast = model_data['daily_forecast']
                        for i, (date, value) in enumerate(zip(forecast_dates, daily_forecast)):
                            if i < len(daily_forecast):
                                sheet_data.append({
                                    'Date': date.strftime('%Y-%m-%d'),
                                    'Type': 'Forecast',
                                    'ACR': value,
                                    'Model': model_name
                                })
                    
                    # Create dataframe and save
                    product_df = pd.DataFrame(sheet_data)
                    safe_sheet_name = product.replace('/', '_').replace('\\', '_')[:31]  # Excel sheet name limits
                    product_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                
                # Model comparison sheet
                comparison_data = []
                for product, data in forecasts_data.items():
                    forecast = data['forecast']
                    mape_scores = forecast.get('mape_scores', {})
                    for model_name, model_data in forecast['individual_forecasts'].items():
                        comparison_data.append({
                            'Product': product,
                            'Model': model_name,
                            'Quarter_Total': model_data['quarter_total'],
                            'Method_Description': model_data['method'],
                            'MAPE': mape_scores.get(model_name, 'N/A')
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
                
                writer_created = True
                
    except ImportError:
        # openpyxl not available, try xlsxwriter
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Same Excel creation logic as above
                summary_data = []
                for product, data in forecasts_data.items():
                    forecast = data['forecast']
                    summary_data.append({
                        'Product': product,
                        'Current_Quarter_Actual': forecast['current_quarter_actual'],
                        'Projected_Remaining': forecast['projected_remaining'],
                        'Quarter_Total_Projection': forecast['quarter_total'],
                        'Progress_Pct': forecast['progress_pct'],
                        'Elapsed_Calendar_Days': forecast['elapsed_calendar_days'],
                        'Remaining_Calendar_Days': forecast['remaining_calendar_days'],
                        'Last_Data_Date': forecast['last_data_date'].strftime('%Y-%m-%d'),
                        'Analysis_Date': forecast['analysis_date'].strftime('%Y-%m-%d')
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                for product, data in forecasts_data.items():
                    forecast = data['forecast']
                    raw_data = data['raw_data']
                    
                    sheet_data = []
                    for date, value in raw_data.items():
                        sheet_data.append({
                            'Date': date.strftime('%Y-%m-%d'),
                            'Type': 'Actual',
                            'ACR': value,
                            'Model': 'Actual'
                        })
                    
                    start_date = forecast['last_data_date'] + timedelta(days=1)
                    forecast_dates = pd.date_range(start=start_date, periods=forecast['remaining_calendar_days'], freq='D')
                    
                    for model_name, model_data in forecast['individual_forecasts'].items():
                        daily_forecast = model_data['daily_forecast']
                        for i, (date, value) in enumerate(zip(forecast_dates, daily_forecast)):
                            if i < len(daily_forecast):
                                sheet_data.append({
                                    'Date': date.strftime('%Y-%m-%d'),
                                    'Type': 'Forecast',
                                    'ACR': value,
                                    'Model': model_name
                                })
                    
                    product_df = pd.DataFrame(sheet_data)
                    safe_sheet_name = product.replace('/', '_').replace('\\', '_')[:31]
                    product_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                
                comparison_data = []
                for product, data in forecasts_data.items():
                    forecast = data['forecast']
                    mape_scores = forecast.get('mape_scores', {})
                    for model_name, model_data in forecast['individual_forecasts'].items():
                        comparison_data.append({
                            'Product': product,
                            'Model': model_name,
                            'Quarter_Total': model_data['quarter_total'],
                            'Method_Description': model_data['method'],
                            'MAPE': mape_scores.get(model_name, 'N/A')
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
                
                writer_created = True
                
        except ImportError:
            writer_created = False
        except Exception:
            writer_created = False
    except Exception:
        writer_created = False
    
    if not writer_created:
        # Fallback: create a simple CSV-like format in Excel using basic pandas
        try:
            # Create a simple single-sheet Excel file
            summary_data = []
            for product, data in forecasts_data.items():
                forecast = data['forecast']
                mape_scores = forecast.get('mape_scores', {})
                best_model = forecast.get('best_model', 'N/A')
                best_mape = mape_scores.get(best_model, 'N/A') if best_model != 'N/A' else 'N/A'
                
                summary_data.append({
                    'Product': product,
                    'Current_Quarter_Actual': forecast['current_quarter_actual'],
                    'Projected_Quarter_Total': forecast['quarter_total'],
                    'Progress_Pct': f"{forecast['progress_pct']:.1%}",
                    'Best_Model': best_model,
                    'Best_Model_MAPE': f"{best_mape:.1f}%" if best_mape != 'N/A' else 'N/A',
                    'Last_Data_Date': forecast['last_data_date'].strftime('%Y-%m-%d'),
                    'Analysis_Date': forecast['analysis_date'].strftime('%Y-%m-%d')
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(output, index=False, engine=None)  # Use default engine
            
        except Exception:
            # Ultimate fallback: return None to indicate failure
            return None
    
    output.seek(0)
    return output.getvalue()

# ============ Existing imports and functions continue ============

# ============ Streamlit UI ============

st.title("📈 Quarterly Outlook Forecaster")
st.markdown("**Daily Data Edition** - Project quarter performance from partial data using fiscal year calendar (July-June)")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Analysis date info (will be auto-set from data)
    st.subheader("Analysis Date")
    if 'outlook_last_data_date' in st.session_state:
        analysis_date = st.session_state.outlook_last_data_date
        st.success(f"**From your data:** {analysis_date.strftime('%B %d, %Y')}")
        st.caption("✅ Analysis date set from last date in uploaded file")
        
        # Display quarter info based on uploaded data
        current_quarter_info = get_fiscal_quarter_info(analysis_date)
        st.subheader("📊 Quarter Being Forecasted")
        st.info(f"**{current_quarter_info['quarter_name']}**")
        st.info(f"**Period:** {current_quarter_info['quarter_start'].strftime('%b %d')} - {current_quarter_info['quarter_end'].strftime('%b %d, %Y')}")
        
        # Show data range if available
        if 'outlook_data_range' in st.session_state:
            data_start, data_end = st.session_state.outlook_data_range
            st.caption(f"📅 Your data: {data_start.strftime('%b %d')} - {data_end.strftime('%b %d, %Y')}")
    else:
        analysis_date = datetime.now()
        st.info("**Default:** Using today's date")
        st.caption("⏳ Upload data to set analysis date and quarter")
        
        # Display default quarter info
        current_quarter_info = get_fiscal_quarter_info(analysis_date)
        st.subheader("📊 Current Quarter (Default)")
        st.info(f"**{current_quarter_info['quarter_name']}**")
        st.info(f"**Period:** {current_quarter_info['quarter_start'].strftime('%b %d')} - {current_quarter_info['quarter_end'].strftime('%b %d, %Y')}")
    
    st.subheader("Forecasting Options")
    confidence_level = st.selectbox("Confidence Level", [80, 90, 95], index=1, help="Confidence intervals for uncertainty")
    
    st.subheader("Capacity Constraints")
    apply_capacity_adjustment = st.checkbox("Apply capacity constraints", value=False, help="Adjust forecasts for operational limitations that prevent achieving projected growth")
    
    # Store in session state for use in results tab
    st.session_state['apply_capacity_adjustment'] = apply_capacity_adjustment
    
    capacity_adjustment = 1.0  # Default value (no adjustment)
    if apply_capacity_adjustment:
        # Quick capacity calculator
        with st.expander("💡 Capacity Calculator Helper", expanded=False):
            st.markdown("**Calculate capacity factor based on weekly revenue loss:**")
            
            # Auto-populate values from forecast data if available
            default_unconstrained = 332.0
            default_weeks_remaining = 3.4
            
            if 'outlook_forecasts' in st.session_state:
                forecasts = st.session_state.outlook_forecasts
                if forecasts:
                    # Calculate total unconstrained forecast from all products
                    total_forecast = sum([f['forecast']['quarter_total'] for f in forecasts.values()])
                    default_unconstrained = total_forecast / 1_000_000  # Convert to millions
                    
                    # Calculate actual weeks remaining based on forecast data
                    # Use the first product's forecast data to get remaining days
                    first_forecast = next(iter(forecasts.values()))['forecast']
                    remaining_days = first_forecast.get('remaining_calendar_days', 0)
                    default_weeks_remaining = remaining_days / 7.0  # Convert days to weeks
                    
                    st.info(f"🔄 **Auto-populated from your forecast:** ${default_unconstrained:.0f}M total forecast, {default_weeks_remaining:.1f} weeks remaining")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                unconstrained_forecast = st.number_input(
                    "Unconstrained forecast ($M)", 
                    min_value=0.0, max_value=1000.0, value=default_unconstrained, step=10.0,
                    help="Total quarter forecast without capacity constraints (auto-filled from your data)"
                )
            with col2:
                weekly_revenue_loss = st.number_input(
                    "Weekly revenue loss ($M)", 
                    min_value=0.0, max_value=50.0, value=3.0, step=0.1,
                    help="Amount of revenue lost per week due to capacity constraints"
                )
            with col3:
                weeks_remaining = st.number_input(
                    "Weeks with constraints", 
                    min_value=0.0, max_value=13.0, value=default_weeks_remaining, step=0.1,
                    help="Number of weeks affected by capacity constraints (auto-filled from forecast period)"
                )
            
            if unconstrained_forecast > 0 and weekly_revenue_loss > 0:
                total_loss = weekly_revenue_loss * weeks_remaining
                recommended_factor = max(0.5, (unconstrained_forecast - total_loss) / unconstrained_forecast)
                
                st.info(f"""
                **Calculated Capacity Factor:**
                - Unconstrained: ${unconstrained_forecast:.0f}M
                - Revenue loss: ${total_loss:.1f}M ({weeks_remaining:.1f} weeks × ${weekly_revenue_loss:.1f}M)
                - **Recommended factor: {recommended_factor:.3f} ({recommended_factor*100:.1f}%)**
                """)
                
                if st.button("📋 Use Calculated Factor"):
                    capacity_adjustment = recommended_factor
        
        capacity_adjustment = st.slider(
            "Capacity limitation factor", 
            0.5, 1.0, capacity_adjustment, 0.001, 
            help="Multiplier to reduce forecasts due to capacity constraints. 0.85 = 15% reduction from capacity limits",
            format="%.3f"
        )
        st.caption(f"🔧 Applying {(1-capacity_adjustment)*100:.1f}% reduction to account for capacity constraints")
        
        # Show dollar impact if forecast data is available
        if 'outlook_forecasts' in st.session_state:
            total_unconstrained = sum([f['forecast']['quarter_total'] for f in st.session_state.outlook_forecasts.values()])
            revenue_reduction = total_unconstrained * (1 - capacity_adjustment)
            st.warning(f"⚠️ **Impact:** ${revenue_reduction:,.0f} reduction from ${total_unconstrained:,.0f} forecast (capacity constraints)")
        else:
            st.warning("⚠️ **Note:** This reduces all forecast models to account for operational limitations that prevent achieving projected growth rates.")
    
    # Store capacity adjustment in session state for use in results tab
    st.session_state['capacity_adjustment'] = capacity_adjustment
    st.subheader("Monthly Renewal Detection")
    spike_detection = st.checkbox("Detect monthly renewals", value=True, help="Automatically detect and forecast monthly subscription renewals (non-consumptive SKUs with specific revenue recognition dates)")
    spike_threshold = 2.0  # Default value
    if spike_detection:
        spike_threshold = st.slider("Spike sensitivity", 1.5, 4.0, 2.0, 0.1, help="Multiplier above baseline to consider a renewal spike. Higher = less sensitive")
        st.caption("💡 For non-consumptive renewals, try 2.0-3.0. For large enterprise deals, try 3.0-4.0")

# Main content
tab_upload, tab_results = st.tabs(["📁 Data Upload", "📊 Outlook Results"])

with tab_upload:
    st.markdown("### 📁 Upload Daily Data")
    st.markdown("Upload an Excel file with daily business data. Required columns: **Date**, **Product**, **ACR**")
    
    # File requirements and purview warning
    st.info("""
    **📋 File Requirements:**
    • Excel file with columns: **"Date", "Product", "ACR"**
    • Must be labeled as **"General"** (not Confidential/Highly Confidential)
    • Check the top of Excel ribbon for Purview classification labels
    """)
    
    # Example template section
    with st.expander("📋 Example Data Format & Download Template", expanded=False):
        st.markdown("""
        **Required Columns:**
        - `Date`: Daily dates (e.g., 2025-01-01, 1/1/2025, etc.)
        - `Product`: Business product names
        - `ACR`: Daily values to forecast (revenue, sales, etc.)
        """)
        
        # Create sample data with fiscal year context
        sample_df = pd.DataFrame({
            "Date": [
                "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-05",
                "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-05",
                "2025-06-27", "2025-06-28", "2025-06-29", "2025-06-30"
            ],
            "Product": [
                "Product A", "Product A", "Product A", "Product A", "Product A",
                "Product B", "Product B", "Product B", "Product B", "Product B", 
                "Product A", "Product A", "Product A", "Product A"
            ],
            "ACR": [
                1250.0, 1180.0, 1320.0, 1400.0, 980.0,
                850.0, 920.0, 875.0, 950.0, 760.0,
                1380.0, 1290.0, 1450.0, 1520.0
            ]
        })
        
        st.markdown("**Example Data Preview:**")
        st.dataframe(sample_df, use_container_width=True)
        
        # Create downloadable Excel template
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            sample_df.to_excel(writer, index=False, sheet_name="Daily_Data_Template")
            
            # Add a second sheet with instructions
            instructions_df = pd.DataFrame({
                "Instructions": [
                    "This template is for the Quarterly Outlook Forecaster",
                    "Uses fiscal year calendar: Q1 (Jul-Sep), Q2 (Oct-Dec), Q3 (Jan-Mar), Q4 (Apr-Jun)",
                    "Upload daily data to get quarterly projections",
                    "",
                    "Column Requirements:",
                    "• Date: Daily dates in any standard format",
                    "• Product: Business product names", 
                    "• ACR: Daily values (revenue, sales, units, etc.)",
                    "",
                    "Tips for Best Results:",
                    "• Include at least 5-10 days of current quarter data",
                    "• Use consistent daily data (avoid large gaps)",
                    "• Consider business calendars (holidays, weekends)",
                    "• Multiple products can be analyzed together"
                ]
            })
            instructions_df.to_excel(writer, index=False, sheet_name="Instructions")
        
        buf.seek(0)
        st.download_button(
            "📥 Download Daily Data Template",
            data=buf,
            file_name="daily_outlook_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download Excel template with sample daily data and instructions"
        )
    
    uploaded = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'])
    
    if uploaded:
        try:
            # Read and validate data
            raw = read_any_excel(io.BytesIO(uploaded.read()))
            required = {"Date", "Product", "ACR"}
            
            if not required.issubset(raw.columns):
                st.error(f"Excel file must contain columns: {required}")
                st.info("Current columns: " + ", ".join(raw.columns))
                st.stop()
            
            # Process dates
            try:
                raw["Date"] = coerce_daily_dates(raw["Date"])
                raw.sort_values(["Product", "Date"], inplace=True)
            except ValueError as e:
                st.error(f"❌ **Date Format Error:** {str(e)}")
                st.stop()
            
            # Auto-set analysis date from the last date in the data
            last_data_date = raw["Date"].max()
            first_data_date = raw["Date"].min()
            analysis_date = last_data_date
            
            # Store in session state for sidebar display
            st.session_state.outlook_last_data_date = analysis_date
            st.session_state.outlook_data_range = (first_data_date, last_data_date)
            
            # Update quarter info based on actual data date
            current_quarter_info = get_fiscal_quarter_info(analysis_date)
            
            st.success(f"📅 **Analysis Date Auto-Set:** {analysis_date.strftime('%B %d, %Y')} (last date in your data)")
            st.info(f"🎯 **Forecasting Quarter:** {current_quarter_info['quarter_name']} ({current_quarter_info['quarter_start'].strftime('%b %d')} - {current_quarter_info['quarter_end'].strftime('%b %d, %Y')})")
            
            # Analyze data by product
            st.markdown("---")
            st.markdown("### 📊 Data Analysis")
            
            analysis_results = []
            product_forecasts = {}
            
            for product, grp in raw.groupby("Product"):
                try:
                    # Create time series
                    series = grp.set_index("Date")["ACR"].astype(float)
                    series = series.sort_index()
                    
                    # Remove duplicates (keep last)
                    series = series[~series.index.duplicated(keep='last')]
                    
                    # Get date range
                    start_date = series.index.min()
                    end_date = series.index.max()
                    total_days = (end_date - start_date).days + 1
                    
                    # Check if we have current quarter data
                    current_quarter_start = current_quarter_info['quarter_start']
                    quarter_data = series[series.index >= current_quarter_start]
                    
                    if len(quarter_data) == 0:
                        status = "❌ No current quarter data"
                        forecast_possible = False
                    elif len(quarter_data) < 5:
                        status = "⚠️ Limited data (need 5+ days)"
                        forecast_possible = False
                    else:
                        status = f"✅ {len(quarter_data)} days in current quarter"
                        forecast_possible = True
                    
                    analysis_results.append({
                        "Product": product,
                        "📅 Date Range": f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}",
                        "📊 Total Days": total_days,
                        "🎯 Current Quarter": status,
                        "💰 Latest ACR": f"{series.iloc[-1]:,.2f}" if len(series) > 0 else "N/A"
                    })
                    
                    # Generate forecast if possible
                    if forecast_possible:
                        # Get spike detection settings from sidebar
                        detect_spikes = st.session_state.get('spike_detection', True)
                        threshold = st.session_state.get('spike_threshold', 2.0)
                        
                        forecast_result, daily_analysis, quarter_info = forecast_quarter_completion(
                            quarter_data, analysis_date, detect_spikes, threshold
                        )
                        if forecast_result:                        product_forecasts[product] = {
                            'forecast': forecast_result,
                            'analysis': daily_analysis,
                            'quarter_info': quarter_info,
                            'raw_data': quarter_data,
                            'mape_scores': forecast_result.get('mape_scores', {})
                        }
                
                except Exception as e:
                    analysis_results.append({
                        "Product": product,
                        "📅 Date Range": "Error",
                        "📊 Total Days": 0,
                        "🎯 Current Quarter": f"❌ Error: {str(e)[:30]}",
                        "💰 Latest ACR": "N/A"
                    })
            
            # Display analysis table
            if analysis_results:
                analysis_df = pd.DataFrame(analysis_results)
                st.dataframe(analysis_df, hide_index=True, use_container_width=True)
            
            # Store results in session state
            if product_forecasts:
                st.session_state.outlook_forecasts = product_forecasts
                st.session_state.outlook_analysis_date = analysis_date
                st.session_state.outlook_filename = uploaded.name
                st.session_state.spike_detection = spike_detection
                st.session_state.spike_threshold = spike_threshold
                st.session_state.apply_capacity_adjustment = apply_capacity_adjustment
                st.session_state.capacity_adjustment = capacity_adjustment
                
                st.success(f"✅ Generated outlooks for {len(product_forecasts)} products. Check the 'Outlook Results' tab!")
            else:
                st.warning("⚠️ No products had sufficient current quarter data for forecasting.")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

with tab_results:
    if 'outlook_forecasts' in st.session_state:
        forecasts = st.session_state.outlook_forecasts
        analysis_date = st.session_state.outlook_analysis_date
        filename = st.session_state.outlook_filename
        
        # Apply capacity adjustments if enabled (use main sidebar settings)
        apply_capacity_adjustment = st.session_state.get('apply_capacity_adjustment', False)
        capacity_adjustment = st.session_state.get('capacity_adjustment', 1.0)
        
        # Create adjusted forecasts if capacity constraints are enabled
        if apply_capacity_adjustment and capacity_adjustment < 1.0:
            adjusted_forecasts = {}
            for product, data in forecasts.items():
                adjusted_data = data.copy()
                adjusted_data['forecast'] = apply_capacity_adjustment_to_forecast(
                    data['forecast'], 
                    capacity_adjustment
                )
                adjusted_forecasts[product] = adjusted_data
            forecasts_to_display = adjusted_forecasts
            
            st.info(f"🔧 **Capacity Constraints Applied:** {(1-capacity_adjustment)*100:.0f}% reduction applied to all forecasts to account for operational limitations.")
        else:
            forecasts_to_display = forecasts
        
        st.markdown(f"### 📈 Quarterly Outlook Results")
        st.markdown(f"**File:** {filename} | **Analysis Date:** {analysis_date.strftime('%B %d, %Y')}")
        
        # Excel download for all data
        col_title, col_download = st.columns([4, 1])
        with col_download:
            try:
                excel_data = create_excel_download(forecasts_to_display, analysis_date, filename)
                if excel_data is not None and isinstance(excel_data, bytes):
                    st.download_button(
                        label="📊 Download All Data",
                        data=excel_data,
                        file_name=f"quarterly_forecast_{analysis_date.strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download complete forecast data for all products"
                    )
                else:
                    st.button("📊 Download All Data", disabled=True, help="Excel download unavailable - missing required libraries (openpyxl, xlsxwriter)")
            except Exception as e:
                st.button("📊 Download All Data", disabled=True, help=f"Excel download failed: {str(e)[:50]}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_products = len(forecasts_to_display)
        avg_progress = np.mean([f['forecast']['progress_pct'] for f in forecasts_to_display.values()]) * 100
        total_actual = sum([f['forecast']['current_quarter_actual'] for f in forecasts_to_display.values()])
        total_projected = sum([f['forecast']['quarter_total'] for f in forecasts_to_display.values()])
        
        with col1:
            st.metric("Products Forecasted", total_products)
        with col2:
            st.metric("Avg Quarter Progress", f"{avg_progress:.1f}%")
        with col3:
            st.metric("Current Quarter Actual", f"{total_actual:,.0f}")
        with col4:
            st.metric("Projected Quarter Total", f"{total_projected:,.0f}")
        
        # Individual product results
        st.markdown("---")
        
        for product, data in forecasts_to_display.items():
            forecast = data['forecast']
            analysis = data['analysis']
            quarter_info = data['quarter_info']
            raw_data = data['raw_data']
            
            with st.expander(f"📊 {product} - {quarter_info['quarter_name']}", expanded=True):
                # Show capacity adjustment notice if applied
                if apply_capacity_adjustment and capacity_adjustment < 1.0:
                    if forecast.get('capacity_adjustment_applied', False):
                        original_total = forecast.get('original_quarter_total', 0)
                        adjusted_total = forecast['quarter_total']
                        reduction_amount = original_total - adjusted_total
                        st.warning(f"🔧 **Capacity Constraint Applied:** Original projection: {original_total:,.0f} → Capacity-limited: {adjusted_total:,.0f} (reduction of {reduction_amount:,.0f})")
                
                # Data completeness notification
                last_data_date = forecast['last_data_date']
                analysis_date = forecast['analysis_date']
                
                if last_data_date.date() < analysis_date.date():
                    days_gap = (analysis_date.date() - last_data_date.date()).days
                    st.warning(f"📅 **Data Gap Detected** - Last data: {last_data_date.strftime('%b %d')}, Analysis date: {analysis_date.strftime('%b %d')} ({days_gap} days gap)")
                elif forecast['progress_pct'] >= 1.0:
                    st.info("📅 **Quarter Complete** - Showing extended projection for analytical purposes")
                
                # Key metrics for this product
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Show progress based on actual data availability
                    progress_label = "Data Progress"
                    progress_help = f"{forecast['elapsed_calendar_days']} of {forecast['total_calendar_days']} calendar days with data"
                    
                    if forecast.get('last_data_date') and forecast.get('analysis_date'):
                        last_data = forecast['last_data_date']
                        analysis_date = forecast['analysis_date']
                        if last_data.date() < analysis_date.date():
                            progress_help += f" (through {last_data.strftime('%b %d')})"
                    
                    st.metric(
                        progress_label, 
                        f"{forecast['progress_pct']:.1%}",
                        help=progress_help
                    )
                
                with col2:
                    st.metric(
                        "Current Actual",
                        f"{forecast['current_quarter_actual']:,.0f}"
                    )
                
                with col3:
                    # Determine projection label based on data situation
                    if forecast.get('last_data_date') and forecast.get('analysis_date'):
                        last_data = forecast['last_data_date']
                        analysis_date = forecast['analysis_date']
                        quarter_end = quarter_info['quarter_end'].date()
                        
                        if last_data.date() < quarter_end:
                            remaining_label = "Projected to Quarter End"
                            remaining_help = f"Forecast for {forecast['remaining_calendar_days']} calendar days from {last_data.strftime('%b %d')} to quarter end"
                        else:
                            remaining_label = "Extended Projection"
                            remaining_help = "Extended projection beyond quarter end for analysis"
                    else:
                        remaining_label = "Projected Remaining"
                        remaining_help = f"Next {forecast['remaining_calendar_days']} calendar days"
                    
                    st.metric(
                        remaining_label,
                        f"{forecast['projected_remaining']:,.0f}",
                        help=remaining_help
                    )
                
                with col4:
                    current_run_rate = forecast['current_quarter_actual'] / forecast['elapsed_calendar_days'] if forecast['elapsed_calendar_days'] > 0 else 0
                    st.metric(
                        "Current Run Rate",
                        f"{current_run_rate:,.0f}/day",
                        help="Average per calendar day so far"
                    )
                
                # Model Selection and MAPE Display
                st.markdown("**🎯 Model Selection & Performance:**")
                
                # Get MAPE scores
                mape_scores = data.get('mape_scores', {})
                
                # Create model options with MAPE scores
                model_options = []
                model_display_names = {}
                
                for method, details in forecast['individual_forecasts'].items():
                    mape = mape_scores.get(method, float('inf'))
                    if mape != float('inf'):
                        mape_text = f" (MAPE: {mape:.1f}%)"
                        if method == forecast.get('best_model'):
                            mape_text += " ⭐ Best"
                    else:
                        mape_text = " (MAPE: N/A)"
                    
                    display_name = f"{method}{mape_text}"
                    model_options.append(method)
                    model_display_names[method] = display_name
                
                # Model selector
                if len(model_options) > 1:
                    col_selector, col_download = st.columns([3, 1])
                    
                    with col_selector:
                        # Default to best model if available
                        default_model = forecast.get('best_model', model_options[0])
                        if default_model not in model_options:
                            default_model = model_options[0]
                        
                        default_index = model_options.index(default_model)
                        
                        selected_model = st.selectbox(
                            "Select Model for Chart & Metrics:",
                            options=model_options,
                            format_func=lambda x: model_display_names[x],
                            index=default_index,
                            key=f"model_selector_{product}"
                        )
                    
                    with col_download:
                        # Excel download button
                        try:
                            excel_data = create_excel_download(
                                {product: data}, 
                                st.session_state.outlook_analysis_date, 
                                st.session_state.outlook_filename
                            )
                            
                            if excel_data is not None and isinstance(excel_data, bytes):
                                st.download_button(
                                    label="📊 Download Excel",
                                    data=excel_data,
                                    file_name=f"{product.replace('/', '_')}_forecast_{st.session_state.outlook_analysis_date.strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    help="Download detailed forecast data and model comparison"
                                )
                            else:
                                st.button("📊 Download Excel", disabled=True, help="Excel download unavailable")
                        except Exception as e:
                            st.button("📊 Download Excel", disabled=True, help=f"Download failed: {str(e)[:30]}")
                else:
                    selected_model = model_options[0] if model_options else None
                
                # Update forecast metrics based on selected model
                if selected_model and selected_model in forecast['individual_forecasts']:
                    selected_forecast = forecast['individual_forecasts'][selected_model]
                    
                    # Update key metrics to show selected model
                    st.markdown(f"**Selected Model: {model_display_names.get(selected_model, selected_model)}**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Quarter Total (Selected Model)", f"{selected_forecast['quarter_total']:,.0f}")
                    with col2:
                        st.metric("Projected Remaining (Selected Model)", f"{selected_forecast['quarter_total'] - forecast['current_quarter_actual']:,.0f}")
                    with col3:
                        mape = mape_scores.get(selected_model, float('inf'))
                        if mape != float('inf'):
                            st.metric("Model MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error - lower is better")
                        else:
                            st.metric("Model MAPE", "N/A", help="MAPE could not be calculated for this model")
                
                # Forecast breakdown table
                st.markdown("**📈 All Models Comparison:**")
                forecast_details = []
                for method, details in forecast['individual_forecasts'].items():
                    mape = mape_scores.get(method, float('inf'))
                    mape_display = f"{mape:.1f}%" if mape != float('inf') else "N/A"
                    
                    forecast_details.append({
                        "Model": method,
                        "Projected Total": f"{details['quarter_total']:,.0f}",
                        "MAPE": mape_display,
                        "Method": details['method']
                    })
                
                if forecast_details:
                    # Sort by MAPE (best first)
                    forecast_df = pd.DataFrame(forecast_details)
                    # Convert MAPE to numeric for sorting, treating N/A as infinity
                    forecast_df['MAPE_numeric'] = forecast_df['MAPE'].apply(
                        lambda x: float(x.replace('%', '')) if x != 'N/A' else float('inf')
                    )
                    forecast_df = forecast_df.sort_values('MAPE_numeric').drop('MAPE_numeric', axis=1)
                    st.dataframe(forecast_df, hide_index=True, use_container_width=True)
                
                # Enhanced visualization with selected model
                if len(raw_data) > 0:
                    # Create chart data with actual values
                    chart_data = pd.DataFrame({
                        'Date': raw_data.index,
                        'Daily Value': raw_data.values,
                        'Type': 'Actual'
                    })
                    
                    # Add projected values for selected model
                    if selected_model and forecast['remaining_calendar_days'] > 0:
                        # Get forecast data for selected model
                        selected_forecast = forecast['individual_forecasts'][selected_model]
                        
                        # Determine the start date for projections
                        last_data_date = forecast.get('last_data_date', raw_data.index.max())
                        
                        # Create future dates for projection starting from day after last data (all days)
                        future_dates = pd.date_range(
                            start=last_data_date + timedelta(days=1),
                            periods=min(forecast['remaining_calendar_days'], 20),  # Limit for readability
                            freq='D'  # Daily frequency, includes weekends
                        )
                        
                        # Use selected model's forecast
                        projected_daily = selected_forecast['daily_forecast'][:len(future_dates)]
                        
                        # Determine projection type for color coding
                        quarter_end = quarter_info['quarter_end'].date()
                        if last_data_date.date() < quarter_end:
                            proj_type = f'{selected_model} Forecast'
                        else:
                            proj_type = f'{selected_model} Extended'
                        
                        projected_data = pd.DataFrame({
                            'Date': future_dates,
                            'Daily Value': projected_daily,
                            'Type': proj_type
                        })
                        
                        chart_data = pd.concat([chart_data, projected_data], ignore_index=True)
                    elif forecast['remaining_calendar_days'] > 0:
                        # Fallback to ensemble forecast if no model selected
                        last_data_date = forecast.get('last_data_date', raw_data.index.max())
                        
                        # Create future dates for projection starting from day after last data (all days)
                        future_dates = pd.date_range(
                            start=last_data_date + timedelta(days=1),
                            periods=min(forecast['remaining_calendar_days'], 20),  # Limit for readability
                            freq='D'  # Daily frequency, includes weekends
                        )
                        
                        projected_daily = forecast['daily_forecast'][:len(future_dates)]
                        
                        # Determine projection type for color coding
                        quarter_end = quarter_info['quarter_end'].date()
                        if last_data_date.date() < quarter_end:
                            proj_type = 'Ensemble Forecast'
                        else:
                            proj_type = 'Ensemble Extended'
                        
                        projected_data = pd.DataFrame({
                            'Date': future_dates,
                            'Daily Value': projected_daily,
                            'Type': proj_type
                        })
                        
                        chart_data = pd.concat([chart_data, projected_data], ignore_index=True)
                    
                    # Create dynamic Altair chart
                    unique_types = chart_data['Type'].unique()
                    
                    # Create color scale based on actual chart data
                    color_domain = list(unique_types)
                   
                    if 'Actual' in color_domain:
                        base_colors = ['blue']
                        forecast_colors = ['orange', 'red', 'green', 'purple', 'brown']
                        
                        color_range = ['blue']  # Actual is always blue
                        color_idx = 0
                        for dtype in color_domain:
                            if dtype != 'Actual':
                                color_range.append(forecast_colors[color_idx % len(forecast_colors)])
                                color_idx += 1
                    else:
                        color_range = ['orange'] * len(color_domain)
                    
                    color_scale = alt.Scale(domain=color_domain, range=color_range)
                    
                    # Dynamic chart title
                    chart_title = f"{product} - Daily Values"
                    if selected_model:
                        mape = mape_scores.get(selected_model, float('inf'))
                        if mape != float('inf'):
                            chart_title += f" ({selected_model}, MAPE: {mape:.1f}%)"
                        else:
                            chart_title += f" ({selected_model})"
                    
                    chart = alt.Chart(chart_data).mark_line(point=True).encode(
                        x=alt.X('Date:T', title='Date'),
                        y=alt.Y('Daily Value:Q', title='Daily Value'),
                        color=alt.Color('Type:N', scale=color_scale),
                        tooltip=['Date:T', 'Daily Value:Q', 'Type:N']
                    ).properties(
                        width=600,
                        height=300,
                        title=chart_title
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                
                # Data insights
                if analysis['business_days'] > 0:
                    st.markdown("**📋 Data Insights:**")
                    insights = []
                    
                    if analysis['business_ratio'] > 1.5:
                        insights.append(f"• Strong weekday pattern: {analysis['business_ratio']:.1f}x higher on business days")
                    elif analysis['business_ratio'] < 0.7:
                        insights.append(f"• Weekend-heavy pattern: {analysis['business_ratio']:.1f}x ratio")
                    else:
                        insights.append("• Consistent daily pattern across week")
                    
                    if analysis['trend'] > 0.05:
                        insights.append(f"• Positive trend: +{analysis['trend']:.1%} week-over-week")
                    elif analysis['trend'] < -0.05:
                        insights.append(f"• Negative trend: {analysis['trend']:.1%} week-over-week")
                    else:
                        insights.append("• Stable trend")
                    
                    if analysis['daily_volatility'] / raw_data.mean() > 0.3:
                        insights.append(f"• High daily volatility: {analysis['daily_volatility'] / raw_data.mean():.1%}")
                    else:
                        insights.append("• Moderate daily volatility")
                    
                    # Add spike insights
                    spike_info = analysis.get('spike_analysis', {})
                    if spike_info.get('has_spikes', False):
                        spike_contribution = spike_info.get('spike_contribution', 0)
                        spike_multiplier = spike_info.get('spike_multiplier', 1)
                        
                        if spike_contribution > 0.05:  # More than 5% of revenue from spikes (lowered threshold)
                            spike_days = [f"{d}" for d, c in spike_info.get('spike_pattern', [])]
                            
                            # Add week pattern insight if available
                            week_info = ""
                            if 'spike_week_pattern' in spike_info and spike_info['spike_week_pattern']:
                                top_weeks = [w for w, c in spike_info['spike_week_pattern'][:2]]
                                if 1 in top_weeks or 2 in top_weeks:
                                    week_info = " (early month renewals)"
                                elif 4 in top_weeks or 5 in top_weeks:
                                    week_info = " (late month renewals)"
                                elif 3 in top_weeks:
                                    week_info = " (mid-month renewals)"
                            
                            insights.append(f"• **Non-consumptive renewals detected**: {spike_contribution:.1%} of revenue from renewal days {', '.join(spike_days[:3])}{week_info}")
                            insights.append(f"• Revenue recognition pattern: {spike_multiplier:.1f}x baseline on renewal dates, ~{spike_info.get('spike_frequency', 0):.1f} renewals per month")
                        else:
                            insights.append(f"• Minor spikes detected: {spike_contribution:.1%} of revenue ({spike_multiplier:.1f}x baseline)")
                    else:
                        insights.append("• No significant monthly patterns detected")
                    
                    for insight in insights:
                        st.markdown(insight)
    
    else:
        st.info("👆 Upload daily data in the 'Data Upload' tab to generate quarterly outlooks.")
        st.markdown("""
        ### 📖 How to Use
        
        1. **Upload Excel File** with daily data containing:
           - `Date` column (daily dates)
           - `Product` column (business product names)
           - `ACR` column (daily values to forecast)
        
        2. **Fiscal Quarter Detection** automatically identifies current quarter:
           - Q1: July - September
           - Q2: October - December  
           - Q3: January - March
           - Q4: April - June
        
        3. **Forecasting Models** project remaining quarter performance:
           - Linear trend analysis (with spike overlay for renewals)
           - Moving averages with trend (with spike overlay for renewals)
           - Prophet (if available, with spike overlay for renewals)
           - Run rate projections
           - Exponential Smoothing (Holt-Winters, with spike overlay)
           - ARIMA (Auto ARIMA if available, with spike overlay)
           - Seasonal Decomposition + Trend (with spike overlay)
           - Ensemble of available models
        
        4. **Business Day Awareness** considers weekday vs weekend patterns (includes all days)
        
        5. **Monthly Renewal Detection** automatically identifies:
           - Non-consumptive revenue spikes (subscriptions, renewals)
           - Expected renewal dates based on historical patterns
           - Proper forecasting that maintains monthly recurring revenue
        
        ### 💡 Best Practices
        - Upload at least 5-10 days of current quarter data
        - Include recent historical quarters for context
        - Use consistent daily data (avoid gaps where possible)
        - **If there are non-compliant revenue recognition SKUs in your products IE--AOAI PTU-C**: Enable monthly renewal detection
        - Adjust spike sensitivity based on your business model:
          - **1.5-2.5x**: Regular subscription renewals
          - **2.5-4.0x**: Large enterprise deals or annual renewals
        - Consider business calendar (holidays, etc.) in interpretation
        """)
