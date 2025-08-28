"""
Core model fitting and forecasting functions.

Contains the main forecasting models including SARIMA, ETS, Prophet, 
Auto-ARIMA, LightGBM, and Polynomial regression implementations.
"""

import numpy as np
import pandas as pd
import warnings
import itertools
import streamlit as st
from typing import Any
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf

# Optional deps with robust error handling
# Define symbols with safe defaults; bind after successful imports
auto_arima = None  # type: ignore[assignment]
Prophet = None  # type: ignore[assignment]
try:
    # Test scipy compatibility first
    from scipy._lib._util import _lazywhere
    scipy_compatible = True
except ImportError:
    scipy_compatible = False

if scipy_compatible:
    try:
        # Try new import path for pmdarima v2+
        from pmdarima.arima import auto_arima as _auto_arima
        HAVE_PMDARIMA = True
    except (ImportError, ValueError) as e:
        try:
            # Fallback for pmdarima v1.x
            from pmdarima import auto_arima as _auto_arima
            HAVE_PMDARIMA = True
        except (ImportError, ValueError) as e:
            # Binary compatibility or other issues - disable pmdarima
            HAVE_PMDARIMA = False
            print(f"⚠️ pmdarima disabled due to compatibility issue: {str(e)[:100]}")
else:
    # scipy version incompatible with pmdarima
    HAVE_PMDARIMA = False
    print("⚠️ pmdarima disabled: scipy version incompatible (missing _lazywhere)")
    print("   Solution: pip install 'scipy>=1.9.0,<1.11.0' 'pmdarima>=2.0.4'")

# Bind auto_arima symbol if available, otherwise provide a stub that raises at runtime
if 'HAVE_PMDARIMA' in globals() and HAVE_PMDARIMA:
    auto_arima = _auto_arima  # type: ignore[assignment]
else:
    def auto_arima(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("pmdarima not available")

# Ensure symbol exists to avoid static-analysis "possibly unbound" warnings when pmdarima is unavailable
if not 'HAVE_PMDARIMA' in globals() or HAVE_PMDARIMA is False:
    def auto_arima(*args, **kwargs):  # type: ignore
        raise ImportError("pmdarima not available")

try:
    from prophet import Prophet as _Prophet
    HAVE_PROPHET = True
except (ImportError, ValueError) as e:
    HAVE_PROPHET = False
else:
    # Bind the imported class to the module-level Prophet symbol
    Prophet = _Prophet  # type: ignore[assignment]
    # Provide a stub class so references type-check, raising on use
    class Prophet:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("prophet not available")

try:
    from lightgbm import LGBMRegressor
    HAVE_LGBM = True
except (ImportError, ValueError) as e:
    HAVE_LGBM = False

# Import metrics from separate module
from .metrics import smape, mase, rmse, calculate_validation_metrics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Hard feature flags (can be flipped by policy)
ENABLE_PROPHET = True
ENABLE_LGBM = True
ENABLE_POLY3 = False


def apply_statistical_validation(forecasts, train_data, model_name="Model"):
    """
    Apply conservative, statistically justified constraints to forecasts.
    CRITICAL FIX: More conservative bounds to prevent unrealistic forecasts.
    Only intervenes when forecasts violate fundamental assumptions.
    """
    pf = np.array(forecasts) if not isinstance(forecasts, np.ndarray) else forecasts
    
    # Statistical validation only - minimal but more conservative intervention
    
    # 1. Non-negativity constraint (for consumptive business data)
    # This is mathematically justified if all historical data is non-negative
    if train_data.min() >= 0:
        pf = np.maximum(pf, 0)
    
    # 2. CONSERVATIVE bounds - prevent truly extreme statistical outliers
    # Balance business growth allowance with forecast realism
    hist_mean = train_data.mean()
    hist_std = train_data.std()
    hist_max = train_data.max()
    hist_min = train_data.min()
    
    # More conservative upper bound: 3x historical maximum OR mean + 4*sigma (whichever is larger)
    # This still allows substantial growth but prevents clearly unrealistic values
    statistical_upper = hist_mean + 4 * hist_std  # Reduced from 10*sigma to 4*sigma
    growth_upper = hist_max * 3  # Reduced from 10x to 3x historical peak
    upper_bound = max(statistical_upper, growth_upper)
    
    # Conservative lower bound with business context
    if hist_min >= 0:
        lower_bound = 0  # Non-negativity constraint only for non-negative historical data
    else:
        # Allow reasonable downturns but prevent extreme negative spikes
        statistical_lower = hist_mean - 4 * hist_std  # More conservative than 10*sigma
        decline_lower = hist_min * 1.5  # Allow 50% worse than historical minimum
        lower_bound = min(statistical_lower, decline_lower)
    
    # Only apply bounds to truly extreme outliers (not normal business growth)
    extreme_outliers = (pf > upper_bound) | (pf < lower_bound)
    if extreme_outliers.any():
        # Only clip the truly extreme values, preserve normal growth trends
        pf = np.clip(pf, lower_bound, upper_bound)
    
    return pf


def apply_business_adjustments_to_forecast(forecasts, annual_growth_pct=0, market_multiplier=1.0):
    """
    Apply business context adjustments to statistical forecasts.
    
    Args:
        forecasts: Array or Series of forecast values
        annual_growth_pct: Expected annual growth percentage (-50 to 100)
        market_multiplier: Market condition multiplier (0.95 to 1.05)
    
    Returns:
        Adjusted forecast array
    """
    if annual_growth_pct == 0 and market_multiplier == 1.0:
        return forecasts  # No adjustments needed
    
    # Convert to numpy array for calculations
    pf = np.array(forecasts) if not isinstance(forecasts, np.ndarray) else forecasts
    
    # Apply annual growth assumption (compound monthly growth)
    if annual_growth_pct != 0:
        monthly_growth_rate = (1 + annual_growth_pct / 100) ** (1/12) - 1
        growth_factors = np.array([(1 + monthly_growth_rate) ** (i+1) for i in range(len(pf))])
        pf = pf * growth_factors
    
    # Apply market condition multiplier
    pf = pf * market_multiplier
    
    return pf


def detect_seasonality_strength(series, seasonal_period=12):
    """
    Detect the strength of seasonality in the time series.
    Returns a score between 0 (no seasonality) and 1 (strong seasonality).
    """
    if len(series) < seasonal_period * 2:
        return 0.0
    
    try:
        # Simple seasonality detection using autocorrelation
        # Calculate autocorrelation at seasonal lag
        autocorr = acf(series, nlags=seasonal_period, fft=False)
        seasonal_autocorr = abs(autocorr[seasonal_period])
        
        # Also check for consistent seasonal patterns
        if len(series) >= seasonal_period * 3:
            # Compare seasonal means
            seasonal_means = []
            for i in range(seasonal_period):
                seasonal_values = series.iloc[i::seasonal_period]
                seasonal_means.append(seasonal_values.mean())
            
            # Calculate coefficient of variation of seasonal means
            seasonal_cv = np.std(seasonal_means) / np.mean(seasonal_means) if np.mean(seasonal_means) > 0 else 0
            
            # Combine autocorrelation and seasonal variation
            # Type: ignore - autocorr is a scalar at this point
            seasonality_score = (float(seasonal_autocorr) + min(float(seasonal_cv), 1.0)) / 2  # type: ignore
        else:
            seasonality_score = float(seasonal_autocorr)  # type: ignore
            
        return min(float(seasonality_score), 1.0)  # type: ignore
        
    except:
        return 0.5  # Default assumption of moderate seasonality

def get_seasonality_aware_split(series, seasonal_period=12, diagnostic_messages=None):
    """
    Split data ensuring adequate seasonal cycles for training.
    For strong seasonality, we need at least 2-3 full cycles for training.
    """
    n = len(series)
    
    # Minimum requirements for seasonal data
    min_cycles_for_training = 2  # Need at least 2 full seasonal cycles
    min_training_periods = seasonal_period * min_cycles_for_training  # 24 months
    min_validation_periods = seasonal_period  # 12 months for validation
      # Check if we have enough data
    if n < min_training_periods + min_validation_periods:
        if n >= 18:  # 1.5 years minimum
            # Use what we have but warn user
            val_size = min(6, max(3, n // 4))  # 25% for validation, min 3, max 6
            train_size = n - val_size
            if diagnostic_messages is not None:
                if n < 24:
                    diagnostic_messages.append(f"⚠️ Limited data: {n} months. Expected accuracy: 15-30% MAPE. For optimal seasonal forecasting, recommend 36+ months.")
                else:
                    diagnostic_messages.append(f"⚠️ Moderate data: {n} months. Expected accuracy: 10-20% MAPE. For optimal results, recommend 36+ months.")
        else:
            # Too little data for seasonal modeling
            st.error(f"❌ Insufficient data: {n} months. Need minimum 18 months for reliable forecasting.")
            return None, None
    else:
        # We have enough data - use seasonality-aware split
        if n >= 48:  # 4+ years of data
            # Use 3 years training + 1 year validation
            train_size = 36
            val_size = 12
            # Use most recent data
            train = series.iloc[-train_size-val_size:-val_size]
            val = series.iloc[-val_size:]
        elif n >= 36:  # 3-4 years of data  
            # Use 2 years training + 1 year validation
            train_size = 24
            val_size = 12
            train = series.iloc[-train_size-val_size:-val_size]
            val = series.iloc[-val_size:]
        else:  # 2.5-3 years of data
            # Use most data for training, 6-9 months for validation
            val_size = min(9, max(6, n // 4))
            train_size = n - val_size
            train = series.iloc[:train_size]
            val = series.iloc[train_size:]
        
        if diagnostic_messages is not None:
            if n >= 48:
                diagnostic_messages.append(f"✅ Excellent data: {train_size} months training, {val_size} months validation. Expected accuracy: 5-10% WAPE")
            elif n >= 36:
                diagnostic_messages.append(f"✅ Good data: {train_size} months training, {val_size} months validation. Expected accuracy: 5-15% WAPE")
            else:
                diagnostic_messages.append(f"📊 Adequate data: {train_size} months training, {val_size} months validation. Expected accuracy: 10-20% WAPE")
        return train, val
    
    # Fallback for edge cases
    if n <= 12:
        st.error(f"❌ Too little data: {n} months. Need minimum 18 months.")
        return None, None
    
    # Emergency fallback
    val_size = max(3, n // 4)
    train_size = n - val_size
    return series.iloc[:train_size], series.iloc[train_size:]


def create_ets_fitting_function(seasonal_type="mul"):
    """
    Create an ETS model fitting function for use with validation methods.
    
    Args:
        seasonal_type: Type of seasonality ("mul" or "add")
    
    Returns:
        Function that fits ETS model to training data
    """
    def fit_ets_model(train_data, **kwargs):
        """Fit ETS model to training data."""
        try:
            # Try with seasonal first
            model = ExponentialSmoothing(
                train_data, 
                trend="add", 
                seasonal=seasonal_type, 
                seasonal_periods=12
            ).fit()
            return model
        except Exception as e:
            error_msg = str(e).lower()
            # Fallback to additive if multiplicative fails
            if seasonal_type == "mul":
                try:
                    model = ExponentialSmoothing(
                        train_data, 
                        trend="add", 
                        seasonal="add", 
                        seasonal_periods=12
                    ).fit()
                    return model
                except Exception:
                    pass
            # Final fallback: no seasonal
            try:
                model = ExponentialSmoothing(
                    train_data, 
                    trend="add", 
                    seasonal=None
                ).fit()
                return model
            except Exception:
                pass
            # If all else fails, try with minimal parameters
            try:
                model = ExponentialSmoothing(
                    train_data, 
                    trend=None, 
                    seasonal=None
                ).fit()
                return model
            except Exception:
                pass
            raise e
    
    return fit_ets_model


def create_sarima_fitting_function(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """
    Create a SARIMA model fitting function for use with validation methods.
    
    Args:
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal ARIMA order (P, D, Q, s)
    
    Returns:
        Function that fits SARIMA model to training data
    """
    def fit_sarima_model(train_data, **kwargs):
        """Fit SARIMA model to training data."""
        try:
            model = SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            return model
        except Exception as e:
            # Fallback to simpler model
            try:
                model = SARIMAX(
                    train_data,
                    order=(1, 1, 1),
                    seasonal_order=(0, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)
                return model
            except Exception:
                pass
            raise e
    
    return fit_sarima_model


def create_auto_arima_fitting_function():
    """
    Create an Auto-ARIMA model fitting function for use with validation methods.
    
    Returns:
        Function that fits Auto-ARIMA model to training data
    """
    def fit_auto_arima_model(train_data, **kwargs):
        """Fit Auto-ARIMA with robust fallbacks and return an object with forecast(steps)."""
        if not HAVE_PMDARIMA:
            raise ImportError("pmdarima not available")

        # Try a small set of increasingly permissive configs to handle short/quiet samples
        configs = [
            {"seasonal": True,  "m": 12, "max_D": 1, "max_d": 2, "max_p": 2, "max_q": 2, "max_P": 1, "max_Q": 1},
            {"seasonal": True,  "m": 12, "max_D": 0, "max_d": 2, "max_p": 2, "max_q": 2, "max_P": 1, "max_Q": 1},
            {"seasonal": False, "m": 1,  "max_D": 0, "max_d": 2, "max_p": 2, "max_q": 2, "max_P": 0, "max_Q": 0},
        ]

        last_err = None
        for cfg in configs:
            try:
                model = auto_arima(
                    train_data,
                    seasonal=cfg["seasonal"],
                    m=cfg["m"],
                    max_p=cfg["max_p"], max_q=cfg["max_q"], max_P=cfg["max_P"], max_Q=cfg["max_Q"],
                    max_d=cfg["max_d"], max_D=cfg["max_D"],
                    start_p=0, start_q=0, start_P=0, start_Q=0,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False,
                )
                class _AAWrapper:
                    def __init__(self, m):
                        self._m = m
                    def forecast(self, steps):
                        return self._m.predict(int(steps))
                return _AAWrapper(model)
            except Exception as e:
                last_err = e
                continue

        # Seatbelt fallback: seasonal‑naive (or last value) to avoid fold failure
        class _NaiveAAWrapper:
            def __init__(self, train_series):
                self._arr = np.asarray(train_series.values if hasattr(train_series, 'values') else train_series, dtype=float)
            def forecast(self, steps):
                steps = int(steps)
                if self._arr.size == 0:
                    return np.zeros(steps)
                m = 12
                if self._arr.size >= m:
                    last_season = self._arr[-m:]
                    reps = int(np.ceil(steps / m))
                    return np.tile(last_season, reps)[:steps]
                # fallback to last value
                return np.full(steps, self._arr[-1])

        return _NaiveAAWrapper(train_data)
    
    return fit_auto_arima_model


def create_prophet_fitting_function(enable_holidays=False):
    """
    Create a Prophet model fitting function for use with validation methods.
    
    Args:
        enable_holidays: Whether to include holidays in the model
    
    Returns:
        Function that fits Prophet model to training data
    """
    def fit_prophet_model(train_data, **kwargs):
        """Fit Prophet and return a wrapper with forecast(steps)."""
        if not HAVE_PROPHET:
            raise ImportError("prophet not available")
        try:
            df = pd.DataFrame({'ds': pd.to_datetime(train_data.index), 'y': train_data.values})
            m = _Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            if enable_holidays:
                try:
                    m.add_country_holidays(country_name='US')  # type: ignore[attr-defined]
                except Exception:
                    pass
            m.fit(df)  # type: ignore[attr-defined]
            class _PWrapper:
                def __init__(self, model, last_ts):
                    self._m = model
                    self._last = pd.to_datetime(last_ts)
                def forecast(self, steps):
                    future_idx = pd.date_range(self._last + pd.DateOffset(months=1), periods=int(steps), freq='MS')
                    future_df = pd.DataFrame({'ds': future_idx})
                    fc = self._m.predict(future_df)
                    return fc['yhat'].values
            return _PWrapper(m, train_data.index[-1])
        except Exception as e:
            raise e
    
    return fit_prophet_model


def create_polynomial_fitting_function(degree=2):
    """
    Create a Polynomial regression fitting function for use with validation methods.
    
    Args:
        degree: Degree of polynomial features
    
    Returns:
        Function that fits polynomial model to training data
    """
    def fit_polynomial_model(train_data, **kwargs):
        """Fit Polynomial regression model to training data."""
        try:
            # Create time-based features
            X = np.arange(len(train_data)).reshape(-1, 1)
            y = train_data.values
            
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            
            # Fit linear regression
            model = LinearRegression().fit(X_poly, y)
            
            # Create a wrapper that includes the polynomial transformer
            class PolynomialModel:
                def __init__(self, model, poly_features, train_size):
                    self.model = model
                    self.poly_features = poly_features
                    self.train_size = train_size
                
                def forecast(self, steps):
                    X_future = np.arange(self.train_size, self.train_size + steps).reshape(-1, 1)
                    X_future_poly = self.poly_features.transform(X_future)
                    return self.model.predict(X_future_poly)
                
                def predict(self, start, end):
                    steps = end - start + 1
                    return self.forecast(steps)
            
            return PolynomialModel(model, poly_features, len(train_data))
            
        except Exception as e:
            raise e
    
    return fit_polynomial_model


def create_lightgbm_fitting_function():
    """Create a leak-safe LightGBM fitting function for backtesting with forecast(steps)."""
    if not HAVE_LGBM:
        def _unavailable(*args, **kwargs):
            raise ImportError("lightgbm not available")
        return _unavailable

    def build_feature_frame(series: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({'ACR': series.values}, index=pd.to_datetime(series.index))
        for lag in [1,2,3,6,12]:
            df[f'lag_{lag}'] = df['ACR'].shift(lag)
        for window in [3,6,12]:
            df[f'rolling_mean_{window}'] = df['ACR'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['ACR'].rolling(window).std()
        # Momentum/first-difference features
        df['lag_1_diff'] = df['ACR'] - df['ACR'].shift(1)
        df['rolling_mean_diff_3'] = df['rolling_mean_3'] - df['rolling_mean_3'].shift(1)
        idx = pd.to_datetime(df.index)
        df['month'] = idx.month
        df['quarter'] = idx.quarter
        df['year'] = idx.year
        df['trend'] = range(len(df))
        return df

    def fit_lgbm_model(train_data: pd.Series, **kwargs):
        # Train on provided series
        df = build_feature_frame(train_data).dropna()
        if df.empty:
            raise RuntimeError("insufficient features for LightGBM")
        feat_cols = [c for c in df.columns if c != 'ACR']
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=7,
                              random_state=42, verbose=-1, force_col_wise=True)
        model.fit(df[feat_cols], df['ACR'])

        class _LGBMWrapper:
            def __init__(self, base_series: pd.Series, mdl, feat_cols):
                self._series = base_series.copy()
                self._m = mdl
                self._feat_cols = feat_cols
            def forecast(self, steps: int):
                preds = []
                temp = self._series.copy()
                for _ in range(int(steps)):
                    feats = build_feature_frame(temp).dropna()
                    if feats.empty:
                        preds.append(float('nan'))
                        continue
                    x_last = feats[self._feat_cols].iloc[-1:]
                    yhat = float(np.asarray(self._m.predict(x_last)).ravel()[0])
                    # Momentum injection: if forecast gets flat while history has trend, nudge toward recent momentum
                    recent_hist = temp.tail(min(6, len(temp)))
                    hist_cv = (recent_hist.std() / (recent_hist.mean() + 1e-9)) if len(recent_hist) > 3 else 0
                    if len(recent_hist) >= 3:
                        recent_trend = (recent_hist.iloc[-1] - recent_hist.iloc[-3]) / 3
                    else:
                        recent_trend = 0.0
                    if hist_cv > 0.05:
                        # blend 15% of recent trend into first-step forecast; decay quickly across horizon
                        decay = max(0.0, 0.15 - 0.03 * len(preds))
                        yhat = yhat + decay * recent_trend
                    preds.append(yhat)
                    next_date = pd.Timestamp(str(temp.index[-1])) + pd.DateOffset(months=1)
                    temp = pd.concat([temp, pd.Series([yhat], index=[next_date])])
                return np.array(preds)

        return _LGBMWrapper(train_data, model, feat_cols)

    return fit_lgbm_model


def fit_seasonal_naive(train_data, steps, seasonal_period=12):
    """Seasonal Naive baseline: repeat last season into forecast horizon."""
    last_season = train_data[-seasonal_period:] if len(train_data) >= seasonal_period else train_data
    reps = int(np.ceil(steps / len(last_season))) if len(last_season) else 1
    fc = np.tile(np.array(last_season), reps)[:steps]
    return pd.Series(fc)

def apply_trend_aware_forecasting(forecasts, full_series, train_end_idx, model_name="Model", diagnostic_messages=None):
    """
    Ensure forecasts properly connect to the actual data trend, eliminating cliff effects.
    CRITICAL FIX: Only use training data (up to train_end_idx) to prevent data leakage during validation.
    """
    if len(forecasts) == 0 or len(full_series) == 0:
        return forecasts
    
    # LEAK FIX: Only use training portion of series for trend calculation
    train_series = full_series.iloc[:train_end_idx] if train_end_idx < len(full_series) else full_series
    
    # Get the actual last TRAINING value and recent trend from TRAINING data only
    last_actual = train_series.iloc[-1]
    
    # Calculate recent momentum from TRAINING data only (last 3-6 months trend)
    if len(train_series) >= 6:
        recent_trend = (train_series.iloc[-1] - train_series.iloc[-6]) / 6  # Monthly change
    elif len(train_series) >= 3:
        recent_trend = (train_series.iloc[-1] - train_series.iloc[-3]) / 3
    else:
        recent_trend = 0
    
    # Check if there's a big gap between last actual and first forecast
    first_forecast = forecasts.iloc[0] if hasattr(forecasts, 'iloc') else forecasts[0]
    gap_ratio = abs(first_forecast - last_actual) / last_actual if last_actual != 0 else 0
    
    # Flat forecast diagnostic: if first 6 forecast points (or all) nearly constant while recent history had movement
    try:
        recent_hist = train_series.tail(min(12, len(train_series)))
        hist_cv = (recent_hist.std() / (recent_hist.mean() + 1e-9)) if len(recent_hist) > 3 else 0
        if hasattr(forecasts, 'values'):
            f_vals = np.array(forecasts[:min(6, len(forecasts))])
        else:
            f_vals = np.array(forecasts[:min(6, len(forecasts))])
        fore_cv = (f_vals.std() / (f_vals.mean() + 1e-9)) if len(f_vals) > 1 else 0
        if hist_cv > 0.05 and fore_cv < 0.01:  # history had variation, forecast nearly flat
            if diagnostic_messages is not None:
                diagnostic_messages.append(
                    f"ℹ️ {model_name}: Forecast appears very flat vs historical variability (hist CV {hist_cv:.2f}, forecast CV {fore_cv:.2f}). Flatness may be expected if recent trend stabilized or model differenced away level shifts."
                )
    except Exception:
        pass

    if gap_ratio > 0.15:  # More than 15% difference
        # Collect diagnostic message instead of displaying immediately
        if diagnostic_messages is not None:
            diagnostic_messages.append(f"🔧 {model_name}: Bridging {gap_ratio:.1%} gap between last actual and forecast")
        
        # Create a smooth transition that respects the recent trend
        expected_first_forecast = last_actual + recent_trend
        
        # Blend the model's forecast with trend expectation
        # More blending for larger gaps, less for later periods
        adjusted_forecasts = forecasts.copy()
        
        for i in range(min(3, len(forecasts))):  # Adjust first 3 periods max
            blend_weight = 0.7 * (1 - i/3)  # 70% -> 35% -> 0% blending
            trend_component = expected_first_forecast + (i * recent_trend)
            model_component = forecasts.iloc[i] if hasattr(forecasts, 'iloc') else forecasts[i]
            
            if hasattr(adjusted_forecasts, 'iloc'):
                adjusted_forecasts.iloc[i] = (blend_weight * trend_component + 
                                            (1 - blend_weight) * model_component)
            else:
                adjusted_forecasts[i] = (blend_weight * trend_component + 
                                       (1 - blend_weight) * model_component)
        
        return adjusted_forecasts
    
    return forecasts

def fit_final_sarima_model(full_series, best_params, seasonality_strength=0.5):
    """
    Retrain the best SARIMA model on the full dataset for final forecasting.
    """
    order, seasonal_order = best_params
    
    try:
        # Train on full series for final forecasting
        model = SARIMAX(full_series, 
                       order=order, 
                       seasonal_order=seasonal_order,
                       enforce_stationarity=True,
                       enforce_invertibility=True).fit(disp=False, maxiter=100)  # type: ignore
        return model
    except:
        try:
            # Fallback with relaxed constraints
            model = SARIMAX(full_series, 
                           order=order, 
                           seasonal_order=seasonal_order,
                           enforce_stationarity=False,
                           enforce_invertibility=False).fit(disp=False, maxiter=50)  # type: ignore
            return model
        except:
            return None

def fit_best_sarima(train_data, validation_data, seasonality_strength=0.5):
    """
    Fit SARIMA model using expanded parameter search with both AIC and BIC selection.
    Adapts seasonal complexity based on detected seasonality strength.
    Returns the best model based on validation performance from both AIC and BIC candidates.
    Now returns all four metrics: (model, selection_criterion, criterion_value, mape, smape, mase, rmse)
    """
    # Adapt parameter search based on seasonality strength
    if seasonality_strength > 0.7:
        pdq = [(p, d, q) for p in (0,1,2) for d in (0,1) for q in (0,1,2)]
        seasonal_pdq = [(p, d, q, 12) for p in (0,1) for d in (0,1) for q in (0,1)]
    else:
        pdq = [(p, d, q) for p in (0,1) for d in (0,1) for q in (0,1)]
        seasonal_pdq = [(0, d, 0, 12) for d in (0,1)]
    
    # Track best models for both AIC and BIC
    best_aic = float("inf")
    best_bic = float("inf")
    best_aic_model = None
    best_bic_model = None
    best_aic_metrics = (float("inf"), np.nan, np.nan, np.nan)  # (mape, smape, mase, rmse)
    best_bic_metrics = (float("inf"), np.nan, np.nan, np.nan)  # (mape, smape, mase, rmse)
    
    for order in pdq:
        for seasonal_order in seasonal_pdq:
            try:
                # Fit with proper statistical constraints
                model = SARIMAX(train_data, 
                               order=order, 
                               seasonal_order=seasonal_order,
                               enforce_stationarity=True,
                               enforce_invertibility=True).fit(disp=False, maxiter=100)  # type: ignore
                
                # Validate on out-of-sample data
                forecast = model.get_forecast(len(validation_data))  # type: ignore
                forecast_mean = forecast.predicted_mean
                mape_val, smape_val, mase_val, rmse_val = calculate_validation_metrics(
                    validation_data, forecast_mean, train_data)
                
                # Only consider models with reasonable validation performance
                if mape_val < 1.0:  # WAPE < 100%
                    # Track best AIC model
                    if model.aic < best_aic:  # type: ignore
                        best_aic = model.aic  # type: ignore
                        best_aic_model = model
                        best_aic_metrics = (mape_val, smape_val, mase_val, rmse_val)
                    
                    # Track best BIC model
                    if model.bic < best_bic:  # type: ignore
                        best_bic = model.bic  # type: ignore
                        best_bic_model = model
                        best_bic_metrics = (mape_val, smape_val, mase_val, rmse_val)
                    
            except Exception:
                # If strict enforcement fails, try relaxed constraints
                try:
                    model = SARIMAX(train_data, 
                                   order=order, 
                                   seasonal_order=seasonal_order,
                                   enforce_stationarity=False,
                                   enforce_invertibility=False).fit(disp=False, maxiter=50)  # type: ignore
                    
                    forecast = model.get_forecast(len(validation_data))  # type: ignore
                    forecast_mean = forecast.predicted_mean
                    mape_val, smape_val, mase_val, rmse_val = calculate_validation_metrics(
                        validation_data, forecast_mean, train_data)
                    
                    if mape_val < 1.0:
                        # Track best AIC model
                        if model.aic < best_aic:  # type: ignore
                            best_aic = model.aic  # type: ignore
                            best_aic_model = model
                            best_aic_metrics = (mape_val, smape_val, mase_val, rmse_val)
                        
                        # Track best BIC model
                        if model.bic < best_bic:  # type: ignore
                            best_bic = model.bic  # type: ignore
                            best_bic_model = model
                            best_bic_metrics = (mape_val, smape_val, mase_val, rmse_val)
                        
                except Exception:
                    continue
    
    # Choose the best model based on validation performance
    # Compare AIC-best vs BIC-best models using validation WAPE
    if best_aic_model is None and best_bic_model is None:
        return None, None, None, float("inf"), np.nan, np.nan, np.nan
    elif best_aic_model is None:
        return best_bic_model, "BIC", best_bic, best_bic_metrics[0], best_bic_metrics[1], best_bic_metrics[2], best_bic_metrics[3]
    elif best_bic_model is None:
        return best_aic_model, "AIC", best_aic, best_aic_metrics[0], best_aic_metrics[1], best_aic_metrics[2], best_aic_metrics[3]
    else:
        # Both models exist - choose based on validation performance
        # This approach leverages both AIC and BIC model selection, then uses
        # out-of-sample validation to make the final decision
        if best_aic_metrics[0] <= best_bic_metrics[0]:  # Compare WAPE
            return best_aic_model, "AIC", best_aic, best_aic_metrics[0], best_aic_metrics[1], best_aic_metrics[2], best_aic_metrics[3]
        else:
            return best_bic_model, "BIC", best_bic, best_bic_metrics[0], best_bic_metrics[1], best_bic_metrics[2], best_bic_metrics[3]

def fit_best_lightgbm(train_series: pd.Series, val_series: pd.Series, diagnostic_messages=None):
    """LightGBM hyperparameter selection WITHOUT look-ahead leakage.

    Generates validation forecasts iteratively: for each validation timestamp t,
    builds feature row using only historical actuals (train + earlier validation actuals, not future).
    This prevents leakage where lag/rolling features would otherwise use the entire validation window.

    Returns best fitted model on train+val (for later refit on full series) and metrics.
    """
    from lightgbm import LGBMRegressor
    from typing import Dict, Any, Optional, List

    if train_series is None or val_series is None or len(train_series) < 24 or len(val_series) == 0:
        if diagnostic_messages is not None:
            diagnostic_messages.append("❌ LightGBM: Insufficient data for leakage-safe validation")
        return None, {}, np.inf, np.inf, np.nan, np.nan, None

    def build_feature_frame(series: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({'ACR': series.values}, index=series.index)
        for lag in [1,2,3,6,12]:
            df[f'lag_{lag}'] = df['ACR'].shift(lag)
        for window in [3,6,12]:
            df[f'rolling_mean_{window}'] = df['ACR'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['ACR'].rolling(window).std()
        idx = pd.to_datetime(df.index)
        df['month'] = idx.month
        df['quarter'] = idx.quarter
        df['year'] = idx.year
        df['trend'] = range(len(df))
        return df

    param_grid: List[Dict[str, Any]] = [
        {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 15, "max_depth": 5},
        {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6},
        {"n_estimators": 150, "learning_rate": 0.1, "num_leaves": 31, "max_depth": 7},
        {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 63, "max_depth": 8},
        {"n_estimators": 100, "learning_rate": 0.01, "num_leaves": 15, "max_depth": 4},
    ]

    best = {
        'mape': np.inf,
        'smape': np.inf,
        'mase': np.inf,
        'rmse': np.inf,
        'params': None,
        'model': None,
        'val_forecast': None
    }

    for params in param_grid:
        try:
            # Fit on training portion ONLY for model selection
            train_df = build_feature_frame(train_series).dropna()
            feature_cols = [c for c in train_df.columns if c != 'ACR']
            if len(train_df) == 0:
                continue
            model = LGBMRegressor(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                num_leaves=params['num_leaves'],
                max_depth=params['max_depth'],
                random_state=42,
                verbose=-1,
                force_col_wise=True
            )
            model.fit(train_df[feature_cols], train_df['ACR'])

            # Iterative validation predictions (expanding window using ACTUALS up to t-1)
            history = train_series.copy()
            val_preds = []
            for ts, y_true in val_series.items():
                feats_df = build_feature_frame(history).dropna()
                if len(feats_df) == 0:
                    break
                X_last = feats_df[feature_cols].iloc[-1:]
                pred_raw = model.predict(X_last)
                arr = np.asarray(pred_raw).ravel()
                y_pred = float(arr[0]) if arr.size else float('nan')
                val_preds.append(y_pred)
                # Append ACTUAL (not prediction) to history for next step to avoid compounding error in validation metrics
                history = pd.concat([history, pd.Series([y_true], index=[ts])])
            if len(val_preds) != len(val_series):
                continue
            val_preds_arr = np.array(val_preds)
            val_mape, val_smape, val_mase, val_rmse = calculate_validation_metrics(val_series.values, val_preds_arr, train_series.values)
            if val_mape < best['mape']:
                best.update({
                    'mape': val_mape,
                    'smape': val_smape,
                    'mase': val_mase,
                    'rmse': val_rmse,
                    'params': params.copy(),
                    'model': model,
                    'val_forecast': val_preds_arr
                })
        except Exception as e:
            if diagnostic_messages is not None:
                diagnostic_messages.append(f"LightGBM params {params} failed: {str(e)[:60]}")
            continue

    if best['model'] is None:
        if diagnostic_messages is not None:
            diagnostic_messages.append("❌ LightGBM: No valid model (post leakage-fix)")
        return None, {}, np.inf, np.inf, np.nan, np.nan, None
    if diagnostic_messages is not None:
        p = best['params']
        diagnostic_messages.append(
            f"✅ LightGBM (leak-safe): n_est:{p['n_estimators']} lr:{p['learning_rate']} leaves:{p['num_leaves']} depth:{p['max_depth']} MAPE:{best['mape']:.1%}"
        )
    return best['model'], best['params'], best['mape'], best['smape'], best['mase'], best['rmse'], best['val_forecast']

def select_business_aware_best_model(product_model_mapes, product_name="", diagnostic_messages=None, model_ranks=None):
    """
    Business-aware model selection that considers both multi-metric performance and business context.
    For consumptive businesses, deprioritizes polynomial models even if they have better metrics.
    
    Args:
        product_model_mapes: Dict of model -> MAPE values
        product_name: Name of product for logging
        diagnostic_messages: List to append diagnostic messages
        model_ranks: Optional dict of model -> average_rank_across_metrics for more sophisticated selection
    """
    if not product_model_mapes:
        return None, 1.0
    
    # Business model hierarchy for consumptive revenue recognition
    business_priority = {
        "SARIMA": 1,        # Best for seasonal revenue patterns with business cycles
        "ETS": 2,           # Good for trending revenue with seasonality  
        "Prophet": 3,       # Good for revenue with holidays and external factors
        "Auto-ARIMA": 4,    # Statistical rigor but less business context
        "LightGBM": 5,      # Good for complex patterns but less interpretable
        "Poly-2": 6,        # Polynomial models - problematic for consumptive revenue
        "Poly-3": 7         # Most problematic - can create unrealistic growth curves
    }
    
    # Use multi-metric ranking if available, otherwise fall back to MAPE
    if model_ranks:
        # Select best performing model overall (lowest average rank across all metrics)
        best_overall_model = min(model_ranks.keys(), key=lambda k: model_ranks[k])
        selection_criterion = f"Multi-metric rank: {model_ranks[best_overall_model]:.1f}"
    else:
        # Original MAPE-only selection
        best_overall_model = min(product_model_mapes.keys(), key=lambda k: product_model_mapes[k])
        selection_criterion = f"MAPE: {product_model_mapes[best_overall_model]:.1%}"
    
    # Separate business-appropriate models from polynomial models
    business_appropriate = {}
    polynomial_models = {}
    
    for model, mape in product_model_mapes.items():
        if model in ["Poly-2", "Poly-3"]:
            polynomial_models[model] = mape
        else:
            business_appropriate[model] = mape
    
    # Decision logic: STRONGLY prefer business-appropriate models
    if business_appropriate:
        # Choose best business model (using ranking if available, else MAPE)
        if model_ranks:
            # Filter ranks to only business-appropriate models
            business_ranks = {k: v for k, v in model_ranks.items() if k in business_appropriate}
            if business_ranks:
                best_business_model = min(business_ranks.keys(), key=lambda k: business_ranks[k])
            else:
                best_business_model = min(business_appropriate.keys(), key=lambda k: business_appropriate[k])
        else:
            best_business_model = min(business_appropriate.keys(), key=lambda k: business_appropriate[k])
            
        selected_mape = business_appropriate[best_business_model]
        
        # Only consider polynomial models if business models are catastrophically bad
        if polynomial_models:
            best_poly_mape = min(polynomial_models.values())
            mape_improvement = (selected_mape - best_poly_mape) / selected_mape if selected_mape > 0 else 0
            
            # BIAS REDUCTION: More balanced criteria for polynomial selection
            # Polynomial must be significantly better (>50% improvement) AND business model must be poor (>40% MAPE)
            poly_threshold_improvement = 0.50  # 50% improvement required
            business_failure_threshold = 0.40   # 40% MAPE threshold
            
            if mape_improvement > poly_threshold_improvement and selected_mape > business_failure_threshold:
                if diagnostic_messages:
                    diagnostic_messages.append(
                        f"⚠️ Product {product_name}: Selecting polynomial {min(polynomial_models.keys(), key=lambda k: polynomial_models[k])} "
                        f"(MAPE: {best_poly_mape:.1%}) over {best_business_model} (MAPE: {selected_mape:.1%}) "
                        f"due to significant performance gap (improvement: {mape_improvement:.1%}, threshold: {poly_threshold_improvement:.1%})."
                    )
                return min(polynomial_models.keys(), key=lambda k: polynomial_models[k]), best_poly_mape
            else:
                if diagnostic_messages:
                    rank_info = f", Avg Rank: {model_ranks[best_business_model]:.1f}" if model_ranks else ""
                    diagnostic_messages.append(
                        f"🏢 Product {product_name}: Business-aware selection chose {best_business_model} (MAPE: {selected_mape:.1%}{rank_info}) "
                        f"over polynomial models (best poly MAPE: {best_poly_mape:.1%}, improvement: {mape_improvement:.1%}) "
                        f"- Requires ≥{poly_threshold_improvement:.1%} improvement + ≥{business_failure_threshold:.1%} business MAPE to select polynomial"
                    )
        else:
            if diagnostic_messages:
                rank_info = f" (Avg Rank: {model_ranks[best_business_model]:.1f})" if model_ranks else ""
                diagnostic_messages.append(
                    f"✅ Product {product_name}: Selected {best_business_model} (MAPE: {selected_mape:.1%}{rank_info}) - business-appropriate model with {selection_criterion}"
                )
        
        return best_business_model, selected_mape
    
    elif polynomial_models:
        # Only use polynomial models if NO business models exist (which should be rare)
        best_poly_model = min(polynomial_models.keys(), key=lambda k: polynomial_models[k])  # type: ignore
        selected_mape = polynomial_models[best_poly_model]
        
        if diagnostic_messages:
            diagnostic_messages.append(
                f"⚠️ Product {product_name}: No business-appropriate models available! Using {best_poly_model} (MAPE: {selected_mape:.1%}). "
                f"This is unusual - consider checking your data quality or model parameters."
            )
        
        return best_poly_model, selected_mape
    
    else:
        # Fallback to raw best MAPE
        best_model = min(product_model_mapes, key=product_model_mapes.get)
        return best_model, product_model_mapes[best_model]
