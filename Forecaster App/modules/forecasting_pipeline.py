"""
Main forecasting application logic and workflow orchestration.

Contains the core forecasting pipeline and main application workflow
that coordinates data processing, model fitting, and results generation.
"""

import io
import itertools
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

# Local imports
from .models import (
    detect_seasonality_strength, apply_statistical_validation, apply_business_adjustments_to_forecast,
    apply_trend_aware_forecasting, fit_final_sarima_model, fit_best_sarima, fit_best_lightgbm,
    select_business_aware_best_model, get_seasonality_aware_split, HAVE_PMDARIMA, HAVE_PROPHET, HAVE_LGBM,
    create_ets_fitting_function, create_sarima_fitting_function, create_auto_arima_fitting_function,
    create_prophet_fitting_function, create_polynomial_fitting_function, fit_seasonal_naive, ENABLE_PROPHET, ENABLE_LGBM,
    create_lightgbm_fitting_function
)
from .metrics import calculate_validation_metrics, comprehensive_validation_suite, walk_forward_validation
from .utils import coerce_month_start
from .ui_components import fy

# Import optional dependencies with fallbacks
Prophet = None
auto_arima = None
LGBMRegressor = None

if HAVE_PROPHET:
    try:
        from prophet import Prophet
    except ImportError:
        pass

if HAVE_PMDARIMA:
    try:
        from pmdarima.arima import auto_arima
    except ImportError:
        try:
            from pmdarima import auto_arima
        except ImportError:
            pass

if HAVE_LGBM:
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        pass

# Model fitting imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------------------------
# Global (module-level) drift carry configuration & helper
# These are populated at runtime by run_forecasting_pipeline so that the per-
# model runner helpers (which operate outside the pipeline function scope)
# can access the selective drift logic without threading many parameters.
# ---------------------------------------------------------------------------
DRIFT_CFG: dict = {
    "enable": False,
    "min_pct": 0.0,
    "max_pct": 0.0,
    "forecast_cv_threshold": 0.0,
    "hist_cv_min": 0.0,
}

_DRIFT_DIAGNOSTICS_REF: list = None
DRIFT_APPLIED_PRODUCTS: set = set()


def _maybe_apply_drift(series: pd.Series, forecast, model_name: str, product: str):
    """Optionally apply gentle linear drift carry to overly flat forecasts.

    Uses global DRIFT_CFG set by run_forecasting_pipeline. Conditions:
      - Drift feature enabled
      - Sufficient history
      - Forecast first segment (up to 6 steps) coefficient of variation below threshold
      - Historical recent CV above minimum (i.e. there *was* some variability)
      - Recent slope (last 6 points) relative to recent mean within [min_pct, max_pct]
    The drift is applied linearly across the forecast horizon and clipped at 0 when
    the historical series is non-negative. Diagnostics appended to global list.
    """
    cfg = DRIFT_CFG
    if not cfg.get("enable"):
        return forecast
    diag = _DRIFT_DIAGNOSTICS_REF
    try:
        if len(series) < 6 or len(forecast) == 0:
            return forecast
        recent_hist = series.tail(min(12, len(series)))
        hist_mean = recent_hist.mean()
        if hist_mean == 0:
            return forecast
        # Relative slope over last up to 6 observations
        if len(recent_hist) >= 6:
            slope = (recent_hist.iloc[-1] - recent_hist.iloc[-6]) / 6.0
        else:
            slope = (recent_hist.iloc[-1] - recent_hist.iloc[0]) / max(1, len(recent_hist) - 1)
        slope_pct = slope / hist_mean
        fc_arr = np.array(forecast, dtype=float)
        first_segment = fc_arr[: min(6, len(fc_arr))]
        mean_first = first_segment.mean()
        fore_cv = first_segment.std() / (mean_first + 1e-9) if mean_first != 0 else 0.0
        hist_cv = recent_hist.std() / (hist_mean + 1e-9)
        if (
            fore_cv < float(cfg.get("forecast_cv_threshold", 0.0))
            and hist_cv > float(cfg.get("hist_cv_min", 0.0))
            and float(cfg.get("min_pct", 0.0)) <= abs(slope_pct) <= float(cfg.get("max_pct", 0.0))
        ):
            adj = fc_arr.copy()
            for i in range(len(adj)):
                adj[i] = adj[i] + slope * (i + 1)
            if series.min() >= 0:
                adj = np.maximum(adj, 0)
            if diag is not None:
                diag.append(
                    f"🪄 Drift carry applied ({model_name}, {product}): slope {slope_pct*100:.2f}%/mo over {len(adj)} steps."
                )
            # Track product-level application for UI badge
            try:
                DRIFT_APPLIED_PRODUCTS.add(product)
            except Exception:
                pass
            return adj
    except Exception as e:  # noqa: BLE001
        if diag is not None:
            diag.append(f"⚠️ Drift carry skipped ({model_name}, {product}): {e}")
    return forecast


def run_forecasting_pipeline(raw_data, models_selected, horizon=12, enable_statistical_validation=True,
                            apply_business_adjustments=False, business_growth_assumption=0,
                            market_multiplier=1.0, market_conditions="Stable",
                            enable_business_aware_selection=True, enable_prophet_holidays=False,
                            enable_backtesting=True,
                            use_backtesting_selection: bool = True,
                            backtest_months: int = 12,
                            backtest_gap: int = 0,
                            validation_horizon: int = 6,
                            fiscal_year_start_month: int = 1,
                            enable_per_product_drift: bool = False,
                            drift_min_pct: float = 0.005,
                            drift_max_pct: float = 0.03,
                            drift_forecast_cv_threshold: float = 0.01,
                            drift_hist_cv_min: float = 0.02):
    """
    Main forecasting pipeline that processes data and runs all selected models.
    
    Args:
        raw_data: Raw input DataFrame with Date, Product, ACR columns
        models_selected: List of model names to run
        horizon: Number of months to forecast
        enable_statistical_validation: Whether to apply statistical validation
        apply_business_adjustments: Whether to apply business adjustments
        business_growth_assumption: Annual growth percentage
        market_multiplier: Market condition multiplier
        market_conditions: Market conditions description
        enable_business_aware_selection: Whether to use business-aware model selection
        enable_prophet_holidays: Whether to include holidays in Prophet
        enable_backtesting: Whether to run backtesting validation analysis
                # Note: Walk-forward and cross-validation removed in favor of simple backtesting
    use_backtesting_selection: If True, override per-product selection using backtesting diagnostics
    enable_per_product_drift: If True, apply gentle drift carry only on products whose model forecast is excessively flat.
    drift_min_pct / drift_max_pct: Monthly pct slope bounds (relative to recent mean) for applying drift.
    drift_forecast_cv_threshold: Max coefficient of variation of first 6 forecast points to consider "flat".
    drift_hist_cv_min: Minimum historical CV (recent 12) indicating some variability exists.
    
    Returns:
        Tuple of (results_dict, avg_mapes_dict, sarima_params_dict, diagnostic_messages,
                 product_metrics_dicts, best_models_per_product, best_mapes_per_product,
                 advanced_validation_results)
    """
    # Configure global drift helper for downstream model routines
    global DRIFT_CFG, _DRIFT_DIAGNOSTICS_REF
    DRIFT_CFG.update({
        "enable": bool(enable_per_product_drift),
        "min_pct": float(drift_min_pct),
        "max_pct": float(drift_max_pct),
        "forecast_cv_threshold": float(drift_forecast_cv_threshold),
        "hist_cv_min": float(drift_hist_cv_min),
    })

    # Initialize containers
    results, mapes, sarima_params = {}, {}, {}
    smapes, mases, rmses = {}, {}, {}
    diagnostic_messages = []
    products = raw_data["Product"].unique()
    backtesting_results = {}  # Store backtesting validation results
    
    # Enhanced logging: Pipeline initialization
    diagnostic_messages.append(f"🚀 **Pipeline Initialization**: Starting forecast pipeline with {len(models_selected)} models for {len(products)} products")
    diagnostic_messages.append(f"📊 **Data Overview**: Total data points: {len(raw_data)}, Date range: {raw_data['Date'].min()} to {raw_data['Date'].max()}")
    diagnostic_messages.append(f"⚙️ **Configuration**: Backtesting: {enable_backtesting}, Business-aware: {enable_business_aware_selection}, Statistical validation: {enable_statistical_validation}")
    
    # Prepare data
    try:
        raw_data["Date"] = coerce_month_start(raw_data["Date"])
        raw_data.sort_values("Date", inplace=True)
        diagnostic_messages.append(f"📅 **Data Preparation**: Successfully standardized dates and sorted data chronologically")
    except ValueError as e:
        raise ValueError(f"Date format error: {str(e)}")
    
    # Count valid products for progress tracking
    diagnostic_messages.append(f"🔍 **Data Validation**: Analyzing data quality for {len(products)} products...")
    valid_products = _get_valid_products(raw_data, diagnostic_messages)
    # Respect user selections; do not auto-add models that weren't chosen
    enforced_models = list(models_selected)
    # Keep only user-chosen models (deduplicated)
    models_selected = list(dict.fromkeys([m for m in models_selected if m in enforced_models]))
    total = len(models_selected) * len(valid_products)
    
    if total == 0:
        raise ValueError("No models selected or no valid products found with sufficient data")
    
    diagnostic_messages.append(f"✅ **Validation Complete**: {len(valid_products)} products passed quality checks, {len(products) - len(valid_products)} products failed")
    diagnostic_messages.append(f"📈 **Processing Scope**: Will run {len(models_selected)} models on {len(valid_products)} products = {total} total operations")
    
    # Initialize progress tracking
    prog = st.progress(0.0, text="Running models…")
    done = 0
    
    # Coerce selected models to allowed defaults + Auto-ARIMA if available
    # Respect user selections but prune unsupported/forbidden options
    try:
        pruned: list[str] = []
        for m in models_selected:
            if m == "Poly-3":
                diagnostic_messages.append("⚠️ Poly-3 disabled by policy; keeping Poly-2 only.")
                continue
            if m == "Auto-ARIMA" and not HAVE_PMDARIMA:
                diagnostic_messages.append("⚠️ Auto-ARIMA not available (pmdarima missing).")
                continue
            pruned.append(m)
        models_selected = pruned
    except Exception:
        pass
    
    # Recompute total after pruning/availability filtering
    total = len(models_selected) * len(valid_products)
    if total == 0:
        raise ValueError("No models selected or supported in this environment. Please select at least one model.")
    
    # Process each product
    diagnostic_messages.append(f"🔄 **Starting Product Processing**: Processing {len(valid_products)} products sequentially...")
    
    for product, grp in raw_data.groupby("Product"):
        if product not in valid_products:
            continue
        
        diagnostic_messages.append(f"📦 **Processing Product**: {product} ({len(grp)} data points)")
        
        # Prepare product data
        series = _prepare_product_series(grp, product, diagnostic_messages)
        if series is None:
            continue
        
        # Get train/validation split
        diagnostic_messages.append(f"✂️ **Data Splitting**: Creating train/validation split for {product} ({len(series)} total months)")
        split_result = get_seasonality_aware_split(series, seasonal_period=12, diagnostic_messages=diagnostic_messages)
        if split_result[0] is None:
            continue
        
        train, val = split_result
        diagnostic_messages.append(f"✅ **Split Complete**: {product} - Training: {len(train)} months, Validation: {len(val)} months")
        
        # Create future date index
        future_idx = pd.date_range(
            pd.Timestamp(series.index[-1]) + pd.DateOffset(months=1), 
            periods=horizon, freq="MS"
        )
        
        # Create actual data DataFrame
        act_df = pd.DataFrame({
            "Product": product, 
            "Date": series.index, 
            "ACR": series.values, 
            "Type": "actual"
        })
        
        # Initialize model results for this product
        for m in models_selected:
            results.setdefault(m, [])
            mapes.setdefault(m, [])
            smapes.setdefault(m, [])
            mases.setdefault(m, [])
            rmses.setdefault(m, [])
        
        # Ensure results dicts include Seasonal-Naive container if selected
        if "Seasonal-Naive" in models_selected:
            results.setdefault("Seasonal-Naive", [])
            mapes.setdefault("Seasonal-Naive", [])
            smapes.setdefault("Seasonal-Naive", [])
            mases.setdefault("Seasonal-Naive", [])
            rmses.setdefault("Seasonal-Naive", [])
        
        # Run each model
        done = _run_models_for_product(
            product, series, train, val, future_idx, act_df,
            models_selected, results, mapes, smapes, mases, rmses,
            sarima_params, diagnostic_messages, horizon,
            enable_statistical_validation, apply_business_adjustments,
            business_growth_assumption, market_multiplier, market_conditions,
            enable_prophet_holidays, enable_backtesting, backtest_months, backtest_gap, validation_horizon,
            fiscal_year_start_month,
            backtesting_results, prog, done, total
        )
    
    prog.empty()
    
    # Process results
    results = _process_model_results(results)
    
    # Calculate metrics and rankings
    avg_mapes, avg_smapes, avg_mases, avg_rmses, model_avg_ranks = _calculate_average_metrics(
        mapes, smapes, mases, rmses, results, products
    )
    
    # Find best models per product (STANDARD selection by multi-metric ranking and optional business-aware filter)
    best_models_per_product, best_mapes_per_product = _find_best_models_per_product(
        products, results, mapes, smapes, mases, rmses, enable_business_aware_selection, diagnostic_messages
    )
    # Persist the standard selection for toggle support
    best_models_per_product_standard = dict(best_models_per_product)
    best_mapes_per_product_standard = dict(best_mapes_per_product)

    # Compute a BACKTESTING-driven selection alternative when backtesting results exist
    best_models_per_product_backtesting: dict[str, str] = {}
    best_mapes_per_product_backtesting: dict[str, float] = {}
    best_model_reasons_backtesting: dict[str, str] = {}
    if backtesting_results:
        for product in list(best_models_per_product_standard.keys()):
            per_model = backtesting_results.get(product, {})
            # Enforce strict eligibility policy and score by backtesting-only WAPE (legacy 'mape')
            def score_validation(v):
                if not v:
                    return (np.inf, np.inf, 9e9)
                bt = v.get('backtesting_validation') or {}
                # Eligibility: at least 2 folds, mean_mase < 1.0, relative WAPE improvement vs seasonal-naive >= 5%, p95 <= 2x mean
                # We expect aggregation to include these; otherwise, mark ineligible.
                folds = bt.get('folds', bt.get('iterations'))
                mean_mase = bt.get('mean_mase')
                p95 = bt.get('p95_mape', np.nan)
                # Prefer recency‑weighted mean if present
                m = bt.get('recent_weighted_mape', bt.get('mape', np.inf))
                # Beat seasonal naive if comparative present
                rel_ok = True
                try:
                    naive = per_model.get('Seasonal-Naive', {}).get('backtesting_validation', {})
                    naive_wape = naive.get('recent_weighted_mape', naive.get('mape', np.inf))
                    if np.isfinite(m) and np.isfinite(naive_wape) and naive_wape > 0:
                        rel_ok = (naive_wape - m) / naive_wape >= 0.05
                except Exception:
                    rel_ok = True
                if folds is not None and int(folds) < 2:
                    return (np.inf, np.inf, 9e9)
                if mean_mase is not None and np.isfinite(mean_mase) and mean_mase >= 1.0:
                    return (np.inf, np.inf, 9e9)
                if np.isfinite(p95) and np.isfinite(m) and p95 > 2.0 * m:
                    return (np.inf, np.inf, 9e9)
                if not rel_ok:
                    return (np.inf, np.inf, 9e9)
                wape_mean = m if np.isfinite(m) else np.inf
                # If fold stats exist elsewhere, read p75; otherwise use mean
                p75 = bt.get('recent_p75_mape', bt.get('p75_mape', wape_mean))
                mase_val = bt.get('mase', bt.get('mean_mase', np.nan))
                return (wape_mean, p75, np.nan_to_num(mase_val, nan=9e9))

            # helper to provide eligibility reason for tooltips
            def eligibility_reason(name: str, v):
                if not v:
                    return False, "no backtesting results", (np.inf, np.inf, 9e9)
                bt = v.get('backtesting_validation') or {}
                folds = bt.get('folds', bt.get('iterations'))
                mean_mase = bt.get('mean_mase')
                p95 = bt.get('p95_mape', np.nan)
                m = bt.get('mape', np.inf)
                p75 = bt.get('p75_mape', m)
                mase_val = bt.get('mase', bt.get('mean_mase', np.nan))
                # seasonal-naive relative improvement
                rel_ok = True
                rel_msg = ""
                try:
                    naive = per_model.get('Seasonal-Naive', {}).get('backtesting_validation', {})
                    naive_wape = naive.get('recent_weighted_mape', naive.get('mape', np.inf))
                    if np.isfinite(m) and np.isfinite(naive_wape) and naive_wape > 0:
                        rel_ok = (naive_wape - m) / naive_wape >= 0.05
                        if not rel_ok:
                            rel_msg = f"<5% better than Seasonal‑Naive (Δ={(naive_wape - m)/naive_wape:.1%})"
                except Exception:
                    rel_ok = True
                if folds is not None and int(folds) < 2:
                    return False, "<2 CV folds", (m, p75, np.nan_to_num(mase_val, nan=9e9))
                if mean_mase is not None and np.isfinite(mean_mase) and mean_mase >= 1.0:
                    return False, "MASE ≥ 1.0 (did not beat seasonal‑naive)", (m, p75, np.nan_to_num(mase_val, nan=9e9))
                if np.isfinite(p95) and np.isfinite(m) and p95 > 2.0 * m:
                    return False, "p95 WAPE > 2× mean (unstable)", (m, p75, np.nan_to_num(mase_val, nan=9e9))
                if not rel_ok:
                    return False, rel_msg or "<5% better than Seasonal‑Naive", (m, p75, np.nan_to_num(mase_val, nan=9e9))
                return True, "eligible", (m, p75, np.nan_to_num(mase_val, nan=9e9))
            if per_model:
                pairs = [(m, score_validation(res)) for m, res in per_model.items()]
                finite_pairs = [(m, s) for m, s in pairs if np.isfinite(s[0])] or pairs

                # Separate non-polynomial and polynomial candidates
                poly_names = {"Poly-2", "Poly-3"}
                non_poly = [(m, s) for m, s in finite_pairs if m not in poly_names]
                polys = [(m, s) for m, s in finite_pairs if m in poly_names]

                # If Business-Aware Selection is enabled, exclude polynomial models from backtesting-driven selection
                if enable_business_aware_selection:
                    if non_poly:
                        best_model, best_score = min(non_poly, key=lambda x: x[1])
                        if polys:
                            # Informative diagnostic when a poly looked good but was excluded
                            best_poly_model, best_poly_score = min(polys, key=lambda x: x[1])
                            diagnostic_messages.append(
                                f"🏢 {product}: Business-aware selection excluded polynomial {best_poly_model} (WAPE {best_poly_score[0]:.1%}); using {best_model} (WAPE {best_score[0]:.1%})."
                            )
                    else:
                        # No non-poly candidates; fall back to overall best but warn
                        best_model, best_score = min(finite_pairs, key=lambda x: x[1])
                        if best_model in poly_names:
                            diagnostic_messages.append(
                                f"⚠️ {product}: No non-polynomial candidates available for backtesting; falling back to {best_model} (WAPE {best_score[0]:.1%})."
                            )
                else:
                    # Deprioritize polynomial: only pick poly if EVERYTHING else fails eligibility
                    if non_poly:
                        best_model, best_score = min(non_poly, key=lambda x: x[1])
                    else:
                        # only polys available
                        best_model, best_score = min(finite_pairs, key=lambda x: x[1])

                best_models_per_product_backtesting[product] = best_model
                best_mapes_per_product_backtesting[product] = float(best_score[0])
                # Build reason string for tooltip
                try:
                    # Evaluate eligibility and capture any exclusions
                    reasons_excluded = []
                    for name, res in per_model.items():
                        ok, reason_txt, sc = eligibility_reason(name, res)
                        if not ok:
                            # if this excluded model would have had lower mean WAPE than winner, highlight
                            try:
                                if np.isfinite(sc[0]) and sc[0] < best_score[0]:
                                    reasons_excluded.append(f"{name}: {reason_txt}")
                            except Exception:
                                pass
                    reason_main = f"Chosen by lowest mean WAPE among eligible models: {best_model} (mean {best_score[0]:.1%}, p75 {best_score[1]:.1%}, MASE {best_score[2]:.2f})."
                    if reasons_excluded:
                        reason_main += " Excluded better-WAPE candidates due to: " + "; ".join(reasons_excluded[:3])
                    best_model_reasons_backtesting[product] = reason_main
                except Exception:
                    best_model_reasons_backtesting[product] = f"Chosen by lowest mean WAPE: {best_model}."

    # Apply UI toggle: if user chose backtesting selection, use that mapping as the primary "best" mapping
    if use_backtesting_selection and best_models_per_product_backtesting:
        best_models_per_product = dict(best_models_per_product_backtesting)
        best_mapes_per_product = dict(best_mapes_per_product_backtesting)
        diagnostic_messages.append("🔁 Using backtesting-driven model selection per product (WAPE-based, CV‑only)")
        # Store a compact method mix for UI KPI
        try:
            from collections import Counter
            cnt = Counter(best_models_per_product_backtesting.values())
            total = sum(cnt.values()) or 1
            mix_str = ", ".join([f"{m} {cnt[m]*100/total:.0f}%" for m in sorted(cnt, key=lambda k: -cnt[k])])
            st.session_state.best_models_per_product_backtesting_mix = mix_str
        except Exception:
            st.session_state.best_models_per_product_backtesting_mix = None
    
    # Create only backtesting hybrid variant for viewing
    if best_models_per_product_backtesting:
        results = _create_hybrid_model(results, best_models_per_product_backtesting, avg_mapes, best_mapes_per_product_backtesting, model_key_name="Best per Product (Backtesting)")

    # Provide diagnostics list reference to global helper
    _DRIFT_DIAGNOSTICS_REF = diagnostic_messages
    # Do not create Standard or Mix variants anymore per policy
    raw_models_per_product = None

    # Store both mappings in session for UI toggle/explanations
    try:
        st.session_state.best_models_per_product_standard = best_models_per_product_standard
        st.session_state.best_models_per_product_backtesting = (
            best_models_per_product_backtesting if best_models_per_product_backtesting else None
        )
        st.session_state.best_model_reasons_backtesting = (
            best_model_reasons_backtesting if best_model_reasons_backtesting else {}
        )
        st.session_state.best_models_per_product_raw = (
            raw_models_per_product if raw_models_per_product else None
        )
        # Provide drift application info
        from .forecasting_pipeline import DRIFT_APPLIED_PRODUCTS  # self-import safe
        st.session_state.drift_applied_products = (
            sorted(list(DRIFT_APPLIED_PRODUCTS)) if DRIFT_APPLIED_PRODUCTS else []
        )
    except Exception:
        pass
    
    # Store metrics in session state for UI access
    st.session_state.model_avg_ranks = model_avg_ranks
    st.session_state.product_mapes = _create_product_metrics_dict(mapes, products, results.keys())
    st.session_state.product_smapes = _create_product_metrics_dict(smapes, products, results.keys())
    st.session_state.product_mases = _create_product_metrics_dict(mases, products, results.keys())
    st.session_state.product_rmses = _create_product_metrics_dict(rmses, products, results.keys())
    
    # Enhanced completion logging
    successful_models = sum(1 for model in results.values() if not model.empty)
    total_expected = len(models_selected) * len(valid_products)
    success_rate = (successful_models / total_expected) * 100 if total_expected > 0 else 0
    
    diagnostic_messages.append(f"🎉 **Pipeline Complete**: Successfully processed {successful_models}/{total_expected} model-product combinations ({success_rate:.1f}% success rate)")
    diagnostic_messages.append(f"📊 **Results Summary**: Generated forecasts for {len(results)} models across {len(valid_products)} products")
    
    # Enhanced backtesting summary with detailed failure analysis
    if backtesting_results:
        diagnostic_messages.append("🧪 **DETAILED BACKTESTING ANALYSIS**")
        
        # Safe backtesting success calculation with null checks
        backtesting_success = 0
        total_backtests = 0
        failed_backtests = []
        skipped_backtests = []
        
        for product_name, product_models in backtesting_results.items():
            if isinstance(product_models, dict):
                diagnostic_messages.append(f"📦 **Product: {product_name}**")
                
                for model_name, model_result in product_models.items():
                    total_backtests += 1
                    
                    if model_result is None:
                        skipped_backtests.append(f"{model_name} for {product_name} (result was None)")
                        diagnostic_messages.append(f"  ❌ **{model_name}**: Backtesting result was None - likely model failed to train")
                    elif isinstance(model_result, dict):
                        backtesting_validation = model_result.get('backtesting_validation')
                        
                        if backtesting_validation is None:
                            skipped_backtests.append(f"{model_name} for {product_name} (no backtesting validation)")
                            diagnostic_messages.append(f"  ⚠️ **{model_name}**: No backtesting validation available - model may have failed or been skipped")
                        elif isinstance(backtesting_validation, dict):
                            if backtesting_validation.get('success'):
                                backtesting_success += 1
                                mape = backtesting_validation.get('mape', 0)
                                test_months = backtesting_validation.get('test_months', 0)
                                diagnostic_messages.append(f"  ✅ **{model_name}**: Backtesting successful - WAPE: {mape*100:.1f}%, Test period: {test_months} months")
                            else:
                                failed_backtests.append(f"{model_name} for {product_name}")
                                error_msg = backtesting_validation.get('error', 'Unknown error')
                                diagnostic_messages.append(f"  ❌ **{model_name}**: Backtesting failed - Error: {error_msg}")
                        else:
                            failed_backtests.append(f"{model_name} for {product_name}")
                            diagnostic_messages.append(f"  ❌ **{model_name}**: Invalid backtesting validation format")
                    else:
                        failed_backtests.append(f"{model_name} for {product_name}")
                        diagnostic_messages.append(f"  ❌ **{model_name}**: Unexpected result type: {type(model_result)}")
                
                diagnostic_messages.append("")  # Add spacing between products
        
        # Summary statistics
        if total_backtests > 0:
            backtesting_rate = (backtesting_success / total_backtests) * 100
            diagnostic_messages.append(f"📊 **BACKTESTING SUMMARY STATISTICS**")
            diagnostic_messages.append(f"  • Total backtests attempted: {total_backtests}")
            diagnostic_messages.append(f"  • Successful backtests: {backtesting_success}")
            diagnostic_messages.append(f"  • Failed backtests: {len(failed_backtests)}")
            diagnostic_messages.append(f"  • Skipped backtests: {len(skipped_backtests)}")
            diagnostic_messages.append(f"  • Success rate: {backtesting_rate:.1f}%")
            
            if failed_backtests:
                diagnostic_messages.append(f"  • Failed combinations: {', '.join(failed_backtests)}")
            if skipped_backtests:
                diagnostic_messages.append(f"  • Skipped combinations: {', '.join(skipped_backtests)}")
        else:
            diagnostic_messages.append("🧪 **Backtesting Summary**: No backtesting results available")
    
    return (results, avg_mapes, sarima_params, diagnostic_messages,
            {'smapes': avg_smapes, 'mases': avg_mases, 'rmses': avg_rmses},
            best_models_per_product, best_mapes_per_product, backtesting_results)


def _get_valid_products(raw_data, diagnostic_messages):
    """Get list of products with sufficient data for forecasting."""
    valid_products = []
    
    for product, grp in raw_data.groupby("Product"):
        try:
            series = grp.set_index("Date")["ACR"].astype(float)
            if not isinstance(series.index, pd.DatetimeIndex):
                series.index = pd.to_datetime(series.index)
            series.index = series.index.to_period("M").to_timestamp(how="start")
            
            if len(series) >= 12:  # Minimum data requirement
                seasonality_strength = detect_seasonality_strength(series)
                split_result = get_seasonality_aware_split(series, seasonal_period=12, diagnostic_messages=None)
                if split_result[0] is not None:
                    valid_products.append(product)
                    
        except Exception as e:
            diagnostic_messages.append(f"❌ Product {product}: Data processing error - {str(e)[:100]}. Skipping.")
            continue
    
    return valid_products


def _prepare_product_series(grp, product, diagnostic_messages):
    """Prepare time series data for a single product."""
    try:
        series = grp.set_index("Date")["ACR"].astype(float)
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)
        series.index = series.index.to_period("M").to_timestamp(how="start")
        
        # Data quality checks
        if len(series) < 12:
            diagnostic_messages.append(f"❌ Product {product}: Insufficient data ({len(series)} months). Skipping.")
            return None
        
        # Remove extreme outliers
        z_scores = np.abs((series - series.mean()) / series.std())
        if (z_scores > 5).any():
            outlier_count = (z_scores > 5).sum()
            diagnostic_messages.append(f"Product {product}: Detected {outlier_count} extreme outliers, capping them.")
            series = series.clip(series.quantile(0.01), series.quantile(0.99))
        
        # Log seasonality detection
        seasonality_strength = detect_seasonality_strength(series)
        if seasonality_strength > 0.6:
            diagnostic_messages.append(f"📈 Product {product}: Strong seasonality detected (strength: {seasonality_strength:.2f})")
        elif seasonality_strength > 0.3:
            diagnostic_messages.append(f"📊 Product {product}: Moderate seasonality detected (strength: {seasonality_strength:.2f})")
        else:
            diagnostic_messages.append(f"📉 Product {product}: Weak seasonality detected (strength: {seasonality_strength:.2f})")
        
        return series
        
    except Exception as e:
        diagnostic_messages.append(f"❌ Product {product}: Data processing error - {str(e)[:100]}. Skipping.")
        return None


def _run_models_for_product(product, series, train, val, future_idx, act_df,
                           models_selected, results, mapes, smapes, mases, rmses,
                           sarima_params, diagnostic_messages, horizon,
                           enable_statistical_validation, apply_business_adjustments,
                           business_growth_assumption, market_multiplier, market_conditions,
                           enable_prophet_holidays, enable_backtesting, backtest_months, backtest_gap, validation_horizon,
                           fiscal_year_start_month,
                           backtesting_results, prog, done, total):
    """Run all selected models for a single product."""
    
    seasonality_strength = detect_seasonality_strength(series)
    
    # SARIMA model
    if "SARIMA" in models_selected:
        diagnostic_messages.append(f"🔧 **Starting SARIMA**: Training SARIMA model for {product} (seasonality strength: {seasonality_strength:.2f})")
        done = _run_sarima_model(
            product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
            sarima_params, diagnostic_messages, seasonality_strength, enable_statistical_validation,
            apply_business_adjustments, business_growth_assumption, market_multiplier, market_conditions,
            enable_backtesting, backtest_months, backtest_gap,
            validation_horizon, fiscal_year_start_month, backtesting_results, prog, done, total
        )
    
    # ETS model
    if "ETS" in models_selected:
        diagnostic_messages.append(f"🔧 **Starting ETS**: Training Exponential Smoothing model for {product}")
        done = _run_ets_model(
            product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
            diagnostic_messages, enable_statistical_validation, apply_business_adjustments,
            business_growth_assumption, market_multiplier, enable_backtesting,
            backtest_months, backtest_gap, validation_horizon, fiscal_year_start_month, backtesting_results,
            prog, done, total
        )
    
    # Seasonal-Naive baseline
    if "Seasonal-Naive" in models_selected:
        try:
            diagnostic_messages.append(f"🔧 **Starting Seasonal-Naive**: Baseline repeat last-season values for {product}")
            # Validation forecast
            pv_naive = fit_seasonal_naive(train.values if hasattr(train, 'values') else train, len(val), seasonal_period=12)
            val_mape, val_smape, val_mase, val_rmse = calculate_validation_metrics(val, pv_naive, train)
            mapes["Seasonal-Naive"].append(val_mape)
            smapes["Seasonal-Naive"].append(val_smape)
            mases["Seasonal-Naive"].append(val_mase)
            rmses["Seasonal-Naive"].append(val_rmse)

            # Future forecast
            pf_naive = fit_seasonal_naive(series.values if hasattr(series, 'values') else series, len(future_idx), seasonal_period=12)
            pf_arr = np.asarray(pf_naive)
            # Apply minimal validation/non-negativity
            if enable_statistical_validation:
                pf_arr = apply_statistical_validation(pf_arr, series, "Seasonal-Naive")
            if apply_business_adjustments:
                pf_arr = apply_business_adjustments_to_forecast(pf_arr, business_growth_assumption, market_multiplier)

            fore_df = pd.DataFrame({"Product": product, "Date": future_idx, "ACR": pf_arr, "Type": "forecast"})
            results["Seasonal-Naive"].append(pd.concat([act_df, fore_df], ignore_index=True))
            diagnostic_messages.append(f"✅ Seasonal-Naive Product {product}: WAPE {val_mape:.1%}")

            # Backtesting
            if enable_backtesting:
                try:
                    if len(series) < (backtest_months + validation_horizon + backtest_gap + 12):
                        required_months = backtest_months + validation_horizon + backtest_gap + 12
                        diagnostic_messages.append(f"⚠️ **Seasonal-Naive Backtesting Skipped**: {product} - Insufficient data for {backtest_months} month backtesting (need {required_months} months, have {len(series)})")
                    else:
                        diagnostic_messages.append(f"🧪 **Seasonal-Naive Backtesting**: Starting backtesting for {product}")
                        # Wrapper fitting function providing forecast()
                        class _NaiveModel:
                            def __init__(self, train_ts):
                                self._train = train_ts
                            def forecast(self, steps):
                                return fit_seasonal_naive(self._train.values if hasattr(self._train, 'values') else self._train, steps, seasonal_period=12)
                        def naive_fit_fn(train_ts, **kwargs):
                            return _NaiveModel(train_ts)
                        validation_results = comprehensive_validation_suite(
                            actual=val.values,
                            forecast=pv_naive,
                            dates=val.index,
                            product_name=product,
                            series=series,
                            model_fitting_func=naive_fit_fn,
                            model_params={},
                            diagnostic_messages=diagnostic_messages,
                            backtest_months=backtest_months,
                            backtest_gap=backtest_gap,
                            validation_horizon=validation_horizon,
                            fiscal_year_start_month=fiscal_year_start_month
                        )
                        if product not in backtesting_results:
                            backtesting_results[product] = {}
                        backtesting_results[product]["Seasonal-Naive"] = validation_results
                        diagnostic_messages.append(f"✅ **Seasonal-Naive Backtesting**: Successfully completed for {product}")
                except Exception as e:
                    error_msg = str(e)[:100]
                    diagnostic_messages.append(f"❌ **Seasonal-Naive Backtesting Failed**: {product} - Error: {error_msg}")
                    if product not in backtesting_results:
                        backtesting_results[product] = {}
                    backtesting_results[product]["Seasonal-Naive"] = None
        except Exception as e:
            diagnostic_messages.append(f"❌ Seasonal-Naive Product {product}: {str(e)[:50]}")
            _add_failed_metrics("Seasonal-Naive", mapes, smapes, mases, rmses)

    # Polynomial models (only Poly-2 allowed; Poly-3 disabled via flag)
    for poly_degree in [2]:
        model_name = f"Poly-{poly_degree}"
        if model_name in models_selected:
            diagnostic_messages.append(f"🔧 **Starting {model_name}**: Training {poly_degree}-degree polynomial model for {product}")
            done = _run_polynomial_model(
                product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
                diagnostic_messages, poly_degree, apply_business_adjustments,
                business_growth_assumption, market_multiplier,
                enable_backtesting,
                backtest_months, backtest_gap, validation_horizon, fiscal_year_start_month, backtesting_results, prog, done, total
            )
    
    # Prophet model (enabled by flag)
    if "Prophet" in models_selected and HAVE_PROPHET and ENABLE_PROPHET:
        diagnostic_messages.append(f"🔧 **Starting Prophet**: Training Facebook Prophet model for {product} (holidays: {enable_prophet_holidays})")
        done = _run_prophet_model(
            product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
            diagnostic_messages, enable_prophet_holidays, enable_statistical_validation,
            apply_business_adjustments, business_growth_assumption, market_multiplier,
            enable_backtesting,
            backtest_months, backtest_gap, validation_horizon, fiscal_year_start_month, backtesting_results, prog, done, total
        )
    
    # Auto-ARIMA model
    if "Auto-ARIMA" in models_selected and HAVE_PMDARIMA:
        diagnostic_messages.append(f"🔧 **Starting Auto-ARIMA**: Training automatic ARIMA model for {product}")
        done = _run_auto_arima_model(
            product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
            diagnostic_messages, enable_statistical_validation, apply_business_adjustments,
            business_growth_assumption, market_multiplier,
            enable_backtesting,
            backtest_months, backtest_gap, validation_horizon, fiscal_year_start_month, backtesting_results, prog, done, total
        )
    
    # LightGBM model (enabled by flag)
    if "LightGBM" in models_selected and HAVE_LGBM and ENABLE_LGBM:
        diagnostic_messages.append(f"🔧 **Starting LightGBM**: Training gradient boosting model for {product}")
        done = _run_lightgbm_model(
            product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
            diagnostic_messages, enable_statistical_validation, apply_business_adjustments,
            business_growth_assumption, market_multiplier,
            enable_backtesting,
            backtest_months, backtest_gap, validation_horizon, fiscal_year_start_month, backtesting_results, prog, done, total
        )
    
    return done


def _run_sarima_model(product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
                     sarima_params, diagnostic_messages, seasonality_strength, enable_statistical_validation,
                     apply_business_adjustments, business_growth_assumption, market_multiplier, market_conditions,
                     enable_backtesting, backtest_months, backtest_gap, validation_horizon,
                     fiscal_year_start_month,
                     backtesting_results, prog, done, total):
    """Run SARIMA model for a product."""
    try:
        # Find best SARIMA parameters
        best_model, selection_criterion, criterion_value, best_validation_mape, best_validation_smape, best_validation_mase, best_validation_rmse = fit_best_sarima(
            train, val, seasonality_strength
        )
        
        if best_model:
            # Store parameters with proper error handling
            try:
                # For SARIMAX models, try to access params differently
                if hasattr(best_model, 'specification'):
                    order = best_model.specification['order']  # type: ignore
                    seasonal_order = best_model.specification['seasonal_order']  # type: ignore
                elif hasattr(best_model, 'order'):
                    order = best_model.order  # type: ignore
                    seasonal_order = getattr(best_model, 'seasonal_order', (0, 0, 0, 12))
                else:
                    # Fallback for other model types
                    order = (1, 1, 1)
                    seasonal_order = (0, 0, 0, 12)
                best_params = (order, seasonal_order)
            except (AttributeError, KeyError, TypeError):
                order = (1, 1, 1)
                seasonal_order = (0, 0, 0, 12)
                best_params = (order, seasonal_order)
            
            # Compute validation forecast from the best (train-fitted) model for diagnostics
            try:
                if hasattr(best_model, 'get_forecast'):
                    pv_val = best_model.get_forecast(len(val)).predicted_mean  # type: ignore
                elif hasattr(best_model, 'forecast'):
                    pv_val = best_model.forecast(len(val))  # type: ignore
                else:
                    pv_val = None
            except Exception:
                pv_val = None

            sarima_params[product] = (order, seasonal_order, criterion_value, selection_criterion)
            mapes["SARIMA"].append(best_validation_mape)
            smapes["SARIMA"].append(best_validation_smape)
            mases["SARIMA"].append(best_validation_mase)
            rmses["SARIMA"].append(best_validation_rmse)
            
            # Fit final model on full series
            final_model = fit_final_sarima_model(series, best_params, seasonality_strength)
            
            if final_model:
                # Generate forecasts with proper error handling
                try:
                    if hasattr(final_model, 'get_forecast'):
                        forecast_result = final_model.get_forecast(len(future_idx))  # type: ignore
                        pf = forecast_result.predicted_mean  # type: ignore
                    elif hasattr(final_model, 'forecast'):
                        pf = final_model.forecast(len(future_idx))  # type: ignore
                    else:
                        # Fallback prediction method
                        pf = final_model.predict(n_periods=int(len(future_idx)))  # type: ignore
                except (AttributeError, TypeError) as e:
                    diagnostic_messages.append(f"❌ SARIMA Product {product}: Forecast generation failed - {str(e)}")
                    _add_failed_metrics("SARIMA", mapes, smapes, mases, rmses)
                    done += 1
                    prog.progress(min(done / total, 1.0), text=f"Running SARIMA on Product {product} ({done}/{total})")
                    return done
                
                # Apply post-processing
                pf = apply_trend_aware_forecasting(pf, series, len(series), "SARIMA", diagnostic_messages)
                pf = _maybe_apply_drift(series, pf, "SARIMA", product)
                if enable_statistical_validation:
                    pf = apply_statistical_validation(pf, series, "SARIMA")
                if apply_business_adjustments:
                    pf = apply_business_adjustments_to_forecast(pf, business_growth_assumption, market_multiplier)
                    diagnostic_messages.append(f"📈 SARIMA Product {product}: Applied business adjustments (Growth: {business_growth_assumption}%, Market: {market_conditions})")
                
                # Store results
                df_f = pd.DataFrame({"Product": product, "Date": future_idx, "ACR": pf, "Type": "forecast"})
                results["SARIMA"].append(pd.concat([act_df, df_f]))
                
                diagnostic_messages.append(f"✅ SARIMA Product {product}: Order {order}, Seasonal {seasonal_order}, {selection_criterion}: {criterion_value:.1f}, WAPE {best_validation_mape:.1%}")

                # Backtesting diagnostics for SARIMA
                if enable_backtesting and pv_val is not None:
                    try:
                        # Check if we have enough data for backtesting
                        if len(series) < (backtest_months + validation_horizon + backtest_gap + 12):
                            required_months = backtest_months + validation_horizon + backtest_gap + 12
                            diagnostic_messages.append(f"⚠️ **SARIMA Backtesting Skipped**: {product} - Insufficient data for {backtest_months} month backtesting (need {required_months} months, have {len(series)})")
                        else:
                            diagnostic_messages.append(f"🧪 **SARIMA Backtesting**: Starting backtesting for {product} with {backtest_months} months validation period")
                            sarima_fit_fn = create_sarima_fitting_function(order=order, seasonal_order=seasonal_order)
                            validation_results = comprehensive_validation_suite(
                                actual=val.values,
                                forecast=pv_val,
                                dates=val.index,
                                product_name=product,
                                series=series,
                                model_fitting_func=sarima_fit_fn,
                                model_params={},
                                diagnostic_messages=diagnostic_messages,
                                backtest_months=backtest_months,
                                backtest_gap=backtest_gap,
                                validation_horizon=validation_horizon,
                                fiscal_year_start_month=fiscal_year_start_month
                            )
                            # Expanding-window CV summary for selection
                            try:
                                cv = walk_forward_validation(
                                    series=series,
                                    model_fitting_func=sarima_fit_fn,
                                    window_size=24,
                                    step_size=validation_horizon,
                                    horizon=validation_horizon,
                                    model_params={},
                                    diagnostic_messages=diagnostic_messages,
                                    gap=backtest_gap
                                )
                                bt_dict = validation_results.get('backtesting_validation', {}) or {}
                                if cv:
                                    bt_dict.update({
                                        'success': True,
                                        'mape': cv.get('mean_mape'),
                                        'p75_mape': cv.get('p75_mape'),
                                        'p95_mape': cv.get('p95_mape'),
                                        'mase': cv.get('mean_mase'),
                                        'folds': cv.get('iterations'),
                                        'backtest_period': backtest_months,
                                        'validation_horizon': validation_horizon
                                    })
                                else:
                                    bt_dict.update({
                                        'success': False,
                                        'error': 'insufficient data for CV',
                                        'folds': 0,
                                        'backtest_period': backtest_months,
                                        'validation_horizon': validation_horizon
                                    })
                                validation_results['backtesting_validation'] = bt_dict
                            except Exception as e:
                                bt_dict = validation_results.get('backtesting_validation', {}) or {}
                                bt_dict.update({
                                    'success': False,
                                    'error': str(e)[:100],
                                    'folds': 0,
                                    'backtest_period': backtest_months,
                                    'validation_horizon': validation_horizon
                                })
                                validation_results['backtesting_validation'] = bt_dict
                            if product not in backtesting_results:
                                backtesting_results[product] = {}
                            backtesting_results[product]["SARIMA"] = validation_results
                            diagnostic_messages.append(f"✅ **SARIMA Backtesting**: Successfully completed backtesting for {product}")
                    except Exception as e:
                        error_msg = str(e)[:100]  # Get more of the error message
                        diagnostic_messages.append(f"❌ **SARIMA Backtesting Failed**: {product} - Error: {error_msg}")
                        if product not in backtesting_results:
                            backtesting_results[product] = {}
                        backtesting_results[product]["SARIMA"] = {
                            'backtesting_validation': {
                                'success': False,
                                'error': error_msg,
                                'backtest_period': backtest_months,
                                'validation_horizon': validation_horizon,
                                'folds': 0
                            }
                        }
            else:
                diagnostic_messages.append(f"❌ SARIMA Product {product}: Final model training failed")
                _add_failed_metrics("SARIMA", mapes, smapes, mases, rmses)
        else:
            diagnostic_messages.append(f"❌ SARIMA Product {product}: Failed to find suitable model")
            _add_failed_metrics("SARIMA", mapes, smapes, mases, rmses)
        
    except Exception as e:
        diagnostic_messages.append(f"❌ SARIMA Product {product}: {str(e)[:50]}")
        _add_failed_metrics("SARIMA", mapes, smapes, mases, rmses)
    
    done += 1
    prog.progress(min(done / total, 1.0), text=f"Running SARIMA on Product {product} ({done}/{total})")
    return done


def _run_ets_model(product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
                  diagnostic_messages, enable_statistical_validation, apply_business_adjustments,
                  business_growth_assumption, market_multiplier, enable_backtesting, 
                  backtest_months, backtest_gap, validation_horizon, fiscal_year_start_month, backtesting_results, 
                  prog, done, total):
    """Run ETS model for a product."""
    try:
        # Robust fitting via helper to avoid hard failures on seasonal mode
        best_ets_config = None
        pv = None
        # Try multiplicative, then additive, then no seasonality
        for seasonal_type_try in ("mul", "add", None):
            try:
                if seasonal_type_try is None:
                    model = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
                else:
                    model = ExponentialSmoothing(train, trend="add", seasonal=seasonal_type_try, seasonal_periods=12).fit()
                pv_try = model.forecast(len(val))
                m0, s0, ma0, r0 = calculate_validation_metrics(val, pv_try, train)
                if best_ets_config is None or (np.isfinite(m0) and m0 < best_ets_config[1]):
                    best_ets_config = (seasonal_type_try if seasonal_type_try is not None else "none", m0, s0, ma0, r0)
                    pv = pv_try
            except Exception:
                continue
        if best_ets_config is None or pv is None:
            raise RuntimeError("ETS fitting failed for all seasonal types")
        # Fit final model on full series using the chosen seasonal type
        seasonal_type = best_ets_config[0]
        if seasonal_type == "none":
            ets_final = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
        else:
            ets_final = ExponentialSmoothing(series, trend="add", seasonal=seasonal_type, seasonal_periods=12).fit()
        pf = ets_final.forecast(len(future_idx))
        # Record metrics
        mapes["ETS"].append(best_ets_config[1])
        smapes["ETS"].append(best_ets_config[2])
        mases["ETS"].append(best_ets_config[3])
        rmses["ETS"].append(best_ets_config[4])
        # Standard post-processing
        pf = apply_trend_aware_forecasting(pf, series, len(series), "ETS", diagnostic_messages)
        pf = _maybe_apply_drift(series, pf, "ETS", product)
        if enable_statistical_validation:
            pf = apply_statistical_validation(pf, series, "ETS")
        if apply_business_adjustments:
            pf = apply_business_adjustments_to_forecast(pf, business_growth_assumption, market_multiplier)
        # Backtesting to compute diagnostics (no forecast adjustments)
        if enable_backtesting:
            try:
                # Check if we have enough data for backtesting
                if len(series) < (backtest_months + validation_horizon + backtest_gap + 12):
                    required_months = backtest_months + validation_horizon + backtest_gap + 12
                    diagnostic_messages.append(f"⚠️ **ETS Backtesting Skipped**: {product} - Insufficient data for {backtest_months} month backtesting (need {required_months} months, have {len(series)})")
                else:
                    diagnostic_messages.append(f"🧪 **ETS Backtesting**: Starting backtesting for {product} with {backtest_months} months validation period")
                    ets_fitting_func = create_ets_fitting_function(seasonal_type)
                    validation_results = comprehensive_validation_suite(
                        actual=val.values,
                        forecast=pv,
                        dates=val.index,
                        product_name=product,
                        series=series,
                        model_fitting_func=ets_fitting_func,
                        model_params={},
                        diagnostic_messages=diagnostic_messages,
                        backtest_months=backtest_months,
                        backtest_gap=backtest_gap,
                        validation_horizon=validation_horizon,
                        fiscal_year_start_month=fiscal_year_start_month
                    )
                    # Expanding-window CV summary for selection
                    try:
                        cv = walk_forward_validation(
                            series=series,
                            model_fitting_func=ets_fitting_func,
                            window_size=24,
                            step_size=validation_horizon,
                            horizon=validation_horizon,
                            model_params={},
                            diagnostic_messages=diagnostic_messages,
                            gap=backtest_gap
                        )
                        bt_dict = validation_results.get('backtesting_validation', {}) or {}
                        if cv:
                            bt_dict.update({
                                'success': True,
                                'mape': cv.get('mean_mape'),
                                'p75_mape': cv.get('p75_mape'),
                                'p95_mape': cv.get('p95_mape'),
                                'mase': cv.get('mean_mase'),
                                'folds': cv.get('iterations'),
                                'backtest_period': backtest_months,
                                'validation_horizon': validation_horizon
                            })
                        else:
                            bt_dict.update({
                                'success': False,
                                'error': 'insufficient data for CV',
                                'folds': 0,
                                'backtest_period': backtest_months,
                                'validation_horizon': validation_horizon
                            })
                        validation_results['backtesting_validation'] = bt_dict
                    except Exception as e:
                        bt_dict = validation_results.get('backtesting_validation', {}) or {}
                        bt_dict.update({
                            'success': False,
                            'error': str(e)[:100],
                            'folds': 0,
                            'backtest_period': backtest_months,
                            'validation_horizon': validation_horizon
                        })
                        validation_results['backtesting_validation'] = bt_dict
                    if product not in backtesting_results:
                        backtesting_results[product] = {}
                    backtesting_results[product]["ETS"] = validation_results
                    diagnostic_messages.append(f"✅ **ETS Backtesting**: Successfully completed backtesting for {product}")
            except Exception as e:
                error_msg = str(e)[:100]  # Get more of the error message
                diagnostic_messages.append(f"❌ **ETS Backtesting Failed**: {product} - Error: {error_msg}")
                if product not in backtesting_results:
                    backtesting_results[product] = {}
                backtesting_results[product]["ETS"] = {
                    'backtesting_validation': {
                        'success': False,
                        'error': error_msg,
                        'backtest_period': backtest_months,
                        'validation_horizon': validation_horizon,
                        'folds': 0
                    }
                }
        # Store results after any corrections
        fore_df = pd.DataFrame({"Product": product, "Date": future_idx, "ACR": pf, "Type": "forecast"})
        results["ETS"].append(pd.concat([act_df, fore_df], ignore_index=True))
        diagnostic_messages.append(f"✅ ETS Product {product}: {seasonal_type} seasonal, WAPE {best_ets_config[1]:.1%}")
    except Exception as e:
        diagnostic_messages.append(f"❌ ETS Product {product}: {str(e)[:50]}")
        _add_failed_metrics("ETS", mapes, smapes, mases, rmses)
    done += 1
    prog.progress(min(done / total, 1.0), text=f"Running ETS on Product {product} ({done}/{total})")
    return done


def _run_polynomial_model(product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
                         diagnostic_messages, degree, apply_business_adjustments,
                         business_growth_assumption, market_multiplier,
                         enable_backtesting,
                         backtest_months, backtest_gap, validation_horizon, fiscal_year_start_month, backtesting_results, prog, done, total):
    """Run polynomial regression model for a product."""
    model_name = f"Poly-{degree}"
    
    try:
        # Prepare data for polynomial fitting
        X_train = np.arange(len(train)).reshape(-1, 1)
        X_val = np.arange(len(train), len(train) + len(val)).reshape(-1, 1)
        
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train)
        X_val_poly = poly_features.transform(X_val)
        
        # Fit and validate
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, train.values)
        pv = poly_model.predict(X_val_poly)
        
        val_mape, val_smape, val_mase, val_rmse = calculate_validation_metrics(val, pv, train)
        
        if val_mape < 2.0:  # Only proceed if reasonable validation performance
            mapes[model_name].append(val_mape)
            smapes[model_name].append(val_smape)
            mases[model_name].append(val_mase)
            rmses[model_name].append(val_rmse)
            
            # Fit final model on full series
            X_full = np.arange(len(series)).reshape(-1, 1)
            X_full_poly = poly_features.fit_transform(X_full)
            final_poly_model = LinearRegression()
            final_poly_model.fit(X_full_poly, np.asarray(series.values))
            
            # Generate forecasts
            X_forecast = np.arange(len(series), len(series) + len(future_idx)).reshape(-1, 1)
            X_forecast_poly = poly_features.transform(X_forecast)
            pf = final_poly_model.predict(X_forecast_poly)
            
            # Apply minimal constraints
            if series.min() >= 0:
                pf = np.maximum(pf, 0)
            
            # Apply business adjustments if enabled
            if apply_business_adjustments:
                pf = apply_business_adjustments_to_forecast(pf, business_growth_assumption, market_multiplier)
            
            # Store results
            fore_df = pd.DataFrame({"Product": product, "Date": future_idx, "ACR": pf, "Type": "forecast"})
            results[model_name].append(pd.concat([act_df, fore_df], ignore_index=True))
            
            diagnostic_messages.append(f"✅ {model_name} Product {product}: WAPE {val_mape:.1%} (pure polynomial)")

            # Backtesting diagnostics for Polynomial
            if enable_backtesting:
                try:
                    # Check if we have enough data for backtesting
                    if len(series) < (backtest_months + validation_horizon + backtest_gap + 12):
                        required_months = backtest_months + validation_horizon + backtest_gap + 12
                        diagnostic_messages.append(f"⚠️ **{model_name} Backtesting Skipped**: {product} - Insufficient data for {backtest_months} month backtesting (need {required_months} months, have {len(series)})")
                    else:
                        diagnostic_messages.append(f"🧪 **{model_name} Backtesting**: Starting backtesting for {product} with {backtest_months} months validation period")
                        poly_fit_fn = create_polynomial_fitting_function(degree=degree)
                        validation_results = comprehensive_validation_suite(
                            actual=val.values,
                            forecast=pv,
                            dates=val.index,
                            product_name=product,
                            # Simple backtesting only
                            series=series,
                            model_fitting_func=poly_fit_fn,
                            model_params={},
                            diagnostic_messages=diagnostic_messages,
                            backtest_months=backtest_months,
                            backtest_gap=backtest_gap,
                            validation_horizon=validation_horizon,
                            fiscal_year_start_month=fiscal_year_start_month
                        )
                        if product not in backtesting_results:
                            backtesting_results[product] = {}
                        backtesting_results[product][model_name] = validation_results
                        diagnostic_messages.append(f"✅ **{model_name} Backtesting**: Successfully completed backtesting for {product}")
                except Exception as e:
                    error_msg = str(e)[:100]  # Get more of the error message
                    diagnostic_messages.append(f"❌ **{model_name} Backtesting Failed**: {product} - Error: {error_msg}")
                    if product not in backtesting_results:
                        backtesting_results[product] = {}
                    backtesting_results[product][model_name] = None
        else:
            diagnostic_messages.append(f"❌ {model_name} Product {product}: Poor fit (WAPE {val_mape:.1%}), skipping")
            _add_failed_metrics(model_name, mapes, smapes, mases, rmses)
        
    except Exception as e:
        diagnostic_messages.append(f"❌ {model_name} Product {product}: {str(e)[:50]}")
        _add_failed_metrics(model_name, mapes, smapes, mases, rmses)
    
    done += 1
    prog.progress(min(done / total, 1.0), text=f"Running {model_name} on Product {product} ({done}/{total})")
    return done


def _run_prophet_model(product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
                      diagnostic_messages, enable_prophet_holidays, enable_statistical_validation,
                      apply_business_adjustments, business_growth_assumption, market_multiplier,
                      enable_backtesting,
                      backtest_months, backtest_gap, validation_horizon, fiscal_year_start_month, backtesting_results, prog, done, total):
    """Run Prophet model for a product."""
    try:
        # Setup holidays if enabled
        holidays_df = None
        if enable_prophet_holidays:
            try:
                from prophet.make_holidays import make_holidays_df
                years = list(range(2010, 2031))
                us_holidays = make_holidays_df(year_list=years, country='US')
                worldwide_holidays = make_holidays_df(year_list=years, country=None)
                holidays_df = pd.concat([us_holidays, worldwide_holidays]).drop_duplicates(subset=['ds']).reset_index(drop=True)
                holidays_df['holiday'] = holidays_df['holiday'].astype(str)
                diagnostic_messages.append(f"📅 Prophet Product {product}: Added {len(holidays_df)} holiday effects")
            except Exception:
                diagnostic_messages.append(f"⚠️ Prophet Product {product}: Could not load holidays, proceeding without them")
                holidays_df = None
        
        # Prepare validation data
        prophet_df = pd.DataFrame({'ds': train.index, 'y': train.values})
        
        # Check if Prophet is available
        if Prophet is None:
            diagnostic_messages.append(f"❌ Prophet Product {product}: Prophet not available")
            _add_failed_metrics("Prophet", mapes, smapes, mases, rmses)
            return done + 1
        
        # Fit and validate
        prophet_model = Prophet(
            yearly_seasonality=True,  # type: ignore
            weekly_seasonality=False,  # type: ignore
            daily_seasonality=False,  # type: ignore
            changepoint_prior_scale=0.05,
            holidays=holidays_df
        )
        prophet_model.fit(prophet_df)
        
        val_future = pd.DataFrame({'ds': val.index})
        val_forecast = prophet_model.predict(val_future)
        pv = val_forecast['yhat'].values
        
        val_mape, val_smape, val_mase, val_rmse = calculate_validation_metrics(val, pv, train)
        mapes["Prophet"].append(val_mape)
        smapes["Prophet"].append(val_smape)
        mases["Prophet"].append(val_mase)
        rmses["Prophet"].append(val_rmse)
        
        # Fit final model and forecast
        full_prophet_df = pd.DataFrame({'ds': series.index, 'y': series.values})
        final_prophet_model = Prophet(
            yearly_seasonality=True,  # type: ignore
            weekly_seasonality=False,  # type: ignore
            daily_seasonality=False,  # type: ignore
            changepoint_prior_scale=0.05,
            holidays=holidays_df
        )
        final_prophet_model.fit(full_prophet_df)
        
        future_df = pd.DataFrame({'ds': future_idx})
        forecast = final_prophet_model.predict(future_df)
        pf = forecast['yhat'].values
        
        # Apply post-processing
        pf = apply_trend_aware_forecasting(pf, series, len(series), "Prophet", diagnostic_messages)
        pf = _maybe_apply_drift(series, pf, "Prophet", product)
        if enable_statistical_validation:
            pf = apply_statistical_validation(pf, series, "Prophet")
        if apply_business_adjustments:
            pf = apply_business_adjustments_to_forecast(pf, business_growth_assumption, market_multiplier)
        
        # Store results
        fore_df = pd.DataFrame({"Product": product, "Date": future_idx, "ACR": pf, "Type": "forecast"})
        results["Prophet"].append(pd.concat([act_df, fore_df], ignore_index=True))
        
        holiday_status = "with holidays" if enable_prophet_holidays and holidays_df is not None else "no holidays"
        diagnostic_messages.append(f"✅ Prophet Product {product}: WAPE {val_mape:.1%} ({holiday_status})")

        # Backtesting diagnostics for Prophet
        if enable_backtesting and HAVE_PROPHET:
            try:
                # Check if we have enough data for backtesting
                if len(series) < (backtest_months + validation_horizon + backtest_gap + 12):
                    required_months = backtest_months + validation_horizon + backtest_gap + 12
                    diagnostic_messages.append(f"⚠️ **Prophet Backtesting Skipped**: {product} - Insufficient data for {backtest_months} month backtesting (need {required_months} months, have {len(series)})")
                else:
                    diagnostic_messages.append(f"🧪 **Prophet Backtesting**: Starting backtesting for {product} with {backtest_months} months validation period")
                    prophet_fit_fn = create_prophet_fitting_function(enable_holidays=bool(holidays_df is not None))
                    validation_results = comprehensive_validation_suite(
                        actual=val.values,
                        forecast=pv,
                        dates=val.index,
                        product_name=product,
                        series=series,
                        model_fitting_func=prophet_fit_fn,
                        model_params={},
                        diagnostic_messages=diagnostic_messages,
                        backtest_months=backtest_months,
                        backtest_gap=backtest_gap,
                        validation_horizon=validation_horizon,
                        fiscal_year_start_month=fiscal_year_start_month
                    )
                    if product not in backtesting_results:
                        backtesting_results[product] = {}
                    backtesting_results[product]["Prophet"] = validation_results
                    diagnostic_messages.append(f"✅ **Prophet Backtesting**: Successfully completed backtesting for {product}")
                    # Add expanding-window CV summary for selection
                    try:
                        prophet_fit_fn_cv = create_prophet_fitting_function(enable_holidays=bool(holidays_df is not None))
                        cv = walk_forward_validation(
                            series=series,
                            model_fitting_func=prophet_fit_fn_cv,
                            window_size=24,
                            step_size=validation_horizon,
                            horizon=validation_horizon,
                            model_params={},
                            diagnostic_messages=diagnostic_messages,
                            gap=backtest_gap
                        )
                        if cv:
                            backtesting_results[product]["Prophet"]["backtesting_validation"] = {
                                'success': True,
                                'mape': cv.get('mean_mape'),
                                'p75_mape': cv.get('p75_mape'),
                                'p95_mape': cv.get('p95_mape'),
                                'mase': cv.get('mean_mase'),
                                'folds': cv.get('iterations'),
                                'backtest_period': backtest_months,
                                'validation_horizon': validation_horizon
                            }
                        else:
                            backtesting_results[product]["Prophet"]["backtesting_validation"] = {
                                'success': False,
                                'error': 'insufficient data for CV',
                                'folds': 0,
                                'backtest_period': backtest_months,
                                'validation_horizon': validation_horizon
                            }
                    except Exception as e:
                        backtesting_results[product]["Prophet"]["backtesting_validation"] = {
                            'success': False,
                            'error': str(e)[:100],
                            'folds': 0,
                            'backtest_period': backtest_months,
                            'validation_horizon': validation_horizon
                        }
            except Exception as e:
                error_msg = str(e)[:100]  # Get more of the error message
                if "prophet not available" in error_msg.lower():
                    diagnostic_messages.append(f"⚠️ **Prophet Backtesting Skipped**: {product} - Prophet package not available")
                else:
                    diagnostic_messages.append(f"❌ **Prophet Backtesting Failed**: {product} - Error: {error_msg}")
                    if product not in backtesting_results:
                        backtesting_results[product] = {}
                    backtesting_results[product]["Prophet"] = {'backtesting_validation': {'success': False, 'error': error_msg, 'backtest_period': backtest_months, 'validation_horizon': validation_horizon, 'folds': 0}}
        elif enable_backtesting and not HAVE_PROPHET:
            diagnostic_messages.append(f"⚠️ **Prophet Backtesting Skipped**: {product} - Prophet package not available")
        
    except Exception as e:
        diagnostic_messages.append(f"❌ Prophet Product {product}: {str(e)[:50]}")
        _add_failed_metrics("Prophet", mapes, smapes, mases, rmses)
    
    done += 1
    prog.progress(min(done / total, 1.0), text=f"Running Prophet on Product {product} ({done}/{total})")
    return done


def _run_auto_arima_model(product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
                         diagnostic_messages, enable_statistical_validation, apply_business_adjustments,
                         business_growth_assumption, market_multiplier,
                         enable_backtesting,
                         backtest_months, backtest_gap, validation_horizon, fiscal_year_start_month, backtesting_results, prog, done, total):
    """Run Auto-ARIMA model for a product."""
    try:
        # Check if auto_arima is available
        if auto_arima is None:
            diagnostic_messages.append(f"❌ Auto-ARIMA Product {product}: pmdarima not available")
            _add_failed_metrics("Auto-ARIMA", mapes, smapes, mases, rmses)
            return done + 1

        # Get auto_arima function
        auto_arima_func = auto_arima

        # Fit model on training window
        auto_model = auto_arima_func(
            train,
            seasonal=True,
            m=12,
            max_p=3, max_d=2, max_q=3,
            max_P=2, max_D=1, max_Q=2,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )

        # Validate on holdout
        pv = auto_model.predict(len(val))
        val_mape, val_smape, val_mase, val_rmse = calculate_validation_metrics(val, pv, train)
        mapes["Auto-ARIMA"].append(val_mape)
        smapes["Auto-ARIMA"].append(val_smape)
        mases["Auto-ARIMA"].append(val_mase)
        rmses["Auto-ARIMA"].append(val_rmse)

        # Fit final model and forecast into the future horizon
        auto_model_full = auto_arima_func(
            series,
            seasonal=True,
            m=12,
            max_p=3, max_d=2, max_q=3,
            max_P=2, max_D=1, max_Q=2,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )

        pf = auto_model_full.predict(len(future_idx))

        # Apply post-processing
        pf = apply_trend_aware_forecasting(pf, series, len(series), "Auto-ARIMA", diagnostic_messages)
        pf = _maybe_apply_drift(series, pf, "Auto-ARIMA", product)
        if enable_statistical_validation:
            pf = apply_statistical_validation(pf, series, "Auto-ARIMA")
        if apply_business_adjustments:
            pf = apply_business_adjustments_to_forecast(pf, business_growth_assumption, market_multiplier)

        # Store results
        fore_df = pd.DataFrame({"Product": product, "Date": future_idx, "ACR": pf, "Type": "forecast"})
        results["Auto-ARIMA"].append(pd.concat([act_df, fore_df], ignore_index=True))

        model_order = auto_model_full.order
        seasonal_order = auto_model_full.seasonal_order
        diagnostic_messages.append(
            f"✅ Auto-ARIMA Product {product}: Order {model_order}, Seasonal {seasonal_order}, WAPE {val_mape:.1%}"
        )

        # Backtesting diagnostics for Auto-ARIMA
        if enable_backtesting:
            try:
                # Check if we have enough data for backtesting
                if len(series) < (backtest_months + validation_horizon + backtest_gap + 12):
                    required_months = backtest_months + validation_horizon + backtest_gap + 12
                    diagnostic_messages.append(f"⚠️ **Auto-ARIMA Backtesting Skipped**: {product} - Insufficient data for {backtest_months} month backtesting (need {required_months} months, have {len(series)})")
                else:
                    diagnostic_messages.append(f"🧪 **Auto-ARIMA Backtesting**: Starting backtesting for {product} with {backtest_months} months validation period")
                    aa_fit_fn = create_auto_arima_fitting_function()
                    validation_results = comprehensive_validation_suite(
                        actual=val.values,
                        forecast=pv,
                        dates=val.index,
                        product_name=product,
                        # Simple backtesting only
                        series=series,
                        model_fitting_func=aa_fit_fn,
                        model_params={},
                        diagnostic_messages=diagnostic_messages,
                        backtest_months=backtest_months,
                        backtest_gap=backtest_gap,
                        validation_horizon=validation_horizon,
                        fiscal_year_start_month=fiscal_year_start_month
                    )
                    if product not in backtesting_results:
                        backtesting_results[product] = {}
                    backtesting_results[product]["Auto-ARIMA"] = validation_results
                    diagnostic_messages.append(f"✅ **Auto-ARIMA Backtesting**: Successfully completed backtesting for {product}")
                    # Add expanding-window CV summary for selection
                    try:
                        aa_fit_fn_cv = create_auto_arima_fitting_function()
                        cv = walk_forward_validation(
                            series=series,
                            model_fitting_func=aa_fit_fn_cv,
                            window_size=24,
                            step_size=validation_horizon,
                            horizon=validation_horizon,
                            model_params={},
                            diagnostic_messages=diagnostic_messages,
                            gap=backtest_gap
                        )
                        if cv:
                            backtesting_results[product]["Auto-ARIMA"]["backtesting_validation"] = {
                                'success': True,
                                'mape': cv.get('mean_mape'),
                                'p75_mape': cv.get('p75_mape'),
                                'p95_mape': cv.get('p95_mape'),
                                'mase': cv.get('mean_mase'),
                                'folds': cv.get('iterations'),
                                'backtest_period': backtest_months,
                                'validation_horizon': validation_horizon
                            }
                        else:
                            backtesting_results[product]["Auto-ARIMA"]["backtesting_validation"] = {
                                'success': False,
                                'error': 'insufficient data for CV',
                                'folds': 0,
                                'backtest_period': backtest_months,
                                'validation_horizon': validation_horizon
                            }
                    except Exception as e:
                        backtesting_results[product]["Auto-ARIMA"]["backtesting_validation"] = {
                            'success': False,
                            'error': str(e)[:100],
                            'folds': 0,
                            'backtest_period': backtest_months,
                            'validation_horizon': validation_horizon
                        }
            except Exception as e:
                error_msg = str(e)[:100]  # Get more of the error message
                diagnostic_messages.append(f"❌ **Auto-ARIMA Backtesting Failed**: {product} - Error: {error_msg}")
                if product not in backtesting_results:
                    backtesting_results[product] = {}
                backtesting_results[product]["Auto-ARIMA"] = {'backtesting_validation': {'success': False, 'error': error_msg, 'backtest_period': backtest_months, 'validation_horizon': validation_horizon, 'folds': 0}}

    except Exception as e:
        diagnostic_messages.append(f"❌ Auto-ARIMA Product {product}: {str(e)[:50]}")
        _add_failed_metrics("Auto-ARIMA", mapes, smapes, mases, rmses)

    done += 1
    prog.progress(min(done / total, 1.0), text=f"Running Auto-ARIMA on Product {product} ({done}/{total})")
    return done


def _run_lightgbm_model(product, series, train, val, future_idx, act_df, results, mapes, smapes, mases, rmses,
                       diagnostic_messages, enable_statistical_validation, apply_business_adjustments,
                       business_growth_assumption, market_multiplier,
                       enable_backtesting,
                       backtest_months, backtest_gap, validation_horizon, fiscal_year_start_month, backtesting_results, prog, done, total):
    """Run LightGBM model for a product."""
    try:
        # Leak-safe LightGBM fitting (uses new signature)
        best_lgbm, best_params, best_mape, best_smape, best_mase, best_rmse, val_forecast = fit_best_lightgbm(
            train, val, diagnostic_messages
        )
        if best_lgbm is None or not np.isfinite(best_mape):
            _add_failed_metrics("LightGBM", mapes, smapes, mases, rmses)
        else:
            mapes["LightGBM"].append(best_mape)
            smapes["LightGBM"].append(best_smape)
            mases["LightGBM"].append(best_mase)
            rmses["LightGBM"].append(best_rmse)

            # Refit on full series (train+val) before forward forecasting
            if LGBMRegressor is None:
                diagnostic_messages.append(f"❌ LightGBM Product {product}: LightGBM not available")
                _add_failed_metrics("LightGBM", mapes, smapes, mases, rmses)
            else:
                # Build features on entire historical series
                def build_features(s):
                    df = pd.DataFrame({'ACR': s.values}, index=s.index)
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
                    return df.dropna()

                hist_feats = build_features(series).dropna()
                feat_cols = [c for c in hist_feats.columns if c != 'ACR']
                final_lgbm = LGBMRegressor(
                    n_estimators=int(best_params.get("n_estimators", 100)),
                    learning_rate=float(best_params.get("learning_rate", 0.1)),
                    num_leaves=int(best_params.get("num_leaves", 31)),
                    max_depth=int(best_params.get("max_depth", 6)),
                    random_state=42,
                    verbose=-1,
                    force_col_wise=True
                )
                final_lgbm.fit(hist_feats[feat_cols], hist_feats['ACR'])

                # Iterative future forecasting using own predictions
                future_preds = []
                temp_series = series.copy()
                for _ in range(len(future_idx)):
                    feats = build_features(temp_series)
                    if len(feats) == 0:
                        break
                    x_last = feats[feat_cols].iloc[-1:]
                    pred_raw = final_lgbm.predict(x_last)
                    yhat = float(np.asarray(pred_raw).ravel()[0])
                    future_preds.append(yhat)
                    next_date = pd.Timestamp(str(temp_series.index[-1])) + pd.DateOffset(months=1)
                    temp_series = pd.concat([temp_series, pd.Series([yhat], index=[next_date])])

                if future_preds:
                    pf = np.array(future_preds)
                    pf = apply_trend_aware_forecasting(pf, series, len(series), "LightGBM", diagnostic_messages)
                    pf = _maybe_apply_drift(series, pf, "LightGBM", product)
                    if enable_statistical_validation:
                        pf = apply_statistical_validation(pf, series, "LightGBM")
                    if apply_business_adjustments:
                        pf = apply_business_adjustments_to_forecast(pf, business_growth_assumption, market_multiplier)
                    fore_df = pd.DataFrame({"Product": product, "Date": future_idx[:len(pf)], "ACR": pf, "Type": "forecast"})
                    results["LightGBM"].append(pd.concat([act_df, fore_df], ignore_index=True))
                    diagnostic_messages.append(f"✅ LightGBM Product {product}: WAPE {best_mape:.1%} (leak-safe)")

                    if enable_backtesting and val_forecast is not None:
                        try:
                            # Check if we have enough data for backtesting
                            if len(series) < (backtest_months + validation_horizon + backtest_gap + 12):
                                required_months = backtest_months + validation_horizon + backtest_gap + 12
                                diagnostic_messages.append(f"⚠️ **LightGBM Backtesting Skipped**: {product} - Insufficient data for {backtest_months} month backtesting (need {required_months} months, have {len(series)})")
                            else:
                                diagnostic_messages.append(f"🧪 **LightGBM Backtesting**: Starting backtesting for {product} with {backtest_months} months validation period")
                                # Simple fitting function reusing leak-safe approach for backtests
                                def lgbm_fit_fn(train_ts, **kwargs):
                                    return fit_best_lightgbm(train_ts.iloc[:-12], train_ts.iloc[-12:], None)[0] if len(train_ts) > 24 else None
                                validation_results = comprehensive_validation_suite(
                                    actual=val.values,
                                    forecast=val_forecast,
                                    dates=val.index,
                                    product_name=product,
                                    # Simple backtesting only
                                    series=series,
                                    model_fitting_func=None,  # For advanced we already have robust metrics; avoid recursion
                                    model_params={},
                                    diagnostic_messages=diagnostic_messages,
                                    backtest_months=backtest_months,
                                    backtest_gap=backtest_gap,
                                    validation_horizon=validation_horizon,
                                    fiscal_year_start_month=fiscal_year_start_month
                                )
                                if product not in backtesting_results:
                                    backtesting_results[product] = {}
                                backtesting_results[product]["LightGBM"] = validation_results
                                diagnostic_messages.append(f"✅ **LightGBM Backtesting**: Successfully completed backtesting for {product}")
                        except Exception as e:
                            error_msg = str(e)[:100]  # Get more of the error message
                            diagnostic_messages.append(f"❌ **LightGBM Backtesting Failed**: {product} - Error: {error_msg}")
                            if product not in backtesting_results:
                                backtesting_results[product] = {}
                            backtesting_results[product]["LightGBM"] = None
                else:
                    _add_failed_metrics("LightGBM", mapes, smapes, mases, rmses)
        
    except Exception as e:
        diagnostic_messages.append(f"❌ LightGBM Product {product}: {str(e)[:50]}")
        _add_failed_metrics("LightGBM", mapes, smapes, mases, rmses)
    
    done += 1
    prog.progress(min(done / total, 1.0), text=f"Running LightGBM on Product {product} ({done}/{total})")
    return done


def _add_failed_metrics(model_name, mapes, smapes, mases, rmses):
    """Add default metrics for failed models."""
    mapes[model_name].append(1.0)
    smapes[model_name].append(1.0)
    mases[model_name].append(np.nan)
    rmses[model_name].append(np.nan)


def _process_model_results(results):
    """Process and clean model results."""
    for key in list(results.keys()):
        if results[key]:
            df_m = pd.concat(results[key])
            df_m["FiscalYear"] = df_m["Date"].apply(fy)
            results[key] = df_m
        else:
            # Remove empty models
            results.pop(key)
    return results


def _calculate_average_metrics(mapes, smapes, mases, rmses, results, products):
    """Calculate average metrics and model rankings."""
    # Calculate average metrics
    avg_mapes = {m: np.mean(mapes[m]) for m in mapes if mapes[m]}
    avg_smapes = {m: np.mean(smapes[m]) for m in smapes if smapes[m]}
    avg_mases = {m: np.mean(mases[m]) for m in mases if mases[m] and not np.isnan(mases[m]).all()}
    avg_rmses = {m: np.mean(rmses[m]) for m in rmses if rmses[m] and not np.isnan(rmses[m]).all()}
    
    # Calculate model rankings
    model_names = list(results.keys())
    metric_ranks = {m: {model: 0 for model in model_names} for m in ["MAPE", "SMAPE", "MASE", "RMSE"]}
    
    # For each product, rank models for each metric
    product_mapes = _create_product_metrics_dict(mapes, products, model_names)
    product_smapes = _create_product_metrics_dict(smapes, products, model_names)
    product_mases = _create_product_metrics_dict(mases, products, model_names)
    product_rmses = _create_product_metrics_dict(rmses, products, model_names)
    
    for product in products:
        # Rank each metric
        for metric_name, product_metrics in [("MAPE", product_mapes), ("SMAPE", product_smapes), 
                                            ("MASE", product_mases), ("RMSE", product_rmses)]:
            metric_vals = {model: product_metrics[model].get(product, np.nan) for model in model_names}
            metric_sorted = sorted((v, k) for k, v in metric_vals.items() if not np.isnan(v))
            
            for rank, (v, k) in enumerate(metric_sorted):
                metric_ranks[metric_name][k] += rank + 1
    
    # Compute average rank for each model
    avg_ranks = {model: np.mean([metric_ranks[m][model] for m in metric_ranks]) for model in model_names}
    
    return avg_mapes, avg_smapes, avg_mases, avg_rmses, avg_ranks


def _create_product_metrics_dict(metrics_by_model, products, model_names):
    """Create product-level metrics dictionary."""
    product_metrics = {}
    
    for model_name in model_names:
        product_metrics[model_name] = {}
        
        if model_name in metrics_by_model:
            for i, product in enumerate(products):
                if i < len(metrics_by_model[model_name]):
                    product_metrics[model_name][product] = metrics_by_model[model_name][i]
                else:
                    product_metrics[model_name][product] = np.nan
    
    return product_metrics


def _find_best_models_per_product(products, results, mapes, smapes, mases, rmses,
                                 enable_business_aware_selection, diagnostic_messages):
    """Find the best model for each product using multi-metric ranking.
    
    ROBUSTNESS IMPROVEMENT: This function now operates on metrics that were calculated
    using a single validation split. For even more robust selection, consider implementing
    multiple validation windows in the future (see _find_best_models_robust_validation).
    """
    best_models_per_product = {}
    best_mapes_per_product = {}
    model_names = list(results.keys())
    
    # Create product metrics dictionaries
    product_mapes = _create_product_metrics_dict(mapes, products, model_names)
    product_smapes = _create_product_metrics_dict(smapes, products, model_names)
    product_mases = _create_product_metrics_dict(mases, products, model_names)
    product_rmses = _create_product_metrics_dict(rmses, products, model_names)
    
    for product in products:
        # Calculate per-product rankings across all 4 metrics
        product_model_metrics = {}
        product_model_ranks = {model: 0 for model in model_names}
        
        for model_name in model_names:
            if product in product_mapes[model_name]:
                product_model_metrics[model_name] = {
                    'MAPE': product_mapes[model_name][product],
                    'SMAPE': product_smapes[model_name].get(product, np.nan),
                    'MASE': product_mases[model_name].get(product, np.nan),
                    'RMSE': product_rmses[model_name].get(product, np.nan)
                }
        
        if product_model_metrics:
            # Calculate rankings for this specific product across all metrics
            for metric in ['MAPE', 'SMAPE', 'MASE', 'RMSE']:
                metric_vals = {model: metrics[metric] for model, metrics in product_model_metrics.items() 
                             if not np.isnan(metrics[metric])}
                if metric_vals:
                    metric_sorted = sorted(metric_vals.items(), key=lambda x: x[1])
                    for rank, (model, _) in enumerate(metric_sorted):
                        product_model_ranks[model] += rank + 1
            
            # BIAS MITIGATION: Add warning about single validation split
            if diagnostic_messages and len(product_model_metrics) > 1:
                diagnostic_messages.append(
                    f"📊 Product {product}: Model selection based on single validation split. "
                    f"Consider enabling advanced validation for more robust selection."
                )
            
            # Select model with best average rank for this product
            if enable_business_aware_selection:
                # Apply business-aware filtering to ranked candidates
                product_model_mapes_dict = {model: product_mapes[model][product] 
                                          for model in product_model_metrics.keys()}
                # Pass ranking information for smarter business-aware selection
                valid_ranks = {model: rank/4 for model, rank in product_model_ranks.items() if rank > 0}  # Average rank
                best_model_for_product, best_mape_for_product = select_business_aware_best_model(
                    product_model_mapes_dict, product, diagnostic_messages, valid_ranks
                )
            else:
                # Pure multi-metric ranking selection
                valid_ranks = {model: rank for model, rank in product_model_ranks.items() if rank > 0}
                if valid_ranks:
                    best_model_for_product = min(valid_ranks.keys(), key=lambda k: valid_ranks[k])
                    best_mape_for_product = product_mapes[best_model_for_product][product]
                    
                    if diagnostic_messages:
                        avg_rank = valid_ranks[best_model_for_product] / 4  # Average across 4 metrics
                        diagnostic_messages.append(
                            f"📊 Product {product}: Multi-metric ranking selected {best_model_for_product} "
                            f"(Avg Rank: {avg_rank:.1f}, MAPE: {best_mape_for_product:.1%})"
                        )
                else:
                    # Fallback to MAPE if ranking fails
                    product_model_mapes_dict = {model: product_mapes[model][product] 
                                              for model in product_model_metrics.keys()}
                    best_model_for_product = min(product_model_mapes_dict.keys(), 
                                               key=lambda k: product_model_mapes_dict[k])
                    best_mape_for_product = product_model_mapes_dict[best_model_for_product]
            
            best_models_per_product[product] = best_model_for_product
            best_mapes_per_product[product] = best_mape_for_product
        else:
            # Fallback to first available model
            best_models_per_product[product] = list(results.keys())[0]
            best_mapes_per_product[product] = 1.0
    
    return best_models_per_product, best_mapes_per_product


def _create_hybrid_model(results, best_models_per_product, avg_mapes, best_mapes_per_product, model_key_name: str = "Best per Product"):
    """Create hybrid 'Best per Product' model combining best forecasts for each product.

    model_key_name: name of the hybrid model key in results
    """
    products = list(best_models_per_product.keys())
    hybrid_results = []
    
    for product in products:
        best_model_name = best_models_per_product[product]
        if best_model_name in results:
            product_data = results[best_model_name]
            product_specific_data = product_data[product_data["Product"] == product].copy()
            if not product_specific_data.empty:
                product_specific_data["BestModel"] = best_model_name  # Add metadata
                hybrid_results.append(product_specific_data)
        else:
            # If the chosen model is missing, fall back to any available model's data for this product
            try:
                for alt_model, df in results.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        alt_data = df[df["Product"] == product].copy()
                        if not alt_data.empty:
                            alt_data["BestModel"] = alt_model
                            hybrid_results.append(alt_data)
                            break
            except Exception:
                pass
    
    if hybrid_results:
        hybrid_df = pd.concat(hybrid_results, ignore_index=True)
        hybrid_df["FiscalYear"] = hybrid_df["Date"].apply(fy)
        results[model_key_name] = hybrid_df
        
        # Calculate average MAPE for hybrid model, excluding infinite values
        mape_values = list(best_mapes_per_product.values())
        finite_mapes = [m for m in mape_values if np.isfinite(m)]
        if finite_mapes:
            hybrid_avg_mape = np.mean(finite_mapes)
        else:
            # If all individual product MAPEs are infinite, use a high but finite value
            hybrid_avg_mape = 1.0  # 100% WAPE as fallback
        avg_mapes[model_key_name] = hybrid_avg_mape
    
    return results
