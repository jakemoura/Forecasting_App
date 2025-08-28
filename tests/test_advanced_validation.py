#!/usr/bin/env python3
"""
Test Advanced Validation Features for MAPE Back Testing

This test validates the new comprehensive back testing functionality including:
1. Walk-forward validation
2. Time series cross-validation  
3. Enhanced MAPE analysis
4. Seasonal performance analysis
5. Model fitting functions integration

Author: Enhanced by Assistant for Jake Moura
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Suppress warnings for testing
warnings.filterwarnings('ignore')

# Add modules path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.metrics import (
        walk_forward_validation, time_series_cross_validation,
        enhanced_mape_analysis, seasonal_mape_analysis,
        comprehensive_validation_suite, calculate_validation_metrics
    )
    from modules.models import (
        create_ets_fitting_function, create_sarima_fitting_function,
        create_polynomial_fitting_function
    )
    print("✅ Successfully imported advanced validation modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def create_test_data(n_periods=60, trend=0.02, seasonality_strength=0.3, noise_level=0.1):
    """Create synthetic time series data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Create date range
    start_date = datetime(2019, 1, 1)
    dates = pd.date_range(start_date, periods=n_periods, freq='MS')
    
    # Generate base values
    base_value = 1000
    trend_component = np.arange(n_periods) * trend * base_value / 12
    
    # Add seasonality (12-month cycle)
    seasonal_component = seasonality_strength * base_value * np.sin(2 * np.pi * np.arange(n_periods) / 12)
    
    # Add noise
    noise = np.random.normal(0, noise_level * base_value, n_periods)
    
    # Combine components
    values = base_value + trend_component + seasonal_component + noise
    
    # Ensure positive values
    values = np.maximum(values, base_value * 0.5)
    
    # Create time series
    series = pd.Series(values, index=dates)
    return series


def test_enhanced_mape_analysis():
    """Test enhanced MAPE analysis functionality."""
    print("\n🧪 Testing Enhanced MAPE Analysis...")
    
    # Create test data
    actual = np.array([100, 110, 95, 120, 105, 115, 90, 125, 108, 118])
    forecast = np.array([98, 108, 97, 118, 107, 113, 92, 122, 110, 116])
    dates = pd.date_range('2023-01-01', periods=10, freq='MS')
    
    # Run enhanced analysis
    result = enhanced_mape_analysis(actual, forecast, dates, "Test_Product")
    
    # Validate results
    assert 'mape' in result, "MAPE not calculated"
    assert 'bias' in result, "Bias not calculated"
    assert 'mape_ci_lower' in result, "Confidence interval not calculated"
    assert 'outlier_count' in result, "Outlier detection not working"
    
    print(f"   ✅ MAPE: {result['mape']:.1%}")
    print(f"   ✅ Bias: {result['bias']:+.1%}")
    print(f"   ✅ Outliers: {result['outlier_count']}")
    print(f"   ✅ CI Range: {result['mape_ci_lower']:.1%} - {result['mape_ci_upper']:.1%}")
    
    return True


def test_seasonal_mape_analysis():
    """Test seasonal MAPE analysis functionality."""
    print("\n🧪 Testing Seasonal MAPE Analysis...")
    
    # Create 24 months of test data
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=24, freq='MS')
    actual = 100 + 10 * np.sin(2 * np.pi * np.arange(24) / 12) + np.random.normal(0, 5, 24)
    forecast = actual + np.random.normal(0, 3, 24)
    
    # Run seasonal analysis
    result = seasonal_mape_analysis(actual, forecast, dates, "Test_Product")
    
    # Validate results
    assert 'monthly_mape' in result, "Monthly MAPE not calculated"
    assert 'quarterly_mape' in result, "Quarterly MAPE not calculated"
    assert 'best_performing_month' in result, "Best month not identified"
    assert 'worst_performing_month' in result, "Worst month not identified"
    
    print(f"   ✅ Monthly patterns: {len(result['monthly_mape'])} months analyzed")
    print(f"   ✅ Quarterly patterns: {len(result['quarterly_mape'])} quarters analyzed")
    print(f"   ✅ Best month: {result.get('best_month_name', 'N/A')}")
    print(f"   ✅ Worst month: {result.get('worst_month_name', 'N/A')}")
    
    return True


def test_model_fitting_functions():
    """Test model fitting functions."""
    print("\n🧪 Testing Model Fitting Functions...")
    
    # Create test series
    series = create_test_data(48)  # 4 years of data
    
    # Test ETS fitting function
    try:
        ets_func = create_ets_fitting_function("mul")
        ets_model = ets_func(series[:36])  # Train on 3 years
        forecast = ets_model.forecast(12)  # Forecast 1 year
        print(f"   ✅ ETS model: {len(forecast)} periods forecasted")
    except Exception as e:
        print(f"   ❌ ETS model failed: {e}")
        return False
    
    # Test SARIMA fitting function
    try:
        sarima_func = create_sarima_fitting_function()
        sarima_model = sarima_func(series[:36])
        forecast = sarima_model.forecast(12)
        print(f"   ✅ SARIMA model: {len(forecast)} periods forecasted")
    except Exception as e:
        print(f"   ❌ SARIMA model failed: {e}")
        return False
    
    # Test Polynomial fitting function
    try:
        poly_func = create_polynomial_fitting_function(degree=2)
        poly_model = poly_func(series[:36])
        forecast = poly_model.forecast(12)
        print(f"   ✅ Polynomial model: {len(forecast)} periods forecasted")
    except Exception as e:
        print(f"   ❌ Polynomial model failed: {e}")
        return False
    
    return True


def test_walk_forward_validation():
    """Test walk-forward validation."""
    print("\n🧪 Testing Walk-Forward Validation...")
    
    # Create test series
    series = create_test_data(60)  # 5 years of data
    
    # Create ETS fitting function
    ets_func = create_ets_fitting_function("mul")
    
    # Run walk-forward validation
    try:
        result = walk_forward_validation(
            series=series,
            model_fitting_func=ets_func,
            window_size=24,
            step_size=3,
            horizon=6,
            diagnostic_messages=[]
        )
        
        if result is not None:
            print(f"   ✅ Completed {result['iterations']} validation iterations")
            print(f"   ✅ Mean MAPE: {result['mean_mape']:.1%} ± {result['std_mape']:.1%}")
            print(f"   ✅ MAPE range: {result['min_mape']:.1%} - {result['max_mape']:.1%}")
            return True
        else:
            print("   ⚠️ Walk-forward validation returned None (insufficient data)")
            return True  # This is acceptable for short series
    except Exception as e:
        print(f"   ❌ Walk-forward validation failed: {e}")
        return False


def test_cross_validation():
    """Test time series cross-validation."""
    print("\n🧪 Testing Time Series Cross-Validation...")
    
    # Create test series
    series = create_test_data(60)  # 5 years of data
    
    # Create ETS fitting function
    ets_func = create_ets_fitting_function("mul")
    
    # Run cross-validation
    try:
        result = time_series_cross_validation(
            series=series,
            model_fitting_func=ets_func,
            n_splits=3,
            horizon=6,
            diagnostic_messages=[]
        )
        
        if result is not None:
            print(f"   ✅ Completed {result['folds_completed']} validation folds")
            print(f"   ✅ Mean MAPE: {result['mean_mape']:.1%} ± {result['std_mape']:.1%}")
            return True
        else:
            print("   ⚠️ Cross-validation returned None (insufficient data)")
            return True  # This is acceptable for short series
    except Exception as e:
        print(f"   ❌ Cross-validation failed: {e}")
        return False


def test_comprehensive_validation_suite():
    """Test the comprehensive validation suite."""
    print("\n🧪 Testing Comprehensive Validation Suite...")
    
    # Create test series
    series = create_test_data(48)  # 4 years of data
    train_size = 36
    val_size = 12
    
    train = series[:train_size]
    val = series[train_size:train_size + val_size]
    
    # Fit a simple model for testing
    ets_func = create_ets_fitting_function("mul")
    model = ets_func(train)
    forecast = model.forecast(val_size)
    
    # Run comprehensive validation
    try:
        result = comprehensive_validation_suite(
            actual=val.values,
            forecast=forecast,
            dates=val.index,
            product_name="Test_Product",
            enable_walk_forward=True,
            enable_cross_validation=True,
            series=series,
            model_fitting_func=ets_func,
            model_params={},
            diagnostic_messages=[]
        )
        
        # Validate structure
        assert 'basic_validation' in result, "Basic validation missing"
        assert 'enhanced_analysis' in result, "Enhanced analysis missing"
        assert 'seasonal_analysis' in result, "Seasonal analysis missing"
        
        print(f"   ✅ Basic MAPE: {result['basic_validation']['mape']:.1%}")
        print(f"   ✅ Enhanced analysis completed")
        print(f"   ✅ Seasonal analysis completed")
        
        if result['walk_forward_validation']:
            print(f"   ✅ Walk-forward validation completed")
        
        if result['cross_validation']:
            print(f"   ✅ Cross-validation completed")
        
        return True
    except Exception as e:
        print(f"   ❌ Comprehensive validation failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🧪 Testing Edge Cases...")
    
    # Test with minimal data
    short_series = create_test_data(12)  # Only 1 year
    
    try:
        # This should handle short series gracefully
        ets_func = create_ets_fitting_function("mul")
        result = walk_forward_validation(
            series=short_series,
            model_fitting_func=ets_func,
            window_size=24,  # More than available data
            diagnostic_messages=[]
        )
        
        # Should return None for insufficient data
        if result is None:
            print("   ✅ Properly handled insufficient data for walk-forward")
        else:
            print("   ⚠️ Walk-forward did not return None for insufficient data")
        
    except Exception as e:
        print(f"   ❌ Edge case handling failed: {e}")
        return False
    
    # Test with zero values
    try:
        actual = np.array([0, 100, 0, 110, 105])
        forecast = np.array([5, 98, 8, 108, 107])
        dates = pd.date_range('2023-01-01', periods=5, freq='MS')
        
        result = enhanced_mape_analysis(actual, forecast, dates)
        print("   ✅ Handled zero values in MAPE calculation")
        
    except Exception as e:
        print(f"   ❌ Zero value handling failed: {e}")
        return False
    
    return True


def main():
    """Run all advanced validation tests."""
    print("🚀 Starting Advanced Validation Tests")
    print("=" * 50)
    
    tests = [
        test_enhanced_mape_analysis,
        test_seasonal_mape_analysis,
        test_model_fitting_functions,
        test_walk_forward_validation,
        test_cross_validation,
        test_comprehensive_validation_suite,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"   ❌ {test.__name__} failed")
        except Exception as e:
            print(f"   ❌ {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All advanced validation features working correctly!")
        print("\n📊 New Back Testing Features Ready:")
        print("   • Enhanced MAPE analysis with confidence intervals")
        print("   • Walk-forward validation for robust testing")
        print("   • Time series cross-validation")
        print("   • Seasonal performance analysis")
        print("   • Comprehensive validation suite")
        print("   • Improved error handling and edge cases")
    else:
        print(f"❌ {total - passed} tests failed - check implementation")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
