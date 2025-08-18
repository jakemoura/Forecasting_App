#!/usr/bin/env python3
"""
Simple test script for the new smart backtesting functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.metrics import simple_backtesting_validation, comprehensive_validation_suite
import pandas as pd
import numpy as np

def create_test_series():
    """Create a test time series for validation."""
    dates = pd.date_range('2020-01-01', periods=48, freq='MS')
    # Create seasonal data with trend
    trend = np.linspace(100, 200, 48)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(48) / 12)
    noise = np.random.normal(0, 5, 48)
    values = trend + seasonal + noise
    return pd.Series(values, index=dates)

def create_simple_model_fitting_function():
    """Create a simple model fitting function for testing."""
    def fit_model(series):
        # Simple linear model for testing
        class SimpleModel:
            def __init__(self, series):
                self.series = series
                self.coef = np.polyfit(range(len(series)), series.values, 1)[0]
                self.intercept = np.polyfit(range(len(series)), series.values, 1)[1]
            
            def forecast(self, steps):
                return np.array([self.intercept + self.coef * (len(self.series) + i) for i in range(steps)])
        
        return SimpleModel(series)
    
    return fit_model

def test_simple_backtesting():
    """Test the simple backtesting functionality."""
    print("🧪 Testing Simple Backtesting...")
    
    # Create test data
    series = create_test_series()
    model_fitting_func = create_simple_model_fitting_function()
    
    # Test with different backtesting periods
    test_cases = [
        (12, 1, 6),   # 12 months backtesting, 1 month gap, 6 months horizon
        (6, 0, 3),    # 6 months backtesting, 0 month gap, 3 months horizon
        (24, 2, 12),  # 24 months backtesting, 2 month gap, 12 months horizon
    ]
    
    for backtest_months, gap, horizon in test_cases:
        print(f"\n📊 Testing: {backtest_months} months backtesting, {gap} month gap, {horizon} months horizon")
        
        try:
            result = simple_backtesting_validation(
                series, model_fitting_func, backtest_months, gap, horizon
            )
            
            if result and result.get('success'):
                print(f"✅ Success: MAPE {result.get('mape', 0):.1%}")
                print(f"   Train months: {result.get('train_months', 0)}")
                print(f"   Test months: {result.get('test_months', 0)}")
                print(f"   Backtest period: {result.get('backtest_period', 0)}")
            else:
                print(f"❌ Failed: {result}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def test_comprehensive_validation_suite():
    """Test the comprehensive validation suite."""
    print("\n🧪 Testing Comprehensive Validation Suite...")
    
    # Create test data
    series = create_test_series()
    model_fitting_func = create_simple_model_fitting_function()
    
    # Split for basic validation
    split_point = len(series) - 12
    train = series.iloc[:split_point]
    val = series.iloc[split_point:]
    
    # Test the comprehensive suite
    try:
        result = comprehensive_validation_suite(
            actual=val.values,
            forecast=val.values * 1.1,  # Simulate some forecast error
            dates=val.index,
            product_name="Test Product",
            enable_walk_forward=False,
            enable_cross_validation=False,
            series=series,
            model_fitting_func=model_fitting_func,
            model_params={},
            diagnostic_messages=[],
            backtest_months=12,
            backtest_gap=1,
            validation_horizon=6,
            fiscal_year_start_month=1
        )
        
        print("✅ Comprehensive validation suite completed")
        print(f"   Basic validation MAPE: {result.get('basic_validation', {}).get('mape', 0):.1%}")
        
        backtesting = result.get('backtesting_validation')
        if backtesting and backtesting.get('success'):
            print(f"   Backtesting MAPE: {backtesting.get('mape', 0):.1%}")
            print(f"   Method recommendation: {result.get('method_recommendation', {}).get('recommended', 'unknown')}")
        else:
            print("   Backtesting failed or not available")
            
    except Exception as e:
        print(f"❌ Error in comprehensive validation suite: {str(e)}")

def main():
    """Run all tests."""
    print("🚀 Starting Smart Backtesting Tests...")
    print("=" * 50)
    
    test_simple_backtesting()
    test_comprehensive_validation_suite()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")

if __name__ == "__main__":
    main()
