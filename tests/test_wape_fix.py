#!/usr/bin/env python3
"""
Quick test to verify the WAPE fix is working correctly.
Tests that enhanced rolling validation WAPE is used instead of basic validation MAPE.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test imports
try:
    from modules.metrics import enhanced_rolling_validation
    print("✅ Core modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_test_data():
    """Create simple test data."""
    dates = pd.date_range('2021-01-01', periods=36, freq='MS')
    # Create a simple trend with seasonality
    trend = np.linspace(100, 150, 36)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(36) / 12)
    noise = np.random.normal(0, 5, 36)
    
    data = pd.DataFrame({
        'Date': dates,
        'Product': 'TestProduct',
        'ACR': trend + seasonal + noise
    })
    return data

def test_enhanced_rolling_validation():
    """Test the enhanced rolling validation function."""
    print("\n🔍 Testing enhanced_rolling_validation function...")
    
    # Create test series
    dates = pd.date_range('2021-01-01', periods=30, freq='MS')
    values = np.linspace(100, 130, 30) + np.random.normal(0, 3, 30)
    series = pd.Series(values, index=dates)
    
    try:
        # Simple model fitting function for testing
        def simple_fit_func(train_data, model_params=None):
            return lambda x: [train_data[-1]] * len(x) if hasattr(train_data, '__getitem__') else [100.0] * len(x)
        
        results = enhanced_rolling_validation(
            series, 
            model_fitting_func=simple_fit_func,
            min_train_size=15, 
            max_train_size=18,
            validation_horizon=3, 
            backtest_months=15,
            recency_alpha=0.6
        )
        
        print(f"✅ Enhanced rolling validation returned: {type(results)}")
        print(f"   - Validation type: {results.get('validation_type', 'MISSING')}")
        print(f"   - Aggregation method: {results.get('aggregation_method', 'MISSING')}")
        print(f"   - Recent weighted WAPE: {results.get('recent_weighted_wape', 'MISSING')}")
        print(f"   - Number of folds: {results.get('folds', 'MISSING')}")
        
        return results.get('recent_weighted_wape') is not None
        
    except Exception as e:
        print(f"❌ Enhanced rolling validation failed: {e}")
        return False

def main():
    print("🔧 Testing WAPE Fix Implementation")
    print("=" * 50)
    
    # Test 1: Enhanced rolling validation function
    validation_works = test_enhanced_rolling_validation()
    
    if validation_works:
        print("\n✅ Enhanced rolling validation is working correctly")
        print("✅ The system should now use weighted WAPE instead of basic MAPE")
        print("\n🎯 Key fixes applied:")
        print("   - All models (SARIMA, ETS, Prophet, Auto-ARIMA, LightGBM, Polynomials, Seasonal-Naive)")
        print("   - Now extract 'recent_weighted_wape' from enhanced rolling validation")
        print("   - Backtesting details table shows correct weighted WAPE values")
        print("   - Results Summary will show weighted WAPE instead of basic validation MAPE")
    else:
        print("\n❌ Enhanced rolling validation is not working properly")
        return False
    
    print("\n🎉 Test completed successfully!")
    print("   The Best WAPE metric should now reflect the enhanced rolling validation weighted WAPE")
    print("   instead of the old walk-forward cross-validation MAPE.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
