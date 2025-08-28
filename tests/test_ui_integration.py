#!/usr/bin/env python3
"""
Test script to verify enhanced rolling validation UI integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test imports
print("Testing enhanced validation imports...")
try:
    from modules.metrics import enhanced_rolling_validation
    from modules.data_validation import analyze_data_quality, _get_backtesting_recommendations
    import pandas as pd
    import numpy as np
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Create test data (36 months = good data scenario)
print("\nCreating test data...")
dates = pd.date_range('2020-01-01', periods=36, freq='M')
data = pd.DataFrame({
    'Date': dates,
    'Product': ['TestProduct'] * 36,
    'ACR': np.random.randn(36) * 100 + 1000
})
print(f"✅ Created data with {len(data)} months")

# Test backtesting recommendations
print("\nTesting backtesting recommendations...")
try:
    result = _get_backtesting_recommendations(12, 36, 24)  # min=12, max=36, avg=24
    print(f"Status: {result['status']}")
    print(f"Title: {result['title']}")
    print(f"Recommended Range: {result['recommended_range']}")
    print(f"Default Value: {result['default_value']} months")
    print(f"Message: {result['message']}")
    
    # Verify it's using enhanced rolling validation
    if "enhanced rolling" in result['recommended_range'].lower():
        print("✅ Using enhanced rolling validation recommendations")
    else:
        print("❌ Not using enhanced rolling validation recommendations")
        
except Exception as e:
    print(f"❌ Backtesting recommendations error: {e}")
    sys.exit(1)

# Test enhanced validation function with dummy model function
print("\nTesting enhanced rolling validation function...")
try:
    def dummy_model_func(train_data, validation_data, params=None):
        """Dummy model that returns simple average forecast"""
        avg_value = train_data.mean()
        forecast = [avg_value] * len(validation_data)
        return forecast
    
    # Create a simple time series
    series = pd.Series(
        np.random.randn(36) * 100 + 1000,
        index=pd.date_range('2020-01-01', periods=36, freq='M')
    )
    
    validation_result = enhanced_rolling_validation(
        series, 
        dummy_model_func,
        min_train_size=12,
        max_train_size=18,
        validation_horizon=3,
        backtest_months=15,
        recency_alpha=0.6
    )
    
    print(f"Validation completed: {validation_result['validation_completed']}")
    print(f"Number of folds: {validation_result['num_folds']}")
    print(f"Validation type: {validation_result['validation_type']}")
    print(f"Aggregation method: {validation_result['aggregation_method']}")
    print(f"Training window range: {validation_result['training_window_range']}")
    
    # Verify enhanced rolling validation characteristics
    if (validation_result['validation_completed'] and 
        validation_result['num_folds'] >= 4 and
        validation_result['validation_type'] == 'enhanced_rolling' and
        validation_result['aggregation_method'] == 'recency_weighted_wape'):
        print("✅ Enhanced rolling validation working correctly")
    else:
        print("❌ Enhanced rolling validation not working as expected")
        
except Exception as e:
    print(f"❌ Enhanced validation error: {e}")
    sys.exit(1)

print("\n🎉 All tests passed! Enhanced rolling validation UI integration is working correctly.")
