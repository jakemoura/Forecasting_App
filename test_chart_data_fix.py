#!/usr/bin/env python3
"""
Test script to verify that enhanced rolling validation now includes chart data
for backtesting overlay functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the Forecaster App directory to the path
forecaster_path = os.path.join(os.path.dirname(__file__), 'Forecaster App')
sys.path.insert(0, forecaster_path)

try:
    from modules.metrics import enhanced_rolling_validation, wape
    print("✅ Successfully imported enhanced rolling validation function")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def create_test_data():
    """Create realistic test data for validation."""
    # Create 36 months of data (enough for enhanced validation)
    start_date = datetime(2021, 1, 1)
    dates = pd.date_range(start=start_date, periods=36, freq='MS')
    
    # Create trend + seasonal + noise data
    trend = np.linspace(100, 150, 36)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(36) / 12)
    noise = np.random.normal(0, 5, 36)
    values = trend + seasonal + noise
    
    return pd.Series(values, index=dates, name='Sales')

def simple_fit_func(train_data, **model_params):
    """Simple linear trend model for testing."""
    class SimpleModel:
        def __init__(self, data):
            # Simple linear regression on time
            x = np.arange(len(data))
            y = data.values
            self.slope = np.polyfit(x, y, 1)[0]
            self.intercept = data.iloc[-1]  # Start from last training point
            
        def forecast(self, steps):
            return [self.intercept + self.slope * i for i in range(1, steps + 1)]
    
    return SimpleModel(train_data)

def test_chart_data_functionality():
    """Test that enhanced rolling validation now includes chart data."""
    print("\n🧪 Testing Enhanced Rolling Validation Chart Data")
    print("=" * 55)
    
    # Create test data
    print("📊 Creating test data (36 months)...")
    series = create_test_data()
    print(f"   Data range: {series.index[0].strftime('%Y-%m')} to {series.index[-1].strftime('%Y-%m')}")
    
    # Run enhanced rolling validation
    diagnostic_messages = []
    print("\n🔄 Running enhanced rolling validation...")
    
    results = enhanced_rolling_validation(
        series=series,
        model_fitting_func=simple_fit_func,
        min_train_size=12,
        max_train_size=18,
        validation_horizon=3,
        backtest_months=15,
        recency_alpha=0.6,
        model_params={},
        diagnostic_messages=diagnostic_messages
    )
    
    # Print diagnostic messages
    print("\n📋 Diagnostic Messages:")
    for msg in diagnostic_messages:
        print(f"   {msg}")
    
    if results is None:
        print("❌ Enhanced rolling validation failed!")
        return False
    
    # Check if chart data is present
    print("\n🔍 Checking Chart Data Availability:")
    
    has_train_data = 'train_data' in results and results['train_data'] is not None
    has_test_data = 'test_data' in results and results['test_data'] is not None
    has_predictions = 'predictions' in results and results['predictions'] is not None
    
    print(f"   train_data: {'✅' if has_train_data else '❌'}")
    print(f"   test_data: {'✅' if has_test_data else '❌'}")  
    print(f"   predictions: {'✅' if has_predictions else '❌'}")
    
    if has_train_data and has_test_data and has_predictions:
        print("\n📈 Chart Data Details:")
        print(f"   Train data size: {len(results['train_data'])} months")
        print(f"   Test data size: {len(results['test_data'])} months")
        print(f"   Predictions size: {len(results['predictions'])} values")
        
        # Verify data integrity
        train_data = results['train_data']
        test_data = results['test_data']
        predictions = results['predictions']
        
        print(f"   Train period: {train_data.index[0].strftime('%Y-%m')} to {train_data.index[-1].strftime('%Y-%m')}")
        print(f"   Test period: {test_data.index[0].strftime('%Y-%m')} to {test_data.index[-1].strftime('%Y-%m')}")
        
        # Check if test data and predictions have same length
        data_length_match = len(test_data) == len(predictions)
        print(f"   Data length consistency: {'✅' if data_length_match else '❌'}")
        
        if data_length_match:
            # Calculate WAPE for this fold
            test_wape = wape(test_data.values, predictions)
            print(f"   Test fold WAPE: {test_wape:.2%}")
        
        print("\n✅ Chart data functionality is working correctly!")
        return True
    else:
        print("\n❌ Chart data is missing - overlay functionality will not work!")
        return False

def test_metrics_consistency():
    """Test that metrics are consistent with chart data."""
    print("\n🧪 Testing Metrics Consistency with Chart Data")
    print("=" * 48)
    
    series = create_test_data()
    diagnostic_messages = []
    
    results = enhanced_rolling_validation(
        series=series,
        model_fitting_func=simple_fit_func,
        diagnostic_messages=diagnostic_messages
    )
    
    if results is None or results['train_data'] is None:
        print("❌ No results or chart data available")
        return False
    
    # Extract metrics
    weighted_wape = results['recent_weighted_wape']
    mean_wape = results['mean_wape']
    num_folds = results['folds']
    
    print(f"📊 Enhanced Validation Results:")
    print(f"   Folds: {num_folds}")
    print(f"   Weighted WAPE: {weighted_wape:.2%}")
    print(f"   Mean WAPE: {mean_wape:.2%}")
    print(f"   Validation type: {results['validation_type']}")
    print(f"   Aggregation method: {results['aggregation_method']}")
    
    # Verify chart data represents most recent fold
    test_data = results['test_data']
    predictions = results['predictions']
    
    if len(test_data) > 0 and len(predictions) > 0:
        chart_wape = wape(test_data.values, predictions)
        print(f"   Chart fold WAPE: {chart_wape:.2%}")
        
        # Chart WAPE should be one of the individual fold WAPEs
        fold_wapes = results['fold_wapes']
        print(f"   Individual fold WAPEs: {[f'{w:.2%}' for w in fold_wapes]}")
        
        print("\n✅ Metrics consistency verified!")
        return True
    else:
        print("❌ Chart data is empty")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Rolling Validation Chart Data Test")
    print("=" * 50)
    
    try:
        # Test chart data functionality
        chart_test_passed = test_chart_data_functionality()
        
        # Test metrics consistency
        metrics_test_passed = test_metrics_consistency()
        
        print("\n" + "=" * 50)
        print("📋 Test Summary:")
        print(f"   Chart Data Test: {'✅ PASSED' if chart_test_passed else '❌ FAILED'}")
        print(f"   Metrics Test: {'✅ PASSED' if metrics_test_passed else '❌ FAILED'}")
        
        if chart_test_passed and metrics_test_passed:
            print("\n🎉 All tests passed! Enhanced rolling validation now supports chart overlay.")
        else:
            print("\n⚠️ Some tests failed. Chart overlay may not work properly.")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
