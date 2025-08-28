#!/usr/bin/env python3
"""
Quick test script for enhanced rolling validation
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the modules path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Forecaster App', 'modules'))

def test_enhanced_rolling_validation():
    """Test the enhanced rolling validation function"""
    try:
        from metrics import enhanced_rolling_validation
        
        # Create synthetic monthly data (24 months)
        dates = pd.date_range('2022-01-01', periods=24, freq='MS')
        # Simple trend + seasonal + noise
        trend = np.linspace(100, 120, 24)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(24) / 12)
        noise = np.random.normal(0, 2, 24)
        values = trend + seasonal + noise
        
        series = pd.Series(values, index=dates)
        
        # Simple model fitting function (naive seasonal)
        def naive_model_fit(train_data, **kwargs):
            class NaiveModel:
                def __init__(self, data):
                    self.data = data
                def forecast(self, steps):
                    # Simple repeat last year pattern
                    if len(self.data) >= 12:
                        return np.tile(self.data[-12:], (steps // 12) + 1)[:steps]
                    else:
                        return np.repeat(self.data.mean(), steps)
            return NaiveModel(train_data.values)
        
        # Test enhanced rolling validation
        print("🧪 Testing Enhanced Rolling Validation...")
        
        results = enhanced_rolling_validation(
            series=series,
            model_fitting_func=naive_model_fit,
            min_train_size=12,
            max_train_size=18,
            validation_horizon=3,
            backtest_months=15,
            recency_alpha=0.6,
            diagnostic_messages=[]
        )
        
        if results:
            print("✅ Enhanced rolling validation successful!")
            print(f"   - Folds: {results['folds']}")
            print(f"   - Mean WAPE: {results['mean_wape']:.1%}")
            print(f"   - Recent Weighted WAPE: {results['recent_weighted_wape']:.1%}")
            print(f"   - P75 WAPE: {results['p75_wape']:.1%}")
            print(f"   - P95 WAPE: {results['p95_wape']:.1%}")
            print(f"   - Fold consistency: {results['fold_consistency']:.1%}")
            print(f"   - Trend improving: {results['trend_improving']}")
            print(f"   - Method: {results['method']}")
            return True
        else:
            print("❌ Enhanced rolling validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Enhanced Rolling Validation Implementation...")
    success = test_enhanced_rolling_validation()
    if success:
        print("\n🎉 All tests passed! Enhanced rolling validation is working.")
    else:
        print("\n💥 Tests failed. Check implementation.")
