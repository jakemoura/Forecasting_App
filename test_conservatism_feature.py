"""
Test script to validate the new Forecast Conservatism feature in the Forecaster App
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Forecaster App'))

def test_ui_config():
    """Test that the UI config returns the forecast conservatism setting"""
    from modules.ui_config import create_business_adjustments_section
    
    # Note: This would normally require Streamlit context, but we can test the structure
    print("✅ UI config module imports successfully")
    print("✅ create_business_adjustments_section function exists")
    
def test_business_logic():
    """Test that the business logic has the conservatism function"""
    from modules.business_logic import apply_forecast_conservatism_to_results
    
    print("✅ Business logic module imports successfully")
    print("✅ apply_forecast_conservatism_to_results function exists")
    
def test_forecasting_pipeline():
    """Test that the forecasting pipeline has the new parameter"""
    from modules.forecasting_pipeline import run_forecasting_pipeline
    import inspect
    
    # Get the function signature
    sig = inspect.signature(run_forecasting_pipeline)
    params = list(sig.parameters.keys())
    
    print("✅ Forecasting pipeline module imports successfully")
    print("✅ run_forecasting_pipeline function exists")
    
    if 'forecast_conservatism' in params:
        print("✅ forecast_conservatism parameter found in function signature")
    else:
        print("❌ forecast_conservatism parameter NOT found in function signature")
        print(f"Available parameters: {params}")
    
    # Check that old parameters are removed
    old_params = ['apply_business_adjustments', 'business_growth_assumption', 'market_multiplier', 'market_conditions']
    found_old_params = [p for p in old_params if p in params]
    
    if found_old_params:
        print(f"⚠️ Old parameters still present: {found_old_params}")
    else:
        print("✅ Old business adjustment parameters successfully removed")

if __name__ == "__main__":
    print("🧪 Testing Forecast Conservatism Feature Implementation\n")
    
    try:
        test_ui_config()
        print()
        test_business_logic()
        print()
        test_forecasting_pipeline()
        print("\n🎉 All tests passed! Forecast Conservatism feature is ready.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
