"""
Test script to verify the Excel download function works correctly
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.ui_components import create_excel_download

def test_excel_function():
    """Test the create_excel_download function"""
    
    print("Testing create_excel_download function...")
    
    # Create sample forecast data
    analysis_date = datetime(2025, 4, 15)
    
    # Sample forecast result structure
    sample_forecast = {
        'quarter_info': {
            'quarter_name': 'FY25 Q4',
            'quarter_start': datetime(2025, 4, 1),
            'quarter_end': datetime(2025, 6, 30)
        },
        'forecasts': {
            'Linear Trend': {
                'forecast': np.array([1000, 1100, 1200, 1300, 1400]),
                'quarter_total': 15000,
                'daily_avg': 1200,
                'remaining_total': 6000
            },
            'Moving Average': {
                'forecast': np.array([1050, 1050, 1050, 1050, 1050]),
                'quarter_total': 14250,
                'daily_avg': 1050,
                'remaining_total': 5250
            }
        },
        'best_model': 'Linear Trend',
        'actual_to_date': 9000,
        'summary': {
            'quarter_total': 15000,
            'daily_run_rate': 1200
        },
        'quarter_progress': {
            'days_completed': 10,
            'days_remaining': 5,
            'total_days': 15,
            'completion_pct': 66.7
        },
        'model_evaluation': {
            'Linear Trend': 12.5,
            'Moving Average': 15.2
        }
    }
    
    # Create sample historical data
    dates = pd.date_range(start=datetime(2025, 4, 1), end=datetime(2025, 4, 10), freq='D')
    values = np.random.normal(1000, 100, len(dates))
    sample_data = pd.Series(values, index=dates)
    
    # Sample forecasts data structure
    forecasts_data = {
        'Product A': {
            'forecast': sample_forecast,
            'data': sample_data
        },
        'Product B': {
            'forecast': sample_forecast,
            'data': sample_data
        }
    }
    
    try:
        # Test the function
        excel_data = create_excel_download(
            forecasts_data=forecasts_data,
            analysis_date=analysis_date,
            filename="test_forecast.xlsx",
            conservatism_factor=0.97,
            capacity_factor=0.85
        )
        
        if excel_data:
            print("✅ Excel function executed successfully!")
            print(f"   Generated {len(excel_data)} bytes of Excel data")
            
            # Save to file to verify
            with open("test_output.xlsx", "wb") as f:
                f.write(excel_data)
            print("   Test file saved as 'test_output.xlsx'")
            
            return True
        else:
            print("❌ Excel function returned None")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Excel function: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_excel_function()
    if success:
        print("\n🎉 All tests passed! The Excel download function is working correctly.")
    else:
        print("\n💥 Test failed! Check the error messages above.")
