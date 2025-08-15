"""
Simple test for the monthly breakdown functionality
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add the modules directory to the path  
sys.path.insert(0, 'modules')

try:
    from ui_components import display_monthly_splits, create_excel_download
    print("✅ Successfully imported functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test basic functionality
print("Testing basic functionality...")

# Create sample data
dates = pd.date_range('2025-04-01', '2025-04-15', freq='D')
values = np.random.normal(1000, 100, len(dates))
sample_series = pd.Series(values, index=dates)

# Create sample forecast structure
sample_forecast = {
    'quarter_info': {
        'quarter_name': 'FY25 Q4',
        'quarter_start': datetime(2025, 4, 1),
        'quarter_end': datetime(2025, 6, 30)
    },
    'forecasts': {
        'Linear Trend': {
            'forecast': np.array([1100, 1200, 1300, 1400, 1500]),
            'quarter_total': 20000,
            'daily_avg': 1200,
            'remaining_total': 6000
        }
    },
    'best_model': 'Linear Trend',
    'actual_to_date': 14000,
    'summary': {
        'quarter_total': 20000,
        'daily_run_rate': 1200
    },
    'quarter_progress': {
        'days_completed': 15,
        'days_remaining': 5,
        'total_days': 20,
        'completion_pct': 75.0
    }
}

forecasts_data = {
    'Test Product': {
        'forecast': sample_forecast,
        'data': sample_series
    }
}

# Test Excel generation
print("Testing Excel generation...")
try:
    excel_bytes = create_excel_download(
        forecasts_data=forecasts_data,
        analysis_date=datetime(2025, 4, 15),
        filename="test.xlsx",
        conservatism_factor=0.97,
        capacity_factor=0.85
    )
    
    if excel_bytes:
        print(f"✅ Excel generation successful! Generated {len(excel_bytes)} bytes")
        
        # Save to file
        with open('test_monthly_breakdown.xlsx', 'wb') as f:
            f.write(excel_bytes)
        print("✅ File saved as test_monthly_breakdown.xlsx")
    else:
        print("❌ Excel generation returned None")
        
except Exception as e:
    print(f"❌ Excel generation error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed!")
