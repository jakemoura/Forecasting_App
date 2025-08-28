
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    # Test if we can import the main components
    from modules.fiscal_calendar import get_fiscal_quarter_info
    from modules.data_processing import analyze_daily_data
    from modules.quarterly_forecasting import forecast_quarter_completion
    print("OUTLOOK_IMPORT_SUCCESS")
    
    # Test fiscal calendar logic
    from datetime import datetime
    test_date = datetime(2024, 8, 15)  # Should be Q1
    quarter_info = get_fiscal_quarter_info(test_date)
    
    if quarter_info and quarter_info.get('quarter') == 1:
        print("OUTLOOK_FISCAL_SUCCESS")
    else:
        print(f"OUTLOOK_FISCAL_ERROR: {quarter_info}")
    
    # Test data processing
    import pandas as pd
    import numpy as np
    
    test_data = pd.DataFrame({
        'Date': pd.date_range('2024-08-01', '2024-08-15', freq='D'),
        'Product': ['Test'] * 15,
        'Value': np.random.randint(50, 150, 15)
    })
    
    analysis = analyze_daily_data(test_data, 'Value')
    if 'weekday_avg' in analysis:
        print("OUTLOOK_DATA_SUCCESS")
    else:
        print(f"OUTLOOK_DATA_ERROR: {analysis}")
    
except Exception as e:
    print(f"OUTLOOK_ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
