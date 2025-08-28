
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    # Test if we can import the main components
    from modules.ui_config import setup_page_config
    from modules.data_validation import validate_data_format
    from modules.utils import read_any_excel
    print("FORECASTER_IMPORT_SUCCESS")
    
    # Test basic data validation
    import pandas as pd
    test_data = pd.DataFrame({
        'Date': ['2023-01-01', '2023-02-01', '2023-03-01'],
        'Product': ['Test'] * 3,
        'ACR': [100, 110, 120]
    })
    
    is_valid, msg = validate_data_format(test_data)
    print(f"FORECASTER_VALIDATION_SUCCESS: {is_valid}")
    
except Exception as e:
    print(f"FORECASTER_ERROR: {str(e)}")
    sys.exit(1)
