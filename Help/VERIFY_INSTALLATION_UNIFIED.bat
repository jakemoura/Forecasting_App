@echo off
echo ===============================================
echo UNIFIED VERIFICATION - Testing All Components
echo ===============================================
echo.
echo Checking installation for both forecasting apps:
echo   - Main Forecasting App (forecaster_app.py)
echo   - Quarterly Outlook Forecaster (streamlit_outlook_forecaster.py)
echo.

echo Testing Python and core packages...
python test_models.py
echo.

echo ===============================================
echo Testing Main Forecasting App
===============================================
python -c "
import sys
sys.path.append('.')

try:
    print('Testing main forecasting app imports...')
    import forecaster_app
    print('✓ Main forecasting app imported successfully')
    
    # Test model detection variables
    from forecaster_app import HAVE_PMDARIMA, HAVE_PROPHET, HAVE_LGBM
    print(f'Main App Model Detection:')
    print(f'  HAVE_PMDARIMA: {HAVE_PMDARIMA}')
    print(f'  HAVE_PROPHET: {HAVE_PROPHET}')
    print(f'  HAVE_LGBM: {HAVE_LGBM}')
    
    if HAVE_PMDARIMA:
        print('✓ Auto-ARIMA available in main app')
    else:
        print('⚠ Auto-ARIMA will NOT be available in main app')
        
except Exception as e:
    print(f'✗ Main app import failed: {e}')
"

echo.
echo ===============================================
echo Testing Quarterly Outlook Forecaster
===============================================
python -c "
import sys
sys.path.append('.')

try:
    print('Testing outlook forecaster imports...')
    import streamlit_outlook_forecaster
    print('✓ Outlook forecaster imported successfully')
    
    # Test Excel functionality
    try:
        import openpyxl, xlsxwriter
        print('✓ Excel export functionality ready (openpyxl + xlsxwriter)')
    except ImportError as e:
        print(f'⚠ Excel export may be limited: {e}')
    
    # Test forecasting models
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        print('✓ Machine learning models available')
    except ImportError:
        print('✗ ML models not available')
        
except Exception as e:
    print(f'✗ Outlook forecaster import failed: {e}')
"

echo.
echo ===============================================
echo Testing Excel Export Functionality
===============================================
python -c "
try:
    import pandas as pd
    import openpyxl
    import xlsxwriter
    
    # Test creating a simple Excel file
    test_data = pd.DataFrame({'test': [1, 2, 3]})
    
    # Test openpyxl engine
    test_data.to_excel('test_openpyxl.xlsx', engine='openpyxl', index=False)
    print('✓ openpyxl engine working')
    
    # Test xlsxwriter engine  
    test_data.to_excel('test_xlsxwriter.xlsx', engine='xlsxwriter', index=False)
    print('✓ xlsxwriter engine working')
    
    # Clean up test files
    import os
    try:
        os.remove('test_openpyxl.xlsx')
        os.remove('test_xlsxwriter.xlsx')
    except:
        pass
        
    print('✓ Excel export functionality fully operational')
    
except Exception as e:
    print(f'⚠ Excel export issue: {e}')
    print('Consider running: INSTALL_EXCEL_SUPPORT.bat')
"

echo.
echo ===============================================
echo VERIFICATION COMPLETE
===============================================
echo.
echo If all tests above passed with checkmarks (✓), both apps are ready!
echo.
echo To start the apps:
echo   MAIN FORECASTING: RUN_FORECAST_APP.bat
echo   OUTLOOK FORECASTER: RUN_OUTLOOK_FORECASTER.bat
echo.
echo Your browser will open automatically for either app.
echo Check the sidebar sections for green checkmarks indicating available models.
echo.
pause
