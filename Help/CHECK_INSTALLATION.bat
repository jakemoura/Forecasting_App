@echo off
echo.
echo ================================
echo  Package Installation Checker
echo ================================
echo.
echo Checking which forecasting app packages are currently installed...
echo.

echo [Core Packages]
python -c "import streamlit; print('✅ Streamlit:', streamlit.__version__)" 2>nul || echo "❌ Streamlit: Not installed"
python -c "import pandas; print('✅ Pandas:', pandas.__version__)" 2>nul || echo "❌ Pandas: Not installed"
python -c "import numpy; print('✅ NumPy:', numpy.__version__)" 2>nul || echo "❌ NumPy: Not installed"
python -c "import sklearn; print('✅ Scikit-learn:', sklearn.__version__)" 2>nul || echo "❌ Scikit-learn: Not installed"
python -c "import statsmodels; print('✅ Statsmodels:', statsmodels.__version__)" 2>nul || echo "❌ Statsmodels: Not installed"

echo.
echo [Excel Support]
python -c "import openpyxl; print('✅ OpenPyXL:', openpyxl.__version__)" 2>nul || echo "❌ OpenPyXL: Not installed"
python -c "import xlrd; print('✅ XLRD:', xlrd.__version__)" 2>nul || echo "❌ XLRD: Not installed"
python -c "import pyxlsb; print('✅ PyXLSB: Installed')" 2>nul || echo "❌ PyXLSB: Not installed"

echo.
echo [Visualization]
python -c "import altair; print('✅ Altair:', altair.__version__)" 2>nul || echo "❌ Altair: Not installed"
python -c "import matplotlib; print('✅ Matplotlib:', matplotlib.__version__)" 2>nul || echo "❌ Matplotlib: Not installed"
python -c "import plotly; print('✅ Plotly:', plotly.__version__)" 2>nul || echo "❌ Plotly: Not installed"
python -c "import seaborn; print('✅ Seaborn:', seaborn.__version__)" 2>nul || echo "❌ Seaborn: Not installed"

echo.
echo [Advanced Forecasting]
python -c "import pmdarima; print('✅ pmdarima (Auto-ARIMA):', pmdarima.__version__)" 2>nul || echo "❌ pmdarima: Not installed"
python -c "import lightgbm; print('✅ LightGBM:', lightgbm.__version__)" 2>nul || echo "❌ LightGBM: Not installed"
python -c "import prophet; print('✅ Prophet:', prophet.__version__)" 2>nul || echo "❌ Prophet: Not installed"

echo.
echo [Configuration Files]
if exist ".streamlit\config.toml" (
    echo ✅ Local Streamlit config: EXISTS
) else (
    echo ❌ Local Streamlit config: NOT FOUND
)

if exist "%USERPROFILE%\.streamlit\config.toml" (
    echo ✅ User Streamlit config: EXISTS
) else (
    echo ❌ User Streamlit config: NOT FOUND
)

echo.
echo ================================
echo  Installation Status Summary
echo ================================
echo.
echo Use this to verify:
echo   - Before running SETUP.bat (should see mostly ❌)
echo   - After running SETUP.bat (should see mostly ✅)
echo   - After running UNINSTALL_PACKAGES.bat (should see mostly ❌)
echo.

echo.
echo Press any key to exit...
pause >nul
