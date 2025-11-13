@echo off
REM ========================================
REM  Forecasting Apps Setup (Optimized)
REM  Python 3.11-3.12 Recommended
REM ========================================

echo.
echo ========================================
echo  Forecasting Apps Setup
echo ========================================
echo.

REM Check Python version first
echo [1/9] Checking Python installation...
where python >nul 2>&1
if errorlevel 1 goto PYTHON_NOT_FOUND

echo Python found:
python --version
echo.

echo.
echo [2/9] Upgrading pip...
python -m pip install --upgrade pip --quiet --no-warn-script-location

echo.
echo [3/9] Installing build tools...
python -m pip install --upgrade setuptools wheel --quiet --no-warn-script-location

echo.
echo [4/9] Installing CORE packages...
echo Installing numpy...
python -m pip install numpy --quiet --no-warn-script-location

echo Installing pandas...
python -m pip install pandas --quiet --no-warn-script-location

echo Installing scikit-learn...
python -m pip install scikit-learn --quiet --no-warn-script-location

echo Installing streamlit...
python -m pip install streamlit --quiet --no-warn-script-location

echo.
echo [5/9] Installing scipy and statsmodels...
python -m pip install scipy --quiet --no-warn-script-location
python -m pip install statsmodels --quiet --no-warn-script-location

echo.
echo [6/9] Installing Excel support...
python -m pip install openpyxl xlsxwriter --quiet --no-warn-script-location

echo.
echo [7/9] Installing visualization packages...
python -m pip install altair matplotlib plotly --quiet --no-warn-script-location

echo.
echo [8/9] Installing OPTIONAL forecasting models...
echo Installing pmdarima...
python -m pip install pmdarima --quiet --no-warn-script-location

echo Installing xgboost...
python -m pip install xgboost --quiet --no-warn-script-location

echo Installing lightgbm...
python -m pip install lightgbm --quiet --no-warn-script-location

echo Installing prophet...
python -m pip install prophet --quiet --no-warn-script-location

echo.
echo [9/9] Configuring Streamlit...
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
(
echo [browser]
echo gatherUsageStats = false
echo [general]
echo email = ""
echo [server]
echo headless = false
echo address = "127.0.0.1"
echo port = 8501
) > "%USERPROFILE%\.streamlit\config.toml"

echo.
echo ========================================
echo  INSTALLATION COMPLETE
echo ========================================
echo.
echo Testing packages...
echo.

python -c "import streamlit, pandas, numpy, sklearn" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Some core packages failed!
) else (
    echo SUCCESS! Core packages installed.
    echo Apps are ready to use!
)

echo.
echo Next steps:
echo 1. Run RUN_FORECAST_APP.bat to start the Forecaster
echo 2. Run RUN_OUTLOOK_FORECASTER.bat for Quarterly Outlook
echo.
pause
exit /b 0

:PYTHON_NOT_FOUND
echo.
echo ========================================
echo ERROR: Python not found!
echo ========================================
echo.
echo Please install Python 3.11 or 3.12
echo Visit https://python.org/downloads/
echo CHECK "Add Python to PATH" during install
echo.
pause
exit /b 1
