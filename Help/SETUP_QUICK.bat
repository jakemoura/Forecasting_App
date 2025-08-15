@echo off

:: ========================================
::  QUICK Forecasting Apps Setup (Backup)
:: ========================================
echo.
echo ========================================
echo  QUICK Forecasting Apps Setup
echo ========================================
echo.
echo This is a QUICK installation that installs basic packages only:
echo   - Core Streamlit, pandas, numpy, scikit-learn
echo   - Excel support (openpyxl, xlsxwriter)
echo   - Visualization packages
echo   - LightGBM and XGBoost only
echo.
echo SKIPPED in QUICK mode:
echo   - Prophet (complex installation)
echo   - pmdarima (long compilation time)
echo.
echo Installation time: ~3-5 minutes
echo.
echo For full capabilities, use the main SETUP.bat instead.
echo.
echo Press any key to continue with QUICK installation or Ctrl+C to cancel...
pause >nul

echo.
echo [1/8] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo ERROR: Python not found on your system
    echo ========================================
    echo.
    echo Please install Python first. We recommend one of these options:
    echo.
    echo OPTION 1 - Microsoft Store ^(Recommended for Windows 10/11^):
    echo   1. Open Microsoft Store
    echo   2. Search for "Python 3.12" or "Python"
    echo   3. Install the latest Python version
    echo   4. This handles PATH setup automatically
    echo.
    echo OPTION 2 - Python.org ^(Traditional method^):
    echo   1. Visit https://python.org/downloads/
    echo   2. Download Python 3.10 or newer
    echo   3. During installation, CHECK "Add Python to PATH"
    echo   4. Choose "Install for all users" if you have admin rights
    echo.
    echo After installing Python:
    echo   - Close this window
    echo   - Open a new Command Prompt or PowerShell
    echo   - Run this setup script again
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)
echo Python is available!
echo Detected version:
python --version

echo.
echo [2/8] Upgrading pip...
python -m pip install --upgrade pip --no-warn-script-location

echo.
echo [3/8] Installing core packages...
echo This may take a few minutes
python -m pip install --no-warn-script-location --timeout 300 streamlit pandas numpy scikit-learn statsmodels

echo.
echo [4/8] Installing Excel support packages...
echo Installing comprehensive Excel functionality for export features
python -m pip install --no-warn-script-location --timeout 300 openpyxl xlsxwriter xlrd pyxlsb

echo.
echo [5/8] Installing visualization packages...
python -m pip install --no-warn-script-location --timeout 300 altair matplotlib plotly seaborn

echo.
echo [6/8] Installing basic ML packages...
echo Installing LightGBM and XGBoost
python -m pip install --no-warn-script-location --timeout 300 lightgbm
python -m pip install --no-warn-script-location --timeout 300 xgboost
echo Basic ML models: LightGBM and XGBoost installed

echo.
echo [6.5/8] Optional: pmdarima (Auto-ARIMA) installation...
echo pmdarima provides additional forecasting accuracy but may take time to install
echo.
choice /c YN /m "Install pmdarima for Auto-ARIMA forecasting? (Y/N)"
if errorlevel 2 (
    echo Skipping pmdarima installation (can install later with Help\FORCE_INSTALL_PMDARIMA.bat)
) else (
    echo Installing pmdarima with force installer for Python 3.11+ compatibility...
    if exist "FORCE_INSTALL_PMDARIMA.bat" (
        call "FORCE_INSTALL_PMDARIMA.bat"
    ) else if exist "INSTALL_PMDARIMA_PYTHON311.bat" (
        call "INSTALL_PMDARIMA_PYTHON311.bat"
    ) else (
        echo Force installers not found, trying standard installation...
        python -m pip install --no-warn-script-location --timeout 600 pmdarima
    )
)

echo.
echo [7/8] Configuring Streamlit...
if not exist "%USERPROFILE%\.streamlit" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
)
echo [browser] > "%USERPROFILE%\.streamlit\config.toml" 2>nul
echo gatherUsageStats = false >> "%USERPROFILE%\.streamlit\config.toml" 2>nul
echo serverAddress = "127.0.0.1" >> "%USERPROFILE%\.streamlit\config.toml" 2>nul
echo [global] >> "%USERPROFILE%\.streamlit\config.toml" 2>nul
echo showWarningOnDirectExecution = false >> "%USERPROFILE%\.streamlit\config.toml" 2>nul
echo [server] >> "%USERPROFILE%\.streamlit\config.toml" 2>nul
echo headless = false >> "%USERPROFILE%\.streamlit\config.toml" 2>nul
echo enableCORS = false >> "%USERPROFILE%\.streamlit\config.toml" 2>nul
echo enableXsrfProtection = false >> "%USERPROFILE%\.streamlit\config.toml" 2>nul
echo address = "127.0.0.1" >> "%USERPROFILE%\.streamlit\config.toml" 2>nul
echo port = 8501 >> "%USERPROFILE%\.streamlit\config.toml" 2>nul

echo [general] > "%USERPROFILE%\.streamlit\credentials.toml" 2>nul
echo email = "" >> "%USERPROFILE%\.streamlit\credentials.toml" 2>nul

if not exist ".streamlit" (
    mkdir ".streamlit" 2>nul
)
echo [browser] > ".streamlit\config.toml" 2>nul
echo gatherUsageStats = false >> ".streamlit\config.toml" 2>nul
echo [server] >> ".streamlit\config.toml" 2>nul
echo port = 8501 >> ".streamlit\config.toml" 2>nul

echo [general] > ".streamlit\credentials.toml" 2>nul
echo email = "" >> ".streamlit\credentials.toml" 2>nul
echo Streamlit configured!

echo.
echo [8/8] Testing installation...
echo Testing core packages...
python -c "import streamlit, pandas, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Some core packages may have issues
    set core_success=false
) else (
    echo Core packages working
    set core_success=true
)

echo Testing Excel functionality...
python -c "import openpyxl" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Excel export may have issues
)

echo Testing ML packages...
python -c "import lightgbm" >nul 2>&1
if %errorlevel% equ 0 (
    echo LightGBM: Available
) else (
    echo LightGBM: Not available
)

python -c "import xgboost" >nul 2>&1
if %errorlevel% equ 0 (
    echo XGBoost: Available
) else (
    echo XGBoost: Not available
)

echo.
echo ========================================
echo  QUICK SETUP COMPLETED
echo ========================================
echo.

if "%core_success%"=="true" (
    echo SUCCESS: Quick installation completed successfully!
    echo Basic forecasting models (LightGBM, XGBoost) are ready.
    echo Both forecasting apps will work with available models.
    echo.
    echo To install additional models for maximum accuracy:
    echo   - pmdarima (Auto-ARIMA): Run FORCE_INSTALL_PMDARIMA.bat (in this Help folder)
    echo   - For Python 3.11+: Run INSTALL_PMDARIMA_PYTHON311.bat (in this Help folder)
    echo   - Prophet: Run the main ..\SETUP.bat
) else (
    echo WARNING: Some core packages may have installation issues.
    echo Apps may still work with available packages.
)

echo.
echo MAIN FORECAST APP:
echo   - Run: ..\RUN_FORECAST_APP.bat
echo.
echo OUTLOOK FORECASTER:
echo   - Run: ..\RUN_OUTLOOK_FORECASTER.bat
echo.
echo Press any key to exit...
pause >nul
