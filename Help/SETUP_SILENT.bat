@echo off
setlocal enabledelayedexpansion

title Installing Python Dependencies
echo =============================================
echo Installing Python Dependencies
echo =============================================
echo This may take 10-15 minutes...
echo Please wait while packages are installed.
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python and try again.
    exit /b 1
)

echo Python is available!
python --version
echo.

REM Create a simple pip install command that won't hang
echo Installing all required packages...
echo This may take several minutes, please be patient...
echo.

REM Use a single pip install command with all packages
python -m pip install --quiet --disable-pip-version-check --timeout 600 --retries 2 numpy pandas matplotlib seaborn openpyxl xlsxwriter scipy statsmodels scikit-learn prophet pmdarima lightgbm xgboost streamlit plotly

if errorlevel 1 (
    echo WARNING: Some packages may have failed to install.
    echo You can run SETUP.bat manually later for detailed installation.
) else (
    echo All packages installed successfully!
)

echo.
echo Installation completed!
exit /b 0