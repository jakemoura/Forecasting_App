@echo off
echo.
echo ================================
echo  Forecasting App Package Uninstaller
echo ================================
echo.
echo This will REMOVE ALL packages installed by SETUP.bat
echo This is useful for testing clean installations.
echo.
echo WARNING: This will uninstall packages that might be used by other Python projects!
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

echo.
echo [1/8] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Nothing to uninstall.
    pause
    exit /b 1
)
echo Python is available!

echo.
echo [2/8] Uninstalling advanced forecasting packages...
echo Uninstalling Prophet...
python -m pip uninstall -y prophet
echo Uninstalling pmdarima (Auto-ARIMA)...
python -m pip uninstall -y pmdarima
echo Uninstalling LightGBM...
python -m pip uninstall -y lightgbm

echo.
echo [3/8] Uninstalling optional visualization packages...
python -m pip uninstall -y matplotlib plotly seaborn

echo.
echo [4/8] Uninstalling visualization packages...
python -m pip uninstall -y altair

echo.
echo [5/8] Uninstalling Excel support packages...
python -m pip uninstall -y openpyxl xlrd pyxlsb

echo.
echo [6/8] Uninstalling core forecasting packages...
python -m pip uninstall -y statsmodels scikit-learn

echo.
echo [7/8] Uninstalling core data packages...
echo Note: pandas and numpy are often used by other packages, so we'll be careful
python -m pip uninstall -y pandas numpy

echo.
echo [8/8] Uninstalling Streamlit...
python -m pip uninstall -y streamlit

echo.
echo Cleaning up configuration files...
if exist ".streamlit\config.toml" (
    del ".streamlit\config.toml"
    echo Removed local Streamlit config
)
if exist ".streamlit\credentials.toml" (
    del ".streamlit\credentials.toml"
    echo Removed local Streamlit credentials
)
if exist ".streamlit" (
    rmdir ".streamlit"
    echo Removed local .streamlit directory
)

if exist "%USERPROFILE%\.streamlit\config.toml" (
    del "%USERPROFILE%\.streamlit\config.toml"
    echo Removed user Streamlit config
)
if exist "%USERPROFILE%\.streamlit\credentials.toml" (
    del "%USERPROFILE%\.streamlit\credentials.toml"
    echo Removed user Streamlit credentials
)

echo.
echo Cleaning pip cache...
python -m pip cache purge

echo.
echo ================================
echo  UNINSTALL COMPLETE!
echo ================================
echo.
echo All forecasting app packages have been removed.
echo Streamlit configuration files have been cleaned up.
echo.
echo To reinstall everything:
echo   1. Run SETUP.bat
echo   2. Wait for installation to complete
echo   3. Test with RUN_FORECAST_APP.bat
echo.
echo Note: If you have other Python projects, you may need to reinstall
echo packages like pandas, numpy, etc. for those projects.
echo.

echo.
echo Press any key to exit...
pause >nul
