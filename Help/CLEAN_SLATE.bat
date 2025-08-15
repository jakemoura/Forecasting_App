@echo off
echo.
echo ================================
echo  CLEAN SLATE - Complete Package Reset
echo ================================
echo.
echo This script will:
echo   1. List all currently installed packages
echo   2. Uninstall ALL forecasting app packages
echo   3. Clean up pip cache and config files
echo   4. Show final package state
echo.
echo This is perfect for testing clean installs!
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

echo.
echo [BEFORE] Current Python packages:
echo ================================================
python -m pip list
echo ================================================

echo.
echo [1/4] Uninstalling forecasting packages in dependency order...

REM Uninstall high-level packages first
echo Removing high-level packages...
python -m pip uninstall -y streamlit prophet pmdarima lightgbm

REM Uninstall visualization packages
echo Removing visualization packages...
python -m pip uninstall -y altair matplotlib plotly seaborn

REM Uninstall data processing packages
echo Removing data processing packages...
python -m pip uninstall -y openpyxl xlrd pyxlsb

REM Uninstall ML/stats packages
echo Removing ML and statistics packages...
python -m pip uninstall -y statsmodels scikit-learn

REM Uninstall core data packages (be careful - other projects might use these)
echo Removing core data packages (pandas, numpy)...
python -m pip uninstall -y pandas numpy

echo.
echo [2/4] Cleaning up dependency packages that may have been left behind...
python -m pip uninstall -y ^
    pytz ^
    python-dateutil ^
    six ^
    kiwisolver ^
    cycler ^
    fonttools ^
    packaging ^
    pyparsing ^
    pillow ^
    patsy ^
    scipy ^
    joblib ^
    threadpoolctl ^
    et-xmlfile ^
    xlwt ^
    charset-normalizer ^
    certifi ^
    urllib3 ^
    requests ^
    idna ^
    click ^
    jinja2 ^
    markupsafe ^
    itsdangerous ^
    werkzeug ^
    blinker ^
    importlib-metadata ^
    zipp ^
    jsonschema ^
    attrs ^
    pyrsistent ^
    tornado ^
    pyarrow ^
    protobuf ^
    gitpython ^
    gitdb ^
    smmap ^
    watchdog ^
    validators ^
    toml ^
    rich ^
    typing-extensions ^
    tzlocal ^
    pydeck ^
    pympler

echo.
echo [3/4] Cleaning up configuration and cache...
if exist ".streamlit" rmdir /s /q ".streamlit"
if exist "%USERPROFILE%\.streamlit" rmdir /s /q "%USERPROFILE%\.streamlit"
python -m pip cache purge
echo Configuration and cache cleaned!

echo.
echo [4/4] Final cleanup - removing any remaining orphaned packages...
python -m pip autoremove -y 2>nul || echo "autoremove not available, skipping..."

echo.
echo [AFTER] Remaining Python packages:
echo ================================================
python -m pip list
echo ================================================

echo.
echo ================================
echo  CLEAN SLATE COMPLETE!
echo ================================
echo.
echo Your Python environment has been reset.
echo Ready for a fresh SETUP.bat installation!
echo.
echo Next steps:
echo   1. Run SETUP.bat to reinstall everything
echo   2. Test the complete installation process
echo   3. Verify all packages work correctly
echo.

echo.
echo Press any key to exit...
pause >nul
