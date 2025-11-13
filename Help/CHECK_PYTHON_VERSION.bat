@echo off
echo.
echo ========================================
echo  PYTHON VERSION CHECKER
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.12 from:
    echo   - Microsoft Store (search "Python 3.12")
    echo   - https://python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo Current Python version:
python --version
echo.

python -c "import sys; print(f'Full version: {sys.version}'); print(f'Version info: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"

echo.
echo ========================================
echo  COMPATIBILITY ANALYSIS
echo ========================================
echo.

python -c "import sys; exit(0 if sys.version_info >= (3, 14) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo Status: CRITICAL ISSUE - Python 3.14 detected
    echo.
    echo Python 3.14 is TOO NEW for data science packages!
    echo.
    echo PROBLEMS:
    echo   - No pre-built wheels for most packages
    echo   - statsmodels WILL FAIL
    echo   - pmdarima WILL FAIL
    echo   - Requires C++ compilation (slow and fails)
    echo.
    echo SOLUTION:
    echo   1. Uninstall Python 3.14
    echo   2. Install Python 3.12 (RECOMMENDED)
    echo   3. Run SETUP.bat again
    echo.
    echo See PYTHON_VERSION_GUIDE.md for detailed instructions
    echo.
    goto :end_check
)

python -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo Status: WARNING - Python 3.13 detected
    echo.
    echo Python 3.13 is NEW and some packages may require compilation.
    echo.
    echo RECOMMENDED: Use Python 3.12 instead for best compatibility
    echo.
    echo Current version may work but you might encounter issues with:
    echo   - statsmodels
    echo   - pmdarima
    echo   - Prophet
    echo.
    goto :end_check
)

python -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo Status: EXCELLENT - Python 3.12 detected
    echo.
    echo This is the RECOMMENDED version!
    echo   - All packages have pre-built wheels
    echo   - Fast installation
    echo   - No compilation needed
    echo   - Full compatibility
    echo.
    echo You can proceed with SETUP.bat
    echo.
    goto :end_check
)

python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo Status: EXCELLENT - Python 3.11 detected
    echo.
    echo This is also a RECOMMENDED version!
    echo   - All packages have pre-built wheels
    echo   - Fast installation
    echo   - Full compatibility
    echo.
    echo You can proceed with SETUP.bat
    echo.
    goto :end_check
)

python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo Status: GOOD - Python 3.10 detected
    echo.
    echo This version works well!
    echo   - Good package compatibility
    echo   - Stable and tested
    echo.
    echo You can proceed with SETUP.bat
    echo.
    goto :end_check
)

echo Status: OLD VERSION - Python 3.9 or earlier detected
echo.
echo This version is outdated.
echo RECOMMENDED: Upgrade to Python 3.12
echo.

:end_check
echo ========================================
echo.

echo Installed location:
python -c "import sys; print(sys.executable)"

echo.
echo ========================================
echo  QUICK PACKAGE TEST
echo ========================================
echo.

echo Testing core package installations...
echo.

python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [X] streamlit - NOT INSTALLED
) else (
    python -c "import streamlit; print('[OK] streamlit version:', streamlit.__version__)"
)

python -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo [X] pandas - NOT INSTALLED
) else (
    python -c "import pandas; print('[OK] pandas version:', pandas.__version__)"
)

python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [X] numpy - NOT INSTALLED
) else (
    python -c "import numpy; print('[OK] numpy version:', numpy.__version__)"
)

python -c "import scipy" >nul 2>&1
if errorlevel 1 (
    echo [X] scipy - NOT INSTALLED
) else (
    python -c "import scipy; print('[OK] scipy version:', scipy.__version__)"
)

python -c "import statsmodels" >nul 2>&1
if errorlevel 1 (
    echo [X] statsmodels - NOT INSTALLED
) else (
    python -c "import statsmodels; print('[OK] statsmodels version:', statsmodels.__version__)"
)

python -c "import pmdarima" >nul 2>&1
if errorlevel 1 (
    echo [X] pmdarima - NOT INSTALLED
) else (
    python -c "import pmdarima; print('[OK] pmdarima version:', pmdarima.__version__)"
)

echo.
echo ========================================
echo  RECOMMENDATIONS
echo ========================================
echo.

python -c "import sys; exit(0 if sys.version_info >= (3, 14) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo ACTION REQUIRED:
    echo   1. Read PYTHON_VERSION_GUIDE.md
    echo   2. Uninstall Python 3.14
    echo   3. Install Python 3.12
    echo   4. Run SETUP.bat
    echo.
) else (
    python -c "import streamlit, pandas, numpy" >nul 2>&1
    if errorlevel 1 (
        echo ACTION REQUIRED:
        echo   Run SETUP.bat to install required packages
        echo.
    ) else (
        echo Your setup looks good!
        echo   Run SETUP.bat to install any missing optional packages
        echo   Or run the apps directly with RUN_FORECAST_APP.bat
        echo.
    )
)

pause
