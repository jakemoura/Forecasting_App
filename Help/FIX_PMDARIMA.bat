@echo off
echo.
echo ========================================
echo  QUICK FIX FOR PMDARIMA INSTALLATION
echo ========================================
echo.
echo This will fix the scipy compatibility issue that prevents pmdarima from working.
echo Error: "_lazywhere" missing from scipy
echo Solution: Install compatible scipy and pmdarima versions
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

echo.
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python first.
    pause
    exit /b 1
)
echo Python found:
python --version

echo.
echo [2/5] Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel --quiet --no-warn-script-location

echo.
echo [3/5] Installing build dependencies in correct order...
echo Installing numpy first (required for pmdarima build)...
python -m pip install "numpy>=1.21.0" --upgrade --no-warn-script-location
if errorlevel 1 (
    echo ERROR: Failed to install numpy
    pause
    exit /b 1
)

echo Installing scipy (includes _lazywhere function needed by pmdarima)...
python -m pip install "scipy>=1.9.0" --upgrade --no-warn-script-location
if errorlevel 1 (
    echo ERROR: Failed to install scipy
    pause
    exit /b 1
)

echo Installing Cython (required for pmdarima compilation)...
python -m pip install "Cython>=0.29.0,<3.0.0" --no-warn-script-location

echo.
echo [4/5] Installing pmdarima (trying multiple methods)...
echo Method 1: Pre-built wheels (fastest)...
python -m pip install pmdarima --no-warn-script-location
if errorlevel 1 (
    echo Method 1 failed, trying Method 2...
    echo Method 2: Specific version with pre-built wheels...
    python -m pip install "pmdarima==2.0.4" --no-warn-script-location
    if errorlevel 1 (
        echo Method 2 failed, trying Method 3...
        echo Method 3: Latest version without build isolation...
        python -m pip install pmdarima --no-build-isolation --no-warn-script-location
        if errorlevel 1 (
            echo All methods failed. Trying one last approach...
            python -m pip install "pmdarima>=1.8.0,<2.0.0" --no-warn-script-location
        )
    )
)

echo.
echo [5/5] Testing installation...
python -c "import pmdarima; print('pmdarima version:', pmdarima.__version__)" 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: pmdarima import failed
    echo Showing detailed error:
    python -c "import pmdarima"
) else (
    echo.
    echo SUCCESS! pmdarima is now working correctly.
)

echo.
echo Verifying scipy has _lazywhere function...
python -c "from scipy.stats._stats_py import _lazywhere; print('scipy _lazywhere: Available')" 2>nul
if errorlevel 1 (
    echo WARNING: _lazywhere still not available
    echo Current scipy version:
    python -c "import scipy; print('scipy version:', scipy.__version__)"
    echo.
    echo You may need to upgrade scipy further:
    echo   python -m pip install scipy --upgrade
) else (
    echo scipy _lazywhere: Confirmed working
)

echo.
echo Testing auto_arima functionality...
python -c "import pmdarima as pm; import numpy as np; data = np.random.randn(30).cumsum(); model = pm.auto_arima(data, start_p=0, start_q=0, max_p=1, max_q=1, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore'); print('auto_arima test: PASSED')" 2>nul
if errorlevel 1 (
    echo WARNING: auto_arima test failed
    echo Showing detailed error:
    python -c "import pmdarima as pm; import numpy as np; data = np.random.randn(30).cumsum(); model = pm.auto_arima(data, start_p=0, start_q=0, max_p=1, max_q=1, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')"
) else (
    echo auto_arima test: PASSED
)

echo.
echo ========================================
echo  FIX COMPLETE
echo ========================================
echo.
echo Package versions installed:
python -c "import numpy, scipy, pmdarima; print('numpy:', numpy.__version__); print('scipy:', scipy.__version__); print('pmdarima:', pmdarima.__version__)"

echo.
echo Press any key to exit...
pause >nul
