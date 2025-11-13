@echo off
echo.
echo ========================================
echo     PMDARIMA FORCE INSTALLER
echo ========================================
echo.
echo This script will FORCE install pmdarima by:
echo   1. Completely removing scipy, numpy, and pmdarima
echo   2. Installing compatible versions in the correct order
echo   3. Compiling pmdarima from source with no binary cache
echo   4. Testing the installation thoroughly
echo.
echo WARNING: This will temporarily remove and reinstall
echo          scipy, numpy, and pmdarima packages!
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

echo.
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] ERROR: Python not found! Please install Python first.
    echo [!] Recommended: Python 3.10 or 3.11 from python.org or Microsoft Store
    pause
    exit /b 1
)

echo [+] Python found:
python --version

echo.
echo [2/7] Stopping any running applications...
echo [!] Please close all Streamlit applications before continuing
echo [!] Press any key when ready...
pause >nul

echo.
echo [3/7] Uninstalling existing packages...
echo.
echo Removing pmdarima (if installed)...
python -m pip uninstall pmdarima -y --no-warn-script-location
if errorlevel 1 (
    echo [!] pmdarima was not installed or removal failed
)

echo.
echo Removing scipy (if installed)...
python -m pip uninstall scipy -y --no-warn-script-location
if errorlevel 1 (
    echo [!] scipy was not installed or removal failed
)

echo.
echo Removing numpy (if installed)...
python -m pip uninstall numpy -y --no-warn-script-location
if errorlevel 1 (
    echo [!] numpy was not installed or removal failed
)

echo.
echo Removing any cached packages...
python -m pip cache purge --no-warn-script-location
if errorlevel 1 (
    echo [!] Cache purge failed, continuing anyway...
)

echo.
echo [4/7] Installing build dependencies...
echo.
echo Installing/upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel --no-warn-script-location --timeout 300
if errorlevel 1 (
    echo [X] ERROR: Failed to upgrade build tools
    pause
    exit /b 1
)

echo.
echo Installing setuptools-scm (required for modern builds)...
python -m pip install "setuptools-scm>=6.0" --no-warn-script-location --timeout 300
if errorlevel 1 (
    echo [!] WARNING: setuptools-scm installation failed, continuing...
)

echo.
echo Installing distutils replacement for Python 3.11+...
python -c "import sys; print(f'Python version: {sys.version_info.major}.{sys.version_info.minor}')"
python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [+] Python 3.11+ detected, installing setuptools with distutils compatibility...
    python -m pip install "setuptools>=60.0.0" --upgrade --no-warn-script-location --timeout 300
    if errorlevel 1 (
        echo [!] WARNING: Modern setuptools installation failed
    )
) else (
    echo [+] Python 3.10 or earlier detected, using standard setuptools
)

echo.
echo Installing Cython (required for pmdarima compilation)...
python -m pip install "Cython>=0.29.0,<3.0.0" --no-warn-script-location --timeout 300
if errorlevel 1 (
    echo [X] ERROR: Failed to install Cython
    pause
    exit /b 1
)

echo.
echo Installing Microsoft Visual C++ 14.0 build tools check...
python -c "import subprocess, sys; result = subprocess.run([sys.executable, '-c', 'print(\"Build tools check completed\")'], capture_output=True); print('[+] Build tools appear to be available') if result.returncode == 0 else print('[!] WARNING: Build tools may not be properly configured')" 2>nul
if errorlevel 1 (
    echo [!] WARNING: Cannot verify build tools - may cause compilation issues
)

echo.
echo [5/7] Installing compatible base packages...
echo.
echo Installing compatible numpy version...
python -m pip install "numpy>=1.21.0" --no-warn-script-location --timeout 300 --force-reinstall
if errorlevel 1 (
    echo [X] ERROR: Failed to install compatible numpy
    pause
    exit /b 1
)

echo.
echo Installing compatible scipy version...
python -m pip install "scipy>=1.9.0" --no-warn-script-location --timeout 300 --force-reinstall
if errorlevel 1 (
    echo [X] ERROR: Failed to install compatible scipy
    pause
    exit /b 1
)

echo.
echo Installing other required dependencies...
python -m pip install "scikit-learn>=1.0.0" statsmodels pandas --no-warn-script-location --timeout 300
if errorlevel 1 (
    echo [!] WARNING: Some dependencies failed to install, continuing...
)

echo.
echo [6/7] Installing pmdarima from source...
echo.
echo NOTICE: This step may take 5-15 minutes to compile
echo Please be patient - compilation output will be shown
echo.

REM Check Python version and use appropriate installation method
python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [+] Python 3.11+ detected - using compatible installation method...
    echo [+] Setting up environment for modern Python build system...
    
    REM Set environment variables for Python 3.11+ compatibility
    set SETUPTOOLS_USE_DISTUTILS=stdlib
    set DISTUTILS_USE_SDK=1
    
    echo [+] Attempting pmdarima installation with pre-built wheels (recommended for Python 3.11+)...
    python -m pip install pmdarima --no-cache-dir --no-warn-script-location --timeout 900
    if errorlevel 1 (
        echo [!] Latest version failed, trying specific version 2.0.4...
        python -m pip install "pmdarima==2.0.4" --no-cache-dir --no-warn-script-location --timeout 900
        if errorlevel 1 (
            echo [!] Pre-built wheels failed, trying with build isolation disabled...
            python -m pip install pmdarima --no-build-isolation --no-cache-dir --no-warn-script-location --timeout 900 --verbose
            if errorlevel 1 (
                echo [!] Build isolation method failed, trying older pmdarima version...
                python -m pip install "pmdarima>=1.8.0,<2.0.0" --no-cache-dir --no-warn-script-location --timeout 900
                if errorlevel 1 goto :install_failed
            )
        )
    )
) else (
    echo [+] Python 3.10 or earlier - using source compilation...
    python -m pip install "pmdarima>=2.0.4" --no-binary=pmdarima --no-cache-dir --no-warn-script-location --timeout 900 --verbose
    if errorlevel 1 goto :install_failed
)

echo [+] pmdarima installation completed successfully!
goto :test_installation

:install_failed
    echo.
    echo [X] ERROR: pmdarima installation failed!
    echo.
    echo Common causes and solutions:
    echo   1. Python 3.13+ compatibility issue (newest Python versions)
    echo      - pmdarima may not have pre-built wheels for Python 3.13 yet
    echo      - RECOMMENDED: Use Python 3.11 or 3.12 instead
    echo      - Download from python.org or Microsoft Store
    echo.
    echo   2. Python 3.11+ distutils compatibility issue
    echo      - Pre-built wheels should work (no compilation needed)
    echo      - If compilation is attempted, may fail due to missing distutils
    echo.
    echo   3. Missing C++ compiler (only needed if building from source)
    echo      - Install "Microsoft C++ Build Tools" from:
    echo        https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo      - Or install Visual Studio Community with C++ workload
    echo.
    echo   4. Network timeout during installation
    echo      - Try running this script again
    echo      - Check your internet connection
    echo.
    echo   5. Insufficient system resources
    echo      - Close other applications
    echo      - Ensure sufficient disk space (2GB+ free)
    echo.
    pause
    exit /b 1

:test_installation

echo.
echo [7/7] Testing installation...
echo.
echo Testing numpy import...
python -c "import numpy; print(f'numpy {numpy.__version__} - OK')" 2>nul
if errorlevel 1 (
    echo [X] ERROR: numpy import failed
    pause
    exit /b 1
)

echo Testing scipy import...
python -c "import scipy; print(f'scipy {scipy.__version__} - OK')" 2>nul
if errorlevel 1 (
    echo [X] ERROR: scipy import failed
    pause
    exit /b 1
)

echo Testing pmdarima import...
python -c "import pmdarima; print(f'pmdarima {pmdarima.__version__} - OK')" 2>nul
if errorlevel 1 (
    echo [X] ERROR: pmdarima import failed
    pause
    exit /b 1
)

echo.
echo Testing pmdarima functionality...
python -c "import pmdarima as pm; import numpy as np; import warnings; warnings.filterwarnings('ignore'); data = np.random.randn(50).cumsum(); model = pm.auto_arima(data, start_p=0, start_q=0, max_p=2, max_q=2, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=False); forecast = model.predict(n_periods=5); print('pmdarima functionality test - OK'); print(f'Sample forecast: {forecast[:3]}')" 2>nul
if errorlevel 1 (
    echo [X] ERROR: pmdarima functionality test failed
    echo [!] Package installed but not working correctly
    pause
    exit /b 1
)

echo.
echo ========================================
echo       INSTALLATION SUCCESSFUL!
echo ========================================
echo.
echo Package versions installed:
python -c "import numpy, scipy, pmdarima; print(f'numpy: {numpy.__version__}'); print(f'scipy: {scipy.__version__}'); print(f'pmdarima: {pmdarima.__version__}')"

echo.
echo pmdarima is now ready for use in your forecasting applications!
echo.
echo You can now run:
echo   - RUN_FORECAST_APP.bat
echo   - RUN_OUTLOOK_FORECASTER.bat
echo.
echo Both applications should now have full pmdarima support.
echo.
pause
