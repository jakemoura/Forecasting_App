@echo off
echo.
echo ========================================
echo   PMDARIMA INSTALLER FOR PYTHON 3.11+
echo ========================================
echo.
echo This script specifically handles Python 3.11+ compatibility issues
echo with pmdarima installation by using pre-built wheels and avoiding
echo the distutils.msvccompiler issue.
echo.

echo [1/5] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] ERROR: Python not found!
    pause
    exit /b 1
)

echo [+] Python found:
python --version

python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [!] NOTE: This installer is optimized for Python 3.11+
    echo [!] For Python 3.10 and earlier, use FORCE_INSTALL_PMDARIMA.bat instead
    echo [!] Continuing anyway...
)

echo.
echo [2/5] Cleaning existing installations...
python -m pip uninstall pmdarima -y --no-warn-script-location >nul 2>&1
python -m pip cache purge --no-warn-script-location >nul 2>&1

echo.
echo [3/5] Installing build dependencies for Python 3.11+...
echo.
echo Upgrading build tools...
python -m pip install --upgrade pip setuptools wheel --no-warn-script-location --timeout 300
if errorlevel 1 (
    echo [X] ERROR: Failed to upgrade build tools
    pause
    exit /b 1
)

echo.
echo Installing modern setuptools with distutils compatibility...
python -m pip install "setuptools>=65.0.0" --upgrade --no-warn-script-location --timeout 300

echo.
echo [4/5] Installing compatible base packages...
echo.
echo Installing numpy (allowing newer versions for Python 3.11+)...
python -m pip install "numpy>=1.21.0" --no-warn-script-location --timeout 300
if errorlevel 1 (
    echo [X] ERROR: Failed to install numpy
    pause
    exit /b 1
)

echo Installing scipy (allowing newer versions for Python 3.11+)...
python -m pip install "scipy>=1.9.0" --no-warn-script-location --timeout 300
if errorlevel 1 (
    echo [X] ERROR: Failed to install scipy
    pause
    exit /b 1
)

echo Installing scikit-learn and statsmodels...
python -m pip install "scikit-learn>=1.0.0" statsmodels --no-warn-script-location --timeout 300

echo.
echo [5/5] Installing pmdarima using pre-built wheels...
echo.
echo Method 1: Latest pmdarima with pre-built wheels (fastest)...
python -m pip install pmdarima --no-warn-script-location --timeout 600
if errorlevel 1 (
    echo [!] Method 1 failed, trying Method 2...
    echo.
    echo Method 2: Specific pmdarima version with broader compatibility...
    python -m pip install "pmdarima==2.0.4" --no-warn-script-location --timeout 600
    if errorlevel 1 (
        echo [!] Method 2 failed, trying Method 3...
        echo.
        echo Method 3: Alternative pmdarima version...
        python -m pip install "pmdarima==1.8.5" --no-warn-script-location --timeout 600
        if errorlevel 1 (
            echo [X] ERROR: All installation methods failed!
            echo.
            echo Recommendations:
            echo 1. Try using Python 3.10 instead of 3.11+
            echo 2. Install Microsoft C++ Build Tools
            echo 3. Run FORCE_INSTALL_PMDARIMA.bat for source compilation
            echo.
            pause
            exit /b 1
        )
    )
)

echo.
echo [+] Testing installation...
python -c "import pmdarima as pm; import numpy as np; import warnings; warnings.filterwarnings('ignore'); print(f'pmdarima version: {pm.__version__}'); data = np.random.randn(30).cumsum(); model = pm.auto_arima(data, start_p=0, start_q=0, max_p=1, max_q=1, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=False); forecast = model.predict(n_periods=3); print('pmdarima functionality test: PASSED'); print(f'Sample forecast: {forecast}')" 2>nul
if errorlevel 1 (
    echo [X] ERROR: pmdarima installed but not working correctly
    pause
    exit /b 1
)

echo.
echo ========================================
echo       INSTALLATION SUCCESSFUL!
echo ========================================
echo.
echo pmdarima has been successfully installed for Python 3.11+
echo.
echo Package versions:
python -c "import numpy, scipy, pmdarima; print(f'numpy: {numpy.__version__}'); print(f'scipy: {scipy.__version__}'); print(f'pmdarima: {pmdarima.__version__}')"

echo.
echo You can now run your forecasting applications!
echo.
pause
