@echo off
echo.
echo ========================================
echo     PMDARIMA DIAGNOSTIC TOOL
echo ========================================
echo.
echo This script will check your pmdarima installation
echo and diagnose any compatibility issues.
echo.

echo [1/5] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] ERROR: Python not found
    pause
    exit /b 1
)

echo [+] Python version:
python --version

REM Check for Python 3.11+ specific issues
python -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 (
    echo [!] NOTICE: Python 3.11+ detected
    echo [!] This version has known distutils compatibility issues with pmdarima
    echo [!] If pmdarima installation fails, try INSTALL_PMDARIMA_PYTHON311.bat
    echo.
    
    REM Check for distutils availability
    python -c "from distutils.msvccompiler import get_build_version" >nul 2>&1
    if errorlevel 1 (
        echo [X] WARNING: distutils.msvccompiler not available (Python 3.11+ issue)
        echo [!] This will cause pmdarima source compilation to fail
        echo [!] Recommendation: Use pre-built wheels or Python 3.10
    ) else (
        echo [+] distutils.msvccompiler: Available
    )
) else (
    echo [+] Python 3.10 or earlier: Good compatibility with pmdarima
)

echo.
echo [2/5] Checking installed package versions...
echo.

echo Checking numpy...
python -c "import numpy; print(f'numpy: {numpy.__version__}')" 2>nul
if errorlevel 1 (
    echo [X] numpy: NOT INSTALLED
    set NUMPY_MISSING=1
) else (
    echo [+] numpy: OK
)

echo Checking scipy...
python -c "import scipy; print(f'scipy: {scipy.__version__}')" 2>nul
if errorlevel 1 (
    echo [X] scipy: NOT INSTALLED
    set SCIPY_MISSING=1
) else (
    echo [+] scipy: OK
)

echo Checking pmdarima...
python -c "import pmdarima; print(f'pmdarima: {pmdarima.__version__}')" 2>nul
if errorlevel 1 (
    echo [X] pmdarima: NOT INSTALLED
    set PMDARIMA_MISSING=1
) else (
    echo [+] pmdarima: OK
)

echo.
echo [3/5] Testing pmdarima functionality...
if defined PMDARIMA_MISSING (
    echo [X] SKIP: pmdarima not installed
    goto :version_check
)

python -c "import pmdarima as pm; import numpy as np; import warnings; warnings.filterwarnings('ignore'); print('[+] Basic import: OK'); data = np.random.randn(30).cumsum(); model = pm.auto_arima(data, start_p=0, start_q=0, max_p=1, max_q=1, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=False); print('[+] auto_arima creation: OK'); forecast = model.predict(n_periods=3); print('[+] Forecasting: OK'); print('[+] pmdarima functionality: FULLY WORKING')" 2>nul
if errorlevel 1 (
    echo [X] pmdarima functionality: FAILED
    set PMDARIMA_BROKEN=1
) else (
    echo [+] pmdarima functionality: OK
)

:version_check
echo.
echo [4/5] Checking version compatibility...
python -c "
import sys
try:
    import numpy
    import scipy
    from packaging import version
    
    numpy_ver = numpy.__version__
    scipy_ver = scipy.__version__
    
    print(f'numpy version: {numpy_ver}')
    print(f'scipy version: {scipy_ver}')
    
    # Check if scipy version is compatible with pmdarima
    if version.parse(scipy_ver) >= version.parse('1.11.0'):
        print('[!] WARNING: scipy version may be too new for pmdarima')
        print('    Recommended: scipy < 1.11.0')
        print('    Run FORCE_INSTALL_PMDARIMA.bat to fix this')
    elif version.parse(scipy_ver) < version.parse('1.9.0'):
        print('[!] WARNING: scipy version may be too old')
        print('    Recommended: scipy >= 1.9.0')
    else:
        print('[+] scipy version is compatible')
        
    # Check numpy compatibility
    if version.parse(numpy_ver) >= version.parse('1.25.0'):
        print('[!] WARNING: numpy version may be too new for pmdarima')
        print('    Recommended: numpy < 1.25.0')
    elif version.parse(numpy_ver) < version.parse('1.21.0'):
        print('[!] WARNING: numpy version may be too old')
        print('    Recommended: numpy >= 1.21.0')
    else:
        print('[+] numpy version is compatible')
        
except ImportError as e:
    print(f'[X] Package missing for version check: {e}')
except Exception as e:
    print(f'[!] Version check error: {e}')
" 2>nul

echo.
echo [5/5] System requirements check...
echo.

echo Checking for C++ compiler (needed for pmdarima compilation)...
where cl >nul 2>&1
if errorlevel 1 (
    echo [!] WARNING: Microsoft C++ compiler not found in PATH
    echo     This may cause pmdarima installation to fail
    echo     Install "Microsoft C++ Build Tools" if needed
) else (
    echo [+] Microsoft C++ compiler: Found
)

echo.
echo Checking disk space...
for /f "tokens=3" %%a in ('dir /-c "%SystemDrive%\" ^| find "bytes free"') do set FREE_SPACE=%%a
echo [+] Free disk space check completed

echo.
echo ========================================
echo           DIAGNOSTIC SUMMARY
echo ========================================
echo.

if defined NUMPY_MISSING (
    echo [X] numpy is missing - install with: pip install numpy
)
if defined SCIPY_MISSING (
    echo [X] scipy is missing - install with: pip install scipy
)
if defined PMDARIMA_MISSING (
    echo [X] pmdarima is missing - run FORCE_INSTALL_PMDARIMA.bat
)
if defined PMDARIMA_BROKEN (
    echo [X] pmdarima is installed but not working - run FORCE_INSTALL_PMDARIMA.bat
)

if not defined NUMPY_MISSING if not defined SCIPY_MISSING if not defined PMDARIMA_MISSING if not defined PMDARIMA_BROKEN (
    echo [+] All packages are installed and working correctly!
    echo [+] pmdarima is ready for use in your forecasting applications
)

echo.
echo RECOMMENDATIONS:
echo.
if defined PMDARIMA_MISSING (
    echo 1. Run FORCE_INSTALL_PMDARIMA.bat to install pmdarima with compatible versions
    echo 2. For Python 3.11+: Try INSTALL_PMDARIMA_PYTHON311.bat (uses pre-built wheels)
)
if defined PMDARIMA_BROKEN (
    echo 1. Run FORCE_INSTALL_PMDARIMA.bat to reinstall pmdarima with compatible versions
    echo 2. For Python 3.11+: Try INSTALL_PMDARIMA_PYTHON311.bat (avoids compilation issues)
)
echo 2. Ensure you have Microsoft C++ Build Tools installed
echo 3. Use Python 3.10 or 3.11 for best compatibility
echo 4. Close all Streamlit applications before running installers
echo.
pause
