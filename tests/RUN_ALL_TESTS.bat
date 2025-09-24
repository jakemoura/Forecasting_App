@echo off
echo ========================================
echo  Forecasting Apps Test Suite Runner
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

echo Available test suites:
echo.
echo 1. Quick Regression Tests (30 seconds)
echo 2. Comprehensive Tests (3 minutes)  
echo 3. Full Advanced Tests (5+ minutes)
echo 4. All Tests (run everything)
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Running quick regression tests...
    python simple_regression_test.py
) else if "%choice%"=="2" (
    echo Running comprehensive tests...
    python working_comprehensive_test.py
) else if "%choice%"=="3" (
    echo Running full advanced regression tests...
    python full_regression_test.py --quick
) else if "%choice%"=="4" (
    echo Running ALL tests...
    echo.
    echo === Quick Regression Tests ===
    python simple_regression_test.py
    echo.
    echo === Comprehensive Tests ===
    python working_comprehensive_test.py
    echo.
    echo === Advanced Tests ===
    python comprehensive_e2e_test.py --quick
) else (
    echo Invalid choice. Running quick tests by default.
    python simple_regression_test.py
)

echo.
echo Test run completed. Press any key to exit...
pause >nul