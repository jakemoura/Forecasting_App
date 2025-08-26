@echo off
REM Regression Test Runner for Forecasting Applications
REM Run this batch file to execute the regression test suite

echo.
echo ================================================
echo Forecasting Applications - Regression Test Suite
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo Running Simple Regression Test...
echo.

REM Run the simple regression test
python simple_regression_test.py

if errorlevel 1 (
    echo.
    echo ================================================
    echo Tests FAILED - Please review the output above
    echo ================================================
    pause
    exit /b 1
) else (
    echo.
    echo ================================================
    echo All tests PASSED successfully!
    echo ================================================
    echo.
    echo To run comprehensive tests, use:
    echo   python run_tests.py
    echo.
    pause
)
