@echo off
echo ========================================
echo  Full Regression Test Suite Runner
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

echo Testing both Forecaster App and Quarter Outlook App...
echo.

REM Change to the tests directory (we're already here)
cd /d "%~dp0"

REM Run full regression test
python full_regression_test.py

echo.
echo Test completed. Press any key to exit...
pause >nul