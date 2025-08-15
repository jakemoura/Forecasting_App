@echo off
echo Testing batch file compatibility...
echo.
echo 1. Basic echo test: WORKING
echo.

REM Test variable assignment
set "TEST_VAR=Hello"
echo 2. Variable test: %TEST_VAR%
echo.

REM Test arithmetic
set /a TEST_MATH=5+3
echo 3. Math test: 5+3 = %TEST_MATH%
echo.

REM Test Python availability
echo 4. Testing Python...
python --version 2>nul
if %errorlevel% equ 0 (
    echo    Python is available
    python --version
) else (
    echo    Python not found or not in PATH
)
echo.

REM Test internet connectivity
echo 5. Testing internet connectivity...
ping google.com -n 1 >nul 2>&1
if %errorlevel% equ 0 (
    echo    Internet connection: OK
) else (
    echo    Internet connection: FAILED
)
echo.

REM Test PowerShell availability
echo 6. Testing PowerShell...
powershell -Command "Write-Host 'PowerShell is working'" 2>nul
if %errorlevel% equ 0 (
    echo    PowerShell: OK
) else (
    echo    PowerShell: FAILED
)
echo.

echo 7. Environment info:
echo    OS: %OS%
echo    Processor: %PROCESSOR_ARCHITECTURE%
echo    User: %USERNAME%
echo    Temp: %TEMP%
echo.

echo Diagnostic complete. If you see this message, the batch file is working.
echo.
echo Press any key to continue...
pause >nul
