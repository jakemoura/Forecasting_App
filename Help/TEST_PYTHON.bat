@echo off
echo Testing Python detection...
where python >nul 2>&1
echo ErrorLevel after where: %errorlevel%
if %errorlevel% neq 0 (
    echo Python NOT found
    exit /b 1
) else (
    echo Python FOUND
    python --version
)
pause
