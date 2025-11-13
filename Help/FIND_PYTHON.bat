@echo off
echo.
echo ========================================
echo  FIND PYTHON INSTALLATION
echo ========================================
echo.

echo Searching for Python installations...
echo.

echo Method 1: Checking common locations...
echo.

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python314\python.exe" (
    echo [FOUND] Python 3.14 at:
    echo   C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python314\python.exe
    echo.
)

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe" (
    echo [FOUND] Python 3.13 at:
    echo   C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python313\python.exe
    echo.
)

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe" (
    echo [FOUND] Python 3.12 at:
    echo   C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe
    echo.
)

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    echo [FOUND] Python 3.11 at:
    echo   C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe
    echo.
)

if exist "C:\Users\%USERNAME%\AppData\Local\Microsoft\WindowsApps\python.exe" (
    echo [FOUND] Python (Microsoft Store) at:
    echo   C:\Users\%USERNAME%\AppData\Local\Microsoft\WindowsApps\python.exe
    echo.
)

if exist "C:\Python314\python.exe" (
    echo [FOUND] Python 3.14 at:
    echo   C:\Python314\python.exe
    echo.
)

if exist "C:\Python313\python.exe" (
    echo [FOUND] Python 3.13 at:
    echo   C:\Python313\python.exe
    echo.
)

if exist "C:\Python312\python.exe" (
    echo [FOUND] Python 3.12 at:
    echo   C:\Python312\python.exe
    echo.
)

echo.
echo Method 2: Trying Windows Python Launcher (py)...
echo.

py --version >nul 2>&1
if not errorlevel 1 (
    echo Python launcher works!
    echo Version:
    py --version
    echo.
    echo Location:
    py -c "import sys; print(sys.executable)"
    echo.
) else (
    echo Python launcher (py) not found
)

echo.
echo Method 3: Checking PATH...
echo.

python --version >nul 2>&1
if not errorlevel 1 (
    echo SUCCESS! Python is in PATH!
    echo Version:
    python --version
    echo.
    echo Location:
    python -c "import sys; print(sys.executable)"
    echo.
    echo You can run SETUP.bat now!
) else (
    echo Python is NOT in PATH
    echo.
    echo ========================================
    echo  RECOMMENDATION
    echo ========================================
    echo.
    echo 1. Uninstall current Python
    echo 2. Install Python 3.12 from Microsoft Store
    echo 3. Microsoft Store version automatically adds to PATH
    echo.
    echo See PYTHON_PATH_FIX.md for detailed instructions
)

echo.
echo ========================================
echo  NEXT STEPS
echo ========================================
echo.

python --version >nul 2>&1
if not errorlevel 1 (
    echo Python is working! Next:
    echo   1. Run: CHECK_PYTHON_VERSION.bat
    echo   2. Then run: SETUP.bat
) else (
    py --version >nul 2>&1
    if not errorlevel 1 (
        echo Python launcher works! Next:
        echo   1. Use 'py' instead of 'python'
        echo   2. Or add Python to PATH (see PYTHON_PATH_FIX.md)
    ) else (
        echo Python not found! Next:
        echo   1. Read PYTHON_PATH_FIX.md
        echo   2. Install Python 3.12 from Microsoft Store
        echo   3. Run this script again
    )
)

echo.
pause
