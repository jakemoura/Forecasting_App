@echo off
echo.
echo ========================================
echo  SIMPLE PMDARIMA FIX FOR PYTHON 3.13+
echo ========================================
echo.

echo Checking Python version...
python --version
echo.

echo Installing dependencies...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Installing numpy (required for pmdarima)...
python -m pip install numpy

echo.
echo Installing scipy (required for pmdarima)...
python -m pip install scipy

echo.
echo Installing pmdarima using pre-built wheels (no compilation)...
python -m pip install pmdarima

echo.
echo Testing installation...
python -c "import pmdarima; print('SUCCESS! pmdarima version:', pmdarima.__version__)" 2>nul
if errorlevel 1 (
    echo.
    echo pmdarima import failed. Trying alternative version...
    python -m pip install pmdarima==2.0.4
    python -c "import pmdarima; print('SUCCESS! pmdarima version:', pmdarima.__version__)" 2>nul
    if errorlevel 1 (
        echo.
        echo Still failed. Trying older version...
        python -m pip install "pmdarima>=1.8.0,<2.0.0"
        python -c "import pmdarima; print('SUCCESS! pmdarima version:', pmdarima.__version__)" 2>nul
        if errorlevel 1 (
            echo.
            echo ========================================
            echo  INSTALLATION FAILED
            echo ========================================
            echo.
            echo Python 3.13 may not be fully supported yet.
            echo RECOMMENDED: Install Python 3.12 or 3.11
            echo.
            echo Download from:
            echo   - https://www.python.org/downloads/
            echo   - Microsoft Store (search for "Python 3.12")
            echo.
            echo The app will still work with 6 other forecasting models!
            echo.
            pause
            exit /b 1
        )
    )
)

echo.
echo ========================================
echo  INSTALLATION SUCCESSFUL!
echo ========================================
echo.
python -c "import numpy, scipy, pmdarima; print('numpy:', numpy.__version__); print('scipy:', scipy.__version__); print('pmdarima:', pmdarima.__version__)"
echo.
echo pmdarima is now working!
echo.
pause
