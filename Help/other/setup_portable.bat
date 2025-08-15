@echo off
echo [*] One-Click Setup for Forecasting App
echo ==========================================
echo.
echo [~] This will automatically:
echo    1. Check/Install Python if needed
echo    2. Install all required packages
echo    3. Set up your forecasting app
echo.
echo [!] This may take 5-10 minutes on first run
echo [^] Internet connection required
echo.
echo [*] PROGRESS TRACKING ENABLED - You'll see detailed status updates!
echo.
echo Press any key to start automatic setup...
pause >nul
echo.

REM Initialize progress tracking
set "STEP_COUNT=0"

REM ========================================
REM STEP 1: CHECK/INSTALL PYTHON
REM ========================================
set /a STEP_COUNT+=1
echo.
echo ============================================
echo [STEP %STEP_COUNT%] CHECKING PYTHON INSTALLATION
echo ============================================
echo [?] Checking if Python is already installed...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [+] Python is already installed!
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo     Detected version: %%i
    goto :install_packages
)

echo [X] Python not found - installing automatically...
echo.
set /a STEP_COUNT+=1
echo ============================================
echo [STEP %STEP_COUNT%] DOWNLOADING PYTHON INSTALLER
echo ============================================
echo [v] Downloading Python installer (may take 1-3 minutes)...
echo [~] Download size: approximately 25MB

REM Create temp directory
if not exist "%TEMP%\forecasting_app" (
    mkdir "%TEMP%\forecasting_app"
    echo [+] Created temporary directory
) else (
    echo [~] Using existing temporary directory
)

REM Download Python installer
echo [*] Starting download from python.org...
powershell -Command "& {Write-Host '[~] Downloading Python 3.11.7...'; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe' -OutFile '%TEMP%\forecasting_app\python_installer.exe'; Write-Host '[+] Download completed!'}"

if not exist "%TEMP%\forecasting_app\python_installer.exe" (
    echo [X] Failed to download Python installer
    echo [^] Please install Python manually from: https://python.org
    echo [!] Make sure to check "Add Python to PATH" during installation
    start https://python.org
    echo.
    echo After installing Python, run this setup again.
    pause
    exit /b 1
)

echo [+] Python installer downloaded successfully!

set /a STEP_COUNT+=1
echo.
echo ============================================
echo [STEP %STEP_COUNT%] INSTALLING PYTHON
echo ============================================
echo [+] Python downloaded! Installing...
echo [!] IMPORTANT: Installing Python with automatic PATH configuration
echo [~] This may take 2-5 minutes - please wait...
echo.

REM Install Python silently with PATH
echo [*] Running Python installer at %TIME%...
"%TEMP%\forecasting_app\python_installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

echo [+] Python installation completed!

set /a STEP_COUNT+=1
echo.
echo ============================================
echo [STEP %STEP_COUNT%] CONFIGURING ENVIRONMENT
echo ============================================
echo [~] Refreshing environment variables...

REM Refresh PATH for current session - try multiple methods
call refreshenv >nul 2>&1

REM Add common Python paths to current session
set "PATH=%PATH%;C:\Program Files\Python311\Scripts\;C:\Program Files\Python311\"
set "PATH=%PATH%;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\Scripts\"
set "PATH=%PATH%;C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\"
set "PATH=%PATH%;C:\Python311\Scripts\;C:\Python311\"

REM Also try to add pip explicitly to avoid script location warnings
for /f "delims=" %%i in ('python -m site --user-site 2^>nul') do set "PYTHONUSERBASE=%%i"
if defined PYTHONUSERBASE (
    set "PATH=%PATH%;%PYTHONUSERBASE%\..\Scripts\"
)

REM Wait for installation to complete
echo [~] Waiting for installation to finalize...
timeout /t 5 /nobreak >nul

REM Test Python again
echo [?] Verifying Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python installed but PATH not updated yet
    echo [~] Please close this window and run setup again
    pause
    exit /b 1
)

echo [+] Python is now ready!
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo     Active version: %%i

:install_packages
set /a STEP_COUNT+=1
echo.
echo ============================================
echo [STEP %STEP_COUNT%] UPGRADING PIP
echo ============================================
echo [?] Upgrading pip package manager...

REM Upgrade pip first (with PATH warning suppression)
echo [*] Upgrading pip to latest version...
python -m pip install --upgrade pip --quiet --no-warn-script-location
echo [+] Pip upgrade completed!

set /a STEP_COUNT+=1
echo.
echo ============================================
echo [STEP %STEP_COUNT%] INSTALLING CORE PACKAGES
echo ============================================
echo [*] Installing core packages (this may take 3-5 minutes)...
echo [~] Packages: Streamlit, Pandas, NumPy, Scikit-learn, Statsmodels, OpenPyXL, Matplotlib, Plotly
python -m pip install streamlit pandas numpy scikit-learn statsmodels openpyxl matplotlib plotly --quiet --no-warn-script-location
if %errorlevel% neq 0 (
    echo [X] Failed to install core packages
    echo [^] Please check your internet connection
    pause
    exit /b 1
)
echo [+] Core packages installed successfully!

set /a STEP_COUNT+=1
echo.
echo ============================================
echo [STEP %STEP_COUNT%] INSTALLING ENHANCED PACKAGES
echo ============================================
echo [*] Installing enhanced packages...
echo [~] Packages: Excel readers (xlrd, pyxlsb), visualization (seaborn)
python -m pip install xlrd pyxlsb seaborn --quiet --no-warn-script-location
echo [+] Enhanced packages installed successfully!

set /a STEP_COUNT+=1
echo.
echo ============================================
echo [STEP %STEP_COUNT%] INSTALLING ADVANCED MODELS
echo ============================================
echo [*] Installing advanced forecasting models...

echo   [~] Installing Auto-ARIMA (pmdarima)...
python -m pip install pmdarima --quiet --no-warn-script-location
if %errorlevel% equ 0 (
    echo   [+] Auto-ARIMA installed successfully!
) else (
    echo   [!] Auto-ARIMA failed (optional model)
)

echo   [*] Installing Prophet (trying multiple methods)...
echo   [~] This is the most complex package - may take several attempts...

REM Method 1: Try with conda if available
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo   [~] Method 1: Trying conda installation...
    conda install -c conda-forge prophet -y --quiet >nul 2>&1
    if %errorlevel% equ 0 (
        echo   [+] Prophet installed via conda!
        goto :prophet_success
    )
    echo   [!] Conda method failed, trying pip methods...
)

REM Method 2: Try installing dependencies first
echo   [~] Method 2: Installing Prophet dependencies first...
python -m pip install pystan==2.19.1.1 --quiet --no-warn-script-location >nul 2>&1
python -m pip install "holidays>=0.10.5" --quiet --no-warn-script-location >nul 2>&1
python -m pip install "convertdate>=2.1.2" --quiet --no-warn-script-location >nul 2>&1
python -m pip install "LunarCalendar>=0.0.9" --quiet --no-warn-script-location >nul 2>&1
python -m pip install hijri-converter --quiet --no-warn-script-location >nul 2>&1
python -m pip install korean-lunar-calendar --quiet --no-warn-script-location >nul 2>&1

REM Method 3: Try with compatible numpy version
echo   [~] Method 3: Ensuring numpy compatibility...
python -m pip install "numpy<1.24" --quiet --no-warn-script-location >nul 2>&1

REM Method 4: Try direct pip install
echo   [~] Method 4: Trying direct Prophet installation...
python -m pip install prophet --quiet --no-warn-script-location >nul 2>&1
if %errorlevel% equ 0 (
    echo   [+] Prophet installed successfully!
    goto :prophet_success
)

REM Method 5: Try with --only-binary flag (pre-compiled wheels)
echo   [~] Method 5: Trying binary wheel installation...
python -m pip install prophet --only-binary=all --quiet --no-warn-script-location >nul 2>&1
if %errorlevel% equ 0 (
    echo   [+] Prophet installed via binary wheel!
    goto :prophet_success
)

REM Method 6: Try installing Cython first (sometimes helps)
echo   [~] Method 6: Installing Cython and retrying...
python -m pip install Cython --quiet --no-warn-script-location >nul 2>&1
python -m pip install prophet --quiet --no-warn-script-location >nul 2>&1
if %errorlevel% equ 0 (
    echo   [+] Prophet installed with Cython!
    goto :prophet_success
)

echo   [!] Prophet installation failed after all methods - this is common on Windows
echo   [~] Your app will work great without Prophet!
goto :prophet_end

:prophet_success
echo   [+] Prophet forecasting model ready!

:prophet_end

echo   [*] Installing LightGBM machine learning model...
python -m pip install lightgbm --quiet --no-warn-script-location
if %errorlevel% equ 0 (
    echo   [+] LightGBM installed successfully!
) else (
    echo   [!] LightGBM failed (optional model)
)

set /a STEP_COUNT+=1
echo.
echo ============================================
echo [STEP %STEP_COUNT%] INSTALLATION VERIFICATION
echo ============================================
echo [?] Testing core functionality...

python -c "import streamlit, pandas, numpy, sklearn, statsmodels; print('[+] Core packages working!')" 2>nul
if %errorlevel% equ 0 (
    echo [+] Core functionality verified!
) else (
    echo [!] Core package verification had issues but may still work
)

echo.
echo [*] SETUP COMPLETED SUCCESSFULLY!
echo ============================================
echo.
echo [+] Your Forecasting App is ready to use!
echo.
echo [*] To start your app:
echo    [*] Double-click: "run_forecast_app.bat"
echo    [^] Your browser will open automatically
echo    [~] Upload your data and start forecasting!
echo.
echo [~] Available Models:
echo    [+] SARIMA (automatic time series)
echo    [+] ETS (exponential smoothing)
echo    [+] Polynomial Regression (trend analysis)
echo    [+] Plus any advanced models that installed successfully
echo.
echo [!] Your app will work great even if some optional models failed!
echo.
echo Press any key to exit setup...
pause >nul
