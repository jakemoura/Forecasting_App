@echo off
echo [*] Starting Forecasting App...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Python not found! Please run SETUP.bat first.
    echo [!] If you just installed Python, try opening a new Command Prompt
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Streamlit not installed! Please run SETUP.bat first.
    echo [!] If setup failed, try running Help\SETUP_QUICK.bat
    pause
    exit /b 1
)

REM Check if the target directory exists
if not exist "%~dp0Forecaster App" (
    echo [X] Error: Forecaster App directory not found!
    echo [!] Current directory: %~dp0
    echo [!] Looking for: %~dp0Forecaster App
    pause
    exit /b 1
)

REM Change to the Forecaster App directory
cd /d "%~dp0Forecaster App"

REM Check if the main Python file exists
if not exist "forecaster_app.py" (
    echo [X] Error: forecaster_app.py not found in Forecaster App directory!
    echo [!] Please ensure all files are properly extracted
    pause
    exit /b 1
)

REM Check for port conflicts
netstat -an | findstr ":8501" >nul 2>&1
if %errorlevel% equ 0 (
    echo [!] Warning: Port 8501 appears to be in use
    echo [!] If the app doesn't start, try closing other instances first
    timeout /t 3 >nul
)

echo [+] Python check: OK
echo [+] Streamlit check: OK
echo [+] Directory check: OK
echo [+] Application file check: OK

REM Skip Streamlit email prompt and usage stats
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_GLOBAL_SHOW_WARNING_ON_DIRECT_EXECUTION=false
set STREAMLIT_EMAIL=installer@localhost

REM Create global Streamlit config directory to skip email prompt system-wide
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\config.toml" (
    echo [browser] > "%USERPROFILE%\.streamlit\config.toml"
    echo gatherUsageStats = false >> "%USERPROFILE%\.streamlit\config.toml"
    echo [global] >> "%USERPROFILE%\.streamlit\config.toml"
    echo showWarningOnDirectExecution = false >> "%USERPROFILE%\.streamlit\config.toml"
)
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    echo [general] > "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "installer@localhost" >> "%USERPROFILE%\.streamlit\credentials.toml"
)

REM Ensure local config directory exists
if not exist ".streamlit" mkdir ".streamlit"
if not exist ".streamlit\config.toml" (
    echo [browser] > ".streamlit\config.toml"
    echo gatherUsageStats = false >> ".streamlit\config.toml"
    echo serverAddress = "127.0.0.1" >> ".streamlit\config.toml"
    echo [global] >> ".streamlit\config.toml"
    echo showWarningOnDirectExecution = false >> ".streamlit\config.toml"
    echo [server] >> ".streamlit\config.toml"
    echo headless = false >> ".streamlit\config.toml"
    echo enableCORS = false >> ".streamlit\config.toml"
    echo enableXsrfProtection = false >> ".streamlit\config.toml"
    echo address = "127.0.0.1" >> ".streamlit\config.toml"
    echo port = 8501 >> ".streamlit\config.toml"
    echo [client] >> ".streamlit\config.toml"
    echo showErrorDetails = false >> ".streamlit\config.toml"
    echo toolbarMode = "minimal" >> ".streamlit\config.toml"
)

REM Ensure credentials file exists to prevent email prompt
if not exist ".streamlit\credentials.toml" (
    echo [general] > ".streamlit\credentials.toml"
    echo email = "installer@localhost" >> ".streamlit\credentials.toml"
)

echo [+] Configuration setup: OK

echo.
echo ========================================
echo           IMPORTANT WARNING
echo ========================================
echo.
color 0C
echo  *** DO NOT CLOSE THIS WINDOW ***
echo  *** DO NOT CLOSE THIS WINDOW ***
echo  *** DO NOT CLOSE THIS WINDOW ***
color 07
echo.
echo  Closing this command prompt window
echo  will STOP the web application!
echo.
color 0E
echo  To stop the app: Press Ctrl+C here
echo  or close your browser and then this window
color 07
echo.
echo ========================================
echo.
echo [+] Starting your forecasting app...
echo [+] Your browser will open automatically at: http://localhost:8501
echo [+] Keep this window open while using the app
echo [+] Main Forecaster runs on: http://localhost:8501
echo [+] Outlook Forecaster runs on: http://localhost:8502
echo.

python -m streamlit run forecaster_app.py --browser.gatherUsageStats=false
