@echo off
echo [*] Starting Quarterly Outlook Forecaster...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Python not found! Please run setup_portable.bat first.
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Streamlit not installed! Please run setup_portable.bat first.
    pause
    exit /b 1
)

REM Skip Streamlit email prompt and usage stats
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_GLOBAL_SHOW_WARNING_ON_DIRECT_EXECUTION=false

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
    echo port = 8502 >> ".streamlit\config.toml"
)

REM Ensure credentials file exists to prevent email prompt
if not exist ".streamlit\credentials.toml" (
    echo [general] > ".streamlit\credentials.toml"
    echo email = "" >> ".streamlit\credentials.toml"
)

echo [+] Starting Quarterly Outlook Forecaster...
echo [+] This app specializes in daily data and quarterly projections
echo [+] Uses fiscal year calendar (July-June)
echo [+] Your browser will open automatically
echo [+] Close this window to stop the app
echo.
echo [^] Outlook Forecaster will be available at: http://localhost:8502
echo [^] Main Forecaster runs on: http://localhost:8501
echo.

python -m streamlit run streamlit_outlook_forecaster.py --server.port 8502
