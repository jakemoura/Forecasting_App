#!/bin/bash

# ========================================
#  Unified Forecasting Apps Setup (macOS)
# ========================================

echo
echo "========================================"
echo "  Unified Forecasting Apps Setup (macOS)"
echo "========================================"
echo
echo "This will install ALL Python packages needed for:"
echo "  - Main Forecasting App (forecaster_app.py)"
echo "  - Quarterly Outlook Forecaster (streamlit_outlook_forecaster.py)"
echo "  - Enhanced Excel export functionality"
echo "  - All forecasting models and visualization tools"
echo
echo "FULL INSTALLATION includes all advanced packages:"
echo "  - Prophet, pmdarima (Auto-ARIMA), LightGBM, XGBoost"
echo "  - May take 10-15 minutes to complete"
echo "  - Provides the most accurate forecasting capabilities"
echo
echo "IMPORTANT: Some packages (Prophet, pmdarima) may take 5 to 10 minutes to compile."
echo "If the script appears to hang, please wait or press Ctrl+C to cancel and try again."
echo "The apps will work even if some packages fail."
echo
echo "Note: For a quicker installation option, see Help/SETUP_QUICK.bat"
echo
read -n 1 -s -r -p "Press any key to continue with FULL installation or Ctrl+C to cancel..."

echo
echo "[1/9] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo
    echo "========================================"
    echo "WARNING: Python3 not found on your system"
    echo "========================================"
    echo
    echo "Please install Python 3.10 or newer from https://python.org/downloads/"
    echo "After installing, re-run this setup script."
    read -n 1 -s -r -p "Press any key to continue without Python or Ctrl+C to cancel..."
    # exit 1
fi

echo "Python is available!"
echo "Detected version:"
python3 --version

echo
echo "[2/9] Upgrading pip..."
python3 -m pip install --upgrade pip --timeout 300

echo
echo "[3/9] Installing core packages..."
python3 -m pip install --timeout 300 streamlit pandas numpy scikit-learn statsmodels

echo
echo "[4/9] Installing Excel support packages..."
python3 -m pip install --timeout 300 openpyxl xlsxwriter xlrd pyxlsb

echo
echo "[5/9] Installing visualization packages..."
python3 -m pip install --timeout 300 altair matplotlib plotly seaborn

echo
echo "[6/9] Installing advanced forecasting packages..."
python3 -m pip install --timeout 300 scipy "numpy>=1.21.0,<1.25.0" --force-reinstall
python3 -m pip install --timeout 300 Cython
python3 -m pip install --timeout 600 pmdarima --no-binary=pmdarima --no-cache-dir
python3 -m pip install --timeout 300 lightgbm
python3 -m pip install --timeout 300 xgboost
echo "Installing Prophet (this may take longer and might fail on some systems)"
python3 -m pip install --timeout 300 prophet || {
    echo "Prophet installation failed or timed out, trying alternative method"
    python3 -m pip install --timeout 180 --upgrade setuptools wheel
    python3 -m pip install --timeout 300 prophet || {
        echo "Prophet still failed, trying lightweight version"
        python3 -m pip install --timeout 180 prophet --no-deps || {
            echo "Prophet installation unsuccessful - this is common on macOS"
            echo "The app will work with 6 other forecasting models including XGBoost"
        }
    }
}
echo "Note: If Prophet fails, both apps will still work with 6 other forecasting models including XGBoost!"

echo
echo "[7/9] Ensuring latest visualization packages..."
python3 -m pip install --timeout 300 matplotlib plotly seaborn

echo
echo "[8/9] Configuring Streamlit to skip email prompt..."
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml <<EOF
[browser]
gatherUsageStats = false
serverAddress = "127.0.0.1"
[global]
showWarningOnDirectExecution = false
[server]
headless = false
enableCORS = false
enableXsrfProtection = false
address = "127.0.0.1"
port = 8501
EOF

cat > ~/.streamlit/credentials.toml <<EOF
[general]
email = ""
EOF

mkdir -p .streamlit
cp ~/.streamlit/config.toml .streamlit/config.toml
cp ~/.streamlit/credentials.toml .streamlit/credentials.toml
echo "Streamlit configured for local-only access with auto browser opening!"

echo
echo "[9/9] Final package verification and testing..."
echo "Testing core installation..."
python3 -c "import streamlit" || echo "WARNING: streamlit may have issues"
python3 -c "import pandas" || echo "WARNING: pandas may have issues"
python3 -c "import numpy" || echo "WARNING: numpy may have issues"
echo "Core packages tested."

echo "Testing Excel functionality..."
python3 -c "import openpyxl" || echo "WARNING: Excel export may have issues"

echo "Testing advanced forecasting packages..."
echo "Checking pmdarima..."
python3 -c "import pmdarima" && echo "pmdarima: Available" || echo "pmdarima: Not available"
echo "Checking Prophet..."
python3 -c "from prophet import Prophet" && echo "Prophet: Available" || echo "Prophet: Not available"
echo "Checking LightGBM..."
python3 -c "import lightgbm" && echo "LightGBM: Available" || echo "LightGBM: Not available"
echo "Checking XGBoost..."
python3 -c "import xgboost" && echo "XGBoost: Available" || echo "XGBoost: Not available"
echo "Advanced package testing complete!"

echo
echo "========================================"
echo "  SETUP PROCESS COMPLETED"
echo "========================================"
echo
echo "MAIN FORECAST APP:"
echo "  - Advanced time series forecasting"
echo "  - Multiple ML models available"
echo "  - File: forecaster_app.py"
echo "  - Run: Forecaster App/LAUNCH_FORECASTER.sh"
echo
echo "OUTLOOK FORECASTER:"
echo "  - Quarterly revenue forecasting"
echo "  - File: outlook_forecaster.py"
echo "  - Run: Quarter Outlook App/RUN_OUTLOOK_FORECASTER.bat"
echo
echo "Troubleshooting tools:"
echo "  - Quick installation option: Help/SETUP_QUICK.bat"
echo "  - Package verification: python3 test_packages.py"
echo "  - Clean slate: Help/CLEAN_SLATE.bat"
echo
read -n 1 -s -r -p "Press any key to exit..."
