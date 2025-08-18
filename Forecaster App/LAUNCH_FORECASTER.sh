#!/bin/bash

# Forecaster App Launcher for macOS
# This script launches the refactored forecaster app with smart backtesting

echo "🚀 Launching Forecaster App with Smart Backtesting..."
echo "=================================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3 and try again"
    echo "You can download it from: https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION detected"

# Check if required packages are available
echo "🔍 Checking required packages..."

# Check for streamlit
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "⚠️  Warning: Streamlit not found. Installing required packages..."
    echo "Installing streamlit and other dependencies..."
    pip3 install streamlit pandas numpy scikit-learn
fi

# Check for other optional packages
MISSING_PACKAGES=()
if ! python3 -c "import prophet" &> /dev/null; then
    MISSING_PACKAGES+=("prophet")
fi

if ! python3 -c "import pmdarima" &> /dev/null; then
    MISSING_PACKAGES+=("pmdarima")
fi

if ! python3 -c "import lightgbm" &> /dev/null; then
    MISSING_PACKAGES+=("lightgbm")
fi

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "⚠️  Optional packages not found: ${MISSING_PACKAGES[*]}"
    echo "The app will work with basic models (SARIMA, ETS, Polynomial)"
    echo "For advanced models, you can install missing packages later"
fi

# Check if the main app file exists
if [ ! -f "forecaster_app.py" ]; then
    echo "❌ Error: forecaster_app.py not found in current directory"
    echo "Current directory: $(pwd)"
    echo "Files in directory:"
    ls -la
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✅ All checks passed!"
echo ""
echo "🎯 Launching Forecaster App..."
echo "📊 Features:"
echo "   • Smart backtesting with data-driven recommendations"
echo "   • Automatic fallback to MAPE rankings"
echo "   • Business-focused validation approach"
echo "   • Fast, reliable forecasting"
echo ""
echo "🌐 The app will open in your default web browser"
echo "📱 You can also access it at: http://localhost:8501"
echo ""
echo "⏹️  To stop the app, press Ctrl+C in this terminal"
echo ""

# Launch the app
echo "🚀 Starting Streamlit..."
python3 -m streamlit run forecaster_app.py --server.port 8501 --server.headless false

# If we get here, the app has stopped
echo ""
echo "👋 Forecaster App has stopped"
read -p "Press Enter to exit..."
