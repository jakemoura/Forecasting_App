# Unified Setup Guide for Forecasting Apps

## Overview
This workspace contains a comprehensive, unified setup that supports **both** forecasting applications with all their enhanced features.

## Quick Start

### 🎯 Option 1: Professional Installer (Recommended)
1. **Download and run**: `Forecasting_Apps_Suite_Setup.exe`
2. **Follow the installation wizard** (automatically installs all dependencies)
3. **Launch from Start Menu** or use desktop shortcuts

### 🔧 Option 2: Manual Setup
1. **Run Setup**: Double-click `SETUP.bat` (installs everything needed)
2. **Run Main App**: Double-click `RUN_FORECAST_APP.bat`
3. **Run Outlook App**: Double-click `RUN_OUTLOOK_FORECASTER.bat`

## Applications Included

### 1. Main Forecasting App (`forecaster_app.py`)
- Advanced time series forecasting with 7+ models
- Multiple ML algorithms (SARIMA, ETS, Prophet, LightGBM, XGBoost, etc.)
- Professional charts and business adjustments
- **Launcher:** `RUN_FORECAST_APP.bat`
- **Port:** http://localhost:8501

### 2. Quarterly Outlook Forecaster (`outlook_forecaster.py`)
- Quarterly revenue projection with fiscal year calendar (July-June)
- Enhanced spike detection for subscription renewals
- MAPE-based model selection with dynamic chart updates
- Comprehensive Excel export functionality
- **Launcher:** `RUN_OUTLOOK_FORECASTER.bat`
- **Port:** http://localhost:8502

## Installation Options

### 🎯 Professional Installer (Recommended)
```
Forecasting_Apps_Suite_Setup.exe
```

This **professional Windows installer** provides:
- Automatic installation to Program Files
- Start Menu and Desktop shortcuts  
- Python dependency checking
- Optional automatic setup of Python packages
- Professional uninstall support
- Windows integration

### 🔧 Manual Setup Command
```bash
SETUP.bat
```

This **single command** installs ALL dependencies for both applications:

#### Core Dependencies
- `streamlit>=1.28.0` - Web framework
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0,<1.25.0` - Numerical computing (version controlled for pmdarima compatibility)
- `scikit-learn>=1.0.0` - Machine learning
- `statsmodels>=0.13.0` - Statistical models

#### Excel Export (Enhanced)
- `openpyxl>=3.0.0` - Primary Excel engine (most stable)
- `xlsxwriter>=3.0.0` - Alternative Excel engine (better formatting)
- `xlrd>=2.0.0` - Read older Excel formats
- `pyxlsb>=1.0.0` - Read .xlsb files

#### Visualization
- `altair>=4.2.0` - Streamlit's preferred charting
- `matplotlib>=3.5.0` - Traditional plotting
- `plotly>=5.0.0` - Interactive charts
- `seaborn>=0.11.0` - Statistical visualization

#### Advanced Forecasting Models
- `scipy>=1.9.0,<1.12.0` - Scientific computing (compatible version for pmdarima)
- `pmdarima>=2.0.4` - Auto-ARIMA implementation
- `prophet>=1.1.0` - Facebook Prophet (may fail on some systems)
- `lightgbm>=3.0.0` - Gradient boosting
- `xgboost>=1.5.0` - Extreme gradient boosting
- `Cython>=0.29.0` - Build dependency for pmdarima

## Features Installed

### ✅ All Forecasting Models (7+ algorithms)
1. **Linear/Polynomial Regression** - Traditional statistical approach
2. **Moving Averages with Trend** - Simple trend-following
3. **SARIMA/ETS** - Seasonal decomposition (statsmodels)
4. **Auto-ARIMA** - Automated ARIMA selection (pmdarima)
5. **Prophet** - Facebook's time series model (seasonal + holidays)
6. **LightGBM** - Fast gradient boosting
7. **XGBoost** - Extreme gradient boosting

### ✅ Enhanced Excel Export
- Multi-engine support (openpyxl + xlsxwriter)
- Robust fallback mechanisms
- Multi-sheet exports with detailed model comparisons
- Individual line and consolidated data exports

### ✅ Advanced Analytics
- Spike detection for subscription renewals
- MAPE-based model selection
- Dynamic chart and metrics updates
- Fiscal year calendar support

### ✅ User Experience
- No email signup required
- Auto browser opening
- Local-only access (127.0.0.1)
- Comprehensive error handling
- Different ports for each app to run simultaneously

## File Structure

### Setup Files
- `SETUP.bat` - **Main setup command** (single source of truth)
- `requirements_portable.txt` - Updated unified requirements with version constraints
- `test_packages.py` - Package verification script

### Application Files
- `RUN_FORECAST_APP.bat` - Main app launcher (parent folder)
- `RUN_OUTLOOK_FORECASTER.bat` - Outlook app launcher (parent folder)
- `Forecaster App/forecaster_app.py` - Main forecasting app
- `Quarter Outlook App/outlook_forecaster.py` - Quarterly outlook forecaster

### Verification & Troubleshooting
- `test_packages.py` - Comprehensive package verification
- `Help/VERIFY_INSTALLATION_UNIFIED.bat` - Enhanced verification for both apps
- `Help/CLEAN_SLATE.bat` - Clean installation reset

## Usage Instructions

### First Time Setup (5-15 minutes)
1. Double-click `SETUP.bat` in the main folder
2. Wait for installation to complete (may take 5-15 minutes)
3. Look for "UNIFIED SETUP COMPLETE!" message

### Running the Apps
- **Main Forecasting:** Double-click `RUN_FORECAST_APP.bat` (in main folder)
- **Outlook Forecasting:** Double-click `RUN_OUTLOOK_FORECASTER.bat` (in main folder)
- Both apps can run simultaneously on different ports

### Package Verification
- Run `python test_packages.py` to verify all packages
- Look for ✅ checkmarks indicating successful installation
- ❌ indicates missing packages that may need reinstallation

### Troubleshooting Common Issues

#### Setup Issues
- **Python not found:** Install Python from python.org (3.8+ recommended)
- **Permission errors:** Run Command Prompt as Administrator
- **Package conflicts:** Run `Help/CLEAN_SLATE.bat` then `SETUP.bat`

#### Prophet Installation Failures
- Common on Windows - apps will work with 6 other models
- Try: `pip install prophet --no-deps` if issues persist

#### pmdarima Issues
- Requires specific numpy/scipy versions
- Setup automatically handles version constraints

#### Excel Export Issues
- Multiple engines provide fallbacks (openpyxl → xlsxwriter → pandas)
- Most Excel issues resolve automatically

#### Port Conflicts
- Main app: http://localhost:8501
- Outlook app: http://localhost:8502
- If ports conflict, close other Streamlit instances

## Benefits of Unified Setup

1. **Single Command Installation** - One `SETUP.bat` installs everything
2. **Comprehensive Coverage** - All dependencies for both apps
3. **Enhanced Excel Support** - Multiple engines with robust fallbacks
4. **Version Management** - Handles package compatibility automatically
5. **Consistent Environment** - Same setup process for all users
6. **Easy Launching** - Both apps can be started from main folder
7. **Simultaneous Operation** - Run both apps at once on different ports

## Package Version Strategy

The setup uses specific version constraints to ensure compatibility:
- **numpy**: `>=1.21.0,<1.25.0` (pmdarima compatibility)
- **scipy**: `>=1.9.0,<1.12.0` (pmdarima compatibility)
- **pmdarima**: `>=2.0.4` (latest stable with auto-ARIMA)
- Other packages use minimum version requirements for stability

This approach minimizes conflicts while ensuring all forecasting models work correctly.

**Total Package Count:** 15+ packages
**Installation Time:** 5-15 minutes (depending on system and network)
**Success Rate:** High (works even if some advanced packages fail)

---

**Contact:** Jake Moura (jakemoura@microsoft.com)
**Last Updated:** June 2025
