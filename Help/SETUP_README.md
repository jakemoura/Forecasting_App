# Setup Options

## Main Installation
- **SETUP.bat** (in parent folder) - Full installation with all advanced models
  - Includes Prophet, pmdarima, LightGBM, XGBoost
  - Takes 10-15 minutes
  - Provides maximum forecasting accuracy

## Quick Alternative
- **SETUP_QUICK.bat** (this folder) - Backup quick installation
  - Core packages + LightGBM + XGBoost only
  - Skips Prophet and pmdarima (complex installations)
  - Takes 3-5 minutes
  - Good for testing or when full installation fails

## Python Installation Required
Both setups will guide you to install Python if not found:
- Microsoft Store (recommended for Windows 10/11)
- Python.org (traditional method)

## Other Tools
- **CLEAN_SLATE.bat** - Remove all packages and start fresh
- **VERIFY_INSTALLATION_UNIFIED.bat** - Test what's installed
- **CHECK_INSTALLATION.bat** - Quick package check
