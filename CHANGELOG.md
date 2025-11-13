# Changelog

All notable changes to the Forecasting Applications will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-10-08

### 🎯 Added
- **Smart Fiscal Year YoY Calculations**: Each fiscal year now accurately calculates its own year-over-year growth percentage
  - FY2027 compares against FY2026
  - FY2028 compares against FY2027
  - FY2029 compares against FY2028
- **Enhanced fiscal year analysis**: New `fy_specific_yoy` calculation in `analyze_fiscal_year_coverage()`
- **Python version compatibility documentation**: Added comprehensive guides for Python 3.11-3.12 compatibility
- **New helper scripts**: 
  - `CHECK_PYTHON_VERSION.bat` - Verify Python version compatibility
  - `FIND_PYTHON.bat` - Locate Python installations
  - `FIX_PMDARIMA.bat` - Quick fix for pmdarima installation issues
  - `PYTHON_PATH_FIX.md` - Guide for PATH configuration
  - `PYTHON_VERSION_GUIDE.md` - Detailed Python version recommendations

### 🐛 Fixed
- **UnboundLocalError in fiscal year controls**: Fixed critical bug caused by incorrect indentation in `target_yoy` input code block
- **Distribution method initialization**: Added default initialization to prevent undefined variable errors
- **Fiscal year YoY display bug**: Fixed issue where all fiscal years showed the same YoY percentage (copying from first year)
- **Variable scope issues**: Improved variable initialization and scope management in adjustment controls

### 🔧 Changed
- **Optimized SETUP.bat**: Improved package installation sequence with better error handling
- **Streamlit configuration**: Cleaned up config files for better initial setup experience
- **Package installation order**: numpy and scipy now install before pmdarima to prevent build issues

### 📝 Documentation
- Updated README.md with v1.3.0 features and fixes
- Added comprehensive troubleshooting guides for Python version compatibility
- Enhanced setup documentation with Python 3.11-3.12 recommendations

---

## [1.2.0] - 2025-09-XX

### 🎯 Added
- **Sequential YoY Compounding**: Multi-fiscal year adjustments now properly compound across sequential years
- **Proper Baseline Calculation**: Each year builds from previous year's adjusted values
- **Universal Product Support**: All products apply fiscal year changes to charts AND export data

### 🐛 Fixed
- **MediaFileStorageError Prevention**: Automatic session state cleanup prevents crashes
- **Product Adjustment Failures**: All products now handle adjustments identically
- **File Reference Issues**: Automatic detection and cleanup of invalid session state references

### 🔧 Changed
- **Enhanced Error Handling**: Comprehensive error handling throughout adjustment workflow
- **Improved Reliability**: Better crash prevention and data corruption safeguards

---

## [1.1.0] - 2025-08-XX

### 🎯 Added
- **Live Conservatism Feature**: Real-time forecast adjustment slider (90-110%)
- **Enhanced Backtesting Visualization**: Purple dotted lines showing actual backtesting predictions
- **Fiscal Calendar Integration**: July-June fiscal year support in Quarterly Outlook
- **Renewal Pattern Detection**: Automatic spike detection and monthly renewal forecasting

### 🔧 Changed
- **WAPE-First Accuracy**: Revenue-aligned error measurement as primary metric
- **Business-Aware Safeguards**: Stability checks and reasonableness penalties
- **Multi-Metric Validation**: Robust selection across WAPE, SMAPE, MASE, RMSE

---

## [1.0.0] - 2025-07-XX

### 🎯 Initial Release
- **Forecaster App**: Multi-model time-series forecasting with monthly/weekly data support
- **Quarter Outlook Forecaster**: Daily-to-quarterly projection with fiscal calendar
- **Advanced Models**: SARIMA, ETS, Prophet, Auto-ARIMA, LightGBM, Seasonal-Naive
- **Backtesting Framework**: Walk-forward validation with configurable parameters
- **Excel/CSV Export**: Comprehensive data export with fiscal period formatting
- **Interactive Charts**: Altair-based visualizations with confidence intervals

---

## Version History Summary

| Version | Release Date | Key Features |
|---------|--------------|--------------|
| 1.3.0 | 2025-10-08 | Fiscal Year Intelligence, Bug Fixes, Enhanced Setup |
| 1.2.0 | 2025-09-XX | Sequential YoY Compounding, Universal Product Support |
| 1.1.0 | 2025-08-XX | Live Conservatism, Enhanced Backtesting Visualization |
| 1.0.0 | 2025-07-XX | Initial Release with Core Features |

---

## Upgrade Notes

### Upgrading to 1.3.0
- **Recommended**: Use Python 3.11 or 3.12 for best compatibility
- **Action Required**: Run `SETUP.bat` to update packages with optimized installation sequence
- **Breaking Changes**: None - fully backward compatible
- **Data Migration**: No data migration needed

### Upgrading from 1.1.x to 1.2.0
- **Action Required**: Test multi-fiscal year adjustments to verify sequential compounding
- **Data Migration**: Existing session state will be automatically cleaned up

### Upgrading from 1.0.x to 1.1.0
- **Action Required**: Review WAPE interpretations (new business-aligned thresholds)
- **New Feature**: Explore Live Conservatism slider in sidebar

---

## Support & Troubleshooting

For issues or questions:
1. Check the [README.md](README.md) troubleshooting section
2. Review [SETUP_GUIDE.html](Help/SETUP_GUIDE.html) for installation help
3. See [USER_GUIDE.html](USER_GUIDE.html) for workflow guidance
4. Check Python version compatibility with `Help/CHECK_PYTHON_VERSION.bat`

---

[1.3.0]: https://github.com/jakemoura/Forecasting_App/releases/tag/v1.3.0
[1.2.0]: https://github.com/jakemoura/Forecasting_App/releases/tag/v1.2.0
[1.1.0]: https://github.com/jakemoura/Forecasting_App/releases/tag/v1.1.0
[1.0.0]: https://github.com/jakemoura/Forecasting_App/releases/tag/v1.0.0
