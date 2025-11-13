# Release Notes - Version 1.3.0

**Release Date**: October 8, 2025

## 🎯 What's New

### Smart Fiscal Year YoY Calculations
The headline feature of v1.3.0 is the introduction of **accurate fiscal year-over-year growth calculations**. Previously, all fiscal years displayed the same YoY percentage (copying from the first year). Now, each fiscal year correctly calculates its own unique year-over-year growth:

- **FY2027 YoY%** = (FY2027 total revenue / FY2026 total revenue - 1) × 100
- **FY2028 YoY%** = (FY2028 total revenue / FY2027 total revenue - 1) × 100
- **FY2029 YoY%** = (FY2029 total revenue / FY2028 total revenue - 1) × 100

This provides accurate growth trajectories for multi-year forecasting and planning.

## 🐛 Critical Bug Fixes

### UnboundLocalError Resolution
Fixed a critical bug that prevented fiscal year adjustment controls from loading. The issue was caused by:
- Incorrect indentation in the `target_yoy` input code block
- Variables being referenced before assignment due to scope issues
- Missing initialization of `distribution_method` variable

**Impact**: Users can now successfully access and use multi-fiscal year YoY growth target controls without encountering errors.

### Variable Initialization Improvements
Enhanced variable initialization throughout the adjustment workflow:
- Added default initialization for `distribution_method` 
- Fixed scope management in fiscal year control loops
- Improved error handling for edge cases

## 🔧 Enhancements

### Optimized Setup Process
The `SETUP.bat` installation script has been completely overhauled:
- **Streamlined installation sequence**: Packages install in the correct dependency order
- **Better error handling**: Installation continues even if optional packages fail
- **Python version compatibility**: Optimized for Python 3.11 and 3.12
- **Faster installation**: Removed unnecessary compilation flags, uses pre-built wheels when available

### New Helper Scripts & Documentation
Added comprehensive Python environment tools:
- **`CHECK_PYTHON_VERSION.bat`**: Verify your Python version is compatible
- **`FIND_PYTHON.bat`**: Locate Python installations on your system
- **`FIX_PMDARIMA.bat`**: Quick fix for pmdarima installation issues
- **`PYTHON_PATH_FIX.md`**: Step-by-step guide for PATH configuration
- **`PYTHON_VERSION_GUIDE.md`**: Detailed Python version recommendations

### Configuration Cleanup
- Streamlined Streamlit configuration files for better initial setup
- Removed redundant settings
- Cleaner default configurations

## 📊 Technical Details

### Code Changes
**File**: `Forecaster App/modules/ui_components.py`

**Function**: `analyze_fiscal_year_coverage()`
- Added `fy_specific_yoy` dictionary calculation
- Combines actual and forecast data to get complete fiscal year totals
- Calculates YoY for each fiscal year by comparing to previous year
- Returns individual YoY percentages in the analysis results

**Function**: `create_multi_fiscal_year_adjustment_controls()`
- Fixed indentation of `target_yoy` input block (moved outside conditional)
- Added `distribution_method = 'Smooth'` default initialization
- Improved error handling in input widgets

## 🔄 Upgrade Path

### Upgrading from v1.2.0
1. **Pull latest code** or download the new release
2. **Run `SETUP.bat`** to update packages (recommended but optional)
3. **No data migration needed** - fully backward compatible
4. **Test fiscal year controls** to see the new accurate YoY calculations

### Compatibility
- ✅ **Python 3.12**: Recommended (best compatibility)
- ✅ **Python 3.11**: Recommended (excellent compatibility)
- ✅ **Python 3.10**: Supported (good compatibility)
- ⚠️ **Python 3.13**: Partial (some packages may require compilation)
- ❌ **Python 3.14+**: Not recommended (too new, missing pre-built wheels)

### Breaking Changes
**None** - v1.3.0 is fully backward compatible with v1.2.0

## 📦 Installation

### New Installation
```batch
# Clone the repository
git clone https://github.com/jakemoura/Forecasting_App.git
cd Forecasting_App

# Run setup (installs all dependencies)
.\SETUP.bat

# Launch the app
.\Forecaster App\RUN_FORECAST_APP.bat
```

### Updating Existing Installation
```batch
# Pull latest changes
git pull origin main

# Update packages (optional but recommended)
.\SETUP.bat
```

## 🎓 Usage Examples

### Using the New Fiscal Year YoY Features

1. **Upload your data** with historical and forecast periods
2. **Navigate to the Forecast Results tab**
3. **Expand "Multi-Fiscal Year Growth Targets"** section
4. **View each fiscal year's unique YoY percentage**:
   - FY2027: Shows actual YoY vs FY2026
   - FY2028: Shows actual YoY vs FY2027
   - FY2029: Shows actual YoY vs FY2028
5. **Set growth targets** for each year independently
6. **Apply adjustments** to see forecasts updated with sequential compounding

### Before vs After v1.3.0

**Before (v1.2.0)**:
```
FY2027: Current: 29.8%
FY2028: Current: 29.8%  ← Incorrect (copied from FY2027)
FY2029: Current: 29.8%  ← Incorrect (copied from FY2027)
```

**After (v1.3.0)**:
```
FY2027: Current: 29.8%  (FY2027 vs FY2026)
FY2028: Current: 15.2%  (FY2028 vs FY2027) ← Correct calculation
FY2029: Current: 8.5%   (FY2029 vs FY2028) ← Correct calculation
```

## 🐛 Known Issues

None currently identified in v1.3.0.

If you encounter any issues, please:
1. Check the [troubleshooting guide](README.md#troubleshooting--best-practices)
2. Verify Python version compatibility
3. Review the [CHANGELOG](CHANGELOG.md)
4. Open an issue on GitHub

## 📚 Documentation Updates

- Updated **README.md** with v1.3.0 features
- Created **CHANGELOG.md** for version history
- Added **VERSION.txt** for easy version tracking
- New **RELEASE_NOTES.md** (this document)
- Enhanced setup guides in `Help/` folder

## 🙏 Acknowledgments

Thank you to all users who reported the fiscal year YoY calculation issues and the UnboundLocalError bugs. Your feedback directly led to these improvements!

## 📞 Support

For questions or issues:
- 📖 Read the [README.md](README.md)
- 📘 Check the [USER_GUIDE.html](USER_GUIDE.html)
- 🛠️ Review [SETUP_GUIDE.html](Help/SETUP_GUIDE.html)
- 🔍 See [CHANGELOG.md](CHANGELOG.md) for version history
- 🐛 Report issues on GitHub

---

**Happy Forecasting! 📈**

*The Forecasting Applications Team*
