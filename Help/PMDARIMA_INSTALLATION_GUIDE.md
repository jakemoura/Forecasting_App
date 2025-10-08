# pmdarima (Auto-ARIMA) Installation Guide

## Problem Overview
pmdarima is an advanced forecasting package that can be challenging to install on newer Python versions, especially Python 3.13+.

## Quick Fix Solutions

### Solution 1: Use Pre-built Wheels (FASTEST - Recommended)
Run the updated `FIX_PMDARIMA.bat` script:
```batch
.\FIX_PMDARIMA.bat
```

This will:
1. Install numpy and scipy in the correct order
2. Attempt to use pre-built wheels (no compilation needed)
3. Try multiple installation methods automatically

### Solution 2: Manual Installation
Open PowerShell or Command Prompt and run:
```powershell
# Install build dependencies first
python -m pip install "numpy>=1.21.0"
python -m pip install "scipy>=1.9.0"
python -m pip install "Cython>=0.29.0,<3.0.0"

# Install pmdarima using pre-built wheels
python -m pip install pmdarima
```

### Solution 3: Use Compatible Python Version
**HIGHLY RECOMMENDED for best compatibility:**
- **Python 3.11 or 3.12**: Best compatibility with all packages
- **Python 3.10**: Also works well
- **Python 3.13**: Newest version, but some packages may not have pre-built wheels yet

To check your Python version:
```powershell
python --version
```

## Common Errors and Solutions

### Error: "ModuleNotFoundError: No module named 'numpy'"
**Cause**: numpy needs to be installed before pmdarima can build

**Solution**: Install numpy first, then pmdarima:
```powershell
python -m pip install numpy
python -m pip install pmdarima
```

### Error: "scipy version incompatible (missing _lazywhere)"
**Cause**: Installed scipy version is too old

**Solution**: Upgrade scipy:
```powershell
python -m pip install "scipy>=1.9.0" --upgrade
```

### Error: "Building wheel for pmdarima (pyproject.toml) ... error"
**Cause**: Trying to build from source but missing compiler or using Python 3.13

**Solutions**:
1. **Use pre-built wheels** (add `--only-binary=:all:`):
   ```powershell
   python -m pip install pmdarima --only-binary=:all:
   ```

2. **Downgrade to Python 3.12 or 3.11** (recommended)

3. **Install Microsoft C++ Build Tools** (if you must build from source):
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++" workload

### Error: "ERROR: Failed building wheel for pmdarima"
**Cause**: Python 3.13 compatibility issues

**Solutions**:
1. Try specific version with pre-built wheels:
   ```powershell
   python -m pip install pmdarima==2.0.4
   ```

2. Try without build isolation:
   ```powershell
   python -m pip install pmdarima --no-build-isolation
   ```

3. Use older pmdarima version:
   ```powershell
   python -m pip install "pmdarima>=1.8.0,<2.0.0"
   ```

## Recommended Python Versions

| Python Version | pmdarima Compatibility | Recommendation |
|---------------|------------------------|----------------|
| 3.13.x | ⚠️ Partial (pre-built wheels may not be available) | Wait for package updates or use 3.12 |
| 3.12.x | ✅ Excellent (pre-built wheels available) | **RECOMMENDED** |
| 3.11.x | ✅ Excellent (pre-built wheels available) | **RECOMMENDED** |
| 3.10.x | ✅ Good (fully supported) | Good choice |
| 3.9.x or older | ⚠️ May work but outdated | Upgrade to 3.11+ |

## Installation Scripts Available

1. **FIX_PMDARIMA.bat** (Root folder)
   - Quick fix for pmdarima installation issues
   - Tries multiple installation methods
   - Non-interactive, fast

2. **Help\FORCE_INSTALL_PMDARIMA.bat**
   - Comprehensive installation with full cleanup
   - Uninstalls existing packages first
   - Interactive, thorough

3. **Help\INSTALL_PMDARIMA_PYTHON311.bat**
   - Optimized for Python 3.11+
   - Uses pre-built wheels when available
   - Fast installation

4. **SETUP.bat** (Updated)
   - Now includes improved pmdarima installation
   - Tries pre-built wheels first
   - Falls back to specialized installers if needed

## Testing Your Installation

After installation, test pmdarima:
```powershell
python -c "import pmdarima; print('pmdarima version:', pmdarima.__version__)"
```

Test auto_arima functionality:
```powershell
python -c "import pmdarima as pm; import numpy as np; data = np.random.randn(30).cumsum(); model = pm.auto_arima(data, start_p=0, start_q=0, max_p=1, max_q=1, seasonal=False, stepwise=True, suppress_warnings=True); print('auto_arima test: PASSED')"
```

## Alternative: App Works Without pmdarima

If pmdarima installation continues to fail, **the forecasting app will still work** with these 6 other models:
- Linear Regression
- Prophet (Facebook's forecasting tool)
- LSTM (Deep Learning)
- XGBoost (Gradient Boosting)
- LightGBM (Light Gradient Boosting)
- SARIMA (Seasonal ARIMA - manual configuration)

Only Auto-ARIMA will be unavailable.

## Getting Help

If none of these solutions work:
1. Check your Python version: `python --version`
2. Consider using Python 3.12 or 3.11 for best compatibility
3. Run `Help\CHECK_INSTALLATION.bat` to diagnose issues
4. Check the error messages carefully - they often suggest specific solutions

## Updated Files

The following files have been updated to fix pmdarima installation:
- ✅ `SETUP.bat` - Improved installation sequence
- ✅ `FIX_PMDARIMA.bat` - New quick fix script
- ✅ `Help\FORCE_INSTALL_PMDARIMA.bat` - Enhanced with Python 3.13 support
- ✅ `Help\INSTALL_PMDARIMA_PYTHON311.bat` - Already optimized for Python 3.11+
