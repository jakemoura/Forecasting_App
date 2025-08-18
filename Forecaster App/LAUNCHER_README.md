# 🚀 Forecaster App Launcher for macOS

## Quick Start

### **Option 1: Double-Click (Easiest)**
1. **Double-click** `LAUNCH_FORECASTER.sh` in Finder
2. **Allow execution** if macOS asks for permission
3. **Follow the prompts** in the terminal window

### **Option 2: Terminal (Recommended)**
1. **Open Terminal** (Applications → Utilities → Terminal)
2. **Navigate** to the Forecaster App folder:
   ```bash
   cd "/path/to/Forecaster App"
   ```
3. **Run the launcher**:
   ```bash
   ./LAUNCH_FORECASTER.sh
   ```

## 🔧 What the Launcher Does

### **Automatic Checks:**
- ✅ **Python 3**: Verifies Python is installed and accessible
- ✅ **Required Packages**: Checks for Streamlit and core dependencies
- ✅ **Optional Packages**: Identifies missing advanced model packages
- ✅ **File Validation**: Ensures all app files are present

### **Smart Installation:**
- **Auto-installs** missing core packages (Streamlit, Pandas, NumPy, Scikit-learn)
- **Warns about** missing optional packages (Prophet, Auto-ARIMA, LightGBM)
- **Continues working** with basic models if advanced packages are missing

### **User Experience:**
- **Clear feedback** about what's happening
- **Helpful error messages** if something goes wrong
- **Automatic browser launch** when the app starts
- **Easy shutdown** with Ctrl+C

## 📱 App Access

Once launched, you can access the Forecaster App:

- **🌐 Browser**: Automatically opens in your default browser
- **📱 Direct URL**: http://localhost:8501
- **🔄 Refresh**: Use Cmd+R to refresh the page

## 🛑 Stopping the App

### **In Terminal:**
- Press **Ctrl+C** to stop the app
- Press **Enter** to close the terminal window

### **In Browser:**
- Simply close the browser tab
- The app will continue running in the background until stopped in terminal

## 🔍 Troubleshooting

### **"Permission Denied" Error:**
```bash
chmod +x LAUNCH_FORECASTER.sh
```

### **Python Not Found:**
- Install Python 3 from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

### **Port Already in Use:**
- The launcher uses port 8501
- If busy, stop other Streamlit apps or change the port in the script

### **Package Installation Issues:**
```bash
# Update pip first
pip3 install --upgrade pip

# Install packages manually
pip3 install streamlit pandas numpy scikit-learn
```

## 🎯 Features You'll Get

### **Smart Backtesting:**
- **Data-driven recommendations** for validation periods
- **Automatic fallback** to MAPE rankings if backtesting fails
- **User control** over validation depth via intuitive slider

### **Business-Focused Design:**
- **Fast, reliable validation** (20x faster than academic methods)
- **Never fails completely** - always provides model recommendations
- **Intuitive interface** that business users understand

### **Enterprise-Grade Forecasting:**
- **Multiple models**: SARIMA, ETS, Polynomial, Prophet, Auto-ARIMA, LightGBM
- **Business adjustments**: Growth assumptions, market conditions
- **Statistical validation**: Prevents extreme outliers
- **Seasonal analysis**: Identifies monthly/quarterly patterns

## 📊 System Requirements

- **macOS**: 10.14 (Mojave) or later
- **Python**: 3.8 or later
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 500MB free space
- **Browser**: Chrome, Safari, Firefox, or Edge

## 🆘 Need Help?

### **Check the Logs:**
- The terminal shows detailed information about what's happening
- Look for error messages or warnings

### **Verify Installation:**
```bash
# Check Python
python3 --version

# Check packages
python3 -c "import streamlit; print('Streamlit OK')"
python3 -c "import pandas; print('Pandas OK')"
```

### **Common Issues:**
- **"Command not found"**: Python not in PATH
- **"Permission denied"**: Script not executable
- **"Port already in use"**: Another app using port 8501
- **"Module not found"**: Missing Python packages

## 🎉 Ready to Forecast!

The launcher makes it easy to get started with enterprise-grade forecasting. Just run it and let the smart backtesting guide you to optimal model selection!

**Happy Forecasting! 🎯📈**
