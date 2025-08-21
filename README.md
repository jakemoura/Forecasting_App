# Advanced Forecasting Applications

Two professional-grade Streamlit applications for business revenue forecasting:

- **Forecaster App**: Multi-model time-series forecasting with rigorous backtesting, WAPE-optimized selection, and business-aware model ranking
- **Quarter Outlook Forecaster**: Daily-to-quarterly projection with enhanced backtesting validation, WAPE-first accuracy, fiscal calendar support, and renewal/capacity modeling

## Key Features

### Core Capabilities (Both Apps)
- **🏆 Enhanced Model Selection**: Intelligent backtesting-first approach with fallback to multi-metric ranking
- **🧪 Rigorous Backtesting**: Walk-forward validation optimized per application (monthly vs. daily data)
- **📊 WAPE-First Accuracy**: Revenue-aligned error measurement with business-relevant interpretations
- **🛡️ Business-Aware Safeguards**: Stability checks and reasonableness penalties for reliable forecasts
- **📈 Visual Backtesting**: Interactive charts showing validation points and performance metrics
- **⚙️ Multi-Metric Validation**: Robust selection across WAPE, SMAPE, MASE, RMSE for statistical confidence

### Forecaster App (Monthly/Weekly Data)
- **🔮 Advanced Models**: SARIMA (AIC/BIC), ETS, Prophet (holidays), Auto-ARIMA, LightGBM, Seasonal-Naive
- **🎯 Interactive Adjustments**: Management overrides, growth assumptions, scenario planning
- **📋 Comprehensive Reporting**: Product-level breakdowns with confidence intervals

### Quarter Outlook Forecaster (Daily Data)
- **🗓️ Fiscal Calendar Integration**: Configurable fiscal years (e.g., July-June quarters)
- **🏆 Sophisticated Quarterly Backtesting**: Advanced walk-forward validation with business-oriented rules
- **📈 Visual Backtesting Charts**: Purple dotted lines show actual backtesting prediction trends
- **🔄 Renewal Pattern Detection**: Automatic spike detection and monthly renewal forecasting
- **📊 Streamlined Models**: Core models optimized for daily quarterly forecasting (Run Rate, Linear Trend, Exponential Smoothing, Moving Average, Monthly Renewals)
- **⚡ Intelligent Model Selection**: Only backtested models allowed (overfitting protection)

## Repo structure

- Forecaster App/: Monthly forecaster UI and modules
- Quarter Outlook App/: Outlook (daily → quarterly) forecaster UI and modules
- Help/: Setup scripts, requirements, installer resources
- App/, Installer.lnk, RUN_*.bat: Windows-friendly launchers and helpers

## Quick Start (Windows)

**📁 File Explorer Method (Recommended):**
- **Forecaster App**: Double-click `Forecaster App/RUN_FORECAST_APP.bat`
- **Quarter Outlook Forecaster**: Double-click `Quarter Outlook App/RUN_OUTLOOK_FORECASTER.bat`

**⚠️ Important**: Run .BAT files from File Explorer, not web browsers.

**🚀 New Features**: Both apps now feature enhanced backtesting validation, WAPE-first accuracy, and visual chart indicators showing validation performance.

**📖 Documentation**: See `USER_GUIDE.html` for comprehensive workflow guidance and `Help/SETUP_GUIDE.html` for installation.

## Data Requirements

**Forecaster App:**
- **Format**: Excel/CSV with Date, Product, ACR columns
- **Time index**: Regular monthly/weekly/daily (gaps handled automatically)
- **History**: ≥24 months recommended for backtesting eligibility; ≥12 months minimum
- **Products**: Consistent identifiers across time periods

**Quarterly Outlook Forecaster:**  
- **Format**: Daily data with Date, Product, ACR columns
- **Fiscal calendar**: Configurable (e.g., July-June quarters)
- **Coverage**: Current quarter partial data + historical quarters for model training

## Model Selection & Backtesting

### Forecaster App (Monthly/Weekly Data)

**🏆 Best per Product (Backtesting) - Default & Recommended:**
- **Process**: Rigorous walk-forward validation per product with strict eligibility
- **Eligibility**: ≥24 months history, ≥2 backtesting folds, MASE < 1.0, ≥5% better than Seasonal-Naive
- **Scoring**: Primary WAPE → p75 WAPE → MASE → recent worst-month (tie-breaking hierarchy)
- **Safeguards**: Stability checks (p95 WAPE ≤ 2× mean), polynomial deprioritization for revenue

**📊 Best per Product (Standard) - Fallback:**
- **Process**: Multi-metric ranking across WAPE, SMAPE, MASE, RMSE
- **Use case**: When backtesting eligibility insufficient (<24 months history)
- **Selection**: Best average rank across all validation metrics per product

**🎯 Individual Models:**
- **Purpose**: Consistency across products or domain expertise applications  
- **Options**: SARIMA, ETS, Prophet, Auto-ARIMA, LightGBM, Seasonal-Naive
- **Selection**: Choose when you prefer single-model interpretation

**⚙️ Backtesting Configuration:**
- **Gap**: 0 months (default) | 1-2 months if autocorrelation/lag concerns
- **Horizon**: 6 months (mimics real forecasting) | configurable based on planning needs
- **Step size**: 6 months (captures seasonal cycles)
- **Fallback**: Automatic degradation to Standard selection when history insufficient

### Quarter Outlook Forecaster (Daily Data)

**🏆 Sophisticated Quarterly Backtesting - Default & Recommended:**
- **Training Window**: Rolling 180-365 days (minimum 180, default 365, never <90 days)
- **Validation Folds**: 8-12 weekly folds per quarter (preferably Fridays), starting near end of history
- **Dynamic Horizon**: Forecast from origin date through quarter-end (not fixed horizons)
- **Gap Protection**: `max(lag_days, rolling_window_days)` to prevent data leakage
- **Recency Weighting**: Exponential decay with 2-quarter half-life + 28-day current quarter weighting
- **Multi-Tier Metrics**: Primary WAPE on remaining-quarter sum, Secondary WAPE on quarter total, Tertiary daily MASE
- **EOQ Penalty**: 1.25x penalty if last 5 business days error exceeds 30% threshold
- **Strict Selection**: Only backtested models allowed (no fallback to non-validated models)

**📈 Enhanced Daily Backtesting - Fallback for Shorter History:**
- **Process**: Optimized for 14-179 days of data with 2-day horizon validation
- **Configuration**: 7-day windows, heavy weighting of recent performance  
- **Scoring**: WAPE-first with exponential weighting favoring recent validation folds
- **Models**: Same streamlined set with faster validation cycles

**📊 Standard Mode - Final Fallback:**
- **Process**: Multi-metric ranking when <14 days of data available
- **Use case**: Insufficient data for meaningful backtesting (very rare)
- **Validation**: Basic statistical validation with stability checks

**🎯 Visual Integration:**
- **🔺 Green Triangles**: Backtesting validation start points (where each fold begins prediction)
- **📈 Purple Dotted Lines**: Actual backtesting prediction trends from each validation fold
- **📊 Chart Titles**: Both Standard and Backtesting WAPE displayed with fold counts
- **🎯 Interactive Dropdown**: Real-time model comparison with performance metrics
- **📋 Enhanced Legends**: Clear explanations of all chart elements (historical, forecast, spikes, validation)
- **✅ Validation Indicators**: "✓ Walk-Forward Validated" badge in chart titles
- **📈 Backtesting Breakdown**: Detailed fold-by-fold validation results in expandable section

## WAPE Accuracy & Interpretation

**📊 Primary Metric: WAPE (Weighted Absolute Percentage Error)**
- **Formula**: `sum(|Actual - Forecast|) / sum(|Actual|)`
- **Advantage**: Revenue-aligned (dollar-weighted), robust to zero/small values
- **Business relevance**: 15% WAPE = forecasts typically within 15% of actual revenue

**🎯 Accuracy Interpretation:**
- **0-10%**: Excellent accuracy (professional-grade forecasting)
- **10-20%**: Good accuracy (acceptable for most business decisions)
- **20-30%**: Moderate accuracy (use caution for critical decisions)
- **30%+**: Lower accuracy (consider manual adjustments or business overrides)

**🔧 When WAPE is High:**
- Apply interactive adjustments for management overrides
- Check for outliers or data quality issues  
- Consider longer historical periods if available
- Review polynomial model warnings (deprioritized automatically)

## Developer Setup (Optional)

If you prefer running from source instead of the .BAT files:

1) Python environment

```powershell
# Windows PowerShell
Set-Location "C:\\Users\\jakemoura\\OneDrive - Microsoft\\Desktop\\Forecasting_App"
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install -r .\\Help\\requirements_portable.txt
```

2) Run apps

```powershell
# Forecaster App (monthly)
streamlit run ".\\Forecaster App\\forecaster_app.py"

# Quarterly Outlook Forecaster (daily → quarterly)
streamlit run ".\\Quarter Outlook App\\outlook_forecaster.py"
```

## Contributing

- Create a feature branch: `git checkout -b feat/<short-name>`
- Commit with Conventional Commits (e.g., `feat(ui): add backtesting explainer`)
- Open a Pull Request to main

## Large/binary files

Use Git LFS if you must version Excel/installer binaries:

```powershell
git lfs install
git lfs track "*.xls" "*.xlsx" "*.xlsb" "*.zip"
```

## License & notices

- See `Help/LICENSE.txt` for licensing details
- Do not commit confidential data or secrets. Streamlit secrets (if used): `.streamlit/secrets.toml` (ignored by default)

## Troubleshooting & Best Practices

### Forecaster App
**📊 Model Selection Issues:**
- **Short history (<24 months)**: App automatically falls back to Best per Product (Standard)
- **High WAPE (>30%)**: Apply interactive adjustments, check data quality, or extend historical period
- **Polynomial warnings**: Business-aware selection is enabled by default for revenue forecasting

### Quarter Outlook Forecaster
**📊 Daily Quarterly Forecasting Issues:**
- **Sophisticated backtesting not running**: Requires ≥180 days of data; falls back to Enhanced (≥14 days) or Standard (<14 days)
- **High quarterly backtesting WAPE**: Check EOQ penalty impact, review training window (180-365 days), verify data quality in recent periods
- **Limited validation folds**: Optimal is 8-12 weekly folds; fewer folds may indicate insufficient recent history
- **Purple lines not showing**: Indicates validation fold issues; check backtesting breakdown section for detailed fold results
- **Spike detection**: Ensure sufficient historical data (≥30 days) for reliable monthly renewal pattern detection
- **Model selection**: Only backtested models allowed; if all models fail backtesting, app falls back to 'Run Rate'

### General (Both Apps)
**🔧 Technical Issues:**
- **OneDrive file locks**: Close Excel/apps using files, retry after a few seconds
- **BAT file errors**: Run from File Explorer, not browsers; check Windows execution policy
- **Package conflicts**: Use provided requirements files and clean Python environment

**💡 Best Practices:**
- **Default approach**: Sophisticated Quarterly Backtesting provides optimal accuracy with overfitting protection
- **Chart interpretation**: 
  - 🔺 **Green triangles** = validation start points (where backtesting predictions begin)
  - 📈 **Purple dotted lines** = actual prediction trends from each validation fold
  - 📊 **Chart titles** = show both Standard and Backtesting WAPE with fold counts
- **Performance validation**: Quarterly backtesting uses business-aware metrics (remaining-quarter WAPE, EOQ penalties)
- **Model reliability**: Only validated models allowed - ensures production-ready forecasts
- **Visual validation**: Enhanced charts reveal model behavior during actual backtesting periods
- **Backtesting breakdown**: Expand detailed section to see fold-by-fold validation performance
