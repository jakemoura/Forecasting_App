# Advanced Forecasting Applications

Two professional-grade Streamlit applications for business revenue forecasting:

- **Forecaster App**: Multi-model time-series forecasting with rigorous backtesting, WAPE-optimized selection, and business-aware model ranking
- **Quarterly Outlook Forecaster**: Daily-to-quarterly projection with fiscal calendar support and renewal/capacity modeling

## Key Features

- **🏆 Best per Product Selection**: Intelligent hybrid approach using optimal models per product
- **🧪 Rigorous Backtesting**: Walk-forward validation with strict eligibility criteria (≥24 months, stability checks)
- **📊 WAPE-First Accuracy**: Revenue-aligned error measurement with business-relevant interpretations
- **🛡️ Business-Aware Safeguards**: Deprioritizes polynomial models for revenue forecasting scenarios
- **🔮 Advanced Models**: SARIMA (AIC/BIC), ETS, Prophet (holidays), Auto-ARIMA, LightGBM, Seasonal-Naive
- **⚙️ Multi-Metric Ranking**: Robust selection across WAPE, SMAPE, MASE, RMSE for statistical confidence
- **🎯 Interactive Adjustments**: Management overrides, growth assumptions, scenario planning

## Repo structure

- Forecaster App/: Monthly forecaster UI and modules
- Quarter Outlook App/: Outlook (daily → quarterly) forecaster UI and modules
- Help/: Setup scripts, requirements, installer resources
- App/, Installer.lnk, RUN_*.bat: Windows-friendly launchers and helpers

## Quick Start (Windows)

**📁 File Explorer Method (Recommended):**
- **Forecaster App**: Double-click `Forecaster App/RUN_FORECAST_APP.bat`
- **Quarterly Outlook**: Double-click `Quarter Outlook App/RUN_OUTLOOK_FORECASTER.bat`

**⚠️ Important**: Run .BAT files from File Explorer, not web browsers.

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

**📊 Model Selection Issues:**
- **Short history (<24 months)**: App automatically falls back to Best per Product (Standard)
- **High WAPE (>30%)**: Apply interactive adjustments, check data quality, or extend historical period
- **Polynomial warnings**: Business-aware selection is enabled by default for revenue forecasting

**🔧 Technical Issues:**
- **OneDrive file locks**: Close Excel/apps using files, retry after a few seconds
- **BAT file errors**: Run from File Explorer, not browsers; check Windows execution policy
- **Package conflicts**: Use provided requirements files and clean Python environment

**💡 Best Practices:**
- **Default approach**: Use Best per Product (Backtesting) for highest accuracy
- **Composite vs individual**: Composite models (Best per Product) often outperform single models
- **Interactive adjustments**: Apply management overrides for business strategy changes
- **Model comparison**: Use dropdown to compare approaches; WAPE displayed for transparency
