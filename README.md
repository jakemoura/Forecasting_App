# Forecasting Apps

Two Streamlit applications for business forecasting:

- Forecaster App: multi-model, monthly time-series forecasting with statistical validation and business-aware selection.
- Quarterly Outlook Forecaster: quick quarter projection from partial in-quarter daily data with capacity/renewal adjustments.

## Features

- Multi-model comparison: SARIMA/ETS, Auto-ARIMA, Prophet, LightGBM, Polynomial baselines
- Advanced validation: walk-forward validation and time-series cross-validation (MAPE, SMAPE, MASE, RMSE)
- Auto (per product) selection: choose Standard vs Backtesting based on lower validation error
- Business adjustments: growth assumptions, market multipliers, fiscal calendar awareness
- Outlook-specific: capacity constraints, renewal spike detection, monthly breakdown, Excel exports

## Repo structure

- Forecaster App/: Monthly forecaster UI and modules
- Quarter Outlook App/: Outlook (daily → quarterly) forecaster UI and modules
- Help/: Setup scripts, requirements, installer resources
- App/, Installer.lnk, RUN_*.bat: Windows-friendly launchers and helpers

## Quick start (Windows)

Most users should run the apps using the provided .BAT files from File Explorer (not from a browser):

- Run Forecaster App: double-click `Forecaster App/RUN_FORECAST_APP.bat` (or repo-root `RUN_FORECAST_APP.bat`)
- Run Quarterly Outlook Forecaster: double-click `Quarter Outlook App/RUN_OUTLOOK_FORECASTER.bat` (or repo-root `RUN_OUTLOOK_FORECASTER.bat`)

See `USER_GUIDE.html` for screenshots and workflow tips.

## Data requirements (summary)

- Forecaster App: regular time index (monthly/weekly/daily), product identifiers; ≥ 12–18 periods recommended
- Quarterly Outlook Forecaster: daily data with columns `Date`, `Product`, `ACR`; fiscal calendar supported (e.g., July–June)

## Backtesting guidance

- Validation method
  - Automatic (recommended): app picks the most reliable signal
  - Walk-forward: simulates month-by-month forecasting
  - Cross-validation: checks multiple time splits for stability
- Leakage gap (months): 0–1 for most; 2 if there’s known reporting lag/end-of-month spikes
- Validation horizon (months)
  - 12: default; captures full seasonality
  - 6: near-term focus or shorter history
  - 3: very near-term planning
  - 18–24: only with lots of history and strong seasonality
- Data needs: walk-forward typically needs ~`24 + gap + horizon` months of history; if short, the app reduces folds or falls back to Standard

## Developer setup (optional)

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

## Troubleshooting

- OneDrive file locks can occasionally interfere with Git or app runs; retry after a few seconds or close apps using the files
- If backtesting is unavailable for a product, history is likely too short—use Standard or Auto mode
