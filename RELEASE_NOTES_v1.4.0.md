# Release Notes - Version 1.4.0

**Release Date**: November 13, 2025

## 🎯 What's New

### Live Fiscal-Year Totals Everywhere
- Forecast dashboard now surfaces **per-product fiscal-year totals** directly in the results view, combining actuals, forecast, and non-compliant revenue in real time.
- The fiscal-year growth target panel shows **live totals inside each product expander**, updating as you tweak YoY percentages so planners immediately see the dollar impact.

### Boundary-Aware Growth Adjustments
- Added an optional **"Smooth fiscal year boundary transitions"** toggle that blends July projections with the prior fiscal year when YoY targets change, reducing MoM spikes.
- Manual and fiscal-year adjustments now refresh instantly and keep totals aligned with backtesting-driven models.

## 🔧 Enhancements

### Streamlined Forecast Review
- Model selector sits directly above the product chart for quicker context switching.
- Forecast download names are concise (`BaseName_Model_YYYYMMDD_HHMMSS.xlsx`), eliminating Windows path-length issues while still flagging adjustments.

### Adjustment Workflow Polish
- Fiscal-year growth adjustments only trigger smoothing when user overrides are active, preserving pure statistical curves by default.
- Manual adjustment persistence and reruns have been hardened to prevent stale KPIs when sliders return to zero.

## 🐛 Bug Fixes

- Fixed fiscal-year totals and adjustments so downloads and UI widgets stay synchronized after toggling between manual and FY targets.
- Eliminated rare dimension mismatches when applying sequential adjustments across multiple fiscal years.

## 📊 Technical Details

| Area | Description |
| --- | --- |
| `modules/ui_components.py` | Reworked `display_forecast_results()` to add per-product FY summaries, repositioned the model selector, and shortened download filenames. |
| `modules/ui_components.py` | Updated `create_multi_fiscal_year_adjustment_controls()` and `apply_multi_fiscal_year_adjustments()` to support optional boundary smoothing and live totals. |

## 🔄 Upgrade Notes

- No database or format changes; pull the latest code and restart the Streamlit app.
- Optional: rerun `SETUP.bat` if you keep a frozen environment.

## 🧭 Compatibility

- ✅ Python 3.11 / 3.12 (recommended)
- ⚠️ Python 3.13 (requires latest wheels)

## 🙏 Acknowledgements

Thanks to users stressing July transition artifacts—the new smoothing toggle lets you choose between strict statistical output and business-friendly curves while keeping totals accurate.

---

**Happy forecasting!**
