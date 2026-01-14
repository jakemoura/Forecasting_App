# 📊 Forecaster App

**Enterprise-grade time series forecasting with intelligent model selection and business-focused adjustments.**

[![Version](https://img.shields.io/badge/version-1.5-blue.svg)](https://github.com/jakemoura/Forecasting_App/releases/tag/v1.5)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)

## 🚀 Features

### **Intelligent Model Selection**
- **Smart Backtesting**: Automatically validates multiple models and recommends the best one per product
- **Multiple Algorithms**: SARIMA, ETS, Polynomial, Prophet, Auto-ARIMA, LightGBM
- **Data-Driven Recommendations**: Uses WAPE and MASE metrics for optimal model selection

### **Trend Override Controls** *(New in v1.5)*
- **Override Declining Trends**: Per-product toggle to adjust statistical forecasts
- **Options**:
  - 📊 Statistical Model (Default)
  - ➡️ Flat Trend (Maintain Last Value)
  - 📈 Continue Recent Growth (Avg of Last 6 Months)
  - 🎯 Custom Monthly Growth Rate (Pure Compounding)

### **Business Adjustments**
- **Conservatism Slider**: Apply percentage haircuts to forecasts
- **Manual Adjustments**: Per-product growth/haircut percentages
- **Fiscal Year Growth Targets**: Set YoY targets with smooth distribution
- **Renewals Overlay**: Include non-compliant RevRec data

### **Enhanced Excel Export** *(New in v1.5)*
- **Forecast Summary Sheet**: Documents all adjustments applied
  - Conservatism factor
  - Trend overrides per product
  - Manual adjustments
  - Fiscal year growth targets
  - Renewals overlay status
- **Auto-Select Trend Overrides**: Download option automatically selects when overrides are active

### **Visualization & Analysis**
- **Interactive Charts**: Altair-powered visualizations with backtesting comparisons
- **Month-over-Month Growth Analysis**: Track trends across products
- **Model Performance Comparison**: Side-by-side accuracy metrics

## 📦 Installation

### Prerequisites
- Python 3.8 or later
- pip (Python package manager)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/jakemoura/Forecasting_App.git
cd "Forecaster App"

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run forecaster_app.py
```

### macOS Users
Double-click `LAUNCH_FORECASTER.sh` or run:
```bash
./LAUNCH_FORECASTER.sh
```

## 🎯 Usage

1. **Upload Data**: CSV or Excel file with Date, Product, and ACR columns
2. **Configure Forecast**: Set horizon, select models, adjust conservatism
3. **Run Forecast**: Click "Run Forecast" and wait for backtesting
4. **Review Results**: Analyze per-product forecasts and model recommendations
5. **Apply Adjustments**: Use trend overrides, manual adjustments, or FY targets
6. **Download**: Export to Excel with full summary of adjustments

## 📋 Data Format

Your input file should contain:

| Column | Description | Required |
|--------|-------------|----------|
| Date | Monthly date (YYYY-MM-DD or similar) | ✅ |
| Product | Product/category name | ✅ |
| ACR | Revenue/metric value | ✅ |

## 🔄 Release History

### v1.5 (January 2026)
- ✨ Trend Override toggle for per-product forecast adjustments
- ✨ "Best per Product (With Trend Overrides)" download option
- ✨ Forecast Summary sheet in Excel exports
- 🐛 Fixed 230+ unicode/emoji encoding issues
- 🐛 Fixed custom growth rate to use pure compounding

### v1.4
- Smart backtesting with automatic model selection
- Fiscal year growth targets with smooth distribution
- Conservatism slider for forecast adjustments

### v1.3
- Multi-model support (Prophet, Auto-ARIMA, LightGBM)
- Interactive forecast adjustments
- Renewals overlay integration

## 🛠️ Configuration

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Custom port (default: 8501)
- `STREAMLIT_SERVER_HEADLESS`: Run without browser (default: false)

### Fiscal Year Settings
The app uses July-June fiscal year by default. Modify `fiscal_year_start_month=7` in the code to change.

## 📊 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| Storage | 500 MB | 1 GB |
| Python | 3.8 | 3.10+ |
| Browser | Any modern | Chrome/Edge |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is proprietary software. All rights reserved.

## 🆘 Support

For issues or questions:
- Open a GitHub issue
- Check the [LAUNCHER_README.md](LAUNCHER_README.md) for setup troubleshooting

---

**Happy Forecasting! 🎯📈**
