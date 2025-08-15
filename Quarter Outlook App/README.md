# Quarterly Outlook Forecaster (Modular Version)

A modular, daily data edition of the quarterly business forecaster that projects full quarter performance from partial data using a fiscal year calendar (July-June).

## 📁 Project Structure

```
outlook_forecaster/
├── outlook_forecaster.py          # Main Streamlit application
├── RUN_OUTLOOK_FORECASTER.bat    # Launcher script
├── README.md                      # This file
└── modules/
    ├── __init__.py               # Module initialization
    ├── fiscal_calendar.py        # Fiscal year calendar utilities
    ├── data_processing.py        # Excel reading and data processing
    ├── spike_detection.py        # Monthly renewal spike detection
    ├── forecasting_models.py     # Individual forecasting models
    ├── quarterly_forecasting.py  # Main forecasting engine
    ├── model_evaluation.py       # MAPE calculation and model evaluation
    └── ui_components.py          # Streamlit UI components
```

## 🚀 Key Features

### Modular Architecture
- **Separation of Concerns**: Each module handles specific functionality
- **Maintainability**: Easy to update individual components
- **Testability**: Modules can be tested independently
- **Reusability**: Components can be reused across applications

### Business-Focused Forecasting
- **Fiscal Year Calendar**: Q1 (Jul-Sep), Q2 (Oct-Dec), Q3 (Jan-Mar), Q4 (Apr-Jun)
- **Daily Data Processing**: Handles daily business data with weekend/weekday awareness
- **Multiple Models**: Linear trend, moving averages, Prophet, ARIMA, XGBoost, etc.
- **Automatic Model Selection**: MAPE-based evaluation to select best performing models

### Advanced Features
- **Monthly Renewal Detection**: Automatically detects subscription renewal spikes
- **Capacity Constraints**: Apply operational limitations to forecasts
- **Confidence Intervals**: Uncertainty quantification across models
- **Interactive Visualizations**: Charts and progress indicators
- **Excel Export**: Comprehensive reporting capabilities

## 📊 Module Descriptions

### `fiscal_calendar.py`
- Fiscal quarter detection and calculations
- Business day counting for daily consumptive businesses
- Quarter progress tracking

### `data_processing.py`
- Robust Excel file reading (multiple engines)
- Date parsing and validation
- Daily data pattern analysis

### `spike_detection.py`
- Monthly subscription renewal detection
- Spike pattern analysis by day of month
- Enhanced forecasting for renewal-based businesses

### `forecasting_models.py`
- Linear trend models
- Moving averages with trend adjustment
- Prophet daily models (if available)
- LightGBM and XGBoost time series models
- ARIMA and Exponential Smoothing
- Ensemble model creation

### `quarterly_forecasting.py`
- Main forecasting engine
- Multi-model orchestration
- Capacity constraint application
- Quarter completion projections

### `model_evaluation.py`
- MAPE (Mean Absolute Percentage Error) calculation
- Cross-validation for model selection
- Performance ranking and comparison

### `ui_components.py`
- Streamlit UI components
- Interactive charts and visualizations
- Progress indicators and summary tables
- Excel export functionality

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.7+
- Required packages: streamlit, pandas, numpy, scikit-learn, altair
- Optional packages: prophet, lightgbm, xgboost, statsmodels

### Running the Application
1. Navigate to the `outlook_forecaster` directory
2. Double-click `RUN_OUTLOOK_FORECASTER.bat` (Windows)
3. Or run: `python -m streamlit run outlook_forecaster.py --server.port 8502`

### Data Requirements
Upload an Excel file with these columns:
- **Date**: Daily dates in any standard format
- **Product**: Business product names
- **ACR**: Daily values to forecast (revenue, sales, etc.)

## 📈 How It Works

1. **Data Upload**: Users upload daily business data via Excel
2. **Fiscal Analysis**: System determines current fiscal quarter and progress
3. **Multi-Model Forecasting**: Applies 6+ different forecasting models
4. **Model Evaluation**: Uses MAPE to rank model performance
5. **Ensemble Creation**: Combines best models for final forecast
6. **Business Intelligence**: Provides actionable insights and projections

## 🔧 Configuration Options

### Sidebar Controls
- **Capacity Constraints**: Apply operational limitations (0.5-1.0 multiplier)
- **Monthly Renewal Detection**: Detect subscription spikes (sensitivity 1.5-4.0)
- **Confidence Levels**: 80%, 90%, or 95% confidence intervals

### Advanced Features
- **Capacity Calculator**: Estimate constraints from weekly revenue loss
- **Spike Analysis**: Detailed renewal pattern detection
- **Model Comparison**: Performance metrics and rankings

## 📊 Output & Reporting

### Interactive Results
- Daily data trends with forecast projections
- Quarter completion progress indicators
- Model performance comparisons
- Spike analysis for renewals

### Excel Reports
- Summary of all forecasts by product and model
- Individual product sheets with detailed breakdowns
- Model evaluation metrics and rankings

## 🎯 Business Value

### Quick Decision Making
- Fast quarterly outlook from partial data
- Multiple scenario planning with confidence intervals
- Operational constraint modeling

### Accuracy & Reliability
- Multiple model validation and ensemble methods
- Automatic best model selection
- Business calendar awareness (fiscal year, renewals)

### Scalability
- Handles multiple products simultaneously
- Modular architecture for easy extension
- Enterprise-ready Excel integration

## 🔄 Comparison with Original

### Advantages of Modular Version
- **Better Organization**: Clear separation of functionality
- **Easier Maintenance**: Update individual components without affecting others
- **Enhanced Testability**: Test modules independently
- **Improved Readability**: Smaller, focused files
- **Reusable Components**: Use modules in other projects

### Migration Benefits
- **Same Functionality**: All original features preserved
- **Better Performance**: Optimized imports and processing
- **Future-Proof**: Easier to add new features and models
- **Documentation**: Better code documentation and structure

This modular architecture makes the Outlook Forecaster more maintainable, scalable, and professional while preserving all the powerful forecasting capabilities of the original application.
