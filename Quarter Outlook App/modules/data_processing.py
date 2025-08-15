"""
Data Processing Utilities for Outlook Forecaster

Handles Excel file reading, date parsing, and daily data analysis
for the Quarterly Outlook Forecaster application.
"""

import io
import pandas as pd
from .spike_detection import detect_monthly_spikes


def read_any_excel(buf: io.BytesIO) -> pd.DataFrame:
    """
    Robust Excel reader supporting multiple engines.
    
    Args:
        buf: BytesIO buffer containing Excel file data
        
    Returns:
        pd.DataFrame: Parsed Excel data
        
    Raises:
        RuntimeError: If workbook cannot be read by any available engine
    """
    for eng in ("openpyxl", "xlrd", "pyxlsb"):
        try:
            return pd.read_excel(buf, engine=eng)
        except Exception:
            buf.seek(0)
    raise RuntimeError("Workbook could not be read by available engines.")


def coerce_daily_dates(col: pd.Series) -> pd.Series:
    """
    Convert various date formats to daily timestamps.
    Handles Excel dates, text dates, etc.
    
    Args:
        col: pandas Series containing date values in various formats
        
    Returns:
        pd.Series: Series with properly parsed datetime values
        
    Raises:
        ValueError: If dates cannot be parsed
    """
    # First try standard pandas parsing
    dt = pd.to_datetime(col, errors="coerce")
    
    if dt.isna().any():
        mask = dt.isna()
        failed_dates = col[mask].astype(str)
        
        # Try different date formats for failed dates
        for date_format in ["%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d"]:
            if failed_dates.isna().all():
                break
            try:
                parsed = pd.to_datetime(failed_dates, format=date_format, errors="coerce")
                valid_parsed = ~parsed.isna()
                if valid_parsed.any():
                    dt[mask] = dt[mask].fillna(parsed)
                    mask = dt.isna()
                    failed_dates = col[mask].astype(str) if mask.any() else pd.Series([], dtype=str)
            except (ValueError, TypeError):
                continue
    
    # Final check for any remaining unparsed dates
    if dt.isna().any():
        bad = col[dt.isna()].unique()[:5]
        raise ValueError(f"Unable to parse date strings: {bad}. Supported formats include: 'MM/DD/YYYY', 'YYYY-MM-DD', etc.")
    
    return dt


def analyze_daily_data(series, spike_threshold=2.0):
    """
    Analyze daily business data patterns including spike detection.
    
    Args:
        series: pandas Series with datetime index containing daily business data
        spike_threshold: float, multiplier above baseline to consider a spike
        
    Returns:
        dict: Analysis results including averages, volatility, trends, and spike analysis
    """
    # Check for weekday/weekend patterns using proper datetime indexing
    weekday_mask = series.index.to_series().dt.dayofweek < 5  # Mon-Fri
    weekend_mask = series.index.to_series().dt.dayofweek >= 5  # Sat-Sun
    
    weekday_avg = series[weekday_mask].mean()  # Mon-Fri
    weekend_avg = series[weekend_mask].mean()  # Sat-Sun
    
    # Calculate business day statistics
    business_days = series[weekday_mask]
    weekend_days = series[weekend_mask]
    
    # Detect monthly spikes (subscription renewals, etc.)
    spike_analysis = detect_monthly_spikes(series, spike_threshold)
    
    analysis = {
        'total_days': len(series),
        'business_days': len(business_days),
        'weekend_days': len(weekend_days),
        'weekday_avg': weekday_avg if len(business_days) > 0 else 0,
        'weekend_avg': weekend_avg if len(weekend_days) > 0 else 0,
        'business_ratio': weekday_avg / weekend_avg if weekend_avg > 0 else 1.0,
        'daily_volatility': series.std(),
        'trend': (series.iloc[-7:].mean() - series.iloc[:7].mean()) / series.iloc[:7].mean() if len(series) >= 14 else 0,
        'spike_analysis': spike_analysis
    }
    
    return analysis
