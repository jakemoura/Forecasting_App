"""
Utility functions for data processing and validation.

Contains helper functions for reading Excel files, date processing,
and data validation operations.
"""

import io
import pandas as pd
import numpy as np
from datetime import datetime


def read_any_excel(file_obj):
    """
    Read Excel file with robust engine detection.
    Tries different engines to handle various Excel formats.
    """
    try:
        # Try openpyxl first (most reliable for .xlsx)
        return pd.read_excel(file_obj, engine='openpyxl')
    except Exception as e1:
        try:
            # Try xlrd for older .xls files
            return pd.read_excel(file_obj, engine='xlrd')
        except Exception as e2:
            try:
                # Try pyxlsb for .xlsb files
                return pd.read_excel(file_obj, engine='pyxlsb')
            except Exception as e3:
                # Try with explicit engine specification based on file extension
                try:
                    # Get file info to determine format
                    if hasattr(file_obj, 'name'):
                        filename = file_obj.name.lower()
                    else:
                        filename = str(file_obj).lower()
                    
                    if filename.endswith('.xlsx'):
                        return pd.read_excel(file_obj, engine='openpyxl')
                    elif filename.endswith('.xls'):
                        return pd.read_excel(file_obj, engine='xlrd')
                    elif filename.endswith('.xlsb'):
                        return pd.read_excel(file_obj, engine='pyxlsb')
                    else:
                        # Try openpyxl as final fallback (most compatible)
                        return pd.read_excel(file_obj, engine='openpyxl')
                except Exception as e4:
                    # If all else fails, provide detailed error
                    raise ValueError(f"Failed to read Excel file. Tried engines: openpyxl, xlrd, pyxlsb. Errors: {e1}, {e2}, {e3}, {e4}")


def coerce_month_start(date_series):
    """
    Convert various date formats to month-start datetime format.
    Handles common Excel date formats and text representations.
    """
    if isinstance(date_series, pd.Series):
        # Type: ignore - pandas apply type annotation issue
        return date_series.apply(lambda x: _coerce_single_date(x))  # type: ignore
    else:
        return _coerce_single_date(date_series)


def _coerce_single_date(date_val):
    """Helper function to coerce a single date value."""
    if pd.isna(date_val):
        return pd.NaT
    
    # If already datetime, convert to month start
    if isinstance(date_val, pd.Timestamp):
        return date_val.to_period('M').to_timestamp()
    
    # If string, try to parse various formats
    if isinstance(date_val, str):
        date_val = date_val.strip()
        
        # Check for manual parsing patterns first (e.g., "Jul-21", "Apr-22")
        if '-' in date_val:
            parts = date_val.split('-')
            if len(parts) == 2:
                month_str, year_str = parts
                
                # Map common month abbreviations
                month_mapping = {
                    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', 
                    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
                    'January': '01', 'February': '02', 'March': '03', 'April': '04',
                    'June': '06', 'July': '07', 'August': '08', 'September': '09', 
                    'October': '10', 'November': '11', 'December': '12'
                }
                
                # Check if month is in our mapping (indicating manual parsing needed)
                if month_str in month_mapping:
                    try:
                        # Assume 2-digit years are 20xx
                        year = int(year_str) + 2000 if len(year_str) == 2 else int(year_str)
                        month_num = month_mapping[month_str]
                        parsed = pd.to_datetime(f"{year}-{month_num}-01")
                        return parsed.to_period('M').to_timestamp()
                    except Exception:
                        pass  # Fall through to standard parsing
        
        # Try standard pandas parsing for other formats
        try:
            parsed = pd.to_datetime(date_val)
            return parsed.to_period('M').to_timestamp()
        except Exception:
            pass
    
    # Try direct pandas conversion as final fallback
    try:
        parsed = pd.to_datetime(date_val)
        return parsed.to_period('M').to_timestamp()
    except Exception:
        raise ValueError(f"Cannot parse date: {date_val}")


def debug_chart_data(data, title="Debug Chart Data"):
    """
    Debug function to print chart data structure.
    Useful for troubleshooting data issues.
    """
    print(f"\n=== {title} ===")
    print(f"Type: {type(data)}")
    
    if hasattr(data, 'shape'):
        print(f"Shape: {data.shape}")
    
    if hasattr(data, 'columns'):
        print(f"Columns: {list(data.columns)}")
        
    if hasattr(data, 'dtypes'):
        print("Data types:")
        for col, dtype in data.dtypes.items():
            print(f"  {col}: {dtype}")
    
    if hasattr(data, 'head'):
        print("First few rows:")
        print(data.head())
    
    print("=" * (len(title) + 8))


def validate_data_structure(df, required_columns=None):
    """
    Validate that dataframe has required structure for forecasting.
    
    Args:
        df: pandas DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if required_columns is None:
        required_columns = ["Date", "Product", "ACR"]
    
    # Check if DataFrame is not empty
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check for required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for non-null values in key columns
    for col in required_columns:
        if df[col].isnull().all():
            return False, f"Column '{col}' contains only null values"
    
    # Check data types
    if "Date" in df.columns:
        try:
            pd.to_datetime(df["Date"].dropna().iloc[0])
        except Exception:
            return False, "Date column contains invalid date values"
    
    if "ACR" in df.columns:
        try:
            pd.to_numeric(df["ACR"].dropna(), errors='raise')
        except Exception:
            return False, "ACR column contains non-numeric values"
    
    return True, "Data structure is valid"


def prepare_forecast_data(df, product_col="Product", date_col="Date", value_col="ACR"):
    """
    Prepare raw data for forecasting by cleaning and sorting.
    
    Args:
        df: Input DataFrame
        product_col: Name of product column
        date_col: Name of date column  
        value_col: Name of value column
    
    Returns:
        Cleaned and prepared DataFrame
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Convert dates to proper format
    df_clean[date_col] = coerce_month_start(df_clean[date_col])
    
    # Convert values to numeric
    df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
    
    # Remove rows with invalid dates or values
    df_clean = df_clean.dropna(subset=[date_col, value_col])
    
    # Sort by product and date
    df_clean = df_clean.sort_values([product_col, date_col])
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean


def get_data_summary_stats(df, product_col="Product", date_col="Date", value_col="ACR"):
    """
    Generate summary statistics for forecasting data.
    
    Returns:
        Dictionary with summary statistics per product
    """
    summary = {}
    
    for product, group in df.groupby(product_col):
        series = group.set_index(date_col)[value_col]
        
        summary[product] = {
            'months_count': len(series),
            'date_range': (series.index.min(), series.index.max()),
            'mean_value': series.mean(),
            'std_value': series.std(),
            'min_value': series.min(),
            'max_value': series.max(),
            'has_negatives': (series < 0).any(),
            'missing_months': _count_missing_months(series),
            'data_quality': _assess_data_quality(series)
        }
    
    return summary


def _count_missing_months(series):
    """Count missing months in a time series."""
    if len(series) < 2:
        return 0
    
    # Create full date range
    start_date = series.index.min()
    end_date = series.index.max()
    full_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Count missing dates
    missing = len(full_range) - len(series)
    return missing


def _assess_data_quality(series):
    """Assess data quality for forecasting."""
    if len(series) < 12:
        return "insufficient"
    elif len(series) < 24:
        return "limited"
    elif len(series) < 36:
        return "good"
    else:
        return "excellent"
