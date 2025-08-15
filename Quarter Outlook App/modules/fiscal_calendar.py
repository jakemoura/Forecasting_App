"""
Fiscal Calendar Utilities for Outlook Forecaster

Handles fiscal year calculations, quarter detection, and business day calculations
for the daily data edition of the Quarterly Outlook Forecaster.

Fiscal Year Calendar: July-June
- Q1: July-September
- Q2: October-December  
- Q3: January-March
- Q4: April-June
"""

from datetime import datetime, timedelta


def get_fiscal_quarter_info(date):
    """
    Get fiscal quarter information for a given date.
    Fiscal year runs July-June (Q1: Jul-Sep, Q2: Oct-Dec, Q3: Jan-Mar, Q4: Apr-Jun)
    
    Args:
        date: datetime object
        
    Returns:
        dict: Contains fiscal_year, quarter, quarter_start, quarter_end, quarter_name
    """
    month = date.month
    year = date.year
    
    if month >= 7:  # July-December = Q1, Q2 of fiscal year starting this calendar year
        fiscal_year = year + 1  # FY2024 starts July 2023
        if month <= 9:  # Jul-Sep
            quarter = 1
            quarter_start = datetime(year, 7, 1)
            quarter_end = datetime(year, 9, 30)
        else:  # Oct-Dec
            quarter = 2
            quarter_start = datetime(year, 10, 1)
            quarter_end = datetime(year, 12, 31)
    else:  # January-June = Q3, Q4 of fiscal year that started previous calendar year
        fiscal_year = year  # FY2024 ends June 2024
        if month <= 3:  # Jan-Mar
            quarter = 3
            quarter_start = datetime(year, 1, 1)
            quarter_end = datetime(year, 3, 31)
        else:  # Apr-Jun
            quarter = 4
            quarter_start = datetime(year, 4, 1)
            quarter_end = datetime(year, 6, 30)
    
    return {
        'fiscal_year': fiscal_year,
        'quarter': quarter,
        'quarter_start': quarter_start,
        'quarter_end': quarter_end,
        'quarter_name': f"FY{fiscal_year % 100:02d} Q{quarter}"
    }


def get_business_days_in_period(start_date, end_date):
    """
    Get number of calendar days between two dates (inclusive) - for daily consumptive businesses.
    
    Args:
        start_date: datetime object
        end_date: datetime object
        
    Returns:
        int: Number of days in the period
    """
    return (end_date - start_date).days + 1


def fiscal_quarter_label(date):
    """
    Generate fiscal quarter label for a date.
    
    Args:
        date: datetime object
        
    Returns:
        str: Fiscal quarter label (e.g., "FY25 Q2")
    """
    info = get_fiscal_quarter_info(date)
    return info['quarter_name']
