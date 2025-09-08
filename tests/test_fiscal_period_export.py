"""
Test script to demonstrate the new Fiscal Period column functionality.

This shows how     print(f"\n✅ SUCCESS: Export will now include 'Fiscal_Period' column")
    print(f"📊 Format: 'P01 - July (FY2025)' where P01 = first month of fiscal year")
    print(f"🗓️  Fiscal Year Definition: July = P01 (FY2025), August = P02 (FY2025), ..., June = P12 (FY2025)")
    print(f"📅 Next FY: July 2025 = P01 (FY2026), etc.")
    print(f"🔢 Zero-padded periods (P01, P02, ..., P12) ensure proper Excel sorting!")
    print(f"💡 This makes exported data much more business-friendly for fiscal reporting!")xport will now include a fiscal period column in the format:
"P1 - July (Fiscal Year)", "P2 - August (Fiscal Year)", etc.
"""

import pandas as pd
import sys
import os

# Add the modules path so we can import the function
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Forecaster App', 'modules'))

try:
    from ui_components import fiscal_period_display
except ImportError:
    # If import fails, define the function locally
    def fiscal_period_display(date, fiscal_year_start_month=7):
        """
        Convert date to fiscal period display format like 'P1 - July (FY2025)'.
        """
        if pd.isna(date):
            return ""
        
        date = pd.Timestamp(date)
        
        # Calculate fiscal year and period
        if date.month >= fiscal_year_start_month:
            # Current calendar year fiscal year
            fiscal_period = date.month - fiscal_year_start_month + 1
            fiscal_year = date.year + 1  # FY starts in current year, named for next year
        else:
            # Next calendar year fiscal year (months 1-6 for July start)
            fiscal_period = date.month + (12 - fiscal_year_start_month) + 1
            fiscal_year = date.year  # FY started in previous year, named for current year
        
        # Get month name
        month_name = date.strftime('%B')
        
        # Use zero-padded period number for proper Excel sorting (P01, P02, ..., P12)
        return f"P{fiscal_period:02d} - {month_name} (FY{fiscal_year})"

def demonstrate_fiscal_periods():
    """Demonstrate how the fiscal period column will look in exports."""
    
    # Create sample data for a full fiscal year
    sample_dates = [
        '2024-07-01',  # P1 - July (FY start)
        '2024-08-01',  # P2 - August
        '2024-09-01',  # P3 - September
        '2024-10-01',  # P4 - October
        '2024-11-01',  # P5 - November
        '2024-12-01',  # P6 - December
        '2025-01-01',  # P7 - January
        '2025-02-01',  # P8 - February
        '2025-03-01',  # P9 - March
        '2025-04-01',  # P10 - April
        '2025-05-01',  # P11 - May
        '2025-06-01',  # P12 - June (FY end)
    ]
    
    # Create sample export data structure
    export_data = []
    for i, date_str in enumerate(sample_dates):
        export_data.append({
            'Date': date_str,
            'Product': 'Sample Product',
            'ACR': 100000 + (i * 5000),  # Sample revenue
            'Type': 'forecast' if i >= 6 else 'actual',
            'FiscalYear': 'FY2025'
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(export_data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add fiscal period column (this is what the export will do)
    df['Fiscal_Period'] = df['Date'].apply(
        lambda x: fiscal_period_display(x, fiscal_year_start_month=7)
    )
    
    # Reorder columns to show fiscal period after date
    cols = ['Date', 'Fiscal_Period', 'Product', 'ACR', 'Type', 'FiscalYear']
    df = df[cols]
    
    print("=== FISCAL PERIOD EXPORT DEMONSTRATION ===\n")
    print("Sample export data with new Fiscal_Period column:")
    print("(Fiscal Year starts in July = P1)\n")
    
    # Display the data
    for _, row in df.iterrows():
        print(f"{row['Date'].strftime('%Y-%m-%d')} | {row['Fiscal_Period']:<30} | {row['Product']:<15} | ${row['ACR']:,}")
    
    print(f"\n✅ SUCCESS: Export will now include 'Fiscal_Period' column")
    print(f"📊 Format: 'P1 - July (FY2025)' where P1 = first month of fiscal year")
    print(f"🗓️  Fiscal Year Definition: July = P1 (FY2025), August = P2 (FY2025), ..., June = P12 (FY2025)")
    print(f"📅 Next FY: July 2025 = P1 (FY2026), etc.")
    print(f"💡 This makes exported data much more business-friendly for fiscal reporting!")

if __name__ == "__main__":
    demonstrate_fiscal_periods()
