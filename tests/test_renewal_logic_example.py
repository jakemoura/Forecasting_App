"""
Example test to demonstrate the new 12-month renewal logic.

This shows how the system now handles upfront RevRec renewals by only
projecting the most recent 12 months of data forward.
"""

import pandas as pd
from datetime import datetime, timedelta

def demonstrate_renewal_logic():
    """
    Example of how the new renewal logic works.
    """
    
    # Example: Uploaded file has 36 months of renewal data
    renewal_data = [
        # Old renewals (>12 months ago) - these WON'T be projected
        {'Date': '2023-01-01', 'Product': 'ProductA', 'ACR': 100000},
        {'Date': '2023-06-01', 'Product': 'ProductA', 'ACR': 150000},
        {'Date': '2023-12-01', 'Product': 'ProductB', 'ACR': 200000},
        
        # Recent renewals (last 12 months) - these WILL be projected
        {'Date': '2024-09-01', 'Product': 'ProductA', 'ACR': 120000},  # Will project to 2025-09-01
        {'Date': '2024-12-01', 'Product': 'ProductB', 'ACR': 220000},  # Will project to 2025-12-01
        {'Date': '2025-03-01', 'Product': 'ProductA', 'ACR': 125000},  # Will project to 2026-03-01
    ]
    
    # Forecast period: 2025-09-01 to 2026-08-31
    forecast_start = datetime(2025, 9, 1)
    forecast_end = datetime(2026, 8, 31)
    
    # Calculate 12 months ago from forecast start
    twelve_months_ago = forecast_start - timedelta(days=365)
    
    print("=== UPFRONT REVREC RENEWAL LOGIC EXAMPLE ===\n")
    print(f"Forecast Period: {forecast_start.strftime('%Y-%m-%d')} to {forecast_end.strftime('%Y-%m-%d')}")
    print(f"12-Month Cutoff: {twelve_months_ago.strftime('%Y-%m-%d')}\n")
    
    print("📁 UPLOADED RENEWAL DATA:")
    for renewal in renewal_data:
        date_obj = pd.to_datetime(renewal['Date'])
        is_recent = date_obj >= twelve_months_ago
        status = "✅ WILL PROJECT" if is_recent else "❌ TOO OLD - IGNORED"
        print(f"  {renewal['Date']} | {renewal['Product']} | ${renewal['ACR']:,} | {status}")
    
    print(f"\n🔮 PROJECTED FUTURE RENEWALS:")
    for renewal in renewal_data:
        date_obj = pd.to_datetime(renewal['Date'])
        if date_obj >= twelve_months_ago:
            # Project this renewal forward yearly
            next_year = date_obj.replace(year=date_obj.year + 1)
            if forecast_start <= next_year <= forecast_end:
                print(f"  {next_year.strftime('%Y-%m-%d')} | {renewal['Product']} | ${renewal['ACR']:,} | (from {renewal['Date']})")
    
    print(f"\n💡 RESULT:")
    print(f"- Only 3 recent renewals (last 12 months) are used for projection")
    print(f"- 3 older renewals are ignored to avoid over-forecasting from churned customers")
    print(f"- This prevents stacking renewals from customers who may no longer be active")

if __name__ == "__main__":
    demonstrate_renewal_logic()
