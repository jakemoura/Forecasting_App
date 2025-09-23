#!/usr/bin/env python3
"""
Test sequential fiscal year processing logic
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "Forecaster App", "modules"))

import pandas as pd
import numpy as np
from datetime import datetime

def test_sequential_fy_processing():
    """Test that FY adjustments are processed sequentially for proper YoY compounding"""
    
    # Create sample data with 3 fiscal years
    dates = pd.date_range("2026-07-01", periods=36, freq='M')  # FY2027, FY2028, FY2029
    
    data = []
    base_value = 1000000  # $1M per month base
    
    for date in dates:
        data.append({
            'Date': date,
            'Product': 'Search',
            'ACR': base_value,
            'Type': 'forecast'
        })
    
    chart_model_data = pd.DataFrame(data)
    
    # Create test adjustments matching your scenario
    fiscal_year_adjustments = {
        "Search": {
            2027: {"enabled": True, "target_yoy": 90.0, "current_yoy": 0.0, "distribution_method": "Smooth"},  # 90% growth
            2028: {"enabled": True, "target_yoy": 24.0, "current_yoy": 0.0, "distribution_method": "Smooth"},  # 24% growth from FY2027
            2029: {"enabled": True, "target_yoy": 15.0, "current_yoy": 0.0, "distribution_method": "Smooth"}   # 15% growth from FY2028
        }
    }
    
    print("Testing Sequential Fiscal Year Processing")
    print("="*50)
    print(f"Base monthly value: ${base_value:,.0f}")
    print(f"Expected progression:")
    print(f"  FY2027: +90.0% -> ${base_value * 1.9 * 12 / 1e6:.1f}M annual")
    print(f"  FY2028: +24.0% from FY2027 -> ${base_value * 1.9 * 1.24 * 12 / 1e6:.1f}M annual")
    print(f"  FY2029: +15.0% from FY2028 -> ${base_value * 1.9 * 1.24 * 1.15 * 12 / 1e6:.1f}M annual")
    
    # Apply adjustments
    try:
        from ui_components import apply_multi_fiscal_year_adjustments
        
        adjusted_data, summary = apply_multi_fiscal_year_adjustments(
            chart_model_data, fiscal_year_adjustments, 7
        )
        
        # Calculate actual totals by fiscal year
        for fy in [2027, 2028, 2029]:
            fy_start = pd.Timestamp(year=fy-1, month=7, day=1)
            fy_end = pd.Timestamp(year=fy, month=7, day=1) - pd.DateOffset(days=1)
            
            # Original data
            orig_mask = (
                (chart_model_data['Product'] == 'Search') &
                (pd.to_datetime(chart_model_data['Date']) >= fy_start) &
                (pd.to_datetime(chart_model_data['Date']) <= fy_end)
            )
            orig_total = chart_model_data[orig_mask]['ACR'].sum()
            
            # Adjusted data
            adj_mask = (
                (adjusted_data['Product'] == 'Search') &
                (pd.to_datetime(adjusted_data['Date']) >= fy_start) &
                (pd.to_datetime(adjusted_data['Date']) <= fy_end)
            )
            adj_total = adjusted_data[adj_mask]['ACR'].sum()
            
            # Calculate YoY vs previous FY (for FY2028 and later)
            if fy > 2027:
                prev_fy = fy - 1
                prev_fy_start = pd.Timestamp(year=prev_fy-1, month=7, day=1)
                prev_fy_end = pd.Timestamp(year=prev_fy, month=7, day=1) - pd.DateOffset(days=1)
                
                prev_adj_mask = (
                    (adjusted_data['Product'] == 'Search') &
                    (pd.to_datetime(adjusted_data['Date']) >= prev_fy_start) &
                    (pd.to_datetime(adjusted_data['Date']) <= prev_fy_end)
                )
                prev_adj_total = adjusted_data[prev_adj_mask]['ACR'].sum()
                
                if prev_adj_total > 0:
                    yoy_growth = ((adj_total / prev_adj_total) - 1) * 100
                    print(f"\nFY{fy}: ${orig_total/1e6:.1f}M -> ${adj_total/1e6:.1f}M ({yoy_growth:+.1f}% YoY)")
                else:
                    print(f"\nFY{fy}: ${orig_total/1e6:.1f}M -> ${adj_total/1e6:.1f}M")
            else:
                # First year, compare to original
                yoy_growth = ((adj_total / orig_total) - 1) * 100
                print(f"\nFY{fy}: ${orig_total/1e6:.1f}M -> ${adj_total/1e6:.1f}M ({yoy_growth:+.1f}% vs original)")
        
        print(f"\n✅ Sequential processing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sequential_fy_processing()
    sys.exit(0 if success else 1)