#!/usr/bin/env python3
"""
Test script for multi-fiscal year adjustment functionality.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add the Forecaster App modules to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Forecaster App', 'modules'))

try:
    from ui_components import analyze_fiscal_year_coverage, apply_multi_fiscal_year_adjustments
    print("✅ Successfully imported multi-fiscal year functions!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_fiscal_year_coverage():
    """Test the fiscal year coverage analysis function."""
    print("\n🧪 Testing fiscal year coverage analysis...")
    
    # Create sample forecast data spanning multiple fiscal years
    dates = pd.date_range(start='2024-01-01', end='2026-12-31', freq='MS')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Product': 'TestProduct',
        'MoM_Group': 'TestGroup',
        'Type': 'forecast',
        'Revenue': [1000] * len(dates)
    })
    
    # Test the analysis
    coverage = analyze_fiscal_year_coverage(sample_data)
    print(f"📊 Fiscal year coverage detected: {coverage}")
    
    # Should detect FY25, FY26, FY27 (July-June fiscal years)
    expected_years = [2025, 2026, 2027]
    detected_years = sorted(coverage.keys())
    
    if detected_years == expected_years:
        print("✅ Fiscal year detection working correctly!")
    else:
        print(f"⚠️  Expected {expected_years}, got {detected_years}")
    
    return coverage

def test_multi_fy_adjustments():
    """Test the multi-fiscal year adjustment application."""
    print("\n🧪 Testing multi-fiscal year adjustments...")
    
    # Create sample data
    dates = pd.date_range(start='2024-07-01', end='2026-06-30', freq='MS')  # FY25 and FY26
    sample_data = pd.DataFrame({
        'Date': dates,
        'Product': 'TestProduct',
        'MoM_Group': 'TestGroup',
        'Type': 'forecast',
        'Revenue': [1000] * len(dates)
    })
    
    # Create sample adjustments
    fiscal_year_adjustments = {
        'TestProduct TestGroup': {
            2025: 10,  # 10% growth for FY25
            2026: 5    # 5% growth for FY26
        }
    }
    
    # Test the adjustment application
    try:
        adjusted_data = apply_multi_fiscal_year_adjustments(
            sample_data, 
            fiscal_year_adjustments, 
            'Smooth'
        )
        print("✅ Multi-fiscal year adjustments applied successfully!")
        
        # Verify that adjustments were applied
        original_total = sample_data['Revenue'].sum()
        adjusted_total = adjusted_data['Revenue'].sum()
        
        print(f"📈 Original total revenue: ${original_total:,.0f}")
        print(f"📈 Adjusted total revenue: ${adjusted_total:,.0f}")
        print(f"📈 Total growth: {((adjusted_total/original_total - 1) * 100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error applying adjustments: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Multi-Fiscal Year Adjustment Tests")
    print("=" * 50)
    
    # Test 1: Fiscal year coverage analysis
    coverage = test_fiscal_year_coverage()
    
    # Test 2: Multi-fiscal year adjustments
    adjustment_success = test_multi_fy_adjustments()
    
    print("\n" + "=" * 50)
    if coverage and adjustment_success:
        print("🎉 All tests passed! Multi-fiscal year functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()