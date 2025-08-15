"""
Excel File Diagnostic Tool
For Forecasting App Troubleshooting

This script helps diagnose issues with Excel files before uploading them to the forecasting app.
"""

import pandas as pd
import sys
import io
from pathlib import Path

def diagnose_excel_file(file_path):
    """Diagnose an Excel file for common issues"""
    print(f"\n🔍 Diagnosing Excel file: {file_path}")
    print("=" * 60)
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    # Try different engines
    engines = ["openpyxl", "xlrd", "pyxlsb"]
    successful_engine = None
    
    for engine in engines:
        try:
            print(f"\n📋 Trying engine: {engine}")
            df = pd.read_excel(file_path, engine=engine)
            print(f"✅ Successfully read with {engine}")
            successful_engine = engine
            break
        except Exception as e:
            print(f"❌ Failed with {engine}: {str(e)}")
    
    if successful_engine is None:
        print(f"\n❌ Could not read file with any engine")
        return False
    
    # Analyze the successfully read data
    print(f"\n📊 File Analysis (using {successful_engine}):")
    print(f"   • Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   • Columns: {list(df.columns)}")
    
    # Check for required columns
    required_cols = {"Date", "Line", "ACR"}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        print(f"\n❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Required columns: {required_cols}")
    else:
        print(f"✅ All required columns present: {required_cols}")
    
    # Analyze Date column
    if "Date" in df.columns:
        print(f"\n📅 Date Column Analysis:")
        print(f"   • Sample values: {df['Date'].head(3).tolist()}")
        print(f"   • Data type: {df['Date'].dtype}")
        
        try:
            date_series = pd.to_datetime(df['Date'], errors='coerce')
            invalid_dates = date_series.isna().sum()
            if invalid_dates > 0:
                print(f"   ❌ {invalid_dates} invalid dates found")
                invalid_examples = df.loc[date_series.isna(), 'Date'].head(3).tolist()
                print(f"   Examples: {invalid_examples}")
            else:
                print(f"   ✅ All dates are valid")
                print(f"   • Date range: {date_series.min()} to {date_series.max()}")
        except Exception as e:
            print(f"   ❌ Date parsing error: {e}")
    
    # Analyze ACR column
    if "ACR" in df.columns:
        print(f"\n💰 ACR Column Analysis:")
        print(f"   • Sample values: {df['ACR'].head(3).tolist()}")
        print(f"   • Data type: {df['ACR'].dtype}")
        
        try:
            numeric_series = pd.to_numeric(df['ACR'], errors='coerce')
            invalid_numbers = numeric_series.isna().sum()
            if invalid_numbers > 0:
                print(f"   ❌ {invalid_numbers} non-numeric values found")
                invalid_examples = df.loc[numeric_series.isna(), 'ACR'].head(3).tolist()
                print(f"   Examples: {invalid_examples}")
            else:
                print(f"   ✅ All ACR values are numeric")
                print(f"   • Range: {numeric_series.min():.2f} to {numeric_series.max():.2f}")
        except Exception as e:
            print(f"   ❌ ACR parsing error: {e}")
    
    # Analyze Line column
    if "Line" in df.columns:
        print(f"\n📈 Line Column Analysis:")
        unique_lines = df['Line'].unique()
        print(f"   • Unique lines: {len(unique_lines)}")
        print(f"   • Sample lines: {unique_lines[:5].tolist()}")
    
    print(f"\n{'✅ File appears compatible!' if not missing_cols else '❌ File needs fixing before use'}")
    return len(missing_cols) == 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_excel.py <path_to_excel_file>")
        print("\nExample: python diagnose_excel.py my_data.xlsx")
        
        # Try to find Excel files in current directory
        excel_files = list(Path(".").glob("*.xlsx")) + list(Path(".").glob("*.xls"))
        if excel_files:
            print(f"\nFound Excel files in current directory:")
            for i, file in enumerate(excel_files, 1):
                print(f"  {i}. {file.name}")
            
            try:
                choice = input(f"\nEnter number (1-{len(excel_files)}) to diagnose, or press Enter to exit: ").strip()
                if choice and choice.isdigit():
                    file_idx = int(choice) - 1
                    if 0 <= file_idx < len(excel_files):
                        diagnose_excel_file(excel_files[file_idx])
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
        sys.exit(1)
    
    file_path = sys.argv[1]
    diagnose_excel_file(file_path)
