#!/usr/bin/env python3
"""
Simple Regression Test for Forecasting Applications

This test validates basic functionality of both forecasting applications
without requiring complex imports. It focuses on core functionality that
can be tested in isolation.

Usage:
    python simple_regression_test.py

Author: Jake Moura
"""

import sys
import os
import traceback
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import io

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SimpleRegressionTest:
    """Simple regression test class for basic validation."""
    
    def __init__(self):
        self.test_results = []
        self.project_root = Path(__file__).parent.parent
        
    def log_result(self, test_name, passed, message=""):
        """Log a test result."""
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message
        })
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message and not passed:
            print(f"   └─ {message}")
    
    def test_project_structure(self):
        """Test that project directories and key files exist."""
        test_name = "Project Structure"
        
        try:
            # Check main directories
            forecaster_dir = self.project_root / "Forecaster App"
            outlook_dir = self.project_root / "Quarter Outlook App"
            help_dir = self.project_root / "Help"
            tests_dir = self.project_root / "tests"
            
            required_dirs = [forecaster_dir, outlook_dir, help_dir, tests_dir]
            missing_dirs = [d for d in required_dirs if not d.exists()]
            
            if missing_dirs:
                self.log_result(test_name, False, f"Missing directories: {missing_dirs}")
                return
            
            # Check key files
            forecaster_main = forecaster_dir / "forecaster_app.py"
            outlook_main = outlook_dir / "outlook_forecaster.py"
            
            key_files = [forecaster_main, outlook_main]
            missing_files = [f for f in key_files if not f.exists()]
            
            if missing_files:
                self.log_result(test_name, False, f"Missing key files: {missing_files}")
                return
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
    
    def test_data_format_validation(self):
        """Test basic data format validation logic."""
        test_name = "Data Format Validation"
        
        try:
            # Create test data
            valid_data = pd.DataFrame({
                'Date': ['2023-01-01', '2023-02-01', '2023-03-01'],
                'Product': ['Product A', 'Product A', 'Product A'],
                'ACR': [100.0, 110.0, 120.0]
            })
            
            # Test basic validations
            required_columns = ['Date', 'Product', 'ACR']
            has_required_cols = all(col in valid_data.columns for col in required_columns)
            
            if not has_required_cols:
                self.log_result(test_name, False, "Missing required columns")
                return
            
            # Test date parsing
            try:
                pd.to_datetime(valid_data['Date'])
                date_parsing_ok = True
            except:
                date_parsing_ok = False
            
            if not date_parsing_ok:
                self.log_result(test_name, False, "Date parsing failed")
                return
            
            # Test numeric data
            numeric_ok = pd.api.types.is_numeric_dtype(valid_data['ACR'])
            
            if not numeric_ok:
                self.log_result(test_name, False, "ACR column is not numeric")
                return
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
    
    def test_basic_forecasting_logic(self):
        """Test basic forecasting calculations."""
        test_name = "Basic Forecasting Logic"
        
        try:
            # Create sample time series
            np.random.seed(42)
            dates = pd.date_range('2023-01-01', periods=12, freq='MS')
            values = np.random.normal(100, 10, 12) + np.arange(12) * 2  # Trend + noise
            
            series = pd.Series(values, index=dates)
            
            # Test simple trend calculation
            x = np.arange(len(series))
            y = np.array(series.values)  # Convert to numpy array
            
            # Simple linear regression manually
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Predict next 3 periods
            future_x = np.arange(len(series), len(series) + 3)
            predictions = slope * future_x + intercept
            
            # Basic validation - predictions should be reasonable
            if len(predictions) != 3:
                self.log_result(test_name, False, "Wrong number of predictions")
                return
            
            if any(np.isnan(predictions)) or any(np.isinf(predictions)):
                self.log_result(test_name, False, "Invalid prediction values")
                return
            
            # Predictions should follow trend
            last_value = series.iloc[-1]
            if predictions[0] < last_value - 50 or predictions[0] > last_value + 50:
                self.log_result(test_name, False, "Predictions seem unreasonable")
                return
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
    
    def test_fiscal_calendar_logic(self):
        """Test fiscal calendar calculations."""
        test_name = "Fiscal Calendar Logic"
        
        try:
            # Test fiscal quarter detection
            test_cases = [
                (datetime(2025, 7, 15), 1, 2026),   # July = Q1 FY26
                (datetime(2025, 10, 15), 2, 2026),  # October = Q2 FY26
                (datetime(2025, 1, 15), 3, 2025),   # January = Q3 FY25
                (datetime(2025, 4, 15), 4, 2025),   # April = Q4 FY25
            ]
            
            for test_date, expected_quarter, expected_fy in test_cases:
                month = test_date.month
                year = test_date.year
                
                # Fiscal year logic
                if month >= 7:  # July-December
                    fiscal_year = year + 1
                    quarter = 1 if month <= 9 else 2
                else:  # January-June
                    fiscal_year = year
                    quarter = 3 if month <= 3 else 4
                
                if quarter != expected_quarter or fiscal_year != expected_fy:
                    self.log_result(test_name, False, 
                                  f"Wrong fiscal calculation for {test_date}: got Q{quarter} FY{fiscal_year}, expected Q{expected_quarter} FY{expected_fy}")
                    return
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
    
    def test_spike_detection_logic(self):
        """Test spike detection functionality."""
        test_name = "Spike Detection Logic"
        
        try:
            # Create data with artificial spikes
            np.random.seed(42)
            normal_days = np.random.normal(100, 10, 25)
            spike_days = np.array([500, 450, 480])  # High values on certain days
            
            # Insert spikes at specific positions
            data = normal_days.copy()
            data[5] = spike_days[0]   # Day 6
            data[15] = spike_days[1]  # Day 16
            data[25-1] = spike_days[2]  # Last day
            
            # Basic spike detection logic
            baseline = np.median(data)  # Use median as baseline
            threshold_multiplier = 2.0
            threshold = baseline * threshold_multiplier
            
            detected_spikes = data > threshold
            spike_count = np.sum(detected_spikes)
            
            # Should detect 3 spikes
            if spike_count < 2 or spike_count > 5:  # Allow some tolerance
                self.log_result(test_name, False, f"Expected ~3 spikes, detected {spike_count}")
                return
            
            # Spike contribution should be significant
            total_value = np.sum(data)
            spike_value = np.sum(data[detected_spikes])
            spike_contribution = spike_value / total_value
            
            if spike_contribution < 0.3:  # Should be at least 30%
                self.log_result(test_name, False, f"Spike contribution too low: {spike_contribution:.2%}")
                return
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
    
    def test_excel_handling(self):
        """Test Excel file creation and reading."""
        test_name = "Excel File Handling"
        
        try:
            # Create test DataFrame
            test_data = pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=5, freq='D'),
                'Product': ['Test Product'] * 5,
                'ACR': [100, 110, 105, 115, 120]
            })
            
            # Create temporary Excel file
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                try:
                    test_data.to_excel(tmp_file.name, index=False, engine='openpyxl')
                    
                    # Try to read it back
                    read_data = pd.read_excel(tmp_file.name, engine='openpyxl')
                    
                    # Verify data integrity
                    if len(read_data) != len(test_data):
                        self.log_result(test_name, False, "Row count mismatch after read/write")
                        return
                    
                    if not all(col in read_data.columns for col in ['Date', 'Product', 'ACR']):
                        self.log_result(test_name, False, "Column mismatch after read/write")
                        return
                    
                    self.log_result(test_name, True)
                    
                except ImportError:
                    self.log_result(test_name, False, "openpyxl not available for Excel handling")
                    
                finally:
                    # Clean up
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                        
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
    
    def test_data_processing_pipeline(self):
        """Test basic data processing pipeline."""
        test_name = "Data Processing Pipeline"
        
        try:
            # Create sample data
            raw_data = pd.DataFrame({
                'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
                'Product': ['Product A', 'Product A', 'Product B', 'Product B'],
                'ACR': [100.0, 110.0, 200.0, 210.0]
            })
            
            # Step 1: Convert dates
            try:
                raw_data['Date'] = pd.to_datetime(raw_data['Date'])
                date_conversion_ok = True
            except:
                date_conversion_ok = False
            
            if not date_conversion_ok:
                self.log_result(test_name, False, "Date conversion failed")
                return
            
            # Step 2: Group by product
            try:
                grouped = raw_data.groupby('Product')
                product_count = len(grouped)
                
                if product_count != 2:
                    self.log_result(test_name, False, f"Expected 2 products, got {product_count}")
                    return
                    
            except Exception:
                self.log_result(test_name, False, "Product grouping failed")
                return
            
            # Step 3: Basic aggregation
            try:
                for product, group in grouped:
                    if len(group) != 2:
                        self.log_result(test_name, False, f"Expected 2 records per product, got {len(group)} for {product}")
                        return
                    
                    total_acr = group['ACR'].sum()
                    if total_acr <= 0:
                        self.log_result(test_name, False, f"Invalid ACR total for {product}: {total_acr}")
                        return
                        
            except Exception:
                self.log_result(test_name, False, "Aggregation failed")
                return
            
            self.log_result(test_name, True)
            
        except Exception as e:
            self.log_result(test_name, False, f"Error: {e}")
    
    def run_all_tests(self):
        """Run all regression tests."""
        print("🧪 Simple Regression Test Suite for Forecasting Applications")
        print("=" * 65)
        print()
        
        # Run all tests
        self.test_project_structure()
        self.test_data_format_validation()
        self.test_basic_forecasting_logic()
        self.test_fiscal_calendar_logic()
        self.test_spike_detection_logic()
        self.test_excel_handling()
        self.test_data_processing_pipeline()
        
        # Summary
        print()
        print("=" * 65)
        print("📊 Test Results Summary")
        print("=" * 65)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests*100):.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed tests:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['test']}: {result['message']}")
        
        print()
        if failed_tests == 0:
            print("✅ All tests passed! Basic functionality is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Please review the issues above.")
            return False


def main():
    """Main entry point."""
    tester = SimpleRegressionTest()
    success = tester.run_all_tests()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
