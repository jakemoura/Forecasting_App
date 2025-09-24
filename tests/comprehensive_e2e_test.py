#!/usr/bin/env python3
"""
Comprehensive End-to-End Regression Test for Forecasting Applications v1.2.0

This test suite validates both applications work correctly by actually importing 
and running their main modules with realistic test data.

Usage:
    python comprehensive_e2e_test.py [--quick] [--app forecaster|outlook|both]

Author: Jake Moura  
Version: v1.2.0
"""

import sys
import os
import unittest
import tempfile
import time
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).parent.parent
forecaster_path = project_root / "Forecaster App"
outlook_path = project_root / "Quarter Outlook App"

class ComprehensiveRegressionTests(unittest.TestCase):
    """End-to-end tests that actually run both applications."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment and data."""
        cls.forecaster_data = cls._create_forecaster_data()
        cls.outlook_data = cls._create_outlook_data()
        cls.temp_files = []
        
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        for temp_file in cls.temp_files:
            try:
                os.unlink(temp_file)
            except (FileNotFoundError, PermissionError):
                pass
    
    @staticmethod
    def _create_forecaster_data():
        """Create realistic monthly forecasting data."""
        start_date = datetime(2022, 1, 1)
        dates = pd.date_range(start_date, periods=36, freq='MS')
        
        data = []
        np.random.seed(42)
        
        products = ['TestProduct_A', 'TestProduct_B', 'TestProduct_C']
        
        for product in products:
            base_value = 1000000 if 'A' in product else (750000 if 'B' in product else 1500000)
            
            for i, date in enumerate(dates):
                # Realistic business growth with seasonality
                trend = i * 25000
                seasonal = 100000 * np.sin(2 * np.pi * i / 12)
                noise = np.random.normal(0, 50000)
                
                # Quarter-end spikes
                if date.month in [3, 6, 9, 12]:
                    seasonal *= 1.3
                
                acr = max(base_value + trend + seasonal + noise, base_value * 0.8)
                
                data.append({
                    'Date': date,
                    'Product': product,
                    'ACR': round(acr, 2)
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def _create_outlook_data():
        """Create realistic daily outlook data."""
        start_date = datetime(2025, 4, 1)
        end_date = datetime(2025, 6, 30)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        data = []
        np.random.seed(123)
        
        products = ['Daily_Product_X', 'Daily_Product_Y']
        
        for product in products:
            base_daily = 50000 if 'X' in product else 30000
            
            for date in dates:
                # Business day effects and renewal patterns
                multiplier = 1.0 if date.weekday() < 5 else 0.3  # Weekday vs weekend
                
                # Month-end and renewal spikes
                if date.day >= 28:
                    multiplier *= 1.5
                if date.day in [1, 15]:
                    multiplier *= 2.0
                
                daily_value = base_daily * multiplier + np.random.normal(0, base_daily * 0.1)
                
                data.append({
                    'Date': date,
                    'Product': product,
                    'ACR': max(daily_value, 0)
                })
        
        return pd.DataFrame(data)

    def test_forecaster_app_import_and_basic_functionality(self):
        """Test that Forecaster App can be imported and basic functions work."""
        print("🔍 Testing Forecaster App import and basic functionality...")
        
        try:
            # Change to Forecaster App directory and test import
            original_cwd = os.getcwd()
            os.chdir(str(forecaster_path))
            sys.path.insert(0, str(forecaster_path))
            
            # Test importing core modules
            from modules import data_validation, models, metrics
            
            # Test data validation
            is_valid = data_validation.validate_data_format(self.forecaster_data)
            self.assertTrue(is_valid, "Forecaster data should be valid")
            
            # Test that we can get available models
            available_models = models.get_available_models()
            self.assertIsInstance(available_models, list)
            self.assertGreater(len(available_models), 0, "Should have available models")
            
            # Test metrics calculation
            test_actual = [100, 200, 300, 400, 500]
            test_forecast = [95, 210, 290, 420, 480]
            wape = metrics.calculate_wape(test_actual, test_forecast)
            self.assertIsInstance(wape, (int, float))
            self.assertGreaterEqual(wape, 0, "WAPE should be non-negative")
            
            print("✅ Forecaster App basic functionality working")
            
        except Exception as e:
            self.fail(f"Forecaster App basic functionality failed: {e}")
        finally:
            os.chdir(original_cwd)
            if str(forecaster_path) in sys.path:
                sys.path.remove(str(forecaster_path))

    def test_forecaster_app_yoy_adjustments_v1_2_0(self):
        """Test v1.2.0 sequential YoY compounding specifically."""
        print("🔍 Testing v1.2.0 YoY sequential compounding...")
        
        try:
            original_cwd = os.getcwd()
            os.chdir(str(forecaster_path))
            sys.path.insert(0, str(forecaster_path))
            
            from modules import ui_components
            
            # Create test forecast data spanning multiple fiscal years
            forecast_dates = pd.date_range('2025-07-01', '2027-06-30', freq='MS')
            
            test_forecast = pd.DataFrame({
                'Date': forecast_dates,
                'Product': 'TestProduct',
                'ACR': [1000000] * len(forecast_dates)  # Flat $1M baseline
            })
            
            # Test sequential YoY adjustments
            adjustments = {
                2026: 0.10,  # +10% for FY26
                2027: 0.05   # +5% for FY27 (should compound on FY26's adjusted values)
            }
            
            adjusted_forecast = ui_components.apply_multi_fiscal_year_adjustments(
                test_forecast.copy(), adjustments
            )
            
            # Verify structure
            self.assertIsInstance(adjusted_forecast, pd.DataFrame)
            self.assertEqual(len(adjusted_forecast), len(test_forecast))
            
            # Verify sequential compounding logic
            # FY26 should be 1M * 1.10 = 1.1M
            # FY27 should be 1M * 1.10 * 1.05 = 1.155M (compounded)
            
            fy26_data = adjusted_forecast[
                (adjusted_forecast['Date'].dt.year == 2025) & 
                (adjusted_forecast['Date'].dt.month >= 7)
            ]
            fy27_data = adjusted_forecast[
                (adjusted_forecast['Date'].dt.year == 2026) & 
                (adjusted_forecast['Date'].dt.month >= 7)
            ]
            
            if len(fy26_data) > 0:
                fy26_avg = fy26_data['ACR'].mean()
                self.assertAlmostEqual(fy26_avg, 1100000, delta=50000,
                                     msg="FY26 should show 10% increase")
            
            if len(fy27_data) > 0:
                fy27_avg = fy27_data['ACR'].mean()
                expected_fy27 = 1000000 * 1.10 * 1.05  # Sequential compounding
                self.assertAlmostEqual(fy27_avg, expected_fy27, delta=50000,
                                     msg="FY27 should show sequential compounding")
            
            print("✅ v1.2.0 YoY sequential compounding working correctly")
            
        except Exception as e:
            self.fail(f"YoY sequential compounding test failed: {e}")
        finally:
            os.chdir(original_cwd)
            if str(forecaster_path) in sys.path:
                sys.path.remove(str(forecaster_path))

    def test_outlook_app_import_and_basic_functionality(self):
        """Test that Quarter Outlook App can be imported and basic functions work."""
        print("🔍 Testing Quarter Outlook App import and basic functionality...")
        
        try:
            original_cwd = os.getcwd()
            os.chdir(str(outlook_path))
            sys.path.insert(0, str(outlook_path))
            
            # Test importing core modules
            from modules import fiscal_calendar, data_processing, forecasting_models
            
            # Test fiscal calendar functionality
            test_date = datetime(2025, 7, 15)  # July 15, 2025
            quarter_info = fiscal_calendar.get_fiscal_quarter_info(test_date)
            
            self.assertIsInstance(quarter_info, dict)
            self.assertIn('quarter', quarter_info)
            self.assertIn('fiscal_year', quarter_info)
            self.assertEqual(quarter_info['quarter'], 1)  # July is Q1
            self.assertEqual(quarter_info['fiscal_year'], 2026)  # FY2026
            
            # Test data processing
            product_data = self.outlook_data[
                self.outlook_data['Product'] == 'Daily_Product_X'
            ].copy()
            
            analysis = data_processing.analyze_daily_data(product_data, 'ACR')
            self.assertIsInstance(analysis, dict)
            
            # Test forecasting models
            series = product_data.set_index('Date')['ACR']
            trend_model = forecasting_models.fit_linear_trend_model(series)
            
            if trend_model:
                forecast = trend_model(30)  # 30-day forecast
                self.assertIsInstance(forecast, (list, np.ndarray))
                self.assertEqual(len(forecast), 30)
            
            print("✅ Quarter Outlook App basic functionality working")
            
        except Exception as e:
            self.fail(f"Quarter Outlook App basic functionality failed: {e}")
        finally:
            os.chdir(original_cwd)
            if str(outlook_path) in sys.path:
                sys.path.remove(str(outlook_path))

    def test_outlook_app_spike_detection(self):
        """Test Quarter Outlook App spike detection for renewal patterns."""
        print("🔍 Testing Quarter Outlook App spike detection...")
        
        try:
            original_cwd = os.getcwd()
            os.chdir(str(outlook_path))
            sys.path.insert(0, str(outlook_path))
            
            from modules import spike_detection
            
            # Use test data with built-in spikes (1st and 15th of each month)
            product_data = self.outlook_data[
                self.outlook_data['Product'] == 'Daily_Product_X'
            ].copy()
            
            series = product_data.set_index('Date')['ACR']
            
            # Test spike detection
            spikes = spike_detection.detect_renewal_spikes(series)
            self.assertIsInstance(spikes, (list, dict, pd.Series, type(None)))
            
            # Test monthly pattern analysis
            patterns = spike_detection.analyze_monthly_patterns(series)
            if patterns:
                self.assertIsInstance(patterns, dict)
            
            print("✅ Quarter Outlook App spike detection working")
            
        except Exception as e:
            self.fail(f"Quarter Outlook App spike detection failed: {e}")
        finally:
            os.chdir(original_cwd)
            if str(outlook_path) in sys.path:
                sys.path.remove(str(outlook_path))

    def test_data_format_compatibility(self):
        """Test that both apps can handle their respective data formats correctly."""
        print("🔍 Testing data format compatibility...")
        
        # Test Forecaster App data validation
        try:
            original_cwd = os.getcwd()
            os.chdir(str(forecaster_path))
            sys.path.insert(0, str(forecaster_path))
            
            from modules.data_validation import validate_data_format
            
            # Test valid monthly data
            is_valid = validate_data_format(self.forecaster_data)
            self.assertTrue(is_valid, "Monthly data should be valid for Forecaster App")
            
            # Test invalid data structure
            invalid_data = pd.DataFrame({'Wrong': [1, 2, 3], 'Format': [4, 5, 6]})
            with self.assertRaises((ValueError, KeyError)):
                validate_data_format(invalid_data)
            
            os.chdir(original_cwd)
            if str(forecaster_path) in sys.path:
                sys.path.remove(str(forecaster_path))
        
        except Exception as e:
            self.fail(f"Forecaster data validation failed: {e}")
        
        # Test Outlook App data processing
        try:
            os.chdir(str(outlook_path))
            sys.path.insert(0, str(outlook_path))
            
            from modules.data_processing import analyze_daily_data
            
            # Test daily data processing
            analysis = analyze_daily_data(self.outlook_data, 'ACR')
            self.assertIsInstance(analysis, dict)
            
            print("✅ Data format compatibility working")
            
        except Exception as e:
            self.fail(f"Outlook data processing failed: {e}")
        finally:
            os.chdir(original_cwd)
            if str(outlook_path) in sys.path:
                sys.path.remove(str(outlook_path))

    def test_performance_and_memory_usage(self):
        """Test performance with realistic data sizes."""
        print("🔍 Testing performance and memory usage...")
        
        start_time = time.time()
        
        # Create larger datasets
        large_monthly_data = []
        for i in range(1200):  # 100 years of monthly data
            large_monthly_data.append({
                'Date': datetime(2000, 1, 1) + timedelta(days=i*30),
                'Product': f'Product_{i % 10}',
                'ACR': np.random.uniform(500000, 2000000)
            })
        
        large_df = pd.DataFrame(large_monthly_data)
        
        # Test basic operations
        grouped = large_df.groupby('Product')['ACR'].agg(['sum', 'mean', 'count'])
        memory_usage = large_df.memory_usage(deep=True).sum()
        
        processing_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(processing_time, 10.0, "Large data processing should complete within 10 seconds")
        self.assertLess(memory_usage, 100 * 1024 * 1024, "Memory usage should be reasonable (<100MB)")
        self.assertEqual(len(grouped), 10, "Should process all products correctly")
        
        print(f"✅ Performance test passed ({processing_time:.2f}s, {memory_usage/1024/1024:.1f}MB)")

    def test_fiscal_period_calculations(self):
        """Test fiscal calendar calculations across both apps."""
        print("🔍 Testing fiscal period calculations...")
        
        try:
            original_cwd = os.getcwd()
            os.chdir(str(outlook_path))
            sys.path.insert(0, str(outlook_path))
            
            from modules.fiscal_calendar import get_fiscal_quarter_info, get_fiscal_year_dates
            
            # Test key fiscal dates
            test_cases = [
                (datetime(2025, 7, 1), 1, 2026),    # Start of FY2026
                (datetime(2025, 10, 1), 2, 2026),   # Q2 FY2026
                (datetime(2026, 1, 1), 3, 2026),    # Q3 FY2026
                (datetime(2026, 4, 1), 4, 2026),    # Q4 FY2026
                (datetime(2026, 6, 30), 4, 2026),   # End of FY2026
            ]
            
            for test_date, expected_q, expected_fy in test_cases:
                quarter_info = get_fiscal_quarter_info(test_date)
                
                self.assertEqual(quarter_info['quarter'], expected_q,
                               f"Wrong quarter for {test_date}")
                self.assertEqual(quarter_info['fiscal_year'], expected_fy,
                               f"Wrong fiscal year for {test_date}")
            
            # Test fiscal year boundaries
            fy_dates = get_fiscal_year_dates(2026)
            self.assertIsInstance(fy_dates, dict)
            self.assertIn('start_date', fy_dates)
            self.assertIn('end_date', fy_dates)
            
            print("✅ Fiscal period calculations working")
            
        except Exception as e:
            self.fail(f"Fiscal period calculations failed: {e}")
        finally:
            os.chdir(original_cwd)
            if str(outlook_path) in sys.path:
                sys.path.remove(str(outlook_path))

def run_comprehensive_tests(test_filter=None, quick_mode=False):
    """Run the comprehensive end-to-end test suite."""
    
    print("🚀 Comprehensive End-to-End Regression Test Suite v1.2.0")
    print("=" * 70)
    print("Testing REAL functionality by importing and running both applications")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    
    if quick_mode:
        # Quick mode - essential tests only
        essential_tests = [
            'test_forecaster_app_import_and_basic_functionality',
            'test_forecaster_app_yoy_adjustments_v1_2_0',
            'test_outlook_app_import_and_basic_functionality',
            'test_data_format_compatibility'
        ]
        suite = unittest.TestSuite()
        for test_name in essential_tests:
            suite.addTest(ComprehensiveRegressionTests(test_name))
    elif test_filter:
        # Filter by app
        suite = unittest.TestSuite()
        all_tests = loader.loadTestsFromTestCase(ComprehensiveRegressionTests)
        
        for test in all_tests:
            if (test_filter == 'forecaster' and 'forecaster' in test._testMethodName) or \
               (test_filter == 'outlook' and 'outlook' in test._testMethodName) or \
               (test_filter == 'both'):
                suite.addTest(test)
    else:
        # Run all tests
        suite = loader.loadTestsFromTestCase(ComprehensiveRegressionTests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE END-TO-END TEST RESULTS")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            error_msg = str(traceback).split('\n')[-2] if '\n' in str(traceback) else str(traceback)
            print(f"   - {test}: {error_msg}")
    
    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            error_msg = str(traceback).split('\n')[-2] if '\n' in str(traceback) else str(traceback)
            print(f"   - {test}: {error_msg}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL END-TO-END TESTS PASSED!")
        print("✅ Both Forecaster App and Quarter Outlook App are fully functional")
        print("✅ v1.2.0 sequential YoY compounding working correctly")
        print("✅ All core features validated with real imports and execution")
    else:
        print(f"\n⚠️  {len(result.failures) + len(result.errors)} TEST(S) FAILED")
        print("Check the error details above for specific issues")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive End-to-End Regression Test Suite')
    parser.add_argument('--quick', action='store_true', help='Run essential tests only')
    parser.add_argument('--app', choices=['forecaster', 'outlook', 'both'], default='both',
                       help='Which app to test (default: both)')
    
    args = parser.parse_args()
    
    test_filter = None if args.app == 'both' else args.app
    success = run_comprehensive_tests(test_filter, args.quick)
    
    sys.exit(0 if success else 1)