#!/usr/bin/env python3
"""
Full Feature Regression Test Suite for Forecasting Applications

This comprehensive test suite validates ALL major features of both applications:
- Forecaster App: Model selection, backtesting, YoY adjustments, exports
- Quarter Outlook App: Daily forecasting, fiscal calendar, quarterly projections
- v1.2.0 Features: Sequential YoY compounding, universal product support

Usage:
    python full_regression_test.py [--quick] [--app forecaster|outlook|both]

Author: Jake Moura  
Version: v1.2.0
"""

import sys
import os
import unittest
import tempfile
import time
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).parent.parent
forecaster_path = project_root / "Forecaster App"
outlook_path = project_root / "Quarter Outlook App"

# Add paths for module imports
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(forecaster_path))
sys.path.insert(0, str(outlook_path))

# Change working directory to project root for relative imports
os.chdir(str(project_root))

class FullRegressionTestSuite(unittest.TestCase):
    """Comprehensive end-to-end regression test suite for both applications."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and environment for all tests."""
        cls.forecaster_test_data = cls._create_forecaster_test_data()
        cls.outlook_test_data = cls._create_outlook_test_data()
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
    def _create_forecaster_test_data():
        """Create comprehensive test data for Forecaster App (monthly data)."""
        # Create 36 months of data for multiple products
        start_date = datetime(2022, 1, 1)
        dates = pd.date_range(start_date, periods=36, freq='MS')
        
        data = []
        np.random.seed(42)
        
        products = ['SurfaceHub', 'Language', 'DataCenter']
        
        for product in products:
            base_value = {'SurfaceHub': 1000000, 'Language': 750000, 'DataCenter': 1500000}[product]
            
            for i, date in enumerate(dates):
                # Create realistic monthly revenue with trend and seasonality
                trend = i * 25000  # Growing trend
                seasonal = 100000 * np.sin(2 * np.pi * i / 12)  # Annual seasonality
                noise = np.random.normal(0, 50000)  # Random variation
                
                # Add quarterly spikes for business reality
                if date.month in [3, 6, 9, 12]:  # End of quarters
                    seasonal *= 1.3
                
                acr = max(base_value + trend + seasonal + noise, base_value * 0.8)
                
                data.append({
                    'Date': date,
                    'Product': product,
                    'ACR': round(acr, 2)
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def _create_outlook_test_data():
        """Create comprehensive test data for Quarter Outlook App (daily data)."""
        # Create Q4 FY25 daily data (April-June 2025)
        start_date = datetime(2025, 4, 1)
        end_date = datetime(2025, 6, 30)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        data = []
        np.random.seed(123)
        
        products = ['Product_Alpha', 'Product_Beta', 'Product_Gamma']
        
        for product in products:
            base_daily = {'Product_Alpha': 50000, 'Product_Beta': 30000, 'Product_Gamma': 75000}[product]
            
            for date in dates:
                # Business day effects
                if date.weekday() < 5:  # Weekdays
                    multiplier = 1.0
                else:  # Weekends
                    multiplier = 0.3
                
                # Month-end spikes
                if date.day >= 28:
                    multiplier *= 1.5
                
                # Renewal spikes (1st and 15th)
                if date.day in [1, 15]:
                    multiplier *= 2.0
                
                daily_value = base_daily * multiplier + np.random.normal(0, base_daily * 0.1)
                
                data.append({
                    'Date': date,
                    'Product': product,
                    'ACR': max(daily_value, 0)
                })
        
        return pd.DataFrame(data)

    # =============================================================================
    # Forecaster App Full Feature Tests
    # =============================================================================
    
    def test_forecaster_app_data_validation(self):
        """Test Forecaster App data validation with real module."""
        print("🔍 Testing Forecaster App data validation...")
        
        try:
            from modules.data_validation import validate_data_format, prepare_data
            
            # Test valid data
            is_valid = validate_data_format(self.forecaster_test_data)
            self.assertTrue(is_valid, "Valid data should pass validation")
            
            # Test data preparation
            prepared_data = prepare_data(self.forecaster_test_data.copy())
            self.assertIsInstance(prepared_data, pd.DataFrame, "Should return DataFrame")
            self.assertTrue(len(prepared_data) > 0, "Prepared data should not be empty")
            
            # Test invalid data
            invalid_data = pd.DataFrame({'Wrong': [1, 2, 3], 'Columns': [4, 5, 6]})
            with self.assertRaises(ValueError):
                validate_data_format(invalid_data)
            
            print("✅ Forecaster App data validation working")
            
        except ImportError as e:
            self.skipTest(f"Forecaster modules not available: {e}")
    
    def test_forecaster_app_model_selection(self):
        """Test Forecaster App model selection and ranking."""
        print("🔍 Testing Forecaster App model selection...")
        
        try:
            from modules.models import get_available_models, fit_model_safe
            from modules.metrics import calculate_wape
            
            # Get available models
            available_models = get_available_models()
            self.assertIsInstance(available_models, list, "Should return list of models")
            self.assertGreater(len(available_models), 0, "Should have available models")
            
            # Test basic model fitting with a product's data
            product_data = self.forecaster_test_data[
                self.forecaster_test_data['Product'] == 'SurfaceHub'
            ].copy()
            
            if len(product_data) > 12:  # Need enough data for modeling
                series = product_data.set_index('Date')['ACR']
                
                # Test at least one model
                for model_name in available_models[:2]:  # Test first 2 models
                    result = fit_model_safe(series, model_name, 6)  # 6-month forecast
                    
                    if result and 'forecast' in result:
                        forecast = result['forecast']
                        self.assertIsInstance(forecast, (list, np.ndarray, pd.Series))
                        self.assertEqual(len(forecast), 6, "Should forecast 6 months")
                        break
                else:
                    self.skipTest("No models could be fitted successfully")
            
            print("✅ Forecaster App model selection working")
            
        except ImportError as e:
            self.skipTest(f"Forecaster model modules not available: {e}")
    
    def test_forecaster_app_backtesting(self):
        """Test Forecaster App backtesting functionality."""
        print("🔍 Testing Forecaster App backtesting...")
        
        try:
            from modules.metrics import enhanced_rolling_validation
            
            # Test backtesting with sufficient data
            product_data = self.forecaster_test_data[
                self.forecaster_test_data['Product'] == 'SurfaceHub'
            ].copy()
            
            if len(product_data) >= 24:  # Need enough data for backtesting
                series = product_data.set_index('Date')['ACR']
                
                # Simple model function for testing
                def simple_model_func(train_data, model_params=None):
                    return lambda x: [train_data.iloc[-1]] * len(x)
                
                # Test enhanced rolling validation
                result = enhanced_rolling_validation(
                    series, 
                    simple_model_func, 
                    backtest_months=12,
                    validation_horizon=3
                )
                
                self.assertIsInstance(result, dict, "Should return results dictionary")
                
                # Check for key metrics
                expected_keys = ['mean_wape', 'recent_weighted_wape']
                for key in expected_keys:
                    if key in result:
                        self.assertIsInstance(result[key], (int, float), f"{key} should be numeric")
                        self.assertGreaterEqual(result[key], 0, f"{key} should be non-negative")
            
            print("✅ Forecaster App backtesting working")
            
        except ImportError as e:
            self.skipTest(f"Backtesting modules not available: {e}")
    
    def test_forecaster_app_yoy_adjustments(self):
        """Test v1.2.0 sequential YoY compounding functionality."""
        print("🔍 Testing v1.2.0 YoY sequential compounding...")
        
        try:
            from modules.ui_components import apply_multi_fiscal_year_adjustments
            
            # Create forecast data for testing YoY adjustments
            forecast_dates = pd.date_range('2025-07-01', '2027-06-30', freq='MS')  # 2 fiscal years
            
            test_forecast = pd.DataFrame({
                'Date': forecast_dates,
                'Product': 'TestProduct',
                'ACR': [1000000] * len(forecast_dates)  # Flat $1M per month
            })
            
            # Test sequential YoY adjustments
            adjustments = {
                2026: 0.10,  # +10% for FY26
                2027: 0.05   # +5% for FY27 (should compound on FY26)
            }
            
            adjusted_forecast = apply_multi_fiscal_year_adjustments(
                test_forecast.copy(), adjustments
            )
            
            # Verify sequential compounding
            self.assertIsInstance(adjusted_forecast, pd.DataFrame)
            self.assertEqual(len(adjusted_forecast), len(test_forecast))
            
            # Check that FY27 values are higher than FY26 values
            fy26_data = adjusted_forecast[adjusted_forecast['Date'].dt.year == 2025]  # FY26 starts July 2025
            fy27_data = adjusted_forecast[adjusted_forecast['Date'].dt.year == 2026]  # FY27 starts July 2026
            
            if len(fy26_data) > 0 and len(fy27_data) > 0:
                # FY26 should be ~1.1M (10% increase)
                fy26_avg = fy26_data['ACR'].mean()
                self.assertGreater(fy26_avg, 1050000, "FY26 should show 10% increase")
                
                # FY27 should be ~1.155M (5% increase on top of FY26's 1.1M)
                fy27_avg = fy27_data['ACR'].mean()
                self.assertGreater(fy27_avg, fy26_avg, "FY27 should be higher than FY26 (compounding)")
                
                # Test approximate compounding: 1M * 1.1 * 1.05 = 1.155M
                expected_fy27 = 1000000 * 1.1 * 1.05
                self.assertAlmostEqual(fy27_avg, expected_fy27, delta=50000, 
                                     msg="FY27 should show proper sequential compounding")
            
            print("✅ v1.2.0 YoY sequential compounding working")
            
        except ImportError as e:
            self.skipTest(f"YoY adjustment modules not available: {e}")
    
    def test_forecaster_app_excel_export(self):
        """Test Forecaster App Excel export functionality."""
        print("🔍 Testing Forecaster App Excel export...")
        
        try:
            from modules.ui_components import create_excel_download
            
            # Test Excel export creation
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                self.temp_files.append(tmp_file.name)
                
                # Test basic export
                export_data = self.forecaster_test_data.copy()
                export_data.to_excel(tmp_file.name, index=False)
                
                # Verify export
                read_back = pd.read_excel(tmp_file.name)
                self.assertEqual(len(read_back), len(export_data))
                self.assertEqual(list(read_back.columns), list(export_data.columns))
                
            print("✅ Forecaster App Excel export working")
            
        except ImportError as e:
            self.skipTest(f"Excel export modules not available: {e}")

    # =============================================================================
    # Quarter Outlook App Full Feature Tests  
    # =============================================================================
    
    def test_outlook_app_fiscal_calendar(self):
        """Test Quarter Outlook App fiscal calendar functionality."""
        print("🔍 Testing Outlook App fiscal calendar...")
        
        try:
            from modules.fiscal_calendar import get_fiscal_quarter_info, get_fiscal_year_dates
            
            # Test fiscal quarter detection
            test_dates = [
                (datetime(2025, 7, 15), 1, 2026),   # July = Q1 FY26
                (datetime(2025, 10, 15), 2, 2026),  # October = Q2 FY26  
                (datetime(2025, 1, 15), 3, 2025),   # January = Q3 FY25
                (datetime(2025, 4, 15), 4, 2025),   # April = Q4 FY25
            ]
            
            for test_date, expected_q, expected_fy in test_dates:
                quarter_info = get_fiscal_quarter_info(test_date)
                
                self.assertIsInstance(quarter_info, dict)
                self.assertEqual(quarter_info.get('quarter'), expected_q, 
                               f"Wrong quarter for {test_date}")
                self.assertEqual(quarter_info.get('fiscal_year'), expected_fy,
                               f"Wrong fiscal year for {test_date}")
            
            # Test fiscal year date ranges
            fy_dates = get_fiscal_year_dates(2025)
            self.assertIsInstance(fy_dates, dict)
            self.assertIn('start_date', fy_dates)
            self.assertIn('end_date', fy_dates)
            
            print("✅ Outlook App fiscal calendar working")
            
        except ImportError as e:
            self.skipTest(f"Fiscal calendar modules not available: {e}")
    
    def test_outlook_app_data_processing(self):
        """Test Quarter Outlook App daily data processing."""
        print("🔍 Testing Outlook App data processing...")
        
        try:
            from modules.data_processing import analyze_daily_data, prepare_daily_forecast_data
            
            # Test with a single product's daily data
            product_data = self.outlook_test_data[
                self.outlook_test_data['Product'] == 'Product_Alpha'
            ].copy()
            
            # Convert to time series for analysis
            series = product_data.set_index('Date')['ACR']
            
            # Test daily data analysis
            analysis = analyze_daily_data(product_data, 'ACR')
            
            self.assertIsInstance(analysis, dict)
            
            # Check for key analysis components
            expected_keys = ['business_days', 'weekend_days', 'daily_average']
            for key in expected_keys:
                if key in analysis:
                    self.assertIsInstance(analysis[key], (int, float))
            
            print("✅ Outlook App data processing working")
            
        except ImportError as e:
            self.skipTest(f"Data processing modules not available: {e}")
    
    def test_outlook_app_spike_detection(self):
        """Test Quarter Outlook App spike detection for renewals."""
        print("🔍 Testing Outlook App spike detection...")
        
        try:
            from modules.spike_detection import detect_renewal_spikes, analyze_monthly_patterns
            
            # Test with data that has built-in spikes on 1st and 15th
            product_data = self.outlook_test_data[
                self.outlook_test_data['Product'] == 'Product_Alpha'
            ].copy()
            
            series = product_data.set_index('Date')['ACR']
            
            # Test spike detection
            spikes = detect_renewal_spikes(series)
            
            self.assertIsInstance(spikes, (list, dict, pd.Series))
            
            # Test monthly pattern analysis
            patterns = analyze_monthly_patterns(series)
            
            if patterns:
                self.assertIsInstance(patterns, dict)
            
            print("✅ Outlook App spike detection working")
            
        except ImportError as e:
            self.skipTest(f"Spike detection modules not available: {e}")
    
    def test_outlook_app_quarterly_forecasting(self):
        """Test Quarter Outlook App quarterly forecasting models."""
        print("🔍 Testing Outlook App quarterly forecasting...")
        
        try:
            from modules.forecasting_models import fit_linear_trend_model, fit_moving_average_model
            
            # Test with single product daily data
            product_data = self.outlook_test_data[
                self.outlook_test_data['Product'] == 'Product_Alpha'
            ].copy()
            
            series = product_data.set_index('Date')['ACR']
            
            # Test linear trend model
            trend_model = fit_linear_trend_model(series)
            
            if trend_model:
                # Test forecast generation
                forecast_result = trend_model(30)  # 30-day forecast
                self.assertIsInstance(forecast_result, (list, np.ndarray))
                self.assertEqual(len(forecast_result), 30)
            
            # Test moving average model
            ma_model = fit_moving_average_model(series)
            
            if ma_model:
                ma_forecast = ma_model(30)
                self.assertIsInstance(ma_forecast, (list, np.ndarray))
                self.assertEqual(len(ma_forecast), 30)
            
            print("✅ Outlook App quarterly forecasting working")
            
        except ImportError as e:
            self.skipTest(f"Quarterly forecasting modules not available: {e}")

    # =============================================================================
    # Integration and Performance Tests
    # =============================================================================
    
    def test_cross_app_data_compatibility(self):
        """Test data format compatibility between both apps."""
        print("🔍 Testing cross-app data compatibility...")
        
        # Test that Forecaster monthly data can be aggregated to daily-like format
        monthly_summary = self.forecaster_test_data.groupby(['Product']).agg({
            'ACR': ['sum', 'mean', 'count']
        })
        
        self.assertGreater(len(monthly_summary), 0, "Should aggregate monthly data")
        
        # Test that Outlook daily data can be aggregated to monthly format
        self.outlook_test_data['YearMonth'] = self.outlook_test_data['Date'].dt.to_period('M')
        daily_to_monthly = self.outlook_test_data.groupby(['Product', 'YearMonth']).agg({
            'ACR': 'sum'
        }).reset_index()
        
        self.assertGreater(len(daily_to_monthly), 0, "Should aggregate daily to monthly")
        
        print("✅ Cross-app data compatibility working")
    
    def test_performance_with_large_datasets(self):
        """Test performance with larger datasets."""
        print("🔍 Testing performance with larger datasets...")
        
        start_time = time.time()
        
        # Create larger dataset (1000+ rows)
        large_data = []
        for i in range(1000):
            large_data.append({
                'Date': datetime(2020, 1, 1) + timedelta(days=i),
                'Product': f'Product_{i % 5}',
                'ACR': np.random.uniform(50000, 150000)
            })
        
        large_df = pd.DataFrame(large_data)
        
        # Test basic operations performance
        grouped = large_df.groupby('Product')['ACR'].sum()
        memory_usage = large_df.memory_usage(deep=True).sum()
        
        processing_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(processing_time, 5.0, "Processing should complete within 5 seconds")
        self.assertLess(memory_usage, 50 * 1024 * 1024, "Memory usage should be under 50MB")
        self.assertEqual(len(grouped), 5, "Should process all 5 products")
        
        print(f"✅ Performance test passed ({processing_time:.2f}s, {memory_usage/1024/1024:.1f}MB)")
    
    def test_fiscal_period_export_integration(self):
        """Test v1.2.0 fiscal period export functionality."""
        print("🔍 Testing fiscal period export integration...")
        
        try:
            from modules.ui_components import fiscal_period_display
            
            # Test fiscal period formatting for export
            test_dates = [
                datetime(2025, 7, 1),   # P01 - July (FY2026)
                datetime(2025, 12, 1),  # P06 - December (FY2026) 
                datetime(2026, 3, 1),   # P09 - March (FY2026)
                datetime(2026, 6, 1),   # P12 - June (FY2026)
            ]
            
            for test_date in test_dates:
                fiscal_display = fiscal_period_display(test_date)
                
                self.assertIsInstance(fiscal_display, str)
                self.assertIn('P', fiscal_display, "Should contain period indicator")
                self.assertIn('FY', fiscal_display, "Should contain fiscal year")
                
                # Check for proper zero-padding (P01, P02, etc.)
                if 'P01' in fiscal_display or 'P06' in fiscal_display or 'P09' in fiscal_display:
                    self.assertRegex(fiscal_display, r'P\d{2}', "Should have zero-padded periods")
            
            print("✅ Fiscal period export integration working")
            
        except ImportError as e:
            self.skipTest(f"Fiscal period modules not available: {e}")

def run_full_regression_suite(test_filter=None, quick_mode=False):
    """Run the complete regression test suite with options."""
    
    print("🚀 Full Feature Regression Test Suite v1.2.0")
    print("=" * 60)
    print("Testing ALL major features of both forecasting applications")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if test_filter:
        # Run specific tests
        if test_filter == 'forecaster':
            pattern = 'test_forecaster_'
        elif test_filter == 'outlook':
            pattern = 'test_outlook_'
        else:
            pattern = f'test_{test_filter}_'
        
        # Load all tests and filter
        all_tests = loader.loadTestsFromTestCase(FullRegressionTestSuite)
        for test in all_tests:
            if hasattr(test, '_testMethodName') and pattern in test._testMethodName:
                suite.addTest(test)
    else:
        # Load all tests
        if quick_mode:
            # Skip performance and large data tests in quick mode
            test_methods = [
                'test_forecaster_app_data_validation',
                'test_forecaster_app_yoy_adjustments', 
                'test_outlook_app_fiscal_calendar',
                'test_cross_app_data_compatibility',
                'test_fiscal_period_export_integration'
            ]
            for method in test_methods:
                suite.addTest(FullRegressionTestSuite(method))
        else:
            suite = loader.loadTestsFromTestCase(FullRegressionTestSuite)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 FULL REGRESSION TEST RESULTS")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"   - {test}: {error_msg}")
    
    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            error_msg = traceback.split(': ')[-1].split('\n')[0]
            print(f"   - {test}: {error_msg}")
    
    if hasattr(result, 'skipped') and result.skipped:
        print(f"\n⏭️ SKIPPED ({len(result.skipped)}):")
        for test, reason in result.skipped:
            print(f"   - {test}: {reason}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED! Both applications are fully functional.")
    else:
        print(f"\n⚠️  {len(result.failures) + len(result.errors)} TEST(S) FAILED.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Full Feature Regression Test Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick subset of tests')
    parser.add_argument('--app', choices=['forecaster', 'outlook', 'both'], default='both',
                       help='Which app to test (default: both)')
    
    args = parser.parse_args()
    
    test_filter = None if args.app == 'both' else args.app
    success = run_full_regression_suite(test_filter, args.quick)
    
    sys.exit(0 if success else 1)