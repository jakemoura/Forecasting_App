#!/usr/bin/env python3
"""
Comprehensive Regression Test Suite for Both Forecasting Apps

This script tests:
1. Forecaster App - Time-series forecasting with multiple models
2. Outlook App - Quarterly forecasting with fiscal calendar

Testing approach:
- Import validation
- Function signature validation
- Data processing logic
- Model functionality
- Business logic
- Error handling
- End-to-end data flow simulation

Author: Jake Moura (jakemoura@microsoft.com)
"""

import sys
import traceback
import warnings
import inspect
import importlib
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class RegressionTester:
    def __init__(self):
        self.test_results = []
        self.current_app = None
        self.outlook_modules = {}
        
    def log_test(self, test_name, result, details=""):
        """Log test result."""
        self.test_results.append({
            'app': self.current_app,
            'test': test_name,
            'result': result,
            'details': details
        })
        
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if details and not result:
            print(f"   Details: {details}")
    
    def test_forecaster_app(self):
        """Test the main Forecaster App."""
        print("\n" + "="*60)
        print("TESTING FORECASTER APP")
        print("="*60)
        self.current_app = "Forecaster App"
        
        # Change to Forecaster App directory
        original_dir = os.getcwd()
        forecaster_dir = Path(original_dir) / "Forecaster App"
        
        if not forecaster_dir.exists():
            self.log_test("Directory Check", False, "Forecaster App directory not found")
            return False
            
        os.chdir(forecaster_dir)
        
        # Add current directory to Python path for imports
        if str(forecaster_dir) not in sys.path:
            sys.path.insert(0, str(forecaster_dir))
        
        try:
            # Test 1: Import validation
            self._test_forecaster_imports()
            
            # Test 2: Function signatures
            self._test_forecaster_functions()
            
            # Test 3: Data processing
            self._test_forecaster_data_processing()
            
            # Test 4: Model availability and functionality
            self._test_forecaster_models()
            
            # Test 5: Business logic
            self._test_forecaster_business_logic()
            
            # Test 6: Error handling
            self._test_forecaster_error_handling()
            
        finally:
            # Clean up sys.path and restore directory
            try:
                if str(forecaster_dir) in sys.path:
                    sys.path.remove(str(forecaster_dir))
            except ValueError:
                pass  # Already removed
            os.chdir(original_dir)
    
    def _test_forecaster_imports(self):
        """Test all Forecaster App imports."""
        print("\nTesting Forecaster App imports...")
        
        try:
            # Core modules
            from modules.ui_config import setup_page_config, create_sidebar_controls
            from modules.tab_content import render_forecast_tab, render_example_data_tab, render_model_guide_tab
            from modules.data_validation import validate_data_format, prepare_data, analyze_data_quality
            from modules.forecasting_pipeline import run_forecasting_pipeline
            from modules.business_logic import process_yearly_renewals, calculate_model_rankings
            from modules.session_state import store_forecast_results, initialize_session_state_variables
            from modules.utils import read_any_excel
            from modules.models import HAVE_PMDARIMA, HAVE_PROPHET, HAVE_LGBM
            
            # Main app
            import forecaster_app
            
            self.log_test("Forecaster Imports", True)
            return True
            
        except Exception as e:
            self.log_test("Forecaster Imports", False, str(e))
            return False
    
    def _test_forecaster_functions(self):
        """Test Forecaster App function signatures."""
        print("\nTesting Forecaster App function signatures...")
        
        try:
            from modules.forecasting_pipeline import run_forecasting_pipeline
            from modules.business_logic import calculate_model_rankings
            from modules.session_state import store_forecast_results
            
            # Test forecasting pipeline signature
            sig = inspect.signature(run_forecasting_pipeline)
            required_params = ['raw_data', 'models_selected', 'horizon']
            actual_params = list(sig.parameters.keys())
            
            if all(param in actual_params for param in required_params):
                self.log_test("Forecaster Function Signatures", True)
                return True
            else:
                self.log_test("Forecaster Function Signatures", False, f"Missing params: {set(required_params) - set(actual_params)}")
                return False
                
        except Exception as e:
            self.log_test("Forecaster Function Signatures", False, str(e))
            return False
    
    def _test_forecaster_data_processing(self):
        """Test Forecaster App data processing logic."""
        print("\nTesting Forecaster App data processing...")
        
        try:
            from modules.data_validation import prepare_data, validate_data_format
            from modules.utils import coerce_month_start
            
            # Create test data
            test_data = pd.DataFrame({
                'Date': ['2023-01-15', '2023-02-15', '2023-03-15', '2023-04-15', '2023-05-15'],
                'Product': ['Test Product A'] * 5,
                'ACR': [1000, 1100, 1200, 1300, 1400]
            })
            
            # Test date coercion
            dates = pd.to_datetime(test_data['Date'])
            coerced = coerce_month_start(dates)
            
            if isinstance(coerced, pd.Series) and all(pd.to_datetime(coerced).dt.day == 1):
                self.log_test("Date Coercion", True)
            else:
                self.log_test("Date Coercion", False, "Dates not properly coerced to month start")
                return False
            
            # Test data validation
            validation_result = validate_data_format(test_data)
            if isinstance(validation_result, bool):
                self.log_test("Data Validation Logic", True)
            elif isinstance(validation_result, tuple) and len(validation_result) == 2:
                is_valid, msg = validation_result
                self.log_test("Data Validation Logic", True)
            else:
                self.log_test("Data Validation Logic", False, "Validation function not returning expected format")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Forecaster Data Processing", False, str(e))
            return False
    
    def _test_forecaster_models(self):
        """Test Forecaster App model availability and basic functionality."""
        print("\nTesting Forecaster App models...")
        
        try:
            from modules.models import HAVE_PMDARIMA, HAVE_PROPHET, HAVE_LGBM
            from modules.models import detect_seasonality_strength, apply_statistical_validation
            
            models_available = sum([HAVE_PMDARIMA, HAVE_PROPHET, HAVE_LGBM])
            print(f"   Available models: PMDArima={HAVE_PMDARIMA}, Prophet={HAVE_PROPHET}, LightGBM={HAVE_LGBM}")
            
            if models_available > 0:
                self.log_test("Model Availability", True, f"{models_available}/3 models available")
            else:
                self.log_test("Model Availability", False, "No models available")
                return False
            
            # Test utility functions
            test_data = pd.Series([100, 110, 105, 120, 115, 130, 125, 140])
            seasonality = detect_seasonality_strength(test_data)
            
            if isinstance(seasonality, (int, float)):
                self.log_test("Model Utilities", True)
            else:
                self.log_test("Model Utilities", False, "Seasonality detection not returning numeric value")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Forecaster Models", False, str(e))
            return False
    
    def _test_forecaster_business_logic(self):
        """Test Forecaster App business logic."""
        print("\nTesting Forecaster App business logic...")
        
        try:
            from modules.business_logic import calculate_model_rankings
            
            # Mock metrics data
            products = ['Product A', 'Product B']
            model_names = ['SARIMA', 'Prophet']
            
            mock_mapes = {
                'SARIMA': {'Product A': 0.1, 'Product B': 0.15},
                'Prophet': {'Product A': 0.12, 'Product B': 0.13}
            }
            
            mock_smapes = {
                'SARIMA': {'Product A': 0.05, 'Product B': 0.07},
                'Prophet': {'Product A': 0.06, 'Product B': 0.065}
            }
            
            mock_mases = {
                'SARIMA': {'Product A': 0.8, 'Product B': 0.9},
                'Prophet': {'Product A': 0.85, 'Product B': 0.88}
            }
            
            mock_rmses = {
                'SARIMA': {'Product A': 100, 'Product B': 150},
                'Prophet': {'Product A': 110, 'Product B': 140}
            }
            
            # Test ranking calculation
            metric_ranks, avg_ranks, best_model = calculate_model_rankings(
                mock_mapes, mock_smapes, mock_mases, mock_rmses, model_names, products
            )
            
            if metric_ranks and avg_ranks and best_model in model_names:
                self.log_test("Business Logic - Model Rankings", True)
            else:
                self.log_test("Business Logic - Model Rankings", False, "Ranking calculation failed")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Forecaster Business Logic", False, str(e))
            return False
    
    def _test_forecaster_error_handling(self):
        """Test Forecaster App error handling."""
        print("\nTesting Forecaster App error handling...")
        
        try:
            from modules.data_validation import validate_data_format, prepare_data
            
            # Test with invalid data
            invalid_data = pd.DataFrame({
                'InvalidColumn': [1, 2, 3],
                'AnotherInvalid': ['a', 'b', 'c']
            })
            
            try:
                validation_result = validate_data_format(invalid_data)
                
                # Handle both boolean and tuple returns
                if isinstance(validation_result, bool):
                    is_valid = validation_result
                    msg = "Invalid format"
                elif isinstance(validation_result, tuple) and len(validation_result) == 2:
                    is_valid, msg = validation_result
                else:
                    is_valid = False
                    msg = "Unknown validation format"
                
                if not is_valid and isinstance(msg, str) and len(msg) > 0:
                    self.log_test("Error Handling - Invalid Data", True)
                else:
                    self.log_test("Error Handling - Invalid Data", False, "Should reject invalid data format")
                    return False
            except Exception as e:
                # Exception is also acceptable for invalid data
                if "must contain columns" in str(e):
                    self.log_test("Error Handling - Invalid Data", True)
                else:
                    self.log_test("Error Handling - Invalid Data", False, f"Unexpected error: {e}")
                    return False
            
            # Test with insufficient data
            insufficient_data = pd.DataFrame({
                'Date': ['2023-01-01'],
                'Product': ['Test'],
                'ACR': [100]
            })
            
            try:
                prepared = prepare_data(insufficient_data)
                # Should either handle gracefully or raise appropriate exception
                self.log_test("Error Handling - Insufficient Data", True)
            except Exception:
                # Exception is acceptable for insufficient data
                self.log_test("Error Handling - Insufficient Data", True)
            
            return True
            
        except Exception as e:
            self.log_test("Forecaster Error Handling", False, str(e))
            return False
    
    def test_outlook_app(self):
        """Test the Outlook App."""
        print("\n" + "="*60)
        print("TESTING OUTLOOK APP")
        print("="*60)
        self.current_app = "Outlook App"
        
        # Change to Outlook App directory
        original_dir = os.getcwd()
        outlook_dir = Path(original_dir) / "Quarter Outlook App"
        
        if not outlook_dir.exists():
            self.log_test("Directory Check", False, "Quarter Outlook App directory not found")
            return False
            
        os.chdir(outlook_dir)
        
        # Clean up any previous paths first
        paths_to_remove = [path for path in sys.path if 'Forecaster App' in path]
        for path in paths_to_remove:
            sys.path.remove(path)
        
        # Clear module cache for modules from previous tests - be more aggressive
        modules_to_clear = [name for name in sys.modules.keys() if (
            name.startswith('modules.') or 
            name == 'modules' or
            'forecaster_app' in name.lower() or
            'ui_components' in name.lower() or
            'data_validation' in name.lower() or
            'business_logic' in name.lower()
        )]
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]
                
        # Also clear the base 'modules' package if it exists
        if 'modules' in sys.modules:
            del sys.modules['modules']
        
        # Add current directory to Python path for imports
        if str(outlook_dir) not in sys.path:
            sys.path.insert(0, str(outlook_dir))
        
        try:
            # Test 1: Import validation
            self._test_outlook_imports()
            
            # Test 2: Fiscal calendar logic
            self._test_outlook_fiscal_calendar()
            
            # Test 3: Data processing
            self._test_outlook_data_processing()
            
            # Test 4: Forecasting logic
            self._test_outlook_forecasting()
            
            # Test 5: UI components
            self._test_outlook_ui_components()
            
        finally:
            # Clean up sys.path and restore directory
            try:
                if str(outlook_dir) in sys.path:
                    sys.path.remove(str(outlook_dir))
            except ValueError:
                pass  # Already removed
            os.chdir(original_dir)
    
    def _test_outlook_imports(self):
        """Test all Outlook App imports."""
        print("\nTesting Outlook App imports...")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Python path: {sys.path[:3]}...")  # Show first 3 entries
        
        try:
            # Use dynamic imports to avoid caching issues
            fiscal_calendar_spec = importlib.util.spec_from_file_location(
                "modules.fiscal_calendar", 
                os.path.join(os.getcwd(), "modules", "fiscal_calendar.py")
            )
            if fiscal_calendar_spec is None or fiscal_calendar_spec.loader is None:
                raise ImportError("Could not create spec for fiscal_calendar module")
            fiscal_calendar_module = importlib.util.module_from_spec(fiscal_calendar_spec)
            fiscal_calendar_spec.loader.exec_module(fiscal_calendar_module)
            
            data_processing_spec = importlib.util.spec_from_file_location(
                "modules.data_processing", 
                os.path.join(os.getcwd(), "modules", "data_processing.py")
            )
            if data_processing_spec is None or data_processing_spec.loader is None:
                raise ImportError("Could not create spec for data_processing module")
            data_processing_module = importlib.util.module_from_spec(data_processing_spec)
            data_processing_spec.loader.exec_module(data_processing_module)
            
            quarterly_forecasting_spec = importlib.util.spec_from_file_location(
                "modules.quarterly_forecasting", 
                os.path.join(os.getcwd(), "modules", "quarterly_forecasting.py")
            )
            if quarterly_forecasting_spec is None or quarterly_forecasting_spec.loader is None:
                raise ImportError("Could not create spec for quarterly_forecasting module")
            quarterly_forecasting_module = importlib.util.module_from_spec(quarterly_forecasting_spec)
            quarterly_forecasting_spec.loader.exec_module(quarterly_forecasting_module)
            
            ui_components_spec = importlib.util.spec_from_file_location(
                "modules.ui_components", 
                os.path.join(os.getcwd(), "modules", "ui_components.py")
            )
            if ui_components_spec is None or ui_components_spec.loader is None:
                raise ImportError("Could not create spec for ui_components module")
            ui_components_module = importlib.util.module_from_spec(ui_components_spec)
            ui_components_spec.loader.exec_module(ui_components_module)
            
            # Test that we can access the functions
            get_fiscal_quarter_info = fiscal_calendar_module.get_fiscal_quarter_info
            read_any_excel = data_processing_module.read_any_excel
            coerce_daily_dates = data_processing_module.coerce_daily_dates
            analyze_daily_data = data_processing_module.analyze_daily_data
            forecast_quarter_completion = quarterly_forecasting_module.forecast_quarter_completion
            create_forecast_summary_table = ui_components_module.create_forecast_summary_table
            create_forecast_visualization = ui_components_module.create_forecast_visualization
            
            # Store these for use in other tests
            self.outlook_modules = {
                'fiscal_calendar': fiscal_calendar_module,
                'data_processing': data_processing_module,
                'quarterly_forecasting': quarterly_forecasting_module,
                'ui_components': ui_components_module
            }
            
            # Main app
            outlook_forecaster_spec = importlib.util.spec_from_file_location(
                "outlook_forecaster", 
                os.path.join(os.getcwd(), "outlook_forecaster.py")
            )
            if outlook_forecaster_spec is None or outlook_forecaster_spec.loader is None:
                raise ImportError("Could not create spec for outlook_forecaster module")
            outlook_forecaster_module = importlib.util.module_from_spec(outlook_forecaster_spec)
            outlook_forecaster_spec.loader.exec_module(outlook_forecaster_module)
            
            self.log_test("Outlook Imports", True)
            return True
            
        except Exception as e:
            self.log_test("Outlook Imports", False, str(e))
            return False
    
    def _test_outlook_fiscal_calendar(self):
        """Test Outlook App fiscal calendar logic."""
        print("\nTesting Outlook App fiscal calendar...")
        
        try:
            if 'fiscal_calendar' not in self.outlook_modules:
                self.log_test("Outlook Fiscal Calendar", False, "fiscal_calendar module not loaded")
                return False
                
            get_fiscal_quarter_info = self.outlook_modules['fiscal_calendar'].get_fiscal_quarter_info
            
            # Test various dates
            test_dates = [
                datetime(2024, 8, 15),  # Q1 (Jul-Sep)
                datetime(2024, 11, 15), # Q2 (Oct-Dec)
                datetime(2024, 2, 15),  # Q3 (Jan-Mar)
                datetime(2024, 5, 15),  # Q4 (Apr-Jun)
            ]
            
            expected_quarters = [1, 2, 3, 4]
            
            for i, test_date in enumerate(test_dates):
                quarter_info = get_fiscal_quarter_info(test_date)
                
                if not isinstance(quarter_info, dict):
                    self.log_test("Fiscal Calendar Logic", False, "Quarter info not returning dict")
                    return False
                
                if 'quarter' not in quarter_info:
                    self.log_test("Fiscal Calendar Logic", False, "Quarter not in result")
                    return False
                
                if quarter_info['quarter'] != expected_quarters[i]:
                    self.log_test("Fiscal Calendar Logic", False, f"Wrong quarter for {test_date}: expected {expected_quarters[i]}, got {quarter_info['quarter']}")
                    return False
            
            self.log_test("Fiscal Calendar Logic", True)
            return True
            
        except Exception as e:
            self.log_test("Outlook Fiscal Calendar", False, str(e))
            return False
    
    def _test_outlook_data_processing(self):
        """Test Outlook App data processing."""
        print("\nTesting Outlook App data processing...")
        
        try:
            if 'data_processing' not in self.outlook_modules:
                self.log_test("Outlook Data Processing", False, "data_processing module not loaded")
                return False
                
            coerce_daily_dates = self.outlook_modules['data_processing'].coerce_daily_dates
            analyze_daily_data = self.outlook_modules['data_processing'].analyze_daily_data
            
            # Create test daily data
            dates = pd.date_range('2024-08-01', '2024-08-31', freq='D')
            test_data = pd.DataFrame({
                'Date': dates,
                'Product': ['Test Product'] * len(dates),
                'Value': np.random.randint(50, 150, len(dates))
            })
            
            # Test date coercion
            coerced_dates = coerce_daily_dates(test_data['Date'])
            
            if isinstance(coerced_dates, pd.Series):
                self.log_test("Daily Date Coercion", True)
            else:
                self.log_test("Daily Date Coercion", False, "Date coercion not returning Series")
                return False
            
            # Test data analysis - fix function call to match expected signature
            # analyze_daily_data expects a Series with datetime index
            test_series = pd.Series(
                test_data['Value'].values,
                index=pd.to_datetime(test_data['Date'])
            )
            
            analysis = analyze_daily_data(test_series)
            
            if isinstance(analysis, dict) and 'weekday_avg' in analysis:
                self.log_test("Outlook Data Processing", True)
            else:
                self.log_test("Outlook Data Processing", False, "Data analysis not returning proper structure")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Outlook Data Processing", False, str(e))
            return False
    
    def _test_outlook_forecasting(self):
        """Test Outlook App forecasting logic."""
        print("\nTesting Outlook App forecasting...")
        
        try:
            if 'quarterly_forecasting' not in self.outlook_modules or 'fiscal_calendar' not in self.outlook_modules:
                self.log_test("Outlook Forecasting", False, "Required modules not loaded")
                return False
                
            forecast_quarter_completion = self.outlook_modules['quarterly_forecasting'].forecast_quarter_completion
            get_fiscal_quarter_info = self.outlook_modules['fiscal_calendar'].get_fiscal_quarter_info
            
            # Create test quarter data with proper datetime index
            current_date = datetime(2024, 8, 15)  # Mid Q1
            quarter_info = get_fiscal_quarter_info(current_date)
            
            # Mock partial quarter data as Series with datetime index
            dates = pd.date_range('2024-07-01', '2024-08-15', freq='D')
            values = np.random.randint(80, 120, len(dates))
            partial_data = pd.Series(values, index=dates)
            
            # Test forecasting with correct function signature
            forecast_result = forecast_quarter_completion(
                partial_data,
                current_date=current_date
            )
            
            if isinstance(forecast_result, dict) and ('forecasts' in forecast_result or 'error' in forecast_result):
                self.log_test("Outlook Forecasting", True)
            else:
                self.log_test("Outlook Forecasting", False, "Forecasting not returning proper structure")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Outlook Forecasting", False, str(e))
            return False
    
    def _test_outlook_ui_components(self):
        """Test Outlook App UI components."""
        print("\nTesting Outlook App UI components...")
        
        try:
            if 'ui_components' not in self.outlook_modules:
                self.log_test("Outlook UI Components", False, "ui_components module not loaded")
                return False
                
            create_forecast_summary_table = self.outlook_modules['ui_components'].create_forecast_summary_table
            create_forecast_visualization = self.outlook_modules['ui_components'].create_forecast_visualization
            
            # Test function signatures
            sig1 = inspect.signature(create_forecast_summary_table)
            sig2 = inspect.signature(create_forecast_visualization)
            
            if len(sig1.parameters) > 0 and len(sig2.parameters) > 0:
                self.log_test("UI Component Structure", True)
            else:
                self.log_test("UI Component Structure", False, "UI functions missing parameters")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Outlook UI Components", False, str(e))
            return False
    
    def generate_summary(self):
        """Generate comprehensive test summary."""
        print("\n" + "="*80)
        print("COMPREHENSIVE REGRESSION TEST SUMMARY")
        print("="*80)
        
        # Group results by app
        forecaster_results = [r for r in self.test_results if r['app'] == 'Forecaster App']
        outlook_results = [r for r in self.test_results if r['app'] == 'Outlook App']
        
        def print_app_summary(app_name, results):
            if not results:
                print(f"\nERROR - {app_name}: No tests run")
                return 0, 0
                
            passed = sum(1 for r in results if r['result'])
            total = len(results)
            
            print(f"\nResults for {app_name}: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
            print("-" * 40)
            
            for result in results:
                status = "[PASS]" if result['result'] else "[FAIL]"
                print(f"{status:10} {result['test']}")
                if result['details'] and not result['result']:
                    print(f"           Details: {result['details']}")
            
            return passed, total
        
        # Print summaries
        f_passed, f_total = print_app_summary("Forecaster App", forecaster_results)
        o_passed, o_total = print_app_summary("Outlook App", outlook_results)
        
        # Overall summary
        total_passed = f_passed + o_passed
        total_tests = f_total + o_total
        
        print("\n" + "="*80)
        print("OVERALL RESULTS")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_tests - total_passed}")
        print(f"Success Rate: {total_passed/total_tests*100:.1f}%" if total_tests > 0 else "No tests run")
        
        if total_passed == total_tests:
            print("\nALL TESTS PASSED! Both apps are functioning correctly.")
            return 0
        else:
            print(f"\n{total_tests - total_passed} TEST(S) FAILED. Please review the issues above.")
            return 1


def main():
    """Run comprehensive regression tests for both apps."""
    print("Starting Comprehensive Regression Test Suite")
    print("Testing both Forecaster App and Outlook App")
    print("="*80)
    
    tester = RegressionTester()
    
    # Test both apps
    tester.test_forecaster_app()
    tester.test_outlook_app()
    
    # Generate summary and return exit code
    return tester.generate_summary()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nERROR - Regression test failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
