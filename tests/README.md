# Forecasting Applications Test Suite

This directory contains comprehensive test suites for validating the functionality of both forecasting applications.

## Test Files

### 1. `simple_regression_test.py`
**Recommended for regular use**

A lightweight regression test that validates core functionality without complex dependencies:
- Project structure validation
- Data format validation  
- Basic forecasting logic
- Fiscal calendar calculations
- Spike detection logic
- Excel file handling
- Data processing pipeline

**Usage:**
```bash
cd tests
python simple_regression_test.py
```

### 2. `working_comprehensive_test.py`
**Complete test suite with full coverage**

A comprehensive test suite that validates:
- Project structure and module organization
- Data validation and processing logic
- Mathematical operations and forecasting algorithms
- Excel file operations
- Fiscal calendar calculations
- Spike detection functionality
- Memory usage and performance
- Error handling and edge cases
- WAPE (Weighted Absolute Percentage Error) calculations

**Usage:**
```bash
cd tests
python working_comprehensive_test.py
```

### 3. `run_tests.py`
Test runner utility that handles path setup and provides a clean interface for running the comprehensive test suite.

## Test Coverage

The test suites validate:

### Forecaster App
- ✅ Module imports (ui_config, data_validation, forecasting_pipeline, etc.)
- ✅ Data validation and preparation
- ✅ Model availability (Prophet, PMDARIMA, LightGBM)
- ✅ Business logic calculations
- ✅ Forecasting pipeline execution
- ✅ Excel file reading/writing

### Quarter Outlook App  
- ✅ Module imports (fiscal_calendar, data_processing, forecasting_models, etc.)
- ✅ Fiscal calendar calculations
- ✅ Daily data processing
- ✅ Spike detection functionality
- ✅ Quarterly forecasting
- ✅ Model evaluation and WAPE calculations

### Integration Tests
- ✅ Excel reading capabilities
- ✅ Error handling with edge cases
- ✅ Memory and performance with large datasets
- ✅ Utility functions and helpers

## Running Tests

### Quick Validation (Recommended)
For quick validation of core functionality:
```bash
python simple_regression_test.py
```

### Full Test Suite
For comprehensive testing:
```bash
python run_tests.py
```

### Development Workflow
1. Run simple regression test after any changes
2. Run comprehensive test suite before commits
3. Use specific tests to debug individual components

## Expected Output

### Successful Run
```
🧪 Simple Regression Test Suite for Forecasting Applications
=================================================================

✅ PASS: Project Structure
✅ PASS: Data Format Validation
✅ PASS: Basic Forecasting Logic
✅ PASS: Fiscal Calendar Logic
✅ PASS: Spike Detection Logic
✅ PASS: Excel File Handling
✅ PASS: Data Processing Pipeline

=================================================================
📊 Test Results Summary
=================================================================
Total tests: 7
Passed: 7
Failed: 0
Success rate: 100.0%

✅ All tests passed! Basic functionality is working correctly.
```

### Failed Test Example
```
❌ FAIL: Data Format Validation
   └─ Missing required columns

⚠️  Some tests failed. Please review the issues above.
```

## Troubleshooting

### Import Errors
If you see import errors:
1. Make sure you're running from the correct directory
2. Check that all required modules exist in their expected locations
3. Verify Python path setup

### Missing Dependencies
If tests fail due to missing packages:
```bash
pip install pandas numpy openpyxl
```

### Path Issues
Tests automatically handle path setup, but if you encounter issues:
1. Run from the `tests/` directory
2. Check that parent directories exist
3. Verify project structure matches expected layout

## Adding New Tests

### For Simple Tests
Add new test methods to the `SimpleRegressionTest` class in `simple_regression_test.py`:

```python
def test_new_functionality(self):
    """Test new functionality."""
    test_name = "New Functionality"
    
    try:
        # Your test logic here
        result = some_function()
        
        if not result:
            self.log_result(test_name, False, "Test failed")
            return
            
        self.log_result(test_name, True)
        
    except Exception as e:
        self.log_result(test_name, False, f"Error: {e}")
```

### For Comprehensive Tests
Add new test methods to the `ForecastingRegressionTests` class in `comprehensive_regression_test.py`:

```python
def test_new_feature(self):
    """Test new feature functionality."""
    try:
        # Import required modules
        from modules.new_module import new_function
        
        # Test logic
        result = new_function(test_data)
        self.assertIsInstance(result, expected_type)
        
    except Exception as e:
        self.fail(f"New feature test failed: {e}")
```

## Maintenance

- Run tests regularly during development
- Update tests when adding new features
- Keep test data realistic but simple
- Document any new test requirements
- Consider performance impact of comprehensive tests
