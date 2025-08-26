#!/usr/bin/env python3
"""
Test Runner for Forecasting Applications

This script runs the comprehensive regression test suite from the tests directory.
It handles path setup and provides a clean interface for running tests.

Usage:
    python run_tests.py [--verbose] [--specific-test TestName]

Author: Jake Moura
"""

import sys
import os
import argparse
from pathlib import Path

def setup_paths():
    """Set up Python paths for importing modules."""
    # Get the project root directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Add necessary paths to sys.path
    forecaster_path = project_root / "Forecaster App"
    outlook_path = project_root / "Quarter Outlook App"
    tests_path = project_root / "tests"
    
    paths_to_add = [str(project_root), str(forecaster_path), str(outlook_path), str(tests_path)]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print(f"📂 Project root: {project_root}")
    print(f"📂 Added paths: {paths_to_add}")

def run_tests(verbose=False, specific_test=None):
    """Run the comprehensive regression tests."""
    setup_paths()
    
    try:
        # Import the test module
        from working_comprehensive_test import run_working_comprehensive_tests, WorkingRegressionTests
        import unittest
        
        if specific_test:
            # Run specific test
            suite = unittest.TestSuite()
            suite.addTest(WorkingRegressionTests(specific_test))
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            result = runner.run(suite)
        else:
            # Run all tests
            success = run_working_comprehensive_tests()
            return success
            
    except ImportError as e:
        print(f"❌ Failed to import test modules: {e}")
        print("Make sure you're running this from the project root or tests directory.")
        return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run forecasting application tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--specific-test", "-t", help="Run a specific test method")
    parser.add_argument("--list-tests", "-l", action="store_true", help="List available tests")
    
    args = parser.parse_args()
    
    if args.list_tests:
        setup_paths()
        try:
            from working_comprehensive_test import WorkingRegressionTests
            import unittest
            
            print("📋 Available tests:")
            test_methods = [method for method in dir(WorkingRegressionTests) 
                           if method.startswith('test_')]
            for test_method in test_methods:
                print(f"  - {test_method}")
        except Exception as e:
            print(f"❌ Failed to list tests: {e}")
        return
    
    print("🧪 Forecasting Applications Test Runner")
    print("=" * 50)
    
    success = run_tests(verbose=args.verbose, specific_test=args.specific_test)
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
