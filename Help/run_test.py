#!/usr/bin/env python3
"""
Simple test runner for the comprehensive regression test
"""

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    try:
        from comprehensive_regression_test import RegressionTester
        
        print("Starting Comprehensive Regression Test Suite")
        print("Testing both Forecaster App and Outlook App")
        print("="*80)
        
        tester = RegressionTester()
        
        # Test Forecaster App
        print("\nTesting Forecaster App...")
        tester.test_forecaster_app()
        
        # Test Outlook App  
        print("\nTesting Outlook App...")
        tester.test_outlook_app()
        
        # Generate summary
        print("\nGenerating summary...")
        tester.generate_summary()
        
        print("\nTest suite completed successfully!")
        
    except Exception as e:
        print(f"ERROR: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
