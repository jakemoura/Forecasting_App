#!/usr/bin/env python3
"""Debug version of comprehensive test to identify issues."""

import sys
import os
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

print("🔧 Debug Test Starting...")

# Add project paths to sys.path
project_root = Path(__file__).parent.parent
print(f"Project root: {project_root}")

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Forecaster App"))
sys.path.insert(0, str(project_root / "Quarter Outlook App"))

def safe_import(module_name):
    """Safely import a module without failing."""
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"Failed to import {module_name}: {e}")
        return None

print("Attempting basic imports...")

try:
    import importlib
    print("✅ importlib imported")
except Exception as e:
    print(f"❌ importlib failed: {e}")

try:
    import pandas as pd
    print("✅ pandas imported")
except Exception as e:
    print(f"❌ pandas failed: {e}")

try:
    import numpy as np
    print("✅ numpy imported")
except Exception as e:
    print(f"❌ numpy failed: {e}")

print("Creating test class...")

class DebugTest(unittest.TestCase):
    def test_basic(self):
        """Basic test."""
        print("Running basic test...")
        self.assertTrue(True)
        
    def test_pandas(self):
        """Test pandas works."""
        print("Testing pandas...")
        df = pd.DataFrame({'A': [1, 2, 3]})
        self.assertEqual(len(df), 3)

print("Setting up test runner...")

if __name__ == "__main__":
    print("Starting test execution...")
    unittest.main(verbosity=2)
