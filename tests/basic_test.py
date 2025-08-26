#!/usr/bin/env python3
"""Very basic test to check if anything is working."""

print("Starting basic test...")

import sys
import unittest

print("Imports successful, creating test class...")

class BasicTest(unittest.TestCase):
    def test_basic(self):
        print("Basic test running...")
        self.assertTrue(True)

print("About to run tests...")

if __name__ == "__main__":
    print("In main block...")
    unittest.main(verbosity=2)
