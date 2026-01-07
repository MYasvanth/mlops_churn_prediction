#!/usr/bin/env python3
"""Simple test runner for CI/CD"""

import subprocess
import sys

def run_tests():
    """Run all tests"""
    try:
        # Run unit tests
        result = subprocess.run(['python', '-m', 'pytest', 'tests/', '-v'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
        print("All tests passed")
        return True
        
    except Exception as e:
        print(f"Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)