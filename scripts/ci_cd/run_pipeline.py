#!/usr/bin/env python3
"""Manual CI/CD Pipeline Trigger"""

import subprocess
import sys
import os

def run_pipeline_step(step_name, command):
    """Run a pipeline step and return success status"""
    print(f"\n=== {step_name} ===")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[PASS] {step_name} completed successfully")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"[FAIL] {step_name} failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] {step_name} execution error: {e}")
        return False

def main():
    """Run complete CI/CD pipeline manually"""
    print("Starting Manual CI/CD Pipeline...")
    
    steps = [
        ("Quality Check", "python scripts/ci_cd/quality_check.py"),
        ("Smoke Test", "python scripts/ci_cd/smoke_test.py"),
        ("Model Retraining", "python scripts/ci_cd/retrain_model.py")
    ]
    
    passed = 0
    total = len(steps)
    
    for step_name, command in steps:
        if run_pipeline_step(step_name, command):
            passed += 1
    
    print(f"\n=== PIPELINE SUMMARY ===")
    print(f"Steps passed: {passed}/{total}")
    
    if passed == total:
        print("SUCCESS: All pipeline steps completed!")
        return True
    else:
        print("FAILURE: Some pipeline steps failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)