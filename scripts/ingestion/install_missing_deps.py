#!/usr/bin/env python3
"""
Script to install missing dependencies for the MLOps churn prediction project
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {cmd}")
            return True
        else:
            print(f"❌ {cmd} - Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {cmd} - Exception: {str(e)}")
        return False

def main():
    """Install missing dependencies"""
    print("🔧 Installing Missing Dependencies")
    print("=" * 50)
    
    # List of dependencies to install
    dependencies = [
        "omegaconf",           # Missing from config loader
        "lightgbm",            # Missing for LightGBM models
        "evidently",           # For monitoring
        "streamlit",           # For dashboard
        "optuna-dashboard",    # For hyperparameter optimization UI
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for dep in dependencies:
        print(f"📦 Installing {dep}...")
        if run_command(f"pip install {dep}"):
            success_count += 1
    
    print("\n" + "=" * 50)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 50)
    print(f"Successfully installed: {success_count}/{total_count} packages")
    
    if success_count == total_count:
        print("🎉 All dependencies installed successfully!")
        print("\n📋 Next steps:")
        print("1. Test model loading: python scripts/test_model_loading.py")
        print("2. Test complete deployment: python scripts/test_complete_deployment.py")
        print("3. Start all services: python scripts/start_complete_deployment.py")
    else:
        print("⚠️  Some dependencies failed to install")
        print("You may need to install them manually:")
        for dep in dependencies:
            print(f"   pip install {dep}")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
