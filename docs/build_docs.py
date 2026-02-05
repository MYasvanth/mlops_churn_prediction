#!/usr/bin/env python3
"""
Documentation build script for MLOps Churn Prediction project
"""

import subprocess
import sys
from pathlib import Path

def build_docs():
    """Build Sphinx documentation"""
    docs_dir = Path(__file__).parent
    
    print("Building MLOps Documentation...")
    
    # Install documentation dependencies
    print("Installing documentation dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], cwd=docs_dir, check=True)
    
    # Build HTML documentation
    print("Building HTML documentation...")
    result = subprocess.run([
        "sphinx-build", "-b", "html", ".", "_build/html"
    ], cwd=docs_dir)
    
    if result.returncode == 0:
        print("Documentation built successfully!")
        print(f"Open: {docs_dir / '_build' / 'html' / 'index.html'}")
        return True
    else:
        print("Documentation build failed!")
        return False

if __name__ == "__main__":
    success = build_docs()
    sys.exit(0 if success else 1)