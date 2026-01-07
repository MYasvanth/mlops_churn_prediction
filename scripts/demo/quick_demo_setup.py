#!/usr/bin/env python3
"""
Quick Demo Setup Script
Prepares the churn prediction project for interview demonstration
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def install_dependencies():
    """Install required packages for demo"""
    packages = [
        "streamlit",
        "omegaconf", 
        "lightgbm",
        "evidently",
        "optuna-dashboard"
    ]
    
    print("📦 Installing demo dependencies...")
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package}")

def start_api_server():
    """Start FastAPI server"""
    print("🚀 Starting API server...")
    try:
        subprocess.Popen([
            sys.executable, 
            "scripts/monitoring/run_fastapi_server.py"
        ])
        time.sleep(3)
        
        # Test API health
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server running on http://localhost:8000")
            return True
    except Exception as e:
        print(f"❌ API server failed: {e}")
    return False

def start_streamlit():
    """Start Streamlit dashboard"""
    print("📊 Starting Streamlit dashboard...")
    try:
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "src/deployment/streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
        time.sleep(5)
        print("✅ Streamlit dashboard running on http://localhost:8501")
        return True
    except Exception as e:
        print(f"❌ Streamlit failed: {e}")
    return False

def check_models():
    """Check if models are available"""
    models_dir = Path("models/staging")
    if models_dir.exists():
        model_count = len(list(models_dir.glob("*")))
        print(f"✅ Found {model_count} models in staging")
        return True
    else:
        print("⚠️ No models found - run training first")
        return False

def check_data():
    """Check if data is available"""
    data_file = Path("data/processed/train.csv")
    if data_file.exists():
        print("✅ Training data available")
        return True
    else:
        print("⚠️ Training data not found")
        return False

def main():
    """Main demo setup"""
    print("🎯 MLOps Churn Prediction - Demo Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not check_data():
        print("❌ Please run data ingestion first")
        return
    
    if not check_models():
        print("❌ Please run model training first")
        return
    
    # Install dependencies
    install_dependencies()
    
    # Start services
    api_success = start_api_server()
    streamlit_success = start_streamlit()
    
    print("\n🎉 Demo Setup Complete!")
    print("=" * 50)
    
    if api_success:
        print("🔗 API Documentation: http://localhost:8000/docs")
        print("🔗 Health Check: http://localhost:8000/health")
    
    if streamlit_success:
        print("🔗 Dashboard: http://localhost:8501")
    
    print("\n📋 Demo Commands:")
    print("# Test API")
    print("curl http://localhost:8000/health")
    print("curl http://localhost:8000/models")
    
    print("\n# Run training")
    print("python src/pipelines/unified_training_pipeline.py")
    
    print("\n# Run monitoring")
    print("python scripts/monitoring/run_monitoring.py")
    
    print("\n🎬 Ready for demo! Follow DEMO_GUIDE.md")

if __name__ == "__main__":
    main()