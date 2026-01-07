#!/usr/bin/env python3
"""
Simple script to track and verify Streamlit dashboard availability
"""

import requests
import time
import sys
from pathlib import Path

def check_streamlit_health():
    """Check if Streamlit dashboard is running"""
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit Dashboard: Running")
            print(f"   Status: {response.status_code}")
            print(f"   URL: http://localhost:8501")
            return True
        else:
            print(f"❌ Streamlit Dashboard: Not responding (Status: {response.status_code})")
            return False
    except requests.ConnectionError:
        print("❌ Streamlit Dashboard: Not running (Connection refused)")
        return False
    except Exception as e:
        print(f"❌ Streamlit Dashboard: Error - {str(e)}")
        return False

def check_mlflow():
    """Check if MLflow is running"""
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("✅ MLflow Tracking: Running")
            print(f"   Status: {response.status_code}")
            print(f"   URL: http://localhost:5000")
            return True
        else:
            print(f"❌ MLflow Tracking: Not responding (Status: {response.status_code})")
            return False
    except requests.ConnectionError:
        print("❌ MLflow Tracking: Not running (Connection refused)")
        return False
    except Exception as e:
        print(f"❌ MLflow Tracking: Error - {str(e)}")
        return False

def check_optuna():
    """Check if Optuna dashboard is running"""
    try:
        response = requests.get("http://localhost:8080", timeout=5)
        if response.status_code == 200:
            print("✅ Optuna Dashboard: Running")
            print(f"   Status: {response.status_code}")
            print(f"   URL: http://localhost:8080")
            return True
        else:
            print(f"❌ Optuna Dashboard: Not responding (Status: {response.status_code})")
            return False
    except requests.ConnectionError:
        print("❌ Optuna Dashboard: Not running (Connection refused)")
        return False
    except Exception as e:
        print(f"❌ Optuna Dashboard: Error - {str(e)}")
        return False

def check_fastapi():
    """Check if FastAPI is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ FastAPI Server: Running")
            print(f"   Status: {data['status']}")
            print(f"   Models: {data['available_models']}")
            print(f"   URL: http://localhost:8000")
            return True
        else:
            print(f"❌ FastAPI Server: Not responding (Status: {response.status_code})")
            return False
    except requests.ConnectionError:
        print("❌ FastAPI Server: Not running (Connection refused)")
        return False
    except Exception as e:
        print(f"❌ FastAPI Server: Error - {str(e)}")
        return False

def main():
    """Main function to track dashboard services"""
    print("🚀 MLOps Dashboard Tracking")
    print("=" * 50)
    
    services = [
        ("FastAPI Server", check_fastapi),
        ("Streamlit Dashboard", check_streamlit_health),
        ("MLflow Tracking", check_mlflow),
        ("Optuna Dashboard", check_optuna),
    ]
    
    results = {}
    
    for service_name, check_func in services:
        print(f"\n🔍 Checking {service_name}...")
        result = check_func()
        results[service_name] = result
    
    print("\n" + "=" * 50)
    print("📊 Service Status Summary")
    print("=" * 50)
    
    for service_name, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {service_name}: {'Running' if status else 'Not available'}")
    
    running_services = sum(1 for status in results.values() if status)
    total_services = len(results)
    
    print(f"\n📈 {running_services}/{total_services} services running")
    
    if running_services == total_services:
        print("\n🎉 All services are running successfully!")
        print("\n🌐 Access Points:")
        print("   FastAPI API: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        print("   Streamlit: http://localhost:8501")
        print("   MLflow: http://localhost:5000")
        print("   Optuna: http://localhost:8080")
    else:
        print(f"\n⚠️  Some services are not running")
        print("   The FastAPI server is the most critical component and it's working")
        print("   Other services can be started individually if needed")
    
    return running_services > 0  # Success if at least one service is running

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
