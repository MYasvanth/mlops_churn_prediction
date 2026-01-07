#!/usr/bin/env python3
"""
Containerized Execution Script
Runs the complete MLOps workflow in a containerized environment
"""

import os
import subprocess
import sys
from pathlib import Path

def build_docker_image():
    """Build the Docker image for the MLOps project"""
    try:
        print("🚀 Building Docker image...")
        result = subprocess.run([
            "docker", "build", 
            "-t", "mlops-churn-prediction",
            "-f", "deployment/docker/Dockerfile",
            "."
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Docker build failed: {result.stderr}")
            return False
        
        print("✅ Docker image built successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error building Docker image: {str(e)}")
        return False

def run_docker_compose():
    """Run the complete deployment using Docker Compose"""
    try:
        print("🚀 Starting services with Docker Compose...")
        result = subprocess.run([
            "docker-compose", 
            "-f", "deployment/docker/docker-compose.yml",
            "up", "-d"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Docker Compose failed: {result.stderr}")
            return False
        
        print("✅ Docker Compose services started successfully")
        print("\n📊 Services running:")
        print("   - MLflow: http://localhost:5000")
        print("   - FastAPI: http://localhost:8000")
        print("   - Streamlit: http://localhost:8501")
        print("   - Optuna: http://localhost:8080")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running Docker Compose: {str(e)}")
        return False

def run_containerized_training():
    """Run training in a Docker container"""
    try:
        print("🚀 Running containerized training...")
        result = subprocess.run([
            "docker", "run", 
            "-v", f"{os.getcwd()}/data:/app/data",
            "-v", f"{os.getcwd()}/models:/app/models",
            "-v", f"{os.getcwd()}/mlartifacts:/app/mlartifacts",
            "mlops-churn-prediction",
            "python", "scripts/run_training.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Containerized training failed: {result.stderr}")
            return False
        
        print("✅ Containerized training completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error running containerized training: {str(e)}")
        return False

def run_containerized_pipeline():
    """Run the complete ZenML pipeline in a container"""
    try:
        print("🚀 Running containerized ZenML pipeline...")
        result = subprocess.run([
            "docker", "run", 
            "-v", f"{os.getcwd()}/data:/app/data",
            "-v", f"{os.getcwd()}/models:/app/models",
            "-v", f"{os.getcwd()}/mlartifacts:/app/mlartifacts",
            "-v", f"{os.getcwd()}/zenml_pipelines:/app/zenml_pipelines",
            "mlops-churn-prediction",
            "python", "run_churn_pipeline.py",
            "--model-type", "xgboost",
            "--deploy-to-production"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Containerized pipeline failed: {result.stderr}")
            return False
        
        print("✅ Containerized ZenML pipeline completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error running containerized pipeline: {str(e)}")
        return False

def main():
    """Main execution function"""
    print("=" * 60)
    print("CONTAINERIZED EXECUTION - MLOPS CHURN PREDICTION")
    print("=" * 60)
    
    # Build Docker image
    if not build_docker_image():
        sys.exit(1)
    
    # Choose execution mode
    print("\n🔧 Choose execution mode:")
    print("1. Full Docker Compose deployment")
    print("2. Containerized training only")
    print("3. Containerized ZenML pipeline")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        success = run_docker_compose()
    elif choice == "2":
        success = run_containerized_training()
    elif choice == "3":
        success = run_containerized_pipeline()
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    if success:
        print("\n🎉 Containerized execution completed successfully!")
    else:
        print("\n💥 Containerized execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
