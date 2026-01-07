#!/usr/bin/env python3
"""
Complete Deployment Script
Starts all components of the MLOps churn prediction system
"""

import os
import sys
import time
import subprocess
import threading
import signal
import logging
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)

class DeploymentManager:
    """Manager for deploying all system components"""
    
    def __init__(self):
        self.processes = []
        self.running = False
        
    def start_mlflow(self):
        """Start MLflow tracking server"""
        try:
            cmd = ["mlflow", "ui", "--port", "5000", "--host", "0.0.0.0"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(process)
            logger.info("✅ MLflow server started on http://localhost:5000")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start MLflow: {str(e)}")
            return False
    
    def start_fastapi(self):
        """Start FastAPI model server"""
        try:
            cmd = ["python", "scripts/run_fastapi_server.py"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(process)
            logger.info("✅ FastAPI server started on http://localhost:8000")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start FastAPI server: {str(e)}")
            return False
    
    def start_streamlit(self):
        """Start Streamlit dashboard"""
        try:
            cmd = [
                "streamlit", "run", 
                "src/deployment/streamlit_app.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0"
            ]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(process)
            logger.info("✅ Streamlit dashboard started on http://localhost:8501")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start Streamlit: {str(e)}")
            return False
    
    def start_optuna_dashboard(self):
        """Start Optuna dashboard with proper database initialization"""
        try:
            # Ensure the Optuna database is properly initialized
            import optuna
            import os
            
            if not os.path.exists("optuna_studies.db"):
                logger.info("📝 Creating Optuna database...")
                # Create a test study to initialize the database
                study = optuna.create_study(
                    storage='sqlite:///optuna_studies.db',
                    study_name='churn_prediction_study'
                )
                # Run a simple optimization to populate the database
                def objective(trial):
                    x = trial.suggest_float('learning_rate', 0.01, 0.3)
                    return (x - 0.1) ** 2
                study.optimize(objective, n_trials=3)
                logger.info("✅ Optuna database initialized")
            
            cmd = [
                "optuna-dashboard",
                "--port", "8080",
                "--host", "0.0.0.0",
                "sqlite:///optuna_studies.db"
            ]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(process)
            logger.info("✅ Optuna dashboard started on http://localhost:8080")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start Optuna dashboard: {str(e)}")
            return False
    
    def start_monitoring(self):
        """Start monitoring service"""
        try:
            cmd = ["python", "scripts/run_monitoring.py", "--mode", "continuous", "--interval", "30"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(process)
            logger.info("✅ Monitoring service started")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring: {str(e)}")
            return False
    
    def monitor_processes(self):
        """Monitor running processes and restart if needed"""
        while self.running:
            for i, process in enumerate(self.processes):
                if process.poll() is not None:  # Process has terminated
                    logger.warning(f"Process {i} terminated with return code {process.returncode}")
                    # TODO: Implement restart logic if needed
            time.sleep(5)
    
    def start_all_services(self, services):
        """Start all specified services with proper delays"""
        self.running = True
        
        # Start services based on selection
        service_starters = {
            'mlflow': self.start_mlflow,
            'fastapi': self.start_fastapi,
            'streamlit': self.start_streamlit,
            'optuna': self.start_optuna_dashboard,
            'monitoring': self.start_monitoring
        }
        
        started_services = []
        
        # Start services in sequence with delays
        for service in services:
            if service in service_starters:
                logger.info(f"Starting {service} service...")
                if service_starters[service]():
                    started_services.append(service)
                    # Add delay between service startups
                    if service != services[-1]:  # Don't sleep after last service
                        time.sleep(3)  # 3-second delay between services
            else:
                logger.warning(f"Unknown service: {service}")
        
        # Start process monitor in background
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        # Wait a bit for services to fully initialize
        time.sleep(5)
        
        return started_services
    
    def stop_all_services(self):
        """Stop all running services"""
        self.running = False
        logger.info("🛑 Stopping all services...")
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                logger.error(f"Error stopping process: {str(e)}")
        
        self.processes.clear()
        logger.info("✅ All services stopped")
    
    def print_status(self):
        """Print deployment status"""
        print("\n" + "="*60)
        print("🚀 MLOps Churn Prediction System - Deployment Status")
        print("="*60)
        
        urls = [
            ("MLflow Tracking", "http://localhost:5000"),
            ("FastAPI Server", "http://localhost:8000"),
            ("Streamlit Dashboard", "http://localhost:8501"),
            ("Optuna Dashboard", "http://localhost:8080"),
            ("API Documentation", "http://localhost:8000/docs")
        ]
        
        for name, url in urls:
            print(f"📍 {name}: {url}")
        
        print("\n📊 Monitoring:")
        print("   - Data drift detection: Enabled")
        print("   - Performance monitoring: Enabled")
        print("   - Alert system: Enabled")
        
        print("\n⚡ Quick Commands:")
        print("   Monitor logs:      tail -f logs/monitoring.log")
        print("   Run single test:   python scripts/run_monitoring.py")
        print("   Stop services:     Ctrl+C")
        
        print("="*60)
        print("Press Ctrl+C to stop all services")
        print("="*60)

def signal_handler(sig, frame):
    """Handle interrupt signal"""
    print("\n🛑 Received interrupt signal, shutting down...")
    deployment.stop_all_services()
    sys.exit(0)

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy MLOps churn prediction system")
    parser.add_argument('--services', nargs='+', 
                       default=['mlflow', 'fastapi', 'streamlit', 'optuna', 'monitoring'],
                       help='Services to start (mlflow, fastapi, streamlit, optuna, monitoring)')
    parser.add_argument('--setup-only', action='store_true',
                       help='Only run setup, don\'t start services')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Don\'t start monitoring service')
    
    args = parser.parse_args()
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    global deployment
    deployment = DeploymentManager()
    
    try:
        # Run monitoring setup
        logger.info("🔧 Running monitoring setup...")
        setup_result = subprocess.run(
            ["python", "scripts/setup_monitoring.py"],
            capture_output=True,
            text=True
        )
        
        if setup_result.returncode == 0:
            logger.info("✅ Monitoring setup completed")
        else:
            logger.error(f"❌ Monitoring setup failed: {setup_result.stderr}")
        
        if args.setup_only:
            logger.info("Setup completed. Exiting.")
            return
        
        # Filter services if needed
        services_to_start = args.services
        if args.no_monitoring and 'monitoring' in services_to_start:
            services_to_start.remove('monitoring')
        
        # Start services
        logger.info("🚀 Starting services...")
        started_services = deployment.start_all_services(services_to_start)
        
        if not started_services:
            logger.error("❌ No services started successfully")
            return
        
        # Print status
        deployment.print_status()
        
        # Keep main thread alive
        while deployment.running:
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        deployment.stop_all_services()
        sys.exit(1)

if __name__ == "__main__":
    main()
