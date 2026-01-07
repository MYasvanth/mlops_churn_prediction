import pytest
import subprocess
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_deploy_model_script():
    """Test model deployment script with correct path"""
    script_path = "scripts/deployment/deploy_model.py"
    result = subprocess.run(['python', script_path, '--help'], capture_output=True, text=True, cwd=project_root)
    # Check if script runs without errors (help command)
    assert result.returncode == 0

def test_run_fastapi_server_script():
    """Test FastAPI server startup script"""
    script_path = "scripts/monitoring/run_fastapi_server.py"
    # Test script exists and is executable (don't actually start server in tests)
    assert os.path.exists(os.path.join(project_root, script_path))

def test_start_complete_deployment_script():
    """Test complete deployment orchestration script"""
    script_path = "scripts/deployment/start_complete_deployment.py"
    assert os.path.exists(os.path.join(project_root, script_path))

def test_run_data_drift_monitor_script():
    """Test data drift monitoring script"""
    script_path = "scripts/monitoring/run_data_drift_monitor.py"
    result = subprocess.run(['python', script_path, '--help'], capture_output=True, text=True, cwd=project_root)
    # Check if script runs without errors (help command)
    assert result.returncode == 0

def test_run_monitoring_script():
    """Test general monitoring script"""
    script_path = "scripts/monitoring/run_monitoring.py"
    result = subprocess.run(['python', script_path, '--help'], capture_output=True, text=True, cwd=project_root)
    assert result.returncode == 0

def test_model_loading_script():
    """Test model loading verification script"""
    script_path = "scripts/testing/test_model_loading.py"
    if os.path.exists(os.path.join(project_root, script_path)):
        result = subprocess.run(['python', script_path], capture_output=True, text=True, cwd=project_root)
        assert result.returncode == 0

def test_deployment_verification_script():
    """Test deployment verification script"""
    script_path = "scripts/deployment/final_deployment_verification.py"
    if os.path.exists(os.path.join(project_root, script_path)):
        result = subprocess.run(['python', script_path], capture_output=True, text=True, cwd=project_root)
        assert result.returncode == 0
