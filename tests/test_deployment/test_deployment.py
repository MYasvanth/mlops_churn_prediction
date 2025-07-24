import pytest
import subprocess

def test_deploy_model_script():
    result = subprocess.run(['python', 'scripts/deploy_model.py'], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Model deployed" in result.stdout or "Deployment successful" in result.stdout or result.stdout != ""

def test_run_inference_script():
    result = subprocess.run(['python', 'scripts/run_inference.py'], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Inference" in result.stdout or result.stdout != ""

def test_run_ingestion_script():
    result = subprocess.run(['python', 'scripts/run_ingestion.py'], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Ingestion" in result.stdout or result.stdout != ""

def test_run_monitoring_script():
    result = subprocess.run(['python', 'scripts/run_monitoring.py'], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Monitoring" in result.stdout or result.stdout != ""

def test_run_training_script():
    result = subprocess.run(['python', 'scripts/run_training.py'], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Training" in result.stdout or result.stdout != ""
