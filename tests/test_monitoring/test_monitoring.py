import pytest
from src.monitoring.data_drift_monitor import DataDriftMonitor
from src.monitoring.alert_system import AlertSystem
import pandas as pd

@pytest.fixture
def sample_data():
    # Create sample dataframes for drift detection
    df_ref = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })
    df_new = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 6],  # slight drift in feature1
        'feature2': [10, 20, 30, 40, 50]
    })
    return df_ref, df_new

def test_data_drift_detection(sample_data):
    df_ref, df_new = sample_data
    monitor = DataDriftMonitor()
    drift_report = monitor.detect_drift(df_ref, df_new)
    assert isinstance(drift_report, dict)
    assert 'feature1' in drift_report

def test_alert_system_send_email(monkeypatch):
    alert_system = AlertSystem()
    # Mock the send_email method to avoid actual email sending
    monkeypatch.setattr(alert_system, 'send_email', lambda subject, body: True)
    result = alert_system.send_email("Test Subject", "Test Body")
    assert result is True
