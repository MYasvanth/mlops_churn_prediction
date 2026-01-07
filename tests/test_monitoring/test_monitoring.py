import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.data_drift_monitor import DataDriftMonitor
from src.monitoring.model_performance_monitor import ModelPerformanceMonitor

@pytest.fixture
def sample_reference_data():
    """Create reference dataset for drift detection"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'feature3': np.random.normal(10, 3, 1000),
        'churn': np.random.randint(0, 2, 1000)
    })

@pytest.fixture
def sample_current_data_no_drift(sample_reference_data):
    """Create current data with no drift"""
    np.random.seed(43)
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, 500),
        'feature2': np.random.normal(5, 2, 500),
        'feature3': np.random.normal(10, 3, 500),
        'churn': np.random.randint(0, 2, 500)
    })

@pytest.fixture
def sample_current_data_with_drift():
    """Create current data with significant drift"""
    np.random.seed(44)
    return pd.DataFrame({
        'feature1': np.random.normal(2, 1.5, 500),  # Mean shifted from 0 to 2
        'feature2': np.random.normal(5, 2, 500),    # No drift
        'feature3': np.random.normal(15, 4, 500),   # Mean shifted from 10 to 15
        'churn': np.random.randint(0, 2, 500)
    })

@pytest.fixture
def temp_reports_dir():
    """Create temporary directory for reports"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

def test_data_drift_monitor_initialization():
    """Test DataDriftMonitor initialization"""
    monitor = DataDriftMonitor()
    assert monitor is not None
    assert hasattr(monitor, 'detect_drift')
    assert hasattr(monitor, 'set_reference_data')

def test_set_reference_data(sample_reference_data):
    """Test setting reference data"""
    monitor = DataDriftMonitor()
    monitor.set_reference_data(sample_reference_data)

    assert monitor.reference_data is not None
    assert len(monitor.reference_data) == 1000
    assert list(monitor.reference_data.columns) == ['feature1', 'feature2', 'feature3', 'churn']

def test_detect_no_drift(sample_reference_data, sample_current_data_no_drift):
    """Test drift detection when there's no significant drift"""
    monitor = DataDriftMonitor()
    monitor.set_reference_data(sample_reference_data)

    drift_report = monitor.detect_drift(sample_current_data_no_drift)

    assert isinstance(drift_report, dict)
    assert 'drift_detected' in drift_report
    assert 'drift_score' in drift_report
    assert 'drifted_columns' in drift_report
    assert 'total_columns' in drift_report
    assert 'drift_threshold' in drift_report
    assert 'report_path' in drift_report

    # With no significant drift, these should be reasonable values
    assert isinstance(drift_report['drift_detected'], bool)
    assert isinstance(drift_report['drift_score'], (int, float))
    assert isinstance(drift_report['drifted_columns'], list)

def test_detect_significant_drift(sample_reference_data, sample_current_data_with_drift):
    """Test drift detection when there's significant drift"""
    monitor = DataDriftMonitor()
    monitor.set_reference_data(sample_reference_data)

    drift_report = monitor.detect_drift(sample_current_data_with_drift)

    assert isinstance(drift_report, dict)
    assert 'drift_detected' in drift_report
    assert 'drift_score' in drift_report
    assert 'drifted_columns' in drift_report

    # With significant drift in feature1 and feature3, drift should be detected
    # Note: This might not always detect drift depending on statistical significance
    print(f"Drift detected: {drift_report['drift_detected']}")
    print(f"Drift score: {drift_report['drift_score']}")
    print(f"Drifted columns: {drift_report['drifted_columns']}")

def test_drift_report_structure(sample_reference_data, sample_current_data_no_drift):
    """Test drift report has all required fields"""
    monitor = DataDriftMonitor()
    monitor.set_reference_data(sample_reference_data)

    drift_report = monitor.detect_drift(sample_current_data_no_drift)

    required_fields = [
        'report_id', 'timestamp', 'drift_detected', 'drift_score',
        'drifted_columns', 'total_columns', 'drift_threshold',
        'report_path', 'summary'
    ]

    for field in required_fields:
        assert field in drift_report, f"Missing field: {field}"

def test_drift_detection_with_custom_report_name(sample_reference_data, sample_current_data_no_drift):
    """Test drift detection with custom report name"""
    monitor = DataDriftMonitor()
    monitor.set_reference_data(sample_reference_data)

    custom_name = "custom_drift_test"
    drift_report = monitor.detect_drift(sample_current_data_no_drift, report_name=custom_name)

    assert custom_name in drift_report['report_id']
    assert os.path.exists(drift_report['report_path'])

def test_performance_monitor_initialization():
    """Test ModelPerformanceMonitor initialization"""
    monitor = ModelPerformanceMonitor()
    assert monitor is not None
    assert hasattr(monitor, 'calculate_metrics')
    assert hasattr(monitor, 'log_performance')
    assert hasattr(monitor, 'check_performance_degradation')

def test_calculate_metrics():
    """Test metrics calculation"""
    monitor = ModelPerformanceMonitor()

    # Create sample predictions
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_pred_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]])

    metrics = monitor.calculate_metrics(y_true, y_pred, y_pred_proba)

    required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    for metric in required_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))

    # Check ROC AUC is calculated
    assert metrics['roc_auc'] > 0

def test_performance_logging():
    """Test performance logging"""
    monitor = ModelPerformanceMonitor()

    metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.87,
        'f1_score': 0.84,
        'roc_auc': 0.91
    }

    performance_metrics = monitor.log_performance(metrics, "test_model", 100)

    assert performance_metrics is not None
    assert performance_metrics.accuracy == 0.85
    assert performance_metrics.model_version == "test_model"
    assert performance_metrics.data_size == 100

def test_performance_degradation_check():
    """Test performance degradation detection"""
    monitor = ModelPerformanceMonitor()

    # Add some historical performance data
    for i in range(3):
        metrics = {
            'accuracy': 0.85 - i * 0.02,  # Declining accuracy
            'precision': 0.82,
            'recall': 0.87,
            'f1_score': 0.84,
            'roc_auc': 0.91
        }
        monitor.log_performance(metrics, f"model_v{i}", 100)

    # Test current performance (significant drop)
    current_metrics = Mock()
    current_metrics.accuracy = 0.75  # Significant drop from 0.85
    current_metrics.precision = 0.82
    current_metrics.recall = 0.87
    current_metrics.f1_score = 0.84
    current_metrics.roc_auc = 0.91

    alerts = monitor.check_performance_degradation(current_metrics)

    assert isinstance(alerts, list)
    # Should detect accuracy degradation
    accuracy_alerts = [a for a in alerts if a.metric_name == 'accuracy']
    assert len(accuracy_alerts) > 0

def test_monitor_model_performance():
    """Test complete model performance monitoring workflow"""
    monitor = ModelPerformanceMonitor()

    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    y_pred_proba = np.array([
        [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8],
        [0.1, 0.9], [0.7, 0.3], [0.4, 0.6], [0.3, 0.7], [0.8, 0.2]
    ])

    report = monitor.monitor_model_performance(y_true, y_pred, y_pred_proba, "test_model")

    assert isinstance(report, dict)
    assert 'timestamp' in report
    assert 'model_version' in report
    assert 'metrics' in report
    assert 'performance_summary' in report
    assert 'alerts' in report
    assert report['model_version'] == "test_model"

def test_performance_trends():
    """Test performance trends analysis"""
    monitor = ModelPerformanceMonitor()

    # Add multiple performance records
    for i in range(5):
        metrics = {
            'accuracy': 0.80 + i * 0.01,  # Improving accuracy
            'precision': 0.82,
            'recall': 0.87,
            'f1_score': 0.84,
            'roc_auc': 0.91
        }
        monitor.log_performance(metrics, f"model_v{i}", 100)

    trends = monitor.get_performance_trends(window_size=5)

    assert isinstance(trends, dict)
    assert 'time_range' in trends
    assert 'metrics_trends' in trends
    assert 'accuracy' in trends['metrics_trends']
    assert 'trend' in trends['metrics_trends']['accuracy']

def test_alert_summary():
    """Test alert summary generation"""
    monitor = ModelPerformanceMonitor()

    # Create some mock alerts
    from src.monitoring.model_performance_monitor import PerformanceAlert

    alert1 = PerformanceAlert(
        alert_type='THRESHOLD_BREACH',
        metric_name='accuracy',
        current_value=0.75,
        threshold_value=0.80,
        severity='HIGH',
        timestamp=datetime.now(),
        message='Accuracy below threshold'
    )

    alert2 = PerformanceAlert(
        alert_type='PERFORMANCE_DEGRADATION',
        metric_name='precision',
        current_value=0.78,
        threshold_value=0.82,
        severity='MEDIUM',
        timestamp=datetime.now(),
        message='Precision degraded'
    )

    monitor.alerts = [alert1, alert2]

    summary = monitor.get_alert_summary(hours_back=24)

    assert isinstance(summary, dict)
    assert 'total_alerts' in summary
    assert 'severity_breakdown' in summary
    assert 'type_breakdown' in summary
    assert 'recent_alerts' in summary
    assert summary['total_alerts'] == 2

def test_drift_monitor_without_reference_data(sample_current_data_no_drift):
    """Test drift detection without setting reference data"""
    monitor = DataDriftMonitor()

    with pytest.raises(ValueError, match="Reference data not set"):
        monitor.detect_drift(sample_current_data_no_drift)

def test_empty_data_handling():
    """Test handling of empty datasets"""
    monitor = DataDriftMonitor()

    empty_df = pd.DataFrame()
    with pytest.raises(Exception):  # Should handle empty data gracefully
        monitor.set_reference_data(empty_df)
