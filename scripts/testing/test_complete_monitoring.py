#!/usr/bin/env python3
"""
Complete monitoring system test
Tests data drift, performance monitoring, and alert system integration
"""

import pandas as pd
import numpy as np
from src.monitoring.data_drift_monitor import DataDriftMonitor
from src.monitoring.model_performance_monitor import ModelPerformanceMonitor
from src.monitoring.alert_system import AlertManager, AlertType, AlertSeverity

def test_complete_monitoring():
    """Test the complete monitoring system"""
    print("🚀 Testing Complete Monitoring System")
    print("="*50)
    
    # 1. Test Data Drift Monitoring
    print("\n1. Testing Data Drift Monitoring...")
    drift_monitor = DataDriftMonitor()
    
    # Load reference data
    reference_data = pd.read_csv("data/processed/raw_data.csv")
    drift_monitor.set_reference_data(reference_data)
    
    # Create modified data to simulate drift
    current_data = reference_data.copy()
    np.random.seed(42)
    
    # Introduce more significant drift
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        if col in current_data.columns:
            current_data[col] = current_data[col] * (1 + np.random.normal(0.3, 0.1, len(current_data)))
    
    # Run drift detection
    drift_report = drift_monitor.detect_drift(current_data, report_name="complete_test_drift")
    
    print(f"   Drift Detected: {drift_report.drift_detected}")
    print(f"   Drift Score: {drift_report.drift_score:.3f}")
    print(f"   Drifted Columns: {len(drift_report.drifted_columns)}/{drift_report.total_columns}")
    print(f"   Alert Triggered: {drift_report.alert_triggered}")
    
    # 2. Test Performance Monitoring
    print("\n2. Testing Performance Monitoring...")
    performance_monitor = ModelPerformanceMonitor()
    
    # Create sample predictions
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])  # Some wrong predictions
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8, 0.6, 0.7, 0.3, 0.9])
    
    # Calculate metrics
    metrics = performance_monitor.calculate_metrics(y_true, y_pred, y_pred_proba)
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    print(f"   ROC AUC: {metrics['roc_auc']:.3f}")
    
    # 3. Test Alert System
    print("\n3. Testing Alert System...")
    alert_manager = AlertManager()
    
    # Create test alerts
    alert1 = alert_manager.create_alert(
        alert_type=AlertType.DATA_DRIFT,
        severity=AlertSeverity.HIGH,
        title="Test Data Drift Alert",
        message="This is a test alert for data drift detection",
        source="test_script"
    )
    
    alert2 = alert_manager.create_alert(
        alert_type=AlertType.MODEL_DEGRADATION,
        severity=AlertSeverity.MEDIUM,
        title="Test Performance Alert",
        message="This is a test alert for model performance degradation",
        source="test_script"
    )
    
    print(f"   Alert 1 Created: {alert1.id}")
    print(f"   Alert 2 Created: {alert2.id}")
    
    # Get alert statistics
    stats = alert_manager.get_alert_statistics()
    print(f"   Total Alerts: {stats['total_alerts']}")
    print(f"   Active Alerts: {stats['active_alerts']}")
    
    print("\n✅ Complete Monitoring System Test Completed Successfully!")
    return True

if __name__ == "__main__":
    test_complete_monitoring()
