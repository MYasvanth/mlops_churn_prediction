#!/usr/bin/env python3
"""
Performance Monitoring Integration Test
Tests the model performance monitoring functionality
"""

import numpy as np
from src.monitoring.model_performance_monitor import ModelPerformanceMonitor

def test_performance_monitoring():
    """Test the performance monitoring functionality"""
    print("🧪 Testing Performance Monitoring Integration")
    print("="*50)
    
    # Initialize the performance monitor
    monitor = ModelPerformanceMonitor(config_path="configs/monitoring/monitoring_config.yaml")
    
    # Create test data
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1])  # Some errors
    y_pred_proba = np.array([
        [0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1],
        [0.2, 0.8], [0.4, 0.6], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9],
        [0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4], [0.9, 0.1],
        [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]
    ])
    
    print("1. Testing Metric Calculation...")
    # Test metric calculation
    metrics = monitor.calculate_metrics(y_true, y_pred, y_pred_proba)
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    print(f"   ROC AUC: {metrics['roc_auc']:.3f}")
    
    print("\n2. Testing Performance Logging...")
    # Test performance logging
    performance_metrics = monitor.log_performance(
        metrics, "test_model_v1", len(y_true)
    )
    print(f"   Logged model version: {performance_metrics.model_version}")
    print(f"   Data size: {performance_metrics.data_size}")
    
    print("\n3. Testing Performance Degradation Check...")
    # Test degradation check
    alerts = monitor.check_performance_degradation(performance_metrics)
    print(f"   Alerts generated: {len(alerts)}")
    for alert in alerts:
        print(f"   - {alert.metric_name}: {alert.current_value:.3f} vs threshold {alert.threshold_value:.3f}")
        print(f"     Message: {alert.message}")
    
    print("\n4. Testing Complete Monitoring Workflow...")
    # Test complete workflow
    report = monitor.monitor_model_performance(
        y_true, y_pred, y_pred_proba, "test_model_v1"
    )
    print(f"   Monitoring completed for model: {report['model_version']}")
    print(f"   Total alerts in report: {len(report['alerts'])}")
    
    print("\n5. Testing Performance Trends...")
    # Test trend analysis
    trends = monitor.get_performance_trends(window_size=5)
    if 'error' not in trends:
        print(f"   Trend analysis completed for {len(monitor.performance_history)} data points")
        for metric, trend_data in trends['metrics_trends'].items():
            print(f"   {metric}: {trend_data['trend']} ({trend_data['change']:+.3f})")
    else:
        print(f"   Trend analysis: {trends['error']}")
    
    print("\n6. Testing Alert Summary...")
    # Test alert summary
    alert_summary = monitor.get_alert_summary(hours_back=24)
    print(f"   Total alerts in last 24h: {alert_summary['total_alerts']}")
    print(f"   Severity breakdown: {alert_summary['severity_breakdown']}")
    
    print("\n✅ Performance Monitoring Integration Test Completed Successfully!")
    return True

if __name__ == "__main__":
    test_performance_monitoring()
