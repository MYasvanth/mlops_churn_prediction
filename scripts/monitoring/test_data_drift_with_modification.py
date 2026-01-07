import pandas as pd
import numpy as np
from src.monitoring.data_drift_monitor import DataDriftMonitor

def test_data_drift_with_modification():
    """Test data drift detection with modified data to simulate actual drift"""
    
    # Initialize the DataDriftMonitor
    monitor = DataDriftMonitor(config_path="configs/monitoring/monitoring_config.yaml")
    
    # Load reference data
    reference_data = pd.read_csv("data/processed/raw_data.csv")
    monitor.set_reference_data(reference_data)
    
    # Create modified current data to simulate drift
    current_data = reference_data.copy()
    
    # Simulate drift by modifying some columns
    np.random.seed(42)  # For reproducible results
    
    # 1. Modify numerical columns (introduce drift)
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        if col in current_data.columns:
            # Add some noise and shift the distribution
            current_data[col] = current_data[col] * (1 + np.random.normal(0.1, 0.05, len(current_data)))
    
    # 2. Modify categorical columns (change some categories)
    categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']
    for col in categorical_cols:
        if col in current_data.columns:
            # Randomly change some categories
            mask = np.random.random(len(current_data)) < 0.2  # Change 20% of values
            unique_vals = current_data[col].unique()
            if len(unique_vals) > 1:
                current_data.loc[mask, col] = np.random.choice(unique_vals, size=mask.sum())
    
    print(f"Reference data shape: {reference_data.shape}")
    print(f"Modified current data shape: {current_data.shape}")
    
    # Run drift detection
    drift_report = monitor.detect_drift(current_data, report_name="simulated_drift_test")
    
    # Print the results
    print("\n" + "="*50)
    print("DATA DRIFT DETECTION RESULTS")
    print("="*50)
    print(f"Drift Detected: {drift_report.drift_detected}")
    print(f"Drift Score: {drift_report.drift_score:.3f}")
    print(f"Drifted Columns: {len(drift_report.drifted_columns)}/{drift_report.total_columns}")
    if drift_report.drifted_columns:
        print(f"Drifted Column Names: {drift_report.drifted_columns}")
    print(f"Alert Triggered: {drift_report.alert_triggered}")
    print(f"Alert Severity: {drift_report.alert_severity}")
    print(f"Report Path: {drift_report.report_path}")
    
    return drift_report

if __name__ == "__main__":
    test_data_drift_with_modification()
