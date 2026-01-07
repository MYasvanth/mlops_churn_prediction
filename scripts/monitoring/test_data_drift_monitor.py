import pandas as pd
from src.monitoring.data_drift_monitor import DataDriftMonitor

def test_data_drift_monitor():
    # Initialize the DataDriftMonitor
    monitor = DataDriftMonitor(config_path="configs/monitoring/monitoring_config.yaml")
    
    # Load reference data (using the same data as configured)
    reference_data = pd.read_csv("data/processed/raw_data.csv")
    monitor.set_reference_data(reference_data)
    
    # Use the same data as current data to test the drift detection
    # In a real scenario, this would be different production data
    current_data = pd.read_csv("data/processed/raw_data.csv")
    
    # Run drift detection
    drift_report = monitor.detect_drift(current_data)
    
    # Print the results
    print(f"Drift Detected: {drift_report.drift_detected}")
    print(f"Drift Score: {drift_report.drift_score}")
    print(f"Drifted Columns: {drift_report.drifted_columns}")
    print(f"Total Columns: {drift_report.total_columns}")
    print(f"Report Path: {drift_report.report_path}")

if __name__ == "__main__":
    test_data_drift_monitor()
