import pandas as pd
from src.monitoring.data_drift_monitor import DataDriftMonitor

def main():
    # Load reference and current data
    reference_data = pd.read_csv("data/processed/train.csv")
    current_data = pd.read_csv("data/processed/test.csv")
    
    # Instantiate DataDriftMonitor with config path
    monitor = DataDriftMonitor(config_path="configs/monitoring/monitoring_config.yaml")
    
    # Set reference data
    monitor.set_reference_data(reference_data)
    
    # Detect drift on current data
    drift_report = monitor.detect_drift(current_data)
    
    # Print drift report summary
    print("Drift Report ID:", drift_report.report_id)
    print("Timestamp:", drift_report.timestamp)
    print("Drift Detected:", drift_report.drift_detected)
    print("Drift Score:", drift_report.drift_score)
    print("Drifted Columns:", drift_report.drifted_columns)
    print("Report Path:", drift_report.report_path)

if __name__ == "__main__":
    main()
