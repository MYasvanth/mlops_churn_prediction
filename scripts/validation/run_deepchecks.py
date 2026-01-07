import os
from deepchecks.tabular import Dataset as TabularDataset
from deepchecks.tabular.suites import full_suite as tabular_full_suite
import pandas as pd
import os
from src.data.data_loader import DataLoader

def main():
    # Load processed data
    loader = DataLoader()
    df = loader.load_processed_data('train')
    X, y = loader.get_feature_target_split(df)
    
    # Create Deepchecks dataset (without 'name' argument)
    dataset = TabularDataset(X, label=y, cat_features=None)
    
    # Run full suite of checks
    suite = tabular_full_suite()
    result = suite.run(dataset)
    
    # Save report
    report_path = os.path.join('reports', 'deepchecks_report.html')
    os.makedirs('reports', exist_ok=True)
    result.save_as_html(report_path)
    print(f"Deepchecks report saved to {report_path}")

if __name__ == "__main__":
    main()
