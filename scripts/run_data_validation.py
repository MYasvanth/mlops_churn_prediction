import os
import pandas as pd
from scipy.stats import ks_2samp
from src.data.data_loader import DataLoader

def basic_data_validation(df: pd.DataFrame):
    print("Data Validation Report")
    print("======================")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe(include='all'))
    
def data_drift_detection(df_ref: pd.DataFrame, df_new: pd.DataFrame, threshold=0.05):
    print("\nData Drift Detection Report")
    print("===========================")
    drifted_features = []
    for col in df_ref.columns:
        if col in df_new.columns and pd.api.types.is_numeric_dtype(df_ref[col]):
            stat, p_value = ks_2samp(df_ref[col].dropna(), df_new[col].dropna())
            if p_value < threshold:
                drifted_features.append(col)
                print(f"Feature '{col}' drift detected (p-value={p_value:.4f})")
    if not drifted_features:
        print("No numeric feature drift detected.")
    return drifted_features

def main():
    loader = DataLoader()
    df_train = loader.load_processed_data('train')
    df_test = loader.load_processed_data('test')
    
    print("Training Data Validation:")
    basic_data_validation(df_train)
    
    print("\nTest Data Validation:")
    basic_data_validation(df_test)
    
    data_drift_detection(df_train, df_test)

if __name__ == "__main__":
    main()
