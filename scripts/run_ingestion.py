import sys
import os
from src.data.data_ingestion import ingest_data
sys.path.append(os.path.abspath('.'))
if __name__ == "__main__":
    ingest_data("data/raw/Customer_Churn.csv", "data/processed/train.csv")
