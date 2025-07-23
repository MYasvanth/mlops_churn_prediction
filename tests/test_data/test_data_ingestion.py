import pytest
import pandas as pd
from src.data.data_ingestion import DataIngestion
from src.utils.config_loader import ConfigLoader

@pytest.fixture
def data_config():
    config_loader = ConfigLoader()
    params = config_loader.load_params()
    return config_loader.get_data_config(params)

def test_load_csv_data(data_config):
    ingestion = DataIngestion(data_config)
    df = ingestion.load_csv_data(data_config.raw_data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_clean_column_names(data_config):
    ingestion = DataIngestion(data_config)
    df = ingestion.load_csv_data(data_config.raw_data_path)
    df_clean = ingestion.clean_column_names(df)
    assert all(col == col.strip() for col in df_clean.columns)

def test_handle_data_types(data_config):
    ingestion = DataIngestion(data_config)
    df = ingestion.load_csv_data(data_config.raw_data_path)
    df_processed = ingestion.handle_data_types(df)
    # Check that no string values remain in TotalCharges column
    assert not df_processed['TotalCharges'].apply(lambda x: isinstance(x, str)).any()

def test_remove_duplicates(data_config):
    ingestion = DataIngestion(data_config)
    df = ingestion.load_csv_data(data_config.raw_data_path)
    df_no_dup = ingestion.remove_duplicates(df)
    assert len(df_no_dup) <= len(df)

def test_handle_missing_values(data_config):
    ingestion = DataIngestion(data_config)
    df = ingestion.load_csv_data(data_config.raw_data_path)
    df_processed = ingestion.handle_missing_values(df)
    assert df_processed.isnull().sum().sum() <= df.isnull().sum().sum()

def test_validate_data_schema(data_config):
    ingestion = DataIngestion(data_config)
    df = ingestion.load_csv_data(data_config.raw_data_path)
    df_processed = ingestion.handle_data_types(df)
    assert ingestion.validate_data_schema(df_processed)
