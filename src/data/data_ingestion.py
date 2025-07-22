# data_ingestion.py
"""
Data ingestion module for Customer Churn MLOps project.
Handles loading and initial processing of raw customer data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader, DataConfig
from src.utils.helpers import create_directories, save_json, log_data_quality_report

logger = get_logger(__name__)


class DataIngestion:
    """
    Handles data ingestion from various sources and formats.
    Supports CSV, Excel, and database connections.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        
    def load_csv_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file with error handling.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded dataframe
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        try:
            # Read CSV with proper handling of different data types
            df = pd.read_csv(
                filepath,
                encoding='utf-8',
                low_memory=False
            )
            
            self.logger.info(f"Successfully loaded data from {filepath}")
            self.logger.info(f"Data shape: {df.shape}")
            
            return df
            
        except pd.errors.EmptyDataError:
            self.logger.error(f"Empty data file: {filepath}")
            raise
        except pd.errors.ParserError as e:
            self.logger.error(f"Error parsing CSV file {filepath}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading data from {filepath}: {e}")
            raise
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize column names.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with cleaned column names
        """
        # Remove leading/trailing whitespace
        df.columns = df.columns.str.strip()
        
        # Log column cleaning
        original_columns = df.columns.tolist()
        self.logger.info(f"Original columns: {original_columns}")
        
        return df
    
    def handle_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle and convert appropriate data types.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with corrected data types
        """
        df_processed = df.copy()
        
        # Convert TotalCharges to numeric (it might be stored as string)
        if 'TotalCharges' in df_processed.columns:
            # Replace empty strings with NaN
            df_processed['TotalCharges'] = df_processed['TotalCharges'].replace(' ', np.nan)
            # Convert to numeric
            df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
            self.logger.info("Converted TotalCharges to numeric type")
        
        # Ensure SeniorCitizen is integer
        if 'SeniorCitizen' in df_processed.columns:
            df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(int)
            self.logger.info("Converted SeniorCitizen to integer type")
        
        # Ensure tenure is integer
        if 'tenure' in df_processed.columns:
            df_processed['tenure'] = df_processed['tenure'].astype(int)
            self.logger.info("Converted tenure to integer type")
        
        # Ensure MonthlyCharges is float
        if 'MonthlyCharges' in df_processed.columns:
            df_processed['MonthlyCharges'] = df_processed['MonthlyCharges'].astype(float)
            self.logger.info("Converted MonthlyCharges to float type")
        
        return df_processed
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with duplicates removed
        """
        initial_rows = len(df)
        df_deduplicated = df.drop_duplicates()
        final_rows = len(df_deduplicated)
        
        duplicates_removed = initial_rows - final_rows
        if duplicates_removed > 0:
            self.logger.warning(f"Removed {duplicates_removed} duplicate rows")
        else:
            self.logger.info("No duplicate rows found")
        
        return df_deduplicated
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with missing values handled
        """
        df_processed = df.copy()
        
        # Log missing values before processing
        missing_before = df_processed.isnull().sum()
        if missing_before.sum() > 0:
            self.logger.info("Missing values before processing:")
            for col, count in missing_before[missing_before > 0].items():
                percentage = (count / len(df_processed)) * 100
                self.logger.info(f"  {col}: {count} ({percentage:.2f}%)")
        
        # Handle TotalCharges missing values
        if 'TotalCharges' in df_processed.columns:
            # Fill missing TotalCharges with 0 for customers with 0 tenure
            mask = (df_processed['TotalCharges'].isnull()) & (df_processed['tenure'] == 0)
            df_processed.loc[mask, 'TotalCharges'] = 0.0
            self.logger.info("Filled TotalCharges missing values for zero tenure customers")
        
        # Log missing values after processing
        missing_after = df_processed.isnull().sum()
        if missing_after.sum() > 0:
            self.logger.warning("Missing values after processing:")
            for col, count in missing_after[missing_after > 0].items():
                percentage = (count / len(df_processed)) * 100
                self.logger.warning(f"  {col}: {count} ({percentage:.2f}%)")
        
        return df_processed
    
    def validate_data_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that the data matches expected schema.
        
        Args:
            df: Input dataframe
            
        Returns:
            True if validation passes, False otherwise
        """
        expected_columns = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
        
        missing_columns = set(expected_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(expected_columns)
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        if extra_columns:
            self.logger.warning(f"Extra columns found: {extra_columns}")
        
        # Validate data types
        if 'SeniorCitizen' in df.columns:
            if not df['SeniorCitizen'].dtype in ['int64', 'int32']:
                self.logger.error("SeniorCitizen should be integer type")
                return False
        
        if 'tenure' in df.columns:
            if not df['tenure'].dtype in ['int64', 'int32']:
                self.logger.error("tenure should be integer type")
                return False
        
        self.logger.info("Data schema validation passed")
        return True
    
    def generate_data_report(self, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data report with JSON serializable types.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary containing data report
    """
    def serialize_value(value):
        """Convert values to JSON serializable format"""
        if isinstance(value, (np.integer, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return serialize_value(value.tolist())
        elif isinstance(value, dict):
            return {str(k): serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [serialize_value(item) for item in value]
        return value

    # Basic statistics
    report = {
        'total_rows': serialize_value(len(df)),
        'total_columns': serialize_value(len(df.columns)),
        'memory_usage_mb': serialize_value(df.memory_usage(deep=True).sum() / (1024 ** 2)),
        'duplicate_rows': serialize_value(df.duplicated().sum()),
        'missing_values': serialize_value(df.isnull().sum().to_dict()),
        'data_types': {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
    }

    # Target distribution
    if 'Churn' in df.columns:
        report['churn_distribution'] = serialize_value(df['Churn'].value_counts().to_dict())

    # Numerical statistics
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    if len(numerical_columns) > 0:
        report['numerical_stats'] = serialize_value(df[numerical_columns].describe().to_dict())

    # Categorical statistics
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        report['categorical_stats'] = {}
        for col in categorical_columns:
            if col != 'customerID':
                report['categorical_stats'][col] = serialize_value(df[col].value_counts().to_dict())

    return report
    
    def run_data_ingestion(self) -> pd.DataFrame:
        """
        Main method to run the complete data ingestion pipeline.
        
        Returns:
            Processed dataframe
        """
        self.logger.info("Starting data ingestion process")
        
        # Load raw data
        df = self.load_csv_data(self.config.raw_data_path)
        log_data_quality_report(df, "raw")
        
        # Clean column names
        df = self.clean_column_names(df)
        
        # Handle data types
        df = self.handle_data_types(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Validate schema
        if not self.validate_data_schema(df):
            raise ValueError("Data schema validation failed")
        
        # Generate and save data report
        report = self.generate_data_report(df)
        report_path = "reports/data_quality/ingestion_report.json"
        save_json(report, report_path)
        
        # Save processed data
        output_dir = os.path.dirname(self.config.processed_data_path)
        create_directories([output_dir])
        
        df.to_csv(self.config.processed_data_path, index=False)
        self.logger.info(f"Saved processed data to: {self.config.processed_data_path}")
        
        log_data_quality_report(df, "processed")
        self.logger.info("Data ingestion process completed successfully")
        
        return df


def main():
    """Main function to run data ingestion."""
    try:
        # Load configuration
        config_loader = ConfigLoader()
        params = config_loader.load_params()
        data_config = config_loader.get_data_config(params)
        
        # Run data ingestion
        ingestion = DataIngestion(data_config)
        df = ingestion.run_data_ingestion()
        
        logger.info(f"Data ingestion completed. Final shape: {df.shape}")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()