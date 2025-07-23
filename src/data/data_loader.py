"""
Data loading module for customer churn prediction.
Handles loading, splitting, and basic transformations of data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading and preprocessing class for customer churn dataset."""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize data loader with configuration."""
        self.config = self._load_config(config_path)
        self.label_encoders = {}
        self.data_info = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load data configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default data configuration."""
        return {
            'data_paths': {
                'raw_data': 'data/raw/customer_data.csv',
                'processed_data': 'data/processed/',
                'external_data': 'data/external/'
            },
            'preprocessing': {
                'target_column': 'Churn',
                'id_column': 'customerID',
                'categorical_columns': [
                    'gender', 'Partner', 'Dependents', 'PhoneService',
                    'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod'
                ],
                'numerical_columns': [
                    'tenure', 'MonthlyCharges', 'TotalCharges'
                ],
                'binary_columns': [
                    'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                    'PaperlessBilling'
                ]
            },
            'splitting': {
                'test_size': 0.2,
                'validation_size': 0.1,
                'random_state': 42,
                'stratify': True
            }
        }
    
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load raw customer churn data."""
        if file_path is None:
            file_path = self.config['data_paths']['raw_data']
        
        try:
            logger.info(f"Loading raw data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Store data info
            self.data_info['raw_data'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"Raw data loaded successfully. Shape: {df.shape}")
            return df
            
        except FileNotFoundError:
            logger.error(f"Raw data file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for processing."""
        logger.info("Starting data cleaning process")
        df_cleaned = df.copy()
        
        # Handle TotalCharges column (convert to numeric)
        if 'TotalCharges' in df_cleaned.columns:
            df_cleaned['TotalCharges'] = pd.to_numeric(
                df_cleaned['TotalCharges'], 
                errors='coerce'
            )
        
        # Handle missing values
        missing_before = df_cleaned.isnull().sum().sum()
        
        # Fill missing TotalCharges with 0 (new customers)
        if 'TotalCharges' in df_cleaned.columns:
            df_cleaned['TotalCharges'].fillna(0, inplace=True)
        
        # Drop rows with missing target values
        target_col = self.config['preprocessing']['target_column']
        if target_col in df_cleaned.columns:
            df_cleaned = df_cleaned.dropna(subset=[target_col])
        
        missing_after = df_cleaned.isnull().sum().sum()
        logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
        
        # Remove duplicate rows
        duplicates_before = df_cleaned.duplicated().sum()
        df_cleaned = df_cleaned.drop_duplicates()
        duplicates_after = df_cleaned.duplicated().sum()
        logger.info(f"Duplicates removed: {duplicates_before} -> {duplicates_after}")
        
        # Store cleaning info
        self.data_info['cleaned_data'] = {
            'shape': df_cleaned.shape,
            'missing_values_handled': missing_before - missing_after,
            'duplicates_removed': duplicates_before - duplicates_after,
            'cleaned_at': datetime.now().isoformat()
        }
        
        logger.info(f"Data cleaning completed. Final shape: {df_cleaned.shape}")
        return df_cleaned
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                fit_encoders: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        logger.info("Encoding categorical variables")
        df_encoded = df.copy()
        
        categorical_columns = self.config['preprocessing']['categorical_columns']
        
        for col in categorical_columns:
            if col in df_encoded.columns:
                if fit_encoders:
                    # Fit and transform
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                    logger.info(f"Fitted encoder for {col}: {le.classes_}")
                else:
                    # Transform only (use existing encoder)
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        df_encoded[col] = df_encoded[col].astype(str)
                        df_encoded[col] = df_encoded[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        logger.warning(f"No encoder found for {col}")
        
        # Handle target variable
        target_col = self.config['preprocessing']['target_column']
        if target_col in df_encoded.columns and fit_encoders:
            le_target = LabelEncoder()
            df_encoded[target_col] = le_target.fit_transform(df_encoded[target_col])
            self.label_encoders[target_col] = le_target
            logger.info(f"Target variable encoded: {le_target.classes_}")
        
        return df_encoded
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from existing data."""
        logger.info("Creating additional features")
        df_features = df.copy()
        
        # Feature engineering
        if 'tenure' in df_features.columns and 'MonthlyCharges' in df_features.columns:
            # Customer lifetime value
            df_features['CLV'] = df_features['tenure'] * df_features['MonthlyCharges']
        
        if 'TotalCharges' in df_features.columns and 'MonthlyCharges' in df_features.columns:
            # Average monthly charges
            df_features['AvgMonthlyCharges'] = (
                df_features['TotalCharges'] / (df_features['tenure'] + 1)
            )
        
        # Service usage count
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        available_services = [col for col in service_columns if col in df_features.columns]
        if available_services:
            # Count of services used (assuming 1 means Yes/True)
            df_features['ServiceCount'] = df_features[available_services].sum(axis=1)
        
        # Tenure groups
        if 'tenure' in df_features.columns:
            df_features['TenureGroup'] = pd.cut(
                df_features['tenure'],
                bins=[0, 12, 24, 48, 72, 100],
                labels=['0-1 year', '1-2 years', '2-4 years', '4-6 years', '6+ years']
            )
            # Convert to numeric
            df_features['TenureGroup'] = df_features['TenureGroup'].cat.codes
        
        # Monthly charges groups
        if 'MonthlyCharges' in df_features.columns:
            df_features['ChargeGroup'] = pd.cut(
                df_features['MonthlyCharges'],
                bins=[0, 35, 65, 95, 200],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            # Convert to numeric
            df_features['ChargeGroup'] = df_features['ChargeGroup'].cat.codes
        
        logger.info(f"Feature engineering completed. New shape: {df_features.shape}")
        return df_features
    
    def split_data(self, df: pd.DataFrame, 
                save_splits: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data into train/validation/test sets")
        
        # Separate features and target
        target_col = self.config['preprocessing']['target_column']
        id_col = self.config['preprocessing']['id_column']
        
        # Remove ID column from features
        feature_cols = [col for col in df.columns if col not in [target_col, id_col]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split configuration
        test_size = self.config['splitting']['test_size']
        val_size = self.config['splitting']['validation_size']
        random_state = self.config['splitting']['random_state']
        stratify = self.config['splitting']['stratify']
        
        # First split: train+val vs test
        stratify_y = y if stratify else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, 
            random_state=random_state, 
            stratify=stratify_y
        )
        
        # Second split: train vs validation
        val_size_adjusted = val_size / (1 - test_size)
        stratify_y_temp = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_y_temp
        )
        
        # Combine features and target
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Store split information
        self.data_info['splits'] = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'train_churn_rate': y_train.mean(),
            'val_churn_rate': y_val.mean(),
            'test_churn_rate': y_test.mean(),
            'split_at': datetime.now().isoformat()
        }
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_df)} samples")
        logger.info(f"  Test: {len(test_df)} samples")
        
        # Save splits if requested
        if save_splits:
            self.save_splits(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, 
                val_df: pd.DataFrame, 
                test_df: pd.DataFrame):
        """Save train, validation, and test sets to files."""
        processed_path = Path(self.config['data_paths']['processed_data'])
        processed_path.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_df.to_csv(processed_path / 'train.csv', index=False)
        val_df.to_csv(processed_path / 'validation.csv', index=False)
        test_df.to_csv(processed_path / 'test.csv', index=False)
        
        # Save feature information
        feature_info = {
            'feature_columns': list(train_df.columns),
            'target_column': self.config['preprocessing']['target_column'],
            'categorical_columns': self.config['preprocessing']['categorical_columns'],
            'numerical_columns': self.config['preprocessing']['numerical_columns'],
            'created_features': ['CLV', 'AvgMonthlyCharges', 'ServiceCount', 'TenureGroup', 'ChargeGroup']
        }
        
        with open(processed_path / 'feature_info.yaml', 'w') as f:
            yaml.dump(feature_info, f)
        
        logger.info(f"Data splits saved to {processed_path}")
    
    def save_encoders(self, output_path: str = "models/encoders/"):
        """Save label encoders for later use."""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        for col_name, encoder in self.label_encoders.items():
            encoder_path = Path(output_path) / f"{col_name}_encoder.joblib"
            joblib.dump(encoder, encoder_path)
        
        logger.info(f"Encoders saved to {output_path}")
    
    def load_encoders(self, input_path: str = "models/encoders/"):
        """Load previously saved label encoders."""
        encoder_path = Path(input_path)
        if not encoder_path.exists():
            logger.warning(f"Encoder path does not exist: {input_path}")
            return
        
        for encoder_file in encoder_path.glob("*_encoder.joblib"):
            col_name = encoder_file.stem.replace("_encoder", "")
            encoder = joblib.load(encoder_file)
            self.label_encoders[col_name] = encoder
        
        logger.info(f"Encoders loaded from {input_path}")
    
    def load_processed_data(self, data_type: str = "train") -> pd.DataFrame:
        """Load processed data (train, validation, or test)."""
        processed_path = Path(self.config['data_paths']['processed_data'])
        file_path = processed_path / f"{data_type}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {data_type} data: {df.shape}")
        return df
    
    def get_feature_target_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Split dataframe into features and target."""
        target_col = self.config['preprocessing']['target_column']
        id_col = self.config['preprocessing']['id_column']
        
        # Remove ID column if present
        feature_cols = [col for col in df.columns if col not in [target_col, id_col]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        return X, y
    
    def process_full_pipeline(self, input_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Run the complete data processing pipeline."""
        logger.info("Starting full data processing pipeline")
        
        # Load raw data
        df_raw = self.load_raw_data(input_path)
        
        # Clean data
        df_cleaned = self.clean_data(df_raw)
        
        # Encode categorical variables
        df_encoded = self.encode_categorical_variables(df_cleaned, fit_encoders=True)
        
        # Create features
        df_features = self.create_features(df_encoded)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df_features, save_splits=True)
        
        # Save encoders
        self.save_encoders()
        
        # Save data processing info
        self.save_data_info()
        
        logger.info("Full data processing pipeline completed")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'raw': df_raw,
            'processed': df_features
        }
    
    def save_data_info(self, output_path: str = "reports/data_quality/data_info.yaml"):
        """Save data processing information."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.data_info, f, default_flow_style=False)
        
        logger.info(f"Data processing info saved to {output_path}")
    
    def load_data_info(self, input_path: str = "reports/data_quality/data_info.yaml"):
        """Load data processing information."""
        try:
            with open(input_path, 'r') as f:
                self.data_info = yaml.safe_load(f)
            logger.info(f"Data processing info loaded from {input_path}")
        except FileNotFoundError:
            logger.warning(f"Data info file not found: {input_path}")
    
    def get_data_summary(self) -> Dict:
        """Get summary of data processing."""
        return {
            'config': self.config,
            'data_info': self.data_info,
            'encoders': list(self.label_encoders.keys()),
            'processing_status': 'completed' if self.data_info else 'not_started'
        }

def main():
    """Main function for testing data loader."""
    # Example usage
    loader = DataLoader()
    
    try:
        # Run full pipeline
        datasets = loader.process_full_pipeline()
        
        # Print summary
        print("\nData Processing Summary:")
        print(f"Raw data shape: {datasets['raw'].shape}")
        print(f"Processed data shape: {datasets['processed'].shape}")
        print(f"Train set: {datasets['train'].shape}")
        print(f"Validation set: {datasets['validation'].shape}")
        print(f"Test set: {datasets['test'].shape}")
        
        # Print feature information
        train_X, train_y = loader.get_feature_target_split(datasets['train'])
        print(f"\nFeatures: {train_X.shape[1]}")
        print(f"Target distribution: {train_y.value_counts().to_dict()}")
        
        # Print encoder information
        print(f"\nEncoders created: {list(loader.label_encoders.keys())}")
        
    except FileNotFoundError:
        print("Sample data file not found. Please ensure data/raw/customer_data.csv exists.")
    except Exception as e:
        print(f"Error in data processing: {str(e)}")

if __name__ == "__main__":
    main()