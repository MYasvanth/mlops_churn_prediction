#!/usr/bin/env python3
"""
Complete pipeline to train churn prediction models from raw data
Handles data ingestion, preprocessing, training, and deployment
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import unified components
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

from models.unified_model_interface import UnifiedModelTrainer, UnifiedModelEvaluator
from models.unified_model_registry import UnifiedModelRegistry
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

class RawDataTrainer:
    """Complete pipeline for training from raw data"""
    
    def __init__(self, config_path: str = "configs/model/unified_model_config.yaml"):
        self.config = load_config(config_path)
        self.data_path = Path("data/raw")
        self.processed_path = Path("data/processed")
        self.models_path = Path("models")
        
    def load_raw_data(self, filename: str = "churn_data.csv") -> pd.DataFrame:
        """Load raw data from data/raw directory"""
        try:
            raw_file = self.data_path / filename
            if not raw_file.exists():
                # Create sample data if file doesn't exist
                logger.warning(f"Raw data file not found: {raw_file}")
                logger.info("Creating sample data for demonstration")
                return self._create_sample_data()
            
            df = pd.read_csv(raw_file)
            logger.info(f"Loaded raw data: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample churn data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'customerID': [f'CUST_{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(1, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
            'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples),
            'TotalCharges': np.random.uniform(18.25, 8684.8, n_samples),
            'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
        }
        
        df = pd.DataFrame(data)
        
        # Save sample data
        self.data_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.data_path / "churn_data.csv", index=False)
        logger.info("Created and saved sample data")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess raw data for training"""
        try:
            # Handle missing values
            df = df.replace(' ', np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df = df.dropna()
            
            # Drop customer ID
            df = df.drop('customerID', axis=1)
            
            # Encode categorical variables
            categorical_cols = df.select_dtypes(include=['object']).columns
            categorical_cols = [col for col in categorical_cols if col != 'Churn']
            
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            
            # Separate features and target
            X = df.drop('Churn', axis=1)
            y = df['Churn']
            
            # Scale numerical features
            numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
            
            # Save processed data
            self.processed_path.mkdir(parents=True, exist_ok=True)
            train_df = pd.concat([X, y], axis=1)
            train_df.to_csv(self.processed_path / "train.csv", index=False)
            
            logger.info(f"Preprocessed data: X={X.shape}, y={y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = "xgboost") -> Dict[str, Any]:
        """Train model using unified interface"""
        try:
            trainer = UnifiedModelTrainer(model_type)
            model = trainer.train(X, y)
            
            # Evaluate model
            evaluator = UnifiedModelEvaluator()
            metrics = evaluator.evaluate(model, X, y)
            
            # Save model
            self.models_path.mkdir(parents=True, exist_ok=True)
            model_path = self.models_path / f"{model_type}_model.pkl"
            trainer.save_model(str(model_path))
            
            result = {
                "model": model,
                "metrics": metrics,
                "model_path": str(model_path),
                "model_type": model_type,
                "passed_validation": evaluator.validate_performance(metrics)
            }
            
            logger.info(f"Training completed: {model_type}, metrics={metrics}")
            return result
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def register_and_deploy(self, training_result: Dict[str, Any], deploy_to_production: bool = False) -> str:
        """Register and deploy model"""
        try:
            if not training_result["passed_validation"]:
                logger.warning("Model not registered due to validation failure")
                return ""
            
            registry = UnifiedModelRegistry()
            model_path = training_result["model_path"]
            model_type = training_result["model_type"]
            metrics = training_result["metrics"]
            
            # Register model
            model_id = registry.register_model(
                model_path=model_path,
                model_type=model_type,
                metrics=metrics,
                stage="staging"
            )
            
            # Deploy if requested
            if deploy_to_production:
                deployment_id = registry.promote_to_production(model_id)
                logger.info(f"Model deployed to production: {deployment_id}")
            else:
                deployment_id = registry.deploy_to_staging(model_id)
                logger.info(f"Model deployed to staging: {deployment_id}")
            
            return model_id
            
        except Exception as e:
            logger.error(f"Error registering/deploying model: {str(e)}")
            raise
    
    def run_complete_pipeline(self, model_type: str = "xgboost", deploy_to_production: bool = False) -> Dict[str, Any]:
        """Run complete pipeline from raw data to deployment"""
        try:
            logger.info("Starting complete pipeline from raw data...")
            
            # Step 1: Load raw data
            raw_data = self.load_raw_data()
            
            # Step 2: Preprocess data
            X, y = self.preprocess_data(raw_data)
            
            # Step 3: Train model
            training_result = self.train_model(X, y, model_type)
            
            # Step 4: Register and deploy
            model_id = self.register_and_deploy(training_result, deploy_to_production)
            
            logger.info("Complete pipeline execution finished successfully")
            
            return {
                "model_id": model_id,
                "metrics": training_result["metrics"],
                "model_type": model_type,
                "data_shape": raw_data.shape
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Train churn prediction from raw data")
    parser.add_argument(
        "--model-type",
        choices=["xgboost", "lightgbm", "random_forest", "logistic_regression", "svm"],
        default="xgboost",
        help="Model type to train"
    )
    parser.add_argument(
        "--deploy-to-production",
        action="store_true",
        help="Deploy model to production after training"
    )
    parser.add_argument(
        "--raw-data-file",
        default="churn_data.csv",
        help="Raw data filename in data/raw/"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("RAW DATA TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Deploy to Production: {args.deploy_to_production}")
    logger.info(f"Raw Data File: {args.raw_data_file}")
    
    try:
        trainer = RawDataTrainer()
        result = trainer.run_complete_pipeline(
            model_type=args.model_type,
            deploy_to_production=args.deploy_to_production
        )
        
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Model ID: {result['model_id']}")
        logger.info(f"Final Metrics: {result['metrics']}")
        logger.info(f"Data Processed: {result['data_shape'][0]} samples, {result['data_shape'][1]} features")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
