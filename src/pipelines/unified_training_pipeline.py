"""
Unified Training Pipeline for Churn Prediction
Integrates all model types with unified configuration and registry
"""

import os
import json
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mlflow
import mlflow.sklearn
from pathlib import Path
import logging

from ..utils.logger import get_logger
from ..utils.config_loader import load_config
from ..models.unified_model_interface import UnifiedModelTrainer, UnifiedModelEvaluator
from ..models.unified_model_registry_fixed import UnifiedModelRegistry
from ..data.data_loader import DataLoader

logger = get_logger(__name__)

class UnifiedTrainingPipeline:
    """Unified training pipeline supporting all model types."""
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path or "configs/model/unified_model_config.yaml")
        self.registry = UnifiedModelRegistry()
        self.data_loader = DataLoader()
        
    def prepare_data(self, data_path: str, target_column: str = "Churn") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        try:
            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} rows from {data_path}")
            
            # Handle target encoding
            if target_column in df.columns:
                if df[target_column].dtype == 'object':
                    df[target_column] = df[target_column].map({'Yes': 1, 'No': 0})
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                   hyperparameters: Dict[str, Any] = None) -> Any:
        """Train a specific model type."""
        try:
            trainer = UnifiedModelTrainer(model_type)
            model = trainer.train(X_train, y_train, hyperparameters)
            
            logger.info(f"Successfully trained {model_type} model")
            return model
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {str(e)}")
            raise
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            evaluator = UnifiedModelEvaluator()
            metrics = evaluator.evaluate(model, X_test, y_test)
            
            logger.info(f"Model evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def register_model(self, model: Any, model_type: str, metrics: Dict[str, float],
                      feature_columns: List[str], stage: str = "staging") -> str:
        """Register model with unified registry."""
        try:
            metadata = {
                "model_name": f"churn_{model_type}",
                "model_type": model_type,
                "performance_metrics": metrics,
                "hyperparameters": {},  # Could be extracted from model
                "feature_columns": feature_columns,
                "created_by": "unified_pipeline",
                "description": f"Churn prediction model using {model_type}"
            }
            
            model_id = self.registry.register_model(model, metadata, stage)
            logger.info(f"Model registered with ID: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def run_training(self, model_type: str, data_path: str = None, 
                    hyperparameters: Dict[str, Any] = None, 
                    stage: str = "staging") -> Dict[str, Any]:
        """Run complete training pipeline for a specific model type."""
        try:
            # Set up MLflow experiment
            mlflow.set_experiment("churn_prediction_unified")
            
            with mlflow.start_run():
                # Prepare data
                if data_path is None:
                    data_path = "data/processed/train.csv"
                
                X_train, X_test, y_train, y_test = self.prepare_data(data_path)
                
                # Log parameters
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("data_path", data_path)
                mlflow.log_param("stage", stage)
                
                # Train model
                model = self.train_model(model_type, X_train, y_train, hyperparameters)
                
                # Evaluate model
                metrics = self.evaluate_model(model, X_test, y_test)
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Register model
                model_id = self.register_model(
                    model, model_type, metrics, 
                    X_train.columns.tolist(), stage
                )
                
                # Save model locally
                model_path = f"models/{stage}/{model_id}/model.pkl"
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                import joblib
                joblib.dump(model, model_path)
                
                # Log model with MLflow
                mlflow.sklearn.log_model(model, "model")
                
                result = {
                    "model_id": model_id,
                    "model_type": model_type,
                    "metrics": metrics,
                    "model_path": model_path,
                    "stage": stage
                }
                
                logger.info(f"Training pipeline completed for {model_type}")
                return result
                
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise
    
    def train_multiple_models(self, model_types: List[str], 
                            data_path: str = None) -> Dict[str, Dict[str, Any]]:
        """Train multiple model types and compare results."""
        try:
            results = {}
            
            for model_type in model_types:
                try:
                    logger.info(f"Training {model_type} model...")
                    result = self.run_training(model_type, data_path)
                    results[model_type] = result
                except Exception as e:
                    logger.error(f"Error training {model_type}: {str(e)}")
                    results[model_type] = {"error": str(e)}
            
            # Find best model
            best_model = None
            best_score = -1
            
            for model_type, result in results.items():
                if "error" not in result and "metrics" in result:
                    score = result["metrics"].get("f1_score", 0)
                    if score > best_score:
                        best_score = score
                        best_model = model_type
            
            if best_model:
                logger.info(f"Best model: {best_model} with F1 score: {best_score}")
                results["best_model"] = best_model
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multiple model training: {str(e)}")
            raise
    
    def auto_select_best_model(self, data_path: str = None) -> Dict[str, Any]:
        """Automatically select and train the best model."""
        supported_models = self.config.get("models", {}).get("supported_types", [])
        return self.train_multiple_models(supported_models, data_path)

# Convenience functions
def train_single_model(model_type: str, data_path: str = None, 
                      hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Train a single model type."""
    pipeline = UnifiedTrainingPipeline()
    return pipeline.run_training(model_type, data_path, hyperparameters)

def train_all_models(data_path: str = None) -> Dict[str, Dict[str, Any]]:
    """Train all supported model types."""
    pipeline = UnifiedTrainingPipeline()
    supported_models = pipeline.config.get("models", {}).get("supported_types", [])
    return pipeline.train_multiple_models(supported_models, data_path)

def auto_train_best_model(data_path: str = None) -> Dict[str, Any]:
    """Automatically train and select the best model."""
    pipeline = UnifiedTrainingPipeline()
    return pipeline.auto_select_best_model(data_path)

if __name__ == "__main__":
    # Example usage
    pipeline = UnifiedTrainingPipeline()
    
    # Train single model
    # result = train_single_model("xgboost")
    
    # Train all models
    results = train_all_models()
    print(json.dumps(results, indent=2, default=str))
