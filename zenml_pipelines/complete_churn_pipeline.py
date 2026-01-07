"""
Complete ZenML Pipeline for Churn Prediction
End-to-end pipeline from data loading to model deployment
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import mlflow
import joblib
from pathlib import Path
import logging

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.mlflow import MlflowIntegration
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

# Import unified components
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.unified_model_interface import UnifiedModelTrainer, UnifiedModelEvaluator
from models.unified_model_registry_fixed import UnifiedModelRegistry
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

@step
def data_loader(data_path: str = "data/processed") -> pd.DataFrame:
    """Load and validate training data."""
    try:
        train_path = Path(data_path) / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        df = pd.read_csv(train_path)
        logger.info(f"Loaded training data: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

@step
def data_preprocessor(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess data for training."""
    try:
        # Separate features and target
        target_col = 'Churn' if 'Churn' in data.columns else 'churn'
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Convert target variable from string to numeric (No -> 0, Yes -> 1)
        y = y.map({'No': 0, 'Yes': 1})
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
        
        # Handle missing values
        X = X.fillna(X.median())
        
        logger.info(f"Preprocessed data: X={X.shape}, y={y.shape}")
        logger.info(f"Target variable distribution: {y.value_counts().to_dict()}")
        return X, y
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

@step
def model_trainer(X: pd.DataFrame, y: pd.Series, model_type: str = "xgboost") -> Dict[str, Any]:
    """Train model using unified interface."""
    try:
        config = load_config("configs/model/unified_model_config.yaml")
        trainer = UnifiedModelTrainer(model_type)
        
        # Train model
        model = trainer.train(X, y)
        
        # Evaluate model
        evaluator = UnifiedModelEvaluator()
        metrics = evaluator.evaluate(model, X, y)
        
        # Save model temporarily
        model_path = Path("models/temp") / f"{model_type}_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
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

@step
def model_validator(training_result: Dict[str, Any]) -> bool:
    """Validate model performance against thresholds."""
    try:
        metrics = training_result["metrics"]
        evaluator = UnifiedModelEvaluator()
        is_valid = evaluator.validate_performance(metrics)
        
        if is_valid:
            logger.info("Model passed validation")
        else:
            logger.warning("Model failed validation")
        
        return is_valid
    
    except Exception as e:
        logger.error(f"Error validating model: {str(e)}")
        raise

@step
def model_registry(training_result: Dict[str, Any], is_valid: bool) -> str:
    """Register model in unified registry."""
    try:
        if not is_valid:
            logger.warning("Model not registered due to validation failure")
            return ""
        
        registry = UnifiedModelRegistry()
        model_path = training_result["model_path"]
        model_type = training_result["model_type"]
        metrics = training_result["metrics"]
        
        # Register model
        # Load the model from the saved path
        model = joblib.load(model_path)
        
        # Prepare metadata
        metadata = {
            "model_type": model_type,
            "metrics": metrics,
            "model_path": model_path
        }
        
        model_id = registry.register_model(
            model=model,
            metadata=metadata,
            stage="staging"
        )
        
        logger.info(f"Model registered: {model_id}")
        return model_id
    
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise

@step
def model_deployer(model_id: str, deploy_to_production: bool = False) -> str:
    """Deploy model to staging or production."""
    try:
        if not model_id:
            logger.warning("No model to deploy")
            return ""
        
        registry = UnifiedModelRegistry()
        
        if deploy_to_production:
            # Get model info to find current stage
            model_info = registry.get_model_info(model_id)
            current_stage = model_info.get("stage", "staging")
            
            # Promote from current stage to production
            success = registry.promote_model(model_id, current_stage, "production")
            if success:
                deployment_id = f"{model_id}_production"
                logger.info(f"Model deployed to production: {deployment_id}")
            else:
                logger.error("Failed to promote model to production")
                deployment_id = ""
        else:
            # For staging deployment, just return the model ID (already in staging)
            deployment_id = model_id
            logger.info(f"Model remains in staging: {deployment_id}")
        
        return deployment_id
    
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        raise

@pipeline
def churn_training_pipeline(
    data_path: str = "data/processed",
    model_type: str = "xgboost",
    deploy_to_production: bool = False
) -> None:
    """Complete end-to-end churn prediction pipeline."""
    
    # Data loading and preprocessing
    raw_data = data_loader(data_path)
    X, y = data_preprocessor(raw_data)
    
    # Model training
    training_result = model_trainer(X, y, model_type)
    
    # Model validation
    is_valid = model_validator(training_result)
    
    # Model registration
    model_id = model_registry(training_result, is_valid)
    
    # Model deployment
    deployment_id = model_deployer(model_id, deploy_to_production)
    
    logger.info("Pipeline execution completed successfully")

def run_pipeline(
    model_type: str = "xgboost",
    deploy_to_production: bool = False
) -> None:
    """Run the complete churn prediction pipeline."""
    try:
        logger.info("Starting churn prediction pipeline...")
        
        # Run pipeline
        pipeline_instance = churn_training_pipeline(
            model_type=model_type,
            deploy_to_production=deploy_to_production
        )
        
        # In ZenML 0.84.2, calling the pipeline function automatically executes it
        # and returns a PipelineRunResponse object
        run_response = pipeline_instance
        logger.info(f"Pipeline run completed with ID: {run_response.id}")
        
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run churn prediction pipeline")
    parser.add_argument("--model-type", default="xgboost", choices=["xgboost", "lightgbm", "random_forest", "logistic_regression", "svm"])
    parser.add_argument("--deploy-to-production", action="store_true")
    parser.add_argument("--data-path", default="data/processed")
    
    args = parser.parse_args()
    
    run_pipeline(
        model_type=args.model_type,
        deploy_to_production=args.deploy_to_production
    )
