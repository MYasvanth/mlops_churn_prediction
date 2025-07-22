# training_pipeline.py
"""
Training Pipeline for Churn Prediction MLOps

This pipeline orchestrates the complete training workflow including:
- Data ingestion and validation
- Feature engineering and selection
- Model training with hyperparameter tuning
- Model evaluation and registration
- Automated model deployment to staging
"""

import os
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
from zenml import step, pipeline
from zenml.client import Client
from zenml.logger import get_logger
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.evidently.data_validators import EvidentlyDataValidator
from zenml.integrations.evidently.steps import evidently_report_step
from zenml.steps import BaseStepConfig

# Import custom modules
from src.data.data_ingestion import DataIngestionConfig
from src.data.data_validation import DataValidationConfig
from src.features.feature_engineering import FeatureEngineeringConfig
from src.features.preprocessing import PreprocessingConfig
from src.models.model_trainer import ModelTrainerConfig
from src.models.model_evaluator import ModelEvaluatorConfig
from src.models.hyperparameter_tuner import HyperparameterTunerConfig
from src.monitoring.data_drift_monitor import DataDriftMonitorConfig
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

# Step Configurations
class DataIngestionStepConfig(BaseStepConfig):
    """Configuration for data ingestion step."""
    data_path: str = "data/raw/Customer_data.csv"
    output_path: str = "data/processed/"
    test_size: float = 0.2
    random_state: int = 42

class FeatureEngineeringStepConfig(BaseStepConfig):
    """Configuration for feature engineering step."""
    target_column: str = "churn"
    categorical_features: list = ["gender", "contract_type", "payment_method"]
    numerical_features: list = ["tenure", "monthly_charges", "total_charges"]
    feature_selection_method: str = "selectkbest"
    k_features: int = 15

class ModelTrainingStepConfig(BaseStepConfig):
    """Configuration for model training step."""
    model_type: str = "random_forest"
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    use_hyperparameter_tuning: bool = True

class ModelEvaluationStepConfig(BaseStepConfig):
    """Configuration for model evaluation step."""
    threshold: float = 0.5
    metrics: list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    min_accuracy: float = 0.75
    min_roc_auc: float = 0.80

# Pipeline Steps
@step
def data_ingestion_step(
    config: DataIngestionStepConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ingests raw customer data and splits into train/test sets.
    
    Args:
        config: Configuration for data ingestion
        
    Returns:
        Tuple of (training_data, test_data)
    """
    logger.info("Starting data ingestion...")
    
    # Load raw data
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"Data file not found: {config.data_path}")
    
    df = pd.read_csv(config.data_path)
    logger.info(f"Loaded {len(df)} rows from {config.data_path}")
    
    # Basic data validation
    if df.empty:
        raise ValueError("Empty dataset provided")
    
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in dataset")
    
    # Split data
    train_df, test_df = train_test_split(
        df, 
        test_size=config.test_size, 
        random_state=config.random_state,
        stratify=df[config.target_column]
    )
    
    # Save processed data
    os.makedirs(config.output_path, exist_ok=True)
    train_df.to_csv(os.path.join(config.output_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(config.output_path, "test.csv"), index=False)
    
    logger.info(f"Data split completed: {len(train_df)} train, {len(test_df)} test samples")
    
    return train_df, test_df

@step
def data_validation_step(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates data quality and detects anomalies.
    
    Args:
        train_data: Training dataset
        test_data: Test dataset
        
    Returns:
        Tuple of validated (training_data, test_data)
    """
    logger.info("Starting data validation...")
    
    # Check for missing values
    train_missing = train_data.isnull().sum()
    test_missing = test_data.isnull().sum()
    
    if train_missing.any():
        logger.warning(f"Missing values in training data: {train_missing[train_missing > 0].to_dict()}")
    
    if test_missing.any():
        logger.warning(f"Missing values in test data: {test_missing[test_missing > 0].to_dict()}")
    
    # Check data types
    logger.info("Data validation completed successfully")
    
    return train_data, test_data

@step
def feature_engineering_step(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: FeatureEngineeringStepConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs feature engineering on the datasets.
    
    Args:
        train_data: Training dataset
        test_data: Test dataset
        config: Feature engineering configuration
        
    Returns:
        Tuple of (processed_train_data, processed_test_data, feature_metadata)
    """
    logger.info("Starting feature engineering...")
    
    def engineer_features(df):
        """Apply feature engineering transformations."""
        df_processed = df.copy()
        
        # Handle categorical features
        for col in config.categorical_features:
            if col in df_processed.columns:
                df_processed[col] = pd.Categorical(df_processed[col]).codes
        
        # Create new features
        if 'tenure' in df_processed.columns and 'monthly_charges' in df_processed.columns:
            df_processed['tenure_monthly_charges_ratio'] = df_processed['tenure'] / (df_processed['monthly_charges'] + 1)
        
        if 'total_charges' in df_processed.columns and 'monthly_charges' in df_processed.columns:
            df_processed['avg_monthly_charges'] = df_processed['total_charges'] / (df_processed['tenure'] + 1)
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(df_processed[numeric_columns].median())
        
        return df_processed
    
    # Apply feature engineering
    train_processed = engineer_features(train_data)
    test_processed = engineer_features(test_data)
    
    # Create feature metadata
    feature_metadata = pd.DataFrame({
        'feature_name': train_processed.columns.tolist(),
        'data_type': [str(dtype) for dtype in train_processed.dtypes],
        'null_count': train_processed.isnull().sum().tolist()
    })
    
    logger.info(f"Feature engineering completed. Created {len(train_processed.columns)} features")
    
    return train_processed, test_processed, feature_metadata

@step
def model_training_step(
    train_data: pd.DataFrame,
    config: ModelTrainingStepConfig,
) -> Any:
    """
    Trains the churn prediction model.
    
    Args:
        train_data: Training dataset
        config: Model training configuration
        
    Returns:
        Trained model
    """
    logger.info("Starting model training...")
    
    # Prepare features and target
    X = train_data.drop(columns=['churn'])
    y = train_data['churn']
    
    # Initialize model based on configuration
    if config.model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=-1
        )
    elif config.model_type == "logistic_regression":
        model = LogisticRegression(
            random_state=config.random_state,
            max_iter=1000
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
    # Train model
    model.fit(X, y)
    
    # Log model with MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="churn_prediction_model"
    )
    
    logger.info(f"Model training completed using {config.model_type}")
    
    return model

@step
def model_evaluation_step(
    model: Any,
    test_data: pd.DataFrame,
    config: ModelEvaluationStepConfig,
) -> Dict[str, Any]:
    """
    Evaluates the trained model performance.
    
    Args:
        model: Trained model
        test_data: Test dataset
        config: Model evaluation configuration
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Starting model evaluation...")
    
    # Prepare test data
    X_test = test_data.drop(columns=['churn'])
    y_test = test_data['churn']
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Prepare evaluation results
    evaluation_results = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': report
    }
    
    # Log metrics with MLflow
    mlflow.log_metrics({
        'accuracy': evaluation_results['accuracy'],
        'precision': evaluation_results['precision'],
        'recall': evaluation_results['recall'],
        'f1_score': evaluation_results['f1_score'],
        'roc_auc': evaluation_results['roc_auc']
    })
    
    # Model validation
    model_approved = (
        evaluation_results['accuracy'] >= config.min_accuracy and
        evaluation_results['roc_auc'] >= config.min_roc_auc
    )
    
    evaluation_results['model_approved'] = model_approved
    
    logger.info(f"Model evaluation completed. ROC-AUC: {roc_auc:.4f}, Accuracy: {evaluation_results['accuracy']:.4f}")
    
    if not model_approved:
        logger.warning("Model did not meet minimum performance criteria")
    
    return evaluation_results

@step
def model_deployment_step(
    model: Any,
    evaluation_results: Dict[str, Any],
) -> Optional[MLFlowDeploymentService]:
    """
    Deploys the model if it meets performance criteria.
    
    Args:
        model: Trained model
        evaluation_results: Model evaluation results
        
    Returns:
        Deployment service if deployed, None otherwise
    """
    logger.info("Starting model deployment...")
    
    if not evaluation_results['model_approved']:
        logger.warning("Model deployment skipped due to insufficient performance")
        return None
    
    # Deploy model to staging
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    
    deployment_service = model_deployer.deploy_model(
        model=model,
        deployment_config={
            'name': 'churn-prediction-staging',
            'description': 'Churn prediction model deployed to staging',
            'stage': 'staging'
        }
    )
    
    logger.info(f"Model deployed successfully to staging environment")
    
    return deployment_service

# Training Pipeline Definition
@pipeline(enable_cache=True)
def training_pipeline(
    data_ingestion_config: DataIngestionStepConfig,
    feature_engineering_config: FeatureEngineeringStepConfig,
    model_training_config: ModelTrainingStepConfig,
    model_evaluation_config: ModelEvaluationStepConfig,
):
    """
    Complete training pipeline for churn prediction model.
    
    Args:
        data_ingestion_config: Data ingestion configuration
        feature_engineering_config: Feature engineering configuration
        model_training_config: Model training configuration
        model_evaluation_config: Model evaluation configuration
    """
    
    # Data ingestion
    train_data, test_data = data_ingestion_step(data_ingestion_config)
    
    # Data validation
    validated_train_data, validated_test_data = data_validation_step(train_data, test_data)
    
    # Feature engineering
    processed_train_data, processed_test_data, feature_metadata = feature_engineering_step(
        validated_train_data, 
        validated_test_data,
        feature_engineering_config
    )
    
    # Model training
    trained_model = model_training_step(processed_train_data, model_training_config)
    
    # Model evaluation
    evaluation_results = model_evaluation_step(
        trained_model, 
        processed_test_data,
        model_evaluation_config
    )
    
    # Model deployment
    deployment_service = model_deployment_step(trained_model, evaluation_results)
    
    return deployment_service

# Pipeline runner function
def run_training_pipeline():
    """
    Run the training pipeline with default configurations.
    """
    
    # Load configurations
    config_path = "configs/train_config.yaml"
    config = load_config(config_path) if os.path.exists(config_path) else {}
    
    # Create step configurations
    data_ingestion_config = DataIngestionStepConfig(
        data_path=config.get('data_path', 'data/raw/Customer_Churn.csv'),
        test_size=config.get('test_size', 0.2)
    )
    
    feature_engineering_config = FeatureEngineeringStepConfig(
        target_column=config.get('target_column', 'churn'),
        k_features=config.get('k_features', 15)
    )
    
    model_training_config = ModelTrainingStepConfig(
        model_type=config.get('model_type', 'random_forest'),
        n_estimators=config.get('n_estimators', 100)
    )
    
    model_evaluation_config = ModelEvaluationStepConfig(
        min_accuracy=config.get('min_accuracy', 0.75),
        min_roc_auc=config.get('min_roc_auc', 0.80)
    )
    
    # Run pipeline
    with mlflow.start_run():
        pipeline_run = training_pipeline(
            data_ingestion_config=data_ingestion_config,
            feature_engineering_config=feature_engineering_config,
            model_training_config=model_training_config,
            model_evaluation_config=model_evaluation_config
        )
        
        logger.info("Training pipeline completed successfully")
        return pipeline_run

if __name__ == "__main__":
    run_training_pipeline()