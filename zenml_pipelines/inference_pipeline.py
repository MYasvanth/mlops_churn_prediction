# inference_pipeline.py
"""
Inference Pipeline for Churn Prediction MLOps

This pipeline handles batch inference for churn prediction including:
- Data loading and preprocessing
- Model loading from registry
- Batch prediction generation
- Results storage and monitoring
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import mlflow
import mlflow.sklearn
from zenml import step, pipeline
from zenml.client import Client
from zenml.logger import get_logger
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import BaseStepConfig

# Import custom modules
from src.data.data_loader import DataLoader
from src.features.preprocessing import Preprocessor
from src.models.model_registry import ModelRegistry
from src.monitoring.data_drift_monitor import DataDriftMonitor
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

# Step Configurations
class InferenceDataLoadingConfig(BaseStepConfig):
    """Configuration for inference data loading step."""
    input_data_path: str = "data/raw/inference_data.csv"
    batch_size: int = 1000
    preprocessing_config: Dict[str, Any] = {}

class ModelLoadingConfig(BaseStepConfig):
    """Configuration for model loading step."""
    model_name: str = "churn_prediction_model"
    model_stage: str = "production"
    model_version: Optional[str] = None

class PredictionConfig(BaseStepConfig):
    """Configuration for prediction step."""
    prediction_threshold: float = 0.5
    include_probabilities: bool = True
    output_path: str = "data/predictions/"

class ResultsStorageConfig(BaseStepConfig):
    """Configuration for results storage step."""
    storage_format: str = "csv"  # csv, parquet, json
    include_metadata: bool = True
    archive_predictions: bool = True

# Pipeline Steps
@step
def load_inference_data_step(
    config: InferenceDataLoadingConfig,
) -> pd.DataFrame:
    """
    Loads data for batch inference.
    
    Args:
        config: Configuration for data loading
        
    Returns:
        DataFrame containing inference data
    """
    logger.info("Loading inference data...")
    
    if not os.path.exists(config.input_data_path):
        raise FileNotFoundError(f"Inference data file not found: {config.input_data_path}")
    
    # Load data
    df = pd.read_csv(config.input_data_path)
    
    if df.empty:
        raise ValueError("Empty inference dataset provided")
    
    logger.info(f"Loaded {len(df)} rows for inference from {config.input_data_path}")
    
    return df

@step
def preprocess_inference_data_step(
    raw_data: pd.DataFrame,
    config: InferenceDataLoadingConfig,
) -> pd.DataFrame:
    """
    Preprocesses inference data to match training data format.
    
    Args:
        raw_data: Raw inference data
        config: Configuration for preprocessing
        
    Returns:
        Preprocessed data ready for inference
    """
    logger.info("Preprocessing inference data...")
    
    def apply_preprocessing(df):
        """Apply the same preprocessing as training data."""
        df_processed = df.copy()
        
        # Handle categorical features (same as training)
        categorical_features = ["gender", "contract_type", "payment_method"]
        for col in categorical_features:
            if col in df_processed.columns:
                df_processed[col] = pd.Categorical(df_processed[col]).codes
        
        # Create engineered features (same as training)
        if 'tenure' in df_processed.columns and 'monthly_charges' in df_processed.columns:
            df_processed['tenure_monthly_charges_ratio'] = df_processed['tenure'] / (df_processed['monthly_charges'] + 1)
        
        if 'total_charges' in df_processed.columns and 'monthly_charges' in df_processed.columns:
            df_processed['avg_monthly_charges'] = df_processed['total_charges'] / (df_processed['tenure'] + 1)
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(df_processed[numeric_columns].median())
        
        return df_processed
    
    processed_data = apply_preprocessing(raw_data)
    
    logger.info(f"Preprocessing completed. Shape: {processed_data.shape}")
    
    return processed_data

@step
def load_model_step(
    config: ModelLoadingConfig,
) -> Any:
    """
    Loads the trained model from MLflow registry.
    
    Args:
        config: Configuration for model loading
        
    Returns:
        Loaded model
    """
    logger.info("Loading model from registry...")
    
    client = mlflow.tracking.MlflowClient()
    
    try:
        if config.model_version:
            model_version = client.get_model_version(
                name=config.model_name,
                version=config.model_version
            )
        else:
            model_version = client.get_latest_versions(
                name=config.model_name,
                stages=[config.model_stage]
            )[0]
        
        model_uri = f"models:/{config.model_name}/{model_version.version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        logger.info(f"Successfully loaded model {config.model_name} version {model_version.version}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@step
def data_drift_detection_step(
    inference_data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Detects data drift in inference data compared to training data.
    
    Args:
        inference_data: Current inference data
        
    Returns:
        Dictionary containing drift detection results
    """
    logger.info("Performing data drift detection...")
    
    # Load reference data (training data)
    reference_data_path = "data/processed/train.csv"
    
    if not os.path.exists(reference_data_path):
        logger.warning("Reference data not found. Skipping drift detection.")
        return {"drift_detected": False, "message": "Reference data not available"}
    
    reference_data = pd.read_csv(reference_data_path)
    
    # Remove target column if present
    if 'churn' in reference_data.columns:
        reference_data = reference_data.drop(columns=['churn'])
    
    # Ensure same columns
    common_columns = list(set(reference_data.columns) & set(inference_data.columns))
    reference_data = reference_data[common_columns]
    inference_data = inference_data[common_columns]
    
    # Simple drift detection (can be enhanced with Evidently)
    drift_results = {
        "drift_detected": False,
        "drift_score": 0.0,
        "drifted_features": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Compare basic statistics
    for col in common_columns:
        if pd.api.types.is_numeric_dtype(reference_data[col]):
            ref_mean = reference_data[col].mean()
            inf_mean = inference_data[col].mean()
            ref_std = reference_data[col].std()
            
            if ref_std > 0:
                drift_score = abs(ref_mean - inf_mean) / ref_std
                if drift_score > 2.0:  # Threshold for drift
                    drift_results["drifted_features"].append({
                        "feature": col,
                        "drift_score": drift_score,
                        "reference_mean": ref_mean,
                        "inference_mean": inf_mean
                    })
    
    if drift_results["drifted_features"]:
        drift_results["drift_detected"] = True
        drift_results["drift_score"] = max([f["drift_score"] for f in drift_results["drifted_features"]])
        logger.warning(f"Data drift detected in {len(drift_results['drifted_features'])} features")
    else:
        logger.info("No significant data drift detected")
    
    return drift_results

@step
def batch_prediction_step(
    model: Any,
    processed_data: pd.DataFrame,
    config: PredictionConfig,
) -> pd.DataFrame:
    """
    Generates batch predictions using the loaded model.
    
    Args:
        model: Loaded model
        processed_data: Preprocessed inference data
        config: Configuration for prediction
        
    Returns:
        DataFrame containing predictions
    """
    logger.info("Generating batch predictions...")
    
    # Make predictions
    predictions = model.predict(processed_data)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'prediction': predictions,
        'prediction_binary': (predictions > config.prediction_threshold).astype(int),
        'timestamp': datetime.now().isoformat()
    })
    
    # Add probabilities if requested
    if config.include_probabilities:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)
            results_df['probability_class_0'] = probabilities[:, 0]
            results_df['probability_class_1'] = probabilities[:, 1]
        else:
            logger.warning("Model does not support probability predictions")
    
    # Add row identifiers if available
    if 'customer_id' in processed_data.columns:
        results_df['customer_id'] = processed_data['customer_id'].values
    else:
        results_df['row_id'] = range(len(results_df))
    
    logger.info(f"Generated {len(results_df)} predictions")
    
    # Log prediction statistics
    churn_predictions = results_df['prediction_binary'].sum()
    total_predictions = len(results_df)
    churn_rate = churn_predictions / total_predictions
    
    logger.info(f"Predicted churn rate: {churn_rate:.2%} ({churn_predictions}/{total_predictions})")
    
    return results_df

@step
def store_predictions_step(
    predictions: pd.DataFrame,
    original_data: pd.DataFrame,
    drift_results: Dict[str, Any],
    config: ResultsStorageConfig,
) -> str:
    """
    Stores prediction results and metadata.
    
    Args:
        predictions: Prediction results
        original_data: Original inference data
        drift_results: Data drift detection results
        config: Configuration for results storage
        
    Returns:
        Path to stored results
    """
    logger.info("Storing prediction results...")
    
    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine predictions with original data if requested
    if config.include_metadata:
        # Add selected original features
        metadata_columns = ['customer_id'] if 'customer_id' in original_data.columns else []
        if metadata_columns:
            for col in metadata_columns:
                if col in original_data.columns:
                    predictions[col] = original_data[col].values
    
    # Save predictions
    output_filename = f"predictions_{timestamp}"
    
    if config.storage_format == "csv":
        output_path = os.path.join(config.output_path, f"{output_filename}.csv")
        predictions.to_csv(output_path, index=False)
    elif config.storage_format == "parquet":
        output_path = os.path.join(config.output_path, f"{output_filename}.parquet")
        predictions.to_parquet(output_path, index=False)
    elif config.storage_format == "json":
        output_path = os.path.join(config.output_path, f"{output_filename}.json")
        predictions.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported storage format: {config.storage_format}")
    
    # Save drift results
    drift_output_path = os.path.join(config.output_path, f"drift_results_{timestamp}.json")
    import json
    with open(drift_output_path, 'w') as f:
        json.dump(drift_results, f, indent=2)
    
    # Save summary statistics
    summary_stats = {
        "total_predictions": len(predictions),
        "churn_predictions": int(predictions['prediction_binary'].sum()),
        "churn_rate": float(predictions['prediction_binary'].mean()),
        "timestamp": datetime.now().isoformat(),
        "drift_detected": drift_results.get("drift_detected", False),
        "drift_score": drift_results.get("drift_score", 0.0)
    }
    
    summary_path = os.path.join(config.output_path, f"summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logger.info(f"Prediction results stored at: {output_path}")
    logger.info(f"Summary statistics stored at: {summary_path}")
    
    return output_path

@step
def prediction_monitoring_step(
    predictions: pd.DataFrame,
    drift_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Monitors prediction quality and triggers alerts if needed.
    
    Args:
        predictions: Prediction results
        drift_results: Data drift detection results
        
    Returns:
        Dictionary containing monitoring results
    """
    logger.info("Monitoring prediction quality...")
    
    monitoring_results = {
        "timestamp": datetime.now().isoformat(),
        "total_predictions": len(predictions),
        "churn_rate": float(predictions['prediction_binary'].mean()),
        "drift_detected": drift_results.get("drift_detected", False),
        "alerts": []
    }
    
    # Check for unusual churn rates
    expected_churn_rate = 0.25  # Expected baseline churn rate
    current_churn_rate = monitoring_results["churn_rate"]
    
    if abs(current_churn_rate - expected_churn_rate) > 0.1:
        alert = {
            "type": "churn_rate_anomaly",
            "message": f"Unusual churn rate detected: {current_churn_rate:.2%} (expected: {expected_churn_rate:.2%})",
            "severity": "warning" if abs(current_churn_rate - expected_churn_rate) < 0.2 else "critical"
        }
        monitoring_results["alerts"].append(alert)
        logger.warning(alert["message"])
    
    # Check for data drift
    if drift_results.get("drift_detected", False):
        alert = {
            "type": "data_drift",
            "message": f"Data drift detected with score: {drift_results.get('drift_score', 0):.2f}",
            "severity": "warning" if drift_results.get('drift_score', 0) < 3.0 else "critical",
            "drifted_features": drift_results.get("drifted_features", [])
        }
        monitoring_results["alerts"].append(alert)
        logger.warning(alert["message"])
    
    # Check prediction confidence
    if 'probability_class_1' in predictions.columns:
        low_confidence_threshold = 0.6
        low_confidence_predictions = (predictions['probability_class_1'] < low_confidence_threshold).sum()
        low_confidence_rate = low_confidence_predictions / len(predictions)
        
        if low_confidence_rate > 0.5:
            alert = {
                "type": "low_confidence",
                "message": f"High rate of low-confidence predictions: {low_confidence_rate:.2%}",
                "severity": "warning"
            }
            monitoring_results["alerts"].append(alert)
            logger.warning(alert["message"])
    
    logger.info(f"Monitoring completed. Generated {len(monitoring_results['alerts'])} alerts")
    
    return monitoring_results

# Inference Pipeline Definition
@pipeline(enable_cache=True)
def inference_pipeline(
    data_loading_config: InferenceDataLoadingConfig,
    model_loading_config: ModelLoadingConfig,
    prediction_config: PredictionConfig,
    results_storage_config: ResultsStorageConfig,
):
    """
    Complete inference pipeline for churn prediction.
    
    Args:
        data_loading_config: Data loading configuration
        model_loading_config: Model loading configuration
        prediction_config: Prediction configuration
        results_storage_config: Results storage configuration
    """
    
    # Load inference data
    raw_data = load_inference_data_step(data_loading_config)
    
    # Preprocess data
    processed_data = preprocess_inference_data_step(raw_data, data_loading_config)
    
    # Load model
    model = load_model_step(model_loading_config)
    
    # Detect data drift
    drift_results = data_drift_detection_step(processed_data)
    
    # Generate predictions
    predictions = batch_prediction_step(model, processed_data, prediction_config)
    
    # Store results
    output_path = store_predictions_step(
        predictions, 
        raw_data, 
        drift_results,
        results_storage_config
    )
    
    # Monitor predictions
    monitoring_results = prediction_monitoring_step(predictions, drift_results)
    
    return output_path, monitoring_results

# Real-time inference step for single predictions
@step
def realtime_prediction_step(
    model: Any,
    input_data: Dict[str, Any],
    config: PredictionConfig,
) -> Dict[str, Any]:
    """
    Generates real-time prediction for a single customer.
    
    Args:
        model: Loaded model
        input_data: Customer data for prediction
        config: Prediction configuration
        
    Returns:
        Dictionary containing prediction results
    """
    logger.info("Generating real-time prediction...")
    
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # Apply preprocessing (same as batch)
    def preprocess_single_record(df):
        df_processed = df.copy()
        
        # Handle categorical features
        categorical_features = ["gender", "contract_type", "payment_method"]
        for col in categorical_features:
            if col in df_processed.columns:
                df_processed[col] = pd.Categorical(df_processed[col]).codes
        
        # Create engineered features
        if 'tenure' in df_processed.columns and 'monthly_charges' in df_processed.columns:
            df_processed['tenure_monthly_charges_ratio'] = df_processed['tenure'] / (df_processed['monthly_charges'] + 1)
        
        if 'total_charges' in df_processed.columns and 'monthly_charges' in df_processed.columns:
            df_processed['avg_monthly_charges'] = df_processed['total_charges'] / (df_processed['tenure'] + 1)
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(0)
        
        return df_processed
    
    processed_df = preprocess_single_record(df)
    
    # Make prediction
    prediction = model.predict(processed_df)[0]
    prediction_binary = int(prediction > config.prediction_threshold)
    
    result = {
        "customer_id": input_data.get("customer_id", "unknown"),
        "prediction": float(prediction),
        "prediction_binary": prediction_binary,
        "churn_risk": "High" if prediction_binary == 1 else "Low",
        "timestamp": datetime.now().isoformat()
    }
    
    # Add probability if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(processed_df)[0]
        result["probability_no_churn"] = float(probabilities[0])
        result["probability_churn"] = float(probabilities[1])
        result["confidence"] = float(max(probabilities))
    
    logger.info(f"Real-time prediction completed: {result['churn_risk']} risk")
    
    return result

# Real-time inference pipeline
@pipeline(enable_cache=False)
def realtime_inference_pipeline(
    model_loading_config: ModelLoadingConfig,
    prediction_config: PredictionConfig,
    input_data: Dict[str, Any],
):
    """
    Real-time inference pipeline for single customer prediction.
    
    Args:
        model_loading_config: Model loading configuration
        prediction_config: Prediction configuration
        input_data: Customer data for prediction
    """
    
    # Load model
    model = load_model_step(model_loading_config)
    
    # Generate prediction
    prediction_result = realtime_prediction_step(model, input_data, prediction_config)
    
    return prediction_result

# Pipeline runner functions
def run_batch_inference_pipeline(input_data_path: str = None):
    """
    Run the batch inference pipeline with default configurations.
    
    Args:
        input_data_path: Path to inference data file
    """
    
    # Load configurations
    config_path = "configs/inference_config.yaml"
    config = load_config(config_path) if os.path.exists(config_path) else {}
    
    # Create step configurations
    data_loading_config = InferenceDataLoadingConfig(
        input_data_path=input_data_path or config.get('input_data_path', 'data/raw/inference_data.csv'),
        batch_size=config.get('batch_size', 1000)
    )
    
    model_loading_config = ModelLoadingConfig(
        model_name=config.get('model_name', 'churn_prediction_model'),
        model_stage=config.get('model_stage', 'production')
    )
    
    prediction_config = PredictionConfig(
        prediction_threshold=config.get('prediction_threshold', 0.5),
        include_probabilities=config.get('include_probabilities', True),
        output_path=config.get('output_path', 'data/predictions/')
    )
    
    results_storage_config = ResultsStorageConfig(
        storage_format=config.get('storage_format', 'csv'),
        include_metadata=config.get('include_metadata', True)
    )
    
    # Run pipeline
    pipeline_run = inference_pipeline(
        data_loading_config=data_loading_config,
        model_loading_config=model_loading_config,
        prediction_config=prediction_config,
        results_storage_config=results_storage_config
    )
    
    logger.info("Batch inference pipeline completed successfully")
    return pipeline_run

def run_realtime_inference(customer_data: Dict[str, Any]):
    """
    Run real-time inference for a single customer.
    
    Args:
        customer_data: Dictionary containing customer features
        
    Returns:
        Dictionary containing prediction results
    """
    
    # Load configurations
    config_path = "configs/inference_config.yaml"
    config = load_config(config_path) if os.path.exists(config_path) else {}
    
    # Create step configurations
    model_loading_config = ModelLoadingConfig(
        model_name=config.get('model_name', 'churn_prediction_model'),
        model_stage=config.get('model_stage', 'production')
    )
    
    prediction_config = PredictionConfig(
        prediction_threshold=config.get('prediction_threshold', 0.5),
        include_probabilities=config.get('include_probabilities', True)
    )
    
    # Run pipeline
    prediction_result = realtime_inference_pipeline(
        model_loading_config=model_loading_config,
        prediction_config=prediction_config,
        input_data=customer_data
    )
    
    logger.info("Real-time inference completed successfully")
    return prediction_result

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "batch":
            input_path = sys.argv[2] if len(sys.argv) > 2 else None
            run_batch_inference_pipeline(input_path)
        elif sys.argv[1] == "realtime":
            # Example customer data
            example_customer = {
                "customer_id": "CUST_001",
                "gender": "Male",
                "tenure": 12,
                "monthly_charges": 79.99,
                "total_charges": 960.0,
                "contract_type": "Month-to-month",
                "payment_method": "Credit card"
            }
            result = run_realtime_inference(example_customer)
            print(f"Prediction result: {result}")
    else:
        # Default: run batch inference
        run_batch_inference_pipeline()