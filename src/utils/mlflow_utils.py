import mlflow
from src.utils.config_loader import load_config
from omegaconf import OmegaConf
from pathlib import Path

# Load MLflow configuration from config file
try:
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        config = OmegaConf.load(config_path)
        mlflow_config = config.get("mlflow", {})
        tracking_uri = mlflow_config.get("tracking_uri", "file:./mlruns")
        experiment_name = mlflow_config.get("experiment_name", "default_experiment")
    else:
        # Fallback to default values
        tracking_uri = "file:./mlruns"
        experiment_name = "default_experiment"
except Exception as e:
    # Fallback to default values if config loading fails
    tracking_uri = "file:./mlruns"
    experiment_name = "default_experiment"

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

def start_mlflow_run(run_name=None):
    """Start an MLflow run with an optional run name."""
    # End any active run before starting a new one
    if mlflow.active_run() is not None:
        mlflow.end_run()
    return mlflow.start_run(run_name=run_name)

def log_params(params: dict):
    """Log multiple parameters to MLflow."""
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_metrics(metrics: dict):
    """Log multiple metrics to MLflow."""
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

from mlflow.models.signature import infer_signature

def log_model(model, name="model", input_example=None):
    """Log a model to MLflow with signature and input example."""
    import logging
    logger = logging.getLogger(__name__)
    signature = None
    if input_example is not None:
        try:
            signature = infer_signature(input_example, model.predict(input_example))
            logger.info(f"Inferred signature: {signature}")
        except Exception as e:
            logger.warning(f"Failed to infer signature: {e}")
            signature = None
    logger.info(f"Logging model with name={name}, input_example={input_example is not None}, signature={signature is not None}")
    mlflow.sklearn.log_model(model, name=name, signature=signature, input_example=input_example)
