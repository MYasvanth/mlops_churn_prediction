import mlflow
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from src.utils.config_loader import load_config

config = load_config()
mlflow_config = config.get("mlflow", {})
tracking_uri = mlflow_config.get("tracking_uri", "file:./mlruns")
experiment_name = mlflow_config.get("experiment_name", "default_experiment")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

class ModelTrainer:
    def __init__(self, model_type=None):
        self.config = load_config()
        self.model_type = model_type or self.config.get("model_training", {}).get("model_type", "xgboost")
        self.model = self._get_model()

    def _get_model(self):
        """Get model based on configuration."""
        model_config = self.config.get("model_training", {})
        hyperparams = model_config.get("hyperparameters", {})
        
        if self.model_type == "xgboost":
            params = hyperparams.get("xgboost", {})
            return xgb.XGBClassifier(**params)
        elif self.model_type == "lightgbm":
            params = hyperparams.get("lightgbm", {})
            return LGBMClassifier(**params)
        elif self.model_type == "random_forest":
            params = hyperparams.get("random_forest", {})
            return RandomForestClassifier(**params)
        elif self.model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X, y):
        """Train the model and log to MLflow."""
        if mlflow.active_run() is not None:
            mlflow.end_run()
            
        with mlflow.start_run():
            # Log model type and parameters
            mlflow.log_param("model_type", self.model_type)
            
            # Log hyperparameters based on model type
            model_config = self.config.get("model_training", {})
            hyperparams = model_config.get("hyperparameters", {})
            if self.model_type in hyperparams:
                for key, value in hyperparams[self.model_type].items():
                    mlflow.log_param(key, value)
            
            # Train model
            self.model.fit(X, y)
            
            # Log training metrics
            accuracy = self.model.score(X, y)
            mlflow.log_metric("training_accuracy", accuracy)
            
            # Log the model with signature and input example
            from src.utils.mlflow_utils import log_model
            import numpy as np
            input_example = np.array(X[:5])
            log_model(self.model, name="model", input_example=input_example)
            
        return self.model
