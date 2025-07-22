"""
Configuration loader utilities for the Customer Churn MLOps project.
Handles loading and validation of YAML configuration files.
"""

import os
from omegaconf import DictConfig, OmegaConf
from typing import Any
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from src.utils.logger import get_logger
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Configuration loader using Hydra"""
    
    @staticmethod
    def load_config(config_path: str = None) -> DictConfig:
        """Load configuration using Hydra"""
        if config_path:
            return OmegaConf.load(config_path)
        return hydra.compose(config_name="config")
    
    @staticmethod
    def save_config(cfg: DictConfig, path: str) -> None:
        """Save configuration to file"""
        OmegaConf.save(cfg, path)
        
    @staticmethod
    def merge_configs(base_cfg: DictConfig, override_cfg: DictConfig) -> DictConfig:
        """Merge two configurations"""
        return OmegaConf.merge(base_cfg, override_cfg)


@dataclass
class DataConfig:
    """Data configuration parameters."""
    raw_data_path: str
    processed_data_path: str
    validation_rules: Dict[str, Any]


@dataclass
class FeatureConfig:
    """Feature engineering configuration parameters."""
    categorical_features: list
    numerical_features: list
    binary_features: list
    target_column: str
    encoding_method: str
    scaling_method: str


@dataclass
class ModelConfig:
    """Model training configuration parameters."""
    model_type: str
    test_size: float
    random_state: int
    hyperparameters: Dict[str, Any]


@dataclass
class MLflowConfig:
    """MLflow configuration parameters."""
    tracking_uri: str
    experiment_name: str
    model_registry_name: str


@dataclass
class OptunaConfig:
    """Optuna configuration parameters."""
    n_trials: int
    direction: str
    metric: str


@dataclass
class MonitoringConfig:
    """Monitoring configuration parameters."""
    data_drift_threshold: float
    model_performance_threshold: float
    alert_email: str


class ConfigLoader:
    """Handles loading and validation of configuration files."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.logger = get_logger(self.__class__.__name__)
        
    def load_config(self, config_file: str) -> DictConfig:
        """
        Load configuration using OmegaConf.

        Args:
            config_file: Name of the configuration file

        Returns:
            DictConfig containing configuration parameters
        """
        config_path = self.config_dir / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            config = OmegaConf.load(config_path)
            self.logger.info(f"Successfully loaded configuration from {config_path}")
            return config

        except Exception as e:
            self.logger.error(f"Unexpected error loading config {config_path}: {e}")
            raise

    def load_params(self, params_file: str = "params.yaml") -> DictConfig:
        """Load parameters from params.yaml file using OmegaConf."""
        params_path = Path(params_file)

        if not params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_path}")

        try:
            params = OmegaConf.load(params_path)
            self.logger.info(f"Successfully loaded parameters from {params_path}")
            return params

        except Exception as e:
            self.logger.error(f"Unexpected error loading parameters {params_path}: {e}")
            raise
    
    def get_data_config(self, params: Dict[str, Any]) -> DataConfig:
        """Extract data configuration from parameters."""
        data_params = params.get("data_ingestion", {})
        validation_params = params.get("data_validation", {})
        
        return DataConfig(
            raw_data_path=data_params.get("raw_data_path", "data/raw/Customer_data.csv"),
            processed_data_path=data_params.get("processed_data_path", "data/processed/raw_data.csv"),
            validation_rules=validation_params.get("validation_rules", {})
        )
    
    def get_feature_config(self, params: Dict[str, Any]) -> FeatureConfig:
        """Extract feature engineering configuration from parameters."""
        feature_params = params.get("feature_engineering", {})
        
        return FeatureConfig(
            categorical_features=feature_params.get("categorical_features", []),
            numerical_features=feature_params.get("numerical_features", []),
            binary_features=feature_params.get("binary_features", []),
            target_column=feature_params.get("target_column", "Churn"),
            encoding_method=feature_params.get("encoding_method", "label_encoding"),
            scaling_method=feature_params.get("scaling_method", "standard")
        )
    
    def get_model_config(self, params: Dict[str, Any]) -> ModelConfig:
        """Extract model training configuration from parameters."""
        model_params = params.get("model_training", {})
        
        return ModelConfig(
            model_type=model_params.get("model_type", "xgboost"),
            test_size=model_params.get("test_size", 0.2),
            random_state=model_params.get("random_state", 42),
            hyperparameters=model_params.get("hyperparameters", {})
        )
    
    def get_mlflow_config(self, params: Dict[str, Any]) -> MLflowConfig:
        """Extract MLflow configuration from parameters."""
        mlflow_params = params.get("mlflow", {})
        
        return MLflowConfig(
            tracking_uri=mlflow_params.get("tracking_uri", "mlruns"),
            experiment_name=mlflow_params.get("experiment_name", "customer_churn_prediction"),
            model_registry_name=mlflow_params.get("model_registry_name", "churn_model")
        )
    
    def get_optuna_config(self, params: Dict[str, Any]) -> OptunaConfig:
        """Extract Optuna configuration from parameters."""
        optuna_params = params.get("optuna", {})
        
        return OptunaConfig(
            n_trials=optuna_params.get("n_trials", 100),
            direction=optuna_params.get("direction", "maximize"),
            metric=optuna_params.get("metric", "f1_score")
        )
    
    def get_monitoring_config(self, params: Dict[str, Any]) -> MonitoringConfig:
        """Extract monitoring configuration from parameters."""
        monitoring_params = params.get("monitoring", {})
        
        return MonitoringConfig(
            data_drift_threshold=monitoring_params.get("data_drift_threshold", 0.1),
            model_performance_threshold=monitoring_params.get("model_performance_threshold", 0.8),
            alert_email=monitoring_params.get("alert_email", "admin@company.com")
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        required_sections = ["data_ingestion", "feature_engineering", "model_training"]
        
        for section in required_sections:
            if section not in config:
                self.logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Additional validation logic can be added here
        return True


# Global configuration loader instance
config_loader = ConfigLoader()


def load_config(config_file: str = "params.yaml") -> DictConfig:
    """
    Convenience function to load configuration.

    Args:
        config_file: Configuration file name

    Returns:
        DictConfig configuration object
    """
    return config_loader.load_params(config_file)