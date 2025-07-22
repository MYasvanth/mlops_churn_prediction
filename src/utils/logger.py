# logger.py
"""
Logging utilities for the Customer Churn MLOps project.
Provides centralized logging configuration with different levels and formats.
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

def get_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Create and configure a logger with both file and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(
            log_path / log_file,
            mode='a',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_mlflow_logging():
    """Setup MLflow logging configuration."""
    import mlflow
    
    # Suppress MLflow INFO logs
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("mlflow.tracking").setLevel(logging.WARNING)
    logging.getLogger("mlflow.utils").setLevel(logging.WARNING)


def log_system_info(logger: logging.Logger):
    """Log system information for debugging purposes."""
    import platform
    import psutil
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info("===========================")


class MLflowLogger:
    """Wrapper class for MLflow logging with additional utilities."""
    
    def __init__(self, experiment_name: str):
        import mlflow
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.logger = get_logger(self.__class__.__name__)
    
    def log_params(self, params: dict):
        """Log parameters to MLflow."""
        import mlflow
        mlflow.log_params(params)
        self.logger.info(f"Logged parameters: {params}")
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to MLflow."""
        import mlflow
        mlflow.log_metrics(metrics, step=step)
        self.logger.info(f"Logged metrics: {metrics}")
    
    def log_artifact(self, artifact_path: str):
        """Log artifact to MLflow."""
        import mlflow
        mlflow.log_artifact(artifact_path)
        self.logger.info(f"Logged artifact: {artifact_path}")
    
    def log_model(self, model, model_name: str, **kwargs):
        """Log model to MLflow."""
        import mlflow
        mlflow.sklearn.log_model(model, model_name, **kwargs)
        self.logger.info(f"Logged model: {model_name}")


# Global logger instance
main_logger = get_logger(
    "mlops_churn_prediction",
    log_file=f"mlops_{datetime.now().strftime('%Y%m%d')}.log"
)