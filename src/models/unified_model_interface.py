"""
Unified Model Interface for Churn Prediction
This module provides a consistent interface for all model types
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
import mlflow
from pathlib import Path
import logging

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass

class UnifiedModelTrainer:
    """Unified model trainer supporting all model types."""
    
    def __init__(self, model_type: str, config_path: str = None):
        self.model_type = model_type.lower()
        self.config = load_config(config_path or "configs/model/unified_model_config.yaml")
        self.model = None
        self.model_config = self._get_model_config()
        
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration based on model type."""
        model_configs = self.config.get("models", {}).get("model_configs", {})
        if self.model_type not in model_configs:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model_configs[self.model_type]
    
    def _create_model(self, params: Dict[str, Any] = None) -> Any:
        """Create model instance with given parameters."""
        if params is None:
            params = self.model_config.get("default_params", {})
        
        if self.model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(**params)
        elif self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params)
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)
        elif self.model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)
        elif self.model_type == "svm":
            from sklearn.svm import SVC
            return SVC(**params, probability=True)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              hyperparameters: Dict[str, Any] = None) -> Any:
        """Train the model with optional hyperparameters."""
        try:
            # Create model with hyperparameters if provided
            if hyperparameters:
                self.model = self._create_model(hyperparameters)
            else:
                self.model = self._create_model()
            
            # Train model
            self.model.fit(X, y)
            
            logger.info(f"Successfully trained {self.model_type} model")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training {self.model_type} model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            predictions = self.model.predict(X)
            # Create dummy probabilities
            proba = np.zeros((len(predictions), 2))
            proba[:, 0] = 1 - predictions
            proba[:, 1] = predictions
            return proba
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.model_type,
            "model_config": self.model_config,
            "is_trained": self.model is not None,
            "model_class": type(self.model).__name__ if self.model else None
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")

class UnifiedModelEvaluator:
    """Unified model evaluator for all model types."""
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path or "configs/model/unified_model_config.yaml")
        self.evaluation_config = self.config.get("evaluation", {})
        
    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='weighted'),
            "recall": recall_score(y, y_pred, average='weighted'),
            "f1_score": f1_score(y, y_pred, average='weighted'),
            "roc_auc": roc_auc_score(y, y_pred_proba)
        }
        
        return metrics
    
    def validate_performance(self, metrics: Dict[str, float]) -> bool:
        """Validate if model meets performance thresholds."""
        thresholds = self.evaluation_config.get("thresholds", {})
        
        for metric, value in metrics.items():
            min_threshold = thresholds.get(f"min_{metric}")
            if min_threshold and value < min_threshold:
                logger.warning(f"Model failed {metric} threshold: {value} < {min_threshold}")
                return False
        
        return True

class UnifiedModelLoader:
    """Unified model loader for all model types."""
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path or "configs/model/unified_model_config.yaml")
        
    def load_model(self, model_path: str, model_type: str = None) -> Any:
        """Load model from file."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        
        if model_type and model_type != type(model).__name__.lower():
            logger.warning(f"Model type mismatch: expected {model_type}, got {type(model).__name__}")
        
        return model
    
    def load_latest_model(self, stage: str = "production") -> Tuple[Any, str]:
        """Load the latest model from registry."""
        # This will be integrated with unified registry
        model_dir = Path("models") / stage
        model_files = list(model_dir.glob("**/model.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"No models found in {stage} stage")
        
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        model = joblib.load(latest_model)
        
        return model, str(latest_model)
I f   s u b m i t t e d ,   t h i s   r e p o r t   w i l l   b e   u s e d   b y   c o r e   m a i n t a i n e r s   t o   i m p r o v e  
 f u t u r e   r e l e a s e s   o f   c o n d a .  
 