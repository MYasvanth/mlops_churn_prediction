# model_registry.py
"""
Model registry for managing trained models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from pathlib import Path
import json
import shutil
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

from ..utils.logger import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)

@dataclass
class ModelMetadata:
    """Data class for model metadata."""
    model_name: str
    model_version: str
    model_type: str
    training_date: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    model_path: str
    stage: str  # staging, production, archived
    created_by: str
    description: str

class ModelRegistry:
    """
    Model registry for managing trained models throughout their lifecycle.
    """
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize the ModelRegistry.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = load_config(config_path)
        self.registry_config = self.config.get('model_registry', {})
        
        # Set up directories
        self.models_dir = Path("models")
        self.staging_dir = self.models_dir / "staging"
        self.production_dir = self.models_dir / "production"
        self.archived_dir = self.models_dir / "archived"
        
        # Create directories
        for directory in [self.staging_dir, self.production_dir, self.archived_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Registry file
        self.registry_file = self.models_dir / "model_registry.json"
        
        # Initialize MLflow client
        self.mlflow_client = MlflowClient()
        
        # Load existing registry
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """
        Load the model registry from file.
        
        Returns:
            Dict[str, Any]: Registry data
        """
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading registry: {str(e)}")
                return {"models": {}}
        else:
            return {"models": {}}
    
    def _save_registry(self):
        """Save the model registry to file."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")
    
    def register_model(self, model: Any, metadata: ModelMetadata, 
                      overwrite: bool = False) -> str:
        """
        Register a new model.
        
        Args:
            model: Trained model object
            metadata (ModelMetadata): Model metadata
            overwrite (bool): Whether to overwrite existing model
            
        Returns:
            str: Model ID
        """
        try:
            # Generate model ID
            model_id = f"{metadata.model_name}_{metadata.model_version}"
            
            # Check if model already exists
            if model_id in self.registry["models"] and not overwrite:
                raise ValueError(f"Model {model_id} already exists. Use overwrite=True to replace.")
            
            # Determine stage directory
            stage_dir = self._get_stage_directory(metadata.stage)
            
            # Create model directory
            model_dir = stage_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
            
            # Update metadata with actual path
            metadata.model_path = str(model_path)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            # Register with MLflow
            self._register_with_mlflow(model, metadata, model_id)
            
            # Update registry
            self.registry["models"][model_id] = asdict(metadata)
            self._save_registry()
            
            logger.info(f"Model {model_id} registered successfully in {metadata.stage} stage")
            return model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def _get_stage_directory(self, stage: str) -> Path:
        """
        Get directory for a specific stage.
        
        Args:
            stage (str): Model stage
            
        Returns:
            Path: Stage directory
        """
        if stage == "staging":
            return self.staging_dir
        elif stage == "production":
            return self.production_dir
        elif stage == "archived":
            return self.archived_dir
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def _register_with_mlflow(self, model: Any, metadata: ModelMetadata, model_id: str):
        """
        Register model with MLflow.
        
        Args:
            model: Trained model
            metadata (ModelMetadata): Model metadata
            model_id (str): Model ID
        """
        try:
            # Log model with MLflow
            with mlflow.start_run():
                # Log parameters
                for param_name, param_value in metadata.hyperparameters.items():
                    mlflow.log_param(param_name, param_value)
                
                # Log metrics
                for metric_name, metric_value in metadata.performance_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=metadata.model_name
                )
                
                # Add tags
                mlflow.set_tag("model_version", metadata.model_version)
                mlflow.set_tag("model_type", metadata.model_type)
                mlflow.set_tag("stage", metadata.stage)
                mlflow.set_tag("created_by", metadata.created_by)
                
        except Exception as e:
            logger.warning(f"Failed to register with MLflow: {str(e)}")
    
    def load_model(self, model_id: str) -> Any:
        """
        Load a model from the registry.
        
        Args:
            model_id (str): Model ID
            
        Returns:
            Any: Loaded model
        """
        try:
            if model_id not in self.registry["models"]:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.registry["models"][model_id]
            model_path = metadata["model_path"]
            
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = joblib.load(model_path)
            logger.info(f"Model {model_id} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            raise
    
    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """
        Get metadata for a model.
        
        Args:
            model_id (str): Model ID
            
        Returns:
            ModelMetadata: Model metadata
        """
        if model_id not in self.registry["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata_dict = self.registry["models"][model_id]
        return ModelMetadata(**metadata_dict)
    
    def list_models(self, stage: Optional[str] = None, 
                   model_type: Optional[str] = None) -> List[str]:
        """
        List models in the registry.
        
        Args:
            stage (Optional[str]): Filter by stage
            model_type (Optional[str]): Filter by model type
            
        Returns:
            List[str]: List of model IDs
        """
        models = []
        
        for model_id, metadata in self.registry["models"].items():
            if stage and metadata["stage"] != stage:
                continue
            if model_type and metadata["model_type"] != model_type:
                continue
            models.append(model_id)
        
        return models
    
    def get_latest_model(self, model_name: str, stage: str = "production") -> Optional[str]:
        """
        Get the latest model for a given name and stage.
        
        Args:
            model_name (str): Model name
            stage (str): Model stage
            
        Returns:
            Optional[str]: Latest model ID or None
        """
        models = self.list_models(stage=stage)
        model_versions = []
        
        for model_id in models:
            metadata = self.registry["models"][model_id]
            if metadata["model_name"] == model_name:
                model_versions.append((model_id, metadata["training_date"]))
        
        if not model_versions:
            return None
        
        # Sort by training date and return latest
        model_versions.sort(key=lambda x: x[1], reverse=True)
        return model_versions[0][0]
    
    def promote_model(self, model_id: str, target_stage: str) -> bool:
        """
        Promote a model to a different stage.
        
        Args:
            model_id (str): Model ID
            target_stage (str): Target stage
            
        Returns:
            bool: True if successful
        """
        try:
            if model_id not in self.registry["models"]:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.registry["models"][model_id]
            current_stage = metadata["stage"]
            
            if current_stage == target_stage:
                logger.info(f"Model {model_id} is already in {target_stage} stage")
                return True
            
            # Get directories
            current_dir = self._get_stage_directory(current_stage)
            target_dir = self._get_stage_directory(target_stage)
            
            # Move model files
            current_model_dir = current_dir / model_id
            target_model_dir = target_dir / model_id
            
            if current_model_dir.exists():
                shutil.move(str(current_model_dir), str(target_model_dir))
                
                # Update metadata
                metadata["stage"] = target_stage
                metadata["model_path"] = str(target_model_dir / "model.pkl")
                
                # Update registry
                self.registry["models"][model_id] = metadata
                self._save_registry()
                
                logger.info(f"Model {model_id} promoted from {current_stage} to {target_stage}")
                return True
            else:
                logger.error(f"Model directory not found: {current_model_dir}")
                return False
                
        except Exception as e:
            logger.error(f"Error promoting model {model_id}: {str(e)}")
            return False
    
    def archive_model(self, model_id: str) -> bool:
        """
        Archive a model.
        
        Args:
            model_id (str): Model ID
            
        Returns:
            bool: True if successful
        """
        return self.promote_model(model_id, "archived")
    
    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id (str): Model ID
            force (bool): Force deletion even if in production
            
        Returns:
            bool: True if successful
        """
        try:
            if model_id not in self.registry["models"]:
                raise ValueError(f"Model {model_id} not found in registry")
            
            metadata = self.registry["models"][model_id]
            
            # Check if model is in production
            if metadata["stage"] == "production" and not force:
                raise ValueError(f"Cannot delete production model {model_id}. Use force=True to override.")
            
            # Remove model files
            model_path = Path(metadata["model_path"])
            model_dir = model_path.parent
            
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Remove from registry
            del self.registry["models"][model_id]
            self._save_registry()
            
            logger.info(f"Model {model_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {str(e)}")
            return False
    
    def compare_models(self, model_ids: List[str], 
                      metrics: List[str] = ["accuracy", "precision", "recall", "f1_score"]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            model_ids (List[str]): List of model IDs to compare
            metrics (List[str]): List of metrics to compare
            
        Returns:
            pd.DataFrame: Comparison results
        """
        comparison_data = []
        
        for model_id in model_ids:
            if model_id not in self.registry["models"]:
                logger.warning(f"Model {model_id} not found in registry")
                continue
            
            metadata = self.registry["models"][model_id]
            row = {
                "model_id": model_id,
                "model_name": metadata["model_name"],
                "model_version": metadata["model_version"],
                "model_type": metadata["model_type"],
                "stage": metadata["stage"],
                "training_date": metadata["training_date"]
            }
            
            # Add metrics
            for metric in metrics:
                row[metric] = metadata["performance_metrics"].get(metric, None)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_model_performance_history(self, model_name: str) -> pd.DataFrame:
        """
        Get performance history for a model.
        
        Args:
            model_name (str): Model name
            
        Returns:
            pd.DataFrame: Performance history
        """
        history_data = []
        
        for model_id, metadata in self.registry["models"].items():
            if metadata["model_name"] == model_name:
                row = {
                    "model_id": model_id,
                    "model_version": metadata["model_version"],
                    "training_date": metadata["training_date"],
                    "stage": metadata["stage"]
                }
                
                # Add all metrics
                for metric_name, metric_value in metadata["performance_metrics"].items():
                    row[metric_name] = metric_value
                
                history_data.append(row)
        
        df = pd.DataFrame(history_data)
        if not df.empty:
            df = df.sort_values("training_date")
        
        return df
    
    def cleanup_old_models(self, keep_last_n: int = 5) -> int:
        """
        Clean up old models, keeping only the last N versions.
        
        Args:
            keep_last_n (int): Number of versions to keep
            
        Returns:
            int: Number of models deleted
        """
        deleted_count = 0
        
        # Group models by name
        model_groups = {}
        for model_id, metadata in self.registry["models"].items():
            model_name = metadata["model_name"]
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append((model_id, metadata["training_date"]))
        
        # Clean up each group
        for model_name, models in model_groups.items():
            # Sort by training date
            models.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the last N models (excluding production models)
            models_to_delete = []
            non_production_count = 0
            
            for model_id, _ in models:
                metadata = self.registry["models"][model_id]
                if metadata["stage"] != "production":
                    non_production_count += 1
                    if non_production_count > keep_last_n:
                        models_to_delete.append(model_id)
            
            # Delete old models
            for model_id in models_to_delete:
                if self.delete_model(model_id):
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old models")
        return deleted_count
    
    def export_registry(self, output_path: str):
        """
        Export the model registry to a file.
        
        Args:
            output_path (str): Output file path
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
            logger.info(f"Registry exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting registry: {str(e)}")
            raise
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the model registry.
        
        Returns:
            Dict[str, Any]: Registry statistics
        """
        stats = {
            "total_models": len(self.registry["models"]),
            "by_stage": {},
            "by_model_type": {},
            "by_model_name": {}
        }
        
        for model_id, metadata in self.registry["models"].items():
            # Count by stage
            stage = metadata["stage"]
            stats["by_stage"][stage] = stats["by_stage"].get(stage, 0) + 1
            
            # Count by model type
            model_type = metadata["model_type"]
            stats["by_model_type"][model_type] = stats["by_model_type"].get(model_type, 0) + 1
            
            # Count by model name
            model_name = metadata["model_name"]
            stats["by_model_name"][model_name] = stats["by_model_name"].get(model_name, 0) + 1
        
        return stats