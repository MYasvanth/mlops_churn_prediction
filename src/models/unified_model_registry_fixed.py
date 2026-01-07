"""
Unified Model Registry for Churn Prediction
Handles model registration, versioning, and lifecycle management
"""

import os
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

class UnifiedModelRegistry:
    """Unified model registry for managing all model types."""
    
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path or "configs/model/unified_model_config.yaml")
        self.registry_config = self.config.get("registry", {})
        self.mlflow_client = MlflowClient() if self.registry_config.get("mlflow_integration") else None
        
    def register_model(self, model: Any, metadata: Dict[str, Any], 
                      stage: str = "staging") -> str:
        """Register a trained model with metadata."""
        try:
            # Generate model ID
            model_id = f"{metadata.get('model_type', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save model locally
            model_dir = Path("models") / stage / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata_file = model_dir / "metadata.json"
            full_metadata = {
                "model_id": model_id,
                "model_path": str(model_path),
                "stage": stage,
                "registered_at": datetime.now().isoformat(),
                **metadata
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
            
            logger.info(f"Model registered successfully: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def get_model_info(self, model_id: str, stage: str = None) -> Dict[str, Any]:
        """Get model information by ID, searching across stages if needed."""
        # If stage is specified, check only that stage
        if stage:
            metadata_file = Path("models") / stage / model_id / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            raise FileNotFoundError(f"Model metadata not found: {metadata_file}")
        
        # If stage is not specified, search across all stages
        stages = ["staging", "production", "archived"]
        for stage_name in stages:
            metadata_file = Path("models") / stage_name / model_id / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
        
        raise FileNotFoundError(f"Model metadata not found in any stage: {model_id}")
    
    def list_models(self, stage: str = None) -> List[Dict[str, Any]]:
        """List all registered models."""
        models_dir = Path("models")
        stages = [stage] if stage else ["staging", "production", "archived"]
        
        all_models = []
        for stage_name in stages:
            stage_dir = models_dir / stage_name
            if stage_dir.exists():
                for model_dir in stage_dir.iterdir():
                    if model_dir.is_dir():
                        try:
                            model_info = self.get_model_info(model_dir.name, stage_name)
                            all_models.append(model_info)
                        except FileNotFoundError:
                            continue
        
        return sorted(all_models, key=lambda x: x.get("registered_at", ""), reverse=True)
    
    def load_model(self, model_id: str, stage: str = None) -> Any:
        """Load a registered model, searching across stages if needed."""
        # If stage is specified, check only that stage
        if stage:
            model_path = Path("models") / stage / model_id / "model.pkl"
            if model_path.exists():
                return joblib.load(model_path)
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # If stage is not specified, search across all stages
        stages = ["staging", "production", "archived"]
        for stage_name in stages:
            model_path = Path("models") / stage_name / model_id / "model.pkl"
            if model_path.exists():
                return joblib.load(model_path)
        
        raise FileNotFoundError(f"Model file not found in any stage: {model_id}")
    
    def promote_model(self, model_id: str, from_stage: str, to_stage: str) -> bool:
        """Promote model from one stage to another."""
        try:
            model = self.load_model(model_id, from_stage)
            metadata = self.get_model_info(model_id, from_stage)
            
            # Create a completely new metadata object for the promoted model
            promoted_metadata = {
                "model_id": model_id,  # Keep the same model_id
                "model_path": "",  # Will be set below
                "stage": to_stage,
                "registered_at": datetime.now().isoformat(),
                "model_type": metadata.get("model_type", "unknown"),
                "metrics": metadata.get("metrics", {}),
                "promoted_from": from_stage,
                "original_registered_at": metadata.get("registered_at", "")
            }
            
            # Save model to production directory
            model_dir = Path("models") / to_stage / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
            
            # Update model_path in metadata
            promoted_metadata['model_path'] = str(model_path)
            
            # Save updated metadata
            metadata_file = model_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(promoted_metadata, f, indent=2, default=str)
            
            logger.info(f"Model promoted: {model_id} from {from_stage} to {to_stage}")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model: {str(e)}")
            return False
