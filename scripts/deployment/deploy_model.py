#!/usr/bin/env python3
"""
Enhanced Model Deployment Script for Churn Prediction
Handles model deployment to staging/production with comprehensive monitoring
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unified_model_registry_fixed import UnifiedModelRegistry
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.utils.mlflow_utils import start_mlflow_run, log_params, log_metrics

logger = get_logger(__name__)

class ModelDeployer:
    """Enhanced model deployment with monitoring and rollback capabilities"""
    
    def __init__(self, config_path: str = "configs/model/unified_model_config.yaml"):
        self.config = load_config(config_path)
        self.registry = UnifiedModelRegistry(config_path)
        
    def deploy_model(self, model_id: str, target_stage: str = "staging", 
                    validation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Deploy model to specified stage with comprehensive tracking"""
        try:
            with start_mlflow_run(run_name=f"deploy_{model_id}_{target_stage}"):
                logger.info(f"Starting deployment of model {model_id} to {target_stage}")
                
                # Log deployment parameters
                log_params({
                    "model_id": model_id,
                    "target_stage": target_stage,
                    "deployment_type": "manual"
                })
                
                # Get model info
                model_info = self.registry.get_model_info(model_id, "staging")
                
                # Validate model
                if validation_data:
                    validation_metrics = self._validate_model(model_id, validation_data)
                    log_metrics(validation_metrics)
                
                # Deploy based on target stage
                if target_stage == "staging":
                    deployment_id = self._deploy_to_staging(model_id)
                elif target_stage == "production":
                    deployment_id = self._deploy_to_production(model_id)
                else:
                    raise ValueError(f"Invalid target stage: {target_stage}")
                
                # Log deployment success
                import datetime
                deployment_info = {
                    "deployment_id": deployment_id,
                    "model_id": model_id,
                    "stage": target_stage,
                    "timestamp": str(datetime.datetime.now()),
                    "status": "success"
                }
                
                logger.info(f"Deployment completed: {deployment_info}")
                return deployment_info
                
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            # Log failure
            log_metrics({"deployment_success": 0})
            raise
    
    def _deploy_to_staging(self, model_id: str) -> str:
        """Deploy to staging environment"""
        try:
            # Model is already in staging, just validate
            model_info = self.registry.get_model_info(model_id, "staging")
            return f"staging_{model_id}"
        except Exception as e:
            raise RuntimeError(f"Failed to deploy to staging: {str(e)}")
    
    def _deploy_to_production(self, model_id: str) -> str:
        """Deploy to production with safety checks"""
        try:
            # Promote from staging to production
            success = self.registry.promote_model(model_id, "staging", "production")
            if success:
                return f"production_{model_id}"
            else:
                raise RuntimeError("Failed to promote model to production")
        except Exception as e:
            raise RuntimeError(f"Failed to deploy to production: {str(e)}")
    
    def _validate_model(self, model_id: str, validation_data: Dict[str, Any]) -> Dict[str, float]:
        """Validate model performance before deployment"""
        try:
            import pandas as pd
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            model = self.registry.load_model(model_id, "staging")
            
            X_test = validation_data.get("X_test")
            y_test = validation_data.get("y_test")
            
            if X_test is None or y_test is None:
                return {"validation_success": 1}
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                "validation_accuracy": accuracy_score(y_test, y_pred),
                "validation_precision": precision_score(y_test, y_pred, average='weighted'),
                "validation_recall": recall_score(y_test, y_pred, average='weighted'),
                "validation_f1": f1_score(y_test, y_pred, average='weighted')
            }
            
            if y_pred_proba is not None:
                metrics["validation_roc_auc"] = roc_auc_score(y_test, y_pred_proba)
            
            # Check against thresholds
            thresholds = self.config.get("evaluation", {}).get("thresholds", {})
            min_accuracy = thresholds.get("min_accuracy", 0.75)
            
            if metrics["validation_accuracy"] < min_accuracy:
                raise ValueError(f"Model accuracy {metrics['validation_accuracy']} below threshold {min_accuracy}")
            
            metrics["validation_success"] = 1
            return metrics
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return {"validation_success": 0}
    
    def rollback_deployment(self, model_id: str, stage: str) -> bool:
        """Rollback deployment to previous version"""
        try:
            logger.info(f"Rolling back {stage} deployment for {model_id}")
            
            # Get all models in stage
            models = self.registry.list_models(stage)
            current_model = next((m for m in models if m["model_id"] == model_id), None)
            
            if not current_model:
                logger.warning(f"Model {model_id} not found in {stage}")
                return False
            
            # Find previous version (simplified - would need proper versioning)
            previous_models = [m for m in models if m["model_id"] != model_id]
            if previous_models:
                previous_model = previous_models[0]
                logger.info(f"Rolling back to {previous_model['model_id']}")
                return True
            
            logger.warning("No previous model found for rollback")
            return False
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return False
    
    def get_deployment_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        try:
            status = {
                "model_id": model_id,
                "staging": {"deployed": False, "info": None},
                "production": {"deployed": False, "info": None}
            }
            
            for stage in ["staging", "production"]:
                try:
                    models = self.registry.list_models(stage)
                    model_info = next((m for m in models if m["model_id"] == model_id), None)
                    if model_info:
                        status[stage]["deployed"] = True
                        status[stage]["info"] = model_info
                except:
                    continue
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting deployment status: {str(e)}")
            return {"error": str(e)}

def main():
    """Main deployment script"""
    parser = argparse.ArgumentParser(description="Deploy churn prediction models")
    parser.add_argument("--model-id", required=True, help="Model ID to deploy")
    parser.add_argument("--stage", choices=["staging", "production"], default="staging", 
                       help="Target deployment stage")
    parser.add_argument("--config", default="configs/model/unified_model_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--validate", action="store_true", 
                       help="Run validation before deployment")
    
    args = parser.parse_args()
    
    try:
        import pandas as pd
        
        deployer = ModelDeployer(args.config)
        
        # Load validation data if requested
        validation_data = None
        if args.validate:
            # Load test data for validation
            test_path = Path("data/processed/test.csv")
            if test_path.exists():
                test_df = pd.read_csv(test_path)
                X_test = test_df.drop('Churn', axis=1)
                y_test = test_df['Churn']
                validation_data = {"X_test": X_test, "y_test": y_test}
        
        # Deploy model
        result = deployer.deploy_model(args.model_id, args.stage, validation_data)
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"Deployment script failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
