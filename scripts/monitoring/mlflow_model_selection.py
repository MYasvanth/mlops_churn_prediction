#!/usr/bin/env python3
"""
MLflow-based automated model selection and promotion script.
Automatically finds the best model from MLflow experiments and promotes it to production.
"""

import sys
from pathlib import Path
import mlflow
import pandas as pd
from datetime import datetime
import json
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MLflowModelSelector:
    """Automated model selection using MLflow tracking"""
    
    def __init__(self, config_path="configs/monitoring/monitoring_config.yaml"):
        self.config = load_config(config_path)
        mlflow_config = self.config.get('mlflow', {})
        
        # Set up MLflow tracking
        self.tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5000')
        self.experiment_name = mlflow_config.get('experiment_name', 'churn_monitoring')
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        # Model registry paths
        self.production_dir = Path("models/production")
        self.production_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MLflowModelSelector initialized with tracking URI: {self.tracking_uri}")
    
    def get_best_model_from_mlflow(self, metric='f1_score', ascending=False):
        """
        Find the best model from MLflow experiments based on specified metric
        
        Args:
            metric: Metric to optimize (default: f1_score)
            ascending: Whether to sort ascending (False for maximize, True for minimize)
            
        Returns:
            dict: Best run information including model URI and metrics
        """
        try:
            # Get all runs from the experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                logger.error(f"Experiment '{self.experiment_name}' not found")
                return None
                
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            if runs.empty:
                logger.warning("No runs found in MLflow experiment")
                return None
            
            # Filter runs that have the target metric and a model
            valid_runs = runs[
                runs[f'metrics.{metric}'].notna() & 
                runs['artifact_uri'].notna()
            ].copy()
            
            if valid_runs.empty:
                logger.warning(f"No runs with metric '{metric}' and artifacts found")
                return None
            
            # Sort by the target metric
            valid_runs = valid_runs.sort_values(
                by=f'metrics.{metric}', 
                ascending=ascending
            )
            
            # Get the best run
            best_run = valid_runs.iloc[0]
            
            best_model_info = {
                'run_id': best_run['run_id'],
                'experiment_id': best_run['experiment_id'],
                'artifact_uri': best_run['artifact_uri'],
                'metrics': {
                    'f1_score': best_run.get('metrics.f1_score', 0),
                    'accuracy': best_run.get('metrics.accuracy', 0),
                    'precision': best_run.get('metrics.precision', 0),
                    'recall': best_run.get('metrics.recall', 0),
                    'roc_auc': best_run.get('metrics.roc_auc', 0)
                },
                'params': {
                    'model_type': best_run.get('params.model_type', 'unknown'),
                    'model_version': best_run.get('params.model_version', 'unknown')
                },
                'start_time': best_run['start_time'],
                'end_time': best_run['end_time']
            }
            
            logger.info(f"Best model found: {best_model_info['params']['model_type']} "
                       f"with {metric}={best_model_info['metrics'][metric]:.4f}")
            
            return best_model_info
            
        except Exception as e:
            logger.error(f"Error finding best model from MLflow: {str(e)}")
            return None
    
    def promote_model_to_production(self, model_info):
        """
        Promote the selected model to production
        
        Args:
            model_info: Model information from get_best_model_from_mlflow
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create unique model ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{model_info['params']['model_type']}_{timestamp}"
            
            # Create production model directory
            model_dir = self.production_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Download model artifacts from MLflow
            client = mlflow.tracking.MlflowClient()
            
            # Get the model artifact
            model_artifact_path = f"runs:/{model_info['run_id']}/model"
            
            # Download model file
            local_model_path = model_dir / "model.pkl"
            mlflow.artifacts.download_artifacts(
                artifact_uri=model_artifact_path,
                dst_path=str(model_dir)
            )
            
            # Create metadata file
            metadata = {
                'model_id': model_id,
                'run_id': model_info['run_id'],
                'experiment_id': model_info['experiment_id'],
                'promotion_timestamp': datetime.now().isoformat(),
                'performance_metrics': model_info['metrics'],
                'model_type': model_info['params']['model_type'],
                'mlflow_artifact_uri': model_info['artifact_uri'],
                'training_timestamp': model_info['start_time'].isoformat() if hasattr(model_info['start_time'], 'isoformat') else str(model_info['start_time'])
            }
            
            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model promoted to production: {model_id}")
            logger.info(f"Performance metrics: F1={model_info['metrics']['f1_score']:.4f}, "
                       f"Accuracy={model_info['metrics']['accuracy']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model to production: {str(e)}")
            return False
    
    def cleanup_old_production_models(self, keep_last_n=1):
        """
        Clean up old production models, keeping only the most recent N models
        
        Args:
            keep_last_n: Number of recent models to keep
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all production models
            model_dirs = [d for d in self.production_dir.iterdir() if d.is_dir()]
            
            if len(model_dirs) <= keep_last_n:
                return True
                
            # Sort by creation time (newest first)
            model_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)
            
            # Remove old models
            for old_dir in model_dirs[keep_last_n:]:
                try:
                    shutil.rmtree(old_dir)
                    logger.info(f"Removed old production model: {old_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {old_dir.name}: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old production models: {str(e)}")
            return False
    
    def run_automated_selection(self):
        """Run complete automated model selection and promotion"""
        logger.info("🚀 Starting automated model selection...")
        
        # Find best model from MLflow
        best_model = self.get_best_model_from_mlflow()
        
        if not best_model:
            logger.error("No suitable model found for promotion")
            return False
        
        # Promote to production
        success = self.promote_model_to_production(best_model)
        
        if success:
            # Clean up old models
            self.cleanup_old_production_models()
            logger.info("✅ Automated model selection completed successfully")
        else:
            logger.error("❌ Automated model selection failed")
        
        return success

def main():
    """Main function for automated model selection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated MLflow model selection")
    parser.add_argument('--config', default='configs/monitoring/monitoring_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--metric', default='f1_score',
                       help='Metric to optimize (default: f1_score)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up old production models')
    
    args = parser.parse_args()
    
    try:
        selector = MLflowModelSelector(args.config)
        
        if args.cleanup:
            selector.cleanup_old_production_models()
        else:
            selector.run_automated_selection()
            
    except Exception as e:
        logger.error(f"Automated selection failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
