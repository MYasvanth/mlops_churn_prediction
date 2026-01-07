#!/usr/bin/env python3
"""
Script to migrate existing models from local storage to MLflow tracking
"""

import sys
from pathlib import Path
import mlflow
import pandas as pd
import json
from datetime import datetime
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelMigrator:
    """Migrate existing models to MLflow"""
    
    def __init__(self, config_path="configs/monitoring/monitoring_config.yaml"):
        self.config = load_config(config_path)
        mlflow_config = self.config.get('mlflow', {})
        
        # Set up MLflow tracking
        self.tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5000')
        self.experiment_name = mlflow_config.get('experiment_name', 'churn_monitoring')
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"ModelMigrator initialized with tracking URI: {self.tracking_uri}")
    
    def migrate_staging_models(self):
        """Migrate all staging models to MLflow"""
        staging_dir = Path("models/staging")
        
        if not staging_dir.exists():
            logger.error("Staging directory not found")
            return False
        
        model_dirs = [d for d in staging_dir.iterdir() if d.is_dir()]
        
        if not model_dirs:
            logger.warning("No staging models found to migrate")
            return False
        
        migrated_count = 0
        
        for model_dir in model_dirs:
            try:
                # Load model metadata
                metadata_path = model_dir / "metadata.json"
                if not metadata_path.exists():
                    logger.warning(f"No metadata found for {model_dir.name}")
                    continue
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Load model
                model_path = model_dir / "model.pkl"
                if not model_path.exists():
                    logger.warning(f"No model file found for {model_dir.name}")
                    continue
                
                model = joblib.load(model_path)
                
                # Create MLflow run
                run_name = f"migrated_{model_dir.name}"
                
                with mlflow.start_run(run_name=run_name):
                    # Log parameters
                    mlflow.log_params({
                        'model_type': metadata.get('model_type', 'unknown'),
                        'model_id': metadata.get('model_id', 'unknown'),
                        'training_timestamp': metadata.get('training_timestamp', 'unknown')
                    })
                    
                    # Log metrics
                    performance_metrics = metadata.get('performance_metrics', {})
                    mlflow.log_metrics(performance_metrics)
                    
                    # Log model
                    mlflow.sklearn.log_model(model, "model")
                    
                    # Log metadata as tag
                    mlflow.set_tag("migration_source", "local_staging")
                    mlflow.set_tag("original_model_id", model_dir.name)
                
                migrated_count += 1
                logger.info(f"Migrated model: {model_dir.name}")
                
            except Exception as e:
                logger.error(f"Failed to migrate {model_dir.name}: {str(e)}")
                continue
        
        logger.info(f"Successfully migrated {migrated_count}/{len(model_dirs)} models to MLflow")
        return migrated_count > 0
    
    def migrate_production_model(self):
        """Migrate production model to MLflow"""
        production_dir = Path("models/production")
        
        if not production_dir.exists():
            logger.error("Production directory not found")
            return False
        
        model_dirs = [d for d in production_dir.iterdir() if d.is_dir()]
        
        if not model_dirs:
            logger.warning("No production models found to migrate")
            return False
        
        # Get the most recent production model
        model_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        model_dir = model_dirs[0]
        
        try:
            # Load model metadata
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                logger.warning(f"No metadata found for {model_dir.name}")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load model
            model_path = model_dir / "model.pkl"
            if not model_path.exists():
                logger.warning(f"No model file found for {model_dir.name}")
                return False
            
            model = joblib.load(model_path)
            
            # Create MLflow run
            run_name = f"production_{model_dir.name}"
            
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_params({
                    'model_type': metadata.get('model_type', 'unknown'),
                    'model_id': metadata.get('model_id', 'unknown'),
                    'promotion_timestamp': metadata.get('promotion_timestamp', 'unknown'),
                    'status': 'production'
                })
                
                # Log metrics
                performance_metrics = metadata.get('performance_metrics', {})
                mlflow.log_metrics(performance_metrics)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Log metadata as tag
                mlflow.set_tag("migration_source", "local_production")
                mlflow.set_tag("original_model_id", model_dir.name)
                mlflow.set_tag("production_model", "true")
            
            logger.info(f"Migrated production model: {model_dir.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate production model {model_dir.name}: {str(e)}")
            return False

def main():
    """Main function for model migration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate local models to MLflow")
    parser.add_argument('--config', default='configs/monitoring/monitoring_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--staging', action='store_true',
                       help='Migrate staging models')
    parser.add_argument('--production', action='store_true',
                       help='Migrate production model')
    
    args = parser.parse_args()
    
    try:
        migrator = ModelMigrator(args.config)
        
        success = True
        
        if args.staging:
            success &= migrator.migrate_staging_models()
        
        if args.production:
            success &= migrator.migrate_production_model()
        
        if not args.staging and not args.production:
            # Default: migrate both
            success = migrator.migrate_staging_models()
            success &= migrator.migrate_production_model()
        
        if success:
            logger.info("✅ Model migration completed successfully")
        else:
            logger.warning("⚠️ Model migration completed with some issues")
            
    except Exception as e:
        logger.error(f"Model migration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
