#!/usr/bin/env python3
"""
Complete end-to-end churn prediction training script
From data loading to model deployment
"""

import argparse
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Churn Prediction Pipeline")
    parser.add_argument(
        "--model-type",
        choices=["xgboost", "lightgbm", "random_forest", "logistic_regression", "svm", "ensemble"],
        default="xgboost",
        help="Model type to train"
    )
    parser.add_argument(
        "--data-path",
        default="data/processed",
        help="Path to processed data"
    )
    parser.add_argument(
        "--deploy-to-production",
        action="store_true",
        help="Deploy model to production after training"
    )
    parser.add_argument(
        "--zenml-server",
        action="store_true",
        help="Start ZenML server for monitoring"
    )
    parser.add_argument(
        "--hyperparameter-optimization",
        action="store_true",
        help="Enable hyperparameter optimization using Optuna"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials (if hyperparameter-optimization is enabled)"
    )
    parser.add_argument(
        "--cross-validation-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size ratio"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--monitoring-enabled",
        action="store_true",
        default=True,
        help="Enable monitoring and drift detection"
    )
    parser.add_argument(
        "--monitoring-config",
        default="configs/monitoring/monitoring_config.yaml",
        help="Path to monitoring configuration file"
    )
    parser.add_argument(
        "--deployment-config",
        default="configs/deployment/deployment_config.yaml",
        help="Path to deployment configuration file"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("CHURN PREDICTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Data Path: {args.data_path}")
    logger.info(f"Deploy to Production: {args.deploy_to_production}")
    
    try:
        # Import pipeline
        import sys
        sys.path.append(str(Path(__file__).parent))
        
        from zenml_pipelines.complete_churn_pipeline import run_pipeline
        
        # Execute pipeline
        run_pipeline(
            model_type=args.model_type,
            deploy_to_production=args.deploy_to_production
        )
        
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
