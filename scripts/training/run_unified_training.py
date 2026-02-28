#!/usr/bin/env python3
"""
Unified Training Script for Churn Prediction
Supports training single or multiple models with unified configuration
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.pipelines.unified_training_pipeline import (
    train_single_model, 
    train_all_models, 
    auto_train_best_model
)

def main():
    """Main function for unified training."""
    parser = argparse.ArgumentParser(description="Unified model training for churn prediction")
    
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["xgboost", "lightgbm", "random_forest", "logistic_regression", "svm", "all", "auto"],
        default="auto",
        help="Model type to train"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/train.csv",
        help="Path to training data"
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        choices=["staging", "production"],
        default="staging",
        help="Model stage for registration"
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/model/unified_model_config.yaml",
        help="Path to unified configuration"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to save training results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--tune-hparams",
        action="store_true",
        help="Enable hyperparameter tuning using Optuna"
    )
    
    args = parser.parse_args()
    
    print("Starting Unified Model Training")
    print(f"   - Model Type: {args.model_type}")
    print(f"   - Data Path: {args.data_path}")
    print(f"   - Stage: {args.stage}")
    print(f"   - Config: {args.config_path}")
    
    try:
        # Ensure data path exists
        if not Path(args.data_path).exists():
            print(f"Data file not found: {args.data_path}")
            return 1
        
        # Run training based on model type
        if args.model_type == "all":
            print("Training all supported models...")
            results = train_all_models(args.data_path, tune_hparams=args.tune_hparams)
            
        elif args.model_type == "auto":
            print("Auto-selecting best model...")
            results = auto_train_best_model(args.data_path, tune_hparams=args.tune_hparams)
            
        else:
            print(f"Training {args.model_type} model...")
            result = train_single_model(
                model_type=args.model_type,
                data_path=args.data_path,
                tune_hparams=args.tune_hparams
            )
            results = {args.model_type: result}
        
        # Save results if output path provided
        if args.output_path:
            with open(args.output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {args.output_path}")
        
        # Print results
        print("\nTraining Results:")
        if isinstance(results, dict):
            if "best_model" in results:
                print(f"   - Best Model: {results['best_model']}")
                del results['best_model']
            
            for model_type, result in results.items():
                if isinstance(result, dict) and "error" not in result:
                    print(f"   - {model_type}:")
                    print(f"     - Model ID: {result.get('model_id', 'N/A')}")
                    metrics = result.get('metrics', {})
                    f1 = metrics.get('f1_score', 'N/A')
                    acc = metrics.get('accuracy', 'N/A')
                    auc = metrics.get('roc_auc', 'N/A')
                    
                    f1_str = f"{f1:.4f}" if isinstance(f1, (int, float, np.number)) else str(f1)
                    acc_str = f"{acc:.4f}" if isinstance(acc, (int, float, np.number)) else str(acc)
                    auc_str = f"{auc:.4f}" if isinstance(auc, (int, float, np.number)) else str(auc)
                    
                    print(f"     - F1 Score: {f1_str}")
                    print(f"     - Accuracy: {acc_str}")
                    print(f"     - ROC AUC: {auc_str}")
                elif isinstance(result, dict) and "error" in result:
                    print(f"   - {model_type}: Error - {result['error']}")
        
        print("Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
