#!/usr/bin/env python3
"""
Enhanced Experiment Tracking Script for Churn Prediction
Comprehensive experiment tracking with MLflow integration
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.mlflow_utils import start_mlflow_run, log_params, log_metrics, log_model
from utils.config_loader import load_config
from utils.logger import get_logger
from models.unified_model_interface import UnifiedModelTrainer, UnifiedModelEvaluator
from models.unified_model_registry_fixed import UnifiedModelRegistry

logger = get_logger(__name__)

class ExperimentTracker:
    """Comprehensive experiment tracking with MLflow"""
    
    def __init__(self, config_path: str = "configs/model/unified_model_config.yaml"):
        self.config = load_config(config_path)
        self.registry = UnifiedModelRegistry(config_path)
        
    def run_experiment(self, model_type: str, experiment_name: str = None, 
                      hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a complete experiment with tracking"""
        try:
            # Set experiment name
            if experiment_name:
                import mlflow
                mlflow.set_experiment(experiment_name)
            
            with start_mlflow_run(run_name=f"{model_type}_experiment"):
                logger.info(f"Starting experiment for {model_type}")
                
                # Load data
                X_train, y_train, X_test, y_test = self._load_data()
                
                # Get model configuration
                model_config = self.config["models"]["model_configs"][model_type]
                params = hyperparameters or model_config["default_params"]
                
                # Log experiment parameters
                log_params({
                    "model_type": model_type,
                    **params
                })
                
                # Train model
                trainer = UnifiedModelTrainer(model_type)
                model = trainer.train(X_train, y_train, **params)
                
                # Evaluate model
                evaluator = UnifiedModelEvaluator()
                metrics = evaluator.evaluate(model, X_test, y_test)
                
                # Log metrics
                log_metrics(metrics)
                
                # Log model
                log_model(model, name=f"{model_type}_model", 
                         input_example=X_test.iloc[:5])
                
                # Register model if metrics meet thresholds
                model_id = None
                if evaluator.validate_performance(metrics):
                    model_path = f"models/experiments/{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    trainer.save_model(model_path)
                    
                    model_id = self.registry.register_model(
                        model=model,
                        metadata={
                            "model_type": model_type,
                            "hyperparameters": params,
                            "metrics": metrics,
                            "experiment_run": True
                        },
                        stage="staging"
                    )
                
                experiment_result = {
                    "model_type": model_type,
                    "metrics": metrics,
                    "model_id": model_id,
                    "experiment_id": mlflow.active_run().info.run_id,
                    "status": "success"
                }
                
                logger.info(f"Experiment completed: {experiment_result}")
                return experiment_result
                
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise
    
    def run_hyperparameter_tuning(self, model_type: str, n_trials: int = 10) -> List[Dict[str, Any]]:
        """Run hyperparameter tuning with experiment tracking"""
        try:
            import optuna
            
            study_name = f"{model_type}_tuning"
            storage_name = f"sqlite:///{study_name}.db"
            
            def objective(trial):
                # Get hyperparameter space
                model_config = self.config["models"]["model_configs"][model_type]
                space = model_config["hyperparameter_space"]
                
                # Sample hyperparameters
                params = {}
                for param, values in space.items():
                    if isinstance(values[0], (int, float)):
                        if len(values) == 2 and isinstance(values[0], (int, float)):
                            if isinstance(values[0], int):
                                params[param] = trial.suggest_int(param, values[0], values[1])
                            else:
                                params[param] = trial.suggest_float(param, values[0], values[1])
                        else:
                            params[param] = trial.suggest_categorical(param, values)
                    else:
                        params[param] = trial.suggest_categorical(param, values)
                
                # Run experiment
                result = self.run_experiment(model_type, f"{model_type}_tuning", params)
                return result["metrics"]["f1_score"]
            
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                direction="maximize",
                load_if_exists=True
            )
            
            study.optimize(objective, n_trials=n_trials)
            
            # Get best trial
            best_trial = study.best_trial
            
            logger.info(f"Best trial: {best_trial.params}")
            logger.info(f"Best F1 score: {best_trial.value}")
            
            return {
                "best_params": best_trial.params,
                "best_score": best_trial.value,
                "study_name": study_name
            }
            
        except ImportError:
            logger.warning("Optuna not installed, skipping hyperparameter tuning")
            return {}
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            raise
    
    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments"""
        try:
            import mlflow
            
            results = {}
            for experiment_name in experiment_names:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                    results[experiment_name] = runs.to_dict('records')
            
            # Create comparison summary
            summary = {}
            for exp_name, runs in results.items():
                if runs:
                    best_run = max(runs, key=lambda x: x.get('metrics.f1_score', 0))
                    summary[exp_name] = {
                        "best_f1_score": best_run.get('metrics.f1_score', 0),
                        "best_accuracy": best_run.get('metrics.accuracy', 0),
                        "total_runs": len(runs)
                    }
            
            return {
                "detailed_results": results,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Experiment comparison failed: {str(e)}")
            raise
    
    def _load_data(self):
        """Load training and testing data"""
        try:
            train_path = Path("data/processed/train.csv")
            test_path = Path("data/processed/test.csv")
            
            if not train_path.exists() or not test_path.exists():
                # Create sample data if not exists
                self._create_sample_data()
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            X_train = train_df.drop('Churn', axis=1)
            y_train = train_df['Churn']
            X_test = test_df.drop('Churn', axis=1)
            y_test = test_df['Churn']
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _create_sample_data(self):
        """Create sample data for experiments"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.datasets import make_classification
            
            # Create synthetic churn data
            X, y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=2,
                random_state=42
            )
            
            # Create DataFrame
            feature_names = [f'feature_{i}' for i in range(20)]
            df = pd.DataFrame(X, columns=feature_names)
            df['Churn'] = y
            
            # Split data
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save data
            Path("data/processed").mkdir(parents=True, exist_ok=True)
            train_df.to_csv("data/processed/train.csv", index=False)
            test_df.to_csv("data/processed/test.csv", index=False)
            
            logger.info("Created sample data for experiments")
            
        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")
            raise

def main():
    """Main experiment tracking script"""
    parser = argparse.ArgumentParser(description="Run churn prediction experiments")
    parser.add_argument("--model-type", choices=["xgboost", "lightgbm", "random_forest", "logistic_regression", "svm"],
                       required=True, help="Model type to experiment with")
    parser.add_argument("--experiment-name", help="MLflow experiment name")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=10, help="Number of tuning trials")
    parser.add_argument("--config", default="configs/model/unified_model_config.yaml",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    try:
        tracker = ExperimentTracker(args.config)
        
        if args.tune:
            result = tracker.run_hyperparameter_tuning(args.model_type, args.trials)
            print(json.dumps(result, indent=2))
        else:
            result = tracker.run_experiment(args.model_type, args.experiment_name)
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        logger.error(f"Experiment script failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
