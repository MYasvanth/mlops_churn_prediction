# hyperparameter_tuner.py
"""
Hyperparameter tuning module using Optuna.
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, Optional, List
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
import joblib
import mlflow
import logging
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)

@dataclass
class TuningResult:
    """Data class to store tuning results."""
    best_params: Dict[str, Any]
    best_score: float
    best_trial: optuna.trial.FrozenTrial
    study: optuna.study.Study
    model_name: str

class HyperparameterTuner:
    """
    Hyperparameter tuning class using Optuna optimization.
    """
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize the HyperparameterTuner.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = load_config(config_path)
        self.tuning_config = self.config.get('hyperparameter_tuning', {})
        
        # Set up directories
        self.results_dir = Path("reports/hyperparameter_tuning")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Tuning parameters
        self.n_trials = self.tuning_config.get('n_trials', 100)
        self.timeout = self.tuning_config.get('timeout', 3600)  # 1 hour
        self.cv_folds = self.tuning_config.get('cv_folds', 5)
        self.scoring = self.tuning_config.get('scoring', 'f1_weighted')
        self.random_state = self.config.get('random_state', 42)
        
        # Set up cross-validation
        self.cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                                 random_state=self.random_state)
        
        # Set up scorer
        if self.scoring == 'f1_weighted':
            self.scorer = make_scorer(f1_score, average='weighted')
        else:
            self.scorer = self.scoring
    
    def tune_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> TuningResult:
        """
        Tune Random Forest hyperparameters.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            TuningResult: Tuning results
        """
        logger.info("Starting Random Forest hyperparameter tuning")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state
            }
            
            # Create model
            model = RandomForestClassifier(**params)
            
            # Perform cross-validation
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=self.cv, scoring=self.scorer, n_jobs=-1)
            
            return scores.mean()
        
        # Create study
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=self.random_state))
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Create result
        result = TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_trial=study.best_trial,
            study=study,
            model_name="RandomForest"
        )
        
        # Log results
        self._log_tuning_results(result)
        
        logger.info(f"Random Forest tuning completed. Best score: {result.best_score:.4f}")
        return result
    
    def tune_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> TuningResult:
        """
        Tune Logistic Regression hyperparameters.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            TuningResult: Tuning results
        """
        logger.info("Starting Logistic Regression hyperparameter tuning")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'random_state': self.random_state
            }
            
            # Handle elasticnet penalty
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                params['solver'] = 'saga'  # Only saga supports elasticnet
            
            # Handle solver constraints
            if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
                params['solver'] = 'liblinear'
            
            # Create model
            model = LogisticRegression(**params)
            
            # Perform cross-validation
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=self.cv, scoring=self.scorer, n_jobs=-1)
            
            return scores.mean()
        
        # Create study
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=self.random_state))
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Create result
        result = TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_trial=study.best_trial,
            study=study,
            model_name="LogisticRegression"
        )
        
        # Log results
        self._log_tuning_results(result)
        
        logger.info(f"Logistic Regression tuning completed. Best score: {result.best_score:.4f}")
        return result
    
    def tune_svm(self, X_train: pd.DataFrame, y_train: pd.Series) -> TuningResult:
        """
        Tune SVM hyperparameters.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            TuningResult: Tuning results
        """
        logger.info("Starting SVM hyperparameter tuning")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'random_state': self.random_state
            }
            
            # Add degree for polynomial kernel
            if params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
            
            # Create model
            model = SVC(**params, probability=True)
            
            # Perform cross-validation
            scores = cross_val_score(model, X_train, y_train, 
                                   cv=self.cv, scoring=self.scorer, n_jobs=-1)
            
            return scores.mean()
        
        # Create study
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=self.random_state))
        
        # Optimize
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Create result
        result = TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_trial=study.best_trial,
            study=study,
            model_name="SVM"
        )
        
        # Log results
        self._log_tuning_results(result)
        
        logger.info(f"SVM tuning completed. Best score: {result.best_score:.4f}")
        return result
    
    def tune_model(self, model_name: str, X_train: pd.DataFrame, 
                   y_train: pd.Series) -> TuningResult:
        """
        Tune hyperparameters for a specific model.
        
        Args:
            model_name (str): Name of the model to tune
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            TuningResult: Tuning results
        """
        model_name = model_name.lower()
        
        if model_name == 'randomforest':
            return self.tune_random_forest(X_train, y_train)
        elif model_name == 'logisticregression':
            return self.tune_logistic_regression(X_train, y_train)
        elif model_name == 'svm':
            return self.tune_svm(X_train, y_train)
        else:
            raise ValueError(f"Model {model_name} not supported for tuning")
    
    def tune_multiple_models(self, models: List[str], X_train: pd.DataFrame, 
                           y_train: pd.Series) -> Dict[str, TuningResult]:
        """
        Tune hyperparameters for multiple models.
        
        Args:
            models (List[str]): List of model names to tune
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Dict[str, TuningResult]: Dictionary of tuning results
        """
        results = {}
        
        for model_name in models:
            try:
                result = self.tune_model(model_name, X_train, y_train)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error tuning {model_name}: {str(e)}")
        
        # Save comparison results
        self._save_comparison_results(results)
        
        return results
    
    def _log_tuning_results(self, result: TuningResult):
        """
        Log tuning results to MLflow.
        
        Args:
            result (TuningResult): Tuning results
        """
        try:
            with mlflow.start_run(nested=True):
                mlflow.set_tag("model_name", result.model_name)
                mlflow.set_tag("tuning_type", "hyperparameter_optimization")
                
                # Log best parameters
                for param_name, param_value in result.best_params.items():
                    mlflow.log_param(f"best_{param_name}", param_value)
                
                # Log best score
                mlflow.log_metric("best_cv_score", result.best_score)
                mlflow.log_metric("n_trials", len(result.study.trials))
                
                # Log study statistics
                completed_trials = [t for t in result.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if completed_trials:
                    scores = [t.value for t in completed_trials]
                    mlflow.log_metric("mean_cv_score", np.mean(scores))
                    mlflow.log_metric("std_cv_score", np.std(scores))
                
                # Save study object
                study_path = self.results_dir / f"{result.model_name}_study.pkl"
                joblib.dump(result.study, study_path)
                mlflow.log_artifact(str(study_path))
                
        except Exception as e:
            logger.warning(f"Failed to log tuning results to MLflow: {str(e)}")
    
    def _save_comparison_results(self, results: Dict[str, TuningResult]):
        """
        Save comparison results for multiple models.
        
        Args:
            results (Dict[str, TuningResult]): Dictionary of tuning results
        """
        comparison_data = []
        
        for model_name, result in results.items():
            comparison_data.append({
                'model_name': model_name,
                'best_score': result.best_score,
                'best_params': result.best_params,
                'n_trials': len(result.study.trials)
            })
        
        # Sort by best score
        comparison_data.sort(key=lambda x: x['best_score'], reverse=True)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = self.results_dir / f"model_comparison_{timestamp}.json"
        
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"Model comparison results saved to {comparison_path}")
        
        # Log best model
        if comparison_data:
            best_model = comparison_data[0]
            logger.info(f"Best model: {best_model['model_name']} with score: {best_model['best_score']:.4f}")
    
    def get_best_model(self, results: Dict[str, TuningResult]) -> TuningResult:
        """
        Get the best model from tuning results.
        
        Args:
            results (Dict[str, TuningResult]): Dictionary of tuning results
            
        Returns:
            TuningResult: Best tuning result
        """
        best_model = None
        best_score = -1
        
        for model_name, result in results.items():
            if result.best_score > best_score:
                best_score = result.best_score
                best_model = result
        
        return best_model
    
    def create_tuned_model(self, result: TuningResult) -> Any:
        """
        Create a model with the best hyperparameters.
        
        Args:
            result (TuningResult): Tuning result
            
        Returns:
            Any: Model with best hyperparameters
        """
        model_name = result.model_name.lower()
        
        if model_name == 'randomforest':
            return RandomForestClassifier(**result.best_params)
        elif model_name == 'logisticregression':
            return LogisticRegression(**result.best_params)
        elif model_name == 'svm':
            return SVC(**result.best_params, probability=True)
        else:
            raise ValueError(f"Model {model_name} not supported")
    
    def plot_optimization_history(self, result: TuningResult):
        """
        Plot optimization history.
        
        Args:
            result (TuningResult): Tuning result
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get trial values
            trials = result.study.trials
            values = [trial.value for trial in trials if trial.value is not None]
            
            if not values:
                logger.warning("No completed trials to plot")
                return
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(values) + 1), values, 'b-', alpha=0.7)
            plt.xlabel('Trial Number')
            plt.ylabel('CV Score')
            plt.title(f'Optimization History - {result.model_name}')
            plt.grid(True, alpha=0.3)
            
            # Add best value line
            best_value = result.best_score
            plt.axhline(y=best_value, color='r', linestyle='--', 
                       label=f'Best Score: {best_value:.4f}')
            plt.legend()
            
            # Save plot
            plot_path = self.results_dir / f"{result.model_name}_optimization_history.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Optimization history plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting optimization history: {str(e)}")
    
    def plot_parameter_importance(self, result: TuningResult):
        """
        Plot parameter importance.
        
        Args:
            result (TuningResult): Tuning result
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get parameter importance
            importance = optuna.importance.get_param_importances(result.study)
            
            if not importance:
                logger.warning("No parameter importance to plot")
                return
            
            # Create plot
            params = list(importance.keys())
            values = list(importance.values())
            
            plt.figure(figsize=(10, 6))
            plt.barh(params, values)
            plt.xlabel('Importance')
            plt.title(f'Parameter Importance - {result.model_name}')
            plt.tight_layout()
            
            # Save plot
            plot_path = self.results_dir / f"{result.model_name}_parameter_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Parameter importance plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Error plotting parameter importance: {str(e)}")
    
    def generate_tuning_report(self, result: TuningResult):
        """
        Generate a comprehensive tuning report.
        
        Args:
            result (TuningResult): Tuning result
        """
        # Create plots
        self.plot_optimization_history(result)
        self.plot_parameter_importance(result)
        
        # Create text report
        report = f"""
Hyperparameter Tuning Report
============================

Model: {result.model_name}
Best Score: {result.best_score:.4f}
Number of Trials: {len(result.study.trials)}

Best Parameters:
{json.dumps(result.best_params, indent=2)}

Study Statistics:
- Direction: {result.study.direction}
- Number of Trials: {len(result.study.trials)}
- Best Trial Number: {result.best_trial.number}

Trial Information:
- Best Trial Value: {result.best_trial.value:.4f}
- Best Trial Duration: {result.best_trial.duration}
        """
        
        # Save report
        report_path = self.results_dir / f"{result.model_name}_tuning_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Tuning report saved to {report_path}")