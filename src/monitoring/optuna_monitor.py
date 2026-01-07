"""
Optuna Monitoring and Optimization Module
Provides comprehensive Optuna integration for hyperparameter optimization
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import sqlite3
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class OptunaMonitor:
    """Optuna monitoring and optimization management"""
    
    def __init__(self, storage_url: str = "sqlite:///optuna_studies.db"):
        self.storage_url = storage_url
        self.study_prefix = "churn_prediction"
        
    def create_study(self, study_name: str, direction: str = "maximize") -> optuna.Study:
        """Create a new Optuna study"""
        try:
            study = optuna.create_study(
                study_name=f"{self.study_prefix}_{study_name}",
                direction=direction,
                storage=self.storage_url,
                load_if_exists=True
            )
            logger.info(f"Created study: {study_name}")
            return study
        except Exception as e:
            logger.error(f"Failed to create study {study_name}: {str(e)}")
            raise
    
    def get_studies(self) -> List[str]:
        """Get list of all studies"""
        try:
            study_summaries = optuna.get_all_study_summaries(storage=self.storage_url)
            return [s.study_name for s in study_summaries 
                   if s.study_name.startswith(self.study_prefix)]
        except Exception as e:
            logger.error(f"Failed to get studies: {str(e)}")
            return []
    
    def get_study(self, study_name: str) -> optuna.Study:
        """Get existing study"""
        try:
            return optuna.load_study(
                study_name=study_name,
                storage=self.storage_url
            )
        except Exception as e:
            logger.error(f"Failed to load study {study_name}: {str(e)}")
            raise
    
    def get_trial_count(self, study_name: str) -> int:
        """Get number of trials in study"""
        try:
            study = self.get_study(study_name)
            return len(study.trials)
        except:
            return 0
    
    def get_best_value(self, study_name: str) -> Optional[float]:
        """Get best value from study"""
        try:
            study = self.get_study(study_name)
            return study.best_value if study.best_value is not None else None
        except:
            return None
    
    def get_optimization_history(self, study_name: str) -> pd.DataFrame:
        """Get optimization history as DataFrame"""
        try:
            study = self.get_study(study_name)
            trials = study.trials
            
            history = []
            for trial in trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    history.append({
                        'trial_number': trial.number,
                        'value': trial.value,
                        'params': trial.params,
                        'datetime': trial.datetime_start
                    })
            
            return pd.DataFrame(history)
        except Exception as e:
            logger.error(f"Failed to get optimization history: {str(e)}")
            return pd.DataFrame()
    
    def get_parameter_importance(self, study_name: str) -> pd.DataFrame:
        """Get parameter importance analysis"""
        try:
            study = self.get_study(study_name)
            
            if len(study.trials) < 2:
                return pd.DataFrame()
            
            importance = optuna.importance.get_param_importances(study)
            
            importance_df = pd.DataFrame(
                list(importance.items()),
                columns=['parameter', 'importance']
            )
            
            return importance_df.sort_values('importance', ascending=False)
            
        except Exception as e:
            logger.error(f"Failed to get parameter importance: {str(e)}")
            return pd.DataFrame()
    
    def get_parallel_coordinates(self, study_name: str) -> pd.DataFrame:
        """Get data for parallel coordinates plot"""
        try:
            study = self.get_study(study_name)
            trials = study.trials
            
            if len(trials) < 2:
                return pd.DataFrame()
            
            data = []
            for trial in trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    trial_data = {'value': trial.value}
                    trial_data.update(trial.params)
                    data.append(trial_data)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to get parallel coordinates data: {str(e)}")
            return pd.DataFrame()

class OptunaOptimizer:
    """Optuna-based hyperparameter optimization"""
    
    def __init__(self, storage_url: str = "sqlite:///optuna_studies.db"):
        self.storage_url = storage_url
        self.monitor = OptunaMonitor(storage_url)
    
    def optimize_model(self, model_name: str, model_class, X_train, y_train, X_val, y_val, 
                      n_trials: int = 100) -> Dict[str, Any]:
        """Optimize model hyperparameters based on model type"""
        
        if model_name.lower() == "lightgbm":
            return self._optimize_lightgbm(model_class, X_train, y_train, X_val, y_val, n_trials)
        elif model_name.lower() == "xgboost":
            return self._optimize_xgboost(model_class, X_train, y_train, X_val, y_val, n_trials)
        else:
            return self._optimize_sklearn(model_class, X_train, y_train, X_val, y_val, n_trials)
    
    def _optimize_lightgbm(self, model_class, X_train, y_train, X_val, y_val, n_trials):
        """Optimize LightGBM hyperparameters"""
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            from sklearn.metrics import accuracy_score
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study = self.monitor.create_study("lightgbm_optimization")
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _optimize_xgboost(self, model_class, X_train, y_train, X_val, y_val, n_trials):
        """Optimize XGBoost hyperparameters"""
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            from sklearn.metrics import accuracy_score
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study = self.monitor.create_study("xgboost_optimization")
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _optimize_sklearn(self, model_class, X_train, y_train, X_val, y_val, n_trials):
        """Optimize scikit-learn model hyperparameters"""
        def objective(trial):
            if "RandomForest" in str(model_class):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
            elif "LogisticRegression" in str(model_class):
                params = {
                    'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
                    'max_iter': trial.suggest_int('max_iter', 100, 1000),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                }
            else:
                params = {}
            
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            from sklearn.metrics import accuracy_score
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study_name = f"{model_class.__name__.lower()}_optimization"
        study = self.monitor.create_study(study_name)
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
