# model_evaluator.py
"""
Model evaluation module for churn prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import joblib
import mlflow
import logging
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)

@dataclass
class EvaluationMetrics:
    """Data class to store evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]

class ModelEvaluator:
    """
    Model evaluation class with comprehensive metrics and visualizations.
    """
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize the ModelEvaluator.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = load_config(config_path)
        self.evaluation_config = self.config.get('evaluation', {})
        
        # Set up directories
        self.reports_dir = Path("reports/model_performance")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation thresholds
        self.min_accuracy = self.evaluation_config.get('min_accuracy', 0.8)
        self.min_precision = self.evaluation_config.get('min_precision', 0.7)
        self.min_recall = self.evaluation_config.get('min_recall', 0.7)
        self.min_f1_score = self.evaluation_config.get('min_f1_score', 0.7)
        self.min_roc_auc = self.evaluation_config.get('min_roc_auc', 0.8)
        
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, 
                      y_test: pd.Series, model_name: str = "model") -> EvaluationMetrics:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            
        Returns:
            EvaluationMetrics: Evaluation metrics
        """
        try:
            logger.info(f"Evaluating model: {model_name}")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log metrics to MLflow
            self._log_metrics_to_mlflow(metrics, model_name)
            
            # Generate visualizations
            self._generate_evaluation_plots(y_test, y_pred, y_pred_proba, model_name)
            
            # Save detailed report
            self._save_evaluation_report(metrics, model_name)
            
            logger.info(f"Model evaluation completed for {model_name}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> EvaluationMetrics:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (pd.Series): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities
            
        Returns:
            EvaluationMetrics: Calculated metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=conf_matrix,
            classification_report=class_report
        )
    
    def _log_metrics_to_mlflow(self, metrics: EvaluationMetrics, model_name: str):
        """
        Log metrics to MLflow.
        
        Args:
            metrics (EvaluationMetrics): Evaluation metrics
            model_name (str): Name of the model
        """
        try:
            mlflow.log_metric("accuracy", metrics.accuracy)
            mlflow.log_metric("precision", metrics.precision)
            mlflow.log_metric("recall", metrics.recall)
            mlflow.log_metric("f1_score", metrics.f1_score)
            mlflow.log_metric("roc_auc", metrics.roc_auc)
            
            # Log additional metrics
            mlflow.log_metric("precision_class_0", metrics.classification_report['0']['precision'])
            mlflow.log_metric("recall_class_0", metrics.classification_report['0']['recall'])
            mlflow.log_metric("precision_class_1", metrics.classification_report['1']['precision'])
            mlflow.log_metric("recall_class_1", metrics.classification_report['1']['recall'])
            
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {str(e)}")
    
    def _generate_evaluation_plots(self, y_true: pd.Series, y_pred: np.ndarray, 
                                  y_pred_proba: np.ndarray, model_name: str):
        """
        Generate evaluation plots.
        
        Args:
            y_true (pd.Series): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
        """
        # Create confusion matrix plot
        self._plot_confusion_matrix(y_true, y_pred, model_name)
        
        # Create ROC curve
        self._plot_roc_curve(y_true, y_pred_proba, model_name)
        
        # Create precision-recall curve
        self._plot_precision_recall_curve(y_true, y_pred_proba, model_name)
        
        # Create feature importance plot if available
        # This would be implemented based on the model type
    
    def _plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, 
                              model_name: str):
        """
        Plot confusion matrix.
        
        Args:
            y_true (pd.Series): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
        """
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot
        plt.savefig(self.reports_dir / f'{model_name}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        try:
            mlflow.log_artifact(str(self.reports_dir / f'{model_name}_confusion_matrix.png'))
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix to MLflow: {str(e)}")
    
    def _plot_roc_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                       model_name: str):
        """
        Plot ROC curve.
        
        Args:
            y_true (pd.Series): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(self.reports_dir / f'{model_name}_roc_curve.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        try:
            mlflow.log_artifact(str(self.reports_dir / f'{model_name}_roc_curve.png'))
        except Exception as e:
            logger.warning(f"Failed to log ROC curve to MLflow: {str(e)}")
    
    def _plot_precision_recall_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                                    model_name: str):
        """
        Plot precision-recall curve.
        
        Args:
            y_true (pd.Series): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(self.reports_dir / f'{model_name}_precision_recall_curve.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        try:
            mlflow.log_artifact(str(self.reports_dir / f'{model_name}_precision_recall_curve.png'))
        except Exception as e:
            logger.warning(f"Failed to log PR curve to MLflow: {str(e)}")
    
    def _save_evaluation_report(self, metrics: EvaluationMetrics, model_name: str):
        """
        Save detailed evaluation report.
        
        Args:
            metrics (EvaluationMetrics): Evaluation metrics
            model_name (str): Name of the model
        """
        report = {
            'model_name': model_name,
            'metrics': {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'roc_auc': metrics.roc_auc
            },
            'classification_report': metrics.classification_report,
            'confusion_matrix': metrics.confusion_matrix.tolist()
        }
        
        import json
        report_path = self.reports_dir / f'{model_name}_evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
    
    def validate_model_performance(self, metrics: EvaluationMetrics) -> bool:
        """
        Validate if model performance meets minimum requirements.
        
        Args:
            metrics (EvaluationMetrics): Evaluation metrics
            
        Returns:
            bool: True if model meets requirements, False otherwise
        """
        checks = [
            metrics.accuracy >= self.min_accuracy,
            metrics.precision >= self.min_precision,
            metrics.recall >= self.min_recall,
            metrics.f1_score >= self.min_f1_score,
            metrics.roc_auc >= self.min_roc_auc
        ]
        
        if all(checks):
            logger.info("Model performance validation passed")
            return True
        else:
            logger.warning("Model performance validation failed")
            logger.warning(f"Accuracy: {metrics.accuracy:.3f} (required: {self.min_accuracy})")
            logger.warning(f"Precision: {metrics.precision:.3f} (required: {self.min_precision})")
            logger.warning(f"Recall: {metrics.recall:.3f} (required: {self.min_recall})")
            logger.warning(f"F1-score: {metrics.f1_score:.3f} (required: {self.min_f1_score})")
            logger.warning(f"ROC AUC: {metrics.roc_auc:.3f} (required: {self.min_roc_auc})")
            return False
    
    def compare_models(self, evaluation_results: Dict[str, EvaluationMetrics]) -> str:
        """
        Compare multiple models and return the best one.
        
        Args:
            evaluation_results (Dict[str, EvaluationMetrics]): Dictionary of model evaluations
            
        Returns:
            str: Name of the best model
        """
        best_model = None
        best_score = -1
        
        # Use a composite score for comparison
        for model_name, metrics in evaluation_results.items():
            # Weighted average of metrics
            composite_score = (
                0.2 * metrics.accuracy +
                0.2 * metrics.precision +
                0.2 * metrics.recall +
                0.2 * metrics.f1_score +
                0.2 * metrics.roc_auc
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_model = model_name
        
        logger.info(f"Best model: {best_model} with composite score: {best_score:.3f}")
        return best_model