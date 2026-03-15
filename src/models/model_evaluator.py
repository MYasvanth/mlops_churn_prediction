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
from sklearn.model_selection import learning_curve, validation_curve
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
    bias_variance_analysis: Dict[str, Any] = None
    confidence_intervals: Dict[str, Tuple[float, float]] = None
    statistical_significance: Dict[str, Any] = None

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
        self.statistical_reports_dir = Path("reports/statistical_analysis")
        self.statistical_reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluation thresholds
        self.min_accuracy = self.evaluation_config.get('min_accuracy', 0.8)
        self.min_precision = self.evaluation_config.get('min_precision', 0.7)
        self.min_recall = self.evaluation_config.get('min_recall', 0.7)
        self.min_f1_score = self.evaluation_config.get('min_f1_score', 0.7)
        self.min_roc_auc = self.evaluation_config.get('min_roc_auc', 0.8)
        
        # Statistical analysis configuration
        self.confidence_level = self.evaluation_config.get('confidence_level', 0.95)
        self.n_bootstrap = self.evaluation_config.get('n_bootstrap', 1000)
        self.alpha = self.evaluation_config.get('alpha', 0.05)
        
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
            
            # Calculate metrics with confidence intervals
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(y_test, y_pred, y_pred_proba)
            metrics.confidence_intervals = confidence_intervals
            
            # Log metrics to MLflow
            self._log_metrics_to_mlflow(metrics, model_name)
            
            # Generate visualizations
            self._generate_evaluation_plots(y_test, y_pred, y_pred_proba, model_name)
            
            # Analyze learning curves for bias-variance diagnosis
            if hasattr(model, 'fit'):
                bias_variance_analysis = self._analyze_learning_curves(model, X_test, y_test, model_name)
                metrics.bias_variance_analysis = bias_variance_analysis
            
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
        # Use binary metrics for the positive class (Churn=1)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
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
        Plot confusion matrix with fix for clipping issues.
        
        Args:
            y_true (pd.Series): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Name of the model
        """
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'],
                   ax=ax)
        
        # Set titles and labels
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Robust fix for clipping: manually set y-limits
        # In some matplotlib versions, the first and last rows are cut in half
        bottom, top = ax.get_ylim()
        if bottom < top:
            ax.set_ylim(bottom + 0.5, top - 0.5)
        else:
            ax.set_ylim(bottom - 0.5, top + 0.5)
            
        plt.tight_layout()
        
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
            'confidence_intervals': metrics.confidence_intervals,
            'classification_report': metrics.classification_report,
            'confusion_matrix': metrics.confusion_matrix.tolist(),
            'bias_variance_analysis': metrics.bias_variance_analysis,
            'statistical_significance': metrics.statistical_significance
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
    
    def find_optimal_threshold(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                               metric_name: str = 'f1') -> Tuple[float, float]:
        """
        Find the optimal classification threshold that maximizes a specific metric.
        
        Args:
            y_true (pd.Series): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            metric_name (str): Metric to optimize ('f1', 'f2', or 'balanced_accuracy')
            
        Returns:
            Tuple[float, float]: (Optimal threshold, maximum metric value)
        """
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_score = -1
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric_name == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric_name == 'f2':
                from sklearn.metrics import fbeta_score
                score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            elif metric_name == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = accuracy_score(y_true, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimal threshold found: {best_threshold:.3f} for {metric_name} score: {best_score:.3f}")
        return best_threshold, best_score

    def evaluate_with_threshold(self, model: Any, X_test: pd.DataFrame, 
                               y_test: pd.Series, threshold: float = 0.5, 
                               model_name: str = "model") -> EvaluationMetrics:
        """
        Evaluate a model using a specific probability threshold.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            threshold (float): Classification threshold
            model_name (str): Name of the model
            
        Returns:
            EvaluationMetrics: Evaluation metrics
        """
        try:
            logger.info(f"Evaluating model: {model_name} with threshold: {threshold:.3f}")
            
            # Get probabilities
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Apply threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Log metrics to MLflow
            self._log_metrics_to_mlflow(metrics, f"{model_name}_threshold_{threshold:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name} with threshold: {str(e)}")
            raise

    def compare_models(self, evaluation_results: Dict[str, EvaluationMetrics]) -> str:
        """
        Compare multiple models and return the best one.
        
        Args:
            evaluation_results (Dict[str, EvaluationMetrics]): Dictionary of model evaluations
            
        Returns:
            str: Name of the best model
        """
        # Perform statistical comparison
        statistical_comparison = self.compare_models_statistically(evaluation_results)
        
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
        
        # Save statistical comparison results
        if statistical_comparison:
            self._save_statistical_comparison(statistical_comparison)
        
        logger.info(f"Best model: {best_model} with composite score: {best_score:.3f}")
        return best_model
    
    def _analyze_learning_curves(self, model, X, y, model_name: str) -> Dict[str, Any]:
        """
        Analyze learning curves for bias-variance diagnosis.
        
        Args:
            model: Trained model
            X: Features
            y: Target
            model_name: Name of the model
            
        Returns:
            Dict: Bias-variance analysis results
        """
        try:
            # Generate learning curves
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, cv=5, train_sizes=train_sizes, 
                scoring='accuracy', n_jobs=-1, random_state=42
            )
            
            # Calculate means and stds
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Bias-variance diagnosis
            final_train_score = train_mean[-1]
            final_val_score = val_mean[-1]
            gap = final_train_score - final_val_score
            
            # Diagnosis logic
            if final_train_score < 0.8 and final_val_score < 0.8:
                diagnosis = "HIGH_BIAS_UNDERFIT"
                recommendation = "Increase model complexity, add features, reduce regularization"
            elif final_train_score > 0.9 and gap > 0.1:
                diagnosis = "HIGH_VARIANCE_OVERFIT"
                recommendation = "Reduce model complexity, add regularization, get more data"
            elif gap < 0.05:
                diagnosis = "GOOD_FIT"
                recommendation = "Model is well-balanced"
            else:
                diagnosis = "MODERATE_OVERFIT"
                recommendation = "Consider slight regularization or more data"
            
            # Plot learning curves
            self._plot_learning_curves(train_sizes_abs, train_mean, train_std, 
                                     val_mean, val_std, model_name, diagnosis)
            
            analysis = {
                'final_train_score': float(final_train_score),
                'final_val_score': float(final_val_score),
                'train_val_gap': float(gap),
                'diagnosis': diagnosis,
                'recommendation': recommendation,
                'train_scores': train_mean.tolist(),
                'val_scores': val_mean.tolist(),
                'train_sizes': train_sizes_abs.tolist()
            }
            
            # Log to MLflow
            mlflow.log_metric("train_val_gap", gap)
            mlflow.log_param("bias_variance_diagnosis", diagnosis)
            
            logger.info(f"Learning curve analysis: {diagnosis}")
            return analysis
            
        except Exception as e:
            logger.warning(f"Learning curve analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _plot_learning_curves(self, train_sizes, train_mean, train_std, 
                             val_mean, val_std, model_name: str, diagnosis: str):
        """
        Plot learning curves with bias-variance diagnosis.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot training scores
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        # Plot validation scores
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'Learning Curves - {model_name}\nDiagnosis: {diagnosis}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add diagnosis annotation
        gap = train_mean[-1] - val_mean[-1]
        plt.annotate(f'Train-Val Gap: {gap:.3f}', 
                    xy=(0.7, 0.1), xycoords='axes fraction',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save plot
        plt.savefig(self.reports_dir / f'{model_name}_learning_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        try:
            mlflow.log_artifact(str(self.reports_dir / f'{model_name}_learning_curves.png'))
        except Exception as e:
            logger.warning(f"Failed to log learning curves to MLflow: {str(e)}")
    
    def _calculate_confidence_intervals(self, y_true: pd.Series, y_pred: np.ndarray, 
                                       y_pred_proba: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for all metrics.
        
        Args:
            y_true (pd.Series): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities
            
        Returns:
            Dict[str, Tuple[float, float]]: Confidence intervals for each metric
        """
        try:
            from ..utils.confidence_interval_utils import BootstrapCI
            
            ci_calculator = BootstrapCI(
                confidence=self.confidence_level, 
                n_bootstrap=self.n_bootstrap
            )
            
            return {
                'accuracy': ci_calculator.accuracy_ci(y_true, y_pred),
                'precision': ci_calculator.precision_ci(y_true, y_pred),
                'recall': ci_calculator.recall_ci(y_true, y_pred),
                'f1_score': ci_calculator.f1_ci(y_true, y_pred),
                'roc_auc': ci_calculator.roc_auc_ci(y_true, y_pred_proba)
            }
        except Exception as e:
            logger.warning(f"Failed to calculate confidence intervals: {str(e)}")
            return {}
    
    def compare_models_statistically(self, evaluation_results: Dict[str, EvaluationMetrics]) -> Dict[str, Any]:
        """
        Compare multiple models with statistical significance testing.
        
        Args:
            evaluation_results (Dict[str, EvaluationMetrics]): Dictionary of model evaluations
            
        Returns:
            Dict[str, Any]: Statistical comparison results
        """
        try:
            from ..utils.statistical_utils import StatisticalAnalyzer
            
            analyzer = StatisticalAnalyzer(alpha=self.alpha)
            
            # Get cross-validation scores from MLflow for each model (1 most recent run)
            model_scores = {}
            for model_name in evaluation_results.keys():
                scores = self._get_cv_scores_from_mlflow(model_name, max_runs=1)
                if scores:
                    model_scores[model_name] = scores
            
            # Perform pairwise statistical comparisons
            comparison_results = {}
            model_names = list(model_scores.keys())
            
            for i, model_a in enumerate(model_names):
                for model_b in model_names[i+1:]:
                    scores_a = model_scores[model_a]
                    scores_b = model_scores[model_b]
                    
                    # Ensure equal length for paired tests
                    min_len = min(len(scores_a), len(scores_b))
                    scores_a = scores_a[:min_len]
                    scores_b = scores_b[:min_len]
                    
                    comparison_key = f"{model_a}_vs_{model_b}"
                    comparison_results[comparison_key] = {
                        'paired_ttest': analyzer.paired_ttest(scores_a, scores_b),
                        'wilcoxon_test': analyzer.wilcoxon_test(scores_a, scores_b),
                        'effect_size': analyzer.cohens_d(scores_a, scores_b),
                        'practical_significance': analyzer.practical_significance(scores_a, scores_b)
                    }
            
            # Apply multiple comparison correction
            p_values = []
            for comparison in comparison_results.values():
                p_values.append(comparison['paired_ttest']['p_value'])
            
            if p_values:
                correction_results = analyzer.bonferroni_correction(p_values)
                comparison_results['multiple_comparison_correction'] = correction_results
            
            logger.info(f"Statistical comparison completed for {len(model_names)} models")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Statistical comparison failed: {str(e)}")
            return {}
    
    def _get_cv_scores_from_mlflow(self, model_name: str, max_runs: int = 1) -> List[float]:
        """
        Get cross-validation scores from MLflow experiments.
        
        Args:
            model_name (str): Name of the model
            max_runs (int): Maximum number of recent runs to use (default: 1)
            
        Returns:
            List[float]: List of accuracy scores from MLflow runs
        """
        try:
            import mlflow
            
            # Search for runs with this model type, ordered by most recent
            runs = mlflow.search_runs(
                filter_string=f"params.model_type = '{model_name}'",
                order_by=["start_time DESC"],
                max_results=max_runs  # Limit to most recent runs
            )
            
            if not runs.empty:
                # Extract accuracy scores from most recent runs only
                scores = runs['metrics.accuracy'].dropna().tolist()[:max_runs]
                logger.info(f"Found {len(scores)} most recent run for {model_name}")
                return scores
            else:
                logger.warning(f"No MLflow runs found for {model_name}")
                return []
                
        except Exception as e:
            logger.warning(f"Failed to get CV scores from MLflow for {model_name}: {str(e)}")
            return []
    
    def _save_statistical_comparison(self, comparison_results: Dict[str, Any]):
        """
        Save statistical comparison results to file.
        
        Args:
            comparison_results (Dict[str, Any]): Statistical comparison results
        """
        try:
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.statistical_reports_dir / f"model_comparison_stats_{timestamp}.json"
            
            with open(report_path, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            logger.info(f"Statistical comparison report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save statistical comparison: {str(e)}")