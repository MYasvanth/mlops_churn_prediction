# confidence_interval_utils.py
"""
Bootstrap confidence interval calculations for ML metrics.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, List, Callable
import logging

logger = logging.getLogger(__name__)

class BootstrapCI:
    """Bootstrap confidence interval calculations for ML metrics"""
    
    def __init__(self, confidence: float = 0.95, n_bootstrap: int = 1000, random_state: int = 42):
        """
        Initialize BootstrapCI.
        
        Args:
            confidence (float): Confidence level (e.g., 0.95 for 95% CI)
            n_bootstrap (int): Number of bootstrap samples
            random_state (int): Random seed for reproducibility
        """
        self.confidence = confidence
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        np.random.seed(random_state)
    
    def _bootstrap_metric(self, y_true: np.array, y_pred: np.array, 
                         metric_func: Callable, **kwargs) -> List[float]:
        """
        Generic bootstrap sampling for any metric.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels or probabilities
            metric_func (Callable): Metric function to bootstrap
            **kwargs: Additional arguments for metric function
            
        Returns:
            List[float]: Bootstrap scores
        """
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for i in range(self.n_bootstrap):
            # Set seed for each iteration for reproducibility
            np.random.seed(self.random_state + i)
            
            # Bootstrap sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            try:
                score = metric_func(y_true_boot, y_pred_boot, **kwargs)
                bootstrap_scores.append(score)
            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed: {str(e)}")
                continue
        
        if len(bootstrap_scores) < self.n_bootstrap * 0.8:
            logger.warning(f"Only {len(bootstrap_scores)}/{self.n_bootstrap} bootstrap samples succeeded")
        
        return bootstrap_scores
    
    def _calculate_ci(self, bootstrap_scores: List[float]) -> Tuple[float, float]:
        """
        Calculate confidence interval from bootstrap scores.
        
        Args:
            bootstrap_scores (List[float]): Bootstrap metric scores
            
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound)
        """
        if not bootstrap_scores:
            logger.error("No bootstrap scores available for CI calculation")
            return (0.0, 0.0)
        
        alpha = 1 - self.confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_scores, lower_percentile)
        upper_bound = np.percentile(bootstrap_scores, upper_percentile)
        
        return (float(lower_bound), float(upper_bound))
    
    def accuracy_ci(self, y_true: np.array, y_pred: np.array) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for accuracy.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound)
        """
        try:
            bootstrap_scores = self._bootstrap_metric(y_true, y_pred, accuracy_score)
            return self._calculate_ci(bootstrap_scores)
        except Exception as e:
            logger.error(f"Accuracy CI calculation failed: {str(e)}")
            return (0.0, 0.0)
    
    def precision_ci(self, y_true: np.array, y_pred: np.array, 
                    average: str = 'binary') -> Tuple[float, float]:
        """
        Bootstrap confidence interval for precision.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            average (str): Averaging strategy for multiclass
            
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound)
        """
        try:
            bootstrap_scores = self._bootstrap_metric(
                y_true, y_pred, precision_score, 
                average=average, zero_division=0
            )
            return self._calculate_ci(bootstrap_scores)
        except Exception as e:
            logger.error(f"Precision CI calculation failed: {str(e)}")
            return (0.0, 0.0)
    
    def recall_ci(self, y_true: np.array, y_pred: np.array, 
                 average: str = 'binary') -> Tuple[float, float]:
        """
        Bootstrap confidence interval for recall.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            average (str): Averaging strategy for multiclass
            
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound)
        """
        try:
            bootstrap_scores = self._bootstrap_metric(
                y_true, y_pred, recall_score, 
                average=average, zero_division=0
            )
            return self._calculate_ci(bootstrap_scores)
        except Exception as e:
            logger.error(f"Recall CI calculation failed: {str(e)}")
            return (0.0, 0.0)
    
    def f1_ci(self, y_true: np.array, y_pred: np.array, 
             average: str = 'binary') -> Tuple[float, float]:
        """
        Bootstrap confidence interval for F1-score.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            average (str): Averaging strategy for multiclass
            
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound)
        """
        try:
            bootstrap_scores = self._bootstrap_metric(
                y_true, y_pred, f1_score, 
                average=average, zero_division=0
            )
            return self._calculate_ci(bootstrap_scores)
        except Exception as e:
            logger.error(f"F1 CI calculation failed: {str(e)}")
            return (0.0, 0.0)
    
    def roc_auc_ci(self, y_true: np.array, y_pred_proba: np.array) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for ROC-AUC.
        
        Args:
            y_true (np.array): True labels
            y_pred_proba (np.array): Predicted probabilities
            
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound)
        """
        try:
            bootstrap_scores = self._bootstrap_metric(y_true, y_pred_proba, roc_auc_score)
            return self._calculate_ci(bootstrap_scores)
        except Exception as e:
            logger.error(f"ROC-AUC CI calculation failed: {str(e)}")
            return (0.0, 0.0)
    
    def comprehensive_ci(self, y_true: np.array, y_pred: np.array, 
                        y_pred_proba: np.array) -> dict:
        """
        Calculate confidence intervals for all common metrics.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            y_pred_proba (np.array): Predicted probabilities
            
        Returns:
            dict: Dictionary with CI for all metrics
        """
        try:
            results = {
                'confidence_level': self.confidence,
                'n_bootstrap': self.n_bootstrap,
                'metrics': {
                    'accuracy': {
                        'ci': self.accuracy_ci(y_true, y_pred),
                        'point_estimate': float(accuracy_score(y_true, y_pred))
                    },
                    'precision': {
                        'ci': self.precision_ci(y_true, y_pred),
                        'point_estimate': float(precision_score(y_true, y_pred, zero_division=0))
                    },
                    'recall': {
                        'ci': self.recall_ci(y_true, y_pred),
                        'point_estimate': float(recall_score(y_true, y_pred, zero_division=0))
                    },
                    'f1_score': {
                        'ci': self.f1_ci(y_true, y_pred),
                        'point_estimate': float(f1_score(y_true, y_pred, zero_division=0))
                    },
                    'roc_auc': {
                        'ci': self.roc_auc_ci(y_true, y_pred_proba),
                        'point_estimate': float(roc_auc_score(y_true, y_pred_proba))
                    }
                }
            }
            
            # Add CI width information
            for metric_name, metric_data in results['metrics'].items():
                ci_lower, ci_upper = metric_data['ci']
                ci_width = ci_upper - ci_lower
                margin_of_error = ci_width / 2
                
                metric_data['ci_width'] = float(ci_width)
                metric_data['margin_of_error'] = float(margin_of_error)
                metric_data['relative_margin'] = float(margin_of_error / metric_data['point_estimate']) if metric_data['point_estimate'] > 0 else 0.0
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive CI calculation failed: {str(e)}")
            return {'error': str(e)}
    
    def format_ci_report(self, y_true: np.array, y_pred: np.array, 
                        y_pred_proba: np.array) -> str:
        """
        Generate a formatted confidence interval report.
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted labels
            y_pred_proba (np.array): Predicted probabilities
            
        Returns:
            str: Formatted CI report
        """
        try:
            ci_results = self.comprehensive_ci(y_true, y_pred, y_pred_proba)
            
            if 'error' in ci_results:
                return f"Error generating CI report: {ci_results['error']}"
            
            report_lines = [
                f"Confidence Interval Report ({ci_results['confidence_level']*100:.0f}% CI, {ci_results['n_bootstrap']} bootstrap samples)",
                "=" * 80
            ]
            
            for metric_name, metric_data in ci_results['metrics'].items():
                point_est = metric_data['point_estimate']
                ci_lower, ci_upper = metric_data['ci']
                margin_error = metric_data['margin_of_error']
                
                report_lines.append(
                    f"{metric_name.upper():>12}: {point_est:.4f} ± {margin_error:.4f} "
                    f"(95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
                )
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"CI report formatting failed: {str(e)}")
            return f"Error formatting CI report: {str(e)}"