# model_performance_monitor.py
"""
Model Performance Monitoring Module

This module provides functionality to monitor model performance metrics,
track degradation, and generate alerts when performance drops below thresholds.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import mlflow
import mlflow.sklearn
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class to store model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    timestamp: datetime
    model_version: str
    data_size: int

@dataclass
class PerformanceAlert:
    """Data class to store performance alert information"""
    alert_type: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str
    timestamp: datetime
    message: str

class ModelPerformanceMonitor:
    """
    Monitor model performance metrics and detect degradation
    """
    
    def __init__(self, config_path: str = "configs/monitoring_config.yaml"):
        """
        Initialize the model performance monitor
        
        Args:
            config_path: Path to monitoring configuration file
        """
        self.config = load_config(config_path)
        self.performance_config = self.config.get('performance_monitoring', {})
        self.thresholds = self.performance_config.get('thresholds', {})
        self.alert_config = self.performance_config.get('alerts', {})
        
        # Initialize performance history
        self.performance_history: List[PerformanceMetrics] = []
        self.alerts: List[PerformanceAlert] = []
        
        # MLflow setup
        self.mlflow_tracking_uri = self.config.get('mlflow', {}).get('tracking_uri')
        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        logger.info("ModelPerformanceMonitor initialized")
    
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        try:
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
            
            # ROC AUC if probabilities are provided
            if y_pred_proba is not None:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multi-class classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                     multi_class='ovr', average='weighted')
            
            # Confusion matrix details
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Class-wise metrics
            precision_per_class = precision_score(y_true, y_pred, average=None)
            recall_per_class = recall_score(y_true, y_pred, average=None)
            f1_per_class = f1_score(y_true, y_pred, average=None)
            
            metrics['precision_per_class'] = precision_per_class.tolist()
            metrics['recall_per_class'] = recall_per_class.tolist()
            metrics['f1_per_class'] = f1_per_class.tolist()
            
            logger.info(f"Calculated metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def log_performance(self, 
                       metrics: Dict[str, float], 
                       model_version: str,
                       data_size: int) -> PerformanceMetrics:
        """
        Log performance metrics to history and MLflow
        
        Args:
            metrics: Performance metrics dictionary
            model_version: Version of the model
            data_size: Size of the evaluation dataset
            
        Returns:
            PerformanceMetrics object
        """
        try:
            # Create performance metrics object
            performance_metrics = PerformanceMetrics(
                accuracy=metrics.get('accuracy', 0.0),
                precision=metrics.get('precision', 0.0),
                recall=metrics.get('recall', 0.0),
                f1_score=metrics.get('f1_score', 0.0),
                roc_auc=metrics.get('roc_auc', 0.0),
                timestamp=datetime.now(),
                model_version=model_version,
                data_size=data_size
            )
            
            # Add to history
            self.performance_history.append(performance_metrics)
            
            # Log to MLflow
            with mlflow.start_run():
                mlflow.log_metrics({
                    'accuracy': performance_metrics.accuracy,
                    'precision': performance_metrics.precision,
                    'recall': performance_metrics.recall,
                    'f1_score': performance_metrics.f1_score,
                    'roc_auc': performance_metrics.roc_auc,
                    'data_size': data_size
                })
                mlflow.log_param('model_version', model_version)
                mlflow.log_param('timestamp', performance_metrics.timestamp.isoformat())
            
            logger.info(f"Performance logged for model {model_version}")
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error logging performance: {str(e)}")
            raise
    
    def check_performance_degradation(self, 
                                    current_metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """
        Check for performance degradation and generate alerts
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            List of performance alerts
        """
        alerts = []
        
        try:
            # Check against absolute thresholds
            threshold_checks = [
                ('accuracy', current_metrics.accuracy, self.thresholds.get('min_accuracy', 0.8)),
                ('precision', current_metrics.precision, self.thresholds.get('min_precision', 0.8)),
                ('recall', current_metrics.recall, self.thresholds.get('min_recall', 0.8)),
                ('f1_score', current_metrics.f1_score, self.thresholds.get('min_f1_score', 0.8)),
                ('roc_auc', current_metrics.roc_auc, self.thresholds.get('min_roc_auc', 0.8))
            ]
            
            for metric_name, current_value, threshold in threshold_checks:
                if current_value < threshold:
                    alert = PerformanceAlert(
                        alert_type='THRESHOLD_BREACH',
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold_value=threshold,
                        severity='HIGH',
                        timestamp=datetime.now(),
                        message=f"{metric_name} ({current_value:.4f}) below threshold ({threshold:.4f})"
                    )
                    alerts.append(alert)
            
            # Check for relative degradation
            if len(self.performance_history) >= 2:
                recent_window = self.performance_config.get('comparison_window', 5)
                recent_metrics = self.performance_history[-recent_window:]
                
                if len(recent_metrics) >= 2:
                    # Calculate average of recent metrics
                    avg_accuracy = np.mean([m.accuracy for m in recent_metrics[:-1]])
                    avg_precision = np.mean([m.precision for m in recent_metrics[:-1]])
                    avg_recall = np.mean([m.recall for m in recent_metrics[:-1]])
                    avg_f1 = np.mean([m.f1_score for m in recent_metrics[:-1]])
                    avg_roc_auc = np.mean([m.roc_auc for m in recent_metrics[:-1]])
                    
                    degradation_threshold = self.thresholds.get('degradation_threshold', 0.05)
                    
                    degradation_checks = [
                        ('accuracy', current_metrics.accuracy, avg_accuracy),
                        ('precision', current_metrics.precision, avg_precision),
                        ('recall', current_metrics.recall, avg_recall),
                        ('f1_score', current_metrics.f1_score, avg_f1),
                        ('roc_auc', current_metrics.roc_auc, avg_roc_auc)
                    ]
                    
                    for metric_name, current_value, avg_value in degradation_checks:
                        if avg_value > 0 and (avg_value - current_value) / avg_value > degradation_threshold:
                            alert = PerformanceAlert(
                                alert_type='PERFORMANCE_DEGRADATION',
                                metric_name=metric_name,
                                current_value=current_value,
                                threshold_value=avg_value,
                                severity='MEDIUM',
                                timestamp=datetime.now(),
                                message=f"{metric_name} degraded by {((avg_value - current_value) / avg_value * 100):.2f}%"
                            )
                            alerts.append(alert)
            
            # Add alerts to history
            self.alerts.extend(alerts)
            
            # Log alerts
            for alert in alerts:
                logger.warning(f"Performance Alert: {alert.message}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking performance degradation: {str(e)}")
            raise
    
    def monitor_model_performance(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray] = None,
                                 model_version: str = "unknown") -> Dict:
        """
        Complete model performance monitoring workflow
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_version: Model version identifier
            
        Returns:
            Dictionary containing metrics and alerts
        """
        try:
            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
            
            # Log performance
            performance_metrics = self.log_performance(
                metrics, model_version, len(y_true)
            )
            
            # Check for degradation
            alerts = self.check_performance_degradation(performance_metrics)
            
            # Generate monitoring report
            report = {
                'timestamp': datetime.now().isoformat(),
                'model_version': model_version,
                'metrics': metrics,
                'performance_summary': {
                    'accuracy': performance_metrics.accuracy,
                    'precision': performance_metrics.precision,
                    'recall': performance_metrics.recall,
                    'f1_score': performance_metrics.f1_score,
                    'roc_auc': performance_metrics.roc_auc
                },
                'alerts': [
                    {
                        'type': alert.alert_type,
                        'metric': alert.metric_name,
                        'current_value': alert.current_value,
                        'threshold': alert.threshold_value,
                        'severity': alert.severity,
                        'message': alert.message
                    }
                    for alert in alerts
                ],
                'data_size': len(y_true)
            }
            
            logger.info("Model performance monitoring completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error in model performance monitoring: {str(e)}")
            raise
    
    def get_performance_trends(self, 
                              window_size: int = 10) -> Dict:
        """
        Get performance trends over time
        
        Args:
            window_size: Window size for trend calculation
            
        Returns:
            Dictionary containing trend analysis
        """
        try:
            if len(self.performance_history) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            recent_metrics = self.performance_history[-window_size:]
            
            # Calculate trends
            timestamps = [m.timestamp for m in recent_metrics]
            accuracies = [m.accuracy for m in recent_metrics]
            precisions = [m.precision for m in recent_metrics]
            recalls = [m.recall for m in recent_metrics]
            f1_scores = [m.f1_score for m in recent_metrics]
            roc_aucs = [m.roc_auc for m in recent_metrics]
            
            trends = {
                'time_range': {
                    'start': timestamps[0].isoformat(),
                    'end': timestamps[-1].isoformat()
                },
                'metrics_trends': {
                    'accuracy': {
                        'values': accuracies,
                        'trend': 'increasing' if accuracies[-1] > accuracies[0] else 'decreasing',
                        'change': accuracies[-1] - accuracies[0]
                    },
                    'precision': {
                        'values': precisions,
                        'trend': 'increasing' if precisions[-1] > precisions[0] else 'decreasing',
                        'change': precisions[-1] - precisions[0]
                    },
                    'recall': {
                        'values': recalls,
                        'trend': 'increasing' if recalls[-1] > recalls[0] else 'decreasing',
                        'change': recalls[-1] - recalls[0]
                    },
                    'f1_score': {
                        'values': f1_scores,
                        'trend': 'increasing' if f1_scores[-1] > f1_scores[0] else 'decreasing',
                        'change': f1_scores[-1] - f1_scores[0]
                    },
                    'roc_auc': {
                        'values': roc_aucs,
                        'trend': 'increasing' if roc_aucs[-1] > roc_aucs[0] else 'decreasing',
                        'change': roc_aucs[-1] - roc_aucs[0]
                    }
                }
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating performance trends: {str(e)}")
            raise
    
    def save_monitoring_report(self, 
                              report: Dict, 
                              output_path: str = "reports/monitoring_reports/") -> str:
        """
        Save monitoring report to file
        
        Args:
            report: Monitoring report dictionary
            output_path: Output directory path
            
        Returns:
            Path to saved report
        """
        try:
            import os
            os.makedirs(output_path, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_monitoring_report_{timestamp}.json"
            filepath = os.path.join(output_path, filename)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Monitoring report saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving monitoring report: {str(e)}")
            raise
    
    def get_alert_summary(self, 
                         hours_back: int = 24) -> Dict:
        """
        Get summary of alerts from the last N hours
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            Dictionary containing alert summary
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_alerts = [
                alert for alert in self.alerts 
                if alert.timestamp >= cutoff_time
            ]
            
            # Group alerts by severity
            severity_counts = {}
            for alert in recent_alerts:
                severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            
            # Group alerts by type
            type_counts = {}
            for alert in recent_alerts:
                type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
            
            summary = {
                'time_range': f"Last {hours_back} hours",
                'total_alerts': len(recent_alerts),
                'severity_breakdown': severity_counts,
                'type_breakdown': type_counts,
                'recent_alerts': [
                    {
                        'type': alert.alert_type,
                        'metric': alert.metric_name,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in recent_alerts[-10:]  # Last 10 alerts
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating alert summary: {str(e)}")
            raise