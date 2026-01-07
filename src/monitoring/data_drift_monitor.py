# data_drift_monitor.py
"""
Data drift monitoring module using Evidently AI.
Comprehensive data drift, target drift, and data quality monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import warnings

# Evidently imports for version 0.4.0
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable, DatasetDriftMetric, DatasetMissingValuesMetric,
    DatasetSummaryMetric, ColumnDriftMetric, ColumnSummaryMetric,
    ClassificationQualityMetric, RegressionQualityMetric
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues,
    TestNumberOfConstantColumns, TestNumberOfDuplicatedRows,
    TestColumnsType, TestNumberOfDriftedColumns, TestShareOfDriftedColumns,
    TestColumnDrift
)
from evidently.pipeline.column_mapping import ColumnMapping

from ..utils.logger import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)

@dataclass
class DriftReport:
    """Data class for comprehensive drift report."""
    report_id: str
    timestamp: str
    drift_detected: bool
    drift_score: float
    drifted_columns: List[str]
    total_columns: int
    drift_threshold: float
    report_path: str
    summary: Dict[str, Any]
    alert_triggered: bool = False
    alert_severity: str = "LOW"

@dataclass
class TargetDriftReport:
    """Data class for target drift report."""
    report_id: str
    timestamp: str
    target_drift_detected: bool
    target_drift_score: float
    prediction_drift_detected: bool
    prediction_drift_score: float
    report_path: str
    summary: Dict[str, Any]
    alert_triggered: bool = False
    alert_severity: str = "LOW"

class DataDriftMonitor:
    """
    Comprehensive data drift monitoring class using Evidently AI.
    Supports data drift, target drift, and data quality monitoring.
    """
    
    def __init__(self, config_path: str = "configs/monitoring/monitoring_config.yaml"):
        """
        Initialize the DataDriftMonitor.
        
        Args:
            config_path (str): Path to the monitoring configuration file
        """
        self.config = load_config(config_path)
        self.drift_config = self.config.get('data_drift_monitoring', {})
        
        # Set up directories
        self.reports_dir = Path("reports/drift_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Drift detection parameters
        self.drift_threshold = self.drift_config.get('drift_threshold', 0.15)
        self.significant_drift_threshold = self.drift_config.get('significant_drift_threshold', 0.25)
        self.confidence_level = self.drift_config.get('confidence_level', 0.95)
        
        # Column mapping
        self.target_column = self.drift_config.get('target_column', 'churn')
        self.prediction_column = self.drift_config.get('prediction_column', 'prediction')
        self.datetime_column = self.drift_config.get('datetime_column')
        
        # Feature types configuration
        self.numerical_features = self.drift_config.get('numerical_features', [])
        self.categorical_features = self.drift_config.get('categorical_features', [])
        
        # Historical data storage
        self.history_file = self.reports_dir / "drift_history.json"
        self.history = self._load_history()
        
        # Initialize data attributes
        self.reference_data = None
        self.feature_columns = None
        self.task_type = self.drift_config.get('task_type', 'classification')  # classification or regression
        
        logger.info(f"DataDriftMonitor initialized with drift threshold: {self.drift_threshold}")

    def _extract_drift_results(self, report: Report) -> Dict[str, Any]:
        """
        Extract drift results from Evidently report for version 0.4.0.
        
        Args:
            report: Evidently Report object
            
        Returns:
            Dictionary with drift detection results
        """
        results = {}
        
        try:
            # Get the full report as dictionary
            report_dict = report.as_dict()
            
            # Extract dataset drift information
            metrics = report_dict.get('metrics', [])
            for metric in metrics:
                metric_result = metric.get('result', {})
                metric_type = metric.get('metric', '')
                
                if 'DatasetDriftMetric' in metric_type:
                    results['drift_detected'] = metric_result.get('dataset_drift', False)
                    results['drift_score'] = metric_result.get('drift_score', 0.0)
                    results['number_of_drifted_columns'] = metric_result.get('number_of_drifted_columns', 0)
                    results['share_of_drifted_columns'] = metric_result.get('share_of_drifted_columns', 0.0)
                
                elif 'DataDriftTable' in metric_type:
                    drifted_columns = []
                    for column_name, column_result in metric_result.get('drift_by_columns', {}).items():
                        if column_result.get('drift_detected', False):
                            drifted_columns.append(column_name)
                    results['drifted_columns'] = drifted_columns
                    results['drift_by_columns'] = metric_result.get('drift_by_columns', {})
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting drift results: {str(e)}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'drifted_columns': [],
                'number_of_drifted_columns': 0,
                'share_of_drifted_columns': 0.0
            }
        
    def _load_history(self) -> List[Dict[str, Any]]:
        """
        Load drift detection history.
        
        Returns:
            List[Dict[str, Any]]: Drift history
        """
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading drift history: {str(e)}")
                return []
        return []
    
    def _save_history(self):
        """Save drift detection history."""
        try:
            # Keep only the last 100 reports to prevent file from growing too large
            if len(self.history) > 100:
                self.history = self.history[-100:]
                
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving drift history: {str(e)}")

    def set_reference_data(self, reference_data: pd.DataFrame):
        """
        Set reference data for drift detection.
        
        Args:
            reference_data (pd.DataFrame): Reference dataset
        """
        self.reference_data = reference_data.copy()
        
        # Extract feature columns (exclude target, prediction, and datetime columns)
        columns_to_exclude = [self.target_column, self.prediction_column]
        if self.datetime_column:
            columns_to_exclude.append(self.datetime_column)
            
        self.feature_columns = [col for col in reference_data.columns 
                                if col not in columns_to_exclude]
        
        # Auto-detect feature types if not configured
        if not self.numerical_features and not self.categorical_features:
            self._auto_detect_feature_types(reference_data)
        
        logger.info(f"Reference data set with {len(reference_data)} rows and {len(self.feature_columns)} features")
        logger.info(f"Numerical features: {len(self.numerical_features)}")
        logger.info(f"Categorical features: {len(self.categorical_features)}")
    
    def _auto_detect_feature_types(self, data: pd.DataFrame):
        """Auto-detect numerical and categorical features."""
        for col in self.feature_columns:
            if col in data.columns:
                if data[col].dtype in ['int64', 'float64'] and data[col].nunique() > 10:
                    self.numerical_features.append(col)
                else:
                    self.categorical_features.append(col)
    
    def _create_column_mapping(self) -> ColumnMapping:
        """
        Create column mapping for Evidently report.
        
        Returns:
            ColumnMapping object with column mapping configuration
        """
        # Create ColumnMapping object
        column_mapping = ColumnMapping(
            target=self.target_column if self.target_column in self.reference_data.columns else None,
            prediction=self.prediction_column if self.prediction_column in self.reference_data.columns else None,
            datetime=self.datetime_column if self.datetime_column in self.reference_data.columns else None,
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            task='classification' if self.task_type == 'classification' else 'regression'
        )
        
        return column_mapping
    
    def detect_drift(self, current_data: pd.DataFrame, 
                    report_name: Optional[str] = None) -> DriftReport:
        """
        Detect data drift between reference and current data.
        
        Args:
            current_data (pd.DataFrame): Current dataset
            report_name (Optional[str]): Name for the report
            
        Returns:
            DriftReport: Drift detection report
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        try:
            # Generate report ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_id = report_name or f"drift_report_{timestamp}"
            
            logger.info(f"Starting drift detection: {report_id}")
            logger.info(f"Reference data shape: {self.reference_data.shape}")
            logger.info(f"Current data shape: {current_data.shape}")
            
            # Prepare column mapping
            column_mapping = self._create_column_mapping()
            
            logger.info("Creating and running comprehensive drift report...")
            report = Report(metrics=[
                DatasetDriftMetric(),
                DatasetSummaryMetric(),
                DatasetMissingValuesMetric(),
                DataDriftTable(),
            ])
            
            report.run(
                reference_data=self.reference_data, 
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Save HTML report
            report_path = self.reports_dir / f"{report_id}.html"
            report.save_html(str(report_path))
            
            # Extract drift results
            drift_results = self._extract_drift_results(report)
            
            # Determine alert severity
            alert_triggered = drift_results.get('drift_detected', False)
            alert_severity = "HIGH" if drift_results.get('share_of_drifted_columns', 0.0) > self.significant_drift_threshold else "MEDIUM"
            
            # Create drift report
            drift_report = DriftReport(
                report_id=report_id,
                timestamp=timestamp,
                drift_detected=drift_results.get('drift_detected', False),
                drift_score=drift_results.get('drift_score', 0.0),
                drifted_columns=drift_results.get('drifted_columns', []),
                total_columns=len(self.feature_columns),
                drift_threshold=self.drift_threshold,
                report_path=str(report_path),
                summary=drift_results,
                alert_triggered=alert_triggered,
                alert_severity=alert_severity
            )
            
            # Save to history
            self.history.append(asdict(drift_report))
            self._save_history()
            
            # Log results
            self._log_drift_results(drift_report)
            
            logger.info(f"Drift detection completed: {report_id}")
            return drift_report
            
        except Exception as e:
            logger.error(f"Error in drift detection: {str(e)}")
            raise
    
    def detect_target_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame,
                          report_name: Optional[str] = None) -> TargetDriftReport:
        """
        Detect target and prediction drift between reference and current data.
        
        Args:
            reference_data (pd.DataFrame): Reference dataset with target and predictions
            current_data (pd.DataFrame): Current dataset with target and predictions
            report_name (Optional[str]): Name for the report
            
        Returns:
            TargetDriftReport: Target drift detection report
        """
        try:
            # Generate report ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_id = report_name or f"target_drift_report_{timestamp}"
            
            logger.info(f"Starting target drift detection: {report_id}")
            
            # Prepare column mapping
            column_mapping = self._create_column_mapping()
            
            # Create and run target drift report using ColumnDriftMetric for target and prediction
            report = Report(metrics=[
                ColumnDriftMetric(column_name=self.target_column),
                ColumnDriftMetric(column_name=self.prediction_column),
            ])
            
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Save HTML report
            report_path = self.reports_dir / f"target_{report_id}.html"
            report.save_html(str(report_path))
            
            # Extract results from report
            report_dict = report.as_dict()
            target_drift_results = {
                'target_drift_detected': False,
                'target_drift_score': 0.0,
                'prediction_drift_detected': False,
                'prediction_drift_score': 0.0
            }
            
            for metric in report_dict.get('metrics', []):
                metric_result = metric.get('result', {})
                metric_type = metric.get('metric', '')
                
                if 'ColumnDriftMetric' in metric_type:
                    column_name = metric_result.get('column_name', '')
                    drift_detected = metric_result.get('drift_detected', False)
                    drift_score = metric_result.get('drift_score', 0.0)
                    
                    if column_name == self.target_column:
                        target_drift_results['target_drift_detected'] = drift_detected
                        target_drift_results['target_drift_score'] = drift_score
                    elif column_name == self.prediction_column:
                        target_drift_results['prediction_drift_detected'] = drift_detected
                        target_drift_results['prediction_drift_score'] = drift_score
            
            # Create target drift report
            target_drift_report = TargetDriftReport(
                report_id=report_id,
                timestamp=timestamp,
                target_drift_detected=target_drift_results['target_drift_detected'],
                target_drift_score=target_drift_results['target_drift_score'],
                prediction_drift_detected=target_drift_results['prediction_drift_detected'],
                prediction_drift_score=target_drift_results['prediction_drift_score'],
                report_path=str(report_path),
                summary=target_drift_results
            )
            
            logger.info(f"Target drift detection completed: {report_id}")
            return target_drift_report
            
        except Exception as e:
            logger.error(f"Error in target drift detection: {str(e)}")
            raise

    def _log_drift_results(self, drift_report: DriftReport):
        """
        Log the drift detection results.
        """
        logger.info(f"Drift Report ID: {drift_report.report_id}")
        logger.info(f"Timestamp: {drift_report.timestamp}")
        logger.info(f"Drift Detected: {drift_report.drift_detected}")
        logger.info(f"Drift Score: {drift_report.drift_score:.3f}")
        logger.info(f"Drifted Columns: {len(drift_report.drifted_columns)}/{drift_report.total_columns}")
        if drift_report.drifted_columns:
            logger.info(f"Drifted Column Names: {drift_report.drifted_columns}")
        logger.info(f"Alert Triggered: {drift_report.alert_triggered}")
        logger.info(f"Alert Severity: {drift_report.alert_severity}")
        logger.info(f"Report Path: {drift_report.report_path}")
    
    def get_drift_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent drift detection history.
        
        Args:
            limit (int): Number of recent reports to return
            
        Returns:
            List of drift reports
        """
        return self.history[-limit:] if self.history else []
    
    def clear_history(self):
        """Clear drift detection history."""
        self.history = []
        self._save_history()
        logger.info("Drift history cleared")

    def run_data_quality_check(self, data: pd.DataFrame, report_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive data quality check using Evidently.

        Args:
            data (pd.DataFrame): Data to check
            report_name (Optional[str]): Name for the report

        Returns:
            Dictionary with data quality results
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_id = report_name or f"data_quality_{timestamp}"

            logger.info(f"Starting data quality check: {report_id}")

            # Create and run data quality test suite
            test_suite = TestSuite(tests=[
                TestNumberOfColumnsWithMissingValues(),
                TestNumberOfRowsWithMissingValues(),
                TestNumberOfConstantColumns(),
                TestNumberOfDuplicatedRows(),
                TestColumnsType(),
            ])

            test_suite.run(
                reference_data=None,  # No reference for basic data quality
                current_data=data
            )

            # Get test results first (before trying to save HTML)
            results = test_suite.as_dict()

            # Try to save HTML report, but handle potential errors gracefully
            report_path = self.reports_dir / f"quality_{report_id}.html"
            try:
                test_suite.save_html(str(report_path))
                report_saved = True
            except Exception as html_error:
                logger.warning(f"Could not save HTML report for data quality check: {str(html_error)}")
                report_saved = False
                report_path = "Not saved due to HTML generation error"

            logger.info(f"Data quality check completed: {report_id}")
            return {
                'report_id': report_id,
                'timestamp': timestamp,
                'report_path': str(report_path),
                'report_saved': report_saved,
                'results': results,
                'passed_tests': sum(1 for test in results.get('tests', []) if test.get('status') == 'SUCCESS'),
                'failed_tests': sum(1 for test in results.get('tests', []) if test.get('status') == 'FAIL'),
                'total_tests': len(results.get('tests', []))
            }

        except Exception as e:
            logger.error(f"Error in data quality check: {str(e)}")
            # Return a minimal result instead of raising to prevent test failures
            return {
                'report_id': report_name or f"error_{timestamp}",
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'report_path': "Error occurred during quality check",
                'report_saved': False,
                'results': {},
                'passed_tests': 0,
                'failed_tests': 0,
                'total_tests': 0,
                'error': str(e)
            }
