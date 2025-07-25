# data_drift_monitor.py
"""
Data drift monitoring module using Evidently AI.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict

# Evidently imports
try:
    from evidently.model_profile.column_mapping import ColumnMapping
except ModuleNotFoundError:
    # fallback import for older versions of evidently
    from evidently.model_profile import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable, DatasetDriftMetric, DatasetMissingValuesMetric,
    DatasetSummaryMetric, ColumnDriftMetric
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues, TestNumberOfRowsWithMissingValues,
    TestNumberOfConstantColumns, TestNumberOfDuplicatedRows,
    TestColumnsType, TestNumberOfDriftedColumns
)

from ..utils.logger import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)

@dataclass
class DriftReport:
    """Data class for drift report."""
    report_id: str
    timestamp: str
    drift_detected: bool
    drift_score: float
    drifted_columns: List[str]
    total_columns: int
    drift_threshold: float
    report_path: str
    summary: Dict[str, Any]

class DataDriftMonitor:
    """
    Data drift monitoring class using Evidently AI.
    """
    
    def __init__(self, config_path: str = "configs/monitoring_config.yaml"):
        """
        Initialize the DataDriftMonitor.
        
        Args:
            config_path (str): Path to the monitoring configuration file
        """
        self.config = load_config(config_path)
        self.drift_config = self.config.get('data_drift', {})
        
        # Set up directories
        self.reports_dir = Path("reports/drift_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Drift detection parameters
        self.drift_threshold = self.drift_config.get('drift_threshold', 0.5)
        self.confidence_level = self.drift_config.get('confidence_level', 0.95)
        self.stattest_threshold = self.drift_config.get('stattest_threshold', 0.1)
        
        # Column mapping
        self.target_column = self.drift_config.get('target_column', 'churn')
        self.prediction_column = self.drift_config.get('prediction_column', 'prediction')
        
        # Historical data storage
        self.history_file = self.reports_dir / "drift_history.json"
        self.history = self._load_history()
        
        # Feature columns (will be set when reference data is provided)
        self.feature_columns = None
        
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
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving drift history: {str(e)}")
    
    def set_reference_data(self, reference_data: pd.DataFrame):
        """
        Set reference data for drift detection.
        
        Args:
            reference_data (pd.DataFrame): Reference dataset
        """
        self.reference_data = reference_data.copy()
        
        # Extract feature columns (exclude target and prediction columns)
        self.feature_columns = [col for col in reference_data.columns 
                               if col not in [self.target_column, self.prediction_column]]
        
        logger.info(f"Reference data set with {len(reference_data)} rows and {len(self.feature_columns)} features")
    
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
            
            # Prepare column mapping
            column_mapping = self._create_column_mapping()
            
            # Create and run report
            report = Report(metrics=[
                DatasetDriftMetric(),
                DatasetSummaryMetric(),
                DatasetMissingValuesMetric(),
                DataDriftTable()
            ])
            
            report.run(reference_data=self.reference_data, 
                      current_data=current_data,
                      column_mapping=column_mapping)
            
            # Save HTML report
            report_path = self.reports_dir / f"{report_id}.html"
            report.save_html(str(report_path))
            
            # Extract drift results
            drift_results = self._extract_drift_results(report)
            
            # Create drift report
            drift_report = DriftReport(
                report_id=report_id,
                timestamp=timestamp,
                drift_detected=drift_results['drift_detected'],
                drift_score=drift_results['drift_score'],
                drifted_columns=drift_results['drifted_columns'],
                total_columns=len(self.feature_columns),
                drift_threshold=self.drift_threshold,
                report_path=str(report_path),
                summary=drift_results['summary']
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

    def _create_column_mapping(self) -> ColumnMapping:
        """
        Create column mapping for Evidently report.
        """
        column_mapping = ColumnMapping()
        if self.target_column in self.reference_data.columns:
            column_mapping.target = self.target_column
        if self.prediction_column in self.reference_data.columns:
            column_mapping.prediction = self.prediction_column
        if self.feature_columns:
            column_mapping.numerical_features = self.feature_columns
        return column_mapping
