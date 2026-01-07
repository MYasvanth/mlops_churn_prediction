#!/usr/bin/env python3
"""
Monitoring Runner Script
Runs comprehensive monitoring for the churn prediction system
"""

import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.data_drift_monitor import DataDriftMonitor
from src.monitoring.model_performance_monitor import ModelPerformanceMonitor
from src.monitoring.alert_system import AlertManager, AlertType, AlertSeverity
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.data.data_loader import DataLoader

logger = get_logger(__name__)

class MonitoringRunner:
    """Main monitoring runner class"""
    
    def __init__(self, config_path="configs/monitoring/monitoring_config.yaml"):
        """Initialize monitoring runner"""
        self.config = load_config(config_path)
        self.data_drift_monitor = DataDriftMonitor(config_path)
        self.performance_monitor = ModelPerformanceMonitor(config_path)
        self.alert_manager = AlertManager(config_path)
        
        # Load reference data for drift detection
        self.reference_data = None
        self.load_reference_data()
        
        logger.info("MonitoringRunner initialized")
    
    def load_reference_data(self):
        """Load reference data for drift detection"""
        try:
            # Use the reference data path from monitoring configuration
            reference_path = self.config['data_drift_monitoring']['reference_data_path']
            
            if Path(reference_path).exists():
                self.reference_data = pd.read_csv(reference_path)
                self.data_drift_monitor.set_reference_data(self.reference_data)
                logger.info(f"Reference data loaded from {reference_path}")
            else:
                logger.warning(f"Reference data file not found: {reference_path}")
                
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
    
    def run_data_drift_monitoring(self, current_data=None):
        """Run data drift monitoring"""
        try:
            if current_data is None:
                # Load current production data from monitoring configuration
                current_data_path = self.config['data_drift_monitoring']['current_data_path']
                
                if Path(current_data_path).exists():
                    current_data = pd.read_csv(current_data_path)
                else:
                    logger.warning(f"Current data not available for drift monitoring: {current_data_path}")
                    return None
            
            # Run drift detection
            drift_report = self.data_drift_monitor.detect_drift(current_data)
            
            # Generate alert if drift detected
            if drift_report.drift_detected:
                alert = self.alert_manager.create_and_send_alert(
                    alert_type=AlertType.DATA_DRIFT,
                    severity=AlertSeverity.HIGH,
                    title="Data Drift Detected",
                    message=f"Data drift detected with score {drift_report.drift_score:.3f}. "
                           f"Drifted columns: {drift_report.drifted_columns}",
                    source="data_drift_monitor",
                    metadata={
                        'drift_score': drift_report.drift_score,
                        'drifted_columns': drift_report.drifted_columns,
                        'total_columns': drift_report.total_columns,
                        'report_path': drift_report.report_path
                    }
                )
                logger.warning(f"Data drift alert created: {alert}")
            
            return drift_report
            
        except Exception as e:
            logger.error(f"Error in data drift monitoring: {str(e)}")
            return None
    
    def run_performance_monitoring(self, y_true, y_pred, y_pred_proba=None, model_version="production"):
        """Run model performance monitoring"""
        try:
            # Run performance monitoring
            performance_report = self.performance_monitor.monitor_model_performance(
                y_true, y_pred, y_pred_proba, model_version
            )
            
            # Check for performance alerts
            if performance_report.get('alerts'):
                for alert_info in performance_report['alerts']:
                    alert = self.alert_manager.create_and_send_alert(
                        alert_type=AlertType.MODEL_DEGRADATION,
                        severity=AlertSeverity.HIGH if alert_info['severity'] == 'HIGH' else AlertSeverity.MEDIUM,
                        title=f"Model Performance Alert: {alert_info['metric']}",
                        message=alert_info['message'],
                        source="performance_monitor",
                        metadata=alert_info
                    )
                    logger.warning(f"Performance alert created: {alert}")
            
            # Save monitoring report
            report_path = self.performance_monitor.save_monitoring_report(performance_report)
            logger.info(f"Performance monitoring report saved: {report_path}")
            
            return performance_report
            
        except Exception as e:
            logger.error(f"Error in performance monitoring: {str(e)}")
            return None
    
    def run_comprehensive_monitoring(self):
        """Run comprehensive monitoring including data drift and performance"""
        logger.info("🚀 Starting comprehensive monitoring")
        
        results = {}
        
        # 1. Data Drift Monitoring
        logger.info("1. Running data drift monitoring...")
        drift_result = self.run_data_drift_monitoring()
        results['data_drift'] = drift_result
        
        # 2. Performance Monitoring (placeholder - would need actual predictions)
        logger.info("2. Performance monitoring requires actual predictions (skipping for now)")
        
        # 3. Alert Summary
        logger.info("3. Generating alert summary...")
        alert_summary = self.alert_manager.get_alert_statistics()
        results['alert_summary'] = alert_summary
        
        # 4. Performance Trends
        logger.info("4. Generating performance trends...")
        trends = self.performance_monitor.get_performance_trends()
        results['performance_trends'] = trends
        
        logger.info("✅ Comprehensive monitoring completed")
        return results
    
    def run_continuous_monitoring(self, interval_minutes=60):
        """Run monitoring continuously at specified intervals"""
        logger.info(f"Starting continuous monitoring (interval: {interval_minutes} minutes)")
        
        try:
            while True:
                start_time = datetime.now()
                
                logger.info(f"🔄 Monitoring cycle started at {start_time}")
                
                # Run comprehensive monitoring
                results = self.run_comprehensive_monitoring()
                
                # Log results
                cycle_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Monitoring cycle completed in {cycle_time:.2f} seconds")
                
                # Wait for next cycle
                logger.info(f"⏰ Next monitoring cycle in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {str(e)}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run ML model monitoring")
    parser.add_argument('--mode', choices=['single', 'continuous'], default='single',
                       help='Monitoring mode: single run or continuous')
    parser.add_argument('--interval', type=int, default=60,
                       help='Interval in minutes for continuous monitoring')
    parser.add_argument('--config', default='configs/monitoring/monitoring_config.yaml',
                       help='Path to monitoring configuration file')
    
    args = parser.parse_args()
    
    try:
        # Initialize monitoring runner
        runner = MonitoringRunner(args.config)
        
        if args.mode == 'continuous':
            runner.run_continuous_monitoring(args.interval)
        else:
            results = runner.run_comprehensive_monitoring()
            
            # Print summary
            print("\n📊 Monitoring Results Summary:")
            print("=" * 50)
            
            if results['data_drift']:
                drift = results['data_drift']
                print(f"Data Drift: {'DETECTED' if drift.drift_detected else 'No drift'}")
                if drift.drift_detected:
                    print(f"  - Score: {drift.drift_score:.3f}")
                    print(f"  - Drifted Columns: {len(drift.drifted_columns)}/{drift.total_columns}")
            
            alert_summary = results['alert_summary']
            print(f"Alerts: {alert_summary['active_alerts']} active, {alert_summary['total_alerts']} total")
            
            print(f"Performance Trends: Available" if results['performance_trends'] else "Performance Trends: Insufficient data")
            
    except Exception as e:
        logger.error(f"Error running monitoring: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
