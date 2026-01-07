#!/usr/bin/env python3
"""
Monitoring Setup Script
Sets up the monitoring system including directories, databases, and initial configuration
"""

import os
import sys
import logging
from pathlib import Path
import sqlite3
import yaml
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

class MonitoringSetup:
    """Class to set up monitoring infrastructure"""
    
    def __init__(self, config_path="configs/monitoring/monitoring_config.yaml"):
        self.config = load_config(config_path)
        self.storage_config = self.config.get('storage', {})
        
    def create_directories(self):
        """Create necessary directories for monitoring"""
        directories = [
            self.storage_config.get('performance_reports', 'reports/performance_reports'),
            self.storage_config.get('drift_reports', 'reports/drift_reports'),
            'logs',
            'monitoring',
            'data/monitoring'
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        return True
    
    def setup_alert_database(self):
        """Set up alert database"""
        db_path = self.storage_config.get('alerts_database', 'monitoring/alerts.db')
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    alert_type TEXT,
                    severity TEXT,
                    title TEXT,
                    message TEXT,
                    timestamp DATETIME,
                    source TEXT,
                    metadata TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME,
                    resolved_by TEXT
                )
            ''')
            
            # Create performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    model_version TEXT,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    roc_auc REAL,
                    data_size INTEGER
                )
            ''')
            
            # Create drift reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drift_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    drift_score REAL,
                    drift_detected BOOLEAN,
                    drifted_columns TEXT,
                    total_columns INTEGER,
                    report_path TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Alert database setup completed: {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up alert database: {str(e)}")
            return False
    
    def setup_optuna_database(self):
        """Set up Optuna database"""
        optuna_config = self.config.get('optuna', {})
        storage_url = optuna_config.get('storage_url', 'sqlite:///optuna_studies.db')
        
        if storage_url.startswith('sqlite:///'):
            db_path = storage_url.replace('sqlite:///', '')
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Optuna database path: {db_path}")
        
        return True
    
    def create_sample_data(self):
        """Create sample monitoring data for testing"""
        try:
            # Create sample performance data
            sample_performance = {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.87,
                'f1_score': 0.84,
                'roc_auc': 0.89
            }
            
            sample_file = Path('data/monitoring/sample_performance.json')
            sample_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(sample_file, 'w') as f:
                import json
                json.dump(sample_performance, f, indent=2)
            
            logger.info("Sample monitoring data created")
            return True
            
        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")
            return False
    
    def verify_configuration(self):
        """Verify monitoring configuration"""
        required_configs = [
            'data_drift_monitoring',
            'performance_monitoring', 
            'alerts',
            'storage'
        ]
        
        missing_configs = []
        for config_key in required_configs:
            if config_key not in self.config:
                missing_configs.append(config_key)
        
        if missing_configs:
            logger.warning(f"Missing configuration sections: {missing_configs}")
            return False
        
        logger.info("Configuration verification passed")
        return True
    
    def setup_logging(self):
        """Set up logging configuration"""
        logging_config = self.config.get('logging', {})
        log_file = logging_config.get('file', 'logs/monitoring.log')
        
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Logging setup completed: {log_file}")
        return True
    
    def run_setup(self):
        """Run complete monitoring setup"""
        logger.info("🚀 Starting monitoring setup...")
        
        steps = [
            ("Creating directories", self.create_directories),
            ("Setting up logging", self.setup_logging),
            ("Verifying configuration", self.verify_configuration),
            ("Setting up alert database", self.setup_alert_database),
            ("Setting up Optuna database", self.setup_optuna_database),
            ("Creating sample data", self.create_sample_data)
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            try:
                logger.info(f"Step: {step_name}")
                result = step_func()
                results[step_name] = result
                if result:
                    logger.info(f"✅ {step_name} - SUCCESS")
                else:
                    logger.warning(f"⚠️ {step_name} - FAILED")
            except Exception as e:
                logger.error(f"❌ {step_name} - ERROR: {str(e)}")
                results[step_name] = False
        
        # Summary
        success_count = sum(1 for result in results.values() if result)
        total_count = len(results)
        
        logger.info(f"\n📊 Setup Summary: {success_count}/{total_count} steps completed successfully")
        
        if success_count == total_count:
            logger.info("🎉 Monitoring setup completed successfully!")
            return True
        else:
            logger.warning("⚠️ Monitoring setup completed with some issues")
            return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up monitoring system")
    parser.add_argument('--config', default='configs/monitoring/monitoring_config.yaml',
                       help='Path to monitoring configuration file')
    parser.add_argument('--skip-db', action='store_true',
                       help='Skip database setup')
    
    args = parser.parse_args()
    
    try:
        setup = MonitoringSetup(args.config)
        
        if args.skip_db:
            # Only create directories and verify config
            setup.create_directories()
            setup.setup_logging()
            setup.verify_configuration()
            logger.info("Basic setup completed (databases skipped)")
        else:
            setup.run_setup()
            
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
