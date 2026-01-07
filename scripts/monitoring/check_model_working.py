#!/usr/bin/env python3
"""
Model Working Verification Script
This script checks if the trained model is working correctly
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_trainer import ModelTrainer
from src.features.feature_engineering import ChurnFeatureEngineer
from src.data.data_loader import DataLoader
from src.utils.logger import get_logger

# Setup logging
logger = get_logger(__name__)

class ModelChecker:
    """Class to check model working status"""
    
    def __init__(self, model_path=None, data_path=None):
        self.model_path = model_path or "models/production/model.pkl"
        self.data_path = data_path or "data/processed/test_data.csv"
        self.model = None
        self.preprocessor = None
        
    def check_model_exists(self):
        """Check if model file exists"""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return False
        
        model_files = [
            self.model_path,
            self.model_path.replace('.pkl', '_preprocessor.pkl'),
            self.model_path.replace('.pkl', '_metrics.json')
        ]
        
        missing_files = [f for f in model_files if not os.path.exists(f)]
        if missing_files:
            logger.warning(f"Missing model files: {missing_files}")
            return False
            
        logger.info("✅ All model files found")
        return True
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load(self.model_path)
            logger.info("✅ Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def load_preprocessor(self):
        """Load the preprocessor"""
        try:
            preprocessor_path = self.model_path.replace('.pkl', '_preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("✅ Preprocessor loaded successfully")
                return True
            else:
                logger.warning("⚠️  Preprocessor not found, using default")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to load preprocessor: {e}")
            return False
    
    def check_model_prediction(self):
        """Test model prediction with sample data"""
        try:
            # Create sample customer data
            sample_data = pd.DataFrame({
                'customer_id': [1, 2, 3],
                'tenure': [12, 24, 36],
                'monthly_charges': [29.85, 56.95, 70.35],
                'total_charges': [357.2, 1366.8, 2532.6],
                'gender': ['Male', 'Female', 'Male'],
                'contract_type': ['Month-to-month', 'One year', 'Two year'],
                'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer'],
                'churn': [0, 1, 0]
            })
            
            # Prepare features
            feature_engineer = ChurnFeatureEngineer()
            X_sample = sample_data.drop(['customer_id', 'churn'], axis=1)
            
            # Make predictions
            if self.preprocessor:
                X_processed = self.preprocessor.transform(X_sample)
            else:
                # Simple preprocessing
                X_processed = pd.get_dummies(X_sample)
                X_processed = X_processed.fillna(0)
            
            predictions = self.model.predict(X_processed)
            probabilities = self.model.predict_proba(X_processed)
            
            logger.info("✅ Model prediction successful")
            logger.info(f"Predictions: {predictions}")
            logger.info(f"Probabilities: {probabilities}")
            
            # Validate predictions
            if len(predictions) == 3 and all(p in [0, 1] for p in predictions):
                logger.info("✅ Predictions are valid")
                return True
            else:
                logger.error("❌ Invalid predictions")
                return False
                
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return False
    
    def check_model_metrics(self):
        """Check model performance metrics"""
        try:
            metrics_path = self.model_path.replace('.pkl', '_metrics.json')
            if os.path.exists(metrics_path):
                import json
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                logger.info("📊 Model Metrics:")
                for metric, value in metrics.items():
                    logger.info(f"  {metric}: {value}")
                
                # Check if metrics meet minimum thresholds
                min_accuracy = 0.7
                if metrics.get('accuracy', 0) >= min_accuracy:
                    logger.info("✅ Model accuracy meets minimum threshold")
                    return True
                else:
                    logger.warning("⚠️  Model accuracy below threshold")
                    return False
            else:
                logger.warning("⚠️  Metrics file not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to check metrics: {e}")
            return False
    
    def run_full_check(self):
        """Run complete model verification"""
        logger.info("🔍 Starting model working verification...")
        
        checks = [
            ("Model Files", self.check_model_exists),
            ("Model Loading", self.load_model),
            ("Preprocessor Loading", self.load_preprocessor),
            ("Prediction Test", self.check_model_prediction),
            ("Metrics Check", self.check_model_metrics)
        ]
        
        results = {}
        for check_name, check_func in checks:
            logger.info(f"\n📋 {check_name}:")
            results[check_name] = check_func()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📋 MODEL VERIFICATION SUMMARY")
        logger.info("="*50)
        
        all_passed = all(results.values())
        for check_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{check_name}: {status}")
        
        if all_passed:
            logger.info("\n🎉 All checks passed! Model is working correctly")
        else:
            logger.error("\n❌ Some checks failed. Model needs attention")
            
        return all_passed, results

def main():
    """Main function to run model checks"""
    checker = ModelChecker()
    success, results = checker.run_full_check()
    
    if success:
        print("\n🎉 Model is working correctly!")
        return 0
    else:
        print("\n❌ Model verification failed")
        return 1

if __name__ == "__main__":
    exit(main())
