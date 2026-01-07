#!/usr/bin/env python3
"""
Comprehensive test script to verify unified model alignment
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unified_model_interface import UnifiedModelTrainer, UnifiedModelEvaluator
from src.models.unified_model_registry import UnifiedModelRegistry
from src.pipelines.unified_training_pipeline import UnifiedTrainingPipeline
from src.utils.config_loader import load_config

def test_model_consistency():
    """Test that all model types are consistently supported."""
    print("🔧 Testing model consistency...")
    
    config = load_config("configs/model/unified_model_config.yaml")
    supported_models = config.get("models", {}).get("supported_types", [])
    
    expected_models = {"xgboost", "lightgbm", "random_forest", "logistic_regression", "svm"}
    actual_models = set(supported_models)
    
    if not expected_models.issubset(actual_models):
        missing = expected_models - actual_models
        print(f"❌ Missing models: {missing}")
        return False
    
    # Test each model type
    for model_type in expected_models:
        try:
            trainer = UnifiedModelTrainer(model_type)
            info = trainer.get_model_info()
            if info["model_type"] != model_type:
                print(f"❌ Model type mismatch for {model_type}")
                return False
        except Exception as e:
            print(f"❌ Error testing {model_type}: {e}")
            return False
    
    print("✅ Model consistency verified")
    return True

def test_configuration_consistency():
    """Test that configuration is properly unified."""
    print("🔧 Testing configuration consistency...")
    
    try:
        config = load_config("configs/model/unified_model_config.yaml")
        
        # Check all required sections exist
        required_sections = ["models", "preprocessing", "evaluation", "registry"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing section: {section}")
                return False
        
        # Check model configurations
        model_configs = config.get("models", {}).get("model_configs", {})
        for model_type in ["xgboost", "lightgbm", "random_forest", "logistic_regression", "svm"]:
            if model_type not in model_configs:
                print(f"❌ Missing config for {model_type}")
                return False
        
        print("✅ Configuration consistency verified")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_registry_integration():
    """Test unified registry integration."""
    print("🔧 Testing registry integration...")
    
    try:
        registry = UnifiedModelRegistry()
        
        # Test registry initialization
        stats = registry.get_registry_statistics()
        if "total_models" not in stats:
            print("❌ Registry statistics missing")
            return False
        
        # Test MLflow integration
        import mlflow
        if not mlflow.get_tracking_uri():
            print("❌ MLflow not configured")
            return False
        
        print("✅ Registry integration verified")
        return True
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_pipeline_integration():
    """Test unified pipeline integration."""
    print("🔧 Testing pipeline integration...")
    
    try:
        pipeline = UnifiedTrainingPipeline()
        
        # Test pipeline initialization
        config = pipeline.config
        if "models" not in config:
            print("❌ Pipeline configuration missing")
            return False
        
        # Test supported models
        supported_models = config.get("models", {}).get("supported_types", [])
        if len(supported_models) < 3:
            print(f"❌ Insufficient supported models: {len(supported_models)}")
            return False
        
        print("✅ Pipeline integration verified")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_end_to_end_workflow():
    """Test end-to-end workflow with sample data."""
    print("🔧 Testing end-to-end workflow...")
    
    try:
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        sample_data = pd.DataFrame({
            'tenure': np.random.randint(1, 72, n_samples),
            'monthly_charges': np.random.uniform(20, 120, n_samples),
            'total_charges': np.random.uniform(100, 5000, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'contract_type':
