#!/usr/bin/env python3
"""
Validation script for unified model alignment
Tests all unified components and ensures consistency
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.unified_model_interface import UnifiedModelTrainer, UnifiedModelEvaluator
from src.models.unified_model_registry import UnifiedModelRegistry
from src.pipelines.unified_training_pipeline import UnifiedTrainingPipeline
from src.utils.config_loader import load_config

def test_unified_configuration():
    """Test unified configuration loading."""
    print("🔧 Testing unified configuration...")
    
    try:
        config = load_config("configs/model/unified_model_config.yaml")
        
        # Check required sections
        required_sections = ["models", "preprocessing", "evaluation", "registry"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing section: {section}")
                return False
        
        # Check supported models
        supported_models = config.get("models", {}).get("supported_types", [])
        expected_models = ["xgboost", "lightgbm", "random_forest", "logistic_regression", "svm"]
        
        for model in expected_models:
            if model not in supported_models:
                print(f"❌ Missing model type: {model}")
                return False
        
        print("✅ Unified configuration validated")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_unified_interface():
    """Test unified model interface."""
    print("🔧 Testing unified model interface...")
    
    try:
        # Test each model type
        model_types = ["xgboost", "lightgbm", "random_forest", "logistic_regression"]
        
        for model_type in model_types:
            trainer = UnifiedModelTrainer(model_type)
            info = trainer.get_model_info()
            
            if info["model_type"] != model_type:
                print(f"❌ Model type mismatch: {info['model_type']} != {model_type}")
                return False
        
        print("✅ Unified interface validated")
        return True
        
    except Exception as e:
        print(f"❌ Interface test failed: {e}")
        return False

def test_unified_registry():
    """Test unified model registry."""
    print("🔧 Testing unified model registry...")
    
    try:
        registry = UnifiedModelRegistry()
        stats = registry.get_registry_statistics()
        
        # Registry should initialize successfully
        if "total_models" not in stats:
            print("❌ Registry statistics missing")
            return False
        
        print("✅ Unified registry validated")
        return True
        
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_unified_pipeline():
    """Test unified training pipeline."""
    print("🔧 Testing unified training pipeline...")
    
    try:
        pipeline = UnifiedTrainingPipeline()
        
        # Test pipeline initialization
        config = pipeline.config
        if "models" not in config:
            print("❌ Pipeline configuration missing")
            return False
        
        print("✅ Unified pipeline validated")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_model_consistency():
    """Test model consistency across components."""
    print("🔧 Testing model consistency...")
    
    try:
        # Check if all components use the same model types
        config = load_config("configs/model/unified_model_config.yaml")
        supported_models = set(config.get("models", {}).get("supported_types", []))
        
        # Expected models
        expected_models = {"xgboost", "lightgbm", "random_forest", "logistic_regression", "svm"}
        
        if not expected_models.issubset(supported_models):
            missing = expected_models - supported_models
            print(f"❌ Missing models: {missing}")
            return False
        
        print("✅ Model consistency validated")
        return True
        
    except Exception as e:
        print(f"❌ Consistency test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Starting Model Alignment Validation")
    print("=" * 50)
    
    tests = [
        ("Unified Configuration", test_unified_configuration),
        ("Unified Interface", test_unified_interface),
        ("Unified Registry", test_unified_registry),
        ("Unified Pipeline", test_unified_pipeline),
        ("Model Consistency", test_model_consistency)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 Validation Results:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All validation tests passed!")
        print("✅ Model alignment is complete and consistent")
        return 0
    else:
        print("❌ Some validation tests failed")
        print("🔧 Please check the failed components")
        return 1

if __name__ == "__main__":
    sys.exit(main())
