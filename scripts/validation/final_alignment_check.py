#!/usr/bin/env python3
"""
Final comprehensive alignment check for unified model system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def check_all_components():
    """Check all components for alignment."""
    print("🔍 Final Alignment Check")
    print("=" * 50)
    
    checks = []
    
    # 1. Check unified configuration
    try:
        from src.utils.config_loader import load_config
        config = load_config("configs/model/unified_model_config.yaml")
        
        # Check model types
        supported_models = config.get("models", {}).get("supported_types", [])
        expected = ["xgboost", "lightgbm", "random_forest", "logistic_regression", "svm"]
        
        if set(expected).issubset(set(supported_models)):
            print("✅ Model types aligned")
            checks.append(True)
        else:
            print("❌ Model types not aligned")
            checks.append(False)
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        checks.append(False)
    
    # 2. Check unified interface
    try:
        from src.models.unified_model_interface import UnifiedModelTrainer
        trainer = UnifiedModelTrainer("xgboost")
        info = trainer.get_model_info()
        
        if info["model_type"] == "xgboost":
            print("✅ Unified interface aligned")
            checks.append(True)
        else:
            print("❌ Unified interface not aligned")
            checks.append(False)
            
    except Exception as e:
        print(f"❌ Interface error: {e}")
        checks.append(False)
    
    # 3. Check unified registry
    try:
        from src.models.unified_model_registry import UnifiedModelRegistry
        registry = UnifiedModelRegistry()
        print("✅ Unified registry initialized")
        checks.append(True)
    except Exception as e:
        print(f"❌ Registry error: {e}")
        checks.append(False)
    
    # 4. Check unified pipeline
    try:
        from src.pipelines.unified_training_pipeline import UnifiedTrainingPipeline
        pipeline = UnifiedTrainingPipeline()
        config = pipeline.config
        
        if "models" in config and "preprocessing" in config:
            print("✅ Unified pipeline aligned")
            checks.append(True)
        else:
            print("❌ Unified pipeline not aligned")
            checks.append(False)
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        checks.append(False)
    
    # 5. Check ZenML unified pipeline
    try:
        from zenml_pipelines.unified_training_pipeline import UnifiedDataIngestionConfig
        config = UnifiedDataIngestionConfig()
        print("✅ ZenML unified pipeline aligned")
        checks.append(True)
    except Exception as e:
        print(f"❌ ZenML pipeline error: {e}")
        checks.append(False)
    
    # Final result
    print("\n" + "=" * 50)
    if all(checks):
        print("🎉 ALL COMPONENTS ALIGNED SUCCESSFULLY!")
        print("✅ Model alignment is complete and consistent")
        return True
    else:
        print("❌ Some components need alignment")
        return False

if __name__ == "__main__":
    success = check_all_components()
    sys.exit(0 if success else 1)
