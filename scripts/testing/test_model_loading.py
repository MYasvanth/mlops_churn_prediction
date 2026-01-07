#!/usr/bin/env python3
"""
Test script to check model loading functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_model_registry():
    """Test model registry functionality"""
    try:
        from src.models.unified_model_registry_fixed import UnifiedModelRegistry

        print("[TEST] Testing model registry...")

        # Initialize registry
        registry = UnifiedModelRegistry()
        print("[PASS] Model registry initialized")

        # List models
        models = registry.list_models()
        print(f"[PASS] Found {len(models)} models")

        if models:
            # Test loading first model
            model_id = models[0]['model_id']
            stage = models[0]['stage']

            print(f"[TEST] Testing model loading: {model_id} (stage: {stage})")

            try:
                model = registry.load_model(model_id, stage)
                print(f"[PASS] Model loaded successfully: {type(model).__name__}")

                # Test model info
                model_info = registry.get_model_info(model_id, stage)
                print(f"[PASS] Model info retrieved: {model_info['model_id']}")

                return True

            except Exception as e:
                print(f"[FAIL] Model loading failed: {str(e)}")
                return False
        else:
            print("[FAIL] No models found in registry")
            return False

    except Exception as e:
        print(f"[FAIL] Model registry test failed: {str(e)}")
        return False

def test_direct_model_access():
    """Test direct model file access"""
    try:
        import joblib
        import json

        print("\n[TEST] Testing direct model file access...")

        # Check model directories
        models_dir = Path("models/staging")
        if not models_dir.exists():
            print("[FAIL] Models directory not found")
            return False

        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        print(f"[PASS] Found {len(model_dirs)} model directories")

        if model_dirs:
            model_dir = model_dirs[0]

            # Check model file
            model_file = model_dir / "model.pkl"
            if model_file.exists():
                print(f"[PASS] Model file exists: {model_file}")

                # Try to load model
                try:
                    model = joblib.load(model_file)
                    print(f"[PASS] Direct model loading successful: {type(model).__name__}")
                except Exception as e:
                    print(f"[FAIL] Direct model loading failed: {str(e)}")
                    return False
            else:
                print(f"[FAIL] Model file not found: {model_file}")
                return False

            # Check metadata file
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                print(f"[PASS] Metadata file exists: {metadata_file}")

                # Try to load metadata
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    print(f"[PASS] Metadata loaded: {metadata['model_id']}")
                except Exception as e:
                    print(f"[FAIL] Metadata loading failed: {str(e)}")
                    return False
            else:
                print(f"[FAIL] Metadata file not found: {metadata_file}")
                return False

            return True
        else:
            print("[FAIL] No model directories found")
            return False

    except Exception as e:
        print(f"[FAIL] Direct model access test failed: {str(e)}")
        return False

def main():
    """Run model loading tests"""
    print("[START] Running Model Loading Tests")
    print("=" * 50)

    tests = [
        ("Model Registry", test_model_registry),
        ("Direct Model Access", test_direct_model_access)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            status = "[PASS]" if result else "[FAIL]"
            print(f"{status} {test_name}\n")
        except Exception as e:
            print(f"[ERROR] in {test_name}: {str(e)}\n")
            results[test_name] = False

    # Summary
    print("=" * 50)
    print("MODEL LOADING TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("[SUCCESS] Model loading tests passed!")
        return True
    else:
        print("[FAILURE] Model loading tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
