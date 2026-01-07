#!/usr/bin/env python3
"""
Test script for production endpoints with correct stage parameter
"""

import sys
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_with_staging_stage():
    """Test endpoints with explicit staging stage"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing endpoints with stage='staging'")
    print("=" * 50)
    
    # Test model info with staging stage
    try:
        response = requests.get(f"{base_url}/models/xgboost_20250822_013133?stage=staging")
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model info with staging stage: {model_info['model_id']}")
            print(f"   Type: {model_info['model_type']}")
            print(f"   Accuracy: {model_info['performance_metrics'].get('accuracy', 'N/A'):.3f}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Model info test failed: {str(e)}")
        return False

def test_prediction_with_staging():
    """Test prediction with staging stage"""
    base_url = "http://localhost:8000"
    
    try:
        # Sample features (adjust based on actual model requirements)
        sample_features = [1.0] * 10
        
        payload = {
            "features": sample_features,
            "model_id": "xgboost_20250822_013133",
            "stage": "staging"
        }
        
        response = requests.post(f"{base_url}/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction with staging stage successful")
            print(f"   Model: {result['model_id']}")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Probability: {result['probability']:.3f}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction test failed: {str(e)}")
        return False

def test_batch_prediction_with_staging():
    """Test batch prediction with staging stage"""
    base_url = "http://localhost:8000"
    
    try:
        # Sample batch data
        batch_data = [
            [1.0] * 10,
            [0.5] * 10,
            [0.0] * 10
        ]
        
        payload = {
            "data": batch_data,
            "model_id": "xgboost_20250822_013133",
            "stage": "staging"
        }
        
        response = requests.post(f"{base_url}/predict/batch", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch prediction with staging stage successful")
            print(f"   Model: {result['model_id']}")
            print(f"   Predictions: {len(result['predictions'])} samples")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch prediction test failed: {str(e)}")
        return False

def main():
    """Run staging stage tests"""
    print("🚀 Testing Endpoints with Stage='staging'")
    print("=" * 50)
    
    tests = [
        ("Model Info with Staging", test_with_staging_stage),
        ("Prediction with Staging", test_prediction_with_staging),
        ("Batch Prediction with Staging", test_batch_prediction_with_staging)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}\n")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {str(e)}\n")
            results[test_name] = False
    
    # Summary
    print("=" * 50)
    print("📊 STAGING STAGE TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All staging stage tests passed!")
        print("\n💡 Note: The API defaults to 'production' stage, but models are in 'staging'")
        print("   Use ?stage=staging parameter or promote models to production")
        return True
    else:
        print("❌ Some staging stage tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
