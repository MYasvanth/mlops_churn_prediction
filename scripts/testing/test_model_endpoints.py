#!/usr/bin/env python3
"""
Test model endpoints and API functionality
"""

import requests
import json
import time
from pathlib import Path

class ModelEndpointTester:
    """Test model serving endpoints"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.health_endpoint = f"{base_url}/health"
        self.predict_endpoint = f"{base_url}/predict"
        self.model_info_endpoint = f"{base_url}/model/info"
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        try:
            response = requests.get(self.health_endpoint, timeout=10)
            if response.status_code == 200:
                print("✅ Health endpoint working")
                return True
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health endpoint error: {e}")
            return False
    
    def test_model_info(self):
        """Test model info endpoint"""
        try:
            response = requests.get(self.model_info_endpoint, timeout=10)
            if response.status_code == 200:
                info = response.json()
                print("✅ Model info endpoint working")
                print(f"   Model: {info.get('model_name', 'Unknown')}")
                print(f"   Version: {info.get('version', 'Unknown')}")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return False
    
    def test_prediction_endpoint(self):
        """Test prediction endpoint"""
        test_data = {
            "customer_id": 12345,
            "tenure": 24,
            "monthly_charges": 65.50,
            "total_charges": 1572.0,
            "gender": "Male",
            "contract_type": "One year",
            "payment_method": "Electronic check"
        }
        
        try:
            response = requests.post(
                self.predict_endpoint,
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Prediction endpoint working")
                print(f"   Prediction: {result.get('prediction')}")
                print(f"   Probability: {result.get('probability')}")
                return True
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all endpoint tests"""
        print("🔍 Testing Model Endpoints...")
        
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Model Info", self.test_model_info),
            ("Prediction", self.test_prediction_endpoint)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}:")
            results[test_name] = test_func()
            time.sleep(1)  # Rate limiting
        
        print("\n" + "="*50)
        print("📋 ENDPOINT TEST SUMMARY")
        print("="*50)
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        return all(results.values())

if __name__ == "__main__":
    tester = ModelEndpointTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All endpoint tests passed!")
    else:
        print("\n❌ Some endpoint tests failed")
