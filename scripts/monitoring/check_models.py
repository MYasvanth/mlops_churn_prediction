#!/usr/bin/env python3
"""Check available models and their status"""

import requests

def main():
    try:
        response = requests.get('http://localhost:8000/models')
        models = response.json()
        
        print("Available models:")
        for model in models:
            print(f"  - {model['model_id']}: {model['status']}")
            
        # Try to get a specific model prediction
        if models:
            model_id = models[0]['model_id']
            print(f"\nTesting prediction with model: {model_id}")
            
            sample_features = [0.5] * 20
            response = requests.post(
                f'http://localhost:8000/models/{model_id}/predict',
                json={'features': sample_features}
            )
            
            if response.status_code == 200:
                prediction = response.json()
                print(f"✅ Prediction successful: {prediction}")
            else:
                print(f"❌ Prediction failed: {response.status_code} - {response.text}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
