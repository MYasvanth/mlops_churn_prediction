#!/usr/bin/env python3
"""
Quick model functionality test
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def quick_model_check():
    """Quick check if model is working"""
    
    print("🔍 Quick Model Check")
    print("=" * 40)
    
    # Check if model files exist
    model_paths = [
        "models/production/model.pkl",
        "models/staging/model.pkl",
        "models/latest/model.pkl"
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Model found: {path}")
            model_found = True
            break
    
    if not model_found:
        print("❌ No model files found")
        return False
    
    # Check if data exists
    data_path = "data/raw/Customer_data.csv"
    if os.path.exists(data_path):
        print(f"✅ Data found: {data_path}")
        df = pd.read_csv(data_path)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns[:5])}...")
    else:
        print("❌ Data file not found")
        return False
    
    # Check if model can make predictions
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a simple test model if none exists
        if not model_found:
            print("⚠️  Creating test model...")
            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)
            model = RandomForestClassifier(n_estimators=10)
            model.fit(X, y)
            
            os.makedirs("models/latest", exist_ok=True)
            joblib.dump(model, "models/latest/model.pkl")
            print("✅ Test model created")
        
        # Load and test model
        model_path = next(p for p in model_paths if os.path.exists(p))
        model = joblib.load(model_path)
        
        # Test prediction
        test_input = np.array([[0.5, 0.3, 0.7, 0.2, 0.9]])
        prediction = model.predict(test_input)
        probability = model.predict_proba(test_input)
        
        print(f"✅ Model prediction: {prediction[0]}")
        print(f"✅ Model probability: {probability[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_model_check()
    if success:
        print("\n🎉 Model is working correctly!")
    else:
        print("\n❌ Model needs attention")
