import json
from src.pipelines.unified_training_pipeline import train_single_model

def main():
    """Updated training script using unified interface."""
    print("🚀 Starting unified model training...")
    
    # Train using unified pipeline
    result = train_single_model(
        model_type="xgboost",
        data_path="data/processed/train.csv",
        hyperparameters=None  # Use default parameters
    )
    
    print("✅ Training completed!")
    print(f"   - Model ID: {result['model_id']}")
    print(f"   - Model Type: {result['model_type']}")
    print(f"   - Metrics: {json.dumps(result['metrics'], indent=2)}")
    print(f"   - Model Path: {result['model_path']}")
    print(f"   - Stage: {result['stage']}")

if __name__ == "__main__":
    main()
