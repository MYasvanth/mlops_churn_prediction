import time
import pytest
from src.data.data_loader import DataLoader
from src.features.feature_engineering import ChurnFeatureEngineer
from src.models.model_trainer import ModelTrainer

@pytest.mark.performance
def test_training_time():
    loader = DataLoader()
    df = loader.load_processed_data('train')
    X, y = loader.get_feature_target_split(df)
    
    feature_engineer = ChurnFeatureEngineer()
    feature_engineer.fit(X, y)
    X_transformed = feature_engineer.transform(X)
    
    trainer = ModelTrainer()
    start_time = time.time()
    model = trainer.train(X_transformed, y)
    end_time = time.time()
    
    training_duration = end_time - start_time
    print(f"Training time: {training_duration:.2f} seconds")
    
    # Assert training time is within acceptable limit (e.g., 60 seconds)
    assert training_duration < 60

@pytest.mark.performance
def test_inference_time():
    loader = DataLoader()
    df = loader.load_processed_data('train')
    X, y = loader.get_feature_target_split(df)
    
    feature_engineer = ChurnFeatureEngineer()
    feature_engineer.fit(X, y)
    X_transformed = feature_engineer.transform(X)
    
    trainer = ModelTrainer()
    model = trainer.train(X_transformed, y)
    
    start_time = time.time()
    predictions = model.predict(X_transformed)
    end_time = time.time()
    
    inference_duration = end_time - start_time
    print(f"Inference time: {inference_duration:.2f} seconds")
    
    # Assert inference time is within acceptable limit (e.g., 10 seconds)
    assert inference_duration < 10
