import pytest
from src.data.data_loader import DataLoader
from src.features.feature_engineering import ChurnFeatureEngineer
from src.models.model_trainer import ModelTrainer

@pytest.mark.integration
def test_full_pipeline():
    # Load data
    loader = DataLoader()
    df = loader.load_processed_data('train')
    X, y = loader.get_feature_target_split(df)
    
    # Feature engineering
    feature_engineer = ChurnFeatureEngineer()
    feature_engineer.fit(X, y)
    X_transformed = feature_engineer.transform(X)
    
    # Model training
    trainer = ModelTrainer()
    model = trainer.train(X_transformed, y)
    
    # Basic assertions
    assert model is not None
    assert hasattr(model, 'predict')
    assert X_transformed.shape[0] == X.shape[0]
