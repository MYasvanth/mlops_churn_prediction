import pytest
from src.models import model_trainer
from src.data.data_loader import DataLoader

@pytest.fixture
def sample_data():
    loader = DataLoader()
    df = loader.load_processed_data("train")
    X, y = loader.get_feature_target_split(df)
    return X, y

def test_train_model(sample_data):
    X, y = sample_data
    trainer = model_trainer.ModelTrainer()
    model = trainer.train(X, y)
    assert model is not None
    assert hasattr(model, "predict")
