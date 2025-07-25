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

def test_train_model_empty_data():
    from src.models.model_trainer import ModelTrainer
    trainer = ModelTrainer()
    import pandas as pd
    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype='int')
    with pytest.raises(ValueError):
        trainer.train(X_empty, y_empty)

def test_train_model_invalid_data():
    from src.models.model_trainer import ModelTrainer
    trainer = ModelTrainer()
    import pandas as pd
    X_invalid = pd.DataFrame({"feature": ["a", "b", "c"]})
    y_invalid = pd.Series([1, 0, 1])
    with pytest.raises(ValueError):
        trainer.train(X_invalid, y_invalid)
