import pytest
import pandas as pd
from src.features.feature_engineering import ChurnFeatureEngineer
from src.data.data_loader import DataLoader

@pytest.fixture
def sample_data():
    loader = DataLoader()
    df = loader.load_processed_data("train")
    X, y = loader.get_feature_target_split(df)
    return X, y

def test_fit_transform(sample_data):
    X, y = sample_data
    feature_engineer = ChurnFeatureEngineer()
    feature_engineer.fit(X, y)
    X_transformed = feature_engineer.transform(X)
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] > 0

def test_get_feature_importance(sample_data):
    X, y = sample_data

def test_fit_transform_empty_data():
    from src.features.feature_engineering import ChurnFeatureEngineer
    feature_engineer = ChurnFeatureEngineer()
    import pandas as pd
    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype='int')
    feature_engineer.fit(X_empty, y_empty)
    X_transformed = feature_engineer.transform(X_empty)
    assert X_transformed.empty

def test_fit_transform_with_nan(sample_data):
    from src.features.feature_engineering import ChurnFeatureEngineer
    feature_engineer = ChurnFeatureEngineer()
    X, y = sample_data
    X.loc[0, X.columns[0]] = None  # introduce NaN
    feature_engineer.fit(X, y)
    X_transformed = feature_engineer.transform(X)
    assert X_transformed.shape[0] == X.shape[0]
    feature_engineer = ChurnFeatureEngineer()
    feature_engineer.fit(X, y)
    importance = feature_engineer.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) > 0
