import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.unified_model_registry_fixed import UnifiedModelRegistry

@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model storage"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def sample_model():
    """Create a sample trained model for testing"""
    from sklearn.ensemble import RandomForestClassifier
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100)
    })
    y = np.random.randint(0, 2, 100)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

@pytest.fixture
def registry_config():
    """Sample registry configuration"""
    return {
        'model_dir': 'models',
        'staging_dir': 'models/staging',
        'production_dir': 'models/production',
        'metadata_file': 'models/metadata.json'
    }

def test_registry_initialization(temp_model_dir, registry_config):
    """Test registry initialization"""
    registry = UnifiedModelRegistry(config=registry_config)
    assert registry is not None
    assert hasattr(registry, 'list_models')
    assert hasattr(registry, 'save_model')
    assert hasattr(registry, 'load_model')

def test_save_and_load_model(temp_model_dir, sample_model, registry_config):
    """Test saving and loading a model"""
    registry = UnifiedModelRegistry(config=registry_config)

    # Save model
    model_id = "test_rf_model"
    metadata = {
        'model_type': 'random_forest',
        'accuracy': 0.85,
        'features': ['feature1', 'feature2', 'feature3']
    }

    save_result = registry.save_model(sample_model, model_id, metadata, stage='staging')
    assert save_result is True

    # Load model
    loaded_model = registry.load_model(model_id, stage='staging')
    assert loaded_model is not None

    # Test loaded model works
    test_data = pd.DataFrame({
        'feature1': [0.5, 0.3],
        'feature2': [0.8, 0.2],
        'feature3': [0.1, 0.9]
    })
    predictions = loaded_model.predict(test_data)
    assert len(predictions) == 2
    assert all(pred in [0, 1] for pred in predictions)

def test_list_models(temp_model_dir, sample_model, registry_config):
    """Test listing models in different stages"""
    registry = UnifiedModelRegistry(config=registry_config)

    # Save models in different stages
    registry.save_model(sample_model, "staging_model", {'stage': 'staging'}, stage='staging')
    registry.save_model(sample_model, "production_model", {'stage': 'production'}, stage='production')

    # List staging models
    staging_models = registry.list_models('staging')
    assert len(staging_models) >= 1
    assert any(model['model_id'] == 'staging_model' for model in staging_models)

    # List production models
    production_models = registry.list_models('production')
    assert len(production_models) >= 1
    assert any(model['model_id'] == 'production_model' for model in production_models)

def test_promote_model(temp_model_dir, sample_model, registry_config):
    """Test promoting model from staging to production"""
    registry = UnifiedModelRegistry(config=registry_config)

    # Save to staging
    model_id = "promote_test_model"
    registry.save_model(sample_model, model_id, {'stage': 'staging'}, stage='staging')

    # Promote to production
    promote_result = registry.promote_to_production(model_id)
    assert promote_result is True

    # Check model exists in production
    production_models = registry.list_models('production')
    assert any(model['model_id'] == model_id for model in production_models)

def test_model_metadata(temp_model_dir, sample_model, registry_config):
    """Test model metadata handling"""
    registry = UnifiedModelRegistry(config=registry_config)

    model_id = "metadata_test_model"
    metadata = {
        'model_type': 'random_forest',
        'accuracy': 0.87,
        'precision': 0.85,
        'recall': 0.82,
        'f1_score': 0.83,
        'features': ['feature1', 'feature2', 'feature3'],
        'training_date': '2024-01-15',
        'version': '1.0.0'
    }

    registry.save_model(sample_model, model_id, metadata, stage='staging')

    # Get model info
    model_info = registry.get_model_info(model_id, stage='staging')
    assert model_info is not None
    assert model_info['model_id'] == model_id
    assert model_info['metadata']['accuracy'] == 0.87
    assert model_info['metadata']['model_type'] == 'random_forest'

def test_delete_model(temp_model_dir, sample_model, registry_config):
    """Test deleting a model"""
    registry = UnifiedModelRegistry(config=registry_config)

    model_id = "delete_test_model"
    registry.save_model(sample_model, model_id, {'stage': 'staging'}, stage='staging')

    # Verify model exists
    staging_models = registry.list_models('staging')
    assert any(model['model_id'] == model_id for model in staging_models)

    # Delete model
    delete_result = registry.delete_model(model_id, stage='staging')
    assert delete_result is True

    # Verify model is gone
    staging_models_after = registry.list_models('staging')
    assert not any(model['model_id'] == model_id for model in staging_models_after)

def test_load_nonexistent_model(temp_model_dir, registry_config):
    """Test loading a model that doesn't exist"""
    registry = UnifiedModelRegistry(config=registry_config)

    with pytest.raises(FileNotFoundError):
        registry.load_model("nonexistent_model", stage='staging')

def test_invalid_stage(temp_model_dir, sample_model, registry_config):
    """Test operations with invalid stage"""
    registry = UnifiedModelRegistry(config=registry_config)

    with pytest.raises(ValueError):
        registry.save_model(sample_model, "test_model", {}, stage='invalid_stage')

def test_model_versioning(temp_model_dir, registry_config):
    """Test model versioning"""
    registry = UnifiedModelRegistry(config=registry_config)

    model_id_base = "version_test_model"

    # Save multiple versions
    for version in ['v1', 'v2', 'v3']:
        model_id = f"{model_id_base}_{version}"
        metadata = {'version': version, 'accuracy': 0.8 + int(version[1]) * 0.02}
        registry.save_model(sample_model, model_id, metadata, stage='staging')

    # List all versions
    staging_models = registry.list_models('staging')
    version_models = [m for m in staging_models if m['model_id'].startswith(model_id_base)]
    assert len(version_models) == 3

    # Check versioning in metadata
    for model in version_models:
        assert 'version' in model['metadata']

def test_concurrent_model_access(temp_model_dir, sample_model, registry_config):
    """Test concurrent model access (basic test)"""
    registry = UnifiedModelRegistry(config=registry_config)

    model_id = "concurrent_test_model"
    registry.save_model(sample_model, model_id, {'stage': 'staging'}, stage='staging')

    # Load model multiple times (simulating concurrent access)
    models = []
    for _ in range(3):
        model = registry.load_model(model_id, stage='staging')
        models.append(model)
        assert model is not None

    assert len(models) == 3

def test_model_file_integrity(temp_model_dir, sample_model, registry_config):
    """Test model file integrity after save/load cycle"""
    registry = UnifiedModelRegistry(config=registry_config)

    model_id = "integrity_test_model"
    original_predictions = sample_model.predict(pd.DataFrame({
        'feature1': [0.1, 0.5, 0.9],
        'feature2': [0.2, 0.6, 0.8],
        'feature3': [0.3, 0.7, 0.4]
    }))

    registry.save_model(sample_model, model_id, {'stage': 'staging'}, stage='staging')
    loaded_model = registry.load_model(model_id, stage='staging')

    loaded_predictions = loaded_model.predict(pd.DataFrame({
        'feature1': [0.1, 0.5, 0.9],
        'feature2': [0.2, 0.6, 0.8],
        'feature3': [0.3, 0.7, 0.4]
    }))

    # Predictions should be identical
    np.testing.assert_array_equal(original_predictions, loaded_predictions)
