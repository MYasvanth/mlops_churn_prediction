import pytest
import requests
import json
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.deployment.api_endpoints import app
from src.models.unified_model_registry_fixed import UnifiedModelRegistry

@pytest.fixture
def client():
    """Test client for FastAPI app"""
    from fastapi.testclient import TestClient
    return TestClient(app)

@pytest.fixture
def sample_features():
    """Sample features for testing predictions"""
    return [0.5, 0.3, 0.8, 0.2, 0.1, 0.9, 0.4, 0.6, 0.7, 0.0]  # 10 features

@pytest.fixture
def mock_registry():
    """Mock model registry for testing"""
    with patch('src.deployment.api_endpoints.UnifiedModelRegistry') as mock_reg:
        mock_instance = Mock()
        mock_instance.list_models.return_value = [
            {'model_id': 'xgboost_20250822_013133', 'stage': 'production', 'model_type': 'xgboost'},
            {'model_id': 'lightgbm_20250820_104713', 'stage': 'staging', 'model_type': 'lightgbm'}
        ]
        mock_instance.load_model.return_value = Mock()
        mock_instance.load_model.return_value.predict.return_value = np.array([1])
        mock_instance.load_model.return_value.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_reg.return_value = mock_instance
        yield mock_instance

def test_root_endpoint(client):
    """Test root endpoint returns API information"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Churn Prediction API" in data["message"]
    assert "endpoints" in data

def test_health_endpoint(client, mock_registry):
    """Test health endpoint returns system status"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "available_models" in data
    assert "timestamp" in data

def test_list_models_endpoint(client, mock_registry):
    """Test models listing endpoint"""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "model_id" in data[0]
    assert "stage" in data[0]

def test_single_prediction_endpoint(client, mock_registry, sample_features):
    """Test single prediction endpoint"""
    payload = {
        "features": sample_features,
        "model_id": "xgboost_20250822_013133"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "model_id" in data
    assert "timestamp" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)

def test_batch_prediction_endpoint(client, mock_registry, sample_features):
    """Test batch prediction endpoint"""
    payload = {
        "data": [sample_features, sample_features],  # Two samples
        "model_id": "xgboost_20250822_013133"
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert "model_id" in data
    assert "timestamp" in data
    assert isinstance(data["predictions"], list)
    assert isinstance(data["probabilities"], list)
    assert len(data["predictions"]) == 2
    assert len(data["probabilities"]) == 2

def test_prediction_without_model_id(client, mock_registry, sample_features):
    """Test prediction with default model selection"""
    payload = {
        "features": sample_features
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

def test_invalid_features_format(client):
    """Test prediction with invalid features format"""
    payload = {
        "features": "invalid_format",
        "model_id": "xgboost_20250822_013133"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_model_not_found(client, mock_registry):
    """Test prediction with non-existent model"""
    mock_registry.load_model.side_effect = FileNotFoundError("Model not found")

    payload = {
        "features": [0.5, 0.3, 0.8, 0.2, 0.1, 0.9, 0.4, 0.6, 0.7, 0.0],
        "model_id": "non_existent_model"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_empty_batch_data(client):
    """Test batch prediction with empty data"""
    payload = {
        "data": [],
        "model_id": "xgboost_20250822_013133"
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400

def test_cors_headers(client):
    """Test CORS headers are present"""
    response = client.options("/predict")
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers

def test_api_documentation_available(client):
    """Test API documentation endpoints"""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/redoc")
    assert response.status_code == 200

def test_openapi_schema(client):
    """Test OpenAPI schema is valid"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
