import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock

# Add project root to path for all tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path"""
    return project_root

@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def sample_customer_data():
    """Create sample customer churn data for testing"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        'customer_id': range(1, n_samples + 1),
        'tenure': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(18.25, 118.75, n_samples),
        'total_charges': np.random.uniform(18.8, 8684.8, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'senior_citizen': np.random.randint(0, 2, n_samples),
        'partner': np.random.randint(0, 2, n_samples),
        'dependents': np.random.randint(0, 2, n_samples),
        'churn': np.random.randint(0, 2, n_samples)
    }

    return pd.DataFrame(data)

@pytest.fixture
def sample_features():
    """Create sample feature data for model testing"""
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    X = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    })
    return X

@pytest.fixture
def sample_target(sample_features):
    """Create sample target data matching the features"""
    np.random.seed(42)
    return np.random.randint(0, 2, len(sample_features))

@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = Mock()
    model.predict.return_value = np.array([0, 1, 0, 1, 0])
    model.predict_proba.return_value = np.array([
        [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]
    ])
    model.feature_importances_ = np.array([0.1, 0.2, 0.15, 0.05, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01])
    return model

@pytest.fixture
def mock_registry():
    """Create a mock model registry"""
    registry = Mock()
    registry.list_models.return_value = [
        {'model_id': 'xgboost_test', 'stage': 'production', 'model_type': 'xgboost', 'accuracy': 0.85},
        {'model_id': 'lightgbm_test', 'stage': 'staging', 'model_type': 'lightgbm', 'accuracy': 0.82}
    ]
    registry.load_model.return_value = Mock()
    registry.save_model.return_value = True
    registry.delete_model.return_value = True
    return registry

@pytest.fixture
def mock_monitor():
    """Create a mock performance monitor"""
    monitor = Mock()
    monitor.calculate_metrics.return_value = {
        'accuracy': 0.85, 'precision': 0.82, 'recall': 0.87,
        'f1_score': 0.84, 'roc_auc': 0.91
    }
    monitor.monitor_model_performance.return_value = {
        'timestamp': '2024-01-01T00:00:00',
        'model_version': 'test_model',
        'metrics': {'accuracy': 0.85},
        'performance_summary': {'accuracy': 0.85},
        'alerts': []
    }
    return monitor

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Add any cleanup logic here if needed

# Custom markers
def pytest_configure(config):
    """Add custom markers"""
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow (skip in CI)")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")

# Environment detection
@pytest.fixture(scope="session")
def has_gpu():
    """Check if GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

@pytest.fixture(scope="session")
def has_lightgbm():
    """Check if LightGBM is available"""
    try:
        import lightgbm
        return True
    except ImportError:
        return False

@pytest.fixture(scope="session")
def has_evidently():
    """Check if Evidently is available"""
    try:
        import evidently
        return True
    except ImportError:
        return False

# Test data persistence
@pytest.fixture(scope="session")
def test_data_dir(project_root_path):
    """Create and return test data directory"""
    test_data_path = project_root_path / "test_data"
    test_data_path.mkdir(exist_ok=True)
    return test_data_path

@pytest.fixture
def test_model_dir(temp_directory):
    """Create test model directory"""
    model_dir = os.path.join(temp_directory, "models")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

@pytest.fixture
def test_reports_dir(temp_directory):
    """Create test reports directory"""
    reports_dir = os.path.join(temp_directory, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir

# Configuration fixtures
@pytest.fixture
def default_config():
    """Default configuration for testing"""
    return {
        'model_registry': {
            'model_dir': 'models',
            'staging_dir': 'models/staging',
            'production_dir': 'models/production'
        },
        'monitoring': {
            'performance_monitoring': {
                'thresholds': {
                    'min_accuracy': 0.8,
                    'min_precision': 0.8,
                    'degradation_threshold': 0.05
                }
            }
        },
        'mlflow': {
            'tracking_uri': 'sqlite:///test_mlflow.db'
        }
    }

# API test fixtures
@pytest.fixture
def api_client():
    """FastAPI test client fixture"""
    try:
        from fastapi.testclient import TestClient
        from src.deployment.api_endpoints import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI not available")

# Database fixtures (if needed in future)
@pytest.fixture
def mock_db_connection():
    """Mock database connection"""
    connection = Mock()
    connection.execute.return_value = Mock()
    connection.commit.return_value = None
    return connection

# Logging fixtures
@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests"""
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests

# Random seed fixture for reproducible tests
@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests"""
    np.random.seed(42)

# Performance measurement fixture
@pytest.fixture
def performance_timer():
    """Timer fixture for performance measurements"""
    import time
    start_time = None

    def start():
        nonlocal start_time
        start_time = time.time()

    def stop():
        nonlocal start_time
        if start_time is None:
            raise RuntimeError("Timer not started")
        elapsed = time.time() - start_time
        start_time = None
        return elapsed

    class Timer:
        def __enter__(self):
            start()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def elapsed(self):
            return stop()

    return Timer()
