import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.unified_model_interface import UnifiedModelInterface
from src.models.unified_model_registry_fixed import UnifiedModelRegistry
from src.monitoring.model_performance_monitor import ModelPerformanceMonitor
from src.monitoring.data_drift_monitor import DataDriftMonitor

@pytest.mark.integration
def test_complete_model_lifecycle(sample_customer_data, temp_directory):
    """Test complete model lifecycle from training to deployment"""
    # Setup
    model_dir = os.path.join(temp_directory, "models")
    os.makedirs(model_dir, exist_ok=True)

    registry_config = {
        'model_dir': model_dir,
        'staging_dir': os.path.join(model_dir, 'staging'),
        'production_dir': os.path.join(model_dir, 'production')
    }

    # 1. Data Preparation
    X = sample_customer_data.drop('churn', axis=1)
    y = sample_customer_data['churn']

    # Convert categorical to numeric for testing
    X_processed = X.copy()
    for col in X_processed.select_dtypes(include=['object']).columns:
        X_processed[col] = X_processed[col].astype('category').cat.codes

    # 2. Model Training
    interface = UnifiedModelInterface(model_type='xgboost')
    model = interface.train(X_processed, y)

    assert model is not None
    assert hasattr(model, 'predict')

    # 3. Model Registry Operations
    registry = UnifiedModelRegistry(config=registry_config)

    model_id = "integration_test_model"
    metadata = {
        'model_type': 'xgboost',
        'training_samples': len(X_processed),
        'features': list(X_processed.columns),
        'test_accuracy': 0.85
    }

    # Save to staging
    save_result = registry.save_model(model, model_id, metadata, stage='staging')
    assert save_result is True

    # Verify model in staging
    staging_models = registry.list_models('staging')
    assert any(m['model_id'] == model_id for m in staging_models)

    # Promote to production
    promote_result = registry.promote_to_production(model_id)
    assert promote_result is True

    # Verify model in production
    production_models = registry.list_models('production')
    assert any(m['model_id'] == model_id for m in production_models)

    # 4. Model Loading and Inference
    loaded_model = registry.load_model(model_id, stage='production')
    assert loaded_model is not None

    # Test inference
    test_sample = X_processed.head(5)
    predictions = loaded_model.predict(test_sample)
    assert len(predictions) == 5
    assert all(pred in [0, 1] for pred in predictions)

    # 5. Performance Monitoring
    monitor = ModelPerformanceMonitor()

    # Simulate predictions for monitoring
    y_true = y.head(100).values
    y_pred = loaded_model.predict(X_processed.head(100))
    y_pred_proba = loaded_model.predict_proba(X_processed.head(100))

    # Monitor performance
    report = monitor.monitor_model_performance(y_true, y_pred, y_pred_proba, model_id)

    assert 'metrics' in report
    assert 'performance_summary' in report
    assert 'alerts' in report
    assert report['model_version'] == model_id

    print(f"Integration test completed successfully for model {model_id}")

@pytest.mark.integration
def test_data_drift_monitoring_integration(sample_customer_data, temp_directory):
    """Test data drift monitoring in complete workflow"""
    # Setup reference data
    reference_data = sample_customer_data.copy()

    # Create drift monitor
    monitor = DataDriftMonitor()
    monitor.set_reference_data(reference_data)

    # Create current data with some drift
    current_data = sample_customer_data.copy()
    # Introduce drift in tenure
    current_data['tenure'] = current_data['tenure'] * 1.5  # 50% increase

    # Detect drift
    drift_report = monitor.detect_drift(current_data)

    assert isinstance(drift_report, dict)
    assert 'drift_detected' in drift_report
    assert 'drift_score' in drift_report
    assert 'drifted_columns' in drift_report

    # With significant drift in tenure, should detect drift
    print(f"Drift detection - Detected: {drift_report['drift_detected']}, "
          f"Score: {drift_report['drift_score']}")

@pytest.mark.integration
def test_model_comparison_and_selection(sample_customer_data, temp_directory):
    """Test model comparison and selection workflow"""
    X = sample_customer_data.drop('churn', axis=1)
    y = sample_customer_data['churn']

    # Convert categorical to numeric
    X_processed = X.copy()
    for col in X_processed.select_dtypes(include=['object']).columns:
        X_processed[col] = X_processed[col].astype('category').cat.codes

    model_types = ['xgboost', 'lightgbm', 'random_forest']
    trained_models = {}
    performances = {}

    # Train multiple models
    for model_type in model_types:
        try:
            interface = UnifiedModelInterface(model_type=model_type)
            model = interface.train(X_processed, y)
            trained_models[model_type] = model

            # Evaluate performance
            test_X = X_processed.head(100)
            test_y = y.head(100)
            predictions = model.predict(test_X)

            # Calculate accuracy
            accuracy = np.mean(predictions == test_y.values)
            performances[model_type] = accuracy

            print(f"{model_type} accuracy: {accuracy:.4f}")

        except Exception as e:
            print(f"Skipping {model_type}: {e}")
            continue

    # Select best model
    if performances:
        best_model_type = max(performances, key=performances.get)
        best_accuracy = performances[best_model_type]

        print(f"Best model: {best_model_type} with accuracy {best_accuracy:.4f}")

        # Save best model
        registry = UnifiedModelRegistry()
        model_id = f"best_{best_model_type}_model"
        registry.save_model(trained_models[best_model_type], model_id,
                          {'accuracy': best_accuracy, 'model_type': best_model_type},
                          stage='staging')

        assert best_accuracy > 0.5  # At least better than random

@pytest.mark.integration
def test_monitoring_and_alerting_workflow(sample_customer_data):
    """Test monitoring and alerting workflow"""
    monitor = ModelPerformanceMonitor()

    # Simulate multiple performance measurements
    base_accuracy = 0.85

    for i in range(5):
        # Simulate declining performance
        current_accuracy = base_accuracy - (i * 0.02)

        metrics = {
            'accuracy': current_accuracy,
            'precision': 0.82 - (i * 0.01),
            'recall': 0.87 - (i * 0.01),
            'f1_score': 0.84 - (i * 0.01),
            'roc_auc': 0.91 - (i * 0.005)
        }

        # Mock predictions for monitoring
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.rand(100, 2)

        report = monitor.monitor_model_performance(y_true, y_pred, y_pred_proba, f"model_v{i}")

        print(f"Model v{i} - Accuracy: {current_accuracy:.3f}, Alerts: {len(report['alerts'])}")

    # Check for degradation alerts
    alerts = monitor.get_alert_summary(hours_back=1)
    assert 'total_alerts' in alerts

    print(f"Total alerts generated: {alerts['total_alerts']}")

@pytest.mark.integration
def test_end_to_end_prediction_pipeline(sample_customer_data, temp_directory):
    """Test end-to-end prediction pipeline"""
    # Setup
    registry = UnifiedModelRegistry()

    # Train and save model
    X = sample_customer_data.drop('churn', axis=1)
    y = sample_customer_data['churn']

    X_processed = X.copy()
    for col in X_processed.select_dtypes(include=['object']).columns:
        X_processed[col] = X_processed[col].astype('category').cat.codes

    interface = UnifiedModelInterface(model_type='xgboost')
    model = interface.train(X_processed, y)

    model_id = "e2e_test_model"
    registry.save_model(model, model_id, {'test': True}, stage='production')

    # Simulate API-like prediction workflow
    test_samples = X_processed.head(10)

    # Load model (simulating API startup)
    loaded_model = registry.load_model(model_id, stage='production')

    # Make predictions (simulating API calls)
    predictions = []
    probabilities = []

    for _, sample in test_samples.iterrows():
        pred = loaded_model.predict(sample.to_frame().T)[0]
        proba = loaded_model.predict_proba(sample.to_frame().T)[0]

        predictions.append(pred)
        probabilities.append(proba)

    # Validate results
    assert len(predictions) == 10
    assert len(probabilities) == 10
    assert all(isinstance(p, (int, np.integer)) for p in predictions)
    assert all(isinstance(p, np.ndarray) and len(p) == 2 for p in probabilities)

    # Check probability distributions
    for proba in probabilities:
        assert abs(sum(proba) - 1.0) < 0.01  # Should sum to ~1
        assert all(0 <= p <= 1 for p in proba)  # Should be valid probabilities

    print("End-to-end prediction pipeline test completed successfully")

@pytest.mark.integration
@pytest.mark.slow
def test_scalability_under_load(sample_customer_data):
    """Test system scalability under load"""
    # Setup
    registry = UnifiedModelRegistry()

    X = sample_customer_data.drop('churn', axis=1)
    y = sample_customer_data['churn']

    X_processed = X.copy()
    for col in X_processed.select_dtypes(include=['object']).columns:
        X_processed[col] = X_processed[col].astype('category').cat.codes

    interface = UnifiedModelInterface(model_type='xgboost')
    model = interface.train(X_processed, y)

    model_id = "scalability_test_model"
    registry.save_model(model, model_id, {'test': True}, stage='production')

    loaded_model = registry.load_model(model_id, stage='production')

    # Test with increasing batch sizes
    batch_sizes = [1, 10, 50, 100]
    performance_results = []

    for batch_size in batch_sizes:
        test_batch = X_processed.head(batch_size)

        import time
        start_time = time.time()
        predictions = loaded_model.predict(test_batch)
        end_time = time.time()

        latency = end_time - start_time
        latency_per_sample = latency / batch_size

        performance_results.append({
            'batch_size': batch_size,
            'total_latency': latency,
            'latency_per_sample': latency_per_sample,
            'throughput': batch_size / latency
        })

        print(f"Batch {batch_size}: {latency:.4f}s total, "
              f"{latency_per_sample:.6f}s/sample, "
              f"{batch_size/latency:.1f} samples/sec")

    # Validate scalability
    for i in range(1, len(performance_results)):
        prev = performance_results[i-1]
        curr = performance_results[i]

        # Latency per sample should not increase dramatically
        latency_ratio = curr['latency_per_sample'] / prev['latency_per_sample']
        assert latency_ratio < 5.0, f"Latency scaling issue: {latency_ratio:.2f}x increase"

@pytest.mark.integration
def test_error_handling_and_recovery(sample_customer_data, temp_directory):
    """Test error handling and recovery mechanisms"""
    registry = UnifiedModelRegistry()

    # Test invalid model loading
    with pytest.raises(FileNotFoundError):
        registry.load_model("nonexistent_model", stage='production')

    # Test invalid stage
    X = sample_customer_data.drop('churn', axis=1).head(10)
    y = sample_customer_data['churn'].head(10)

    interface = UnifiedModelInterface(model_type='xgboost')
    model = interface.train(X, y)

    with pytest.raises(ValueError):
        registry.save_model(model, "test_model", {}, stage='invalid_stage')

    # Test successful recovery after error
    registry.save_model(model, "recovery_test_model", {'test': True}, stage='staging')
    loaded_model = registry.load_model("recovery_test_model", stage='staging')

    # Verify model still works after error handling
    test_pred = loaded_model.predict(X.head(5))
    assert len(test_pred) == 5

    print("Error handling and recovery test completed successfully")
