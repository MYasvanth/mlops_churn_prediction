import time
import pytest
import psutil
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader
from src.models.unified_model_interface import UnifiedModelInterface
from src.models.unified_model_registry_fixed import UnifiedModelRegistry
from src.monitoring.model_performance_monitor import ModelPerformanceMonitor

@pytest.fixture
def sample_training_data():
    """Create sample training data for performance testing"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    })
    y = np.random.randint(0, 2, n_samples)

    return X, y

@pytest.fixture
def performance_monitor():
    """Create performance monitor instance"""
    return ModelPerformanceMonitor()

@pytest.mark.performance
def test_training_time_unified_interface(sample_training_data):
    """Test training time using unified model interface"""
    X, y = sample_training_data

    # Test different model types
    model_types = ['xgboost', 'lightgbm', 'random_forest']

    for model_type in model_types:
        interface = UnifiedModelInterface(model_type=model_type)

        start_time = time.time()
        model = interface.train(X, y)
        end_time = time.time()

        training_duration = end_time - start_time
        print(f"{model_type} training time: {training_duration:.2f} seconds")

        # Assert reasonable training time (adjust based on model complexity)
        if model_type == 'xgboost':
            assert training_duration < 30  # XGBoost should be relatively fast
        elif model_type == 'lightgbm':
            assert training_duration < 20  # LightGBM is usually faster
        else:  # random_forest
            assert training_duration < 60  # Random Forest can be slower

@pytest.mark.performance
def test_inference_time_unified_interface(sample_training_data):
    """Test inference time using unified model interface"""
    X, y = sample_training_data

    model_types = ['xgboost', 'lightgbm', 'random_forest']
    batch_sizes = [1, 10, 100, 1000]

    for model_type in model_types:
        interface = UnifiedModelInterface(model_type=model_type)
        model = interface.train(X, y)

        for batch_size in batch_sizes:
            # Create test batch
            test_X = X.head(batch_size)

            start_time = time.time()
            predictions = interface.predict(test_X)
            end_time = time.time()

            inference_duration = end_time - start_time
            latency_per_sample = inference_duration / batch_size

            print(f"{model_type} inference - Batch {batch_size}: {inference_duration:.4f}s "
                  f"({latency_per_sample:.6f}s per sample)")

            # Assert reasonable inference time
            assert inference_duration < 5.0  # Max 5 seconds for any batch
            assert latency_per_sample < 0.01  # Max 10ms per sample

@pytest.mark.performance
def test_memory_usage_during_training(sample_training_data):
    """Test memory usage during model training"""
    X, y = sample_training_data

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    interface = UnifiedModelInterface(model_type='xgboost')
    model = interface.train(X, y)

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    print(f"Memory usage - Initial: {initial_memory:.2f}MB, "
          f"Final: {final_memory:.2f}MB, Increase: {memory_increase:.2f}MB")

    # Assert reasonable memory usage (adjust based on data size)
    assert memory_increase < 500  # Max 500MB increase

@pytest.mark.performance
def test_batch_processing_efficiency(sample_training_data):
    """Test efficiency of batch processing vs individual predictions"""
    X, y = sample_training_data
    test_samples = X.head(100)

    interface = UnifiedModelInterface(model_type='xgboost')
    model = interface.train(X, y)

    # Time individual predictions
    start_time = time.time()
    individual_predictions = []
    for _, row in test_samples.iterrows():
        pred = interface.predict(row.to_frame().T)
        individual_predictions.extend(pred)
    individual_time = time.time() - start_time

    # Time batch prediction
    start_time = time.time()
    batch_predictions = interface.predict(test_samples)
    batch_time = time.time() - start_time

    speedup = individual_time / batch_time
    print(f"Batch processing speedup: {speedup:.2f}x "
          f"(Individual: {individual_time:.4f}s, Batch: {batch_time:.4f}s)")

    # Batch processing should be significantly faster
    assert speedup > 2.0
    assert batch_time < individual_time

@pytest.mark.performance
def test_model_loading_performance():
    """Test model loading performance from registry"""
    registry = UnifiedModelRegistry()

    # Create and save a test model
    X = pd.DataFrame({'feature1': np.random.randn(100), 'feature2': np.random.randn(100)})
    y = np.random.randint(0, 2, 100)

    interface = UnifiedModelInterface(model_type='xgboost')
    model = interface.train(X, y)

    model_id = "performance_test_model"
    registry.save_model(model, model_id, {'test': True}, stage='staging')

    # Test loading performance
    start_time = time.time()
    loaded_model = registry.load_model(model_id, stage='staging')
    load_time = time.time() - start_time

    print(f"Model loading time: {load_time:.4f} seconds")

    # Assert reasonable loading time
    assert load_time < 2.0  # Max 2 seconds to load

    # Verify loaded model works
    test_pred = loaded_model.predict(X.head(5))
    assert len(test_pred) == 5

@pytest.mark.performance
def test_monitoring_performance_overhead(sample_training_data, performance_monitor):
    """Test performance overhead of monitoring"""
    X, y = sample_training_data

    interface = UnifiedModelInterface(model_type='xgboost')
    model = interface.train(X, y)

    # Test predictions without monitoring
    test_data = X.head(100)
    start_time = time.time()
    predictions_no_monitoring = model.predict(test_data)
    time_no_monitoring = time.time() - start_time

    # Test predictions with monitoring
    start_time = time.time()
    predictions_with_monitoring = model.predict(test_data)

    # Simulate monitoring overhead
    y_true = np.random.randint(0, 2, 100)
    y_pred = predictions_with_monitoring
    y_pred_proba = model.predict_proba(test_data)

    report = performance_monitor.monitor_model_performance(
        y_true, y_pred, y_pred_proba, "perf_test_model"
    )
    time_with_monitoring = time.time() - start_time

    monitoring_overhead = time_with_monitoring - time_no_monitoring
    overhead_percentage = (monitoring_overhead / time_no_monitoring) * 100

    print(f"Monitoring overhead: {monitoring_overhead:.4f}s "
          f"({overhead_percentage:.1f}% increase)")

    # Monitoring should not add more than 50% overhead
    assert overhead_percentage < 50.0

@pytest.mark.performance
def test_concurrent_prediction_performance(sample_training_data):
    """Test performance under concurrent load simulation"""
    X, y = sample_training_data

    interface = UnifiedModelInterface(model_type='xgboost')
    model = interface.train(X, y)

    import threading
    import queue

    results = queue.Queue()
    num_threads = 4
    predictions_per_thread = 50

    def predict_worker(thread_id):
        thread_predictions = []
        start_time = time.time()

        for i in range(predictions_per_thread):
            # Random sample for prediction
            sample = X.sample(1)
            pred = model.predict(sample)
            thread_predictions.extend(pred)

        end_time = time.time()
        results.put((thread_id, end_time - start_time, len(thread_predictions)))

    # Start concurrent predictions
    threads = []
    start_time = time.time()

    for i in range(num_threads):
        thread = threading.Thread(target=predict_worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join()

    total_time = time.time() - start_time
    total_predictions = num_threads * predictions_per_thread

    print(f"Concurrent predictions: {total_predictions} in {total_time:.4f}s "
          f"({total_predictions/total_time:.1f} pred/sec)")

    # Assert reasonable throughput
    predictions_per_second = total_predictions / total_time
    assert predictions_per_second > 10  # At least 10 predictions per second

@pytest.mark.performance
@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason="CPU affinity not available")
def test_cpu_utilization_during_training(sample_training_data):
    """Test CPU utilization during training (Unix-like systems only)"""
    X, y = sample_training_data

    process = psutil.Process(os.getpid())

    # Get initial CPU usage
    initial_cpu_times = process.cpu_times()

    interface = UnifiedModelInterface(model_type='xgboost')
    model = interface.train(X, y)

    # Get final CPU usage
    final_cpu_times = process.cpu_times()

    user_time = final_cpu_times.user - initial_cpu_times.user
    system_time = final_cpu_times.system - initial_cpu_times.system
    total_cpu_time = user_time + system_time

    print(f"CPU time used - User: {user_time:.2f}s, System: {system_time:.2f}s, "
          f"Total: {total_cpu_time:.2f}s")

    # Assert reasonable CPU usage
    assert total_cpu_time < 60  # Max 60 seconds CPU time

@pytest.mark.performance
def test_scalability_with_data_size():
    """Test how performance scales with data size"""
    data_sizes = [100, 500, 1000, 2000]
    performance_results = []

    for size in data_sizes:
        # Create dataset of specific size
        X = pd.DataFrame({
            f'feature_{i}': np.random.randn(size) for i in range(5)
        })
        y = np.random.randint(0, 2, size)

        interface = UnifiedModelInterface(model_type='xgboost')

        # Measure training time
        start_time = time.time()
        model = interface.train(X, y)
        training_time = time.time() - start_time

        # Measure inference time
        test_size = min(100, size // 10)
        test_X = X.head(test_size)
        start_time = time.time()
        predictions = interface.predict(test_X)
        inference_time = time.time() - start_time

        performance_results.append({
            'data_size': size,
            'training_time': training_time,
            'inference_time': inference_time,
            'inference_per_sample': inference_time / test_size
        })

        print(f"Size {size}: Train {training_time:.2f}s, "
              f"Infer {inference_time:.4f}s ({inference_time/test_size:.6f}s/sample)")

    # Check that performance scales reasonably (not exponentially worse)
    for i in range(1, len(performance_results)):
        prev = performance_results[i-1]
        curr = performance_results[i]

        # Training time shouldn't increase more than 4x when data size doubles
        time_ratio = curr['training_time'] / prev['training_time']
        size_ratio = curr['data_size'] / prev['data_size']

        if size_ratio > 1.5:  # Only check for significant size increases
            assert time_ratio < size_ratio * 2, f"Training time scaled poorly: {time_ratio:.2f}x for {size_ratio:.2f}x data increase"

@pytest.mark.performance
def test_memory_efficiency():
    """Test memory efficiency across different operations"""
    process = psutil.Process(os.getpid())

    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Create data
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000)
    })
    y = np.random.randint(0, 2, 1000)

    # Train model
    interface = UnifiedModelInterface(model_type='xgboost')
    model = interface.train(X, y)

    training_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Make predictions
    predictions = interface.predict(X.head(100))

    prediction_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Clean up
    del model, interface, X, y
    import gc
    gc.collect()

    final_memory = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Memory usage - Initial: {initial_memory:.1f}MB, "
          f"After training: {training_memory:.1f}MB, "
          f"After prediction: {prediction_memory:.1f}MB, "
          f"After cleanup: {final_memory:.1f}MB")

    # Assert memory doesn't grow unbounded
    assert training_memory - initial_memory < 200  # Max 200MB for training
    assert prediction_memory - training_memory < 50  # Max 50MB for prediction
    assert final_memory < training_memory  # Memory should decrease after cleanup
