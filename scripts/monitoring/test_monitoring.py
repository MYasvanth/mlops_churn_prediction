#!/usr/bin/env python3
"""
Simple test script to verify monitoring components work
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_alert_system():
    """Test alert system functionality"""
    try:
        from src.monitoring.alert_system import AlertManager, AlertType, AlertSeverity
        
        print("✅ Alert system imports successful")
        
        # Test creating alert manager
        alert_manager = AlertManager()
        print("✅ AlertManager initialization successful")
        
        # Test creating alert
        alert = alert_manager.create_alert(
            alert_type=AlertType.DATA_DRIFT,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="This is a test alert",
            source="test_script"
        )
        print(f"✅ Alert creation successful: {alert.id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Alert system test failed: {str(e)}")
        return False

def test_performance_monitor():
    """Test performance monitor functionality"""
    try:
        from src.monitoring.model_performance_monitor import ModelPerformanceMonitor
        
        print("✅ Performance monitor imports successful")
        
        # Test creating performance monitor
        monitor = ModelPerformanceMonitor()
        print("✅ ModelPerformanceMonitor initialization successful")
        
        # Test metrics calculation (dummy data)
        import numpy as np
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        
        metrics = monitor.calculate_metrics(y_true, y_pred)
        print(f"✅ Metrics calculation successful: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitor test failed: {str(e)}")
        return False

def test_streamlit_app():
    """Test Streamlit app imports"""
    try:
        from src.deployment.streamlit_app import ChurnPredictionApp
        
        print("✅ Streamlit app imports successful")
        
        # Test creating app instance
        app = ChurnPredictionApp()
        print("✅ ChurnPredictionApp initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running monitoring system tests...\n")
    
    tests = [
        ("Alert System", test_alert_system),
        ("Performance Monitor", test_performance_monitor),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}\n")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {str(e)}\n")
            results[test_name] = False
    
    # Summary
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Monitoring system is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check dependencies and imports.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
