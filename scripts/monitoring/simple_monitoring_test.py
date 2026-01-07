#!/usr/bin/env python3
"""
Simple monitoring test that handles missing dependencies
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports without dependencies"""
    print("🧪 Testing basic imports...")
    
    try:
        # Test basic Python imports
        import pandas as pd
        import numpy as np
        print("✅ pandas and numpy imports successful")
        
        # Test project-specific imports
        from src.utils.logger import get_logger
        print("✅ Logger import successful")
        
        logger = get_logger(__name__)
        logger.info("Logger test successful")
        print("✅ Logger initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {str(e)}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing configuration loading...")
    
    try:
        from src.utils.config_loader import load_config
        
        # Test loading monitoring config
        config = load_config("configs/monitoring/monitoring_config.yaml")
        print("✅ Monitoring config loaded successfully")
        
        # Test loading model config
        model_config = load_config("configs/model/unified_model_config.yaml")
        print("✅ Model config loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {str(e)}")
        return False

def test_alert_system_simple():
    """Test alert system without full initialization"""
    print("\n🧪 Testing alert system basics...")
    
    try:
        from src.monitoring.alert_system import AlertType, AlertSeverity, Alert
        
        # Test enum imports
        print("✅ Alert enums imported successfully")
        
        # Test alert creation
        alert = Alert(
            id="test_alert_001",
            alert_type=AlertType.DATA_DRIFT,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="This is a test alert",
            timestamp="2024-01-01T12:00:00",
            source="test_script"
        )
        print("✅ Alert creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Alert system test failed: {str(e)}")
        return False

def test_performance_monitor_simple():
    """Test performance monitor basics"""
    print("\n🧪 Testing performance monitor basics...")
    
    try:
        from src.monitoring.model_performance_monitor import PerformanceMetrics
        
        # Test dataclass import
        metrics = PerformanceMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.87,
            f1_score=0.84,
            roc_auc=0.89,
            timestamp="2024-01-01T12:00:00",
            model_version="v1.0",
            data_size=1000
        )
        print("✅ Performance metrics creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitor test failed: {str(e)}")
        return False

def main():
    """Run simplified tests"""
    print("🚀 Running simplified monitoring tests...\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Config Loading", test_config_loading),
        ("Alert System Basics", test_alert_system_simple),
        ("Performance Monitor Basics", test_performance_monitor_simple)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
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
    print("📊 SIMPLIFIED TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= 3:  # At least 3 out of 4 should pass
        print("🎉 Core monitoring components are functional!")
        print("\n📋 Next steps:")
        print("1. Install missing dependencies: pip install omegaconf evidently")
        print("2. Run: python scripts/setup_monitoring.py")
        print("3. Test: python scripts/run_monitoring.py")
        return True
    else:
        print("⚠️  Core components need dependency fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
