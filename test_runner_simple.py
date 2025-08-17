#!/usr/bin/env python3
"""
Simple test runner for basic functionality validation.

This script runs basic validation tests without requiring pytest.
"""

import sys
import traceback
from pathlib import Path
import tempfile

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test that basic modules can be imported"""
    print("Testing basic imports...")
    
    try:
        from gan_cyber_range.research.experiment_framework import ExperimentFramework
        from gan_cyber_range.security.security_orchestrator import SecurityOrchestrator
        from gan_cyber_range.monitoring.metrics_collector import MetricsCollector
        from gan_cyber_range.scalability.auto_scaler import AutoScaler
        from gan_cyber_range.internationalization.localization_manager import LocalizationManager
        print("✓ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_experiment_framework():
    """Test basic experiment framework functionality"""
    print("Testing experiment framework...")
    
    try:
        from gan_cyber_range.research.experiment_framework import ExperimentFramework, ExperimentConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ExperimentFramework(output_dir=Path(temp_dir))
            
            # Test basic initialization
            assert framework.output_dir.exists()
            assert len(framework.methods) == 0
            assert len(framework.data_generators) == 0
            
            # Test configuration creation
            config = ExperimentConfig(
                name="Test Experiment",
                description="Basic test",
                hypothesis="Test hypothesis",
                success_criteria={'accuracy': 0.8},
                baseline_methods=['baseline'],
                evaluation_metrics=['accuracy'],
                num_trials=3
            )
            
            assert config.name == "Test Experiment"
            assert config.num_trials == 3
            
        print("✓ Experiment framework basic functionality works")
        return True
    except Exception as e:
        print(f"✗ Experiment framework test failed: {e}")
        traceback.print_exc()
        return False

def test_security_orchestrator():
    """Test basic security orchestrator functionality"""
    print("Testing security orchestrator...")
    
    try:
        from gan_cyber_range.security.security_orchestrator import (
            SecurityOrchestrator, SecurityPolicy, SecurityLevel
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "security_config.json"
            orchestrator = SecurityOrchestrator(config_path=config_path)
            
            # Test basic initialization
            assert orchestrator.config_path == config_path
            assert len(orchestrator.active_policies) >= 0
            assert len(orchestrator.security_contexts) == 0
            
            # Test policy creation and registration
            policy = SecurityPolicy(
                name="test_policy",
                description="Test policy",
                security_level=SecurityLevel.MEDIUM,
                session_timeout=3600
            )
            
            orchestrator.register_security_policy(policy)
            assert policy.name in orchestrator.active_policies
            
            # Test context creation
            context = orchestrator.create_security_context(
                user_id="test_user",
                ip_address="127.0.0.1",
                user_agent="Test Agent",
                requested_permissions={"read"}
            )
            
            assert context is not None
            assert context.user_id == "test_user"
            assert "read" in context.permissions
            
        print("✓ Security orchestrator basic functionality works")
        return True
    except Exception as e:
        print(f"✗ Security orchestrator test failed: {e}")
        traceback.print_exc()
        return False

def test_metrics_collector():
    """Test basic metrics collector functionality"""
    print("Testing metrics collector...")
    
    try:
        from gan_cyber_range.monitoring.metrics_collector import MetricsCollector, MetricType
        
        collector = MetricsCollector(enable_system_metrics=False)
        
        # Test basic functionality
        assert collector.collection_interval > 0
        assert not collector.running
        
        # Test metric recording
        collector.record_metric(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.GAUGE
        )
        
        collector.increment_counter("test_counter")
        collector.set_gauge("test_gauge", 100.0)
        
        # Test metric retrieval
        current_metrics = collector.get_current_metrics()
        assert isinstance(current_metrics, dict)
        
        summary = collector.get_metrics_summary()
        assert 'collection_status' in summary
        assert 'metric_counts' in summary
        
        print("✓ Metrics collector basic functionality works")
        return True
    except Exception as e:
        print(f"✗ Metrics collector test failed: {e}")
        traceback.print_exc()
        return False

def test_auto_scaler():
    """Test basic auto-scaler functionality"""
    print("Testing auto-scaler...")
    
    try:
        from gan_cyber_range.scalability.auto_scaler import (
            AutoScaler, ScalingPolicy, ResourceType, ScalingMetrics
        )
        
        scaler = AutoScaler()
        
        # Test basic initialization
        assert not scaler.running
        assert len(scaler.scaling_policies) >= 0
        
        # Test policy creation
        policy = ScalingPolicy(
            name="test_policy",
            resource_type=ResourceType.CONTAINER,
            target_utilization=70.0,
            min_instances=1,
            max_instances=10
        )
        
        scaler.add_scaling_policy(policy)
        assert policy.name in scaler.scaling_policies
        
        # Test metrics
        metrics = scaler.get_current_metrics()
        assert isinstance(metrics, ScalingMetrics)
        assert metrics.cpu_utilization >= 0
        
        print("✓ Auto-scaler basic functionality works")
        return True
    except Exception as e:
        print(f"✗ Auto-scaler test failed: {e}")
        traceback.print_exc()
        return False

def test_localization_manager():
    """Test basic localization manager functionality"""
    print("Testing localization manager...")
    
    try:
        from gan_cyber_range.internationalization.localization_manager import (
            LocalizationManager, LocaleConfig, LocaleRegion
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LocalizationManager(base_path=Path(temp_dir))
            
            # Test basic initialization
            assert manager.base_path.exists()
            assert manager.default_locale == "en-US"
            assert len(manager.locale_configs) > 0
            
            # Test locale registration
            config = LocaleConfig(
                locale_code="test-TEST",
                language_code="test",
                country_code="TEST",
                display_name="Test Language",
                native_name="Test Native",
                region=LocaleRegion.NORTH_AMERICA
            )
            
            manager.register_locale(config)
            assert config.locale_code in manager.locale_configs
            
            # Test translation
            manager.add_translation(
                key="test.message",
                text="Test message",
                locale="test-TEST"
            )
            
            translated = manager.translate("test.message", locale="test-TEST")
            assert translated == "Test message"
            
            # Test current locale
            assert manager.get_current_locale() == "en-US"
            manager.set_current_locale("test-TEST")
            assert manager.get_current_locale() == "test-TEST"
            
        print("✓ Localization manager basic functionality works")
        return True
    except Exception as e:
        print(f"✗ Localization manager test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all basic tests"""
    print("=" * 60)
    print("Running Basic Functionality Tests")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_experiment_framework,
        test_security_orchestrator,
        test_metrics_collector,
        test_auto_scaler,
        test_localization_manager
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} raised exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())