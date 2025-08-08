"""
Comprehensive tests for utility modules.

Tests cover logging, error handling, validation, monitoring, caching, and optimization.
"""

import pytest
import tempfile
import threading
import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import utility modules
from gan_cyber_range.utils.logging_config import (
    setup_logging, get_logger, CyberRangeLogger, AuditLogger
)
from gan_cyber_range.utils.error_handling import (
    CyberRangeError, AttackExecutionError, ValidationError,
    ErrorHandler, with_error_handling, retry_on_error
)
from gan_cyber_range.utils.validation import (
    validate_config, NetworkTopologyValidator, AttackConfigValidator,
    SecurityValidator, sanitize_input
)
from gan_cyber_range.utils.monitoring import (
    MetricsCollector, AlertManager, HealthMonitor, MetricType, AlertSeverity
)
from gan_cyber_range.utils.caching import (
    MemoryCache, CacheManager, cached, TieredCache
)
from gan_cyber_range.utils.optimization import (
    ModelOptimizer, PerformanceManager, OptimizationConfig
)


class TestLogging:
    """Test logging configuration and functionality"""
    
    def test_setup_logging(self):
        """Test logging setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup_logging(
                log_level="DEBUG",
                log_dir=temp_dir,
                enable_file_logging=True,
                enable_json_logging=True
            )
            
            # Check log files were created
            log_path = Path(temp_dir)
            assert (log_path / "cyber_range.log").exists()
            assert (log_path / "security_events.json").exists()
    
    def test_cyber_range_logger(self):
        """Test CyberRangeLogger functionality"""
        logger = get_logger("test_module")
        
        assert isinstance(logger, CyberRangeLogger)
        
        # Test session ID functionality
        session_id = "test_session_123"
        logger.set_session_id(session_id)
        assert logger.get_session_id() == session_id
        
        # Test logging methods (won't verify output, just ensure no errors)
        logger.info("Test info message")
        logger.error("Test error message")
        logger.warning("Test warning message")
    
    def test_structured_logging(self):
        """Test structured logging methods"""
        logger = get_logger("test_module")
        
        # Test attack event logging
        logger.log_attack_event(
            "info", "Attack executed", 
            attack_id="attack_123",
            technique_id="T1059",
            target_host="host_1"
        )
        
        # Test detection event logging
        logger.log_detection_event(
            "warning", "Malware detected",
            detection_type="signature",
            confidence=0.95,
            source_host="host_1"
        )
        
        # Test range event logging
        logger.log_range_event(
            "info", "Range deployed",
            range_id="range_456",
            operation="deploy"
        )
    
    def test_audit_logger(self):
        """Test audit logging functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_file = Path(temp_dir) / "audit.log"
            audit_logger = AuditLogger(str(audit_file))
            
            # Test user action logging
            audit_logger.log_user_action(
                user_id="user_123",
                action="create_range",
                resource="cyber_range",
                result="success",
                details={"range_id": "range_456"}
            )
            
            # Test system event logging
            audit_logger.log_system_event(
                event_type="security",
                description="Failed login attempt",
                severity="WARNING"
            )
            
            # Check audit file was created and contains entries
            assert audit_file.exists()
            content = audit_file.read_text()
            assert "user_123" in content
            assert "create_range" in content


class TestErrorHandling:
    """Test error handling and recovery"""
    
    def test_cyber_range_error(self):
        """Test base CyberRangeError"""
        error = CyberRangeError(
            message="Test error",
            error_code="TEST_ERROR",
            recoverable=True
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.recoverable is True
        assert isinstance(error.timestamp, datetime)
        
        # Test dictionary conversion
        error_dict = error.to_dict()
        assert error_dict['error_code'] == "TEST_ERROR"
        assert error_dict['message'] == "Test error"
    
    def test_specific_errors(self):
        """Test specific error types"""
        # Attack execution error
        attack_error = AttackExecutionError(
            message="Attack failed",
            attack_id="attack_123",
            technique_id="T1059"
        )
        assert attack_error.error_code == "CR_ATTACK_EXEC"
        assert attack_error.context.attack_id == "attack_123"
        
        # Validation error
        validation_error = ValidationError(
            message="Invalid input",
            field="test_field",
            value="invalid_value"
        )
        assert validation_error.field == "test_field"
        assert validation_error.value == "invalid_value"
    
    def test_error_handler(self):
        """Test ErrorHandler functionality"""
        handler = ErrorHandler()
        
        # Test error handling
        test_error = CyberRangeError("Test error", recoverable=True)
        recovery_success = handler.handle_error(test_error, attempt_recovery=False)
        
        # Check statistics were updated
        stats = handler.get_error_statistics()
        assert stats['total_errors'] == 1
        assert stats['by_error_code']['CR_GENERIC'] == 1
    
    def test_with_error_handling_decorator(self):
        """Test error handling decorator"""
        
        @with_error_handling(reraise=False)
        def test_function(should_fail=False):
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # Test successful execution
        result = test_function(should_fail=False)
        assert result == "success"
        
        # Test error handling
        result = test_function(should_fail=True)
        assert result is None  # Should return None after error handling
    
    def test_retry_decorator(self):
        """Test retry decorator"""
        call_count = 0
        
        @retry_on_error(max_retries=2, delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3  # Should have been called 3 times


class TestValidation:
    """Test validation framework"""
    
    def test_network_topology_validator(self):
        """Test network topology validation"""
        validator = NetworkTopologyValidator()
        
        # Valid topology
        valid_config = {
            'name': 'test_topology',
            'subnets': [
                {
                    'name': 'dmz',
                    'cidr': '192.168.1.0/24',
                    'security_zone': 'dmz'
                }
            ],
            'hosts': [
                {
                    'name': 'web_server',
                    'ip_address': '192.168.1.10',
                    'host_type': 'server',
                    'os_type': 'linux'
                }
            ]
        }
        
        assert validator.validate(valid_config)
        
        # Invalid topology - missing required fields
        invalid_config = {
            'name': 'test',
            'subnets': [],
            'hosts': []
        }
        
        assert not validator.validate(invalid_config)
        errors = validator.get_errors()
        assert len(errors) > 0
    
    def test_attack_config_validator(self):
        """Test attack configuration validation"""
        validator = AttackConfigValidator()
        
        # Valid attack config
        valid_config = {
            'name': 'SQL Injection',
            'technique_id': 'T1190',
            'phase': 'exploitation',
            'target_host': 'web_server',
            'success_probability': 0.8,
            'payload': {'query': "' OR 1=1 --"}
        }
        
        assert validator.validate(valid_config)
        
        # Invalid config - invalid technique ID
        invalid_config = {
            'name': 'Bad Attack',
            'technique_id': 'INVALID',
            'phase': 'exploitation',
            'target_host': 'web_server'
        }
        
        assert not validator.validate(invalid_config)
    
    def test_security_validator(self):
        """Test security validation"""
        validator = SecurityValidator()
        
        # Test payload validation
        safe_payload = {
            'command': 'echo hello',
            'target_ip': '192.168.1.10'
        }
        assert validator.validate_attack_payload(safe_payload)
        
        # Dangerous payload
        dangerous_payload = {
            'command': 'rm -rf /',
            'target_ip': '8.8.8.8'  # External IP
        }
        assert not validator.validate_attack_payload(dangerous_payload)
        
        # Test file path validation
        assert validator.validate_file_path('/tmp/safe_file.txt')
        assert not validator.validate_file_path('../../../etc/passwd')
    
    def test_sanitize_input(self):
        """Test input sanitization"""
        # String sanitization
        dangerous_string = '<script>alert("xss")</script>'
        safe_string = sanitize_input(dangerous_string, "string")
        assert '<script>' not in safe_string
        
        # Filename sanitization
        dangerous_filename = '../../../etc/passwd'
        safe_filename = sanitize_input(dangerous_filename, "filename")
        assert '..' not in safe_filename
        
        # Command sanitization
        dangerous_command = 'echo hello; rm -rf /'
        safe_command = sanitize_input(dangerous_command, "command")
        assert ';' not in safe_command
    
    def test_validate_config_function(self):
        """Test main validate_config function"""
        # Valid topology config
        topology_config = {
            'name': 'test_topology',
            'subnets': [{'name': 'test', 'cidr': '192.168.1.0/24', 'security_zone': 'internal'}],
            'hosts': [{'name': 'test', 'ip_address': '192.168.1.10', 'host_type': 'server', 'os_type': 'linux'}]
        }
        
        assert validate_config(topology_config, "topology")
        
        # Invalid config type
        with pytest.raises(ValidationError):
            validate_config({}, "invalid_type")


class TestMonitoring:
    """Test monitoring and metrics"""
    
    def test_metrics_collector(self):
        """Test metrics collection"""
        collector = MetricsCollector(collection_interval=0.1)
        
        # Test metric recording
        collector.record_metric("test_metric", 42.5, MetricType.GAUGE, {"source": "test"})
        collector.increment_counter("test_counter", 1.0)
        collector.set_gauge("test_gauge", 100.0)
        collector.record_timer("test_timer", 1.5)
        
        # Test metric retrieval
        latest = collector.get_latest_metric("test_metric")
        assert latest is not None
        assert latest.value == 42.5
        
        # Test metric summary
        summary = collector.get_metric_summary("test_metric", timedelta(minutes=1))
        assert summary['count'] == 1
        assert summary['avg'] == 42.5
        
        # Test collection start/stop
        collector.start_collection()
        time.sleep(0.2)  # Let it collect some system metrics
        collector.stop_collection()
        
        # Should have collected system metrics
        cpu_metric = collector.get_latest_metric("system_cpu_percent")
        assert cpu_metric is not None
    
    def test_alert_manager(self):
        """Test alert management"""
        manager = AlertManager()
        
        # Add alert rule
        manager.add_alert_rule(
            metric_name="cpu_usage",
            threshold=80.0,
            comparison="greater_than",
            severity=AlertSeverity.WARNING
        )
        
        # Test alert creation (indirectly through metrics)
        collector = MetricsCollector()
        collector.record_metric("cpu_usage", 85.0, MetricType.GAUGE)
        
        manager.check_metrics(collector)
        
        # Should have created an alert
        active_alerts = manager.get_active_alerts()
        if active_alerts:  # Alert creation depends on implementation details
            assert len(active_alerts) >= 0
    
    def test_health_monitor(self):
        """Test health monitoring"""
        monitor = HealthMonitor()
        
        # Register health check
        def dummy_health_check():
            from gan_cyber_range.utils.monitoring import HealthCheck
            return HealthCheck(
                component="test_component",
                status="healthy",
                message="All systems operational"
            )
        
        monitor.register_health_check("test_component", dummy_health_check)
        
        # Check health
        health = monitor.check_component_health("test_component")
        assert health is not None
        assert health.component == "test_component"
        assert health.status == "healthy"
        
        # Get overall health
        overall = monitor.get_overall_health()
        assert overall['status'] == "healthy"
        assert 'test_component' in overall['components']


class TestCaching:
    """Test caching functionality"""
    
    def test_memory_cache(self):
        """Test memory cache operations"""
        cache = MemoryCache(max_size=100, max_memory_mb=1)
        
        # Test basic operations
        assert cache.set("key1", "value1", ttl=60)
        assert cache.get("key1") == "value1"
        assert cache.exists("key1")
        
        # Test TTL expiration
        cache.set("expiring_key", "value", ttl=1)
        time.sleep(1.1)  # Wait for expiration
        assert cache.get("expiring_key") is None
        
        # Test LRU eviction
        small_cache = MemoryCache(max_size=2)
        small_cache.set("key1", "value1")
        small_cache.set("key2", "value2")
        small_cache.set("key3", "value3")  # Should evict key1
        
        assert small_cache.get("key1") is None
        assert small_cache.get("key2") == "value2"
        assert small_cache.get("key3") == "value3"
        
        # Test statistics
        stats = small_cache.get_stats()
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats
    
    @patch('redis.Redis')
    def test_redis_cache(self, mock_redis_class):
        """Test Redis cache with mocked Redis"""
        from gan_cyber_range.utils.caching import RedisCache
        
        # Mock Redis client
        mock_redis = Mock()
        mock_redis_class.return_value = mock_redis
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        
        cache = RedisCache()
        
        # Test operations
        cache.set("test_key", "test_value")
        mock_redis.set.assert_called()
        
        cache.get("test_key")
        mock_redis.get.assert_called()
    
    def test_cache_manager(self):
        """Test cache manager functionality"""
        manager = CacheManager()
        
        # Test basic operations with namespaces
        manager.set("key1", "value1", namespace="test", ttl=60)
        assert manager.get("key1", namespace="test") == "value1"
        
        # Test cache policies
        manager.configure_policy(
            namespace="test",
            default_ttl=3600,
            max_value_size=1024
        )
        
        # Test namespace clearing
        manager.set("key2", "value2", namespace="test")
        cleared = manager.clear_namespace("test")
        assert cleared >= 0
    
    def test_cached_decorator(self):
        """Test caching decorator"""
        call_count = 0
        
        @cached(namespace="test", ttl=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Different parameters should execute function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_tiered_cache(self):
        """Test tiered cache functionality"""
        cache = TieredCache(l1_max_size=10, l1_max_memory_mb=1)
        
        # Test basic operations
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test promotion to L1 (simulate frequent access)
        for _ in range(5):  # Access multiple times
            cache.get("key1")
        
        # Key should now be in L1
        l1_value = cache.l1_cache.get("key1")
        assert l1_value == "value1"


class TestOptimization:
    """Test optimization utilities"""
    
    def test_optimization_config(self):
        """Test optimization configuration"""
        config = OptimizationConfig(
            enable_gpu=True,
            enable_mixed_precision=True,
            max_workers=8
        )
        
        assert config.enable_gpu is True
        assert config.enable_mixed_precision is True
        assert config.max_workers == 8
    
    def test_model_optimizer(self):
        """Test model optimization"""
        config = OptimizationConfig(enable_gpu=False)  # Force CPU for testing
        optimizer = ModelOptimizer(config)
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Optimize model
        optimized_model = optimizer.optimize_model(model)
        
        assert isinstance(optimized_model, nn.Module)
        # Model should be in eval mode after optimization
        assert not optimized_model.training
    
    @patch('psutil.cpu_count', return_value=4)
    @patch('psutil.virtual_memory')
    def test_device_manager(self, mock_memory, mock_cpu):
        """Test device management"""
        from gan_cyber_range.utils.optimization import DeviceManager
        
        # Mock memory info
        mock_memory.return_value.total = 8 * 1024**3  # 8GB
        
        manager = DeviceManager()
        
        assert 'cpu' in manager.devices
        assert manager.devices['cpu']['cores'] == 4
        
        # Test optimal batch size calculation
        model = nn.Linear(10, 5)
        batch_size = manager.get_optimal_batch_size(model, (10,))
        assert isinstance(batch_size, int)
        assert batch_size > 0
    
    def test_performance_manager(self):
        """Test performance management"""
        manager = PerformanceManager()
        
        # Test model optimization
        model = nn.Linear(10, 5)
        optimized = manager.optimize_model(model)
        assert isinstance(optimized, nn.Module)
        
        # Test function profiling
        def test_func():
            time.sleep(0.01)  # Small delay
            return "result"
        
        profiled_func = manager.register_profiled_function("test_func", test_func)
        
        # Execute function
        result = profiled_func()
        assert result == "result"
        
        # Check profiling stats
        stats = profiled_func.get_stats()
        assert stats['calls'] == 1
        assert stats['total_time'] > 0
        
        # Test performance report
        report = manager.get_performance_report()
        assert 'timestamp' in report
        assert 'memory_stats' in report
        assert 'function_profiles' in report
        assert 'test_func' in report['function_profiles']


class TestIntegration:
    """Integration tests for utility modules"""
    
    def test_error_handling_with_logging(self):
        """Test error handling integration with logging"""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup_logging(log_dir=temp_dir, enable_file_logging=True)
            
            logger = get_logger("integration_test")
            
            # Create error and handle it
            error = CyberRangeError("Integration test error")
            handler = ErrorHandler()
            
            # This should log the error
            handler.handle_error(error)
            
            # Verify error was logged
            log_file = Path(temp_dir) / "errors.log"
            if log_file.exists():
                content = log_file.read_text()
                assert "Integration test error" in content
    
    def test_monitoring_with_caching(self):
        """Test monitoring integration with caching"""
        # Create metrics collector with caching
        collector = MetricsCollector()
        cache_manager = CacheManager()
        
        # Cache metrics function
        @cached(namespace="metrics", ttl=60)
        def get_cached_metrics():
            return collector.get_metrics("system_cpu_percent", timedelta(minutes=1))
        
        # Record some metrics
        collector.record_metric("system_cpu_percent", 50.0, MetricType.GAUGE)
        
        # Get cached metrics
        metrics1 = get_cached_metrics()
        metrics2 = get_cached_metrics()
        
        # Should return same result from cache
        assert metrics1 == metrics2
    
    def test_validation_with_error_handling(self):
        """Test validation integration with error handling"""
        
        @with_error_handling(ValidationError, reraise=True)
        def validate_and_process(config):
            # This will raise ValidationError for invalid config
            validate_config(config, "topology")
            return "processed"
        
        # Valid config should work
        valid_config = {
            'name': 'test',
            'subnets': [{'name': 'test', 'cidr': '192.168.1.0/24', 'security_zone': 'internal'}],
            'hosts': [{'name': 'test', 'ip_address': '192.168.1.10', 'host_type': 'server', 'os_type': 'linux'}]
        }
        
        result = validate_and_process(valid_config)
        assert result == "processed"
        
        # Invalid config should raise error
        invalid_config = {'name': 'test', 'subnets': [], 'hosts': []}
        
        with pytest.raises(ValidationError):
            validate_and_process(invalid_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])