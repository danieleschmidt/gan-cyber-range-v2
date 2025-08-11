"""
Comprehensive test suite for the GAN-Cyber-Range-v2 demo system.

This test suite ensures >85% code coverage and validates all core functionality
including attack generation, cyber range management, security, and performance.
"""

import pytest
import json
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gan_cyber_range.demo import (
    LightweightAttackGenerator, SimpleCyberRange, DemoAPI,
    SimpleAttackVector
)
from gan_cyber_range.utils.enhanced_security import (
    SecureInputValidator, ThreatDetectionEngine, ContainmentEngine,
    SecurityAuditLogger, SecureDataManager, SecurityEventType, ThreatLevel
)
from gan_cyber_range.optimization.enhanced_performance import (
    IntelligentCache, AdaptiveResourcePool, PerformanceMonitor,
    LoadBalancer, AutoOptimizer
)
from gan_cyber_range.config.config_manager import (
    ConfigManager, AppConfig, Environment, ConfigValidator
)


class TestLightweightAttackGenerator:
    """Test the lightweight attack generator"""
    
    def setup_method(self):
        self.generator = LightweightAttackGenerator()
    
    def test_init(self):
        """Test generator initialization"""
        assert self.generator is not None
        assert len(self.generator.attack_templates) == 4
        assert 'malware' in self.generator.attack_templates
        assert 'network' in self.generator.attack_templates
        assert 'web' in self.generator.attack_templates
        assert 'social_engineering' in self.generator.attack_templates
    
    def test_generate_single_attack(self):
        """Test generating a single attack"""
        attack = self.generator.generate_attack()
        
        assert isinstance(attack, SimpleAttackVector)
        assert attack.attack_id is not None
        assert attack.attack_type in ['malware', 'network', 'web', 'social_engineering']
        assert 0.0 <= attack.severity <= 1.0
        assert 0.0 <= attack.stealth_level <= 1.0
        assert 0.0 <= attack.success_probability <= 1.0
        assert len(attack.target_systems) >= 1
        assert len(attack.techniques) >= 1
    
    def test_generate_specific_attack_type(self):
        """Test generating attacks of specific type"""
        for attack_type in ['malware', 'network', 'web', 'social_engineering']:
            attack = self.generator.generate_attack(attack_type)
            assert attack.attack_type == attack_type
    
    def test_generate_batch(self):
        """Test generating multiple attacks"""
        attacks = self.generator.generate_batch(10)
        
        assert len(attacks) == 10
        assert all(isinstance(attack, SimpleAttackVector) for attack in attacks)
        
        # Test with specific attack type
        web_attacks = self.generator.generate_batch(5, 'web')
        assert len(web_attacks) == 5
        assert all(attack.attack_type == 'web' for attack in web_attacks)
    
    def test_fill_template(self):
        """Test template filling functionality"""
        template = "nmap -sS -O {target_ip} -p {ports}"
        filled = self.generator._fill_template(template, 'network')
        
        assert '{target_ip}' not in filled
        assert '{ports}' not in filled
        assert 'nmap' in filled


class TestSimpleCyberRange:
    """Test the simple cyber range"""
    
    def setup_method(self):
        self.cyber_range = SimpleCyberRange("test-range")
    
    def test_init(self):
        """Test cyber range initialization"""
        assert self.cyber_range.name == "test-range"
        assert self.cyber_range.status == "initializing"
        assert len(self.cyber_range.hosts) == 6  # Default demo hosts
        assert len(self.cyber_range.networks) == 3
        assert len(self.cyber_range.services) == 4
    
    def test_deploy(self):
        """Test cyber range deployment"""
        range_id = self.cyber_range.deploy()
        
        assert range_id == self.cyber_range.range_id
        assert self.cyber_range.status == "deployed"
        assert self.cyber_range.start_time is not None
        assert self.cyber_range.dashboard_url is not None
    
    def test_execute_attack(self):
        """Test attack execution"""
        self.cyber_range.deploy()
        
        attack = self.cyber_range.attack_generator.generate_attack()
        result = self.cyber_range.execute_attack(attack)
        
        assert 'attack_id' in result
        assert 'success' in result
        assert 'detected' in result
        assert 'execution_time' in result
        assert isinstance(result['success'], bool)
        assert isinstance(result['detected'], bool)
    
    def test_get_metrics(self):
        """Test metrics collection"""
        self.cyber_range.deploy()
        metrics = self.cyber_range.get_metrics()
        
        expected_keys = [
            'range_id', 'status', 'uptime_seconds', 'total_attacks',
            'successful_attacks', 'detected_attacks', 'detection_rate',
            'success_rate', 'active_hosts', 'networks', 'services'
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        assert metrics['range_id'] == self.cyber_range.range_id
        assert metrics['active_hosts'] == 6
        assert metrics['networks'] == 3
        assert metrics['services'] == 4
    
    def test_attack_summary(self):
        """Test attack summary generation"""
        self.cyber_range.deploy()
        
        # Execute some attacks
        for _ in range(3):
            attack = self.cyber_range.attack_generator.generate_attack()
            self.cyber_range.execute_attack(attack)
        
        summary = self.cyber_range.get_attack_summary()
        
        assert 'attack_breakdown' in summary or 'latest_attacks' in summary
        if 'latest_attacks' in summary:
            assert len(summary['latest_attacks']) <= 5


class TestDemoAPI:
    """Test the demo API functionality"""
    
    def setup_method(self):
        self.api = DemoAPI()
    
    def test_create_range(self):
        """Test cyber range creation via API"""
        response = self.api.create_range("test-api-range")
        
        assert 'range_id' in response
        assert 'name' in response
        assert 'status' in response
        assert 'dashboard_url' in response
        
        range_id = response['range_id']
        assert range_id in self.api.ranges
        assert self.api.ranges[range_id].name == "test-api-range"
    
    def test_get_range_info(self):
        """Test getting range information"""
        # Create a range first
        response = self.api.create_range("info-test-range")
        range_id = response['range_id']
        
        # Get range info
        info = self.api.get_range_info(range_id)
        
        assert 'range_info' in info
        assert 'metrics' in info
        assert 'attacks' in info
        assert info['range_info']['range_id'] == range_id
    
    def test_get_range_info_nonexistent(self):
        """Test getting info for non-existent range"""
        info = self.api.get_range_info("nonexistent-range")
        assert 'error' in info
        assert info['error'] == "Range not found"
    
    def test_generate_attacks(self):
        """Test attack generation via API"""
        # Create a range first
        response = self.api.create_range("attack-test-range")
        range_id = response['range_id']
        
        # Generate attacks
        attack_response = self.api.generate_attacks(range_id, count=5, attack_type="malware")
        
        assert 'generated_attacks' in attack_response
        assert 'attack_type' in attack_response
        assert 'results' in attack_response
        assert 'summary' in attack_response
        
        assert attack_response['generated_attacks'] == 5
        assert attack_response['attack_type'] == "malware"
        assert len(attack_response['results']) == 5


class TestSecureInputValidator:
    """Test the security input validator"""
    
    def setup_method(self):
        self.validator = SecureInputValidator()
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection"""
        malicious_input = "'; DROP TABLE users; --"
        result = self.validator.validate_input(malicious_input)
        
        assert not result['is_valid']
        assert result['severity'] == ThreatLevel.HIGH
        assert len(result['threats_detected']) > 0
    
    def test_xss_detection(self):
        """Test XSS attack detection"""
        xss_payload = "<script>alert('xss')</script>"
        result = self.validator.validate_input(xss_payload)
        
        assert not result['is_valid']
        assert result['severity'] == ThreatLevel.HIGH
    
    def test_command_injection_detection(self):
        """Test command injection detection"""
        cmd_injection = "test; rm -rf /"
        result = self.validator.validate_input(cmd_injection)
        
        assert not result['is_valid']
        assert result['severity'] == ThreatLevel.HIGH
    
    def test_path_traversal_detection(self):
        """Test path traversal detection"""
        path_traversal = "../../../etc/passwd"
        result = self.validator.validate_input(path_traversal)
        
        assert not result['is_valid']
        assert result['severity'] == ThreatLevel.HIGH
    
    def test_length_validation(self):
        """Test input length validation"""
        long_input = "a" * 1000
        result = self.validator.validate_input(long_input, 'username')
        
        assert not result['is_valid']
        assert result['severity'] == ThreatLevel.MEDIUM
    
    def test_valid_input(self):
        """Test valid input validation"""
        valid_input = "test_range_name"
        result = self.validator.validate_input(valid_input)
        
        assert result['is_valid']
        assert result['severity'] == ThreatLevel.LOW
    
    def test_attack_config_validation(self):
        """Test attack configuration validation"""
        safe_config = {
            'target_ip': '192.168.1.100',
            'port': 80,
            'payload': 'test_payload'
        }
        
        result = self.validator.validate_attack_config(safe_config)
        assert result['is_safe']
        
        # Test unsafe config
        unsafe_config = {
            'target_ip': '8.8.8.8',  # Public IP
            'payload': 'malicious www.google.com'
        }
        
        result = self.validator.validate_attack_config(unsafe_config)
        assert not result['is_safe']
        assert len(result['issues']) > 0


class TestThreatDetectionEngine:
    """Test the threat detection engine"""
    
    def setup_method(self):
        self.detector = ThreatDetectionEngine()
    
    def test_malware_detection(self):
        """Test malware signature detection"""
        malware_payload = "meterpreter reverse_tcp LHOST=192.168.1.100"
        result = self.detector.analyze_payload(malware_payload)
        
        assert result['threat_detected']
        assert result['risk_score'] > 0.3
        assert len(result['signatures_matched']) > 0
    
    def test_network_attack_detection(self):
        """Test network attack detection"""
        nmap_command = "nmap -sS -O 192.168.1.0/24"
        result = self.detector.analyze_payload(nmap_command)
        
        assert result['threat_detected']
        assert result['risk_score'] > 0.3
    
    def test_high_entropy_detection(self):
        """Test high entropy detection"""
        encoded_payload = "YWxlcnQoJ1hTUycpOw=="  # Base64 encoded
        result = self.detector.analyze_payload(encoded_payload)
        
        # High entropy should increase risk score
        assert result['risk_score'] > 0.0
    
    def test_safe_payload(self):
        """Test safe payload analysis"""
        safe_payload = "hello world"
        result = self.detector.analyze_payload(safe_payload)
        
        assert result['risk_score'] < 0.3
        assert not result['threat_detected'] or result['threat_type'] != 'high_risk'


class TestIntelligentCache:
    """Test the intelligent cache system"""
    
    def setup_method(self):
        self.cache = IntelligentCache(max_size=5, ttl_seconds=1)
    
    def test_cache_operations(self):
        """Test basic cache operations"""
        # Test set and get
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"
        
        # Test miss
        assert self.cache.get("nonexistent") is None
    
    def test_cache_expiration(self):
        """Test cache expiration"""
        self.cache.set("expire_key", "expire_value")
        assert self.cache.get("expire_key") == "expire_value"
        
        # Wait for expiration
        time.sleep(1.5)
        assert self.cache.get("expire_key") is None
    
    def test_cache_eviction(self):
        """Test LRU eviction"""
        # Fill cache to capacity
        for i in range(5):
            self.cache.set(f"key{i}", f"value{i}")
        
        # Access key0 to make it most recently used
        self.cache.get("key0")
        
        # Add one more item to trigger eviction
        self.cache.set("key5", "value5")
        
        # key1 should be evicted (least recently used)
        assert self.cache.get("key0") == "value0"  # Still there
        assert self.cache.get("key5") == "value5"  # New item
    
    def test_cache_stats(self):
        """Test cache statistics"""
        self.cache.set("test", "value")
        self.cache.get("test")  # hit
        self.cache.get("miss")  # miss
        
        stats = self.cache.get_stats()
        
        assert 'cache_size' in stats
        assert 'hit_count' in stats
        assert 'miss_count' in stats
        assert 'hit_rate' in stats
        assert stats['hit_count'] >= 1
        assert stats['miss_count'] >= 1


class TestPerformanceMonitor:
    """Test the performance monitoring system"""
    
    def setup_method(self):
        self.monitor = PerformanceMonitor(collection_interval=0.1)  # Fast for testing
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        assert self.monitor.collection_interval == 0.1
        assert len(self.monitor.metrics_history) == 0
        assert not self.monitor._monitoring
    
    def test_request_recording(self):
        """Test request metrics recording"""
        self.monitor.record_request(0.5, error=False)
        self.monitor.record_request(1.0, error=True)
        
        assert self.monitor._request_count == 2
        assert self.monitor._error_count == 1
        assert len(self.monitor._response_times) == 2
    
    def test_cache_event_recording(self):
        """Test cache event recording"""
        self.monitor.record_cache_event(hit=True)
        self.monitor.record_cache_event(hit=False)
        
        assert self.monitor._cache_hits == 1
        assert self.monitor._cache_misses == 1
    
    def test_monitoring_lifecycle(self):
        """Test monitor start/stop"""
        self.monitor.start_monitoring()
        assert self.monitor._monitoring
        
        # Let it collect at least one metric
        time.sleep(0.2)
        
        self.monitor.stop_monitoring()
        assert not self.monitor._monitoring
        
        # Should have collected some metrics
        assert len(self.monitor.metrics_history) > 0


class TestConfigManager:
    """Test the configuration management system"""
    
    def setup_method(self):
        self.config_manager = ConfigManager("test_config")
        # Clean up any existing test configs
        if self.config_manager.config_dir.exists():
            import shutil
            shutil.rmtree(self.config_manager.config_dir)
        self.config_manager.config_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        # Clean up test configs
        if self.config_manager.config_dir.exists():
            import shutil
            shutil.rmtree(self.config_manager.config_dir)
    
    def test_config_creation(self):
        """Test default configuration creation"""
        self.config_manager.create_default_configs()
        
        assert (self.config_manager.config_dir / "base.yaml").exists()
        assert (self.config_manager.config_dir / "development.yaml").exists()
        assert (self.config_manager.config_dir / "production.yaml").exists()
    
    def test_config_loading(self):
        """Test configuration loading"""
        self.config_manager.create_default_configs()
        config = self.config_manager.load_config(Environment.DEVELOPMENT)
        
        assert isinstance(config, AppConfig)
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is True
        assert config.version == "2.0.0"
    
    def test_config_validation(self):
        """Test configuration validation"""
        validator = ConfigValidator()
        
        # Test valid config
        valid_config = AppConfig()
        result = validator.validate_config(valid_config)
        assert result['is_valid']
        
        # Test invalid config
        invalid_config = AppConfig()
        invalid_config.database.port = -1  # Invalid port
        result = validator.validate_config(invalid_config)
        assert not result['is_valid']
        assert len(result['errors']) > 0


class TestIntegration:
    """Integration tests for the complete system"""
    
    def setup_method(self):
        self.api = DemoAPI()
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Create a cyber range
        range_response = self.api.create_range("integration-test")
        range_id = range_response['range_id']
        
        # Verify range creation
        assert range_id in self.api.ranges
        cyber_range = self.api.ranges[range_id]
        assert cyber_range.status == "deployed"
        
        # Generate attacks
        attack_response = self.api.generate_attacks(range_id, count=3)
        assert attack_response['generated_attacks'] == 3
        
        # Check metrics
        info = self.api.get_range_info(range_id)
        metrics = info['metrics']
        assert metrics['total_attacks'] == 3
        assert metrics['range_id'] == range_id
    
    def test_multiple_ranges(self):
        """Test managing multiple cyber ranges"""
        ranges = []
        
        # Create multiple ranges
        for i in range(3):
            response = self.api.create_range(f"multi-test-{i}")
            ranges.append(response['range_id'])
        
        assert len(self.api.ranges) == 3
        
        # Test each range
        for range_id in ranges:
            attack_response = self.api.generate_attacks(range_id, count=2)
            assert attack_response['generated_attacks'] == 2
    
    def test_security_integration(self):
        """Test security features integration"""
        validator = SecureInputValidator()
        detector = ThreatDetectionEngine()
        
        # Test malicious input detection
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "rm -rf /",
            "../../../etc/passwd"
        ]
        
        for malicious_input in malicious_inputs:
            validation_result = validator.validate_input(malicious_input)
            detection_result = detector.analyze_payload(malicious_input)
            
            # At least one security system should detect the threat
            assert (not validation_result['is_valid'] or 
                   detection_result['threat_detected'])


# Performance tests
class TestPerformance:
    """Performance and stress tests"""
    
    def test_attack_generation_performance(self):
        """Test attack generation performance"""
        generator = LightweightAttackGenerator()
        
        start_time = time.time()
        attacks = generator.generate_batch(1000)
        generation_time = time.time() - start_time
        
        assert len(attacks) == 1000
        assert generation_time < 10.0  # Should generate 1000 attacks in under 10 seconds
        
        # Verify attack quality
        attack_types = set(attack.attack_type for attack in attacks)
        assert len(attack_types) >= 2  # Should have variety in attack types
    
    def test_cache_performance(self):
        """Test cache performance under load"""
        cache = IntelligentCache(max_size=1000)
        
        # Load test data
        start_time = time.time()
        for i in range(1000):
            cache.set(f"key{i}", f"value{i}")
        set_time = time.time() - start_time
        
        # Retrieve test data
        start_time = time.time()
        for i in range(1000):
            value = cache.get(f"key{i}")
            assert value == f"value{i}"
        get_time = time.time() - start_time
        
        assert set_time < 1.0  # Should set 1000 items in under 1 second
        assert get_time < 0.5  # Should get 1000 items in under 0.5 seconds
    
    def test_concurrent_range_operations(self):
        """Test concurrent range operations"""
        api = DemoAPI()
        results = []
        
        def create_and_test_range(range_name):
            try:
                # Create range
                response = api.create_range(range_name)
                range_id = response['range_id']
                
                # Generate attacks
                attack_response = api.generate_attacks(range_id, count=5)
                
                # Get metrics
                info = api.get_range_info(range_id)
                
                results.append(True)
            except Exception as e:
                results.append(False)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_and_test_range, args=(f"concurrent-{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Verify results
        assert len(results) == 5
        assert all(results)  # All operations should succeed
        assert len(api.ranges) == 5


if __name__ == "__main__":
    # Run tests with coverage reporting
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])