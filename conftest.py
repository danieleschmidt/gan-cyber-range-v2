"""
Pytest configuration and shared fixtures for GAN-Cyber-Range-v2 tests.

This module provides common fixtures, test utilities, and configuration
for the entire test suite.
"""

import pytest
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch
import torch
import numpy as np

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the test session"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_docker_client():
    """Mock Docker client for testing"""
    with patch('docker.from_env') as mock_docker:
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock common Docker operations
        mock_client.networks.create.return_value = Mock()
        mock_client.containers.create.return_value = Mock()
        mock_client.containers.run.return_value = Mock()
        mock_client.images.pull.return_value = Mock()
        
        yield mock_client


@pytest.fixture
def sample_attack_data():
    """Sample attack data for testing"""
    return [
        "sql injection union select password from users",
        "cross site scripting alert document cookie",
        "malware trojan backdoor persistence registry",
        "phishing email credential harvesting social engineering",
        "network port scanning reconnaissance enumeration",
        "privilege escalation sudo buffer overflow",
        "lateral movement remote desktop protocol",
        "data exfiltration encrypted channel compression",
        "denial of service amplification reflection attack",
        "web application parameter tampering bypass authentication"
    ]


@pytest.fixture
def sample_network_topology():
    """Sample network topology for testing"""
    from gan_cyber_range.core.network_sim import NetworkTopology, HostType, OSType
    
    topology = NetworkTopology("test-topology")
    
    # Add subnets
    topology.add_subnet("dmz", "192.168.1.0/24", "dmz")
    topology.add_subnet("internal", "192.168.2.0/24", "internal")
    topology.add_subnet("management", "192.168.3.0/24", "management")
    
    # Add hosts
    topology.add_host("web-server", "dmz", HostType.SERVER, OSType.LINUX, ["web", "ssh"])
    topology.add_host("mail-server", "dmz", HostType.SERVER, OSType.LINUX, ["email", "ssh"])
    topology.add_host("workstation-1", "internal", HostType.WORKSTATION, OSType.WINDOWS, ["rdp"])
    topology.add_host("workstation-2", "internal", HostType.WORKSTATION, OSType.WINDOWS, ["rdp"])
    topology.add_host("db-server", "internal", HostType.SERVER, OSType.LINUX, ["database", "ssh"])
    topology.add_host("dc", "internal", HostType.SERVER, OSType.WINDOWS, ["directory", "rdp"])
    topology.add_host("monitor", "management", HostType.SERVER, OSType.LINUX, ["monitoring", "ssh"])
    
    return topology


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    with patch('redis.Redis') as mock_redis_class:
        mock_redis = Mock()
        mock_redis_class.return_value = mock_redis
        
        # Mock Redis operations
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = False
        mock_redis.keys.return_value = []
        mock_redis.flushdb.return_value = True
        
        yield mock_redis


@pytest.fixture
def simple_gan_model():
    """Simple GAN model for testing"""
    from gan_cyber_range.core.attack_gan import Generator, Discriminator
    
    generator = Generator(noise_dim=32, output_dim=64, hidden_dims=[128])
    discriminator = Discriminator(input_dim=64, hidden_dims=[128])
    
    return {
        'generator': generator,
        'discriminator': discriminator
    }


@pytest.fixture
def attack_config_sample():
    """Sample attack configuration for testing"""
    return {
        'name': 'SQL Injection Test',
        'technique_id': 'T1190',
        'phase': 'exploitation',
        'target_host': 'web-server',
        'success_probability': 0.8,
        'payload': {
            'query': "' UNION SELECT username, password FROM users --",
            'method': 'POST',
            'parameter': 'id'
        },
        'duration': 30
    }


@pytest.fixture
def network_config_sample():
    """Sample network configuration for testing"""
    return {
        'name': 'test_network',
        'subnets': [
            {
                'name': 'dmz',
                'cidr': '192.168.1.0/24',
                'security_zone': 'dmz',
                'vlan_id': 10
            },
            {
                'name': 'internal',
                'cidr': '192.168.2.0/24',
                'security_zone': 'internal',
                'vlan_id': 20
            }
        ],
        'hosts': [
            {
                'name': 'web-server',
                'ip_address': '192.168.1.10',
                'subnet': 'dmz',
                'host_type': 'server',
                'os_type': 'linux',
                'services': ['web', 'ssh'],
                'security_level': 'high'
            },
            {
                'name': 'workstation',
                'ip_address': '192.168.2.10',
                'subnet': 'internal', 
                'host_type': 'workstation',
                'os_type': 'windows',
                'services': ['rdp'],
                'security_level': 'medium'
            }
        ]
    }


@pytest.fixture
def gan_config_sample():
    """Sample GAN configuration for testing"""
    return {
        'architecture': 'wasserstein',
        'attack_types': ['web', 'malware', 'network'],
        'noise_dim': 100,
        'training_mode': 'standard',
        'epochs': 1000,
        'batch_size': 64
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment for each test"""
    # Force CPU-only for tests to avoid GPU dependencies
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    yield
    
    # Cleanup after each test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    with patch('gan_cyber_range.utils.logging_config.logger') as mock_log:
        yield mock_log


@pytest.fixture
def performance_config():
    """Performance configuration for testing"""
    from gan_cyber_range.utils.optimization import OptimizationConfig
    
    return OptimizationConfig(
        enable_gpu=False,  # Force CPU for tests
        enable_mixed_precision=False,
        enable_data_parallel=False,
        max_workers=2,  # Reduce for tests
        memory_optimization=True
    )


@pytest.fixture
def metrics_data_sample():
    """Sample metrics data for testing"""
    from datetime import datetime
    from gan_cyber_range.utils.monitoring import Metric, MetricType
    
    return [
        Metric("cpu_usage", 45.5, MetricType.GAUGE, datetime.now(), {"host": "server1"}),
        Metric("memory_usage", 67.2, MetricType.GAUGE, datetime.now(), {"host": "server1"}),
        Metric("attacks_total", 15, MetricType.COUNTER, datetime.now(), {"type": "web"}),
        Metric("response_time", 0.125, MetricType.TIMER, datetime.now(), {"endpoint": "/api/v1"})
    ]


@pytest.fixture
def cache_test_data():
    """Test data for caching tests"""
    return {
        "string_data": "Hello, World!",
        "numeric_data": 42,
        "list_data": [1, 2, 3, 4, 5],
        "dict_data": {"key1": "value1", "key2": "value2"},
        "complex_data": {
            "nested": {
                "array": [{"id": 1, "name": "test"}],
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
    }


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.gpu
pytest.mark.network = pytest.mark.network


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "network: mark test as requiring network access")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add unit test marker to all tests by default
        if not any(marker.name in ["integration", "slow", "gpu", "network"] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that might take longer
        if any(keyword in item.name.lower() 
               for keyword in ["train", "deploy", "full", "lifecycle", "concurrent"]):
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to tests that test multiple components
        if any(keyword in item.name.lower() 
               for keyword in ["integration", "end_to_end", "full_range"]):
            item.add_marker(pytest.mark.integration)


class TestUtilities:
    """Utility functions for tests"""
    
    @staticmethod
    def create_temp_file(content: str, suffix: str = ".txt") -> Path:
        """Create a temporary file with content"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)
    
    @staticmethod
    def create_temp_config_file(config_dict: dict, format: str = "json") -> Path:
        """Create a temporary configuration file"""
        import json
        import yaml
        
        if format == "json":
            content = json.dumps(config_dict, indent=2)
            suffix = ".json"
        elif format == "yaml":
            content = yaml.dump(config_dict, default_flow_style=False)
            suffix = ".yaml"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return TestUtilities.create_temp_file(content, suffix)
    
    @staticmethod
    def assert_metrics_equal(metric1, metric2, tolerance=1e-6):
        """Assert that two metrics are approximately equal"""
        assert metric1.name == metric2.name
        assert abs(metric1.value - metric2.value) < tolerance
        assert metric1.metric_type == metric2.metric_type
        assert metric1.labels == metric2.labels
    
    @staticmethod
    def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        
        return False


# Make test utilities available
@pytest.fixture
def test_utils():
    """Provide test utilities to tests"""
    return TestUtilities