"""
Comprehensive tests for CyberRange module.

Tests cover range deployment, management, monitoring, and orchestration.
"""

import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime, timedelta

from gan_cyber_range.core.cyber_range import (
    CyberRange, RangeStatus, ResourceLimits, RangeConfig, 
    RangeMetrics, TrainingProgram
)
from gan_cyber_range.core.network_sim import NetworkTopology, Host, HostType, OSType


class TestResourceLimits:
    """Test ResourceLimits dataclass"""
    
    def test_default_values(self):
        """Test default resource limits"""
        limits = ResourceLimits()
        
        assert limits.cpu_cores == 8
        assert limits.memory_gb == 16
        assert limits.storage_gb == 100
        assert limits.network_bandwidth_mbps == 1000
    
    def test_custom_values(self):
        """Test custom resource limits"""
        limits = ResourceLimits(
            cpu_cores=16,
            memory_gb=32,
            storage_gb=500,
            network_bandwidth_mbps=10000
        )
        
        assert limits.cpu_cores == 16
        assert limits.memory_gb == 32
        assert limits.storage_gb == 500
        assert limits.network_bandwidth_mbps == 10000


class TestRangeConfig:
    """Test RangeConfig dataclass"""
    
    @pytest.fixture
    def sample_topology(self):
        """Create sample network topology"""
        topology = NetworkTopology("test-topology")
        topology.add_subnet("internal", "192.168.1.0/24")
        topology.add_host("host-1", "internal", HostType.WORKSTATION, OSType.WINDOWS)
        return topology
    
    def test_default_config(self, sample_topology):
        """Test default range configuration"""
        limits = ResourceLimits()
        config = RangeConfig(
            name="test-range",
            topology=sample_topology,
            resource_limits=limits
        )
        
        assert config.name == "test-range"
        assert config.topology == sample_topology
        assert config.hypervisor == "docker"
        assert config.isolation_level == "container"
        assert config.monitoring_enabled is True
    
    def test_custom_config(self, sample_topology):
        """Test custom range configuration"""
        limits = ResourceLimits(cpu_cores=4, memory_gb=8)
        config = RangeConfig(
            name="custom-range",
            topology=sample_topology,
            resource_limits=limits,
            hypervisor="kvm",
            isolation_level="vm",
            monitoring_enabled=False
        )
        
        assert config.name == "custom-range"
        assert config.hypervisor == "kvm"
        assert config.isolation_level == "vm"
        assert config.monitoring_enabled is False


class TestRangeMetrics:
    """Test RangeMetrics dataclass"""
    
    def test_default_metrics(self):
        """Test default metrics values"""
        metrics = RangeMetrics()
        
        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.network_traffic == 0.0
        assert metrics.active_attacks == 0
        assert metrics.active_defenses == 0
        assert metrics.incidents_detected == 0
        assert metrics.response_time_avg == 0.0
        assert isinstance(metrics.uptime, timedelta)
    
    def test_metrics_with_values(self):
        """Test metrics with specific values"""
        uptime = timedelta(hours=2, minutes=30)
        metrics = RangeMetrics(
            cpu_usage=45.5,
            memory_usage=60.2,
            network_traffic=1024.0,
            active_attacks=3,
            active_defenses=5,
            incidents_detected=7,
            response_time_avg=2.5,
            uptime=uptime
        )
        
        assert metrics.cpu_usage == 45.5
        assert metrics.memory_usage == 60.2
        assert metrics.network_traffic == 1024.0
        assert metrics.active_attacks == 3
        assert metrics.active_defenses == 5
        assert metrics.incidents_detected == 7
        assert metrics.response_time_avg == 2.5
        assert metrics.uptime == uptime


class TestCyberRange:
    """Test main CyberRange class"""
    
    @pytest.fixture
    def sample_topology(self):
        """Create sample network topology for testing"""
        topology = NetworkTopology("test-topology")
        topology.add_subnet("dmz", "192.168.1.0/24", "dmz")
        topology.add_subnet("internal", "192.168.2.0/24", "internal")
        
        topology.add_host("web-server", "dmz", HostType.SERVER, OSType.LINUX, ["web", "ssh"])
        topology.add_host("workstation-1", "internal", HostType.WORKSTATION, OSType.WINDOWS, ["rdp"])
        topology.add_host("db-server", "internal", HostType.SERVER, OSType.LINUX, ["database", "ssh"])
        
        return topology
    
    @pytest.fixture
    def cyber_range(self, sample_topology):
        """Create CyberRange instance for testing"""
        return CyberRange(
            topology=sample_topology,
            hypervisor="docker",
            container_runtime="docker",
            network_emulation="bridge"
        )
    
    def test_init(self, cyber_range, sample_topology):
        """Test cyber range initialization"""
        assert cyber_range.topology == sample_topology
        assert cyber_range.hypervisor == "docker"
        assert cyber_range.container_runtime == "docker"
        assert cyber_range.network_emulation == "bridge"
        assert cyber_range.status == RangeStatus.INITIALIZING
        assert cyber_range.start_time is None
        assert cyber_range.config is None
        assert isinstance(cyber_range.metrics, RangeMetrics)
        assert len(cyber_range.range_id) > 0
    
    @patch('docker.from_env')
    def test_deploy_success(self, mock_docker, cyber_range):
        """Test successful range deployment"""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock network creation
        mock_network = Mock()
        mock_client.networks.create.return_value = mock_network
        
        # Mock container creation
        mock_container = Mock()
        mock_client.containers.create.return_value = mock_container
        mock_client.containers.run.return_value = mock_container
        
        # Deploy range
        range_id = cyber_range.deploy()
        
        assert range_id == cyber_range.range_id
        assert cyber_range.status == RangeStatus.DEPLOYED
        assert cyber_range.config is not None
        assert cyber_range.start_time is not None
        assert cyber_range.docker_client == mock_client
        
        # Verify networks were created
        assert mock_client.networks.create.called
        
        # Verify containers were created
        assert mock_client.containers.create.called or mock_client.containers.run.called
    
    @patch('docker.from_env')
    def test_deploy_with_custom_limits(self, mock_docker, cyber_range):
        """Test deployment with custom resource limits"""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_client.networks.create.return_value = Mock()
        mock_client.containers.create.return_value = Mock()
        mock_client.containers.run.return_value = Mock()
        
        custom_limits = {
            'cpu_cores': 4,
            'memory_gb': 8,
            'storage_gb': 200
        }
        
        range_id = cyber_range.deploy(
            resource_limits=custom_limits,
            isolation_level="vm",
            monitoring=False
        )
        
        assert range_id == cyber_range.range_id
        assert cyber_range.config.resource_limits.cpu_cores == 4
        assert cyber_range.config.resource_limits.memory_gb == 8
        assert cyber_range.config.isolation_level == "vm"
        assert cyber_range.config.monitoring_enabled is False
    
    @patch('docker.from_env')
    def test_deploy_failure(self, mock_docker, cyber_range):
        """Test deployment failure handling"""
        # Mock Docker client to raise exception
        mock_docker.side_effect = Exception("Docker not available")
        
        with pytest.raises(Exception):
            cyber_range.deploy()
        
        assert cyber_range.status == RangeStatus.ERROR
    
    def test_start_without_deployment(self, cyber_range):
        """Test starting range without deployment"""
        with pytest.raises(RuntimeError):
            cyber_range.start()
    
    @patch('docker.from_env')
    def test_start_after_deployment(self, mock_docker, cyber_range):
        """Test starting range after deployment"""
        # Setup mocks
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_client.networks.create.return_value = Mock()
        mock_client.containers.create.return_value = Mock()
        mock_client.containers.run.return_value = Mock()
        
        # Deploy first
        cyber_range.deploy()
        
        # Mock containers for starting
        mock_container = Mock()
        cyber_range.containers = {"test-container": mock_container}
        
        # Mock network simulator
        cyber_range.network_sim = Mock()
        
        # Start range
        cyber_range.start()
        
        assert cyber_range.status == RangeStatus.RUNNING
        assert mock_container.start.called
        assert cyber_range.network_sim.start.called
    
    @patch('docker.from_env')
    def test_stop_range(self, mock_docker, cyber_range):
        """Test stopping the range"""
        # Setup and deploy
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_client.networks.create.return_value = Mock()
        mock_client.containers.create.return_value = Mock()
        mock_client.containers.run.return_value = Mock()
        
        cyber_range.deploy()
        cyber_range.start()
        
        # Mock components
        mock_container = Mock()
        cyber_range.containers = {"test-container": mock_container}
        cyber_range.attack_engine = Mock()
        cyber_range.network_sim = Mock()
        
        # Stop range
        cyber_range.stop()
        
        assert cyber_range.status == RangeStatus.STOPPED
        assert cyber_range.attack_engine.stop_all_attacks.called
        assert mock_container.stop.called
        assert cyber_range.network_sim.stop.called
    
    def test_execute_attack_not_running(self, cyber_range):
        """Test executing attack when range is not running"""
        attack_config = {"name": "test_attack"}
        
        with pytest.raises(RuntimeError):
            cyber_range.execute_attack(attack_config)
    
    @patch('docker.from_env')
    def test_execute_attack_running(self, mock_docker, cyber_range):
        """Test executing attack when range is running"""
        # Setup and start range
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_client.networks.create.return_value = Mock()
        mock_client.containers.create.return_value = Mock()
        mock_client.containers.run.return_value = Mock()
        
        cyber_range.deploy()
        cyber_range.start()
        
        # Mock attack engine
        cyber_range.attack_engine = Mock()
        cyber_range.attack_engine.execute_attack.return_value = "attack_123"
        
        attack_config = {"name": "test_attack", "technique_id": "T1059"}
        attack_id = cyber_range.execute_attack(attack_config)
        
        assert attack_id == "attack_123"
        assert cyber_range.attack_engine.execute_attack.called_with(attack_config)
    
    def test_event_handling(self, cyber_range):
        """Test event handler registration and triggering"""
        
        # Register event handler
        events_received = []
        
        @cyber_range.on_event('detection')
        def handle_detection(event_data):
            events_received.append(event_data)
        
        # Trigger event
        test_event = {'attack_id': 'test', 'confidence': 0.8}
        cyber_range.trigger_event('detection', test_event)
        
        assert len(events_received) == 1
        assert events_received[0] == test_event
    
    def test_get_metrics(self, cyber_range):
        """Test getting range metrics"""
        metrics = cyber_range.get_metrics()
        
        assert isinstance(metrics, RangeMetrics)
        assert metrics.uptime.total_seconds() >= 0
    
    @patch('docker.from_env')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_take_snapshot(self, mock_mkdir, mock_file, mock_docker, cyber_range):
        """Test taking range snapshot"""
        # Setup range
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_client.networks.create.return_value = Mock()
        mock_client.containers.create.return_value = Mock()
        mock_client.containers.run.return_value = Mock()
        
        cyber_range.deploy()
        
        # Mock container for snapshot
        mock_container = Mock()
        mock_image = Mock()
        mock_image.id = "image_123"
        mock_container.commit.return_value = mock_image
        cyber_range.containers = {"test-container": mock_container}
        
        # Take snapshot
        snapshot_path = cyber_range.take_snapshot("test-snapshot")
        
        assert "test-snapshot" in snapshot_path
        assert mock_container.commit.called
        assert mock_file.called
    
    def test_create_training_program(self, cyber_range):
        """Test creating training program"""
        program = cyber_range.create_training_program(
            duration="2_weeks",
            difficulty="progressive",
            focus_areas=["detection", "response", "forensics"]
        )
        
        assert isinstance(program, TrainingProgram)
        assert program.cyber_range == cyber_range
        assert program.duration == "2_weeks"
        assert program.difficulty == "progressive"
        assert program.focus_areas == ["detection", "response", "forensics"]
    
    def test_get_host_image(self, cyber_range):
        """Test getting appropriate host image"""
        
        test_cases = [
            ("linux", [], "ubuntu:20.04"),
            ("windows", [], "mcr.microsoft.com/windows/servercore:ltsc2019"),
            ("router", [], "frrouting/frr:latest"),
            ("firewall", [], "pfsense/pfsense:latest"),
            ("unknown", [], "ubuntu:20.04")  # Default fallback
        ]
        
        for os_type, services, expected in test_cases:
            result = cyber_range._get_host_image(os_type, services)
            assert result == expected
    
    @patch('docker.from_env')
    def test_deploy_networks(self, mock_docker, cyber_range):
        """Test network deployment"""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        mock_network = Mock()
        mock_client.networks.create.return_value = mock_network
        
        cyber_range.docker_client = mock_client
        cyber_range.config = RangeConfig(
            name="test-range",
            topology=cyber_range.topology,
            resource_limits=ResourceLimits()
        )
        
        cyber_range._deploy_networks()
        
        # Should create main network plus subnet networks
        expected_calls = 1 + len(cyber_range.topology.subnets)
        assert mock_client.networks.create.call_count >= expected_calls
    
    @patch('docker.from_env') 
    def test_deploy_hosts(self, mock_docker, cyber_range):
        """Test host deployment"""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        mock_container = Mock()
        mock_client.containers.create.return_value = mock_container
        
        cyber_range.docker_client = mock_client
        cyber_range.config = RangeConfig(
            name="test-range",
            topology=cyber_range.topology,
            resource_limits=ResourceLimits()
        )
        
        cyber_range._deploy_hosts()
        
        # Should create containers for each host
        assert mock_client.containers.create.call_count == len(cyber_range.topology.hosts)
        assert len(cyber_range.containers) == len(cyber_range.topology.hosts)
    
    def test_generate_vpn_config(self, cyber_range):
        """Test VPN configuration generation"""
        cyber_range.config = RangeConfig(
            name="test-range",
            topology=cyber_range.topology,
            resource_limits=ResourceLimits()
        )
        
        vpn_config = cyber_range._generate_vpn_config()
        
        assert "client" in vpn_config
        assert "test-range" in vpn_config
        assert "remote localhost 1194" in vpn_config


class TestTrainingProgram:
    """Test TrainingProgram class"""
    
    @pytest.fixture
    def sample_topology(self):
        """Create sample topology"""
        topology = NetworkTopology("training-topology")
        topology.add_subnet("internal", "192.168.1.0/24")
        topology.add_host("target", "internal", HostType.SERVER, OSType.LINUX)
        return topology
    
    @pytest.fixture
    def training_program(self, sample_topology):
        """Create training program for testing"""
        cyber_range = CyberRange(sample_topology)
        return TrainingProgram(
            cyber_range=cyber_range,
            duration="1_week",
            difficulty="progressive",
            focus_areas=["detection", "response"]
        )
    
    def test_init(self, training_program):
        """Test training program initialization"""
        assert training_program.duration == "1_week"
        assert training_program.difficulty == "progressive"
        assert training_program.focus_areas == ["detection", "response"]
        assert training_program.scenarios == []
    
    def test_run(self, training_program):
        """Test running training program"""
        team_id = "blue_team_alpha"
        
        results = training_program.run(team_id)
        
        assert isinstance(results, dict)
        assert results['team_id'] == team_id
        assert 'start_time' in results
        assert 'end_time' in results
        assert 'scenarios_completed' in results
        assert 'total_score' in results
        assert 'average_score' in results
        assert isinstance(results['individual_scores'], list)
    
    def test_generate_scenarios(self, training_program):
        """Test scenario generation"""
        training_program._generate_scenarios()
        
        assert len(training_program.scenarios) > 0
        # Should have scenarios for each focus area
        scenario_names = [s['name'] for s in training_program.scenarios]
        assert any('Detection' in name for name in scenario_names)
        assert any('Response' in name for name in scenario_names)
    
    def test_execute_scenario(self, training_program):
        """Test scenario execution"""
        scenario = {
            'name': 'Test Scenario',
            'objectives': ['detect_malware', 'contain_infection']
        }
        team_id = "test_team"
        
        result = training_program._execute_scenario(scenario, team_id)
        
        assert isinstance(result, dict)
        assert result['scenario_name'] == 'Test Scenario'
        assert result['team_id'] == team_id
        assert 'score' in result
        assert 'objectives_met' in result
        assert 'completion_time' in result


class TestRangeStatus:
    """Test RangeStatus enum"""
    
    def test_status_values(self):
        """Test range status enum values"""
        assert RangeStatus.INITIALIZING.value == "initializing"
        assert RangeStatus.DEPLOYED.value == "deployed"
        assert RangeStatus.RUNNING.value == "running"
        assert RangeStatus.PAUSED.value == "paused"
        assert RangeStatus.STOPPED.value == "stopped"
        assert RangeStatus.ERROR.value == "error"


class TestIntegration:
    """Integration tests for cyber range components"""
    
    @pytest.fixture
    def full_topology(self):
        """Create comprehensive topology for integration testing"""
        topology = NetworkTopology("integration-test")
        
        # Add multiple subnets
        topology.add_subnet("dmz", "10.0.1.0/24", "dmz")
        topology.add_subnet("internal", "10.0.2.0/24", "internal")
        topology.add_subnet("management", "10.0.3.0/24", "management")
        
        # Add various hosts
        topology.add_host("web-server", "dmz", HostType.SERVER, OSType.LINUX, ["web", "ssh"])
        topology.add_host("mail-server", "dmz", HostType.SERVER, OSType.LINUX, ["email", "ssh"])
        topology.add_host("workstation-1", "internal", HostType.WORKSTATION, OSType.WINDOWS, ["rdp"])
        topology.add_host("workstation-2", "internal", HostType.WORKSTATION, OSType.WINDOWS, ["rdp"])
        topology.add_host("db-server", "internal", HostType.SERVER, OSType.LINUX, ["database", "ssh"])
        topology.add_host("dc", "internal", HostType.SERVER, OSType.WINDOWS, ["directory", "rdp"])
        topology.add_host("monitor", "management", HostType.SERVER, OSType.LINUX, ["monitoring"])
        
        return topology
    
    @patch('docker.from_env')
    def test_full_range_lifecycle(self, mock_docker, full_topology):
        """Test complete range lifecycle"""
        # Setup mocks
        mock_client = Mock()
        mock_docker.return_value = mock_client
        mock_client.networks.create.return_value = Mock()
        mock_client.containers.create.return_value = Mock()
        mock_client.containers.run.return_value = Mock()
        
        # Create range
        cyber_range = CyberRange(full_topology)
        
        # Test deployment
        range_id = cyber_range.deploy()
        assert cyber_range.status == RangeStatus.DEPLOYED
        assert range_id == cyber_range.range_id
        
        # Test starting
        cyber_range.start()
        assert cyber_range.status == RangeStatus.RUNNING
        
        # Test metrics
        metrics = cyber_range.get_metrics()
        assert isinstance(metrics, RangeMetrics)
        
        # Test event handling
        events = []
        
        @cyber_range.on_event('test')
        def handler(data):
            events.append(data)
        
        cyber_range.trigger_event('test', {'message': 'test'})
        assert len(events) == 1
        
        # Test stopping
        cyber_range.stop()
        assert cyber_range.status == RangeStatus.STOPPED
    
    def test_concurrent_operations(self, full_topology):
        """Test concurrent operations on cyber range"""
        cyber_range = CyberRange(full_topology)
        
        # Test concurrent event handling
        events = []
        lock = threading.Lock()
        
        @cyber_range.on_event('concurrent')
        def handler(data):
            with lock:
                events.append(data)
        
        # Trigger multiple events concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(
                target=cyber_range.trigger_event,
                args=('concurrent', {'id': i})
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        assert len(events) == 10
        assert all('id' in event for event in events)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])