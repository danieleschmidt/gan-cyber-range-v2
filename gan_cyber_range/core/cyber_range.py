"""
Cyber Range orchestration and management system.

This module provides the main CyberRange class that orchestrates virtual environments,
network simulation, attack execution, and blue team training scenarios.
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import subprocess
import docker
from enum import Enum

from .network_sim import NetworkTopology, NetworkSimulator
from .attack_engine import AttackEngine, AttackSimulator

logger = logging.getLogger(__name__)


class RangeStatus(Enum):
    """Cyber range operational status"""
    INITIALIZING = "initializing"
    DEPLOYED = "deployed"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ResourceLimits:
    """Resource allocation limits for cyber range"""
    cpu_cores: int = 8
    memory_gb: int = 16
    storage_gb: int = 100
    network_bandwidth_mbps: int = 1000


@dataclass
class RangeConfig:
    """Configuration for cyber range deployment"""
    name: str
    topology: NetworkTopology
    resource_limits: ResourceLimits
    hypervisor: str = "docker"
    container_runtime: str = "docker"
    network_emulation: str = "bridge"
    isolation_level: str = "container"
    monitoring_enabled: bool = True
    logging_enabled: bool = True
    snapshot_enabled: bool = True
    auto_scale: bool = False


@dataclass
class RangeMetrics:
    """Real-time metrics for cyber range"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_traffic: float = 0.0
    active_attacks: int = 0
    active_defenses: int = 0
    incidents_detected: int = 0
    response_time_avg: float = 0.0
    uptime: timedelta = field(default_factory=lambda: timedelta(0))


class CyberRange:
    """Main cyber range orchestration class"""
    
    def __init__(
        self,
        topology: NetworkTopology,
        hypervisor: str = "docker",
        container_runtime: str = "docker", 
        network_emulation: str = "bridge"
    ):
        self.range_id = str(uuid.uuid4())
        self.topology = topology
        self.hypervisor = hypervisor
        self.container_runtime = container_runtime
        self.network_emulation = network_emulation
        
        # Initialize components
        self.network_sim = NetworkSimulator(topology)
        self.attack_engine = AttackEngine(self)
        
        # Runtime state
        self.status = RangeStatus.INITIALIZING
        self.start_time = None
        self.config = None
        self.metrics = RangeMetrics()
        
        # Container management
        self.docker_client = None
        self.containers = {}
        self.networks = {}
        
        # Event handlers
        self.event_handlers = {
            'detection': [],
            'incident': [],
            'attack_complete': [],
            'defense_triggered': []
        }
        
        # URLs and paths
        self.dashboard_url = None
        self.vpn_config_path = None
        self.log_directory = None
        
        logger.info(f"Initialized CyberRange {self.range_id}")
    
    def deploy(
        self,
        resource_limits: Optional[Dict[str, Any]] = None,
        isolation_level: str = "container",
        monitoring: bool = True
    ) -> str:
        """Deploy the cyber range infrastructure"""
        
        logger.info(f"Deploying cyber range {self.range_id}")
        
        try:
            # Set up configuration
            limits = ResourceLimits(**resource_limits) if resource_limits else ResourceLimits()
            self.config = RangeConfig(
                name=f"cyber-range-{self.range_id[:8]}",
                topology=self.topology,
                resource_limits=limits,
                hypervisor=self.hypervisor,
                isolation_level=isolation_level,
                monitoring_enabled=monitoring
            )
            
            # Initialize Docker client
            self.docker_client = docker.from_env()
            
            # Deploy network infrastructure
            self._deploy_networks()
            
            # Deploy host containers
            self._deploy_hosts()
            
            # Set up monitoring
            if monitoring:
                self._setup_monitoring()
            
            # Set up logging
            self._setup_logging()
            
            # Configure dashboard access
            self._setup_dashboard()
            
            # Update status
            self.status = RangeStatus.DEPLOYED
            self.start_time = datetime.now()
            
            logger.info(f"Cyber range {self.range_id} deployed successfully")
            return self.range_id
            
        except Exception as e:
            self.status = RangeStatus.ERROR
            logger.error(f"Failed to deploy cyber range: {e}")
            raise
    
    def start(self) -> None:
        """Start the cyber range"""
        if self.status != RangeStatus.DEPLOYED:
            raise RuntimeError("Range must be deployed before starting")
            
        logger.info("Starting cyber range")
        
        # Start all containers
        for container in self.containers.values():
            if hasattr(container, 'start'):
                container.start()
        
        # Start network simulation
        self.network_sim.start()
        
        # Update status
        self.status = RangeStatus.RUNNING
        
        logger.info("Cyber range started successfully")
    
    def stop(self) -> None:
        """Stop the cyber range"""
        logger.info("Stopping cyber range")
        
        # Stop attack engine
        self.attack_engine.stop_all_attacks()
        
        # Stop containers
        for container in self.containers.values():
            if hasattr(container, 'stop'):
                container.stop()
        
        # Stop network simulation
        self.network_sim.stop()
        
        # Update status
        self.status = RangeStatus.STOPPED
        
        logger.info("Cyber range stopped")
    
    def destroy(self) -> None:
        """Completely destroy the cyber range"""
        logger.info("Destroying cyber range")
        
        # Stop everything first
        if self.status == RangeStatus.RUNNING:
            self.stop()
        
        # Remove containers
        for container in self.containers.values():
            if hasattr(container, 'remove'):
                container.remove(force=True)
        
        # Remove networks
        for network in self.networks.values():
            if hasattr(network, 'remove'):
                network.remove()
        
        # Clean up files
        if self.log_directory and Path(self.log_directory).exists():
            import shutil
            shutil.rmtree(self.log_directory)
        
        self.containers.clear()
        self.networks.clear()
        
        logger.info("Cyber range destroyed")
    
    def execute_attack(self, attack_config: Dict[str, Any]) -> str:
        """Execute an attack in the cyber range"""
        if self.status != RangeStatus.RUNNING:
            raise RuntimeError("Range must be running to execute attacks")
        
        return self.attack_engine.execute_attack(attack_config)
    
    def create_training_program(
        self,
        duration: str = "1_week",
        difficulty: str = "progressive",
        focus_areas: List[str] = None
    ) -> 'TrainingProgram':
        """Create a training program for blue team"""
        return TrainingProgram(
            cyber_range=self,
            duration=duration,
            difficulty=difficulty,
            focus_areas=focus_areas or ["detection", "response"]
        )
    
    def on_event(self, event_type: str):
        """Decorator for registering event handlers"""
        def decorator(func):
            if event_type in self.event_handlers:
                self.event_handlers[event_type].append(func)
            return func
        return decorator
    
    def trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trigger an event and call all registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
    
    def get_metrics(self) -> RangeMetrics:
        """Get current range metrics"""
        if self.status == RangeStatus.RUNNING and self.start_time:
            self.metrics.uptime = datetime.now() - self.start_time
            
        # Update CPU and memory usage
        self._update_resource_metrics()
        
        return self.metrics
    
    def take_snapshot(self, name: Optional[str] = None) -> str:
        """Take a snapshot of the current range state"""
        if not name:
            name = f"snapshot-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"Taking snapshot: {name}")
        
        # Save container states
        snapshot_data = {
            'timestamp': datetime.now().isoformat(),
            'range_id': self.range_id,
            'topology': self.topology.to_dict(),
            'containers': {},
            'networks': {}
        }
        
        # Commit containers to images
        for name, container in self.containers.items():
            if hasattr(container, 'commit'):
                image = container.commit(repository=f"snapshot-{name}")
                snapshot_data['containers'][name] = image.id
        
        # Save snapshot metadata
        snapshot_path = Path(f"snapshots/{name}.json")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        logger.info(f"Snapshot saved: {snapshot_path}")
        return str(snapshot_path)
    
    def restore_snapshot(self, snapshot_path: str) -> None:
        """Restore range from snapshot"""
        logger.info(f"Restoring from snapshot: {snapshot_path}")
        
        with open(snapshot_path, 'r') as f:
            snapshot_data = json.load(f)
        
        # Stop current range
        if self.status == RangeStatus.RUNNING:
            self.stop()
        
        # Restore containers from images
        for name, image_id in snapshot_data['containers'].items():
            container = self.docker_client.containers.run(
                image_id,
                name=f"{self.config.name}-{name}",
                detach=True,
                network=f"{self.config.name}-network"
            )
            self.containers[name] = container
        
        logger.info("Snapshot restored successfully")
    
    def _deploy_networks(self) -> None:
        """Deploy network infrastructure"""
        logger.info("Deploying network infrastructure")
        
        # Create main network
        main_network = self.docker_client.networks.create(
            name=f"{self.config.name}-network",
            driver="bridge",
            options={
                "com.docker.network.enable_ipv6": "false"
            }
        )
        self.networks['main'] = main_network
        
        # Create subnet networks
        for subnet in self.topology.subnets:
            subnet_network = self.docker_client.networks.create(
                name=f"{self.config.name}-{subnet.name}",
                driver="bridge",
                ipam=docker.types.IPAMConfig(
                    pool_configs=[
                        docker.types.IPAMPool(subnet=subnet.cidr)
                    ]
                )
            )
            self.networks[subnet.name] = subnet_network
    
    def _deploy_hosts(self) -> None:
        """Deploy host containers"""
        logger.info("Deploying host containers")
        
        for host in self.topology.hosts:
            # Select appropriate base image
            image = self._get_host_image(host.os_type, host.services)
            
            # Create container
            container = self.docker_client.containers.create(
                image=image,
                name=f"{self.config.name}-{host.name}",
                hostname=host.name,
                network=f"{self.config.name}-{host.subnet}",
                environment={
                    'HOST_TYPE': host.host_type,
                    'SERVICES': ','.join(host.services)
                },
                mem_limit=f"{self.config.resource_limits.memory_gb // len(self.topology.hosts)}g",
                cpu_count=max(1, self.config.resource_limits.cpu_cores // len(self.topology.hosts))
            )
            
            self.containers[host.name] = container
    
    def _get_host_image(self, os_type: str, services: List[str]) -> str:
        """Get appropriate Docker image for host"""
        base_images = {
            'linux': 'ubuntu:20.04',
            'windows': 'mcr.microsoft.com/windows/servercore:ltsc2019',
            'router': 'frrouting/frr:latest',
            'firewall': 'pfsense/pfsense:latest'
        }
        
        return base_images.get(os_type, 'ubuntu:20.04')
    
    def _setup_monitoring(self) -> None:
        """Set up monitoring infrastructure"""
        logger.info("Setting up monitoring")
        
        # Deploy Prometheus container
        prometheus_container = self.docker_client.containers.run(
            image="prom/prometheus:latest",
            name=f"{self.config.name}-prometheus",
            ports={'9090/tcp': 9090},
            detach=True,
            network=f"{self.config.name}-network"
        )
        self.containers['prometheus'] = prometheus_container
        
        # Deploy Grafana container
        grafana_container = self.docker_client.containers.run(
            image="grafana/grafana:latest",
            name=f"{self.config.name}-grafana",
            ports={'3000/tcp': 3000},
            detach=True,
            network=f"{self.config.name}-network",
            environment={
                'GF_SECURITY_ADMIN_PASSWORD': 'admin123'
            }
        )
        self.containers['grafana'] = grafana_container
    
    def _setup_logging(self) -> None:
        """Set up centralized logging"""
        self.log_directory = Path(f"logs/{self.config.name}")
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Logging configured at: {self.log_directory}")
    
    def _setup_dashboard(self) -> None:
        """Set up web dashboard access"""
        self.dashboard_url = f"http://localhost:3000"
        self.vpn_config_path = f"configs/{self.config.name}-vpn.ovpn"
        
        # Generate VPN config
        vpn_config = self._generate_vpn_config()
        Path(self.vpn_config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.vpn_config_path, 'w') as f:
            f.write(vpn_config)
    
    def _generate_vpn_config(self) -> str:
        """Generate OpenVPN configuration"""
        return f"""client
dev tun
proto udp
remote localhost 1194
resolv-retry infinite
nobind
persist-key
persist-tun
ca ca.crt
cert client.crt
key client.key
comp-lzo
verb 3
# Cyber Range: {self.config.name}
"""
    
    def _update_resource_metrics(self) -> None:
        """Update CPU and memory metrics"""
        try:
            if self.containers:
                total_cpu = 0
                total_memory = 0
                container_count = 0
                
                for container in self.containers.values():
                    if hasattr(container, 'stats'):
                        stats = container.stats(stream=False)
                        if stats:
                            # CPU usage calculation
                            cpu_percent = 0
                            if 'cpu_stats' in stats and 'precpu_stats' in stats:
                                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                                           stats['precpu_stats']['cpu_usage']['total_usage']
                                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                              stats['precpu_stats']['system_cpu_usage']
                                if system_delta > 0:
                                    cpu_percent = (cpu_delta / system_delta) * \
                                                 len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
                            
                            # Memory usage calculation
                            memory_usage = 0
                            if 'memory_stats' in stats:
                                usage = stats['memory_stats'].get('usage', 0)
                                limit = stats['memory_stats'].get('limit', 1)
                                memory_usage = (usage / limit) * 100.0
                            
                            total_cpu += cpu_percent
                            total_memory += memory_usage
                            container_count += 1
                
                if container_count > 0:
                    self.metrics.cpu_usage = total_cpu / container_count
                    self.metrics.memory_usage = total_memory / container_count
                    
        except Exception as e:
            logger.warning(f"Failed to update metrics: {e}")


class TrainingProgram:
    """Blue team training program"""
    
    def __init__(
        self,
        cyber_range: CyberRange,
        duration: str = "1_week",
        difficulty: str = "progressive",
        focus_areas: List[str] = None
    ):
        self.cyber_range = cyber_range
        self.duration = duration
        self.difficulty = difficulty
        self.focus_areas = focus_areas or ["detection", "response"]
        self.scenarios = []
        
    def run(self, team_id: str) -> Dict[str, Any]:
        """Run the training program"""
        logger.info(f"Starting training program for team {team_id}")
        
        results = {
            'team_id': team_id,
            'start_time': datetime.now().isoformat(),
            'scenarios_completed': 0,
            'total_score': 0,
            'individual_scores': []
        }
        
        # Generate scenarios based on focus areas
        self._generate_scenarios()
        
        # Execute scenarios
        for scenario in self.scenarios:
            scenario_result = self._execute_scenario(scenario, team_id)
            results['individual_scores'].append(scenario_result)
            results['scenarios_completed'] += 1
            results['total_score'] += scenario_result.get('score', 0)
        
        results['end_time'] = datetime.now().isoformat()
        results['average_score'] = results['total_score'] / max(1, results['scenarios_completed'])
        
        logger.info(f"Training program completed for team {team_id}")
        return results
    
    def _generate_scenarios(self) -> None:
        """Generate training scenarios"""
        scenario_templates = {
            'detection': {
                'name': 'Malware Detection Challenge',
                'description': 'Detect and analyze malware infections',
                'attacks': ['malware_deployment', 'lateral_movement'],
                'objectives': ['detect_malware', 'identify_c2', 'contain_infection']
            },
            'response': {
                'name': 'Incident Response Drill',
                'description': 'Respond to security incidents',
                'attacks': ['data_exfiltration', 'privilege_escalation'], 
                'objectives': ['isolate_threat', 'preserve_evidence', 'eradicate_threat']
            }
        }
        
        for focus_area in self.focus_areas:
            if focus_area in scenario_templates:
                self.scenarios.append(scenario_templates[focus_area])
    
    def _execute_scenario(self, scenario: Dict[str, Any], team_id: str) -> Dict[str, Any]:
        """Execute a training scenario"""
        logger.info(f"Executing scenario: {scenario['name']}")
        
        # Simulate scenario execution
        score = 75  # Placeholder scoring
        
        return {
            'scenario_name': scenario['name'],
            'team_id': team_id,
            'score': score,
            'objectives_met': len(scenario['objectives']),
            'completion_time': 1800  # 30 minutes
        }