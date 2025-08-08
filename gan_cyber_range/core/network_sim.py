"""
Network simulation and topology generation for cyber range environments.

This module provides network topology generation, host simulation, and 
realistic network traffic patterns for cybersecurity training scenarios.
"""

import logging
import ipaddress
import random
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)


class HostType(Enum):
    """Types of hosts in the network"""
    WORKSTATION = "workstation"
    SERVER = "server"
    ROUTER = "router"
    FIREWALL = "firewall"
    SWITCH = "switch"
    IOT_DEVICE = "iot_device"
    MOBILE_DEVICE = "mobile_device"


class OSType(Enum):
    """Operating system types"""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    ROUTER_OS = "router_os"
    FIREWALL_OS = "firewall_os"
    IOT_OS = "iot_os"


@dataclass
class Vulnerability:
    """Represents a vulnerability in a host"""
    cve_id: str
    severity: float  # CVSS score 0-10
    description: str
    affected_services: List[str]
    exploit_difficulty: str  # "low", "medium", "high"
    public_exploit: bool = False


@dataclass
class Host:
    """Represents a host in the network topology"""
    name: str
    ip_address: str
    subnet: str
    host_type: HostType
    os_type: OSType
    services: List[str] = field(default_factory=list)
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    security_level: str = "medium"  # "low", "medium", "high"
    crown_jewel: bool = False
    monitoring_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert host to dictionary representation"""
        return {
            'name': self.name,
            'ip_address': self.ip_address,
            'subnet': self.subnet,
            'host_type': self.host_type.value,
            'os_type': self.os_type.value,
            'services': self.services,
            'vulnerabilities': [
                {
                    'cve_id': v.cve_id,
                    'severity': v.severity,
                    'description': v.description,
                    'affected_services': v.affected_services,
                    'exploit_difficulty': v.exploit_difficulty,
                    'public_exploit': v.public_exploit
                } for v in self.vulnerabilities
            ],
            'security_level': self.security_level,
            'crown_jewel': self.crown_jewel,
            'monitoring_enabled': self.monitoring_enabled
        }


@dataclass
class Subnet:
    """Represents a network subnet"""
    name: str
    cidr: str
    vlan_id: Optional[int] = None
    gateway: Optional[str] = None
    dns_servers: List[str] = field(default_factory=list)
    security_zone: str = "internal"  # "dmz", "internal", "management", "guest"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert subnet to dictionary representation"""
        return {
            'name': self.name,
            'cidr': self.cidr,
            'vlan_id': self.vlan_id,
            'gateway': self.gateway,
            'dns_servers': self.dns_servers,
            'security_zone': self.security_zone
        }


@dataclass 
class NetworkConnection:
    """Represents a connection between network segments"""
    source: str
    destination: str
    connection_type: str  # "direct", "routed", "vpn", "firewall"
    bandwidth_mbps: int = 1000
    latency_ms: int = 1
    allowed_protocols: List[str] = field(default_factory=lambda: ["tcp", "udp", "icmp"])
    firewall_rules: List[Dict[str, Any]] = field(default_factory=list)


class NetworkTopology:
    """Network topology generator and manager"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.subnets: List[Subnet] = []
        self.hosts: List[Host] = []
        self.connections: List[NetworkConnection] = []
        self.total_hosts = 0
        
    @classmethod
    def generate(
        cls,
        template: str = "enterprise",
        subnets: List[str] = None,
        hosts_per_subnet: Dict[str, int] = None,
        services: List[str] = None,
        vulnerabilities: str = "realistic"
    ) -> 'NetworkTopology':
        """Generate a network topology from template"""
        
        topology = cls(name=f"{template}-topology")
        
        # Default configurations
        if subnets is None:
            subnets = ["dmz", "internal", "management"]
        
        if hosts_per_subnet is None:
            hosts_per_subnet = {"dmz": 5, "internal": 20, "management": 3}
            
        if services is None:
            services = ["web", "database", "email", "file_share"]
        
        # Generate subnets
        topology._generate_subnets(subnets)
        
        # Generate hosts
        for subnet_name, host_count in hosts_per_subnet.items():
            if subnet_name in [s.name for s in topology.subnets]:
                topology._generate_hosts_for_subnet(subnet_name, host_count, services)
        
        # Add vulnerabilities
        if vulnerabilities == "realistic":
            topology._add_realistic_vulnerabilities()
        
        # Generate connections
        topology._generate_connections()
        
        logger.info(f"Generated {template} topology with {len(topology.hosts)} hosts across {len(topology.subnets)} subnets")
        return topology
    
    def add_subnet(
        self,
        name: str,
        cidr: str,
        security_zone: str = "internal",
        vlan_id: Optional[int] = None
    ) -> Subnet:
        """Add a subnet to the topology"""
        
        subnet = Subnet(
            name=name,
            cidr=cidr,
            security_zone=security_zone,
            vlan_id=vlan_id,
            gateway=str(ipaddress.ip_network(cidr).network_address + 1),
            dns_servers=["8.8.8.8", "8.8.4.4"]
        )
        
        self.subnets.append(subnet)
        logger.info(f"Added subnet: {name} ({cidr})")
        return subnet
    
    def add_host(
        self,
        name: str,
        subnet_name: str,
        host_type: Union[HostType, str],
        os_type: Union[OSType, str],
        services: List[str] = None,
        crown_jewel: bool = False
    ) -> Host:
        """Add a host to the topology"""
        
        # Find subnet
        subnet = next((s for s in self.subnets if s.name == subnet_name), None)
        if not subnet:
            raise ValueError(f"Subnet {subnet_name} not found")
        
        # Convert enums if needed
        if isinstance(host_type, str):
            host_type = HostType(host_type)
        if isinstance(os_type, str):
            os_type = OSType(os_type)
        
        # Generate IP address
        ip_address = self._generate_ip_for_subnet(subnet.cidr)
        
        host = Host(
            name=name,
            ip_address=ip_address,
            subnet=subnet_name,
            host_type=host_type,
            os_type=os_type,
            services=services or [],
            crown_jewel=crown_jewel
        )
        
        self.hosts.append(host)
        self.total_hosts += 1
        
        logger.info(f"Added host: {name} ({ip_address}) to subnet {subnet_name}")
        return host
    
    def add_connection(
        self,
        source: str,
        destination: str,
        connection_type: str = "direct",
        bandwidth_mbps: int = 1000
    ) -> NetworkConnection:
        """Add a connection between network segments"""
        
        connection = NetworkConnection(
            source=source,
            destination=destination,
            connection_type=connection_type,
            bandwidth_mbps=bandwidth_mbps
        )
        
        self.connections.append(connection)
        logger.info(f"Added connection: {source} -> {destination}")
        return connection
    
    def get_hosts_by_subnet(self, subnet_name: str) -> List[Host]:
        """Get all hosts in a specific subnet"""
        return [host for host in self.hosts if host.subnet == subnet_name]
    
    def get_hosts_by_type(self, host_type: HostType) -> List[Host]:
        """Get all hosts of a specific type"""
        return [host for host in self.hosts if host.host_type == host_type]
    
    def get_crown_jewels(self) -> List[Host]:
        """Get all crown jewel hosts"""
        return [host for host in self.hosts if host.crown_jewel]
    
    def get_vulnerable_hosts(self, min_severity: float = 7.0) -> List[Host]:
        """Get hosts with high-severity vulnerabilities"""
        vulnerable_hosts = []
        for host in self.hosts:
            for vuln in host.vulnerabilities:
                if vuln.severity >= min_severity:
                    vulnerable_hosts.append(host)
                    break
        return vulnerable_hosts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert topology to dictionary representation"""
        return {
            'name': self.name,
            'subnets': [subnet.to_dict() for subnet in self.subnets],
            'hosts': [host.to_dict() for host in self.hosts],
            'connections': [
                {
                    'source': conn.source,
                    'destination': conn.destination,
                    'connection_type': conn.connection_type,
                    'bandwidth_mbps': conn.bandwidth_mbps,
                    'latency_ms': conn.latency_ms,
                    'allowed_protocols': conn.allowed_protocols
                } for conn in self.connections
            ],
            'total_hosts': self.total_hosts
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save topology to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Topology saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'NetworkTopology':
        """Load topology from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        topology = cls(name=data['name'])
        
        # Load subnets
        for subnet_data in data['subnets']:
            subnet = Subnet(**subnet_data)
            topology.subnets.append(subnet)
        
        # Load hosts
        for host_data in data['hosts']:
            # Convert vulnerabilities
            vulnerabilities = []
            for vuln_data in host_data.get('vulnerabilities', []):
                vuln = Vulnerability(**vuln_data)
                vulnerabilities.append(vuln)
            
            host = Host(
                name=host_data['name'],
                ip_address=host_data['ip_address'],
                subnet=host_data['subnet'],
                host_type=HostType(host_data['host_type']),
                os_type=OSType(host_data['os_type']),
                services=host_data.get('services', []),
                vulnerabilities=vulnerabilities,
                security_level=host_data.get('security_level', 'medium'),
                crown_jewel=host_data.get('crown_jewel', False),
                monitoring_enabled=host_data.get('monitoring_enabled', True)
            )
            topology.hosts.append(host)
        
        # Load connections
        for conn_data in data['connections']:
            connection = NetworkConnection(**conn_data)
            topology.connections.append(connection)
        
        topology.total_hosts = data.get('total_hosts', len(topology.hosts))
        
        logger.info(f"Topology loaded from {filepath}")
        return topology
    
    def _generate_subnets(self, subnet_names: List[str]) -> None:
        """Generate subnets based on names"""
        subnet_configs = {
            "dmz": {"cidr": "192.168.1.0/24", "zone": "dmz", "vlan": 10},
            "internal": {"cidr": "192.168.2.0/24", "zone": "internal", "vlan": 20},
            "management": {"cidr": "192.168.3.0/24", "zone": "management", "vlan": 30},
            "development": {"cidr": "192.168.4.0/24", "zone": "internal", "vlan": 40},
            "guest": {"cidr": "192.168.5.0/24", "zone": "guest", "vlan": 50}
        }
        
        for subnet_name in subnet_names:
            config = subnet_configs.get(subnet_name, {
                "cidr": f"192.168.{len(self.subnets) + 10}.0/24",
                "zone": "internal",
                "vlan": (len(self.subnets) + 1) * 10
            })
            
            self.add_subnet(
                name=subnet_name,
                cidr=config["cidr"],
                security_zone=config["zone"],
                vlan_id=config["vlan"]
            )
    
    def _generate_hosts_for_subnet(self, subnet_name: str, host_count: int, services: List[str]) -> None:
        """Generate hosts for a specific subnet"""
        
        # Host generation rules based on subnet type
        host_rules = {
            "dmz": {
                "types": [HostType.SERVER],
                "os": [OSType.LINUX, OSType.WINDOWS],
                "services": ["web", "email", "dns"],
                "security": "high"
            },
            "internal": {
                "types": [HostType.WORKSTATION, HostType.SERVER],
                "os": [OSType.WINDOWS, OSType.LINUX, OSType.MACOS],
                "services": ["file_share", "database", "application"],
                "security": "medium"
            },
            "management": {
                "types": [HostType.SERVER, HostType.ROUTER, HostType.FIREWALL],
                "os": [OSType.LINUX, OSType.ROUTER_OS, OSType.FIREWALL_OS],
                "services": ["monitoring", "backup", "directory"],
                "security": "high"
            }
        }
        
        rules = host_rules.get(subnet_name, host_rules["internal"])
        
        for i in range(host_count):
            host_name = f"{subnet_name}-host-{i+1:02d}"
            host_type = random.choice(rules["types"])
            os_type = random.choice(rules["os"])
            
            # Select appropriate services
            host_services = random.sample(
                services, 
                min(random.randint(1, 3), len(services))
            )
            
            # Determine if crown jewel (5% chance for servers)
            crown_jewel = (host_type == HostType.SERVER and random.random() < 0.05)
            
            host = self.add_host(
                name=host_name,
                subnet_name=subnet_name,
                host_type=host_type,
                os_type=os_type,
                services=host_services,
                crown_jewel=crown_jewel
            )
            
            host.security_level = rules["security"]
    
    def _add_realistic_vulnerabilities(self) -> None:
        """Add realistic vulnerabilities to hosts"""
        
        # Common vulnerabilities database
        common_vulns = [
            Vulnerability("CVE-2021-44228", 10.0, "Log4j Remote Code Execution", ["web", "application"], "low", True),
            Vulnerability("CVE-2021-34527", 8.8, "Windows Print Spooler Privilege Escalation", ["print"], "medium", True),
            Vulnerability("CVE-2020-1472", 10.0, "Netlogon Elevation of Privilege", ["directory"], "low", True),
            Vulnerability("CVE-2019-0708", 9.8, "RDP Remote Code Execution", ["rdp"], "medium", True),
            Vulnerability("CVE-2017-0144", 8.1, "SMB Remote Code Execution", ["file_share"], "low", True),
            Vulnerability("CVE-2021-26855", 9.8, "Exchange Server SSRF", ["email"], "medium", True),
            Vulnerability("CVE-2020-0796", 10.0, "SMBv3 Remote Code Execution", ["file_share"], "low", True),
            Vulnerability("CVE-2019-19781", 9.8, "Citrix ADC Path Traversal", ["web"], "low", True)
        ]
        
        for host in self.hosts:
            # Vulnerability probability based on security level
            vuln_probability = {
                "low": 0.8,
                "medium": 0.4,
                "high": 0.1
            }.get(host.security_level, 0.4)
            
            if random.random() < vuln_probability:
                # Select relevant vulnerabilities for this host
                applicable_vulns = [
                    vuln for vuln in common_vulns
                    if any(service in host.services for service in vuln.affected_services)
                ]
                
                if applicable_vulns:
                    # Add 1-3 vulnerabilities
                    num_vulns = random.randint(1, min(3, len(applicable_vulns)))
                    selected_vulns = random.sample(applicable_vulns, num_vulns)
                    host.vulnerabilities.extend(selected_vulns)
    
    def _generate_connections(self) -> None:
        """Generate network connections between subnets"""
        
        # Create connections based on typical enterprise patterns
        connection_rules = [
            ("dmz", "internal", "firewall", 100),
            ("internal", "management", "routed", 1000),
            ("dmz", "management", "firewall", 100)
        ]
        
        for source, dest, conn_type, bandwidth in connection_rules:
            if (source in [s.name for s in self.subnets] and 
                dest in [s.name for s in self.subnets]):
                self.add_connection(source, dest, conn_type, bandwidth)
    
    def _generate_ip_for_subnet(self, cidr: str) -> str:
        """Generate an available IP address within a subnet"""
        network = ipaddress.ip_network(cidr)
        
        # Get existing IPs in this subnet
        existing_ips = {
            host.ip_address for host in self.hosts 
            if ipaddress.ip_address(host.ip_address) in network
        }
        
        # Find available IP (skip network and broadcast addresses)
        for ip in network.hosts():
            if str(ip) not in existing_ips:
                return str(ip)
        
        raise ValueError(f"No available IP addresses in subnet {cidr}")


class NetworkSimulator:
    """Network simulation and traffic generation"""
    
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self.is_running = False
        self.traffic_generators = {}
        self.bandwidth_monitors = {}
        
    def start(self) -> None:
        """Start network simulation"""
        logger.info("Starting network simulation")
        
        # Initialize traffic generators for each subnet
        for subnet in self.topology.subnets:
            generator = TrafficGenerator(subnet, self.topology.get_hosts_by_subnet(subnet.name))
            self.traffic_generators[subnet.name] = generator
            generator.start()
        
        self.is_running = True
        logger.info("Network simulation started")
    
    def stop(self) -> None:
        """Stop network simulation"""
        logger.info("Stopping network simulation")
        
        for generator in self.traffic_generators.values():
            generator.stop()
        
        self.traffic_generators.clear()
        self.is_running = False
        
        logger.info("Network simulation stopped")
    
    def get_traffic_stats(self) -> Dict[str, Any]:
        """Get current traffic statistics"""
        stats = {}
        
        for subnet_name, generator in self.traffic_generators.items():
            stats[subnet_name] = generator.get_stats()
        
        return stats


class TrafficGenerator:
    """Generates realistic network traffic for a subnet"""
    
    def __init__(self, subnet: Subnet, hosts: List[Host]):
        self.subnet = subnet
        self.hosts = hosts
        self.is_running = False
        self.packets_sent = 0
        self.bytes_sent = 0
        
    def start(self) -> None:
        """Start generating traffic"""
        self.is_running = True
        # In a real implementation, this would start background threads
        # For now, we'll just mark it as running
        
    def stop(self) -> None:
        """Stop generating traffic"""
        self.is_running = False
        
    def get_stats(self) -> Dict[str, Any]:
        """Get traffic statistics"""
        return {
            'packets_sent': self.packets_sent,
            'bytes_sent': self.bytes_sent,
            'hosts_active': len([h for h in self.hosts if h.monitoring_enabled]),
            'is_running': self.is_running
        }