"""
Network Factory for creating and configuring network topologies.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from ipaddress import IPv4Network, IPv4Address
import random

from ..core.network_sim import NetworkTopology, NetworkHost, NetworkSubnet
from ..utils.error_handling import NetworkConfigurationError

logger = logging.getLogger(__name__)


@dataclass 
class NetworkTemplate:
    """Template for network creation"""
    name: str
    description: str
    subnets: Dict[str, Dict[str, Any]]
    default_services: List[str]
    vulnerability_density: float  # 0.0 - 1.0
    complexity_score: int  # 1-10


class NetworkFactory:
    """Factory for creating network topologies with realistic configurations"""
    
    # Industry-standard network templates
    NETWORK_TEMPLATES = {
        "startup": NetworkTemplate(
            name="Startup Network",
            description="Simple network for small startups",
            subnets={
                "office": {"hosts": 15, "services": ["web", "database", "file_share"]},
                "guest": {"hosts": 5, "services": ["web"]}
            },
            default_services=["web", "database", "file_share"],
            vulnerability_density=0.3,
            complexity_score=2
        ),
        
        "smb_enterprise": NetworkTemplate(
            name="Small-Medium Business",
            description="Standard SMB network infrastructure", 
            subnets={
                "dmz": {"hosts": 3, "services": ["web", "email"]},
                "internal": {"hosts": 25, "services": ["database", "file_share", "print"]},
                "management": {"hosts": 5, "services": ["monitoring", "backup"]}
            },
            default_services=["web", "email", "database", "file_share", "monitoring"],
            vulnerability_density=0.4,
            complexity_score=4
        ),
        
        "enterprise": NetworkTemplate(
            name="Large Enterprise",
            description="Complex enterprise network with multiple zones",
            subnets={
                "dmz": {"hosts": 8, "services": ["web", "email", "dns"]},
                "internal": {"hosts": 100, "services": ["database", "file_share", "app_servers"]},
                "management": {"hosts": 15, "services": ["monitoring", "backup", "security"]},
                "development": {"hosts": 30, "services": ["ci_cd", "testing", "version_control"]},
                "production": {"hosts": 50, "services": ["app_servers", "database", "load_balancers"]}
            },
            default_services=["web", "email", "database", "monitoring", "ci_cd"],
            vulnerability_density=0.5,
            complexity_score=7
        ),
        
        "financial": NetworkTemplate(
            name="Financial Institution",
            description="High-security financial services network",
            subnets={
                "public_dmz": {"hosts": 5, "services": ["web", "api_gateway"]},
                "secure_dmz": {"hosts": 3, "services": ["security_appliances"]},
                "trading": {"hosts": 20, "services": ["trading_systems", "market_data"]},
                "core_banking": {"hosts": 40, "services": ["core_systems", "database"]},
                "management": {"hosts": 10, "services": ["monitoring", "compliance", "audit"]}
            },
            default_services=["web", "database", "monitoring", "security_appliances"],
            vulnerability_density=0.2,  # Lower vulnerability due to security focus
            complexity_score=9
        ),
        
        "healthcare": NetworkTemplate(
            name="Healthcare Network",
            description="HIPAA-compliant healthcare network",
            subnets={
                "patient_portal": {"hosts": 5, "services": ["web", "patient_systems"]},
                "clinical": {"hosts": 50, "services": ["ehr", "medical_devices", "imaging"]},
                "administrative": {"hosts": 30, "services": ["billing", "scheduling", "hr_systems"]},
                "research": {"hosts": 15, "services": ["research_db", "analytics"]},
                "management": {"hosts": 8, "services": ["monitoring", "backup", "compliance"]}
            },
            default_services=["web", "ehr", "monitoring", "backup"],
            vulnerability_density=0.3,
            complexity_score=6
        ),
        
        "manufacturing": NetworkTemplate(
            name="Manufacturing Network",
            description="Industrial control systems network",
            subnets={
                "corporate": {"hosts": 40, "services": ["web", "email", "erp"]},
                "scada": {"hosts": 25, "services": ["hmi", "plc", "historian"]},
                "manufacturing": {"hosts": 60, "services": ["mes", "quality", "maintenance"]},
                "logistics": {"hosts": 20, "services": ["wms", "tracking", "shipping"]},
                "management": {"hosts": 10, "services": ["monitoring", "backup"]}
            },
            default_services=["web", "erp", "scada", "monitoring"],
            vulnerability_density=0.6,  # Higher due to legacy systems
            complexity_score=8
        ),
        
        "cloud_hybrid": NetworkTemplate(
            name="Cloud-Hybrid Network",
            description="Modern cloud-hybrid infrastructure",
            subnets={
                "edge": {"hosts": 8, "services": ["api_gateway", "cdn", "waf"]},
                "app_tier": {"hosts": 30, "services": ["microservices", "containers", "load_balancers"]},
                "data_tier": {"hosts": 15, "services": ["database", "cache", "message_queue"]},
                "devops": {"hosts": 20, "services": ["ci_cd", "monitoring", "logging"]},
                "management": {"hosts": 5, "services": ["orchestration", "secrets_management"]}
            },
            default_services=["api_gateway", "microservices", "database", "ci_cd", "monitoring"],
            vulnerability_density=0.35,
            complexity_score=6
        }
    }
    
    def __init__(self):
        self._ip_pool = IPv4Network("10.0.0.0/8")
        self._allocated_subnets: List[IPv4Network] = []
        
    def create_from_template(self, 
                           template_name: str,
                           scale_factor: float = 1.0,
                           custom_modifications: Optional[Dict[str, Any]] = None) -> NetworkTopology:
        """Create network topology from template"""
        
        if template_name not in self.NETWORK_TEMPLATES:
            available = list(self.NETWORK_TEMPLATES.keys())
            raise NetworkConfigurationError(f"Unknown template: {template_name}. Available: {available}")
            
        template = self.NETWORK_TEMPLATES[template_name]
        logger.info(f"Creating network from template: {template_name}")
        
        # Scale the network based on scale factor
        scaled_subnets = self._scale_template_subnets(template.subnets, scale_factor)
        
        # Apply custom modifications
        if custom_modifications:
            scaled_subnets = self._apply_modifications(scaled_subnets, custom_modifications)
            
        # Generate network topology
        topology = self._build_topology_from_subnets(scaled_subnets, template)
        
        # Add realistic vulnerabilities
        self._inject_vulnerabilities(topology, template.vulnerability_density)
        
        # Add network services
        self._configure_services(topology, template.default_services)
        
        # Set metadata
        topology.metadata = {
            "template": template_name,
            "description": template.description,
            "complexity_score": template.complexity_score,
            "scale_factor": scale_factor
        }
        
        return topology
        
    def create_custom_topology(self,
                             subnet_specs: Dict[str, Dict[str, Any]],
                             network_name: str = "custom_network") -> NetworkTopology:
        """Create a custom network topology"""
        
        logger.info(f"Creating custom network topology: {network_name}")
        
        # Validate subnet specifications
        self._validate_subnet_specs(subnet_specs)
        
        # Build topology
        topology = self._build_topology_from_subnets(subnet_specs)
        
        # Set custom metadata
        topology.metadata = {
            "template": "custom",
            "name": network_name,
            "complexity_score": self._calculate_complexity_score(subnet_specs)
        }
        
        return topology
        
    def create_realistic_internet_topology(self,
                                         organization_count: int = 5,
                                         connection_density: float = 0.3) -> NetworkTopology:
        """Create a realistic internet-like topology for advanced scenarios"""
        
        logger.info(f"Creating internet topology with {organization_count} organizations")
        
        # Create multiple organization networks
        org_networks = []
        for i in range(organization_count):
            # Vary the organization types
            org_templates = ["startup", "smb_enterprise", "enterprise"]
            template_name = random.choice(org_templates)
            
            org_net = self.create_from_template(
                template_name,
                scale_factor=random.uniform(0.5, 1.5),
                custom_modifications={"org_id": i}
            )
            
            org_networks.append(org_net)
            
        # Interconnect organizations
        internet_topology = self._interconnect_networks(org_networks, connection_density)
        
        # Add internet infrastructure
        self._add_internet_infrastructure(internet_topology)
        
        return internet_topology
        
    def create_attack_scenario_topology(self,
                                      scenario_type: str,
                                      difficulty: str = "medium") -> NetworkTopology:
        """Create topology optimized for specific attack scenarios"""
        
        scenario_configs = {
            "lateral_movement": {
                "base_template": "enterprise",
                "modifications": {
                    "trust_relationships": True,
                    "shared_services": True,
                    "vulnerability_clusters": True
                }
            },
            "privilege_escalation": {
                "base_template": "smb_enterprise", 
                "modifications": {
                    "admin_workstations": True,
                    "service_accounts": True,
                    "legacy_systems": True
                }
            },
            "data_exfiltration": {
                "base_template": "financial",
                "modifications": {
                    "data_repositories": True,
                    "monitoring_gaps": True,
                    "external_connections": True
                }
            },
            "supply_chain": {
                "base_template": "manufacturing",
                "modifications": {
                    "vendor_connections": True,
                    "update_servers": True,
                    "build_systems": True
                }
            }
        }
        
        if scenario_type not in scenario_configs:
            raise NetworkConfigurationError(f"Unknown scenario type: {scenario_type}")
            
        config = scenario_configs[scenario_type]
        base_topology = self.create_from_template(config["base_template"])
        
        # Apply scenario-specific modifications
        enhanced_topology = self._apply_scenario_enhancements(
            base_topology, 
            config["modifications"], 
            difficulty
        )
        
        return enhanced_topology
        
    def get_network_statistics(self, topology: NetworkTopology) -> Dict[str, Any]:
        """Get comprehensive statistics about a network topology"""
        
        stats = {
            "total_hosts": len(topology.hosts),
            "total_subnets": len(topology.subnets),
            "services_count": len(topology.get_all_services()) if hasattr(topology, 'get_all_services') else 0,
            "vulnerability_count": len(topology.get_vulnerabilities()) if hasattr(topology, 'get_vulnerabilities') else 0,
            "complexity_score": topology.metadata.get("complexity_score", 0),
            "subnet_distribution": {},
            "service_distribution": {},
            "os_distribution": {}
        }
        
        # Analyze subnet distribution
        for subnet in topology.subnets:
            subnet_name = subnet.name if hasattr(subnet, 'name') else str(subnet)
            host_count = len([h for h in topology.hosts if h.subnet == subnet])
            stats["subnet_distribution"][subnet_name] = host_count
            
        return stats
        
    def validate_topology(self, topology: NetworkTopology) -> Dict[str, Any]:
        """Validate network topology for common issues"""
        
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check for isolated hosts
        isolated_hosts = self._find_isolated_hosts(topology)
        if isolated_hosts:
            validation_results["warnings"].append(f"Found {len(isolated_hosts)} isolated hosts")
            
        # Check subnet sizing
        oversized_subnets = self._find_oversized_subnets(topology)
        if oversized_subnets:
            validation_results["warnings"].append(f"Found {len(oversized_subnets)} oversized subnets")
            
        # Check for security issues
        security_issues = self._check_security_issues(topology)
        if security_issues:
            validation_results["errors"].extend(security_issues)
            validation_results["valid"] = False
            
        # Generate recommendations
        recommendations = self._generate_topology_recommendations(topology)
        validation_results["recommendations"].extend(recommendations)
        
        return validation_results
        
    def _scale_template_subnets(self, 
                              subnets: Dict[str, Dict[str, Any]], 
                              scale_factor: float) -> Dict[str, Dict[str, Any]]:
        """Scale subnet host counts by factor"""
        
        scaled = {}
        for subnet_name, config in subnets.items():
            scaled_config = config.copy()
            original_hosts = config["hosts"]
            scaled_hosts = max(1, int(original_hosts * scale_factor))
            scaled_config["hosts"] = scaled_hosts
            scaled[subnet_name] = scaled_config
            
        return scaled
        
    def _apply_modifications(self, 
                           subnets: Dict[str, Dict[str, Any]], 
                           modifications: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Apply custom modifications to subnet configuration"""
        
        modified = subnets.copy()
        
        # Add additional subnets
        if "additional_subnets" in modifications:
            for subnet_name, config in modifications["additional_subnets"].items():
                modified[subnet_name] = config
                
        # Modify existing subnets
        if "subnet_modifications" in modifications:
            for subnet_name, changes in modifications["subnet_modifications"].items():
                if subnet_name in modified:
                    modified[subnet_name].update(changes)
                    
        return modified
        
    def _build_topology_from_subnets(self, 
                                   subnet_specs: Dict[str, Dict[str, Any]], 
                                   template: Optional[NetworkTemplate] = None) -> NetworkTopology:
        """Build NetworkTopology object from subnet specifications"""
        
        # Create subnets with IP allocation
        subnets = []
        hosts = []
        
        for subnet_name, config in subnet_specs.items():
            # Allocate subnet IP range
            subnet_cidr = self._allocate_subnet_cidr(config["hosts"])
            
            # Create subnet object
            subnet = NetworkSubnet(
                name=subnet_name,
                cidr=str(subnet_cidr),
                vlan_id=len(subnets) + 100  # Simple VLAN ID assignment
            )
            subnets.append(subnet)
            
            # Create hosts in subnet
            subnet_hosts = self._create_hosts_in_subnet(subnet, config)
            hosts.extend(subnet_hosts)
            
        # Create topology
        topology = NetworkTopology(
            subnets=subnets,
            hosts=hosts,
            name=template.name if template else "custom_topology"
        )
        
        return topology
        
    def _allocate_subnet_cidr(self, host_count: int) -> IPv4Network:
        """Allocate appropriate CIDR for host count"""
        
        # Calculate required subnet size
        required_ips = host_count + 10  # Buffer for network/broadcast/gateway
        
        # Find appropriate subnet mask
        subnet_bits = 0
        while (2 ** subnet_bits) < required_ips:
            subnet_bits += 1
            
        prefix_length = 32 - subnet_bits
        
        # Find available subnet in pool
        for subnet in self._ip_pool.subnets(new_prefix=prefix_length):
            if not any(subnet.overlaps(allocated) for allocated in self._allocated_subnets):
                self._allocated_subnets.append(subnet)
                return subnet
                
        raise NetworkConfigurationError("No available IP space for subnet allocation")
        
    def _create_hosts_in_subnet(self, 
                              subnet: NetworkSubnet, 
                              config: Dict[str, Any]) -> List[NetworkHost]:
        """Create hosts within a subnet"""
        
        hosts = []
        subnet_network = IPv4Network(subnet.cidr)
        available_ips = list(subnet_network.hosts())
        
        host_count = config["hosts"]
        services = config.get("services", [])
        
        for i in range(min(host_count, len(available_ips))):
            ip = available_ips[i]
            
            # Generate realistic hostname
            hostname = f"{subnet.name}-host-{i+1:03d}"
            
            # Select random OS (could be more sophisticated)
            os_types = ["Windows 10", "Windows Server 2019", "Ubuntu 20.04", "CentOS 8", "macOS"]
            os_type = random.choice(os_types)
            
            # Create host
            host = NetworkHost(
                hostname=hostname,
                ip_address=str(ip),
                subnet=subnet.name,
                operating_system=os_type,
                services=services.copy()
            )
            
            hosts.append(host)
            
        return hosts
        
    def _inject_vulnerabilities(self, topology: NetworkTopology, density: float):
        """Inject realistic vulnerabilities into topology"""
        
        vulnerability_types = [
            "unpatched_software",
            "weak_passwords", 
            "open_ports",
            "misconfigured_services",
            "outdated_certificates",
            "default_credentials"
        ]
        
        total_hosts = len(topology.hosts)
        vuln_count = int(total_hosts * density)
        
        # Randomly distribute vulnerabilities
        vulnerable_hosts = random.sample(topology.hosts, min(vuln_count, total_hosts))
        
        for host in vulnerable_hosts:
            vuln_type = random.choice(vulnerability_types)
            
            # Add vulnerability to host metadata
            if not hasattr(host, 'vulnerabilities'):
                host.vulnerabilities = []
            host.vulnerabilities.append(vuln_type)
            
        logger.info(f"Injected {len(vulnerable_hosts)} vulnerabilities into topology")
        
    def _configure_services(self, topology: NetworkTopology, default_services: List[str]):
        """Configure network services across topology"""
        
        # This is a simplified implementation
        # In practice, this would configure specific services on appropriate hosts
        service_ports = {
            "web": [80, 443],
            "email": [25, 110, 143, 993, 995],
            "database": [3306, 5432, 1433],
            "file_share": [445, 2049],
            "monitoring": [9090, 3000],
            "dns": [53]
        }
        
        for host in topology.hosts:
            # Add default ports for host services
            if not hasattr(host, 'open_ports'):
                host.open_ports = []
                
            for service in host.services:
                if service in service_ports:
                    host.open_ports.extend(service_ports[service])
                    
        logger.info(f"Configured services across {len(topology.hosts)} hosts")
        
    def _validate_subnet_specs(self, subnet_specs: Dict[str, Dict[str, Any]]):
        """Validate subnet specifications"""
        
        for subnet_name, config in subnet_specs.items():
            if "hosts" not in config:
                raise NetworkConfigurationError(f"Subnet {subnet_name} missing host count")
            if not isinstance(config["hosts"], int) or config["hosts"] < 1:
                raise NetworkConfigurationError(f"Invalid host count for subnet {subnet_name}")
                
    def _calculate_complexity_score(self, subnet_specs: Dict[str, Dict[str, Any]]) -> int:
        """Calculate complexity score for custom topology"""
        
        total_hosts = sum(config["hosts"] for config in subnet_specs.values())
        subnet_count = len(subnet_specs)
        
        # Simple complexity scoring
        if total_hosts < 20 and subnet_count < 3:
            return 2
        elif total_hosts < 50 and subnet_count < 5:
            return 4
        elif total_hosts < 100 and subnet_count < 8:
            return 6
        else:
            return 8
            
    def _find_isolated_hosts(self, topology: NetworkTopology) -> List[NetworkHost]:
        """Find hosts that might be isolated"""
        # Simplified implementation - would need more sophisticated network analysis
        return []
        
    def _find_oversized_subnets(self, topology: NetworkTopology) -> List[str]:
        """Find subnets that are unusually large"""
        oversized = []
        host_counts = {}
        
        for host in topology.hosts:
            subnet = host.subnet
            host_counts[subnet] = host_counts.get(subnet, 0) + 1
            
        # Flag subnets with more than 100 hosts
        for subnet, count in host_counts.items():
            if count > 100:
                oversized.append(subnet)
                
        return oversized
        
    def _check_security_issues(self, topology: NetworkTopology) -> List[str]:
        """Check for potential security issues in topology"""
        issues = []
        
        # Check for hosts without services (might be misconfigured)
        hosts_without_services = [h for h in topology.hosts if not h.services]
        if len(hosts_without_services) > len(topology.hosts) * 0.2:  # More than 20%
            issues.append("High number of hosts without configured services")
            
        return issues
        
    def _generate_topology_recommendations(self, topology: NetworkTopology) -> List[str]:
        """Generate recommendations for topology improvement"""
        recommendations = []
        
        # Analyze host distribution
        host_counts = {}
        for host in topology.hosts:
            subnet = host.subnet
            host_counts[subnet] = host_counts.get(subnet, 0) + 1
            
        # Recommend load balancing
        max_hosts = max(host_counts.values()) if host_counts else 0
        min_hosts = min(host_counts.values()) if host_counts else 0
        
        if max_hosts > min_hosts * 3:
            recommendations.append("Consider rebalancing host distribution across subnets")
            
        return recommendations
        
    def _interconnect_networks(self, networks: List[NetworkTopology], density: float) -> NetworkTopology:
        """Interconnect multiple networks to simulate internet topology"""
        # Simplified implementation - would create routing between networks
        # For now, just combine all networks into one large topology
        
        combined_subnets = []
        combined_hosts = []
        
        for network in networks:
            combined_subnets.extend(network.subnets)
            combined_hosts.extend(network.hosts)
            
        return NetworkTopology(
            subnets=combined_subnets,
            hosts=combined_hosts,
            name="internet_simulation"
        )
        
    def _add_internet_infrastructure(self, topology: NetworkTopology):
        """Add internet infrastructure components"""
        # Add ISP infrastructure, routing, etc.
        # Simplified implementation
        pass
        
    def _apply_scenario_enhancements(self, 
                                   topology: NetworkTopology,
                                   modifications: Dict[str, Any],
                                   difficulty: str) -> NetworkTopology:
        """Apply enhancements for specific attack scenarios"""
        
        # This would add scenario-specific network characteristics
        # For example, trust relationships for lateral movement scenarios
        
        enhanced = topology
        
        if modifications.get("trust_relationships"):
            # Add domain trust configurations
            pass
            
        if modifications.get("vulnerability_clusters"):
            # Group vulnerabilities for realistic attack paths
            pass
            
        return enhanced