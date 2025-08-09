"""
CyberRange Factory for creating and configuring cyber range environments.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from ..core.cyber_range import CyberRange, RangeConfig, ResourceLimits
from ..core.network_sim import NetworkTopology
from ..utils.security import SecurityManager
from ..utils.monitoring import MetricsCollector
from ..utils.error_handling import CyberRangeError

logger = logging.getLogger(__name__)


@dataclass
class RangeTemplateConfig:
    """Template configuration for cyber ranges"""
    name: str
    description: str
    target_audience: str  # "students", "professionals", "researchers"
    difficulty_level: str  # "beginner", "intermediate", "advanced", "expert"
    estimated_duration: str
    learning_objectives: List[str]
    network_template: str
    resource_requirements: ResourceLimits
    security_level: str = "standard"
    enable_monitoring: bool = True
    enable_recording: bool = True


class CyberRangeFactory:
    """Factory for creating cyber range environments with smart templates"""
    
    # Predefined templates for different use cases
    TEMPLATES = {
        "educational_basic": RangeTemplateConfig(
            name="Educational Basic Range",
            description="Basic cyber range for cybersecurity education",
            target_audience="students",
            difficulty_level="beginner",
            estimated_duration="2_hours",
            learning_objectives=[
                "Understand network topology",
                "Identify basic attack patterns",
                "Practice incident response"
            ],
            network_template="small_enterprise",
            resource_requirements=ResourceLimits(cpu_cores=4, memory_gb=8, storage_gb=50)
        ),
        "professional_training": RangeTemplateConfig(
            name="Professional Training Range", 
            description="Comprehensive range for security professionals",
            target_audience="professionals",
            difficulty_level="intermediate",
            estimated_duration="8_hours",
            learning_objectives=[
                "Advanced threat detection",
                "Incident response coordination",
                "Forensic analysis techniques"
            ],
            network_template="enterprise",
            resource_requirements=ResourceLimits(cpu_cores=8, memory_gb=16, storage_gb=100)
        ),
        "research_advanced": RangeTemplateConfig(
            name="Advanced Research Range",
            description="High-fidelity range for cybersecurity research", 
            target_audience="researchers",
            difficulty_level="expert",
            estimated_duration="variable",
            learning_objectives=[
                "Novel attack technique analysis",
                "Defense mechanism evaluation",
                "Performance benchmarking"
            ],
            network_template="complex_enterprise",
            resource_requirements=ResourceLimits(cpu_cores=16, memory_gb=32, storage_gb=200)
        ),
        "ctf_competition": RangeTemplateConfig(
            name="CTF Competition Range",
            description="Competition-focused cyber range",
            target_audience="students",
            difficulty_level="advanced",
            estimated_duration="4_hours",
            learning_objectives=[
                "Rapid vulnerability identification",
                "Exploit development",
                "Flag capture techniques"
            ],
            network_template="ctf_challenges",
            resource_requirements=ResourceLimits(cpu_cores=6, memory_gb=12, storage_gb=75)
        )
    }
    
    def __init__(self, security_manager: Optional[SecurityManager] = None):
        self.security_manager = security_manager or SecurityManager()
        self.metrics_collector = MetricsCollector()
        self._active_ranges: Dict[str, CyberRange] = {}
        
    def create_from_template(self, 
                           template_name: str,
                           custom_config: Optional[Dict[str, Any]] = None) -> CyberRange:
        """Create a cyber range from a predefined template"""
        
        if template_name not in self.TEMPLATES:
            available = list(self.TEMPLATES.keys())
            raise ValueError(f"Unknown template: {template_name}. Available: {available}")
            
        template = self.TEMPLATES[template_name]
        logger.info(f"Creating cyber range from template: {template_name}")
        
        # Validate security clearance
        if not self._validate_template_access(template):
            raise CyberRangeError(f"Insufficient permissions for template: {template_name}")
            
        # Create network topology
        topology = self._create_network_topology(template.network_template, custom_config)
        
        # Configure range
        range_config = RangeConfig(
            name=f"{template.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            topology=topology,
            resource_limits=template.resource_requirements,
            security_level=template.security_level,
            monitoring_enabled=template.enable_monitoring,
            recording_enabled=template.enable_recording
        )
        
        # Apply custom configuration
        if custom_config:
            range_config = self._apply_custom_config(range_config, custom_config)
            
        # Create cyber range
        cyber_range = CyberRange(range_config)
        
        # Apply template-specific enhancements
        cyber_range = self._enhance_with_template_features(cyber_range, template)
        
        # Register for monitoring
        range_id = cyber_range.config.name
        self._active_ranges[range_id] = cyber_range
        self.metrics_collector.register_range(range_id)
        
        return cyber_range
        
    def create_custom_range(self,
                          name: str,
                          topology_config: Dict[str, Any],
                          resource_limits: Optional[ResourceLimits] = None,
                          security_requirements: Optional[Dict[str, Any]] = None) -> CyberRange:
        """Create a custom cyber range with specific requirements"""
        
        logger.info(f"Creating custom cyber range: {name}")
        
        # Validate custom configuration
        self._validate_custom_config(topology_config, security_requirements)
        
        # Create topology
        topology = NetworkTopology.from_config(topology_config)
        
        # Set default resource limits if not provided
        if resource_limits is None:
            resource_limits = self._estimate_resource_requirements(topology)
            
        # Configure range
        range_config = RangeConfig(
            name=name,
            topology=topology,
            resource_limits=resource_limits,
            security_level=security_requirements.get("security_level", "standard") if security_requirements else "standard"
        )
        
        # Create and enhance cyber range
        cyber_range = CyberRange(range_config)
        cyber_range = self._apply_security_enhancements(cyber_range, security_requirements)
        
        # Register
        self._active_ranges[name] = cyber_range
        self.metrics_collector.register_range(name)
        
        return cyber_range
        
    def create_multi_tenant_range(self,
                                base_template: str,
                                tenant_count: int,
                                isolation_level: str = "strict") -> List[CyberRange]:
        """Create multiple isolated ranges for multi-tenant scenarios"""
        
        if tenant_count < 1 or tenant_count > 100:
            raise ValueError("Tenant count must be between 1 and 100")
            
        logger.info(f"Creating {tenant_count} tenant ranges from template: {base_template}")
        
        ranges = []
        base_config = self.TEMPLATES[base_template]
        
        for i in range(tenant_count):
            # Create isolated configuration
            tenant_config = {
                "tenant_id": f"tenant_{i+1:03d}",
                "isolation_level": isolation_level,
                "network_prefix": f"10.{100+i}.0.0/16"
            }
            
            # Create range with tenant-specific configuration
            cyber_range = self.create_from_template(base_template, tenant_config)
            
            # Apply tenant isolation
            cyber_range = self._apply_tenant_isolation(cyber_range, tenant_config)
            
            ranges.append(cyber_range)
            
        return ranges
        
    def clone_range(self, source_range: CyberRange, new_name: str) -> CyberRange:
        """Clone an existing range with a new name"""
        
        logger.info(f"Cloning range {source_range.config.name} to {new_name}")
        
        # Create new configuration based on source
        cloned_config = RangeConfig(
            name=new_name,
            topology=source_range.config.topology,
            resource_limits=source_range.config.resource_limits,
            security_level=source_range.config.security_level
        )
        
        # Create new range
        cloned_range = CyberRange(cloned_config)
        
        # Copy customizations (if any)
        self._copy_range_customizations(source_range, cloned_range)
        
        # Register
        self._active_ranges[new_name] = cloned_range
        self.metrics_collector.register_range(new_name)
        
        return cloned_range
        
    def get_active_ranges(self) -> Dict[str, CyberRange]:
        """Get all currently active ranges"""
        return self._active_ranges.copy()
        
    def get_range_statistics(self) -> Dict[str, Any]:
        """Get statistics about all managed ranges"""
        stats = {
            "total_ranges": len(self._active_ranges),
            "templates_available": len(self.TEMPLATES),
            "ranges_by_template": {},
            "resource_utilization": {}
        }
        
        # Analyze ranges by template
        for range_id, cyber_range in self._active_ranges.items():
            # Try to identify template used (simplified)
            template_name = "custom"
            for tpl_name, template in self.TEMPLATES.items():
                if template.name in cyber_range.config.name:
                    template_name = tpl_name
                    break
                    
            stats["ranges_by_template"][template_name] = stats["ranges_by_template"].get(template_name, 0) + 1
            
        return stats
        
    def shutdown_range(self, range_id: str):
        """Shutdown and cleanup a specific range"""
        if range_id in self._active_ranges:
            cyber_range = self._active_ranges[range_id]
            cyber_range.shutdown()
            del self._active_ranges[range_id]
            self.metrics_collector.unregister_range(range_id)
            logger.info(f"Range {range_id} shutdown and cleaned up")
        else:
            logger.warning(f"Range {range_id} not found in active ranges")
            
    def shutdown_all_ranges(self):
        """Shutdown all managed ranges"""
        range_ids = list(self._active_ranges.keys())
        for range_id in range_ids:
            self.shutdown_range(range_id)
        logger.info("All ranges shutdown successfully")
        
    def _validate_template_access(self, template: RangeTemplateConfig) -> bool:
        """Validate user has access to template"""
        # Check if user has appropriate clearance for template
        required_clearance = {
            "educational_basic": "public",
            "professional_training": "restricted", 
            "research_advanced": "confidential",
            "ctf_competition": "restricted"
        }
        
        template_key = None
        for key, tpl in self.TEMPLATES.items():
            if tpl == template:
                template_key = key
                break
                
        if template_key:
            required = required_clearance.get(template_key, "restricted")
            return self.security_manager.check_clearance_level(required)
            
        return False
        
    def _create_network_topology(self, 
                               template_name: str, 
                               custom_config: Optional[Dict[str, Any]] = None) -> NetworkTopology:
        """Create network topology based on template"""
        
        topology_configs = {
            "small_enterprise": {
                "subnets": ["dmz", "internal"],
                "hosts_per_subnet": {"dmz": 3, "internal": 10},
                "services": ["web", "database", "email"]
            },
            "enterprise": {
                "subnets": ["dmz", "internal", "management"],
                "hosts_per_subnet": {"dmz": 5, "internal": 25, "management": 5},
                "services": ["web", "database", "email", "file_share", "vpn"]
            },
            "complex_enterprise": {
                "subnets": ["dmz", "internal", "management", "development", "production"],
                "hosts_per_subnet": {"dmz": 8, "internal": 50, "management": 10, "development": 20, "production": 30},
                "services": ["web", "database", "email", "file_share", "vpn", "ci_cd", "monitoring"]
            },
            "ctf_challenges": {
                "subnets": ["challenges", "admin"],
                "hosts_per_subnet": {"challenges": 15, "admin": 3},
                "services": ["web", "database", "custom_services"]
            }
        }
        
        config = topology_configs.get(template_name, topology_configs["small_enterprise"])
        
        # Apply custom modifications
        if custom_config and "topology_overrides" in custom_config:
            config.update(custom_config["topology_overrides"])
            
        return NetworkTopology.generate(**config)
        
    def _apply_custom_config(self, range_config: RangeConfig, custom_config: Dict[str, Any]) -> RangeConfig:
        """Apply custom configuration to range config"""
        
        if "resource_multiplier" in custom_config:
            multiplier = custom_config["resource_multiplier"]
            range_config.resource_limits.cpu_cores = int(range_config.resource_limits.cpu_cores * multiplier)
            range_config.resource_limits.memory_gb = int(range_config.resource_limits.memory_gb * multiplier)
            range_config.resource_limits.storage_gb = int(range_config.resource_limits.storage_gb * multiplier)
            
        if "security_level" in custom_config:
            range_config.security_level = custom_config["security_level"]
            
        if "name_suffix" in custom_config:
            range_config.name += f"_{custom_config['name_suffix']}"
            
        return range_config
        
    def _enhance_with_template_features(self, cyber_range: CyberRange, template: RangeTemplateConfig) -> CyberRange:
        """Enhance range with template-specific features"""
        
        # Add template metadata
        cyber_range.metadata = {
            "template_source": template.name,
            "target_audience": template.target_audience,
            "difficulty_level": template.difficulty_level,
            "learning_objectives": template.learning_objectives,
            "estimated_duration": template.estimated_duration
        }
        
        # Configure monitoring based on template
        if template.enable_monitoring:
            cyber_range.enable_comprehensive_monitoring()
            
        if template.enable_recording:
            cyber_range.enable_session_recording()
            
        return cyber_range
        
    def _validate_custom_config(self, 
                              topology_config: Dict[str, Any], 
                              security_requirements: Optional[Dict[str, Any]]):
        """Validate custom configuration parameters"""
        
        required_topology_keys = ["subnets", "hosts_per_subnet"]
        for key in required_topology_keys:
            if key not in topology_config:
                raise ValueError(f"Missing required topology configuration: {key}")
                
        # Validate security requirements if provided
        if security_requirements:
            valid_security_levels = ["public", "restricted", "confidential", "secret"]
            if "security_level" in security_requirements:
                if security_requirements["security_level"] not in valid_security_levels:
                    raise ValueError(f"Invalid security level. Valid options: {valid_security_levels}")
                    
    def _estimate_resource_requirements(self, topology: NetworkTopology) -> ResourceLimits:
        """Estimate resource requirements based on topology complexity"""
        
        total_hosts = len(topology.hosts)
        
        # Base requirements
        base_cpu = 2
        base_memory = 4
        base_storage = 20
        
        # Scale based on host count
        cpu_cores = base_cpu + (total_hosts // 5)
        memory_gb = base_memory + (total_hosts // 3)
        storage_gb = base_storage + (total_hosts * 2)
        
        return ResourceLimits(
            cpu_cores=min(cpu_cores, 32),  # Cap at reasonable limits
            memory_gb=min(memory_gb, 64),
            storage_gb=min(storage_gb, 500)
        )
        
    def _apply_security_enhancements(self, 
                                   cyber_range: CyberRange, 
                                   security_requirements: Optional[Dict[str, Any]]) -> CyberRange:
        """Apply security enhancements based on requirements"""
        
        if not security_requirements:
            return cyber_range
            
        # Apply network isolation
        if security_requirements.get("network_isolation"):
            cyber_range.enable_network_isolation()
            
        # Apply access controls
        if "access_control" in security_requirements:
            cyber_range.configure_access_control(security_requirements["access_control"])
            
        # Apply encryption
        if security_requirements.get("encrypt_communications", False):
            cyber_range.enable_encryption()
            
        return cyber_range
        
    def _apply_tenant_isolation(self, cyber_range: CyberRange, tenant_config: Dict[str, Any]) -> CyberRange:
        """Apply tenant-specific isolation"""
        
        tenant_id = tenant_config["tenant_id"]
        isolation_level = tenant_config["isolation_level"]
        
        # Configure network isolation
        if "network_prefix" in tenant_config:
            cyber_range.configure_tenant_network(tenant_config["network_prefix"])
            
        # Apply isolation policies
        if isolation_level == "strict":
            cyber_range.enable_strict_tenant_isolation(tenant_id)
        elif isolation_level == "moderate":
            cyber_range.enable_moderate_tenant_isolation(tenant_id)
            
        # Tag for tenant management
        cyber_range.set_tenant_tag(tenant_id)
        
        return cyber_range
        
    def _copy_range_customizations(self, source: CyberRange, target: CyberRange):
        """Copy customizations from source range to target"""
        
        # Copy metadata if it exists
        if hasattr(source, 'metadata'):
            target.metadata = source.metadata.copy()
            
        # Copy configuration overrides
        if hasattr(source, 'config_overrides'):
            target.config_overrides = source.config_overrides.copy()
            
        logger.info(f"Copied customizations from {source.config.name} to {target.config.name}")