"""
Comprehensive validation framework for GAN-Cyber-Range-v2.

This module provides validation for configurations, inputs, and system state
to ensure safe and correct operation of the cyber range platform.
"""

import re
import ipaddress
import json
from typing import Dict, Any, List, Optional, Union, Callable, Type
from dataclasses import dataclass
from pathlib import Path
import logging

from .error_handling import CyberRangeError, ErrorSeverity, ErrorContext


logger = logging.getLogger(__name__)


class ValidationError(CyberRangeError):
    """Validation-specific error"""
    
    def __init__(self, message: str, field: str, value: Any = None, **kwargs):
        context = ErrorContext(
            module="validation",
            function="validate",
            additional_info={'field': field, 'value': str(value) if value is not None else None}
        )
        super().__init__(
            message=message,
            error_code="CR_VALIDATION",
            severity=ErrorSeverity.HIGH,
            context=context,
            user_message=f"Validation failed for {field}: {message}",
            recoverable=False,
            **kwargs
        )
        self.field = field
        self.value = value


@dataclass
class ValidationRule:
    """Defines a validation rule"""
    field: str
    validator: Callable[[Any], bool]
    error_message: str
    required: bool = True
    transform: Optional[Callable[[Any], Any]] = None


class Validator:
    """Base validator class"""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.errors: List[ValidationError] = []
    
    def add_rule(
        self,
        field: str,
        validator: Callable[[Any], bool],
        error_message: str,
        required: bool = True,
        transform: Optional[Callable[[Any], Any]] = None
    ) -> 'Validator':
        """Add a validation rule"""
        rule = ValidationRule(field, validator, error_message, required, transform)
        self.rules.append(rule)
        return self
    
    def validate(self, data: Dict[str, Any], strict: bool = False) -> bool:
        """
        Validate data against all rules.
        
        Args:
            data: Data to validate
            strict: If True, raise on first validation error
            
        Returns:
            True if all validations pass, False otherwise
        """
        self.errors.clear()
        valid = True
        
        for rule in self.rules:
            try:
                field_valid = self._validate_field(data, rule)
                if not field_valid:
                    valid = False
                    if strict:
                        raise self.errors[-1]
            except ValidationError:
                valid = False
                if strict:
                    raise
        
        return valid
    
    def get_errors(self) -> List[ValidationError]:
        """Get all validation errors"""
        return self.errors.copy()
    
    def _validate_field(self, data: Dict[str, Any], rule: ValidationRule) -> bool:
        """Validate a single field"""
        
        field_value = data.get(rule.field)
        
        # Check if required field is present
        if rule.required and field_value is None:
            error = ValidationError(
                f"Required field '{rule.field}' is missing",
                field=rule.field,
                value=field_value
            )
            self.errors.append(error)
            return False
        
        # Skip validation if field is not required and not present
        if not rule.required and field_value is None:
            return True
        
        # Apply transformation if provided
        if rule.transform:
            try:
                field_value = rule.transform(field_value)
                data[rule.field] = field_value  # Update data with transformed value
            except Exception as e:
                error = ValidationError(
                    f"Transformation failed for field '{rule.field}': {e}",
                    field=rule.field,
                    value=field_value
                )
                self.errors.append(error)
                return False
        
        # Run validator
        try:
            if not rule.validator(field_value):
                error = ValidationError(
                    rule.error_message,
                    field=rule.field,
                    value=field_value
                )
                self.errors.append(error)
                return False
        except Exception as e:
            error = ValidationError(
                f"Validation function failed for field '{rule.field}': {e}",
                field=rule.field,
                value=field_value
            )
            self.errors.append(error)
            return False
        
        return True


class NetworkTopologyValidator(Validator):
    """Validator for network topology configurations"""
    
    def __init__(self):
        super().__init__()
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup validation rules for network topology"""
        
        # Validate topology name
        self.add_rule(
            'name',
            lambda x: isinstance(x, str) and 3 <= len(x) <= 50 and re.match(r'^[a-zA-Z0-9_-]+$', x),
            "Name must be 3-50 characters, alphanumeric, underscore, or dash only"
        )
        
        # Validate subnets
        self.add_rule(
            'subnets',
            lambda x: isinstance(x, list) and len(x) > 0,
            "Subnets must be a non-empty list"
        )
        
        # Validate hosts
        self.add_rule(
            'hosts',
            lambda x: isinstance(x, list) and len(x) > 0,
            "Hosts must be a non-empty list"
        )
    
    def validate_subnet(self, subnet: Dict[str, Any]) -> bool:
        """Validate individual subnet configuration"""
        
        subnet_validator = Validator()
        
        subnet_validator.add_rule(
            'name',
            lambda x: isinstance(x, str) and len(x) > 0,
            "Subnet name is required"
        )
        
        subnet_validator.add_rule(
            'cidr',
            self._validate_cidr,
            "Invalid CIDR notation"
        )
        
        subnet_validator.add_rule(
            'security_zone',
            lambda x: x in ['dmz', 'internal', 'management', 'guest'],
            "Security zone must be one of: dmz, internal, management, guest"
        )
        
        valid = subnet_validator.validate(subnet, strict=False)
        self.errors.extend(subnet_validator.get_errors())
        
        return valid
    
    def validate_host(self, host: Dict[str, Any]) -> bool:
        """Validate individual host configuration"""
        
        host_validator = Validator()
        
        host_validator.add_rule(
            'name',
            lambda x: isinstance(x, str) and len(x) > 0,
            "Host name is required"
        )
        
        host_validator.add_rule(
            'ip_address',
            self._validate_ip_address,
            "Invalid IP address"
        )
        
        host_validator.add_rule(
            'host_type',
            lambda x: x in ['workstation', 'server', 'router', 'firewall', 'switch', 'iot_device'],
            "Invalid host type"
        )
        
        host_validator.add_rule(
            'os_type',
            lambda x: x in ['windows', 'linux', 'macos', 'router_os', 'firewall_os', 'iot_os'],
            "Invalid OS type"
        )
        
        valid = host_validator.validate(host, strict=False)
        self.errors.extend(host_validator.get_errors())
        
        return valid
    
    def _validate_cidr(self, cidr: str) -> bool:
        """Validate CIDR notation"""
        try:
            ipaddress.ip_network(cidr, strict=False)
            return True
        except (ipaddress.AddressValueError, ValueError):
            return False
    
    def _validate_ip_address(self, ip: str) -> bool:
        """Validate IP address"""
        try:
            ipaddress.ip_address(ip)
            return True
        except (ipaddress.AddressValueError, ValueError):
            return False


class AttackConfigValidator(Validator):
    """Validator for attack configurations"""
    
    def __init__(self):
        super().__init__()
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup validation rules for attack configuration"""
        
        # Validate attack name
        self.add_rule(
            'name',
            lambda x: isinstance(x, str) and len(x) > 0,
            "Attack name is required"
        )
        
        # Validate technique ID (MITRE ATT&CK format)
        self.add_rule(
            'technique_id',
            lambda x: isinstance(x, str) and re.match(r'^T\d{4}(\.\d{3})?$', x),
            "Technique ID must be in MITRE ATT&CK format (e.g., T1059, T1059.001)"
        )
        
        # Validate phase
        self.add_rule(
            'phase',
            lambda x: x in ['reconnaissance', 'weaponization', 'delivery', 'exploitation', 
                           'installation', 'command_control', 'actions'],
            "Invalid attack phase"
        )
        
        # Validate target host
        self.add_rule(
            'target_host',
            lambda x: isinstance(x, str) and len(x) > 0,
            "Target host is required"
        )
        
        # Validate success probability
        self.add_rule(
            'success_probability',
            lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
            "Success probability must be between 0.0 and 1.0",
            required=False
        )
        
        # Validate payload
        self.add_rule(
            'payload',
            lambda x: isinstance(x, dict),
            "Payload must be a dictionary",
            required=False
        )


class GANConfigValidator(Validator):
    """Validator for GAN configuration"""
    
    def __init__(self):
        super().__init__()
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup validation rules for GAN configuration"""
        
        # Validate architecture
        self.add_rule(
            'architecture',
            lambda x: x in ['standard', 'wasserstein', 'conditional', 'cyclic'],
            "Invalid GAN architecture"
        )
        
        # Validate attack types
        self.add_rule(
            'attack_types',
            lambda x: isinstance(x, list) and all(
                t in ['malware', 'network', 'web', 'social_engineering', 'physical'] for t in x
            ),
            "Invalid attack types"
        )
        
        # Validate noise dimension
        self.add_rule(
            'noise_dim',
            lambda x: isinstance(x, int) and 50 <= x <= 1000,
            "Noise dimension must be between 50 and 1000"
        )
        
        # Validate training mode
        self.add_rule(
            'training_mode',
            lambda x: x in ['standard', 'differential_privacy', 'federated'],
            "Invalid training mode"
        )
        
        # Validate epochs
        self.add_rule(
            'epochs',
            lambda x: isinstance(x, int) and 1 <= x <= 10000,
            "Epochs must be between 1 and 10000",
            required=False
        )
        
        # Validate batch size
        self.add_rule(
            'batch_size',
            lambda x: isinstance(x, int) and 1 <= x <= 1024 and (x & (x-1)) == 0,
            "Batch size must be a power of 2 between 1 and 1024",
            required=False
        )


class CyberRangeConfigValidator(Validator):
    """Validator for cyber range configuration"""
    
    def __init__(self):
        super().__init__()
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup validation rules for cyber range configuration"""
        
        # Validate range name
        self.add_rule(
            'name',
            lambda x: isinstance(x, str) and 3 <= len(x) <= 50,
            "Range name must be 3-50 characters"
        )
        
        # Validate hypervisor
        self.add_rule(
            'hypervisor',
            lambda x: x in ['docker', 'kvm', 'vmware', 'virtualbox'],
            "Invalid hypervisor type"
        )
        
        # Validate isolation level
        self.add_rule(
            'isolation_level',
            lambda x: x in ['container', 'vm', 'strict'],
            "Invalid isolation level"
        )
        
        # Validate resource limits
        self.add_rule(
            'resource_limits',
            self._validate_resource_limits,
            "Invalid resource limits configuration"
        )
    
    def _validate_resource_limits(self, limits: Dict[str, Any]) -> bool:
        """Validate resource limits"""
        
        if not isinstance(limits, dict):
            return False
        
        # Validate CPU cores
        cpu_cores = limits.get('cpu_cores', 0)
        if not isinstance(cpu_cores, int) or cpu_cores < 1 or cpu_cores > 64:
            return False
        
        # Validate memory
        memory_gb = limits.get('memory_gb', 0)
        if not isinstance(memory_gb, int) or memory_gb < 1 or memory_gb > 512:
            return False
        
        # Validate storage
        storage_gb = limits.get('storage_gb', 0)
        if not isinstance(storage_gb, int) or storage_gb < 1 or storage_gb > 10000:
            return False
        
        return True


class SecurityValidator:
    """Security-focused validator for preventing malicious configurations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.blocked_patterns = self._load_blocked_patterns()
    
    def validate_attack_payload(self, payload: Dict[str, Any]) -> bool:
        """Validate attack payload for security concerns"""
        
        # Check for potentially dangerous commands
        if 'command' in payload:
            command = str(payload['command']).lower()
            
            # Block dangerous system commands
            dangerous_commands = [
                'rm -rf /',
                'del /f /s /q c:\\',
                'format c:',
                'shutdown',
                'reboot',
                'dd if=/dev/zero',
                'mkfs.',
                ':(){ :|:& };:'  # Fork bomb
            ]
            
            for dangerous in dangerous_commands:
                if dangerous in command:
                    self.logger.error(f"Blocked dangerous command in payload: {dangerous}")
                    return False
        
        # Check for network access outside allowed ranges
        if 'target_ip' in payload:
            try:
                ip = ipaddress.ip_address(payload['target_ip'])
                
                # Block attempts to target external networks
                allowed_ranges = [
                    ipaddress.ip_network('192.168.0.0/16'),
                    ipaddress.ip_network('10.0.0.0/8'),
                    ipaddress.ip_network('172.16.0.0/12'),
                    ipaddress.ip_network('127.0.0.0/8')
                ]
                
                if not any(ip in range_ for range_ in allowed_ranges):
                    self.logger.error(f"Blocked attempt to target external IP: {ip}")
                    return False
                    
            except ValueError:
                self.logger.error(f"Invalid IP address in payload: {payload['target_ip']}")
                return False
        
        return True
    
    def validate_file_path(self, file_path: str) -> bool:
        """Validate file paths to prevent directory traversal"""
        
        path = Path(file_path)
        
        # Check for directory traversal attempts
        if '..' in path.parts:
            self.logger.error(f"Blocked directory traversal attempt: {file_path}")
            return False
        
        # Check for absolute paths outside allowed directories
        if path.is_absolute():
            allowed_roots = ['/tmp', '/var/tmp', './data', './logs', './models']
            
            if not any(str(path).startswith(root) for root in allowed_roots):
                self.logger.error(f"Blocked access to restricted path: {file_path}")
                return False
        
        return True
    
    def validate_network_config(self, config: Dict[str, Any]) -> bool:
        """Validate network configuration for security"""
        
        # Ensure network isolation is enabled
        if config.get('isolation_level') == 'none':
            self.logger.error("Network isolation cannot be disabled")
            return False
        
        # Validate CIDR ranges don't overlap with production networks
        if 'subnets' in config:
            for subnet in config['subnets']:
                cidr = subnet.get('cidr')
                if cidr:
                    try:
                        network = ipaddress.ip_network(cidr)
                        
                        # Block common production network ranges
                        blocked_ranges = [
                            ipaddress.ip_network('8.8.8.0/24'),  # Google DNS
                            ipaddress.ip_network('1.1.1.0/24'),  # Cloudflare DNS
                        ]
                        
                        for blocked in blocked_ranges:
                            if network.overlaps(blocked):
                                self.logger.error(f"Blocked overlap with production network: {cidr}")
                                return False
                                
                    except ValueError:
                        self.logger.error(f"Invalid CIDR in network config: {cidr}")
                        return False
        
        return True
    
    def _load_blocked_patterns(self) -> List[str]:
        """Load patterns of blocked content"""
        
        # In a real implementation, this would load from a configuration file
        return [
            r'(?i)(password|passwd|pwd)\s*=\s*["\']?[^"\'\s]+',  # Hardcoded passwords
            r'(?i)(api[_-]?key|secret[_-]?key)\s*=\s*["\']?[^"\'\s]+',  # API keys
            r'(?i)(token)\s*=\s*["\']?[^"\'\s]+',  # Tokens
            r'(?i)(private[_-]?key)',  # Private keys
        ]


def validate_config(config: Dict[str, Any], config_type: str = "general") -> bool:
    """
    Validate configuration based on type.
    
    Args:
        config: Configuration dictionary to validate
        config_type: Type of configuration ('topology', 'attack', 'gan', 'range')
        
    Returns:
        True if valid, raises ValidationError if invalid
    """
    
    validators = {
        'topology': NetworkTopologyValidator(),
        'attack': AttackConfigValidator(),
        'gan': GANConfigValidator(),
        'range': CyberRangeConfigValidator()
    }
    
    if config_type not in validators:
        raise ValidationError(
            f"Unknown configuration type: {config_type}",
            field="config_type",
            value=config_type
        )
    
    validator = validators[config_type]
    
    if not validator.validate(config, strict=True):
        # If we get here, there was a validation error that wasn't raised in strict mode
        errors = validator.get_errors()
        if errors:
            raise errors[0]  # Raise the first error
    
    # Additional security validation
    security_validator = SecurityValidator()
    
    if config_type == 'attack' and 'payload' in config:
        if not security_validator.validate_attack_payload(config['payload']):
            raise ValidationError(
                "Attack payload failed security validation",
                field="payload",
                value=config['payload']
            )
    
    if config_type in ['topology', 'range'] and 'network' in str(config):
        if not security_validator.validate_network_config(config):
            raise ValidationError(
                "Network configuration failed security validation",
                field="network",
                value=config
            )
    
    logger.info(f"Configuration validation passed for type: {config_type}")
    return True


def sanitize_input(input_data: Any, input_type: str = "string") -> Any:
    """
    Sanitize input data to prevent injection attacks.
    
    Args:
        input_data: Data to sanitize
        input_type: Type of input ('string', 'filename', 'command', 'sql')
        
    Returns:
        Sanitized input data
    """
    
    if input_data is None:
        return None
    
    if input_type == "string":
        # Basic string sanitization
        if isinstance(input_data, str):
            # Remove or escape dangerous characters
            sanitized = re.sub(r'[<>"\';]', '', input_data)
            return sanitized.strip()
    
    elif input_type == "filename":
        # Filename sanitization
        if isinstance(input_data, str):
            # Remove path traversal and dangerous characters
            sanitized = re.sub(r'[^\w\.-]', '_', input_data)
            sanitized = re.sub(r'\.\.', '_', sanitized)
            return sanitized[:255]  # Limit length
    
    elif input_type == "command":
        # Command sanitization - very restrictive
        if isinstance(input_data, str):
            # Only allow alphanumeric, spaces, and safe characters
            sanitized = re.sub(r'[^a-zA-Z0-9\s\-_.]', '', input_data)
            return sanitized.strip()
    
    elif input_type == "sql":
        # SQL input sanitization
        if isinstance(input_data, str):
            # Escape SQL special characters
            sanitized = input_data.replace("'", "''")
            sanitized = sanitized.replace('"', '""')
            sanitized = sanitized.replace(';', '')
            return sanitized
    
    return input_data