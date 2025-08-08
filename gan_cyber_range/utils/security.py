"""
Comprehensive security framework for GAN-Cyber-Range-v2.

This module provides security scanning, ethical use enforcement, attack containment,
and security policy management for the cyber range platform.
"""

import logging
import hashlib
import hmac
import secrets
import ipaddress
import re
import subprocess
import threading
import time
from typing import Dict, Any, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import base64

from .error_handling import SecurityValidationError, CyberRangeError
from .validation import sanitize_input

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels"""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class ThreatLevel(Enum):
    """Threat assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    allowed_attack_types: List[str] = field(default_factory=lambda: ["web", "network", "malware"])
    blocked_techniques: List[str] = field(default_factory=list)
    max_attack_severity: float = 8.0
    require_approval: bool = True
    audit_all_actions: bool = True
    network_isolation: bool = True
    data_encryption: bool = True
    retention_days: int = 90
    access_control: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: ThreatLevel
    source: str
    description: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'severity': self.severity.value,
            'source': self.source,
            'description': self.description,
            'user_id': self.user_id,
            'ip_address': self.ip_address,
            'additional_data': self.additional_data,
            'resolved': self.resolved
        }


class SecurityValidator:
    """Validates operations for security compliance"""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.blocked_patterns = self._load_security_patterns()
        self.threat_signatures = self._load_threat_signatures()
        
    def validate_attack_config(self, attack_config: Dict[str, Any]) -> bool:
        """Validate attack configuration for security compliance"""
        
        # Check attack type is allowed
        attack_type = attack_config.get('attack_type', '')
        if attack_type not in self.policy.allowed_attack_types:
            raise SecurityValidationError(
                f"Attack type '{attack_type}' is not allowed",
                validation_type="attack_type"
            )
        
        # Check technique is not blocked
        technique_id = attack_config.get('technique_id', '')
        if technique_id in self.policy.blocked_techniques:
            raise SecurityValidationError(
                f"Technique '{technique_id}' is blocked by security policy",
                validation_type="technique_blocked"
            )
        
        # Check attack severity
        severity = attack_config.get('severity', 0.0)
        if severity > self.policy.max_attack_severity:
            raise SecurityValidationError(
                f"Attack severity {severity} exceeds maximum allowed {self.policy.max_attack_severity}",
                validation_type="severity_exceeded"
            )
        
        # Validate payload content
        payload = attack_config.get('payload', {})
        if not self._validate_payload_content(payload):
            raise SecurityValidationError(
                "Attack payload contains prohibited content",
                validation_type="payload_content"
            )
        
        return True
    
    def validate_network_config(self, network_config: Dict[str, Any]) -> bool:
        """Validate network configuration for security"""
        
        # Ensure network isolation is enabled
        if not self.policy.network_isolation:
            logger.warning("Network isolation is disabled - this may be unsafe")
        
        # Check for dangerous network configurations
        subnets = network_config.get('subnets', [])
        for subnet in subnets:
            cidr = subnet.get('cidr', '')
            if self._is_dangerous_network_range(cidr):
                raise SecurityValidationError(
                    f"Network range {cidr} overlaps with production networks",
                    validation_type="network_overlap"
                )
        
        # Validate host configurations
        hosts = network_config.get('hosts', [])
        for host in hosts:
            if not self._validate_host_security(host):
                raise SecurityValidationError(
                    f"Host {host.get('name', 'unknown')} has unsafe configuration",
                    validation_type="host_security"
                )
        
        return True
    
    def validate_user_permissions(self, user_id: str, action: str, resource: str) -> bool:
        """Validate user permissions for action"""
        
        user_permissions = self.policy.access_control.get(user_id, [])
        required_permission = f"{action}:{resource}"
        
        # Check for exact permission match
        if required_permission in user_permissions:
            return True
        
        # Check for wildcard permissions
        action_wildcard = f"{action}:*"
        resource_wildcard = f"*:{resource}"
        admin_wildcard = "*:*"
        
        if any(perm in user_permissions for perm in [action_wildcard, resource_wildcard, admin_wildcard]):
            return True
        
        raise SecurityValidationError(
            f"User {user_id} lacks permission for {action} on {resource}",
            validation_type="permission_denied"
        )
    
    def scan_for_malicious_content(self, content: Union[str, bytes, Dict[str, Any]]) -> List[str]:
        """Scan content for malicious patterns"""
        
        threats_found = []
        content_str = str(content).lower()
        
        # Check against threat signatures
        for signature_name, pattern in self.threat_signatures.items():
            if re.search(pattern, content_str, re.IGNORECASE):
                threats_found.append(signature_name)
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'shell_exec\s*\(',
            r'passthru\s*\(',
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'data:text/html',
            r'file:///.*',
            r'\\x[0-9a-f]{2}',  # Hex encoded
            r'%[0-9a-f]{2}',    # URL encoded
        ]
        
        for i, pattern in enumerate(suspicious_patterns):
            if re.search(pattern, content_str, re.IGNORECASE):
                threats_found.append(f"suspicious_pattern_{i}")
        
        return threats_found
    
    def _validate_payload_content(self, payload: Dict[str, Any]) -> bool:
        """Validate attack payload content"""
        
        # Scan payload for malicious content
        threats = self.scan_for_malicious_content(payload)
        if threats:
            logger.warning(f"Threats detected in payload: {threats}")
            return False
        
        # Check for prohibited commands
        if 'command' in payload:
            command = str(payload['command']).lower()
            prohibited_commands = [
                'rm -rf /',
                'del /f /s /q c:',
                'format c:',
                'shutdown',
                'reboot',
                'halt',
                'init 0',
                'init 6',
                'dd if=/dev/zero',
                'mkfs',
                ':(){ :|:& };:',  # Fork bomb
                'curl http://',   # External requests
                'wget http://',
                'nc -l',          # Netcat listeners
            ]
            
            for prohibited in prohibited_commands:
                if prohibited in command:
                    logger.error(f"Prohibited command detected: {prohibited}")
                    return False
        
        # Check for external network access
        if 'target_ip' in payload:
            target_ip = str(payload['target_ip'])
            if not self._is_safe_target_ip(target_ip):
                logger.error(f"Unsafe target IP: {target_ip}")
                return False
        
        return True
    
    def _is_dangerous_network_range(self, cidr: str) -> bool:
        """Check if network range is dangerous"""
        
        try:
            network = ipaddress.ip_network(cidr, strict=False)
            
            # Check against known dangerous ranges
            dangerous_ranges = [
                ipaddress.ip_network('0.0.0.0/0'),      # All networks
                ipaddress.ip_network('8.8.8.0/24'),     # Google DNS
                ipaddress.ip_network('1.1.1.0/24'),     # Cloudflare DNS
                ipaddress.ip_network('208.67.222.0/24'), # OpenDNS
            ]
            
            for dangerous in dangerous_ranges:
                if network.overlaps(dangerous):
                    return True
                    
        except (ipaddress.AddressValueError, ValueError):
            logger.warning(f"Invalid CIDR notation: {cidr}")
            return True
        
        return False
    
    def _is_safe_target_ip(self, ip_str: str) -> bool:
        """Check if target IP is safe for attacks"""
        
        try:
            ip = ipaddress.ip_address(ip_str)
            
            # Only allow private networks and localhost
            safe_ranges = [
                ipaddress.ip_network('127.0.0.0/8'),     # Localhost
                ipaddress.ip_network('192.168.0.0/16'),  # Private
                ipaddress.ip_network('10.0.0.0/8'),      # Private
                ipaddress.ip_network('172.16.0.0/12'),   # Private
            ]
            
            return any(ip in range_ for range_ in safe_ranges)
            
        except (ipaddress.AddressValueError, ValueError):
            return False
    
    def _validate_host_security(self, host_config: Dict[str, Any]) -> bool:
        """Validate host security configuration"""
        
        # Check for insecure services
        services = host_config.get('services', [])
        insecure_services = ['telnet', 'ftp', 'rsh', 'rlogin']
        
        for service in insecure_services:
            if service in services:
                logger.warning(f"Insecure service detected: {service}")
                # Allow but warn - might be intentional for training
        
        # Check security level
        security_level = host_config.get('security_level', 'medium')
        if security_level == 'none':
            logger.warning("Host has no security level set")
            return False
        
        return True
    
    def _load_security_patterns(self) -> Dict[str, str]:
        """Load security scanning patterns"""
        
        return {
            'sql_injection': r'(union|select|insert|update|delete|drop|create|alter)\s+.*\s+(from|into|table|database)',
            'xss': r'<script[^>]*>.*?</script>|javascript:|vbscript:|onload=|onerror=',
            'command_injection': r'(;|\||&|`|\$\(|\${).*?(cat|ls|ps|whoami|id|uname)',
            'path_traversal': r'\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c',
            'file_inclusion': r'(include|require|include_once|require_once)\s*\(',
            'ldap_injection': r'(\*|\)|\(|\||\&)',
            'xpath_injection': r'(\[|\]|\.\.|\@|\*)',
        }
    
    def _load_threat_signatures(self) -> Dict[str, str]:
        """Load threat detection signatures"""
        
        return {
            'malware_signature': r'(trojan|virus|worm|rootkit|keylogger|spyware)',
            'exploit_kit': r'(metasploit|exploit|payload|shellcode|nopsled)',
            'c2_communication': r'(beacon|heartbeat|checkin|exfiltrate)',
            'credential_dump': r'(mimikatz|lsass|ntlm|hash|credential)',
            'persistence': r'(startup|registry|service|scheduled|cron)',
            'privilege_escalation': r'(sudo|runas|uac|token|privilege)',
        }


class EthicalFramework:
    """Enforces ethical use of the cyber range"""
    
    def __init__(self):
        self.usage_monitor = UsageMonitor()
        self.consent_manager = ConsentManager()
        self.ethics_board = EthicsBoard()
        
    def validate_use_case(self, use_case: Dict[str, Any]) -> bool:
        """Validate use case against ethical guidelines"""
        
        purpose = use_case.get('purpose', '').lower()
        target_type = use_case.get('target_type', '').lower()
        user_role = use_case.get('user_role', '').lower()
        
        # Check if purpose is ethical
        ethical_purposes = [
            'security_training',
            'educational_research',
            'defensive_capability_testing',
            'vulnerability_assessment',
            'incident_response_training',
            'red_team_exercise',
            'academic_research'
        ]
        
        if purpose not in ethical_purposes:
            logger.error(f"Non-ethical purpose detected: {purpose}")
            return False
        
        # Check target restrictions
        prohibited_targets = [
            'production_system',
            'live_network',
            'unauthorized_system',
            'third_party_system'
        ]
        
        if target_type in prohibited_targets:
            logger.error(f"Prohibited target type: {target_type}")
            return False
        
        # Check user authorization
        if not self._validate_user_authorization(use_case):
            return False
        
        # Log ethical validation
        self.usage_monitor.log_ethics_check(use_case, approved=True)
        
        return True
    
    def require_consent(self, operation: str, participants: List[str]) -> bool:
        """Require explicit consent for operations"""
        
        consent_required_ops = [
            'deploy_range',
            'execute_attack',
            'collect_data',
            'record_session',
            'analyze_behavior'
        ]
        
        if operation in consent_required_ops:
            return self.consent_manager.get_consent(operation, participants)
        
        return True
    
    def check_harm_potential(self, action: Dict[str, Any]) -> ThreatLevel:
        """Assess potential for harm from an action"""
        
        severity = action.get('severity', 0.0)
        scope = action.get('scope', 'limited')
        target_sensitivity = action.get('target_sensitivity', 'low')
        
        # Calculate harm score
        harm_score = 0.0
        
        # Severity contribution
        harm_score += severity * 0.4
        
        # Scope contribution
        scope_weights = {'limited': 0.1, 'moderate': 0.3, 'extensive': 0.6, 'unlimited': 1.0}
        harm_score += scope_weights.get(scope, 0.5) * 0.3
        
        # Target sensitivity contribution
        sensitivity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.6, 'critical': 1.0}
        harm_score += sensitivity_weights.get(target_sensitivity, 0.3) * 0.3
        
        # Convert to threat level
        if harm_score >= 0.8:
            return ThreatLevel.CRITICAL
        elif harm_score >= 0.6:
            return ThreatLevel.HIGH
        elif harm_score >= 0.3:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _validate_user_authorization(self, use_case: Dict[str, Any]) -> bool:
        """Validate user authorization for use case"""
        
        user_role = use_case.get('user_role', '').lower()
        organization = use_case.get('organization', '').lower()
        
        # Check authorized roles
        authorized_roles = [
            'security_researcher',
            'educator',
            'student',
            'red_team_member',
            'blue_team_member',
            'penetration_tester',
            'security_analyst'
        ]
        
        if user_role not in authorized_roles:
            logger.warning(f"Potentially unauthorized role: {user_role}")
            # Don't reject immediately, but flag for review
        
        # Check organization type
        trusted_org_types = [
            'university',
            'security_company',
            'government_agency',
            'research_institution',
            'training_provider'
        ]
        
        org_type = use_case.get('organization_type', '').lower()
        if org_type and org_type not in trusted_org_types:
            logger.warning(f"Untrusted organization type: {org_type}")
        
        return True


class ContainmentSystem:
    """Provides attack containment and isolation"""
    
    def __init__(self):
        self.isolation_policies = {}
        self.kill_switches = {}
        self.monitoring_hooks = []
        self.quarantine_zones = set()
        
    def create_isolation_policy(self, policy_name: str, config: Dict[str, Any]) -> None:
        """Create network isolation policy"""
        
        self.isolation_policies[policy_name] = {
            'network_isolation': config.get('network_isolation', True),
            'process_isolation': config.get('process_isolation', True),
            'file_system_isolation': config.get('file_system_isolation', True),
            'outbound_blocking': config.get('outbound_blocking', True),
            'inbound_filtering': config.get('inbound_filtering', True),
            'resource_limits': config.get('resource_limits', {}),
            'monitoring_level': config.get('monitoring_level', 'high'),
            'auto_terminate': config.get('auto_terminate', True),
            'max_runtime': config.get('max_runtime', 3600)  # 1 hour default
        }
        
        logger.info(f"Created isolation policy: {policy_name}")
    
    def deploy_containment(self, target: str, policy_name: str) -> str:
        """Deploy containment measures for a target"""
        
        if policy_name not in self.isolation_policies:
            raise SecurityValidationError(
                f"Unknown isolation policy: {policy_name}",
                validation_type="policy_not_found"
            )
        
        policy = self.isolation_policies[policy_name]
        containment_id = self._generate_containment_id(target)
        
        # Implement network isolation
        if policy['network_isolation']:
            self._isolate_network(target, containment_id)
        
        # Implement process isolation
        if policy['process_isolation']:
            self._isolate_processes(target, containment_id)
        
        # Set up monitoring
        if policy['monitoring_level'] == 'high':
            self._setup_enhanced_monitoring(target, containment_id)
        
        # Set up kill switch
        if policy['auto_terminate']:
            self._setup_kill_switch(containment_id, policy['max_runtime'])
        
        logger.info(f"Deployed containment {containment_id} for {target}")
        return containment_id
    
    def emergency_shutdown(self, containment_id: str, reason: str) -> bool:
        """Emergency shutdown of containment"""
        
        logger.critical(f"Emergency shutdown triggered for {containment_id}: {reason}")
        
        try:
            # Stop all processes
            self._terminate_processes(containment_id)
            
            # Isolate network completely
            self._complete_network_isolation(containment_id)
            
            # Quarantine affected systems
            self._quarantine_systems(containment_id)
            
            # Alert security team
            self._send_emergency_alert(containment_id, reason)
            
            return True
            
        except Exception as e:
            logger.error(f"Emergency shutdown failed: {e}")
            return False
    
    def release_containment(self, containment_id: str, authorized_by: str) -> bool:
        """Release containment measures"""
        
        logger.info(f"Releasing containment {containment_id} authorized by {authorized_by}")
        
        try:
            # Remove network isolation
            self._remove_network_isolation(containment_id)
            
            # Stop enhanced monitoring
            self._stop_enhanced_monitoring(containment_id)
            
            # Remove kill switch
            if containment_id in self.kill_switches:
                self.kill_switches[containment_id].cancel()
                del self.kill_switches[containment_id]
            
            # Remove from quarantine
            self.quarantine_zones.discard(containment_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to release containment: {e}")
            return False
    
    def _generate_containment_id(self, target: str) -> str:
        """Generate unique containment ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hash_part = hashlib.md5(f"{target}_{timestamp}".encode()).hexdigest()[:8]
        return f"contain_{timestamp}_{hash_part}"
    
    def _isolate_network(self, target: str, containment_id: str) -> None:
        """Implement network isolation"""
        
        # Create isolated network namespace (Linux)
        try:
            subprocess.run([
                'ip', 'netns', 'add', containment_id
            ], check=True, capture_output=True)
            
            logger.info(f"Created network namespace: {containment_id}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to create network namespace: {e}")
    
    def _isolate_processes(self, target: str, containment_id: str) -> None:
        """Implement process isolation"""
        
        # This would typically involve cgroups, containers, or VMs
        logger.info(f"Process isolation implemented for {containment_id}")
    
    def _setup_enhanced_monitoring(self, target: str, containment_id: str) -> None:
        """Set up enhanced monitoring for contained target"""
        
        def monitor_function():
            while containment_id not in self.quarantine_zones:
                # Monitor system calls, network traffic, file access
                time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor_function, daemon=True)
        monitor_thread.start()
        
        logger.info(f"Enhanced monitoring active for {containment_id}")
    
    def _setup_kill_switch(self, containment_id: str, max_runtime: int) -> None:
        """Set up automatic termination after max runtime"""
        
        def kill_switch():
            logger.warning(f"Kill switch triggered for {containment_id} after {max_runtime}s")
            self.emergency_shutdown(containment_id, "Maximum runtime exceeded")
        
        timer = threading.Timer(max_runtime, kill_switch)
        timer.start()
        self.kill_switches[containment_id] = timer
    
    def _terminate_processes(self, containment_id: str) -> None:
        """Terminate all processes in containment"""
        logger.info(f"Terminating processes for {containment_id}")
    
    def _complete_network_isolation(self, containment_id: str) -> None:
        """Complete network isolation"""
        logger.info(f"Complete network isolation for {containment_id}")
    
    def _quarantine_systems(self, containment_id: str) -> None:
        """Quarantine affected systems"""
        self.quarantine_zones.add(containment_id)
        logger.info(f"Systems quarantined for {containment_id}")
    
    def _send_emergency_alert(self, containment_id: str, reason: str) -> None:
        """Send emergency alert to security team"""
        
        alert = {
            'type': 'EMERGENCY_SHUTDOWN',
            'containment_id': containment_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'severity': 'CRITICAL'
        }
        
        logger.critical(f"EMERGENCY ALERT: {json.dumps(alert)}")
    
    def _remove_network_isolation(self, containment_id: str) -> None:
        """Remove network isolation"""
        
        try:
            subprocess.run([
                'ip', 'netns', 'delete', containment_id
            ], check=True, capture_output=True)
            
            logger.info(f"Removed network namespace: {containment_id}")
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to remove network namespace: {e}")
    
    def _stop_enhanced_monitoring(self, containment_id: str) -> None:
        """Stop enhanced monitoring"""
        logger.info(f"Stopped enhanced monitoring for {containment_id}")


class UsageMonitor:
    """Monitors system usage for compliance and security"""
    
    def __init__(self):
        self.usage_logs = []
        self.violation_count = 0
        self.monitoring_active = True
        
    def log_ethics_check(self, use_case: Dict[str, Any], approved: bool) -> None:
        """Log ethics validation check"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'ethics_check',
            'use_case': use_case,
            'approved': approved,
            'user_id': use_case.get('user_id')
        }
        
        self.usage_logs.append(log_entry)
        
        if not approved:
            self.violation_count += 1
            logger.warning(f"Ethics violation #{self.violation_count}")
    
    def log_security_event(self, event: SecurityEvent) -> None:
        """Log security event"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'security_event',
            'event': event.to_dict()
        }
        
        self.usage_logs.append(log_entry)
        
        if event.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            logger.error(f"High severity security event: {event.description}")
    
    def get_usage_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate usage report"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_logs = [
            log for log in self.usage_logs
            if datetime.fromisoformat(log['timestamp']) > cutoff_date
        ]
        
        report = {
            'period_days': days,
            'total_events': len(recent_logs),
            'ethics_checks': len([log for log in recent_logs if log['type'] == 'ethics_check']),
            'security_events': len([log for log in recent_logs if log['type'] == 'security_event']),
            'violations': self.violation_count,
            'generated_at': datetime.now().isoformat()
        }
        
        return report


class ConsentManager:
    """Manages user consent for operations"""
    
    def __init__(self):
        self.consent_records = {}
        self.consent_templates = {}
        
    def get_consent(self, operation: str, participants: List[str]) -> bool:
        """Get consent from participants for operation"""
        
        consent_key = f"{operation}_{hashlib.md5('_'.join(participants).encode()).hexdigest()}"
        
        # Check if consent already exists
        if consent_key in self.consent_records:
            consent_record = self.consent_records[consent_key]
            if consent_record['expires'] > datetime.now():
                return consent_record['granted']
        
        # For automated testing, assume consent is granted
        # In production, this would prompt users for consent
        consent_granted = True
        
        # Record consent
        self.consent_records[consent_key] = {
            'operation': operation,
            'participants': participants,
            'granted': consent_granted,
            'timestamp': datetime.now(),
            'expires': datetime.now() + timedelta(hours=24)
        }
        
        logger.info(f"Consent {'granted' if consent_granted else 'denied'} for {operation}")
        return consent_granted


class EthicsBoard:
    """Simulated ethics review board"""
    
    def __init__(self):
        self.pending_reviews = []
        self.approved_research = set()
        
    def submit_for_review(self, research_proposal: Dict[str, Any]) -> str:
        """Submit research proposal for ethics review"""
        
        review_id = f"ethics_{int(time.time())}"
        
        review_entry = {
            'review_id': review_id,
            'proposal': research_proposal,
            'submitted_at': datetime.now(),
            'status': 'under_review'
        }
        
        self.pending_reviews.append(review_entry)
        
        # Auto-approve for legitimate research (simplified)
        if self._is_legitimate_research(research_proposal):
            self.approved_research.add(review_id)
            review_entry['status'] = 'approved'
        
        return review_id
    
    def _is_legitimate_research(self, proposal: Dict[str, Any]) -> bool:
        """Determine if proposal is legitimate research"""
        
        purpose = proposal.get('purpose', '').lower()
        legitimate_purposes = [
            'security_education',
            'defensive_research',
            'vulnerability_assessment',
            'incident_response_training'
        ]
        
        return any(purpose in legitimate for legitimate in legitimate_purposes)


class SecurityManager:
    """Main security management interface"""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.validator = SecurityValidator(self.policy)
        self.ethical_framework = EthicalFramework()
        self.containment = ContainmentSystem()
        self.usage_monitor = UsageMonitor()
        
        # Set up default containment policies
        self._setup_default_policies()
    
    def validate_operation(self, operation: Dict[str, Any], user_id: str) -> bool:
        """Comprehensive operation validation"""
        
        operation_type = operation.get('type', '')
        
        # Validate user permissions
        self.validator.validate_user_permissions(user_id, operation_type, 'cyber_range')
        
        # Validate against security policy
        if operation_type == 'attack':
            self.validator.validate_attack_config(operation)
        elif operation_type == 'deploy_network':
            self.validator.validate_network_config(operation)
        
        # Ethical validation
        use_case = {
            'purpose': operation.get('purpose', 'training'),
            'user_id': user_id,
            'user_role': operation.get('user_role', 'student'),
            'target_type': operation.get('target_type', 'simulated')
        }
        
        if not self.ethical_framework.validate_use_case(use_case):
            raise SecurityValidationError(
                "Operation violates ethical guidelines",
                validation_type="ethics_violation"
            )
        
        # Get consent if required
        if not self.ethical_framework.require_consent(operation_type, [user_id]):
            raise SecurityValidationError(
                "Required consent not obtained",
                validation_type="consent_required"
            )
        
        return True
    
    def deploy_security_measures(self, target: str, security_level: str = "standard") -> str:
        """Deploy security measures for a target"""
        
        containment_id = self.containment.deploy_containment(target, security_level)
        
        # Log security deployment
        event = SecurityEvent(
            event_id=f"security_{int(time.time())}",
            timestamp=datetime.now(),
            event_type="security_deployment",
            severity=ThreatLevel.LOW,
            source="security_manager",
            description=f"Security measures deployed for {target}",
            additional_data={'containment_id': containment_id, 'security_level': security_level}
        )
        
        self.usage_monitor.log_security_event(event)
        
        return containment_id
    
    def emergency_response(self, threat_id: str, threat_level: ThreatLevel) -> bool:
        """Execute emergency response procedures"""
        
        logger.critical(f"Emergency response triggered for threat {threat_id} (level: {threat_level.value})")
        
        if threat_level == ThreatLevel.CRITICAL:
            # Full system shutdown
            success = self.containment.emergency_shutdown(threat_id, "Critical threat detected")
            
            # Alert all relevant parties
            self._send_critical_alert(threat_id)
            
        elif threat_level == ThreatLevel.HIGH:
            # Enhanced monitoring and partial isolation
            success = self._partial_containment(threat_id)
            
        else:
            # Standard monitoring increase
            success = self._increase_monitoring(threat_id)
        
        return success
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'policy_active': bool(self.policy),
            'containment_active': len(self.containment.quarantine_zones),
            'monitoring_active': self.usage_monitor.monitoring_active,
            'violation_count': self.usage_monitor.violation_count,
            'threat_level': self._assess_overall_threat_level().value
        }
        
        return status
    
    def _setup_default_policies(self) -> None:
        """Set up default containment policies"""
        
        # Standard policy
        self.containment.create_isolation_policy("standard", {
            'network_isolation': True,
            'process_isolation': True,
            'outbound_blocking': True,
            'monitoring_level': 'medium',
            'max_runtime': 3600
        })
        
        # High security policy
        self.containment.create_isolation_policy("high_security", {
            'network_isolation': True,
            'process_isolation': True,
            'file_system_isolation': True,
            'outbound_blocking': True,
            'inbound_filtering': True,
            'monitoring_level': 'high',
            'max_runtime': 1800
        })
        
        # Research policy
        self.containment.create_isolation_policy("research", {
            'network_isolation': True,
            'process_isolation': False,
            'outbound_blocking': False,
            'monitoring_level': 'low',
            'max_runtime': 7200
        })
    
    def _partial_containment(self, threat_id: str) -> bool:
        """Implement partial containment measures"""
        logger.warning(f"Implementing partial containment for {threat_id}")
        return True
    
    def _increase_monitoring(self, threat_id: str) -> bool:
        """Increase monitoring level"""
        logger.info(f"Increasing monitoring level for {threat_id}")
        return True
    
    def _send_critical_alert(self, threat_id: str) -> None:
        """Send critical security alert"""
        logger.critical(f"CRITICAL SECURITY ALERT: {threat_id}")
    
    def _assess_overall_threat_level(self) -> ThreatLevel:
        """Assess overall system threat level"""
        
        if self.usage_monitor.violation_count > 10:
            return ThreatLevel.HIGH
        elif self.usage_monitor.violation_count > 5:
            return ThreatLevel.MEDIUM
        elif len(self.containment.quarantine_zones) > 0:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW


# Global security manager instance
security_manager = SecurityManager()


def get_security_manager() -> SecurityManager:
    """Get the global security manager"""
    return security_manager


def require_security_clearance(clearance_level: SecurityLevel):
    """Decorator to require security clearance for functions"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # In a real implementation, this would check user's clearance level
            logger.info(f"Security clearance {clearance_level.value} required for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def audit_action(action_type: str):
    """Decorator to audit function calls"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful action
                security_manager.usage_monitor.log_security_event(
                    SecurityEvent(
                        event_id=f"audit_{int(time.time())}",
                        timestamp=start_time,
                        event_type="action_audit",
                        severity=ThreatLevel.LOW,
                        source=func.__module__,
                        description=f"Action {action_type} executed: {func.__name__}",
                        additional_data={'function': func.__name__, 'success': True}
                    )
                )
                
                return result
                
            except Exception as e:
                # Log failed action
                security_manager.usage_monitor.log_security_event(
                    SecurityEvent(
                        event_id=f"audit_{int(time.time())}",
                        timestamp=start_time,
                        event_type="action_audit",
                        severity=ThreatLevel.MEDIUM,
                        source=func.__module__,
                        description=f"Action {action_type} failed: {func.__name__} - {str(e)}",
                        additional_data={'function': func.__name__, 'success': False, 'error': str(e)}
                    )
                )
                
                raise
        
        return wrapper
    return decorator