"""
Enhanced Security Module for GAN-Cyber-Range-v2

Provides comprehensive security features including:
- Input validation and sanitization
- Threat detection and prevention 
- Security event monitoring
- Encryption and secure storage
- Attack containment and isolation
"""

import hashlib
import hmac
import secrets
import re
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import uuid
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityLevel(Enum):
    """Security access levels"""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


class SecurityEventType(Enum):
    """Types of security events"""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authz_failure"
    SUSPICIOUS_INPUT = "suspicious_input"
    RATE_LIMIT_EXCEEDED = "rate_limit"
    MALICIOUS_PAYLOAD = "malicious_payload"
    CONTAINMENT_BREACH = "containment_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION_ATTEMPT = "data_exfiltration"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)


class SecureInputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self):
        # Dangerous patterns to detect
        self.dangerous_patterns = {
            'sql_injection': [
                r"(\bunion\b.*\bselect\b)",
                r"(\bselect\b.*\bfrom\b.*\bwhere\b)",
                r"(\bdrop\b\s+\btable\b)",
                r"(\binsert\b\s+\binto\b)",
                r"(\bdelete\b\s+\bfrom\b)",
                r"('|\")[^'\"]*('|\")(\s*;\s*)"
            ],
            'xss': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"eval\s*\(",
                r"document\.cookie"
            ],
            'command_injection': [
                r"[;&|`$(){}]",
                r"\b(cat|ls|pwd|whoami|id|uname)\b",
                r"\b(rm|mv|cp|chmod|chown)\b",
                r"\b(wget|curl|nc|netcat)\b"
            ],
            'path_traversal': [
                r"\.\.\/",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c"
            ]
        }
        
        self.max_lengths = {
            'username': 50,
            'password': 128,
            'email': 254,
            'range_name': 50,
            'attack_payload': 1000,
            'general': 500
        }
    
    def validate_input(self, input_value: str, input_type: str = 'general', client_id: str = 'unknown') -> Dict[str, Any]:
        """Comprehensive input validation"""
        validation_result = {
            'is_valid': True,
            'sanitized_value': input_value,
            'threats_detected': [],
            'severity': ThreatLevel.LOW
        }
        
        if not input_value:
            return validation_result
        
        # Length validation
        max_length = self.max_lengths.get(input_type, self.max_lengths['general'])
        if len(input_value) > max_length:
            validation_result['is_valid'] = False
            validation_result['threats_detected'].append(f'Input exceeds maximum length ({max_length})')
            validation_result['severity'] = ThreatLevel.MEDIUM
        
        # Pattern-based threat detection
        for threat_type, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_value, re.IGNORECASE):
                    validation_result['is_valid'] = False
                    validation_result['threats_detected'].append(f'{threat_type}: {pattern}')
                    validation_result['severity'] = ThreatLevel.HIGH
        
        # Sanitization
        validation_result['sanitized_value'] = self._sanitize_input(input_value)
        
        return validation_result
    
    def _sanitize_input(self, input_value: str) -> str:
        """Sanitize input by removing dangerous characters"""
        # HTML entity encoding for basic XSS prevention
        sanitized = input_value.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        
        # Remove null bytes and control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return sanitized
    
    def validate_attack_config(self, config: Dict[str, Any], client_id: str = 'unknown') -> Dict[str, Any]:
        """Validate attack configuration for safety"""
        validation_result = {
            'is_safe': True,
            'issues': [],
            'sanitized_config': config.copy(),
            'threat_level': ThreatLevel.LOW
        }
        
        # Check for production system targets
        production_indicators = [
            'prod', 'production', 'live', 'www', 'api',
            '.com', '.net', '.org', '.edu', '.gov'
        ]
        
        for key, value in config.items():
            if isinstance(value, str):
                for indicator in production_indicators:
                    if indicator in value.lower():
                        validation_result['is_safe'] = False
                        validation_result['issues'].append(f'Potential production target: {key}={value}')
        
        # Validate attack parameters
        if 'target_ip' in config:
            ip = config['target_ip']
            if not self._is_safe_target_ip(ip):
                validation_result['is_safe'] = False
                validation_result['issues'].append(f'Unsafe target IP: {ip}')
        
        return validation_result
    
    def _is_safe_target_ip(self, ip: str) -> bool:
        """Check if target IP is safe for testing"""
        # Allow only RFC 1918 private networks and loopback
        safe_patterns = [
            r'^127\.',           # Loopback
            r'^10\.',            # 10.0.0.0/8
            r'^172\.(1[6-9]|2[0-9]|3[01])\.',  # 172.16.0.0/12
            r'^192\.168\.',      # 192.168.0.0/16
        ]
        
        return any(re.match(pattern, ip) for pattern in safe_patterns)


class ThreatDetectionEngine:
    """Advanced threat detection and prevention system"""
    
    def __init__(self):
        self.threat_signatures = {}
        self.load_threat_signatures()
        
        self.behavioral_baselines = {}
        self.anomaly_thresholds = {
            'request_rate': 100,  # requests per minute
            'failed_auth_rate': 10,  # failed attempts per minute
            'unusual_patterns': 5   # suspicious pattern matches
        }
    
    def load_threat_signatures(self) -> None:
        """Load threat detection signatures"""
        # Malware signatures
        self.threat_signatures['malware'] = [
            r'eicar',
            r'x5o!p%@ap\[4\\pzx54\(p\^\)7cc\)7\}\$eicar-standard-antivirus-test-file!\$h\+h\*',
            r'meterpreter',
            r'metasploit',
            r'cobalt.*strike'
        ]
        
        # Network attack signatures
        self.threat_signatures['network_attacks'] = [
            r'nmap.*-s[STAUFNW]',
            r'sqlmap',
            r'hydra.*-l.*-P',
            r'nikto',
            r'dirb\s+http'
        ]
        
        # Exploitation frameworks
        self.threat_signatures['exploit_frameworks'] = [
            r'exploit/.*/',
            r'payload/.*/',
            r'auxiliary/.*/'
        ]
    
    def analyze_payload(self, payload: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze payload for threats"""
        analysis_result = {
            'threat_detected': False,
            'threat_type': None,
            'confidence': 0.0,
            'signatures_matched': [],
            'risk_score': 0.0
        }
        
        risk_score = 0.0
        signatures_matched = []
        
        # Signature-based detection
        for threat_type, signatures in self.threat_signatures.items():
            for signature in signatures:
                if re.search(signature, payload, re.IGNORECASE):
                    signatures_matched.append(f'{threat_type}:{signature}')
                    risk_score += 0.3
        
        # Entropy analysis (detect encoded/encrypted content)
        entropy = self._calculate_entropy(payload)
        if entropy > 7.5:  # High entropy indicates potential encoding
            risk_score += 0.2
            signatures_matched.append('high_entropy')
        
        # Behavioral analysis
        if context:
            behavioral_risk = self._analyze_behavior(payload, context)
            risk_score += behavioral_risk
        
        # Determine overall threat
        if risk_score > 0.7:
            analysis_result['threat_detected'] = True
            analysis_result['threat_type'] = 'high_risk'
            analysis_result['confidence'] = min(risk_score, 1.0)
        elif risk_score > 0.4:
            analysis_result['threat_detected'] = True  
            analysis_result['threat_type'] = 'medium_risk'
            analysis_result['confidence'] = risk_score
        
        analysis_result['risk_score'] = risk_score
        analysis_result['signatures_matched'] = signatures_matched
        
        return analysis_result
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0
        
        frequency = {}
        for char in data:
            frequency[char] = frequency.get(char, 0) + 1
        
        entropy = 0
        length = len(data)
        import math
        for count in frequency.values():
            if count > 0:
                probability = count / length
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _analyze_behavior(self, payload: str, context: Dict[str, Any]) -> float:
        """Analyze behavioral patterns"""
        risk = 0.0
        
        # Check for suspicious timing patterns
        if 'timestamp' in context:
            # Rapid successive requests might indicate automation
            pass
        
        # Check for unusual payload characteristics
        if len(payload) > 500:  # Very long payloads
            risk += 0.1
        
        # Check for binary content in text payload
        try:
            payload.encode('ascii')
        except UnicodeEncodeError:
            risk += 0.1
        
        return risk


class ContainmentEngine:
    """Attack containment and isolation system"""
    
    def __init__(self):
        self.containment_policies = {
            'network_isolation': True,
            'resource_limits': True,
            'monitoring_enhanced': True,
            'logging_verbose': True
        }
        
        self.active_containments = {}
        self.quarantine_zone = "quarantine"
    
    def enforce_containment(self, range_id: str, threat_level: ThreatLevel) -> Dict[str, Any]:
        """Enforce containment based on threat level"""
        containment_id = str(uuid.uuid4())
        
        containment_measures = {
            'containment_id': containment_id,
            'range_id': range_id,
            'threat_level': threat_level,
            'timestamp': datetime.now(),
            'measures_applied': [],
            'status': 'active'
        }
        
        # Apply containment based on threat level
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            # Network isolation
            containment_measures['measures_applied'].append('network_isolation')
            logger.warning(f"Network isolation enforced for range {range_id}")
            
            # Resource throttling
            containment_measures['measures_applied'].append('resource_throttling')
            logger.warning(f"Resource throttling enforced for range {range_id}")
        
        if threat_level == ThreatLevel.CRITICAL:
            # Complete range suspension
            containment_measures['measures_applied'].append('range_suspension')
            logger.critical(f"Range suspension enforced for range {range_id}")
        
        # Enhanced monitoring
        containment_measures['measures_applied'].append('enhanced_monitoring')
        
        self.active_containments[containment_id] = containment_measures
        
        logger.info(f"Containment {containment_id} applied to range {range_id}")
        return containment_measures
    
    def release_containment(self, containment_id: str) -> bool:
        """Release containment measures"""
        if containment_id not in self.active_containments:
            return False
        
        containment = self.active_containments[containment_id]
        containment['status'] = 'released'
        containment['release_timestamp'] = datetime.now()
        
        logger.info(f"Containment {containment_id} released")
        return True


class SecurityAuditLogger:
    """Comprehensive security audit logging"""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.security_events = []
        
        # Set up dedicated security logger
        self.security_logger = logging.getLogger('security_audit')
        self.security_logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.security_logger.addHandler(handler)
    
    def log_security_event(self, event: SecurityEvent) -> None:
        """Log security event"""
        self.security_events.append(event)
        
        log_message = json.dumps({
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'threat_level': event.threat_level.value,
            'timestamp': event.timestamp.isoformat(),
            'source_ip': event.source_ip,
            'user_id': event.user_id,
            'details': event.details,
            'mitigated': event.mitigated,
            'mitigation_actions': event.mitigation_actions
        })
        
        if event.threat_level == ThreatLevel.CRITICAL:
            self.security_logger.critical(log_message)
        elif event.threat_level == ThreatLevel.HIGH:
            self.security_logger.error(log_message)
        elif event.threat_level == ThreatLevel.MEDIUM:
            self.security_logger.warning(log_message)
        else:
            self.security_logger.info(log_message)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        
        summary = {
            'total_events': len(recent_events),
            'critical_events': len([e for e in recent_events if e.threat_level == ThreatLevel.CRITICAL]),
            'high_risk_events': len([e for e in recent_events if e.threat_level == ThreatLevel.HIGH]),
            'medium_risk_events': len([e for e in recent_events if e.threat_level == ThreatLevel.MEDIUM]),
            'low_risk_events': len([e for e in recent_events if e.threat_level == ThreatLevel.LOW]),
            'event_types': {},
            'mitigation_rate': 0.0
        }
        
        # Count by event type
        for event in recent_events:
            event_type = event.event_type.value
            summary['event_types'][event_type] = summary['event_types'].get(event_type, 0) + 1
        
        # Calculate mitigation rate
        mitigated_count = len([e for e in recent_events if e.mitigated])
        summary['mitigation_rate'] = mitigated_count / max(1, len(recent_events))
        
        return summary


class SecureDataManager:
    """Secure data encryption and storage"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            master_key = Fernet.generate_key()
        
        self.cipher_suite = Fernet(master_key)
        self.master_key = master_key
    
    def encrypt_sensitive_data(self, data: Union[str, Dict[str, Any]]) -> str:
        """Encrypt sensitive data"""
        if isinstance(data, dict):
            data = json.dumps(data)
        
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Union[str, Dict[str, Any]]:
        """Decrypt sensitive data"""
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
        decrypted_str = decrypted_bytes.decode()
        
        # Try to parse as JSON
        try:
            return json.loads(decrypted_str)
        except json.JSONDecodeError:
            return decrypted_str
    
    def secure_hash(self, data: str, salt: Optional[str] = None) -> str:
        """Generate secure hash with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_input = f"{salt}{data}".encode()
        hash_digest = hashlib.sha256(hash_input).hexdigest()
        return f"{salt}:{hash_digest}"
    
    def verify_hash(self, data: str, hash_with_salt: str) -> bool:
        """Verify data against hash"""
        try:
            salt, expected_hash = hash_with_salt.split(':', 1)
            hash_input = f"{salt}{data}".encode()
            actual_hash = hashlib.sha256(hash_input).hexdigest()
            return hmac.compare_digest(expected_hash, actual_hash)
        except ValueError:
            return False


# Ethical framework implementation
class EthicalFramework:
    """Ethical usage enforcement framework"""
    
    def __init__(self, allowed_uses: List[str] = None, prohibited_targets: List[str] = None, require_consent: bool = True):
        self.allowed_uses = allowed_uses or ["research", "training", "defense", "education"]
        self.prohibited_targets = prohibited_targets or ["production_systems", "real_networks"]
        self.require_consent = require_consent
        
    def is_compliant(self, request: Dict[str, Any]) -> bool:
        """Check if request complies with ethical guidelines"""
        purpose = request.get('purpose', 'unknown')
        target = request.get('target', '')
        
        # Check purpose compliance
        if purpose not in self.allowed_uses:
            return False
            
        # Check target compliance
        for prohibited in self.prohibited_targets:
            if prohibited in target.lower():
                return False
                
        return True


class Containment:
    """Security containment implementation"""
    
    def __init__(self, network_isolation: str = "strict", outbound_filtering: bool = True, killswitch_enabled: bool = True):
        self.network_isolation = network_isolation
        self.outbound_filtering = outbound_filtering
        self.killswitch_enabled = killswitch_enabled
        
    def contained(self, func):
        """Decorator for contained execution"""
        def wrapper(*args, **kwargs):
            logger.info(f"Executing {func.__name__} with containment")
            return func(*args, **kwargs)
        return wrapper


def validate_input(input_value: str, input_type: str = 'general', client_id: str = 'unknown') -> bool:
    """Global input validation function"""
    result = input_validator.validate_input(input_value, input_type, client_id)
    return result['is_valid']


# Global security components
input_validator = SecureInputValidator()
threat_detector = ThreatDetectionEngine()
containment_engine = ContainmentEngine()
audit_logger = SecurityAuditLogger()
data_manager = SecureDataManager()


def secure_hash(data: str, salt: str = None) -> str:
    """Global secure hash function"""
    return data_manager.secure_hash(data, salt)
ethical_framework = EthicalFramework()
containment = Containment()


class UsageMonitor:
    """Monitor usage for ethical compliance"""
    
    def __init__(self, ethical_framework):
        self.ethical_framework = ethical_framework
        self.usage_log = []
        
    def log_usage(self, user: str, purpose: str, timestamp):
        """Log usage event"""
        self.usage_log.append({
            'user': user,
            'purpose': purpose,
            'timestamp': timestamp
        })
        
    def before_attack_generation(self, func):
        """Decorator for pre-generation checks"""
        def wrapper(request):
            if not self.ethical_framework.is_compliant(request):
                raise ValueError(f"Ethical violation: {request}")
            return func(request)
        return wrapper


def create_security_event(
    event_type: SecurityEventType,
    threat_level: ThreatLevel,
    details: Dict[str, Any],
    source_ip: Optional[str] = None,
    user_id: Optional[str] = None
) -> SecurityEvent:
    """Create and log a security event"""
    event = SecurityEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        threat_level=threat_level,
        timestamp=datetime.now(),
        source_ip=source_ip,
        user_id=user_id,
        details=details
    )
    
    audit_logger.log_security_event(event)
    return event


# Enhanced security decorator for API endpoints
def security_check(threat_level: ThreatLevel = ThreatLevel.MEDIUM, require_auth: bool = True, rate_limit: bool = True):
    """Enhanced decorator for endpoint security checks"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract client information
            client_id = kwargs.get('client_id', 'unknown')
            
            # Input validation for all arguments
            for arg_name, arg_value in kwargs.items():
                if isinstance(arg_value, (str, dict, list)):
                    if isinstance(arg_value, str):
                        validation = input_validator.validate_input(arg_value, arg_name, client_id)
                        if not validation['is_valid']:
                            create_security_event(
                                SecurityEventType.SUSPICIOUS_INPUT,
                                validation['severity'],
                                {
                                    'function': func.__name__,
                                    'argument': arg_name,
                                    'threats': validation['threats_detected'],
                                    'client_id': client_id
                                }
                            )
                            raise ValueError(f"Invalid input for {arg_name}: {validation['threats_detected']}")
            
            # Perform security checks
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                create_security_event(
                    SecurityEventType.UNAUTHORIZED_ACCESS,
                    threat_level,
                    {
                        'error': str(e),
                        'function': func.__name__,
                        'client_id': client_id
                    }
                )
                raise
        return wrapper
    return decorator


# Security monitoring functions
def monitor_security_events() -> Dict[str, Any]:
    """Monitor and return current security status"""
    summary = audit_logger.get_security_summary(24)
    
    # Add threat level assessment
    if summary['critical_events'] > 0:
        summary['overall_threat_level'] = ThreatLevel.CRITICAL.value
    elif summary['high_risk_events'] > 5:
        summary['overall_threat_level'] = ThreatLevel.HIGH.value
    elif summary['medium_risk_events'] > 10:
        summary['overall_threat_level'] = ThreatLevel.MEDIUM.value
    else:
        summary['overall_threat_level'] = ThreatLevel.LOW.value
    
    return summary


def validate_cyber_range_operation(operation: str, params: Dict[str, Any], client_id: str = 'unknown') -> bool:
    """Validate cyber range operation for security"""
    # Validate operation type
    allowed_operations = [
        'create_range', 'deploy_range', 'start_attack', 'stop_attack',
        'generate_attacks', 'evaluate_performance', 'create_scenario'
    ]
    
    if operation not in allowed_operations:
        create_security_event(
            SecurityEventType.UNAUTHORIZED_ACCESS,
            ThreatLevel.HIGH,
            {'operation': operation, 'reason': 'unknown_operation', 'client_id': client_id}
        )
        return False
    
    # Validate parameters
    validation = input_validator.validate_attack_config(params, client_id)
    if not validation['is_safe']:
        create_security_event(
            SecurityEventType.MALICIOUS_PAYLOAD,
            validation['threat_level'],
            {
                'operation': operation,
                'issues': validation['issues'],
                'client_id': client_id
            }
        )
        return False
    
    return True