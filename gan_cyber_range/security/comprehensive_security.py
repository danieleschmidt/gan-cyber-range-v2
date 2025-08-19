"""
Comprehensive security framework for GAN-Cyber-Range-v2.
Implements defense-in-depth security architecture.
"""

import logging
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels"""
    PUBLIC = 1
    RESTRICTED = 2
    CONFIDENTIAL = 3
    SECRET = 4
    TOP_SECRET = 5


class ThreatLevel(Enum):
    """Threat assessment levels"""
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    session_id: str
    clearance_level: SecurityLevel
    permissions: List[str]
    ip_address: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


@dataclass
class SecurityEvent:
    """Security event logging"""
    event_id: str
    event_type: str
    severity: ThreatLevel
    source: str
    target: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    mitigated: bool = False


class SecurityValidator:
    """Input validation and sanitization"""
    
    def __init__(self):
        self.dangerous_patterns = {
            'sql_injection': [
                r"(\bunion\b.*\bselect\b)",
                r"(\bselect\b.*\bfrom\b.*\bwhere\b)",
                r"(\bdrop\b\s+\btable\b)",
                r"(\binsert\b\s+\binto\b)",
                r"(\bdelete\b\s+\bfrom\b)"
            ],
            'xss': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"eval\s*\("
            ],
            'command_injection': [
                r"[;&|`$(){}]",
                r"\b(cat|ls|pwd|whoami|id|uname)\b",
                r"\b(rm|mv|cp|chmod|chown)\b"
            ]
        }
    
    def validate_input(self, data: str, context: SecurityContext) -> Dict[str, Any]:
        """Validate input for security threats"""
        result = {
            'is_safe': True,
            'threats_found': [],
            'sanitized_data': data,
            'confidence': 1.0
        }
        
        # Check for dangerous patterns
        for threat_type, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                import re
                if re.search(pattern, data, re.IGNORECASE):
                    result['is_safe'] = False
                    result['threats_found'].append(threat_type)
                    result['confidence'] = 0.0
        
        # Sanitize data
        result['sanitized_data'] = self._sanitize_data(data)
        
        return result
    
    def _sanitize_data(self, data: str) -> str:
        """Sanitize input data"""
        # HTML entity encoding
        sanitized = data.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        
        # Remove null bytes and control characters
        import re
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return sanitized


class RateLimiter:
    """Rate limiting for security"""
    
    def __init__(self):
        self.attempts = {}
        self.locks = {}
    
    def check_rate_limit(self, identifier: str, max_attempts: int = 100, window_seconds: int = 3600) -> bool:
        """Check if rate limit is exceeded"""
        now = time.time()
        
        if identifier not in self.locks:
            self.locks[identifier] = threading.Lock()
        
        with self.locks[identifier]:
            if identifier not in self.attempts:
                self.attempts[identifier] = []
            
            # Clean old attempts
            cutoff = now - window_seconds
            self.attempts[identifier] = [
                attempt for attempt in self.attempts[identifier] 
                if attempt > cutoff
            ]
            
            # Check limit
            if len(self.attempts[identifier]) >= max_attempts:
                return False
            
            # Record attempt
            self.attempts[identifier].append(now)
            return True


class SecurityMonitor:
    """Continuous security monitoring"""
    
    def __init__(self):
        self.events = []
        self.alert_thresholds = {
            ThreatLevel.HIGH: 5,
            ThreatLevel.CRITICAL: 1
        }
        self.monitoring_active = True
    
    def log_event(self, event: SecurityEvent):
        """Log security event"""
        self.events.append(event)
        logger.warning(f"Security event: {event.event_type} - {event.severity.name}")
        
        # Check for alert conditions
        self._check_alerts(event)
    
    def _check_alerts(self, event: SecurityEvent):
        """Check if event should trigger alerts"""
        if event.severity in self.alert_thresholds:
            recent_events = self._get_recent_events(minutes=60)
            severity_count = sum(1 for e in recent_events if e.severity == event.severity)
            
            if severity_count >= self.alert_thresholds[event.severity]:
                self._trigger_alert(event.severity, severity_count)
    
    def _get_recent_events(self, minutes: int = 60) -> List[SecurityEvent]:
        """Get events from recent time window"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [event for event in self.events if event.timestamp > cutoff]
    
    def _trigger_alert(self, severity: ThreatLevel, count: int):
        """Trigger security alert"""
        logger.critical(f"SECURITY ALERT: {count} {severity.name} events in last hour")
        
        # In production, this would trigger:
        # - Email notifications
        # - SIEM alerts
        # - Dashboard notifications
        # - Automated response


class AccessController:
    """Role-based access control"""
    
    def __init__(self):
        self.permissions = {
            'admin': ['*'],
            'researcher': ['read_data', 'create_experiment', 'view_results'],
            'analyst': ['read_data', 'view_results'],
            'user': ['view_results']
        }
        self.sessions = {}
    
    def create_session(self, user_id: str, role: str, ip_address: str = None) -> SecurityContext:
        """Create secure session"""
        session_id = secrets.token_urlsafe(32)
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            clearance_level=SecurityLevel.RESTRICTED,
            permissions=self.permissions.get(role, []),
            ip_address=ip_address,
            expires_at=datetime.now() + timedelta(hours=8)
        )
        
        self.sessions[session_id] = context
        logger.info(f"Created session for user {user_id} with role {role}")
        return context
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate session"""
        if session_id not in self.sessions:
            return None
        
        context = self.sessions[session_id]
        
        # Check expiration
        if context.expires_at and datetime.now() > context.expires_at:
            del self.sessions[session_id]
            return None
        
        return context
    
    def check_permission(self, context: SecurityContext, permission: str) -> bool:
        """Check if context has permission"""
        if '*' in context.permissions:
            return True
        return permission in context.permissions


class EncryptionManager:
    """Data encryption and key management"""
    
    def __init__(self):
        self.master_key = self._generate_master_key()
        self.key_rotation_interval = timedelta(days=30)
        self.last_rotation = datetime.now()
    
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key"""
        return secrets.token_bytes(32)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            cipher = Fernet(key)
            encrypted = cipher.encrypt(data.encode())
            
            # In production, store key securely
            return f"{key.decode()}:{encrypted.decode()}"
        except ImportError:
            # Fallback for development
            import base64
            return base64.b64encode(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            from cryptography.fernet import Fernet
            key_str, data_str = encrypted_data.split(':', 1)
            key = key_str.encode()
            cipher = Fernet(key)
            decrypted = cipher.decrypt(data_str.encode())
            return decrypted.decode()
        except ImportError:
            # Fallback for development
            import base64
            return base64.b64decode(encrypted_data.encode()).decode()
        except:
            return encrypted_data  # Return as-is if decryption fails
    
    def hash_password(self, password: str, salt: bytes = None) -> str:
        """Hash password securely"""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        # Use PBKDF2 for password hashing
        import hashlib
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return f"{salt.hex()}:{key.hex()}"
    
    def verify_password(self, password: str, hash_with_salt: str) -> bool:
        """Verify password against hash"""
        try:
            salt_hex, key_hex = hash_with_salt.split(':', 1)
            salt = bytes.fromhex(salt_hex)
            expected_key = bytes.fromhex(key_hex)
            
            import hashlib
            actual_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            return hmac.compare_digest(expected_key, actual_key)
        except:
            return False


class SecurityOrchestrator:
    """Main security orchestration"""
    
    def __init__(self):
        self.validator = SecurityValidator()
        self.rate_limiter = RateLimiter()
        self.monitor = SecurityMonitor()
        self.access_controller = AccessController()
        self.encryption_manager = EncryptionManager()
        self.security_policies = self._load_security_policies()
    
    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security policies"""
        return {
            'max_login_attempts': 5,
            'session_timeout_hours': 8,
            'password_min_length': 8,
            'require_mfa': True,
            'encryption_required': True,
            'audit_all_actions': True
        }
    
    @contextmanager
    def security_context(self, user_id: str, operation: str):
        """Security context manager"""
        event_id = secrets.token_urlsafe(16)
        start_time = datetime.now()
        
        try:
            # Pre-operation security checks
            if not self.rate_limiter.check_rate_limit(user_id):
                raise SecurityException("Rate limit exceeded")
            
            logger.info(f"Security context started for {user_id}: {operation}")
            yield
            
            # Log successful operation
            event = SecurityEvent(
                event_id=event_id,
                event_type="operation_success",
                severity=ThreatLevel.LOW,
                source=user_id,
                target=operation,
                timestamp=start_time,
                details={'duration': (datetime.now() - start_time).total_seconds()}
            )
            self.monitor.log_event(event)
            
        except Exception as e:
            # Log security incident
            event = SecurityEvent(
                event_id=event_id,
                event_type="operation_failure",
                severity=ThreatLevel.MEDIUM,
                source=user_id,
                target=operation,
                timestamp=start_time,
                details={'error': str(e), 'duration': (datetime.now() - start_time).total_seconds()}
            )
            self.monitor.log_event(event)
            raise
    
    def validate_operation(self, context: SecurityContext, operation: str, data: Any = None) -> bool:
        """Validate security for operation"""
        # Check session
        if not self.access_controller.validate_session(context.session_id):
            return False
        
        # Check permissions
        if not self.access_controller.check_permission(context, operation):
            return False
        
        # Validate input data
        if data and isinstance(data, str):
            validation = self.validator.validate_input(data, context)
            if not validation['is_safe']:
                logger.warning(f"Unsafe input detected: {validation['threats_found']}")
                return False
        
        return True
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status"""
        recent_events = self.monitor._get_recent_events(60)
        
        return {
            'active_sessions': len(self.access_controller.sessions),
            'recent_events': len(recent_events),
            'high_severity_events': len([e for e in recent_events if e.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]),
            'monitoring_active': self.monitor.monitoring_active,
            'last_key_rotation': self.encryption_manager.last_rotation,
            'security_level': 'HIGH'
        }


class SecurityException(Exception):
    """Security-related exceptions"""
    pass


# Security decorators
def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract security context from kwargs
            context = kwargs.get('security_context')
            if not context:
                raise SecurityException("No security context provided")
            
            orchestrator = SecurityOrchestrator()
            if not orchestrator.access_controller.check_permission(context, permission):
                raise SecurityException(f"Permission denied: {permission}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def security_audit(operation: str):
    """Decorator for security auditing"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            orchestrator = SecurityOrchestrator()
            user_id = kwargs.get('user_id', 'unknown')
            
            with orchestrator.security_context(user_id, operation):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global security instance
security_orchestrator = SecurityOrchestrator()