"""
Security orchestration module for comprehensive cyber range protection.

This module coordinates all security components and enforces security policies
across the entire cyber range infrastructure.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different environments"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatLevel(Enum):
    """Threat severity levels"""
    INFO = "info"
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    name: str
    description: str
    security_level: SecurityLevel
    encryption_required: bool = True
    audit_logging_required: bool = True
    access_control_enabled: bool = True
    threat_detection_enabled: bool = True
    compliance_frameworks: List[str] = field(default_factory=list)
    allowed_operations: Set[str] = field(default_factory=set)
    blocked_operations: Set[str] = field(default_factory=set)
    session_timeout: int = 3600  # seconds
    max_failed_attempts: int = 3
    password_policy: Dict[str, Any] = field(default_factory=dict)
    network_restrictions: Dict[str, Any] = field(default_factory=dict)
    data_retention_days: int = 90
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SecurityContext:
    """Security context for operations"""
    user_id: str
    session_id: str
    security_level: SecurityLevel
    permissions: Set[str]
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    encrypted_data: Dict[str, str] = field(default_factory=dict)
    threat_score: float = 0.0
    compliance_status: Dict[str, bool] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event for monitoring and alerting"""
    event_id: str
    event_type: str
    threat_level: ThreatLevel
    source: str
    target: Optional[str]
    description: str
    details: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    response_actions: List[str] = field(default_factory=list)


class SecurityOrchestrator:
    """Main security orchestration class"""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        encryption_key: Optional[bytes] = None
    ):
        self.config_path = config_path or Path("security_config.json")
        
        # Initialize encryption
        if encryption_key:
            self.cipher_suite = Fernet(encryption_key)
        else:
            self.cipher_suite = Fernet(Fernet.generate_key())
        
        # Security state
        self.active_policies: Dict[str, SecurityPolicy] = {}
        self.security_contexts: Dict[str, SecurityContext] = {}
        self.security_events: List[SecurityEvent] = []
        self.threat_signatures: Dict[str, Any] = {}
        
        # Security components
        self.threat_detector = None
        self.access_controller = None
        self.audit_logger = None
        self.compliance_framework = None
        
        # Monitoring
        self.event_handlers: Dict[str, List[Callable]] = {
            'threat_detected': [],
            'access_denied': [],
            'policy_violation': [],
            'compliance_failure': [],
            'security_breach': []
        }
        
        # Rate limiting and protection
        self.failed_attempts: Dict[str, int] = {}
        self.blocked_ips: Set[str] = set()
        self.active_sessions: Dict[str, SecurityContext] = {}
        
        # Load configuration
        self._load_security_config()
        
        logger.info("Security orchestrator initialized")
    
    def initialize_security_components(self) -> None:
        """Initialize all security components"""
        
        from .threat_detector import ThreatDetector
        from .access_control import AccessController
        from .audit_logger import AuditLogger
        from .compliance_framework import ComplianceFramework
        
        self.threat_detector = ThreatDetector(self)
        self.access_controller = AccessController(self)
        self.audit_logger = AuditLogger(self)
        self.compliance_framework = ComplianceFramework(self)
        
        logger.info("All security components initialized")
    
    def register_security_policy(self, policy: SecurityPolicy) -> None:
        """Register a security policy"""
        self.active_policies[policy.name] = policy
        
        # Generate security event
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type="policy_registered",
            threat_level=ThreatLevel.INFO,
            source="security_orchestrator",
            target=policy.name,
            description=f"Security policy '{policy.name}' registered",
            details={"security_level": policy.security_level.value}
        )
        self._record_security_event(event)
        
        logger.info(f"Registered security policy: {policy.name}")
    
    def create_security_context(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        requested_permissions: Set[str]
    ) -> Optional[SecurityContext]:
        """Create a new security context for a user session"""
        
        # Check if IP is blocked
        if ip_address in self.blocked_ips:
            self._record_access_denied(user_id, ip_address, "Blocked IP address")
            return None
        
        # Check failed attempts
        if self.failed_attempts.get(user_id, 0) >= 3:
            self._record_access_denied(user_id, ip_address, "Too many failed attempts")
            return None
        
        # Determine security level based on requested permissions
        security_level = self._determine_security_level(requested_permissions)
        
        # Create session
        session_id = self._generate_session_id()
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            security_level=security_level,
            permissions=requested_permissions,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        # Store context
        self.security_contexts[session_id] = context
        self.active_sessions[session_id] = context
        
        # Generate security event
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type="session_created",
            threat_level=ThreatLevel.INFO,
            source="security_orchestrator",
            target=user_id,
            description=f"Security context created for user {user_id}",
            details={
                "session_id": session_id,
                "security_level": security_level.value,
                "permissions": list(requested_permissions)
            },
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address
        )
        self._record_security_event(event)
        
        logger.info(f"Created security context for user {user_id}")
        return context
    
    def validate_operation(
        self,
        session_id: str,
        operation: str,
        target: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Validate if an operation is allowed"""
        
        # Check if session exists
        if session_id not in self.security_contexts:
            self._record_access_denied(None, None, f"Invalid session: {session_id}")
            return False
        
        context = self.security_contexts[session_id]
        
        # Update last activity
        context.last_activity = datetime.now()
        
        # Check session timeout
        if self._is_session_expired(context):
            self._invalidate_session(session_id)
            self._record_access_denied(context.user_id, context.ip_address, "Session expired")
            return False
        
        # Check if operation is allowed by policy
        applicable_policy = self._get_applicable_policy(context)
        if applicable_policy:
            if operation in applicable_policy.blocked_operations:
                self._record_access_denied(
                    context.user_id, context.ip_address, 
                    f"Operation blocked by policy: {operation}"
                )
                return False
            
            if applicable_policy.allowed_operations and operation not in applicable_policy.allowed_operations:
                self._record_access_denied(
                    context.user_id, context.ip_address,
                    f"Operation not explicitly allowed: {operation}"
                )
                return False
        
        # Check permissions
        required_permission = self._get_required_permission(operation)
        if required_permission and required_permission not in context.permissions:
            self._record_access_denied(
                context.user_id, context.ip_address,
                f"Insufficient permissions for operation: {operation}"
            )
            return False
        
        # Threat detection
        if self.threat_detector:
            threat_score = self.threat_detector.assess_operation_risk(
                context, operation, target, additional_context
            )
            context.threat_score = max(context.threat_score, threat_score)
            
            if threat_score > 0.8:  # High threat threshold
                self._record_threat_detected(context, operation, threat_score)
                return False
        
        # Record successful operation
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type="operation_authorized",
            threat_level=ThreatLevel.INFO,
            source="security_orchestrator",
            target=target or operation,
            description=f"Operation '{operation}' authorized for user {context.user_id}",
            details={
                "operation": operation,
                "target": target,
                "threat_score": context.threat_score
            },
            user_id=context.user_id,
            session_id=session_id,
            ip_address=context.ip_address
        )
        self._record_security_event(event)
        
        return True
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Invalid encrypted data")
    
    def detect_security_threats(self) -> List[SecurityEvent]:
        """Detect security threats across the system"""
        
        threats = []
        
        # Check for suspicious activity patterns
        if self.threat_detector:
            detected_threats = self.threat_detector.scan_for_threats()
            threats.extend(detected_threats)
        
        # Check for policy violations
        for session_id, context in self.active_sessions.items():
            violations = self._check_policy_violations(context)
            threats.extend(violations)
        
        # Check for compliance failures
        if self.compliance_framework:
            compliance_failures = self.compliance_framework.check_compliance_status()
            threats.extend(compliance_failures)
        
        return threats
    
    def respond_to_threat(
        self,
        threat_event: SecurityEvent,
        response_actions: List[str]
    ) -> None:
        """Respond to a detected security threat"""
        
        logger.warning(f"Responding to threat: {threat_event.event_id}")
        
        for action in response_actions:
            try:
                if action == "block_ip":
                    if threat_event.ip_address:
                        self.blocked_ips.add(threat_event.ip_address)
                        logger.info(f"Blocked IP: {threat_event.ip_address}")
                
                elif action == "invalidate_session":
                    if threat_event.session_id:
                        self._invalidate_session(threat_event.session_id)
                        logger.info(f"Invalidated session: {threat_event.session_id}")
                
                elif action == "escalate_alert":
                    self._escalate_security_alert(threat_event)
                
                elif action == "disable_user":
                    if threat_event.user_id:
                        self._disable_user_temporarily(threat_event.user_id)
                        logger.info(f"Temporarily disabled user: {threat_event.user_id}")
                
                elif action == "increase_monitoring":
                    self._increase_monitoring_level(threat_event)
                
            except Exception as e:
                logger.error(f"Failed to execute response action '{action}': {e}")
        
        # Mark threat as resolved
        threat_event.resolved = True
        threat_event.response_actions = response_actions
        
        # Trigger event handlers
        self._trigger_event_handlers('threat_detected', threat_event)
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        # Filter recent events
        recent_events = [e for e in self.security_events if e.timestamp >= last_24h]
        
        # Categorize events
        event_categories = {}
        threat_levels = {}
        
        for event in recent_events:
            event_categories[event.event_type] = event_categories.get(event.event_type, 0) + 1
            threat_levels[event.threat_level.value] = threat_levels.get(event.threat_level.value, 0) + 1
        
        # Active sessions analysis
        active_session_count = len(self.active_sessions)
        high_threat_sessions = len([
            s for s in self.active_sessions.values() 
            if s.threat_score > 0.5
        ])
        
        # Compliance status
        compliance_status = {}
        if self.compliance_framework:
            compliance_status = self.compliance_framework.get_compliance_summary()
        
        report = {
            'generated_at': now.isoformat(),
            'summary': {
                'total_events_24h': len(recent_events),
                'active_sessions': active_session_count,
                'high_threat_sessions': high_threat_sessions,
                'blocked_ips': len(self.blocked_ips),
                'active_policies': len(self.active_policies)
            },
            'event_breakdown': event_categories,
            'threat_level_distribution': threat_levels,
            'policy_summary': {
                name: {
                    'security_level': policy.security_level.value,
                    'encryption_required': policy.encryption_required,
                    'compliance_frameworks': policy.compliance_frameworks
                }
                for name, policy in self.active_policies.items()
            },
            'compliance_status': compliance_status,
            'security_recommendations': self._generate_security_recommendations(recent_events)
        }
        
        return report
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        
        expired_sessions = []
        for session_id, context in list(self.active_sessions.items()):
            if self._is_session_expired(context):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self._invalidate_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)
    
    def on_security_event(self, event_type: str):
        """Decorator for registering security event handlers"""
        def decorator(func):
            if event_type in self.event_handlers:
                self.event_handlers[event_type].append(func)
            return func
        return decorator
    
    def _load_security_config(self) -> None:
        """Load security configuration from file"""
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load default policies
                for policy_config in config.get('policies', []):
                    policy = SecurityPolicy(**policy_config)
                    self.active_policies[policy.name] = policy
                
                # Load threat signatures
                self.threat_signatures = config.get('threat_signatures', {})
                
                logger.info(f"Loaded security configuration from {self.config_path}")
                
            except Exception as e:
                logger.error(f"Failed to load security config: {e}")
        else:
            # Create default configuration
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default security configuration"""
        
        default_policies = [
            SecurityPolicy(
                name="default_high_security",
                description="Default high security policy",
                security_level=SecurityLevel.HIGH,
                encryption_required=True,
                audit_logging_required=True,
                compliance_frameworks=["GDPR", "SOC2"],
                allowed_operations={"read", "create", "update"},
                blocked_operations={"delete_all", "admin_access"},
                session_timeout=1800,
                max_failed_attempts=3
            ),
            SecurityPolicy(
                name="research_policy",
                description="Policy for research environments",
                security_level=SecurityLevel.MEDIUM,
                encryption_required=True,
                compliance_frameworks=["ISO27001"],
                session_timeout=3600
            )
        ]
        
        for policy in default_policies:
            self.active_policies[policy.name] = policy
        
        logger.info("Created default security configuration")
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now().isoformat()
        random_data = secrets.token_bytes(16)
        combined = f"{timestamp}{random_data.hex()}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _determine_security_level(self, permissions: Set[str]) -> SecurityLevel:
        """Determine security level based on requested permissions"""
        
        admin_permissions = {'admin', 'delete_all', 'system_config'}
        high_risk_permissions = {'create_attack', 'modify_security', 'access_sensitive'}
        
        if any(perm in admin_permissions for perm in permissions):
            return SecurityLevel.CRITICAL
        elif any(perm in high_risk_permissions for perm in permissions):
            return SecurityLevel.HIGH
        elif len(permissions) > 5:
            return SecurityLevel.MEDIUM
        else:
            return SecurityLevel.LOW
    
    def _get_applicable_policy(self, context: SecurityContext) -> Optional[SecurityPolicy]:
        """Get the applicable security policy for a context"""
        
        # Find policy matching security level
        for policy in self.active_policies.values():
            if policy.security_level == context.security_level:
                return policy
        
        # Return default high security policy
        return self.active_policies.get("default_high_security")
    
    def _get_required_permission(self, operation: str) -> Optional[str]:
        """Get required permission for an operation"""
        
        permission_map = {
            'create_attack': 'attack_creation',
            'modify_range': 'range_modification',
            'access_logs': 'log_access',
            'delete_data': 'data_deletion',
            'admin_operation': 'admin',
            'view_sensitive': 'sensitive_data_access'
        }
        
        return permission_map.get(operation)
    
    def _is_session_expired(self, context: SecurityContext) -> bool:
        """Check if a session is expired"""
        
        applicable_policy = self._get_applicable_policy(context)
        timeout = applicable_policy.session_timeout if applicable_policy else 3600
        
        time_since_activity = (datetime.now() - context.last_activity).total_seconds()
        return time_since_activity > timeout
    
    def _invalidate_session(self, session_id: str) -> None:
        """Invalidate a session"""
        
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            
            event = SecurityEvent(
                event_id=self._generate_event_id(),
                event_type="session_invalidated",
                threat_level=ThreatLevel.INFO,
                source="security_orchestrator",
                target=context.user_id,
                description=f"Session invalidated for user {context.user_id}",
                details={"session_id": session_id},
                user_id=context.user_id,
                session_id=session_id,
                ip_address=context.ip_address
            )
            self._record_security_event(event)
    
    def _record_security_event(self, event: SecurityEvent) -> None:
        """Record a security event"""
        
        self.security_events.append(event)
        
        # Log to audit system
        if self.audit_logger:
            self.audit_logger.log_security_event(event)
        
        # Trigger event handlers
        event_type = event.event_type.replace('_', ' ').title().replace(' ', '')
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._trigger_event_handlers('threat_detected', event)
    
    def _record_access_denied(
        self,
        user_id: Optional[str],
        ip_address: Optional[str],
        reason: str
    ) -> None:
        """Record an access denied event"""
        
        # Increment failed attempts
        if user_id:
            self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
        
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type="access_denied",
            threat_level=ThreatLevel.MEDIUM,
            source="security_orchestrator",
            target=user_id,
            description=f"Access denied: {reason}",
            details={"reason": reason},
            user_id=user_id,
            ip_address=ip_address
        )
        self._record_security_event(event)
        
        # Trigger access denied handlers
        self._trigger_event_handlers('access_denied', event)
    
    def _record_threat_detected(
        self,
        context: SecurityContext,
        operation: str,
        threat_score: float
    ) -> None:
        """Record a threat detection event"""
        
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type="threat_detected",
            threat_level=ThreatLevel.HIGH if threat_score > 0.9 else ThreatLevel.MEDIUM,
            source="threat_detector",
            target=context.user_id,
            description=f"High threat score detected for operation: {operation}",
            details={
                "operation": operation,
                "threat_score": threat_score,
                "security_level": context.security_level.value
            },
            user_id=context.user_id,
            session_id=context.session_id,
            ip_address=context.ip_address
        )
        self._record_security_event(event)
    
    def _check_policy_violations(self, context: SecurityContext) -> List[SecurityEvent]:
        """Check for policy violations"""
        
        violations = []
        applicable_policy = self._get_applicable_policy(context)
        
        if applicable_policy:
            # Check session timeout
            if self._is_session_expired(context):
                violation = SecurityEvent(
                    event_id=self._generate_event_id(),
                    event_type="policy_violation",
                    threat_level=ThreatLevel.MEDIUM,
                    source="security_orchestrator",
                    target=context.user_id,
                    description="Session timeout policy violation",
                    details={"policy": applicable_policy.name},
                    user_id=context.user_id,
                    session_id=context.session_id,
                    ip_address=context.ip_address
                )
                violations.append(violation)
        
        return violations
    
    def _escalate_security_alert(self, threat_event: SecurityEvent) -> None:
        """Escalate security alert to administrators"""
        
        escalation_event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type="security_escalation",
            threat_level=ThreatLevel.CRITICAL,
            source="security_orchestrator",
            target="security_team",
            description=f"Escalated security alert: {threat_event.description}",
            details={
                "original_event_id": threat_event.event_id,
                "escalation_reason": "High threat level detected"
            }
        )
        self._record_security_event(escalation_event)
        
        logger.critical(f"Security alert escalated: {threat_event.event_id}")
    
    def _disable_user_temporarily(self, user_id: str) -> None:
        """Temporarily disable a user"""
        
        # Invalidate all sessions for the user
        sessions_to_invalidate = [
            session_id for session_id, context in self.active_sessions.items()
            if context.user_id == user_id
        ]
        
        for session_id in sessions_to_invalidate:
            self._invalidate_session(session_id)
        
        # Add to blocked users (implementation depends on user management system)
        logger.warning(f"User {user_id} temporarily disabled due to security threat")
    
    def _increase_monitoring_level(self, threat_event: SecurityEvent) -> None:
        """Increase monitoring level for specific targets"""
        
        if threat_event.user_id:
            # Increase monitoring for user
            logger.info(f"Increased monitoring level for user: {threat_event.user_id}")
        
        if threat_event.ip_address:
            # Increase monitoring for IP
            logger.info(f"Increased monitoring level for IP: {threat_event.ip_address}")
    
    def _generate_security_recommendations(self, recent_events: List[SecurityEvent]) -> List[str]:
        """Generate security recommendations based on recent events"""
        
        recommendations = []
        
        # Count event types
        threat_events = [e for e in recent_events if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
        access_denied_events = [e for e in recent_events if e.event_type == "access_denied"]
        
        if len(threat_events) > 5:
            recommendations.append("High number of security threats detected - consider increasing security level")
        
        if len(access_denied_events) > 10:
            recommendations.append("Many access denied events - review permission policies")
        
        # Check for specific patterns
        failed_logins = [e for e in recent_events if "failed" in e.description.lower()]
        if len(failed_logins) > 20:
            recommendations.append("High number of failed login attempts - implement additional protection")
        
        if len(self.blocked_ips) > 10:
            recommendations.append("Many blocked IPs - review network security configuration")
        
        return recommendations
    
    def _trigger_event_handlers(self, event_type: str, event: SecurityEvent) -> None:
        """Trigger registered event handlers"""
        
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")


# Security decorators for method protection
def requires_security_context(func):
    """Decorator to require valid security context"""
    def wrapper(self, *args, **kwargs):
        session_id = kwargs.get('session_id')
        if not session_id or session_id not in self.security_orchestrator.security_contexts:
            raise PermissionError("Valid security context required")
        return func(self, *args, **kwargs)
    return wrapper


def requires_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            session_id = kwargs.get('session_id')
            if session_id:
                context = self.security_orchestrator.security_contexts.get(session_id)
                if not context or permission not in context.permissions:
                    raise PermissionError(f"Permission '{permission}' required")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def audit_operation(operation_name: str):
    """Decorator to audit operations"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            session_id = kwargs.get('session_id')
            if hasattr(self, 'security_orchestrator') and session_id:
                self.security_orchestrator.validate_operation(session_id, operation_name)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator