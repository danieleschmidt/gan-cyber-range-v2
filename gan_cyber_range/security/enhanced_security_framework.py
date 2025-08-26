"""
Enhanced Security Framework

Comprehensive security framework with multi-layered protection,
ethical compliance, and advanced monitoring capabilities.
"""

import logging
import hashlib
import hmac
import secrets
import time
import json
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re
from pathlib import Path

from ..utils.robust_error_handler import robust, critical, ErrorSeverity

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for operations"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class AccessLevel(Enum):
    """User access levels"""
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    SYSTEM = "system"


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    access_level: AccessLevel
    session_id: str
    timestamp: datetime
    ip_address: str
    permissions: Set[str] = field(default_factory=set)
    rate_limit_tokens: int = 100
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class SecurityEvent:
    """Security event for monitoring"""
    event_id: str
    event_type: str
    severity: str
    timestamp: datetime
    user_id: Optional[str]
    details: Dict[str, Any]
    resolved: bool = False


class EthicalFramework:
    """Ethical compliance framework"""
    
    def __init__(self):
        self.allowed_uses = {
            "research", "training", "defense", "education", "testing"
        }
        self.prohibited_targets = {
            "production_systems", "real_networks", "unauthorized_systems",
            "critical_infrastructure", "personal_devices"
        }
        self.prohibited_techniques = {
            "actual_malware_deployment", "real_data_destruction",
            "unauthorized_access", "privacy_violation"
        }
        
        self.consent_required = True
        self.monitoring_enabled = True
        self.audit_logging = True
    
    @robust(severity=ErrorSeverity.HIGH)
    def is_compliant(self, request: Dict[str, Any]) -> bool:
        """Check if request complies with ethical guidelines"""
        try:
            # Check purpose
            purpose = request.get("purpose", "").lower()
            if not any(allowed in purpose for allowed in self.allowed_uses):
                logger.warning(f"Non-compliant purpose: {purpose}")
                return False
            
            # Check targets
            targets = request.get("targets", [])
            if isinstance(targets, str):
                targets = [targets]
            
            for target in targets:
                if any(prohibited in target.lower() for prohibited in self.prohibited_targets):
                    logger.warning(f"Prohibited target: {target}")
                    return False
            
            # Check techniques
            techniques = request.get("techniques", [])
            if isinstance(techniques, str):
                techniques = [techniques]
            
            for technique in techniques:
                if any(prohibited in technique.lower() for prohibited in self.prohibited_techniques):
                    logger.warning(f"Prohibited technique: {technique}")
                    return False
            
            # Check consent
            if self.consent_required and not request.get("consent", False):
                logger.warning("Consent required but not provided")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ethical compliance check failed: {e}")
            return False
    
    def get_violation_reason(self, request: Dict[str, Any]) -> Optional[str]:
        """Get specific reason for ethical violation"""
        if not self.is_compliant(request):
            purpose = request.get("purpose", "").lower()
            if not any(allowed in purpose for allowed in self.allowed_uses):
                return f"Purpose '{purpose}' not in allowed uses: {self.allowed_uses}"
            
            targets = request.get("targets", [])
            for target in targets:
                if any(prohibited in target.lower() for prohibited in self.prohibited_targets):
                    return f"Target '{target}' is prohibited"
            
            if self.consent_required and not request.get("consent", False):
                return "Explicit consent is required"
        
        return None


class InputSanitizer:
    """Advanced input sanitization and validation"""
    
    def __init__(self):
        # Dangerous patterns to filter
        self.dangerous_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"javascript:",  # JavaScript URLs
            r"on\w+\s*=",  # Event handlers
            r"eval\s*\(",  # eval() calls
            r"exec\s*\(",  # exec() calls
            r"system\s*\(",  # system() calls
            r"shell_exec\s*\(",  # shell_exec() calls
            r"\.\./",  # Path traversal
            r"file://",  # File protocol
            r"data:",  # Data URLs
            r"vbscript:",  # VBScript
            r"\bSELECT\b.*\bFROM\b",  # SQL injection
            r"\bDROP\b.*\bTABLE\b",  # SQL injection
            r"\bINSERT\b.*\bINTO\b",  # SQL injection
            r"\bUNION\b.*\bSELECT\b",  # SQL injection
        ]
        
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.dangerous_patterns
        ]
        
        # Length limits
        self.max_lengths = {
            "string": 10000,
            "list": 1000,
            "dict": 100,
            "payload": 50000
        }
    
    @robust(severity=ErrorSeverity.HIGH)
    def sanitize_input(self, data: Any, input_type: str = "string") -> Any:
        """Sanitize input data"""
        if data is None:
            return data
        
        if isinstance(data, str):
            return self._sanitize_string(data)
        elif isinstance(data, list):
            return self._sanitize_list(data)
        elif isinstance(data, dict):
            return self._sanitize_dict(data)
        else:
            return data
    
    def _sanitize_string(self, text: str) -> str:
        """Sanitize string input"""
        # Check length
        if len(text) > self.max_lengths["string"]:
            logger.warning(f"String too long: {len(text)} chars")
            text = text[:self.max_lengths["string"]]
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                logger.warning(f"Dangerous pattern detected in: {text[:100]}...")
                # Replace with safe placeholder
                text = pattern.sub("[FILTERED]", text)
        
        return text
    
    def _sanitize_list(self, data: List[Any]) -> List[Any]:
        """Sanitize list input"""
        if len(data) > self.max_lengths["list"]:
            logger.warning(f"List too long: {len(data)} items")
            data = data[:self.max_lengths["list"]]
        
        return [self.sanitize_input(item) for item in data]
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary input"""
        if len(data) > self.max_lengths["dict"]:
            logger.warning(f"Dictionary too large: {len(data)} items")
            # Keep only first N items
            items = list(data.items())[:self.max_lengths["dict"]]
            data = dict(items)
        
        return {
            self._sanitize_string(str(key)): self.sanitize_input(value)
            for key, value in data.items()
        }
    
    def is_safe(self, data: Any) -> bool:
        """Check if input is safe without modification"""
        try:
            if isinstance(data, str):
                return not any(pattern.search(data) for pattern in self.compiled_patterns)
            elif isinstance(data, (list, dict)):
                sanitized = self.sanitize_input(data)
                return sanitized == data
            return True
        except Exception:
            return False


class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self):
        self.user_buckets: Dict[str, Dict[str, Any]] = {}
        self.global_limits = {
            "requests_per_minute": 100,
            "attacks_per_hour": 1000,
            "training_sessions_per_day": 10
        }
        self.user_limits = {
            AccessLevel.GUEST: {"requests_per_minute": 10, "attacks_per_hour": 50},
            AccessLevel.USER: {"requests_per_minute": 50, "attacks_per_hour": 500},
            AccessLevel.ADMIN: {"requests_per_minute": 200, "attacks_per_hour": 2000},
        }
    
    @robust(severity=ErrorSeverity.MEDIUM)
    def check_rate_limit(
        self, 
        user_id: str, 
        access_level: AccessLevel, 
        operation: str
    ) -> bool:
        """Check if operation is within rate limits"""
        now = datetime.now()
        
        # Initialize user bucket if needed
        if user_id not in self.user_buckets:
            self.user_buckets[user_id] = {
                "requests": [],
                "attacks": [],
                "training_sessions": []
            }
        
        bucket = self.user_buckets[user_id]
        
        # Clean old entries
        self._clean_bucket(bucket, now)
        
        # Check specific operation limits
        if operation == "request":
            bucket["requests"].append(now)
            limit = self.user_limits[access_level]["requests_per_minute"]
            recent = [r for r in bucket["requests"] if now - r < timedelta(minutes=1)]
            return len(recent) <= limit
        
        elif operation == "attack_generation":
            bucket["attacks"].append(now)
            limit = self.user_limits[access_level]["attacks_per_hour"]
            recent = [a for a in bucket["attacks"] if now - a < timedelta(hours=1)]
            return len(recent) <= limit
        
        elif operation == "training_session":
            bucket["training_sessions"].append(now)
            recent = [t for t in bucket["training_sessions"] if now - t < timedelta(days=1)]
            return len(recent) <= self.global_limits["training_sessions_per_day"]
        
        return True
    
    def _clean_bucket(self, bucket: Dict[str, List], now: datetime):
        """Clean old entries from rate limit bucket"""
        bucket["requests"] = [r for r in bucket["requests"] if now - r < timedelta(minutes=5)]
        bucket["attacks"] = [a for a in bucket["attacks"] if now - a < timedelta(hours=2)]
        bucket["training_sessions"] = [t for t in bucket["training_sessions"] if now - t < timedelta(days=2)]


class SecurityMonitor:
    """Real-time security monitoring and alerting"""
    
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.alert_thresholds = {
            "failed_auth_attempts": 5,
            "rate_limit_violations": 10,
            "suspicious_patterns": 3
        }
        self.alert_callbacks: List[Callable] = []
    
    @robust(severity=ErrorSeverity.LOW)
    def log_event(
        self, 
        event_type: str, 
        severity: str, 
        details: Dict[str, Any],
        user_id: Optional[str] = None
    ):
        """Log security event"""
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            user_id=user_id,
            details=details
        )
        
        self.events.append(event)
        logger.info(f"Security event: {event_type} - {severity}")
        
        # Check for alert conditions
        self._check_alerts(event)
    
    def _check_alerts(self, event: SecurityEvent):
        """Check if event triggers alerts"""
        recent_events = [
            e for e in self.events 
            if e.timestamp > datetime.now() - timedelta(minutes=15)
        ]
        
        # Count recent events by type
        event_counts = {}
        for e in recent_events:
            event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1
        
        # Check thresholds
        for event_type, threshold in self.alert_thresholds.items():
            if event_counts.get(event_type, 0) >= threshold:
                self._trigger_alert(event_type, event_counts[event_type], threshold)
    
    def _trigger_alert(self, event_type: str, count: int, threshold: int):
        """Trigger security alert"""
        alert_data = {
            "alert_type": "threshold_exceeded",
            "event_type": event_type,
            "count": count,
            "threshold": threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.critical(f"SECURITY ALERT: {event_type} threshold exceeded ({count}/{threshold})")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary"""
        recent_events = [
            e for e in self.events 
            if e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        # Count events by type and severity
        event_types = {}
        severities = {}
        
        for event in recent_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            severities[event.severity] = severities.get(event.severity, 0) + 1
        
        return {
            "total_events_24h": len(recent_events),
            "event_types": event_types,
            "severities": severities,
            "alerts_triggered": sum(
                1 for e in recent_events 
                if e.event_type in self.alert_thresholds
            )
        }


class EnhancedSecurityFramework:
    """Main security framework orchestrator"""
    
    def __init__(self):
        self.ethical_framework = EthicalFramework()
        self.input_sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter()
        self.security_monitor = SecurityMonitor()
        
        # Active sessions
        self.active_sessions: Dict[str, SecurityContext] = {}
        
        # Security configuration
        self.config = {
            "enforce_ethics": True,
            "sanitize_inputs": True,
            "rate_limiting": True,
            "security_monitoring": True,
            "session_timeout": 3600,  # 1 hour
            "max_session_per_user": 5
        }
    
    @critical(max_retries=1)
    def create_security_context(
        self, 
        user_id: str, 
        access_level: AccessLevel,
        ip_address: str,
        permissions: Optional[Set[str]] = None
    ) -> SecurityContext:
        """Create new security context"""
        
        # Check existing sessions
        user_sessions = [
            ctx for ctx in self.active_sessions.values()
            if ctx.user_id == user_id
        ]
        
        if len(user_sessions) >= self.config["max_session_per_user"]:
            # Remove oldest session
            oldest = min(user_sessions, key=lambda x: x.timestamp)
            del self.active_sessions[oldest.session_id]
            logger.info(f"Removed oldest session for user {user_id}")
        
        # Create new context
        context = SecurityContext(
            user_id=user_id,
            access_level=access_level,
            session_id=secrets.token_hex(32),
            timestamp=datetime.now(),
            ip_address=ip_address,
            permissions=permissions or set()
        )
        
        self.active_sessions[context.session_id] = context
        
        self.security_monitor.log_event(
            "session_created",
            "info",
            {"user_id": user_id, "access_level": access_level.value, "ip": ip_address}
        )
        
        return context
    
    @robust(severity=ErrorSeverity.MEDIUM)
    def validate_request(
        self, 
        context: SecurityContext, 
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and sanitize request"""
        
        # Update last activity
        context.last_activity = datetime.now()
        
        # Check session timeout
        session_age = datetime.now() - context.timestamp
        if session_age.total_seconds() > self.config["session_timeout"]:
            raise SecurityError("Session expired")
        
        # Rate limiting
        if self.config["rate_limiting"]:
            if not self.rate_limiter.check_rate_limit(
                context.user_id, 
                context.access_level, 
                "request"
            ):
                self.security_monitor.log_event(
                    "rate_limit_violation",
                    "warning",
                    {"user_id": context.user_id, "request": str(request)[:200]}
                )
                raise SecurityError("Rate limit exceeded")
        
        # Input sanitization
        if self.config["sanitize_inputs"]:
            sanitized_request = self.input_sanitizer.sanitize_input(request, "dict")
            if sanitized_request != request:
                self.security_monitor.log_event(
                    "input_sanitized",
                    "info",
                    {"user_id": context.user_id, "changes": "input_modified"}
                )
            request = sanitized_request
        
        # Ethical compliance
        if self.config["enforce_ethics"]:
            if not self.ethical_framework.is_compliant(request):
                violation_reason = self.ethical_framework.get_violation_reason(request)
                self.security_monitor.log_event(
                    "ethics_violation",
                    "warning",
                    {
                        "user_id": context.user_id, 
                        "reason": violation_reason,
                        "request": str(request)[:200]
                    }
                )
                raise SecurityError(f"Ethical violation: {violation_reason}")
        
        return request
    
    def check_permission(self, context: SecurityContext, permission: str) -> bool:
        """Check if user has specific permission"""
        return (
            permission in context.permissions or
            context.access_level in [AccessLevel.ADMIN, AccessLevel.SUPER_ADMIN, AccessLevel.SYSTEM]
        )
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report"""
        return {
            "active_sessions": len(self.active_sessions),
            "security_events": self.security_monitor.get_security_summary(),
            "configuration": self.config,
            "ethical_compliance": {
                "framework_active": self.config["enforce_ethics"],
                "allowed_uses": list(self.ethical_framework.allowed_uses),
                "prohibited_targets": list(self.ethical_framework.prohibited_targets)
            }
        }


class SecurityError(Exception):
    """Custom security exception"""
    pass


# Global security framework instance
security_framework = EnhancedSecurityFramework()