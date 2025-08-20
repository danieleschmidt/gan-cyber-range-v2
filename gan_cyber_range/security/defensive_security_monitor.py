"""
Defensive Security Monitor for GAN-Cyber-Range-v2

This module provides comprehensive security monitoring specifically designed to ensure
the platform is used only for defensive training purposes and maintains strict ethical
boundaries. It includes real-time monitoring, audit logging, and compliance checks.
"""

import logging
import asyncio
import json
import hashlib
import hmac
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import uuid
import threading
import time
from collections import defaultdict, deque
import re

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events to monitor"""
    TRAINING_SESSION_START = "training_session_start"
    TRAINING_SESSION_END = "training_session_end"
    CONTENT_ACCESS = "content_access"
    SKILL_ASSESSMENT = "skill_assessment"
    CURRICULUM_GENERATION = "curriculum_generation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    COMPLIANCE_VIOLATION = "compliance_violation"
    ETHICAL_BOUNDARY_CHECK = "ethical_boundary_check"
    SYSTEM_HEALTH_CHECK = "system_health_check"


class SecurityLevel(Enum):
    """Security alert levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    FERPA = "ferpa"  # Educational privacy
    ISO27001 = "iso27001"
    NIST = "nist"


@dataclass
class SecurityEvent:
    """Represents a security monitoring event"""
    event_id: str
    event_type: SecurityEventType
    security_level: SecurityLevel
    timestamp: datetime
    source: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    compliance_implications: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)


@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    name: str
    description: str
    policy_type: str  # e.g., "access_control", "data_protection", "ethical_use"
    rules: List[Dict[str, Any]]
    enforcement_level: SecurityLevel
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    violation_actions: List[str] = field(default_factory=list)


@dataclass
class UserActivity:
    """User activity tracking for behavioral analysis"""
    user_id: str
    session_id: str
    activity_type: str
    timestamp: datetime
    resource_accessed: str
    duration: Optional[float] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


class DefensiveSecurityMonitor:
    """Comprehensive security monitoring system for defensive training platform"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.security_events: deque = deque(maxlen=10000)  # Ring buffer for events
        self.user_activities: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Monitoring state
        self.active_sessions: Dict[str, Dict] = {}
        self.user_risk_scores: Dict[str, float] = {}
        self.suspicious_patterns: Dict[str, List] = defaultdict(list)
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alert_callbacks: List[Callable] = []
        
        # Compliance tracking
        self.compliance_violations: Dict[str, List] = defaultdict(list)
        self.audit_log: deque = deque(maxlen=50000)
        
        # Initialize security policies
        self._initialize_security_policies()
        
        # Start monitoring
        self.start_monitoring()
    
    def _initialize_security_policies(self):
        """Initialize default security policies for defensive training"""
        
        # Ethical use policy
        ethical_policy = SecurityPolicy(
            policy_id="ethical_use_001",
            name="Ethical Use of AI Training Platform",
            description="Ensures platform is used only for defensive cybersecurity training",
            policy_type="ethical_use",
            enforcement_level=SecurityLevel.CRITICAL,
            rules=[
                {
                    "rule_id": "no_offensive_content",
                    "description": "Prohibit creation or distribution of offensive attack tools",
                    "condition": "content_type == 'offensive_tool' OR intent == 'malicious'",
                    "action": "block_and_alert"
                },
                {
                    "rule_id": "training_purpose_only", 
                    "description": "All activities must be for legitimate training purposes",
                    "condition": "purpose != 'training' AND purpose != 'education' AND purpose != 'research'",
                    "action": "review_and_alert"
                },
                {
                    "rule_id": "no_real_target_systems",
                    "description": "Prohibit targeting of real production systems",
                    "condition": "target_type == 'production' OR target_ip IN real_network_ranges",
                    "action": "block_immediately"
                }
            ],
            compliance_frameworks=[ComplianceFramework.ISO27001, ComplianceFramework.NIST],
            violation_actions=["immediate_suspension", "admin_notification", "audit_review"]
        )
        
        # Data protection policy
        data_protection_policy = SecurityPolicy(
            policy_id="data_protection_001",
            name="Learner Data Protection",
            description="Protect personal and learning data of platform users",
            policy_type="data_protection",
            enforcement_level=SecurityLevel.HIGH,
            rules=[
                {
                    "rule_id": "encrypt_pii",
                    "description": "All PII must be encrypted at rest and in transit",
                    "condition": "data_contains_pii == True AND encryption == False",
                    "action": "encrypt_and_alert"
                },
                {
                    "rule_id": "data_minimization",
                    "description": "Collect only necessary data for training purposes",
                    "condition": "data_collection_scope > training_requirements",
                    "action": "reduce_scope_and_alert"
                },
                {
                    "rule_id": "retention_limits",
                    "description": "Delete user data after retention period",
                    "condition": "data_age > retention_period",
                    "action": "schedule_deletion"
                }
            ],
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA, ComplianceFramework.FERPA],
            violation_actions=["data_anonymization", "user_notification", "compliance_report"]
        )
        
        # Access control policy
        access_control_policy = SecurityPolicy(
            policy_id="access_control_001", 
            name="Secure Access Control",
            description="Ensure proper authentication and authorization for all resources",
            policy_type="access_control",
            enforcement_level=SecurityLevel.HIGH,
            rules=[
                {
                    "rule_id": "strong_authentication",
                    "description": "Require strong authentication for sensitive operations",
                    "condition": "operation_sensitivity == 'high' AND auth_strength < 2",
                    "action": "require_mfa"
                },
                {
                    "rule_id": "role_based_access",
                    "description": "Enforce role-based access to training content",
                    "condition": "user_role NOT IN allowed_roles FOR resource",
                    "action": "deny_access"
                },
                {
                    "rule_id": "session_management",
                    "description": "Manage session lifecycle securely",
                    "condition": "session_age > max_session_time OR concurrent_sessions > limit",
                    "action": "terminate_session"
                }
            ],
            compliance_frameworks=[ComplianceFramework.ISO27001, ComplianceFramework.NIST],
            violation_actions=["account_lockout", "admin_notification", "security_review"]
        )
        
        self.security_policies[ethical_policy.policy_id] = ethical_policy
        self.security_policies[data_protection_policy.policy_id] = data_protection_policy
        self.security_policies[access_control_policy.policy_id] = access_control_policy
    
    def start_monitoring(self):
        """Start real-time security monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Defensive security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Defensive security monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check for suspicious patterns
                self._detect_suspicious_patterns()
                
                # Validate compliance
                self._check_compliance_violations()
                
                # Update risk scores
                self._update_risk_scores()
                
                # Health check
                self._system_health_check()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer if there's an error
    
    def log_security_event(
        self,
        event_type: SecurityEventType,
        security_level: SecurityLevel,
        source: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        compliance_implications: Optional[List[str]] = None
    ) -> str:
        """Log a security event for monitoring and audit"""
        
        event_id = str(uuid.uuid4())
        
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            security_level=security_level,
            timestamp=datetime.now(),
            source=source,
            user_id=user_id,
            session_id=session_id,
            details=details or {},
            compliance_implications=compliance_implications or []
        )
        
        # Add to event log
        self.security_events.append(event)
        
        # Add to audit log
        self._add_to_audit_log("SECURITY_EVENT", event.__dict__)
        
        # Check if this triggers any policy violations
        self._check_policy_violations(event)
        
        # Send alerts if necessary
        if security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            self._send_alert(event)
        
        logger.info(f"Security event logged: {event_type.value} (Level: {security_level.value})")
        
        return event_id
    
    def track_user_activity(
        self,
        user_id: str,
        session_id: str,
        activity_type: str,
        resource_accessed: str,
        duration: Optional[float] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        """Track user activity for behavioral analysis"""
        
        activity = UserActivity(
            user_id=user_id,
            session_id=session_id,
            activity_type=activity_type,
            timestamp=datetime.now(),
            resource_accessed=resource_accessed,
            duration=duration,
            success=success,
            details=details or {}
        )
        
        # Add to user activity log
        self.user_activities[user_id].append(activity)
        
        # Check for suspicious patterns
        self._analyze_user_behavior(user_id, activity)
        
        # Update session tracking
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "user_id": user_id,
                "start_time": datetime.now(),
                "activities": []
            }
        
        self.active_sessions[session_id]["activities"].append(activity)
    
    def _analyze_user_behavior(self, user_id: str, activity: UserActivity):
        """Analyze user behavior for suspicious patterns"""
        
        user_activities = self.user_activities[user_id]
        
        # Check for rapid sequential access (potential automated behavior)
        recent_activities = [a for a in user_activities 
                           if (activity.timestamp - a.timestamp).total_seconds() < 60]
        
        if len(recent_activities) > 20:  # More than 20 activities in 1 minute
            self.log_security_event(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                security_level=SecurityLevel.MEDIUM,
                source="behavioral_analysis",
                user_id=user_id,
                session_id=activity.session_id,
                details={
                    "pattern": "rapid_sequential_access",
                    "activity_count": len(recent_activities),
                    "timeframe_seconds": 60
                }
            )
        
        # Check for unusual resource access patterns
        accessed_resources = [a.resource_accessed for a in recent_activities]
        unique_resources = len(set(accessed_resources))
        
        if unique_resources > 15:  # Accessing many different resources quickly
            self.log_security_event(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                security_level=SecurityLevel.MEDIUM,
                source="behavioral_analysis",
                user_id=user_id,
                session_id=activity.session_id,
                details={
                    "pattern": "broad_resource_scanning",
                    "unique_resources": unique_resources,
                    "total_activities": len(recent_activities)
                }
            )
        
        # Check for failed access attempts
        failed_activities = [a for a in recent_activities if not a.success]
        if len(failed_activities) > 5:  # Multiple failed attempts
            self.log_security_event(
                event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                security_level=SecurityLevel.HIGH,
                source="behavioral_analysis",
                user_id=user_id,
                session_id=activity.session_id,
                details={
                    "pattern": "multiple_failed_attempts",
                    "failed_count": len(failed_activities),
                    "resources": [a.resource_accessed for a in failed_activities]
                }
            )
    
    def _detect_suspicious_patterns(self):
        """Detect suspicious patterns across all users"""
        
        current_time = datetime.now()
        
        # Check for coordinated attacks (multiple users with similar patterns)
        user_patterns = {}
        
        for user_id, activities in self.user_activities.items():
            recent_activities = [a for a in activities 
                               if (current_time - a.timestamp).total_seconds() < 300]  # 5 minutes
            
            if recent_activities:
                pattern_signature = self._calculate_pattern_signature(recent_activities)
                if pattern_signature not in user_patterns:
                    user_patterns[pattern_signature] = []
                user_patterns[pattern_signature].append(user_id)
        
        # Alert on coordinated patterns
        for pattern, users in user_patterns.items():
            if len(users) > 3:  # 3+ users with similar patterns
                self.log_security_event(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    security_level=SecurityLevel.HIGH,
                    source="pattern_detection",
                    details={
                        "pattern": "coordinated_activity",
                        "user_count": len(users),
                        "users": users,
                        "pattern_signature": pattern
                    }
                )
    
    def _calculate_pattern_signature(self, activities: List[UserActivity]) -> str:
        """Calculate a signature for activity patterns"""
        
        # Create signature based on resource access pattern and timing
        resources = [a.resource_accessed for a in activities[-10:]]  # Last 10 activities
        timings = [(activities[i+1].timestamp - activities[i].timestamp).total_seconds() 
                  for i in range(len(activities)-1) if i < 9]
        
        signature_data = {
            "resource_sequence": resources,
            "timing_pattern": [round(t, 1) for t in timings]
        }
        
        signature_string = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_string.encode()).hexdigest()
    
    def _check_policy_violations(self, event: SecurityEvent):
        """Check if an event violates any security policies"""
        
        for policy_id, policy in self.security_policies.items():
            for rule in policy.rules:
                if self._evaluate_policy_rule(rule, event):
                    violation_details = {
                        "policy_id": policy_id,
                        "policy_name": policy.name,
                        "rule_id": rule["rule_id"],
                        "rule_description": rule["description"],
                        "event_id": event.event_id,
                        "enforcement_level": policy.enforcement_level.value
                    }
                    
                    # Log compliance violation
                    self.compliance_violations[policy_id].append(violation_details)
                    
                    # Log security event for violation
                    self.log_security_event(
                        event_type=SecurityEventType.COMPLIANCE_VIOLATION,
                        security_level=policy.enforcement_level,
                        source="policy_engine",
                        user_id=event.user_id,
                        session_id=event.session_id,
                        details=violation_details,
                        compliance_implications=[f.value for f in policy.compliance_frameworks]
                    )
                    
                    # Execute violation actions
                    self._execute_violation_actions(policy, rule, event)
    
    def _evaluate_policy_rule(self, rule: Dict[str, Any], event: SecurityEvent) -> bool:
        """Evaluate if an event violates a policy rule (simplified logic)"""
        
        # This is a simplified rule evaluation - in reality, you'd have a more
        # sophisticated rule engine
        
        rule_id = rule["rule_id"]
        event_details = event.details
        
        # Example rule evaluations
        if rule_id == "no_offensive_content":
            return (event_details.get("content_type") == "offensive_tool" or 
                   event_details.get("intent") == "malicious")
        
        elif rule_id == "training_purpose_only":
            purpose = event_details.get("purpose", "training")
            return purpose not in ["training", "education", "research"]
        
        elif rule_id == "no_real_target_systems":
            target_type = event_details.get("target_type")
            return target_type == "production"
        
        elif rule_id == "encrypt_pii":
            return (event_details.get("data_contains_pii") and 
                   not event_details.get("encryption", False))
        
        elif rule_id == "strong_authentication":
            return (event_details.get("operation_sensitivity") == "high" and
                   event_details.get("auth_strength", 0) < 2)
        
        return False
    
    def _execute_violation_actions(self, policy: SecurityPolicy, rule: Dict, event: SecurityEvent):
        """Execute actions for policy violations"""
        
        action = rule.get("action", "log_only")
        
        if action == "block_and_alert":
            self._block_operation(event)
            self._send_alert(event, f"Policy violation: {rule['description']}")
        
        elif action == "block_immediately":
            self._block_operation(event)
            self._emergency_alert(event, f"Critical policy violation: {rule['description']}")
        
        elif action == "review_and_alert":
            self._flag_for_review(event)
            self._send_alert(event, f"Policy review required: {rule['description']}")
        
        elif action == "require_mfa":
            self._require_additional_authentication(event)
        
        elif action == "deny_access":
            self._deny_access(event)
        
        elif action == "terminate_session":
            self._terminate_session(event)
    
    def _block_operation(self, event: SecurityEvent):
        """Block the current operation"""
        if event.session_id in self.active_sessions:
            session = self.active_sessions[event.session_id]
            session["blocked"] = True
            session["block_reason"] = "Policy violation"
            session["block_time"] = datetime.now()
            
        logger.warning(f"Operation blocked for event {event.event_id}")
    
    def _send_alert(self, event: SecurityEvent, message: Optional[str] = None):
        """Send security alert"""
        alert_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "security_level": event.security_level.value,
            "timestamp": event.timestamp.isoformat(),
            "message": message or f"Security event: {event.event_type.value}",
            "details": event.details
        }
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Add to audit log
        self._add_to_audit_log("SECURITY_ALERT", alert_data)
    
    def _emergency_alert(self, event: SecurityEvent, message: str):
        """Send emergency security alert"""
        alert_data = {
            "alert_type": "EMERGENCY",
            "event_id": event.event_id,
            "security_level": "CRITICAL",
            "timestamp": event.timestamp.isoformat(),
            "message": message,
            "immediate_action_required": True,
            "details": event.details
        }
        
        # Emergency notifications would go here (email, SMS, etc.)
        logger.critical(f"EMERGENCY ALERT: {message}")
        
        # Add to audit log
        self._add_to_audit_log("EMERGENCY_ALERT", alert_data)
    
    def _check_compliance_violations(self):
        """Check for ongoing compliance violations"""
        
        current_time = datetime.now()
        
        # Check data retention compliance
        for user_id, activities in self.user_activities.items():
            old_activities = [a for a in activities 
                            if (current_time - a.timestamp).days > 90]  # 90-day retention
            
            if old_activities:
                self.log_security_event(
                    event_type=SecurityEventType.COMPLIANCE_VIOLATION,
                    security_level=SecurityLevel.MEDIUM,
                    source="compliance_check",
                    user_id=user_id,
                    details={
                        "violation_type": "data_retention_exceeded",
                        "old_activities_count": len(old_activities),
                        "compliance_frameworks": ["GDPR", "CCPA"]
                    },
                    compliance_implications=["GDPR Article 5", "CCPA Section 1798.105"]
                )
        
        # Check for inactive sessions (potential security risk)
        for session_id, session in list(self.active_sessions.items()):
            if (current_time - session["start_time"]).total_seconds() > 3600:  # 1 hour
                self.log_security_event(
                    event_type=SecurityEventType.SYSTEM_HEALTH_CHECK,
                    security_level=SecurityLevel.LOW,
                    source="compliance_check",
                    user_id=session["user_id"],
                    session_id=session_id,
                    details={
                        "issue": "inactive_session",
                        "duration_hours": (current_time - session["start_time"]).total_seconds() / 3600
                    }
                )
    
    def _update_risk_scores(self):
        """Update risk scores for all users"""
        
        current_time = datetime.now()
        
        for user_id, activities in self.user_activities.items():
            risk_score = 0.0
            
            # Recent activity volume
            recent_activities = [a for a in activities 
                               if (current_time - a.timestamp).total_seconds() < 3600]  # 1 hour
            
            if len(recent_activities) > 100:
                risk_score += 0.3  # High activity volume
            elif len(recent_activities) > 50:
                risk_score += 0.1  # Moderate activity volume
            
            # Failed attempts
            failed_activities = [a for a in recent_activities if not a.success]
            if failed_activities:
                risk_score += len(failed_activities) * 0.05
            
            # Unusual timing patterns
            if self._has_unusual_timing_pattern(recent_activities):
                risk_score += 0.2
            
            # Resource access diversity
            unique_resources = len(set(a.resource_accessed for a in recent_activities))
            if unique_resources > 20:
                risk_score += 0.15
            
            # Cap risk score at 1.0
            risk_score = min(risk_score, 1.0)
            
            # Update risk score
            self.user_risk_scores[user_id] = risk_score
            
            # Alert on high risk scores
            if risk_score > 0.7:
                self.log_security_event(
                    event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                    security_level=SecurityLevel.HIGH,
                    source="risk_assessment",
                    user_id=user_id,
                    details={
                        "risk_score": risk_score,
                        "risk_factors": self._identify_risk_factors(recent_activities),
                        "recommendation": "Enhanced monitoring and possible account review"
                    }
                )
    
    def _has_unusual_timing_pattern(self, activities: List[UserActivity]) -> bool:
        """Check for unusual timing patterns in user activities"""
        
        if len(activities) < 5:
            return False
        
        # Calculate intervals between activities
        intervals = []
        for i in range(len(activities) - 1):
            interval = (activities[i+1].timestamp - activities[i].timestamp).total_seconds()
            intervals.append(interval)
        
        # Check for very regular intervals (potential bot behavior)
        if len(set(intervals)) == 1 and len(intervals) > 5:
            return True
        
        # Check for extremely rapid intervals
        rapid_intervals = [i for i in intervals if i < 1.0]  # Less than 1 second
        return len(rapid_intervals) > len(intervals) * 0.8  # 80% rapid intervals
    
    def _identify_risk_factors(self, activities: List[UserActivity]) -> List[str]:
        """Identify specific risk factors in user activities"""
        
        risk_factors = []
        
        # High activity volume
        if len(activities) > 100:
            risk_factors.append("high_activity_volume")
        
        # Multiple failed attempts
        failed_count = len([a for a in activities if not a.success])
        if failed_count > 5:
            risk_factors.append("multiple_failed_attempts")
        
        # Broad resource access
        unique_resources = len(set(a.resource_accessed for a in activities))
        if unique_resources > 20:
            risk_factors.append("broad_resource_access")
        
        # Unusual timing
        if self._has_unusual_timing_pattern(activities):
            risk_factors.append("unusual_timing_pattern")
        
        return risk_factors
    
    def _system_health_check(self):
        """Perform system health checks"""
        
        health_status = {
            "monitoring_active": self.monitoring_active,
            "active_sessions": len(self.active_sessions),
            "total_users": len(self.user_activities),
            "events_logged": len(self.security_events),
            "violations_detected": sum(len(v) for v in self.compliance_violations.values()),
            "high_risk_users": len([u for u, score in self.user_risk_scores.items() if score > 0.7])
        }
        
        # Check for system resource issues
        if len(self.security_events) > 9500:  # Near ring buffer limit
            self.log_security_event(
                event_type=SecurityEventType.SYSTEM_HEALTH_CHECK,
                security_level=SecurityLevel.MEDIUM,
                source="system_monitor",
                details={
                    "issue": "event_buffer_near_full",
                    "event_count": len(self.security_events),
                    "recommendation": "Archive old events or increase buffer size"
                }
            )
        
        # Check for excessive violations
        total_violations = sum(len(v) for v in self.compliance_violations.values())
        if total_violations > 100:
            self.log_security_event(
                event_type=SecurityEventType.SYSTEM_HEALTH_CHECK,
                security_level=SecurityLevel.HIGH,
                source="system_monitor",
                details={
                    "issue": "high_violation_count",
                    "violation_count": total_violations,
                    "recommendation": "Review security policies and user training"
                }
            )
        
        self._add_to_audit_log("SYSTEM_HEALTH", health_status)
    
    def _add_to_audit_log(self, action_type: str, data: Dict[str, Any]):
        """Add entry to audit log"""
        
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "data": data,
            "hash": self._calculate_audit_hash(action_type, data)
        }
        
        self.audit_log.append(audit_entry)
    
    def _calculate_audit_hash(self, action_type: str, data: Dict[str, Any]) -> str:
        """Calculate hash for audit log integrity"""
        
        hash_input = f"{action_type}:{json.dumps(data, sort_keys=True)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback function for security alerts"""
        self.alert_callbacks.append(callback)
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data"""
        
        current_time = datetime.now()
        
        # Recent events (last 24 hours)
        recent_events = [
            event for event in self.security_events
            if (current_time - event.timestamp).total_seconds() < 86400
        ]
        
        # Event distribution by type
        event_distribution = {}
        for event in recent_events:
            event_type = event.event_type.value
            if event_type not in event_distribution:
                event_distribution[event_type] = 0
            event_distribution[event_type] += 1
        
        # Security level distribution
        security_level_distribution = {}
        for event in recent_events:
            level = event.security_level.value
            if level not in security_level_distribution:
                security_level_distribution[level] = 0
            security_level_distribution[level] += 1
        
        # Top risk users
        top_risk_users = sorted(
            [(user_id, score) for user_id, score in self.user_risk_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Recent violations
        recent_violations = []
        for policy_id, violations in self.compliance_violations.items():
            recent_violations.extend(violations[-5:])  # Last 5 violations per policy
        
        dashboard = {
            "timestamp": current_time.isoformat(),
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            
            # Event statistics
            "total_events_24h": len(recent_events),
            "event_distribution": event_distribution,
            "security_level_distribution": security_level_distribution,
            
            # User statistics
            "active_sessions": len(self.active_sessions),
            "total_monitored_users": len(self.user_activities),
            "high_risk_users": len([s for s in self.user_risk_scores.values() if s > 0.7]),
            "top_risk_users": top_risk_users,
            
            # Compliance statistics
            "total_violations": sum(len(v) for v in self.compliance_violations.values()),
            "recent_violations": recent_violations,
            "compliance_frameworks": ["GDPR", "CCPA", "FERPA", "ISO27001", "NIST"],
            
            # System health
            "system_health": {
                "events_buffer_usage": f"{len(self.security_events)}/10000",
                "audit_log_usage": f"{len(self.audit_log)}/50000",
                "monitoring_uptime": "active"
            }
        }
        
        return dashboard
    
    def export_audit_log(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict]:
        """Export audit log for compliance reporting"""
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)  # Default to last 30 days
        
        if end_date is None:
            end_date = datetime.now()
        
        filtered_log = []
        for entry in self.audit_log:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if start_date <= entry_time <= end_date:
                filtered_log.append(entry)
        
        return filtered_log
    
    def generate_compliance_report(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        
        current_time = datetime.now()
        
        # Filter violations by framework
        framework_violations = []
        for policy_id, violations in self.compliance_violations.items():
            policy = self.security_policies.get(policy_id)
            if policy and framework in policy.compliance_frameworks:
                framework_violations.extend(violations)
        
        # Calculate compliance metrics
        total_events = len(self.security_events)
        violation_rate = len(framework_violations) / max(total_events, 1)
        
        report = {
            "framework": framework.value,
            "report_date": current_time.isoformat(),
            "reporting_period": "last_30_days",
            
            "compliance_summary": {
                "total_violations": len(framework_violations),
                "violation_rate": round(violation_rate * 100, 2),
                "compliance_score": round((1 - violation_rate) * 100, 2)
            },
            
            "violation_details": framework_violations,
            
            "remediation_actions": self._generate_remediation_actions(framework, framework_violations),
            
            "recommendations": self._generate_compliance_recommendations(framework)
        }
        
        return report
    
    def _generate_remediation_actions(self, framework: ComplianceFramework, violations: List[Dict]) -> List[str]:
        """Generate remediation actions for compliance violations"""
        
        actions = []
        
        if framework == ComplianceFramework.GDPR:
            if violations:
                actions.extend([
                    "Review data processing activities for GDPR compliance",
                    "Ensure user consent mechanisms are properly implemented",
                    "Verify data subject rights are being honored",
                    "Update privacy notices as needed"
                ])
        
        elif framework == ComplianceFramework.CCPA:
            if violations:
                actions.extend([
                    "Review consumer data collection practices",
                    "Ensure opt-out mechanisms are functional",
                    "Verify consumer rights request processes"
                ])
        
        elif framework == ComplianceFramework.FERPA:
            if violations:
                actions.extend([
                    "Review educational record access controls",
                    "Verify student consent procedures",
                    "Ensure proper disclosure limitations"
                ])
        
        return actions
    
    def _generate_compliance_recommendations(self, framework: ComplianceFramework) -> List[str]:
        """Generate compliance recommendations"""
        
        recommendations = []
        
        # General recommendations
        recommendations.extend([
            "Conduct regular compliance audits",
            "Provide staff training on compliance requirements",
            "Implement automated compliance monitoring",
            "Maintain comprehensive documentation"
        ])
        
        # Framework-specific recommendations
        if framework == ComplianceFramework.GDPR:
            recommendations.extend([
                "Implement privacy by design principles",
                "Regular data protection impact assessments",
                "Maintain records of processing activities"
            ])
        
        elif framework == ComplianceFramework.ISO27001:
            recommendations.extend([
                "Implement comprehensive information security management system",
                "Regular risk assessments and security reviews",
                "Incident response procedure updates"
            ])
        
        return recommendations
    
    # Placeholder methods for actions (would be implemented based on actual system architecture)
    def _flag_for_review(self, event: SecurityEvent):
        logger.info(f"Event {event.event_id} flagged for review")
    
    def _require_additional_authentication(self, event: SecurityEvent):
        logger.info(f"Additional authentication required for event {event.event_id}")
    
    def _deny_access(self, event: SecurityEvent):
        logger.info(f"Access denied for event {event.event_id}")
    
    def _terminate_session(self, event: SecurityEvent):
        if event.session_id in self.active_sessions:
            del self.active_sessions[event.session_id]
        logger.info(f"Session terminated for event {event.event_id}")


def create_security_monitor(config_path: Optional[Path] = None) -> DefensiveSecurityMonitor:
    """Factory function to create a DefensiveSecurityMonitor instance"""
    return DefensiveSecurityMonitor(config_path)


# Example usage and testing
if __name__ == "__main__":
    # Initialize security monitor
    monitor = create_security_monitor()
    
    # Register alert callback
    def alert_handler(alert_data):
        print(f"SECURITY ALERT: {alert_data['message']}")
        print(f"Level: {alert_data['security_level']}")
        print(f"Details: {alert_data['details']}")
    
    monitor.register_alert_callback(alert_handler)
    
    # Simulate some activities
    print("Simulating normal training activity...")
    
    # Normal training session
    monitor.log_security_event(
        event_type=SecurityEventType.TRAINING_SESSION_START,
        security_level=SecurityLevel.INFO,
        source="training_platform",
        user_id="learner_001",
        session_id="session_123",
        details={
            "purpose": "training",
            "content_type": "defensive_training",
            "scenario": "GAN detection lab"
        }
    )
    
    # Track some user activities
    for i in range(5):
        monitor.track_user_activity(
            user_id="learner_001",
            session_id="session_123",
            activity_type="content_access",
            resource_accessed=f"training_module_{i+1}",
            duration=300.0,  # 5 minutes
            success=True
        )
        time.sleep(1)  # Small delay between activities
    
    print("\nSimulating suspicious activity...")
    
    # Simulate suspicious rapid access pattern
    for i in range(25):
        monitor.track_user_activity(
            user_id="learner_002",
            session_id="session_456", 
            activity_type="rapid_access",
            resource_accessed=f"resource_{i}",
            duration=1.0,
            success=True
        )
    
    # Simulate policy violation
    monitor.log_security_event(
        event_type=SecurityEventType.CONTENT_ACCESS,
        security_level=SecurityLevel.MEDIUM,
        source="content_system",
        user_id="learner_003",
        session_id="session_789",
        details={
            "purpose": "malicious",  # This will trigger policy violation
            "content_type": "offensive_tool",
            "target_type": "production"
        }
    )
    
    time.sleep(2)  # Allow monitoring to process
    
    # Get security dashboard
    print("\nSecurity Dashboard:")
    dashboard = monitor.get_security_dashboard()
    print(f"Total events (24h): {dashboard['total_events_24h']}")
    print(f"Active sessions: {dashboard['active_sessions']}")
    print(f"High risk users: {dashboard['high_risk_users']}")
    print(f"Total violations: {dashboard['total_violations']}")
    
    # Generate compliance report
    print("\nGDPR Compliance Report:")
    gdpr_report = monitor.generate_compliance_report(ComplianceFramework.GDPR)
    print(f"Compliance Score: {gdpr_report['compliance_summary']['compliance_score']}%")
    print(f"Violations: {gdpr_report['compliance_summary']['total_violations']}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("\nMonitoring stopped.")