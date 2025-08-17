"""
Comprehensive test suite for the security framework module.

This module tests all security capabilities including security orchestration,
threat detection, access control, and compliance features.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict
import threading
import time

from gan_cyber_range.security.security_orchestrator import (
    SecurityOrchestrator, SecurityPolicy, SecurityContext, SecurityEvent,
    SecurityLevel, ThreatLevel, requires_security_context, requires_permission
)


@pytest.fixture
def temp_security_dir():
    """Create temporary directory for security files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def security_orchestrator(temp_security_dir):
    """Create security orchestrator instance"""
    config_path = temp_security_dir / "security_config.json"
    return SecurityOrchestrator(config_path=config_path)


@pytest.fixture
def sample_security_policy():
    """Create sample security policy"""
    return SecurityPolicy(
        name="test_policy",
        description="Test security policy",
        security_level=SecurityLevel.HIGH,
        encryption_required=True,
        audit_logging_required=True,
        compliance_frameworks=["GDPR", "SOC2"],
        allowed_operations={"read", "create", "update"},
        blocked_operations={"delete_all", "admin_access"},
        session_timeout=1800,
        max_failed_attempts=3
    )


@pytest.fixture
def sample_security_context():
    """Create sample security context"""
    return SecurityContext(
        user_id="test_user",
        session_id="test_session_123",
        security_level=SecurityLevel.MEDIUM,
        permissions={"read", "create"},
        ip_address="192.168.1.100",
        user_agent="Test Agent",
        created_at=datetime.now(),
        last_activity=datetime.now()
    )


class TestSecurityPolicy:
    """Test cases for SecurityPolicy"""
    
    def test_security_policy_creation(self, sample_security_policy):
        """Test creating a security policy"""
        policy = sample_security_policy
        
        assert policy.name == "test_policy"
        assert policy.security_level == SecurityLevel.HIGH
        assert policy.encryption_required == True
        assert policy.session_timeout == 1800
        assert "read" in policy.allowed_operations
        assert "delete_all" in policy.blocked_operations
    
    def test_security_policy_serialization(self, sample_security_policy):
        """Test security policy serialization"""
        policy = sample_security_policy
        policy_dict = asdict(policy)
        
        assert "name" in policy_dict
        assert "security_level" in policy_dict
        assert "compliance_frameworks" in policy_dict
        
        # Test that enums are properly handled
        assert policy_dict["security_level"] == SecurityLevel.HIGH


class TestSecurityContext:
    """Test cases for SecurityContext"""
    
    def test_security_context_creation(self, sample_security_context):
        """Test creating a security context"""
        context = sample_security_context
        
        assert context.user_id == "test_user"
        assert context.session_id == "test_session_123"
        assert context.security_level == SecurityLevel.MEDIUM
        assert "read" in context.permissions
        assert context.ip_address == "192.168.1.100"
        assert isinstance(context.created_at, datetime)
    
    def test_security_context_threat_score(self, sample_security_context):
        """Test threat score tracking"""
        context = sample_security_context
        
        assert context.threat_score == 0.0
        
        # Simulate threat score increase
        context.threat_score = 0.6
        assert context.threat_score == 0.6


class TestSecurityEvent:
    """Test cases for SecurityEvent"""
    
    def test_security_event_creation(self):
        """Test creating a security event"""
        event = SecurityEvent(
            event_id="test_event_123",
            event_type="access_denied",
            threat_level=ThreatLevel.MEDIUM,
            source="security_orchestrator",
            target="test_user",
            description="Access denied for test operation",
            details={"reason": "insufficient_permissions"}
        )
        
        assert event.event_id == "test_event_123"
        assert event.threat_level == ThreatLevel.MEDIUM
        assert event.resolved == False
        assert "reason" in event.details
        assert isinstance(event.timestamp, datetime)


class TestSecurityOrchestrator:
    """Test cases for SecurityOrchestrator"""
    
    def test_initialization(self, security_orchestrator):
        """Test security orchestrator initialization"""
        orchestrator = security_orchestrator
        
        assert orchestrator is not None
        assert len(orchestrator.active_policies) >= 0
        assert len(orchestrator.security_contexts) == 0
        assert len(orchestrator.security_events) == 0
        assert isinstance(orchestrator.failed_attempts, dict)
        assert isinstance(orchestrator.blocked_ips, set)
    
    def test_register_security_policy(self, security_orchestrator, sample_security_policy):
        """Test registering a security policy"""
        orchestrator = security_orchestrator
        policy = sample_security_policy
        
        orchestrator.register_security_policy(policy)
        
        assert policy.name in orchestrator.active_policies
        assert orchestrator.active_policies[policy.name] == policy
        
        # Check that an event was recorded
        assert len(orchestrator.security_events) > 0
        event = orchestrator.security_events[-1]
        assert event.event_type == "policy_registered"
        assert event.target == policy.name
    
    def test_create_security_context(self, security_orchestrator):
        """Test creating a security context"""
        orchestrator = security_orchestrator
        
        context = orchestrator.create_security_context(
            user_id="test_user",
            ip_address="192.168.1.100",
            user_agent="Test Agent",
            requested_permissions={"read", "create"}
        )
        
        assert context is not None
        assert context.user_id == "test_user"
        assert context.ip_address == "192.168.1.100"
        assert "read" in context.permissions
        assert context.session_id in orchestrator.security_contexts
        assert context.session_id in orchestrator.active_sessions
    
    def test_create_security_context_blocked_ip(self, security_orchestrator):
        """Test creating security context with blocked IP"""
        orchestrator = security_orchestrator
        
        # Block IP address
        blocked_ip = "192.168.1.999"
        orchestrator.blocked_ips.add(blocked_ip)
        
        context = orchestrator.create_security_context(
            user_id="test_user",
            ip_address=blocked_ip,
            user_agent="Test Agent",
            requested_permissions={"read"}
        )
        
        assert context is None
        
        # Check that access denied event was recorded
        denied_events = [e for e in orchestrator.security_events if e.event_type == "access_denied"]
        assert len(denied_events) > 0
    
    def test_create_security_context_too_many_failures(self, security_orchestrator):
        """Test creating security context with too many failed attempts"""
        orchestrator = security_orchestrator
        
        # Simulate failed attempts
        user_id = "failing_user"
        orchestrator.failed_attempts[user_id] = 5  # Exceeds default limit of 3
        
        context = orchestrator.create_security_context(
            user_id=user_id,
            ip_address="192.168.1.100",
            user_agent="Test Agent",
            requested_permissions={"read"}
        )
        
        assert context is None
    
    def test_validate_operation_success(self, security_orchestrator, sample_security_policy):
        """Test successful operation validation"""
        orchestrator = security_orchestrator
        
        # Register policy and create context
        orchestrator.register_security_policy(sample_security_policy)
        context = orchestrator.create_security_context(
            user_id="test_user",
            ip_address="192.168.1.100",
            user_agent="Test Agent",
            requested_permissions={"read", "create"}
        )
        
        # Validate allowed operation
        result = orchestrator.validate_operation(
            session_id=context.session_id,
            operation="read",
            target="test_resource"
        )
        
        assert result == True
        
        # Check that authorized event was recorded
        auth_events = [e for e in orchestrator.security_events if e.event_type == "operation_authorized"]
        assert len(auth_events) > 0
    
    def test_validate_operation_blocked(self, security_orchestrator, sample_security_policy):
        """Test blocked operation validation"""
        orchestrator = security_orchestrator
        
        # Register policy and create context
        orchestrator.register_security_policy(sample_security_policy)
        context = orchestrator.create_security_context(
            user_id="test_user",
            ip_address="192.168.1.100",
            user_agent="Test Agent",
            requested_permissions={"read", "create"}
        )
        
        # Try blocked operation
        result = orchestrator.validate_operation(
            session_id=context.session_id,
            operation="delete_all",  # This is in blocked_operations
            target="test_resource"
        )
        
        assert result == False
        
        # Check that access denied event was recorded
        denied_events = [e for e in orchestrator.security_events if e.event_type == "access_denied"]
        assert len(denied_events) > 0
    
    def test_validate_operation_invalid_session(self, security_orchestrator):
        """Test operation validation with invalid session"""
        orchestrator = security_orchestrator
        
        result = orchestrator.validate_operation(
            session_id="invalid_session",
            operation="read",
            target="test_resource"
        )
        
        assert result == False
    
    def test_encrypt_decrypt_data(self, security_orchestrator):
        """Test data encryption and decryption"""
        orchestrator = security_orchestrator
        
        original_data = "sensitive information"
        
        # Encrypt data
        encrypted_data = orchestrator.encrypt_sensitive_data(original_data)
        assert encrypted_data != original_data
        assert len(encrypted_data) > 0
        
        # Decrypt data
        decrypted_data = orchestrator.decrypt_sensitive_data(encrypted_data)
        assert decrypted_data == original_data
    
    def test_decrypt_invalid_data(self, security_orchestrator):
        """Test decrypting invalid data"""
        orchestrator = security_orchestrator
        
        with pytest.raises(ValueError, match="Invalid encrypted data"):
            orchestrator.decrypt_sensitive_data("invalid_encrypted_data")
    
    def test_respond_to_threat(self, security_orchestrator):
        """Test threat response"""
        orchestrator = security_orchestrator
        
        # Create a threat event
        threat_event = SecurityEvent(
            event_id="threat_123",
            event_type="threat_detected",
            threat_level=ThreatLevel.HIGH,
            source="threat_detector",
            target="test_user",
            description="High threat score detected",
            details={"threat_score": 0.9},
            user_id="test_user",
            session_id="test_session",
            ip_address="192.168.1.100"
        )
        
        # Create a session to block
        context = orchestrator.create_security_context(
            user_id="test_user",
            ip_address="192.168.1.100",
            user_agent="Test Agent",
            requested_permissions={"read"}
        )
        threat_event.session_id = context.session_id
        
        # Respond to threat
        response_actions = ["block_ip", "invalidate_session"]
        orchestrator.respond_to_threat(threat_event, response_actions)
        
        # Check that threat was marked as resolved
        assert threat_event.resolved == True
        assert threat_event.response_actions == response_actions
        
        # Check that IP was blocked
        assert "192.168.1.100" in orchestrator.blocked_ips
        
        # Check that session was invalidated
        assert context.session_id not in orchestrator.active_sessions
    
    def test_generate_security_report(self, security_orchestrator, sample_security_policy):
        """Test generating security report"""
        orchestrator = security_orchestrator
        
        # Set up some data
        orchestrator.register_security_policy(sample_security_policy)
        context = orchestrator.create_security_context(
            user_id="test_user",
            ip_address="192.168.1.100",
            user_agent="Test Agent",
            requested_permissions={"read"}
        )
        
        # Generate report
        report = orchestrator.generate_security_report()
        
        assert "generated_at" in report
        assert "summary" in report
        assert "event_breakdown" in report
        assert "threat_level_distribution" in report
        assert "policy_summary" in report
        assert "security_recommendations" in report
        
        # Check summary statistics
        summary = report["summary"]
        assert "total_events_24h" in summary
        assert "active_sessions" in summary
        assert "blocked_ips" in summary
        assert "active_policies" in summary
        
        assert summary["active_sessions"] >= 1  # We created one session
        assert summary["active_policies"] >= 1  # We registered one policy
    
    def test_cleanup_expired_sessions(self, security_orchestrator):
        """Test cleaning up expired sessions"""
        orchestrator = security_orchestrator
        
        # Create a context and make it expired
        context = orchestrator.create_security_context(
            user_id="test_user",
            ip_address="192.168.1.100",
            user_agent="Test Agent",
            requested_permissions={"read"}
        )
        
        # Manually expire the session
        context.last_activity = datetime.now() - timedelta(hours=2)  # 2 hours ago
        
        # Cleanup expired sessions
        cleaned_count = orchestrator.cleanup_expired_sessions()
        
        assert cleaned_count >= 1
        assert context.session_id not in orchestrator.active_sessions
    
    def test_session_timeout_validation(self, security_orchestrator, sample_security_policy):
        """Test session timeout during operation validation"""
        orchestrator = security_orchestrator
        
        # Use policy with short timeout for testing
        policy = sample_security_policy
        policy.session_timeout = 1  # 1 second timeout
        orchestrator.register_security_policy(policy)
        
        # Create context
        context = orchestrator.create_security_context(
            user_id="test_user",
            ip_address="192.168.1.100",
            user_agent="Test Agent",
            requested_permissions={"read"}
        )
        
        # Wait for session to expire
        time.sleep(2)
        
        # Try to validate operation - should fail due to timeout
        result = orchestrator.validate_operation(
            session_id=context.session_id,
            operation="read",
            target="test_resource"
        )
        
        assert result == False
        assert context.session_id not in orchestrator.active_sessions


class TestSecurityDecorators:
    """Test cases for security decorators"""
    
    def test_requires_security_context_decorator(self, security_orchestrator):
        """Test requires_security_context decorator"""
        
        class TestService:
            def __init__(self):
                self.security_orchestrator = security_orchestrator
            
            @requires_security_context
            def protected_method(self, session_id=None):
                return "success"
        
        service = TestService()
        
        # Test without valid session
        with pytest.raises(PermissionError, match="Valid security context required"):
            service.protected_method()
        
        # Test with valid session
        context = security_orchestrator.create_security_context(
            user_id="test_user",
            ip_address="192.168.1.100",
            user_agent="Test Agent",
            requested_permissions={"read"}
        )
        
        result = service.protected_method(session_id=context.session_id)
        assert result == "success"
    
    def test_requires_permission_decorator(self, security_orchestrator):
        """Test requires_permission decorator"""
        
        class TestService:
            def __init__(self):
                self.security_orchestrator = security_orchestrator
            
            @requires_permission("admin")
            def admin_method(self, session_id=None):
                return "admin_success"
        
        service = TestService()
        
        # Test without admin permission
        context = security_orchestrator.create_security_context(
            user_id="test_user",
            ip_address="192.168.1.100",
            user_agent="Test Agent",
            requested_permissions={"read", "create"}  # No admin permission
        )
        
        with pytest.raises(PermissionError, match="Permission 'admin' required"):
            service.admin_method(session_id=context.session_id)
        
        # Test with admin permission
        admin_context = security_orchestrator.create_security_context(
            user_id="admin_user",
            ip_address="192.168.1.100",
            user_agent="Test Agent",
            requested_permissions={"admin"}
        )
        
        result = service.admin_method(session_id=admin_context.session_id)
        assert result == "admin_success"


class TestSecurityIntegration:
    """Integration tests for the complete security framework"""
    
    def test_complete_security_workflow(self, security_orchestrator, sample_security_policy):
        """Test complete security workflow"""
        orchestrator = security_orchestrator
        
        # 1. Register security policy
        orchestrator.register_security_policy(sample_security_policy)
        
        # 2. Create user session
        context = orchestrator.create_security_context(
            user_id="workflow_user",
            ip_address="192.168.1.200",
            user_agent="Workflow Test Agent",
            requested_permissions={"read", "create", "update"}
        )
        
        assert context is not None
        
        # 3. Perform allowed operations
        operations = ["read", "create", "update"]
        for operation in operations:
            result = orchestrator.validate_operation(
                session_id=context.session_id,
                operation=operation,
                target="test_resource"
            )
            assert result == True
        
        # 4. Try blocked operation
        blocked_result = orchestrator.validate_operation(
            session_id=context.session_id,
            operation="delete_all",
            target="test_resource"
        )
        assert blocked_result == False
        
        # 5. Simulate threat detection
        threat_event = SecurityEvent(
            event_id="workflow_threat",
            event_type="threat_detected",
            threat_level=ThreatLevel.HIGH,
            source="threat_detector",
            target="workflow_user",
            description="Suspicious activity detected",
            details={"threat_score": 0.85},
            user_id="workflow_user",
            session_id=context.session_id,
            ip_address="192.168.1.200"
        )
        
        # 6. Respond to threat
        orchestrator.respond_to_threat(threat_event, ["invalidate_session", "block_ip"])
        
        # 7. Verify threat response
        assert threat_event.resolved == True
        assert context.session_id not in orchestrator.active_sessions
        assert "192.168.1.200" in orchestrator.blocked_ips
        
        # 8. Generate security report
        report = orchestrator.generate_security_report()
        
        assert report["summary"]["active_policies"] >= 1
        assert report["summary"]["blocked_ips"] >= 1
        
        # 9. Verify events were recorded
        event_types = [e.event_type for e in orchestrator.security_events]
        assert "policy_registered" in event_types
        assert "session_created" in event_types
        assert "operation_authorized" in event_types
        assert "access_denied" in event_types
        assert "session_invalidated" in event_types
    
    def test_concurrent_security_operations(self, security_orchestrator, sample_security_policy):
        """Test concurrent security operations"""
        orchestrator = security_orchestrator
        orchestrator.register_security_policy(sample_security_policy)
        
        def create_and_validate_session(user_id: str):
            context = orchestrator.create_security_context(
                user_id=user_id,
                ip_address=f"192.168.1.{hash(user_id) % 255}",
                user_agent="Concurrent Test Agent",
                requested_permissions={"read", "create"}
            )
            
            if context:
                # Perform multiple operations
                for i in range(10):
                    orchestrator.validate_operation(
                        session_id=context.session_id,
                        operation="read",
                        target=f"resource_{i}"
                    )
            
            return context is not None
        
        # Create multiple threads
        threads = []
        results = []
        
        for i in range(5):
            thread = threading.Thread(
                target=lambda user_id=f"user_{i}": results.append(create_and_validate_session(user_id))
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 5
        assert all(results)  # All operations should succeed
        
        # Verify that events were recorded from all threads
        assert len(orchestrator.security_events) > 0
        
        # Verify that multiple sessions were created
        assert len(orchestrator.active_sessions) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])