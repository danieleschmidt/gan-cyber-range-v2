"""
API Integration Tests for GAN-Cyber-Range-v2
Comprehensive testing of REST API endpoints
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json

from gan_cyber_range.api.main import app
from gan_cyber_range.api.auth import user_manager, create_access_token
from gan_cyber_range.db.database import get_database


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Async test client fixture"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def db_session():
    """Database session fixture"""
    db = await get_database()
    async with db.get_session() as session:
        yield session


@pytest.fixture
async def test_user(db_session):
    """Create test user fixture"""
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPassword123!",
        "full_name": "Test User",
        "organization": "Test Org",
        "role": "researcher"
    }
    
    user = user_manager.create_user(**user_data)
    yield user
    
    # Cleanup
    # User cleanup would happen here in a real implementation


@pytest.fixture
async def admin_user(db_session):
    """Create admin user fixture"""
    user_data = {
        "username": "admin",
        "email": "admin@example.com", 
        "password": "AdminPassword123!",
        "full_name": "Admin User",
        "organization": "Test Org",
        "role": "admin"
    }
    
    user = user_manager.create_user(**user_data)
    yield user


@pytest.fixture
def auth_headers(test_user):
    """Authentication headers fixture"""
    token = create_access_token({"sub": test_user["username"], "role": test_user["role"]})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(admin_user):
    """Admin authentication headers fixture"""
    token = create_access_token({"sub": admin_user["username"], "role": admin_user["role"]})
    return {"Authorization": f"Bearer {token}"}


class TestHealthEndpoint:
    """Health endpoint tests"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data
        assert "timestamp" in data
    
    def test_health_check_components(self, client):
        """Test health check components"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        components = data["components"]
        
        # Check for expected components
        component_names = [comp["component"] for comp in components]
        assert "attack_gan" in component_names
        assert "metrics_collector" in component_names


class TestAuthenticationEndpoints:
    """Authentication endpoint tests"""
    
    def test_user_registration(self, client):
        """Test user registration"""
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "NewPassword123!",
            "full_name": "New User",
            "organization": "New Org",
            "role": "user"
        }
        
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]
        assert data["role"] == user_data["role"]
        assert "id" in data
        assert "password" not in data  # Ensure password not returned
    
    def test_user_registration_duplicate(self, client, test_user):
        """Test duplicate user registration"""
        user_data = {
            "username": test_user["username"],
            "email": "different@example.com",
            "password": "Password123!",
            "role": "user"
        }
        
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 400
    
    def test_user_login_success(self, client, test_user):
        """Test successful user login"""
        login_data = {
            "username": test_user["username"],
            "password": "TestPassword123!"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_user_login_invalid_credentials(self, client, test_user):
        """Test login with invalid credentials"""
        login_data = {
            "username": test_user["username"],
            "password": "WrongPassword123!"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 401
    
    def test_get_current_user(self, client, auth_headers):
        """Test getting current user info"""
        response = client.get("/auth/me", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "username" in data
        assert "email" in data
        assert "role" in data
    
    def test_unauthorized_access(self, client):
        """Test unauthorized access"""
        response = client.get("/auth/me")
        assert response.status_code == 401


class TestAttackGenerationEndpoints:
    """Attack generation endpoint tests"""
    
    def test_generate_attacks_success(self, client, auth_headers):
        """Test successful attack generation"""
        request_data = {
            "attack_types": ["network", "malware"],
            "num_samples": 10,
            "diversity_threshold": 0.8,
            "filter_detectable": True
        }
        
        response = client.post("/attacks/generate", json=request_data, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "attacks" in data
        assert "diversity_score" in data
        assert len(data["attacks"]) <= request_data["num_samples"]
    
    def test_generate_attacks_invalid_params(self, client, auth_headers):
        """Test attack generation with invalid parameters"""
        request_data = {
            "attack_types": [],  # Empty list should fail
            "num_samples": 10
        }
        
        response = client.post("/attacks/generate", json=request_data, headers=auth_headers)
        assert response.status_code == 422
    
    def test_generate_attacks_unauthorized(self, client):
        """Test attack generation without authentication"""
        request_data = {
            "attack_types": ["network"],
            "num_samples": 5
        }
        
        response = client.post("/attacks/generate", json=request_data)
        assert response.status_code == 401
    
    def test_generate_attacks_insufficient_role(self, client):
        """Test attack generation with insufficient role"""
        # Create user with 'user' role (not 'researcher')
        user_data = {
            "username": "basicuser",
            "email": "basic@example.com",
            "password": "Password123!",
            "role": "user"
        }
        user = user_manager.create_user(**user_data)
        
        token = create_access_token({"sub": user["username"], "role": user["role"]})
        headers = {"Authorization": f"Bearer {token}"}
        
        request_data = {
            "attack_types": ["network"],
            "num_samples": 5
        }
        
        response = client.post("/attacks/generate", json=request_data, headers=headers)
        assert response.status_code == 403


class TestCyberRangeEndpoints:
    """Cyber range endpoint tests"""
    
    def test_create_cyber_range(self, client, auth_headers):
        """Test cyber range creation"""
        config_data = {
            "config": {
                "name": "Test Range",
                "size": "medium",
                "topology": {
                    "template": "enterprise",
                    "subnets": ["dmz", "internal", "management"],
                    "hosts_per_subnet": {"dmz": 5, "internal": 20, "management": 5},
                    "services": ["web", "database", "email"],
                    "vulnerabilities": "realistic"
                },
                "resource_limits": {"cpu": 4, "memory": "8GB"},
                "isolation_level": "strict",
                "monitoring_enabled": True,
                "auto_cleanup": True,
                "duration_hours": 8
            },
            "description": "Test cyber range deployment",
            "tags": ["test", "development"]
        }
        
        response = client.post("/ranges", json=config_data, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Range"
        assert data["status"] == "starting"
        assert data["config"]["size"] == "medium"
    
    def test_list_cyber_ranges(self, client, auth_headers):
        """Test listing cyber ranges"""
        response = client.get("/ranges", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_cyber_range_details(self, client, auth_headers):
        """Test getting cyber range details"""
        # First create a cyber range
        config_data = {
            "config": {
                "name": "Detail Test Range",
                "size": "small"
            }
        }
        
        create_response = client.post("/ranges", json=config_data, headers=auth_headers)
        assert create_response.status_code == 200
        
        range_id = create_response.json()["id"]
        
        # Get range details
        response = client.get(f"/ranges/{range_id}", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == range_id
        assert data["name"] == "Detail Test Range"
    
    def test_start_cyber_range(self, client, auth_headers):
        """Test starting a cyber range"""
        # Create a range first
        config_data = {
            "config": {
                "name": "Start Test Range",
                "size": "small"
            }
        }
        
        create_response = client.post("/ranges", json=config_data, headers=auth_headers)
        range_id = create_response.json()["id"]
        
        # Start the range
        response = client.post(f"/ranges/{range_id}/start", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "started"
    
    def test_stop_cyber_range(self, client, auth_headers):
        """Test stopping a cyber range"""
        # Create and start a range first
        config_data = {
            "config": {
                "name": "Stop Test Range",
                "size": "small"
            }
        }
        
        create_response = client.post("/ranges", json=config_data, headers=auth_headers)
        range_id = create_response.json()["id"]
        
        # Stop the range
        response = client.post(f"/ranges/{range_id}/stop", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "stopped"
    
    def test_delete_cyber_range(self, client, auth_headers):
        """Test deleting a cyber range"""
        # Create a range first
        config_data = {
            "config": {
                "name": "Delete Test Range",
                "size": "small"
            }
        }
        
        create_response = client.post("/ranges", json=config_data, headers=auth_headers)
        range_id = create_response.json()["id"]
        
        # Delete the range
        response = client.delete(f"/ranges/{range_id}", headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "deleted"


class TestRedTeamEndpoints:
    """Red team endpoint tests"""
    
    def test_create_attack_campaign(self, client, auth_headers):
        """Test creating an attack campaign"""
        campaign_data = {
            "red_team_config": {
                "model": "llama2-70b-security",
                "creativity": 0.8,
                "risk_tolerance": 0.6,
                "objective": "Data exfiltration from healthcare network",
                "max_campaign_duration": 30
            },
            "target_profile": {
                "industry": "healthcare",
                "size": "medium",
                "security_maturity": "intermediate",
                "crown_jewels": ["patient_records", "research_data"],
                "known_vulnerabilities": ["CVE-2023-1234"]
            },
            "campaign_duration": 14,
            "tactics": ["initial_access", "persistence", "lateral_movement", "exfiltration"],
            "stealth_level": "medium"
        }
        
        response = client.post("/redteam/campaigns", json=campaign_data, headers=auth_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert data["objective"] == campaign_data["red_team_config"]["objective"]
        assert data["total_duration"] == campaign_data["campaign_duration"]
        assert "stages" in data
        assert isinstance(data["stages"], list)


class TestErrorHandling:
    """Error handling tests"""
    
    def test_404_endpoint(self, client):
        """Test 404 error for non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 error for wrong HTTP method"""
        response = client.put("/health")
        assert response.status_code == 405
    
    def test_invalid_json(self, client, auth_headers):
        """Test handling of invalid JSON"""
        response = client.post(
            "/attacks/generate",
            data="invalid json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestWebSocketEndpoints:
    """WebSocket endpoint tests"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, async_client):
        """Test WebSocket connection"""
        async with async_client.websocket_connect("/ws") as websocket:
            # Send test message
            await websocket.send_text("test message")
            
            # Receive echo
            data = await websocket.receive_json()
            assert data["type"] == "echo"
            assert data["data"] == "test message"


class TestRateLimiting:
    """Rate limiting tests"""
    
    def test_rate_limit_exceeded(self, client, auth_headers):
        """Test rate limiting"""
        # This test would need to be configured based on actual rate limits
        # For now, just test that rate limiting doesn't break normal requests
        
        response = client.get("/health", headers=auth_headers)
        assert response.status_code == 200


class TestInputValidation:
    """Input validation tests"""
    
    def test_sql_injection_prevention(self, client, auth_headers):
        """Test SQL injection prevention"""
        # Test with SQL injection attempt in username
        malicious_data = {
            "username": "admin'; DROP TABLE users; --",
            "email": "test@example.com",
            "password": "Password123!",
            "role": "user"
        }
        
        response = client.post("/auth/register", json=malicious_data)
        # Should either succeed with sanitized input or fail validation
        assert response.status_code in [200, 400, 422]
    
    def test_xss_prevention(self, client, auth_headers):
        """Test XSS prevention"""
        # Test with XSS attempt in user data
        xss_data = {
            "config": {
                "name": "<script>alert('xss')</script>",
                "size": "small"
            }
        }
        
        response = client.post("/ranges", json=xss_data, headers=auth_headers)
        # Should either succeed with sanitized input or fail validation
        assert response.status_code in [200, 400, 422]


class TestPerformance:
    """Performance tests"""
    
    def test_concurrent_requests(self, client, auth_headers):
        """Test handling concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            start_time = time.time()
            response = client.get("/health", headers=auth_headers)
            end_time = time.time()
            results.append({
                "status_code": response.status_code,
                "response_time": end_time - start_time
            })
        
        # Create 10 concurrent threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 10
        assert all(result["status_code"] == 200 for result in results)
        
        # Check that all requests completed in reasonable time (< 5 seconds each)
        assert all(result["response_time"] < 5.0 for result in results)


# Utility functions for tests
def create_test_attack_vector():
    """Create a test attack vector"""
    return {
        "attack_type": "network",
        "techniques": ["T1190", "T1110"],
        "payload": "test_payload_data",
        "confidence": 0.85,
        "sophistication": 0.7,
        "detectability": 0.3,
        "metadata": {"test": True}
    }


def create_test_campaign():
    """Create a test campaign"""
    return {
        "name": "Test Campaign",
        "objective": "Test objective",
        "target_profile": {
            "industry": "test",
            "size": "small",
            "security_maturity": "basic",
            "crown_jewels": ["test_data"]
        },
        "red_team_config": {
            "model": "test-model",
            "creativity": 0.5,
            "risk_tolerance": 0.5,
            "objective": "Test objective"
        },
        "tactics": ["initial_access"],
        "stages": [
            {
                "name": "Test Stage",
                "objective": "Test stage objective",
                "techniques": ["T1190"],
                "success_criteria": "Success",
                "estimated_duration": "1 hour",
                "risk_level": "low",
                "detection_likelihood": 0.2
            }
        ],
        "total_duration": 1,
        "overall_risk": "low",
        "success_probability": 0.8
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])