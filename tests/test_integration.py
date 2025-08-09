"""
Integration Tests for GAN-Cyber-Range-v2
End-to-end testing of system components working together
"""

import pytest
import asyncio
import uuid
import time
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from gan_cyber_range.api.main import app
from gan_cyber_range.api.auth import user_manager, create_access_token
from gan_cyber_range.core.attack_gan import AttackGAN
from gan_cyber_range.core.cyber_range import CyberRange
from gan_cyber_range.red_team.llm_adversary import RedTeamLLM
from gan_cyber_range.db.database import get_database
from gan_cyber_range.utils.monitoring import MetricsCollector


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
async def db():
    """Database fixture"""
    return await get_database()


@pytest.fixture
async def researcher_user():
    """Create researcher user"""
    user_data = {
        "username": "integration_researcher",
        "email": "researcher@integration.test",
        "password": "ResearchPass123!",
        "role": "researcher"
    }
    return user_manager.create_user(**user_data)


@pytest.fixture
async def admin_user():
    """Create admin user"""
    user_data = {
        "username": "integration_admin",
        "email": "admin@integration.test",
        "password": "AdminPass123!",
        "role": "admin"
    }
    return user_manager.create_user(**user_data)


@pytest.fixture
def researcher_headers(researcher_user):
    """Researcher authentication headers"""
    token = create_access_token({"sub": researcher_user["username"], "role": researcher_user["role"]})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(admin_user):
    """Admin authentication headers"""
    token = create_access_token({"sub": admin_user["username"], "role": admin_user["role"]})
    return {"Authorization": f"Bearer {token}"}


class TestFullWorkflow:
    """Test complete workflows end-to-end"""
    
    def test_complete_attack_generation_workflow(self, client, researcher_headers):
        """Test complete attack generation workflow"""
        # Step 1: Generate attacks
        attack_request = {
            "attack_types": ["network", "malware"],
            "num_samples": 5,
            "diversity_threshold": 0.8,
            "filter_detectable": True
        }
        
        response = client.post("/attacks/generate", json=attack_request, headers=researcher_headers)
        assert response.status_code == 200
        
        attack_data = response.json()
        assert "job_id" in attack_data
        assert len(attack_data["attacks"]) > 0
        
        # Step 2: Use generated attacks in a campaign
        campaign_request = {
            "red_team_config": {
                "model": "llama2-70b-security",
                "creativity": 0.8,
                "risk_tolerance": 0.6,
                "objective": "Test attack execution workflow"
            },
            "target_profile": {
                "industry": "testing",
                "size": "small",
                "security_maturity": "basic",
                "crown_jewels": ["test_data"]
            },
            "campaign_duration": 1,
            "tactics": ["initial_access", "persistence"],
            "stealth_level": "low"
        }
        
        response = client.post("/redteam/campaigns", json=campaign_request, headers=researcher_headers)
        assert response.status_code == 200
        
        campaign_data = response.json()
        campaign_id = campaign_data["id"]
        assert len(campaign_data["stages"]) > 0
        
        # Step 3: Create cyber range for execution
        range_request = {
            "config": {
                "name": "Integration Test Range",
                "size": "small",
                "topology": {
                    "template": "small_office",
                    "subnets": ["internal"],
                    "hosts_per_subnet": {"internal": 3},
                    "services": ["web"],
                    "vulnerabilities": "minimal"
                },
                "auto_cleanup": True,
                "duration_hours": 1
            },
            "description": "Range for integration testing"
        }
        
        response = client.post("/ranges", json=range_request, headers=researcher_headers)
        assert response.status_code == 200
        
        range_data = response.json()
        range_id = range_data["id"]
        
        # Step 4: Start the range
        response = client.post(f"/ranges/{range_id}/start", headers=researcher_headers)
        assert response.status_code == 200
        
        # Step 5: Cleanup
        response = client.delete(f"/ranges/{range_id}", headers=researcher_headers)
        assert response.status_code == 200
    
    def test_training_scenario_workflow(self, client, researcher_headers):
        """Test complete training scenario workflow"""
        # Step 1: Create cyber range for training
        range_request = {
            "config": {
                "name": "Training Range",
                "size": "medium",
                "topology": {
                    "template": "enterprise",
                    "subnets": ["dmz", "internal"],
                    "hosts_per_subnet": {"dmz": 2, "internal": 5},
                    "services": ["web", "database"],
                    "vulnerabilities": "realistic"
                },
                "monitoring_enabled": True
            }
        }
        
        response = client.post("/ranges", json=range_request, headers=researcher_headers)
        assert response.status_code == 200
        range_id = response.json()["id"]
        
        # Step 2: Generate training campaign
        campaign_request = {
            "red_team_config": {
                "model": "llama2-70b-security",
                "creativity": 0.7,
                "risk_tolerance": 0.5,
                "objective": "Training scenario for incident response"
            },
            "target_profile": {
                "industry": "education",
                "size": "medium",
                "security_maturity": "intermediate",
                "crown_jewels": ["student_records", "research_data"]
            },
            "campaign_duration": 3,
            "tactics": ["initial_access", "lateral_movement", "exfiltration"]
        }
        
        response = client.post("/redteam/campaigns", json=campaign_request, headers=researcher_headers)
        assert response.status_code == 200
        
        # Step 3: Start training session (would require training endpoints)
        # This would involve creating training session, running attacks, monitoring responses
        
        # Step 4: Cleanup
        response = client.delete(f"/ranges/{range_id}", headers=researcher_headers)
        assert response.status_code == 200
    
    def test_multi_user_collaboration(self, client):
        """Test multi-user collaboration scenario"""
        # Create multiple users with different roles
        users = []
        for i, role in enumerate(["admin", "researcher", "researcher", "student"]):
            user_data = {
                "username": f"collab_user_{i}",
                "email": f"user{i}@collab.test",
                "password": f"CollabPass{i}123!",
                "role": role
            }
            
            response = client.post("/auth/register", json=user_data)
            assert response.status_code == 200
            
            user = response.json()
            token = create_access_token({"sub": user["username"], "role": user["role"]})
            users.append({
                "user": user,
                "headers": {"Authorization": f"Bearer {token}"}
            })
        
        admin, researcher1, researcher2, student = users
        
        # Admin creates shared resources
        range_request = {
            "config": {
                "name": "Shared Collaboration Range",
                "size": "large"
            }
        }
        
        response = client.post("/ranges", json=range_request, headers=admin["headers"])
        assert response.status_code == 200
        shared_range_id = response.json()["id"]
        
        # Researchers can access and use the range
        response = client.get(f"/ranges/{shared_range_id}", headers=researcher1["headers"])
        assert response.status_code == 200
        
        response = client.get(f"/ranges/{shared_range_id}", headers=researcher2["headers"])
        assert response.status_code == 200
        
        # Student might have limited access (depending on implementation)
        response = client.get(f"/ranges/{shared_range_id}", headers=student["headers"])
        # Status depends on authorization implementation
        assert response.status_code in [200, 403]
        
        # Cleanup
        response = client.delete(f"/ranges/{shared_range_id}", headers=admin["headers"])
        assert response.status_code == 200


class TestSystemIntegration:
    """Test integration between system components"""
    
    @patch('gan_cyber_range.core.attack_gan.AttackGAN')
    def test_attack_gan_integration(self, mock_gan_class, client, researcher_headers):
        """Test AttackGAN integration with API"""
        # Mock AttackGAN behavior
        mock_gan = MagicMock()
        mock_gan.generate.return_value = [
            {
                "type": "network",
                "techniques": ["T1190"],
                "payload": "test_payload",
                "confidence": 0.9,
                "sophistication": 0.8,
                "detectability": 0.3,
                "metadata": {"test": True}
            }
        ]
        mock_gan.diversity_score.return_value = 0.85
        mock_gan_class.return_value = mock_gan
        
        # Test attack generation
        request_data = {
            "attack_types": ["network"],
            "num_samples": 1
        }
        
        response = client.post("/attacks/generate", json=request_data, headers=researcher_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["attacks"]) == 1
        assert data["attacks"][0]["attack_type"] == "network"
        assert data["diversity_score"] == 0.85
        
        # Verify GAN was called correctly
        mock_gan.generate.assert_called_once()
        mock_gan.diversity_score.assert_called_once()
    
    @patch('gan_cyber_range.red_team.llm_adversary.RedTeamLLM')
    def test_red_team_llm_integration(self, mock_llm_class, client, researcher_headers):
        """Test RedTeamLLM integration with API"""
        # Mock RedTeamLLM behavior
        mock_llm = MagicMock()
        mock_llm.generate_campaign.return_value = {
            "name": "Generated Campaign",
            "stages": [
                {
                    "name": "Initial Access",
                    "objective": "Gain foothold",
                    "techniques": ["T1190"],
                    "success_criteria": "Shell access obtained",
                    "duration": "2 hours",
                    "risk_level": "medium",
                    "detection_likelihood": 0.6
                }
            ],
            "overall_risk": "medium",
            "success_probability": 0.7
        }
        mock_llm_class.return_value = mock_llm
        
        # Test campaign generation
        request_data = {
            "red_team_config": {
                "model": "test-model",
                "creativity": 0.8,
                "risk_tolerance": 0.6,
                "objective": "Test integration"
            },
            "target_profile": {
                "industry": "test",
                "size": "small",
                "security_maturity": "basic",
                "crown_jewels": ["data"]
            },
            "campaign_duration": 1,
            "tactics": ["initial_access"]
        }
        
        response = client.post("/redteam/campaigns", json=request_data, headers=researcher_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Generated Campaign"
        assert len(data["stages"]) == 1
        assert data["overall_risk"] == "medium"
        
        # Verify LLM was called correctly
        mock_llm.generate_campaign.assert_called_once()
    
    @patch('gan_cyber_range.core.cyber_range.CyberRange')
    def test_cyber_range_integration(self, mock_range_class, client, researcher_headers):
        """Test CyberRange integration with API"""
        # Mock CyberRange behavior
        mock_range = MagicMock()
        mock_range.range_id = "test-range-123"
        mock_range.is_running.return_value = True
        mock_range.get_dashboard_url.return_value = "http://localhost:8080/dashboard"
        mock_range.get_resource_usage.return_value = {"cpu": 50, "memory": 75}
        mock_range.get_metrics.return_value = {"attacks": 5, "detections": 3}
        
        mock_range_class.return_value = mock_range
        
        # Test range creation
        request_data = {
            "config": {
                "name": "Integration Test Range",
                "size": "small"
            }
        }
        
        response = client.post("/ranges", json=request_data, headers=researcher_headers)
        assert response.status_code == 200
        
        range_id = response.json()["id"]
        
        # Test range details
        response = client.get(f"/ranges/{range_id}", headers=researcher_headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "running"
        assert data["dashboard_url"] == "http://localhost:8080/dashboard"
    
    async def test_database_integration(self, db):
        """Test database integration"""
        # Test database connection
        health = await db.health_check()
        assert health["database"] == "healthy"
        
        # Test session management
        async with db.get_session() as session:
            # Test basic database operations
            result = await session.execute("SELECT 1 as test")
            assert result.scalar() == 1
    
    def test_metrics_collection_integration(self):
        """Test metrics collection integration"""
        metrics_collector = MetricsCollector()
        
        # Test system metrics collection
        metrics = metrics_collector.collect_system_metrics()
        
        assert isinstance(metrics, dict)
        assert "timestamp" in metrics
        
        # Test custom metrics
        metrics_collector.record_metric("test_metric", 42)
        custom_metrics = metrics_collector.get_metrics()
        
        assert "test_metric" in custom_metrics


class TestPerformanceIntegration:
    """Test system performance under load"""
    
    def test_concurrent_attack_generation(self, client, researcher_headers):
        """Test concurrent attack generation requests"""
        import threading
        import time
        
        results = []
        
        def generate_attacks():
            request_data = {
                "attack_types": ["network"],
                "num_samples": 5
            }
            
            start_time = time.time()
            response = client.post("/attacks/generate", json=request_data, headers=researcher_headers)
            end_time = time.time()
            
            results.append({
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 200
            })
        
        # Create 5 concurrent threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=generate_attacks)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 5
        success_count = sum(1 for r in results if r["success"])
        
        # At least some requests should succeed
        assert success_count > 0
        
        # Average response time should be reasonable
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        assert avg_response_time < 30.0  # Should complete within 30 seconds
    
    def test_memory_usage_under_load(self, client, researcher_headers):
        """Test memory usage under load"""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Generate multiple requests
        for i in range(10):
            request_data = {
                "attack_types": ["network"],
                "num_samples": 10
            }
            
            response = client.post("/attacks/generate", json=request_data, headers=researcher_headers)
            assert response.status_code == 200
            
            # Force garbage collection
            gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_database_connection_pooling(self, db):
        """Test database connection pooling under load"""
        async def db_operation():
            async with db.get_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar()
        
        async def run_concurrent_operations():
            tasks = []
            for _ in range(20):
                task = asyncio.create_task(db_operation())
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        # Run concurrent database operations
        results = asyncio.run(run_concurrent_operations())
        
        # All operations should succeed
        assert len(results) == 20
        assert all(result == 1 for result in results)


class TestErrorHandlingIntegration:
    """Test error handling across system components"""
    
    @patch('gan_cyber_range.core.attack_gan.AttackGAN')
    def test_attack_generation_error_handling(self, mock_gan_class, client, researcher_headers):
        """Test error handling in attack generation"""
        # Mock AttackGAN to raise exception
        mock_gan = MagicMock()
        mock_gan.generate.side_effect = Exception("GAN processing error")
        mock_gan_class.return_value = mock_gan
        
        request_data = {
            "attack_types": ["network"],
            "num_samples": 1
        }
        
        response = client.post("/attacks/generate", json=request_data, headers=researcher_headers)
        assert response.status_code == 500
        
        data = response.json()
        assert "error" in data
    
    @patch('gan_cyber_range.db.database.get_database')
    async def test_database_error_handling(self, mock_get_db, client, researcher_headers):
        """Test database error handling"""
        # Mock database to raise exception
        mock_db = AsyncMock()
        mock_db.get_session.side_effect = Exception("Database connection error")
        mock_get_db.return_value = mock_db
        
        # This would require an endpoint that uses database
        # For now, just test that the error is handled gracefully
        response = client.get("/health", headers=researcher_headers)
        # Should still return something, not crash
        assert response.status_code in [200, 500, 503]
    
    def test_invalid_input_error_handling(self, client, researcher_headers):
        """Test handling of invalid input across endpoints"""
        # Test various invalid inputs
        invalid_requests = [
            # Invalid attack generation
            {
                "endpoint": "/attacks/generate",
                "data": {"attack_types": "invalid", "num_samples": -1}
            },
            # Invalid range configuration
            {
                "endpoint": "/ranges",
                "data": {"config": {"name": "", "size": "invalid_size"}}
            },
            # Invalid campaign request
            {
                "endpoint": "/redteam/campaigns",
                "data": {"red_team_config": {}, "target_profile": {}}
            }
        ]
        
        for request_info in invalid_requests:
            response = client.post(
                request_info["endpoint"],
                json=request_info["data"],
                headers=researcher_headers
            )
            
            # Should return validation error, not crash
            assert response.status_code in [400, 422]


class TestSecurityIntegration:
    """Test security integration across components"""
    
    def test_end_to_end_authentication(self, client):
        """Test authentication flow end-to-end"""
        # Step 1: Register user
        user_data = {
            "username": "e2e_user",
            "email": "e2e@test.com",
            "password": "E2EPassword123!",
            "role": "researcher"
        }
        
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 200
        
        # Step 2: Login
        login_data = {
            "username": user_data["username"],
            "password": user_data["password"]
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        
        token_data = response.json()
        token = token_data["access_token"]
        
        # Step 3: Access protected resource
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 200
        
        user_info = response.json()
        assert user_info["username"] == user_data["username"]
        
        # Step 4: Use authenticated access for operations
        attack_request = {
            "attack_types": ["network"],
            "num_samples": 1
        }
        
        response = client.post("/attacks/generate", json=attack_request, headers=headers)
        assert response.status_code == 200
    
    def test_authorization_integration(self, client):
        """Test authorization integration across components"""
        # Create users with different roles
        users = []
        for role in ["user", "researcher", "admin"]:
            user_data = {
                "username": f"authz_{role}",
                "email": f"authz_{role}@test.com",
                "password": f"AuthZ{role.title()}123!",
                "role": role
            }
            
            response = client.post("/auth/register", json=user_data)
            assert response.status_code == 200
            
            # Login to get token
            login_response = client.post("/auth/login", json={
                "username": user_data["username"],
                "password": user_data["password"]
            })
            token = login_response.json()["access_token"]
            
            users.append({
                "role": role,
                "headers": {"Authorization": f"Bearer {token}"}
            })
        
        # Test role-based access
        attack_request = {
            "attack_types": ["network"],
            "num_samples": 1
        }
        
        for user in users:
            response = client.post("/attacks/generate", json=attack_request, headers=user["headers"])
            
            if user["role"] in ["researcher", "admin"]:
                assert response.status_code == 200
            else:
                assert response.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])