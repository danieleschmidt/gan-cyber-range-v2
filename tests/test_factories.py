"""
Tests for factory patterns and object creation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import timedelta

from gan_cyber_range.factories.attack_factory import AttackFactory, AttackConfig
from gan_cyber_range.factories.range_factory import CyberRangeFactory, RangeTemplateConfig
from gan_cyber_range.factories.network_factory import NetworkFactory, NetworkTemplate
from gan_cyber_range.factories.training_factory import TrainingFactory, SkillLevel, TrainingDomain
from gan_cyber_range.utils.security import SecurityManager


class TestAttackFactory:
    """Test suite for AttackFactory"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.security_manager = Mock(spec=SecurityManager)
        self.security_manager.validate_use_case.return_value = True
        self.attack_factory = AttackFactory(self.security_manager)
        
    def test_create_attack_gan_default_config(self):
        """Test creating AttackGAN with default configuration"""
        with patch('gan_cyber_range.factories.attack_factory.AttackGAN') as mock_gan:
            mock_gan_instance = Mock()
            mock_gan.return_value = mock_gan_instance
            
            gan = self.attack_factory.create_attack_gan()
            
            assert gan == mock_gan_instance
            mock_gan.assert_called_once()
            self.security_manager.validate_use_case.assert_called_with("research", "attack_generation")
            
    def test_create_attack_gan_custom_config(self):
        """Test creating AttackGAN with custom configuration"""
        config = AttackConfig(
            gan_architecture="conditional",
            noise_dim=256,
            attack_types=["network", "web"]
        )
        
        with patch('gan_cyber_range.factories.attack_factory.AttackGAN') as mock_gan:
            mock_gan_instance = Mock()
            mock_gan.return_value = mock_gan_instance
            
            gan = self.attack_factory.create_attack_gan(config)
            
            assert gan == mock_gan_instance
            mock_gan.assert_called_once_with(
                architecture="conditional",
                noise_dim=256,
                output_dim=512,
                attack_types=["network", "web"],
                differential_privacy=True,
                privacy_budget=10.0
            )
            
    def test_create_red_team_llm(self):
        """Test creating RedTeamLLM"""
        with patch('gan_cyber_range.factories.attack_factory.RedTeamLLM') as mock_llm:
            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance
            
            llm = self.attack_factory.create_red_team_llm()
            
            assert llm == mock_llm_instance
            self.security_manager.validate_use_case.assert_called_with("research", "red_team_simulation")
            
    def test_security_validation_failure(self):
        """Test security validation prevents creation"""
        self.security_manager.validate_use_case.return_value = False
        
        with pytest.raises(Exception) as exc_info:
            self.attack_factory.create_attack_gan()
            
        assert "not authorized" in str(exc_info.value)
        
    def test_cache_functionality(self):
        """Test caching of created instances"""
        with patch('gan_cyber_range.factories.attack_factory.AttackGAN') as mock_gan:
            mock_gan_instance = Mock()
            mock_gan.return_value = mock_gan_instance
            
            # Create first instance
            gan1 = self.attack_factory.create_attack_gan()
            
            # Create second instance with same config
            gan2 = self.attack_factory.create_attack_gan()
            
            # Should return cached instance
            assert gan1 == gan2
            mock_gan.assert_called_once()  # Called only once due to caching
            
    def test_cache_statistics(self):
        """Test cache statistics reporting"""
        stats = self.attack_factory.get_cache_stats()
        
        expected_keys = ["gan_cache_size", "llm_cache_size", "total_cached_objects"]
        assert all(key in stats for key in expected_keys)
        assert all(isinstance(stats[key], int) for key in expected_keys)
        
    def test_create_training_scenario(self):
        """Test creating training scenarios"""
        scenario = self.attack_factory.create_training_scenario(
            "apt_simulation",
            difficulty="hard",
            custom_params={"duration": "3_hours"}
        )
        
        assert isinstance(scenario, dict)
        assert scenario["attack_types"] == ["reconnaissance", "lateral_movement", "data_exfiltration"]
        assert scenario["duration"] == "3_hours"
        assert scenario["complexity"] > 0.7  # Should be increased for hard difficulty


class TestCyberRangeFactory:
    """Test suite for CyberRangeFactory"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.security_manager = Mock(spec=SecurityManager)
        self.security_manager.check_clearance_level.return_value = True
        self.range_factory = CyberRangeFactory(self.security_manager)
        
    def test_create_from_template_educational_basic(self):
        """Test creating range from educational basic template"""
        with patch('gan_cyber_range.factories.range_factory.CyberRange') as mock_range:
            mock_range_instance = Mock()
            mock_range.return_value = mock_range_instance
            
            cyber_range = self.range_factory.create_from_template("educational_basic")
            
            assert cyber_range == mock_range_instance
            mock_range.assert_called_once()
            
            # Verify range configuration
            call_args = mock_range.call_args[0][0]  # First positional argument (RangeConfig)
            assert "Educational Basic Range" in call_args.name
            assert call_args.resource_limits.cpu_cores == 4
            assert call_args.resource_limits.memory_gb == 8
            
    def test_create_from_template_with_custom_config(self):
        """Test creating range with custom modifications"""
        custom_config = {
            "resource_multiplier": 2.0,
            "security_level": "confidential",
            "name_suffix": "test"
        }
        
        with patch('gan_cyber_range.factories.range_factory.CyberRange') as mock_range:
            mock_range_instance = Mock()
            mock_range.return_value = mock_range_instance
            
            cyber_range = self.range_factory.create_from_template(
                "professional_training", 
                custom_config
            )
            
            # Verify resource scaling
            call_args = mock_range.call_args[0][0]
            assert call_args.resource_limits.cpu_cores == 16  # 8 * 2.0
            assert call_args.resource_limits.memory_gb == 32   # 16 * 2.0
            assert call_args.security_level == "confidential"
            assert call_args.name.endswith("_test")
            
    def test_create_multi_tenant_range(self):
        """Test creating multi-tenant ranges"""
        with patch('gan_cyber_range.factories.range_factory.CyberRange') as mock_range:
            mock_range_instance = Mock()
            mock_range.return_value = mock_range_instance
            
            ranges = self.range_factory.create_multi_tenant_range(
                "educational_basic", 
                tenant_count=3,
                isolation_level="strict"
            )
            
            assert len(ranges) == 3
            assert mock_range.call_count == 3
            
            # Each range should be unique
            assert len(set(id(r) for r in ranges)) == 3
            
    def test_invalid_template_name(self):
        """Test handling of invalid template names"""
        with pytest.raises(ValueError) as exc_info:
            self.range_factory.create_from_template("nonexistent_template")
            
        assert "Unknown template" in str(exc_info.value)
        
    def test_get_active_ranges(self):
        """Test getting active ranges"""
        with patch('gan_cyber_range.factories.range_factory.CyberRange'):
            # Create some ranges
            self.range_factory.create_from_template("educational_basic")
            self.range_factory.create_from_template("professional_training")
            
            active_ranges = self.range_factory.get_active_ranges()
            
            assert len(active_ranges) == 2
            assert all(isinstance(range_id, str) for range_id in active_ranges.keys())
            
    def test_range_statistics(self):
        """Test range statistics generation"""
        with patch('gan_cyber_range.factories.range_factory.CyberRange'):
            # Create some ranges
            self.range_factory.create_from_template("educational_basic")
            self.range_factory.create_from_template("professional_training")
            
            stats = self.range_factory.get_range_statistics()
            
            expected_keys = ["total_ranges", "templates_available", "ranges_by_template"]
            assert all(key in stats for key in expected_keys)
            assert stats["total_ranges"] == 2
            assert stats["templates_available"] == 4  # Number of predefined templates


class TestNetworkFactory:
    """Test suite for NetworkFactory"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.network_factory = NetworkFactory()
        
    def test_create_from_template_startup(self):
        """Test creating network from startup template"""
        with patch('gan_cyber_range.factories.network_factory.NetworkTopology') as mock_topology:
            mock_topology.generate.return_value = Mock()
            
            topology = self.network_factory.create_from_template("startup")
            
            assert topology is not None
            assert topology.metadata["template"] == "startup"
            assert topology.metadata["complexity_score"] == 2
            
    def test_create_from_template_with_scaling(self):
        """Test creating network with scale factor"""
        with patch('gan_cyber_range.factories.network_factory.NetworkTopology') as mock_topology:
            mock_topology.generate.return_value = Mock()
            
            topology = self.network_factory.create_from_template(
                "smb_enterprise", 
                scale_factor=2.0
            )
            
            assert topology.metadata["scale_factor"] == 2.0
            
    def test_create_attack_scenario_topology(self):
        """Test creating topology optimized for attack scenarios"""
        with patch('gan_cyber_range.factories.network_factory.NetworkTopology') as mock_topology:
            mock_topology.generate.return_value = Mock()
            
            topology = self.network_factory.create_attack_scenario_topology(
                "lateral_movement",
                difficulty="medium"
            )
            
            assert topology is not None
            
    def test_invalid_template_name(self):
        """Test handling of invalid template names"""
        with pytest.raises(Exception) as exc_info:
            self.network_factory.create_from_template("nonexistent_template")
            
        assert "Unknown template" in str(exc_info.value)
        
    def test_network_statistics(self):
        """Test network statistics generation"""
        with patch('gan_cyber_range.factories.network_factory.NetworkTopology') as mock_topology:
            mock_topology_instance = Mock()
            mock_topology_instance.hosts = [Mock() for _ in range(10)]
            mock_topology_instance.subnets = [Mock() for _ in range(3)]
            mock_topology_instance.metadata = {"complexity_score": 5}
            mock_topology.generate.return_value = mock_topology_instance
            
            topology = self.network_factory.create_from_template("enterprise")
            stats = self.network_factory.get_network_statistics(topology)
            
            expected_keys = ["total_hosts", "total_subnets", "complexity_score"]
            assert all(key in stats for key in expected_keys)
            assert stats["total_hosts"] == 10
            assert stats["total_subnets"] == 3
            assert stats["complexity_score"] == 5
            
    def test_topology_validation(self):
        """Test topology validation functionality"""
        with patch('gan_cyber_range.factories.network_factory.NetworkTopology') as mock_topology:
            mock_topology_instance = Mock()
            mock_topology_instance.hosts = [Mock() for _ in range(5)]
            mock_topology_instance.subnets = [Mock() for _ in range(2)]
            mock_topology.generate.return_value = mock_topology_instance
            
            topology = self.network_factory.create_from_template("startup")
            validation = self.network_factory.validate_topology(topology)
            
            expected_keys = ["valid", "warnings", "errors", "recommendations"]
            assert all(key in validation for key in expected_keys)
            assert isinstance(validation["valid"], bool)
            assert isinstance(validation["warnings"], list)


class TestTrainingFactory:
    """Test suite for TrainingFactory"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.training_factory = TrainingFactory()
        
    def test_create_comprehensive_program(self):
        """Test creating comprehensive training program"""
        program = self.training_factory.create_comprehensive_program(
            program_name="Advanced Security Training",
            target_audience="professionals",
            domains=[TrainingDomain.INCIDENT_RESPONSE, TrainingDomain.THREAT_HUNTING],
            skill_level=SkillLevel.ADVANCED,
            duration_weeks=8
        )
        
        assert program.name == "Advanced Security Training"
        assert program.target_audience == "professionals"
        assert len(program.modules) > 0
        assert program.completion_criteria["min_modules_completed"] > 0
        
    def test_create_custom_module(self):
        """Test creating custom training module"""
        objective_ids = ["ir_basic_triage", "ir_containment"]
        
        module = self.training_factory.create_custom_module(
            module_name="Custom IR Module",
            objective_ids=objective_ids,
            hands_on_percentage=0.8,
            team_based=True
        )
        
        assert module.name == "Custom IR Module"
        assert module.objectives == objective_ids
        assert module.hands_on_percentage == 0.8
        assert module.team_based is True
        assert module.duration.total_seconds() > 0
        
    def test_create_scenario_based_training(self):
        """Test creating scenario-based training"""
        scenario = self.training_factory.create_scenario_based_training(
            scenario_type="ransomware_outbreak",
            skill_level=SkillLevel.INTERMEDIATE,
            team_size=4,
            include_debrief=True
        )
        
        assert scenario["name"] == "Ransomware Incident Response"
        assert len(scenario["roles"]) <= 4  # Should adjust for team size
        assert "debrief_structure" in scenario
        assert scenario["assessment_criteria"] is not None
        
    def test_create_certification_track(self):
        """Test creating certification track"""
        program = self.training_factory.create_certification_track(
            certification_name="gcih",
            industry_focus="financial_services"
        )
        
        assert "GCIH" in program.name
        assert program.certification_info["certification_name"] == "gcih"
        assert program.certification_info["industry_focus"] == "financial_services"
        assert all(module.certification_eligible for module in program.modules)
        
    def test_create_adaptive_learning_path(self):
        """Test creating adaptive learning path"""
        learner_profile = {
            "id": "learner_123",
            "skill_level": "intermediate",
            "domains": [TrainingDomain.INCIDENT_RESPONSE],
            "weekly_hours": 8,
            "learning_style": "hands_on"
        }
        
        learning_goals = ["improve_incident_response", "advanced_forensics"]
        
        path = self.training_factory.create_adaptive_learning_path(
            learner_profile,
            learning_goals
        )
        
        assert path["learner_id"] == "learner_123"
        assert len(path["learning_modules"]) > 0
        assert "timeline" in path
        assert "adaptation_triggers" in path
        
    def test_invalid_certification_name(self):
        """Test handling of invalid certification names"""
        with pytest.raises(Exception) as exc_info:
            self.training_factory.create_certification_track("invalid_cert")
            
        assert "Unknown certification" in str(exc_info.value)
        
    def test_training_analytics(self):
        """Test training analytics generation"""
        training_data = {
            "completions": {
                "module1": [{"passed": True}, {"passed": False}, {"passed": True}],
                "module2": [{"passed": True}, {"passed": True}]
            },
            "pre_assessment_scores": {
                "incident_response": 60,
                "threat_hunting": 55
            },
            "post_assessment_scores": {
                "incident_response": 85,
                "threat_hunting": 78
            }
        }
        
        analytics = self.training_factory.get_training_analytics(training_data)
        
        expected_keys = ["completion_rates", "skill_improvement", "recommendations"]
        assert all(key in analytics for key in expected_keys)
        
        # Check completion rates
        assert abs(analytics["completion_rates"]["module1"] - 0.67) < 0.01  # 2/3
        assert analytics["completion_rates"]["module2"] == 1.0  # 2/2
        
        # Check skill improvement
        assert analytics["skill_improvement"]["incident_response"] == 25  # 85 - 60
        assert analytics["skill_improvement"]["threat_hunting"] == 23   # 78 - 55
        
    def test_unknown_objective_id(self):
        """Test handling of unknown learning objective IDs"""
        with pytest.raises(Exception) as exc_info:
            self.training_factory.create_custom_module(
                "Test Module",
                ["nonexistent_objective"]
            )
            
        assert "Unknown learning objective" in str(exc_info.value)