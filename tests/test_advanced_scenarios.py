"""
Advanced integration tests for complex cybersecurity training scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from gan_cyber_range.factories.attack_factory import AttackFactory
from gan_cyber_range.factories.range_factory import CyberRangeFactory
from gan_cyber_range.factories.network_factory import NetworkFactory
from gan_cyber_range.factories.training_factory import TrainingFactory, SkillLevel, TrainingDomain
from gan_cyber_range.orchestration.workflow_engine import WorkflowEngine
from gan_cyber_range.orchestration.scenario_orchestrator import ScenarioOrchestrator, ScenarioType
from gan_cyber_range.orchestration.pipeline_manager import PipelineManager, PipelineType


class TestAdvancedTrainingScenarios:
    """Test complex end-to-end training scenarios"""
    
    def setup_method(self):
        """Set up comprehensive test environment"""
        # Create factory instances
        self.attack_factory = Mock(spec=AttackFactory)
        self.range_factory = Mock(spec=CyberRangeFactory)
        self.network_factory = Mock(spec=NetworkFactory)
        self.training_factory = TrainingFactory()
        
        # Create orchestration components
        self.workflow_engine = Mock(spec=WorkflowEngine)
        self.scenario_orchestrator = ScenarioOrchestrator(
            self.workflow_engine,
            self.attack_factory,
            self.range_factory
        )
        self.pipeline_manager = Mock(spec=PipelineManager)
        
    @pytest.mark.asyncio
    async def test_apt_campaign_simulation_scenario(self):
        """Test comprehensive APT campaign simulation"""
        # Create APT scenario configuration
        scenario = self.scenario_orchestrator.create_scenario(
            ScenarioType.RED_TEAM_EXERCISE,
            "Advanced Persistent Threat Campaign",
            custom_parameters={
                "difficulty_level": "expert",
                "duration": timedelta(days=2),
                "max_participants": 12,
                "learning_objectives": [
                    "Execute multi-stage attack campaign",
                    "Maintain persistence across reboots",
                    "Exfiltrate sensitive data undetected",
                    "Document attack methodology"
                ],
                "attack_patterns": [
                    {"type": "spear_phishing", "sophistication": "high", "stealth": "high"},
                    {"type": "lateral_movement", "sophistication": "advanced", "stealth": "high"},
                    {"type": "privilege_escalation", "sophistication": "advanced", "stealth": "medium"},
                    {"type": "data_exfiltration", "sophistication": "high", "stealth": "high"}
                ],
                "environment_requirements": {
                    "network_topology": "enterprise",
                    "host_count": 100,
                    "monitoring_tools": ["siem", "edr", "network_monitor"],
                    "threat_intelligence": True
                }
            }
        )
        
        # Verify scenario configuration
        assert scenario.name == "Advanced Persistent Threat Campaign"
        assert scenario.scenario_type == ScenarioType.RED_TEAM_EXERCISE
        assert scenario.difficulty_level == "expert"
        assert scenario.max_participants == 12
        assert len(scenario.attack_patterns) == 4
        assert scenario.environment_requirements["host_count"] == 100
        
        # Define participant teams
        red_team = [
            {"id": "rt_leader", "role": "red_team_leader", "skill_level": "expert"},
            {"id": "rt_phisher", "role": "social_engineer", "skill_level": "advanced"},
            {"id": "rt_pentester", "role": "penetration_tester", "skill_level": "expert"},
            {"id": "rt_dev", "role": "payload_developer", "skill_level": "advanced"}
        ]
        
        blue_team = [
            {"id": "bt_commander", "role": "incident_commander", "skill_level": "advanced"},
            {"id": "bt_analyst1", "role": "security_analyst", "skill_level": "intermediate"},
            {"id": "bt_analyst2", "role": "security_analyst", "skill_level": "intermediate"},
            {"id": "bt_hunter", "role": "threat_hunter", "skill_level": "advanced"},
            {"id": "bt_forensics", "role": "forensics_specialist", "skill_level": "advanced"}
        ]
        
        all_participants = red_team + blue_team
        
        # Mock workflow execution
        self.workflow_engine.register_workflow.return_value = "apt_workflow_123"
        
        mock_workflow_result = Mock()
        mock_workflow_result.status.value = "completed"
        mock_workflow_result.start_time = datetime.now()
        mock_workflow_result.end_time = datetime.now() + timedelta(hours=8)
        
        self.workflow_engine.execute_workflow = AsyncMock(return_value=mock_workflow_result)
        
        # Execute APT scenario
        execution_id = await self.scenario_orchestrator.execute_scenario(
            scenario.id,
            all_participants,
            execution_options={
                "real_time_monitoring": True,
                "automated_scoring": True,
                "generate_timeline": True
            }
        )
        
        # Verify execution setup
        assert execution_id is not None
        execution = self.scenario_orchestrator.active_executions[execution_id]
        assert execution.scenario_id == scenario.id
        assert len(execution.participants) == 9  # Total participants
        
        # Verify workflow was called with correct parameters
        self.workflow_engine.execute_workflow.assert_called_once()
        call_args = self.workflow_engine.execute_workflow.call_args
        initial_vars = call_args[1]["initial_variables"]
        assert initial_vars["scenario_id"] == scenario.id
        assert len(initial_vars["participants"]) == 9
        
    @pytest.mark.asyncio  
    async def test_incident_response_tabletop_exercise(self):
        """Test tabletop incident response exercise"""
        # Create incident response scenario
        scenario = self.scenario_orchestrator.create_scenario(
            ScenarioType.INCIDENT_RESPONSE,
            "Ransomware Outbreak Response",
            custom_parameters={
                "difficulty_level": "intermediate",
                "duration": timedelta(hours=4),
                "max_participants": 8,
                "learning_objectives": [
                    "Rapid incident classification and triage",
                    "Effective team communication under pressure",
                    "Containment strategy execution",
                    "Stakeholder communication management"
                ],
                "attack_patterns": [
                    {"type": "ransomware", "sophistication": "medium", "spread_rate": "rapid"},
                    {"type": "lateral_movement", "sophistication": "medium", "stealth": "low"}
                ],
                "environment_requirements": {
                    "network_topology": "smb_enterprise",
                    "affected_systems_percentage": 0.3,
                    "business_impact": "high",
                    "media_attention": True
                }
            }
        )
        
        # Define IR team participants
        participants = [
            {"id": "ir_commander", "role": "incident_commander", "skill_level": "advanced"},
            {"id": "ir_tech_lead", "role": "technical_lead", "skill_level": "intermediate"},
            {"id": "ir_analyst1", "role": "security_analyst", "skill_level": "intermediate"},
            {"id": "ir_analyst2", "role": "security_analyst", "skill_level": "beginner"},
            {"id": "ir_comms", "role": "communications", "skill_level": "intermediate"},
            {"id": "ir_legal", "role": "legal_liaison", "skill_level": "intermediate"}
        ]
        
        # Mock successful execution
        self.workflow_engine.register_workflow.return_value = "ir_workflow_456"
        mock_workflow_result = Mock()
        mock_workflow_result.status.value = "completed"
        mock_workflow_result.start_time = datetime.now()
        mock_workflow_result.end_time = datetime.now() + timedelta(hours=4)
        self.workflow_engine.execute_workflow = AsyncMock(return_value=mock_workflow_result)
        
        # Execute scenario
        execution_id = await self.scenario_orchestrator.execute_scenario(
            scenario.id,
            participants
        )
        
        # Test real-time event injection during scenario
        await self.scenario_orchestrator.inject_scenario_event(
            execution_id,
            "media_inquiry",
            {"reporter": "TechNews Daily", "urgency": "high", "deadline": "2_hours"}
        )
        
        await self.scenario_orchestrator.inject_scenario_event(
            execution_id,
            "executive_pressure", 
            {"executive": "CEO", "message": "When will systems be back online?"}
        )
        
        # Test difficulty adaptation based on team performance
        await self.scenario_orchestrator.adapt_scenario_difficulty(
            execution_id,
            "increase_difficulty",
            {"additional_system_compromise": 0.1, "add_compliance_pressure": True}
        )
        
        # Verify events were injected
        execution = self.scenario_orchestrator.active_executions[execution_id]
        assert len(execution.events) >= 3  # 2 injected + 1 adaptation
        
        event_types = [event["event_type"] for event in execution.events]
        assert "media_inquiry" in event_types
        assert "executive_pressure" in event_types
        assert "scenario_adaptation" in event_types
        
    @pytest.mark.asyncio
    async def test_threat_hunting_competition(self):
        """Test competitive threat hunting scenario"""
        # Create multiple hunting teams
        teams = []
        for i in range(3):
            team = [
                {"id": f"team{i+1}_leader", "role": "hunt_leader", "skill_level": "advanced", "team": f"team_{i+1}"},
                {"id": f"team{i+1}_analyst1", "role": "data_analyst", "skill_level": "intermediate", "team": f"team_{i+1}"},
                {"id": f"team{i+1}_analyst2", "role": "threat_intelligence", "skill_level": "intermediate", "team": f"team_{i+1}"}
            ]
            teams.extend(team)
            
        # Create competitive scenario
        scenario = self.scenario_orchestrator.create_scenario(
            ScenarioType.THREAT_HUNTING,
            "Advanced Threat Hunting Competition",
            custom_parameters={
                "difficulty_level": "advanced",
                "duration": timedelta(hours=6),
                "max_participants": 12,
                "competitive": True,
                "learning_objectives": [
                    "Develop novel threat hunting hypotheses",
                    "Execute complex hunt queries across large datasets",
                    "Identify advanced threat indicators",
                    "Present actionable intelligence"
                ],
                "environment_requirements": {
                    "network_topology": "complex_enterprise",
                    "log_volume_gb": 500,
                    "threat_density": 0.02,  # 2% of logs contain threat indicators
                    "false_positive_rate": 0.15,
                    "hunt_tools": ["splunk", "elastic", "kusto", "sigma_rules"]
                },
                "scoring_criteria": {
                    "threat_detection": 40,
                    "false_positive_rate": 20,
                    "methodology_quality": 25,
                    "presentation_quality": 15
                }
            }
        )
        
        # Mock execution
        self.workflow_engine.register_workflow.return_value = "hunt_comp_789"
        mock_workflow_result = Mock()
        mock_workflow_result.status.value = "completed"
        self.workflow_engine.execute_workflow = AsyncMock(return_value=mock_workflow_result)
        
        # Execute competition
        execution_id = await self.scenario_orchestrator.execute_scenario(
            scenario.id,
            teams
        )
        
        # Simulate competition events
        competition_events = [
            {"event_type": "hunt_hypothesis_submitted", "team": "team_1", "hypothesis": "Lateral movement via WMI"},
            {"event_type": "threat_found", "team": "team_2", "threat": "credential_dumping", "confidence": 0.85},
            {"event_type": "false_positive", "team": "team_1", "query": "failed_login_analysis"},
            {"event_type": "hunt_hypothesis_submitted", "team": "team_3", "hypothesis": "Data staging in temp directories"},
            {"event_type": "threat_found", "team": "team_1", "threat": "powershell_obfuscation", "confidence": 0.92}
        ]
        
        for event in competition_events:
            await self.scenario_orchestrator.inject_scenario_event(
                execution_id,
                event["event_type"],
                event
            )
            
        # Verify competitive elements
        execution = self.scenario_orchestrator.active_executions[execution_id]
        assert len(execution.participants) == 9  # 3 teams of 3
        assert len(execution.events) >= len(competition_events)
        
        # Verify team separation in participants
        teams_represented = set(p.get("team") for p in execution.participants)
        assert len(teams_represented) == 3
        
    def test_multi_domain_training_program_creation(self):
        """Test creation of comprehensive multi-domain training program"""
        # Create program covering multiple cybersecurity domains
        program = self.training_factory.create_comprehensive_program(
            program_name="Cybersecurity Leadership Development",
            target_audience="senior_professionals",
            domains=[
                TrainingDomain.INCIDENT_RESPONSE,
                TrainingDomain.THREAT_HUNTING,
                TrainingDomain.DIGITAL_FORENSICS,
                TrainingDomain.SECURITY_OPERATIONS,
                TrainingDomain.PENETRATION_TESTING
            ],
            skill_level=SkillLevel.ADVANCED,
            duration_weeks=16
        )
        
        # Verify program structure
        assert program.name == "Cybersecurity Leadership Development"
        assert program.target_audience == "senior_professionals"
        assert len(program.modules) >= 3  # Should have multiple modules
        assert program.total_duration.days >= 100  # 16 weeks approximately
        
        # Verify completion criteria for senior professionals
        assert program.completion_criteria["min_modules_completed"] > 0
        assert program.completion_criteria["final_assessment_required"] is True
        assert program.completion_criteria["peer_evaluation"] is True  # For professionals
        
        # Test adaptive learning path for individual participants
        senior_analyst_profile = {
            "id": "analyst_001",
            "skill_level": "advanced",
            "domains": [TrainingDomain.INCIDENT_RESPONSE, TrainingDomain.THREAT_HUNTING],
            "weekly_hours": 15,
            "learning_style": "hands_on",
            "experience_years": 8,
            "certifications": ["GCIH", "GCFA"]
        }
        
        learning_goals = [
            "leadership_skills",
            "advanced_threat_analysis",
            "team_coordination",
            "strategic_planning"
        ]
        
        adaptive_path = self.training_factory.create_adaptive_learning_path(
            senior_analyst_profile,
            learning_goals
        )
        
        # Verify adaptive path customization
        assert adaptive_path["learner_id"] == "analyst_001"
        assert len(adaptive_path["learning_modules"]) > 0
        assert adaptive_path["timeline"]["weekly_commitment"] == 15
        
        # Verify hands-on learning style adaptations
        style_adaptations = adaptive_path["learning_modules"][0]["learning_style_adaptations"]
        assert style_adaptations["lab_percentage"] == 0.8
        assert style_adaptations["interactive_simulations"] is True
        
    @pytest.mark.asyncio
    async def test_purple_team_collaborative_exercise(self):
        """Test purple team (collaborative red/blue) exercise"""
        # Create purple team scenario combining offensive and defensive elements
        scenario = self.scenario_orchestrator.create_scenario(
            ScenarioType.PURPLE_TEAM_COLLAB,
            "Collaborative Attack and Defense Exercise",
            custom_parameters={
                "difficulty_level": "expert",
                "duration": timedelta(hours=8),
                "max_participants": 16,
                "collaborative": True,
                "learning_objectives": [
                    "Understand attacker methodologies from defender perspective",
                    "Improve detection capabilities through simulated attacks",
                    "Develop effective communication between red and blue teams",
                    "Validate security controls effectiveness"
                ],
                "phases": [
                    {"name": "planning", "duration": timedelta(hours=1), "collaborative": True},
                    {"name": "attack_execution", "duration": timedelta(hours=3), "red_team_lead": True},
                    {"name": "detection_analysis", "duration": timedelta(hours=2), "blue_team_lead": True},
                    {"name": "improvement_planning", "duration": timedelta(hours=2), "collaborative": True}
                ],
                "environment_requirements": {
                    "network_topology": "enterprise",
                    "real_production_mirror": True,
                    "monitoring_enabled": True,
                    "safe_attack_environment": True
                }
            }
        )
        
        # Define mixed red/blue participants
        purple_team_participants = [
            # Red team members
            {"id": "rt_lead", "role": "red_team_leader", "skill_level": "expert", "team_color": "red"},
            {"id": "rt_exploit", "role": "exploit_developer", "skill_level": "advanced", "team_color": "red"},
            {"id": "rt_social", "role": "social_engineer", "skill_level": "advanced", "team_color": "red"},
            {"id": "rt_recon", "role": "reconnaissance_specialist", "skill_level": "intermediate", "team_color": "red"},
            
            # Blue team members
            {"id": "bt_lead", "role": "defense_team_leader", "skill_level": "expert", "team_color": "blue"},
            {"id": "bt_analyst1", "role": "security_analyst", "skill_level": "advanced", "team_color": "blue"},
            {"id": "bt_analyst2", "role": "security_analyst", "skill_level": "intermediate", "team_color": "blue"},
            {"id": "bt_hunter", "role": "threat_hunter", "skill_level": "advanced", "team_color": "blue"},
            {"id": "bt_forensics", "role": "forensics_specialist", "skill_level": "advanced", "team_color": "blue"},
            
            # Purple team coordinators
            {"id": "purple_coord1", "role": "purple_coordinator", "skill_level": "expert", "team_color": "purple"},
            {"id": "purple_coord2", "role": "purple_coordinator", "skill_level": "advanced", "team_color": "purple"}
        ]
        
        # Mock execution
        self.workflow_engine.register_workflow.return_value = "purple_workflow_999"
        mock_workflow_result = Mock()
        mock_workflow_result.status.value = "completed"
        self.workflow_engine.execute_workflow = AsyncMock(return_value=mock_workflow_result)
        
        # Execute purple team exercise
        execution_id = await self.scenario_orchestrator.execute_scenario(
            scenario.id,
            purple_team_participants
        )
        
        # Simulate collaborative events throughout exercise
        collaborative_events = [
            # Planning phase collaboration
            {"event_type": "joint_planning_session", "phase": "planning", "participants": ["rt_lead", "bt_lead", "purple_coord1"]},
            {"event_type": "attack_vector_discussion", "phase": "planning", "vector": "phishing_campaign"},
            
            # Attack execution with real-time blue team observation
            {"event_type": "attack_initiated", "phase": "attack_execution", "attack_type": "initial_compromise"},
            {"event_type": "blue_team_detection", "phase": "attack_execution", "detection": "suspicious_email_click"},
            {"event_type": "red_team_adaptation", "phase": "attack_execution", "adaptation": "alternate_c2_channel"},
            
            # Detection analysis collaboration
            {"event_type": "joint_log_analysis", "phase": "detection_analysis", "participants": ["rt_exploit", "bt_analyst1"]},
            {"event_type": "detection_gap_identified", "phase": "detection_analysis", "gap": "lateral_movement_detection"},
            
            # Improvement planning
            {"event_type": "control_improvement_proposed", "phase": "improvement_planning", "improvement": "enhanced_email_filtering"},
            {"event_type": "training_need_identified", "phase": "improvement_planning", "need": "advanced_threat_hunting"}
        ]
        
        for event in collaborative_events:
            await self.scenario_orchestrator.inject_scenario_event(
                execution_id,
                event["event_type"],
                event
            )
            
        # Verify purple team collaboration
        execution = self.scenario_orchestrator.active_executions[execution_id]
        assert len(execution.participants) == 11
        
        # Verify team color distribution
        team_colors = [p.get("team_color") for p in execution.participants]
        assert "red" in team_colors
        assert "blue" in team_colors  
        assert "purple" in team_colors
        
        # Verify collaborative events were injected
        collaborative_event_types = [e["event_type"] for e in execution.events]
        assert "joint_planning_session" in collaborative_event_types
        assert "joint_log_analysis" in collaborative_event_types
        
    def test_certification_preparation_pipeline(self):
        """Test comprehensive certification preparation pipeline"""
        # Create CISSP preparation track
        cissp_program = self.training_factory.create_certification_track(
            certification_name="cissp_associate",
            industry_focus="financial_services"
        )
        
        # Create GCIH preparation track  
        gcih_program = self.training_factory.create_certification_track(
            certification_name="gcih",
            industry_focus="healthcare"
        )
        
        # Verify certification-specific customization
        assert cissp_program.certification_info["certification_name"] == "cissp_associate"
        assert cissp_program.certification_info["industry_focus"] == "financial_services"
        assert cissp_program.total_duration.days >= 80  # 12 weeks approximately
        
        assert gcih_program.certification_info["certification_name"] == "gcih"
        assert gcih_program.certification_info["industry_focus"] == "healthcare"
        assert gcih_program.total_duration.days >= 50  # 8 weeks approximately
        
        # Test combined certification track for ambitious learners
        multi_cert_profile = {
            "id": "ambitious_learner",
            "skill_level": "advanced",
            "domains": [
                TrainingDomain.INCIDENT_RESPONSE,
                TrainingDomain.SECURITY_OPERATIONS,
                TrainingDomain.DIGITAL_FORENSICS
            ],
            "weekly_hours": 25,  # Very dedicated learner
            "learning_style": "mixed",
            "target_certifications": ["gcih", "cissp_associate"],
            "timeline_months": 8
        }
        
        learning_goals = [
            "gcih_certification",
            "cissp_associate_preparation", 
            "practical_experience",
            "industry_networking"
        ]
        
        combined_path = self.training_factory.create_adaptive_learning_path(
            multi_cert_profile,
            learning_goals
        )
        
        # Verify intensive learning path
        assert combined_path["timeline"]["weekly_commitment"] == 25
        assert len(combined_path["learning_modules"]) > 10  # Should be comprehensive
        
        # Verify mixed learning style accommodations
        style_adaptations = combined_path["learning_modules"][0]["learning_style_adaptations"]
        assert style_adaptations["balanced_content"] is True
        assert style_adaptations["multiple_formats"] is True
        
    def test_scenario_scaling_and_resource_management(self):
        """Test scenario scaling for different organizational sizes"""
        # Test small team scenario (startup environment)
        small_scenario = self.scenario_orchestrator.create_scenario(
            ScenarioType.INCIDENT_RESPONSE,
            "Small Team IR Exercise",
            custom_parameters={
                "difficulty_level": "intermediate",
                "max_participants": 4,
                "resource_constraints": True,
                "environment_requirements": {
                    "network_topology": "startup",
                    "budget_limitations": True,
                    "tool_restrictions": ["basic_siem", "open_source_tools"]
                }
            }
        )
        
        # Test enterprise scenario (large organization)
        enterprise_scenario = self.scenario_orchestrator.create_scenario(
            ScenarioType.INCIDENT_RESPONSE,
            "Enterprise IR Exercise", 
            custom_parameters={
                "difficulty_level": "expert",
                "max_participants": 20,
                "multi_team": True,
                "environment_requirements": {
                    "network_topology": "complex_enterprise",
                    "multiple_business_units": True,
                    "advanced_tools": ["enterprise_siem", "soar", "threat_intel_platform"],
                    "compliance_requirements": ["sox", "pci_dss", "gdpr"]
                }
            }
        )
        
        # Verify scaling differences
        assert small_scenario.max_participants == 4
        assert enterprise_scenario.max_participants == 20
        
        assert "startup" in small_scenario.environment_requirements["network_topology"]
        assert "complex_enterprise" in enterprise_scenario.environment_requirements["network_topology"]
        
        assert small_scenario.environment_requirements.get("budget_limitations") is True
        assert "compliance_requirements" in enterprise_scenario.environment_requirements
        
        # Test resource allocation differences
        small_required_roles = len(small_scenario.required_roles)
        enterprise_required_roles = len(enterprise_scenario.required_roles)
        
        assert enterprise_required_roles >= small_required_roles  # Enterprise should need more roles