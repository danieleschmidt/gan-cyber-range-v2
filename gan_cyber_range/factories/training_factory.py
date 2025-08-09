"""
Training Factory for creating comprehensive training programs and scenarios.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..core.cyber_range import CyberRange
from ..core.attack_gan import AttackGAN
from ..red_team.llm_adversary import RedTeamLLM
from ..utils.error_handling import TrainingConfigurationError

logger = logging.getLogger(__name__)


class SkillLevel(Enum):
    """Skill level categories"""
    NOVICE = "novice"
    BEGINNER = "beginner" 
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TrainingDomain(Enum):
    """Training domain categories"""
    INCIDENT_RESPONSE = "incident_response"
    PENETRATION_TESTING = "penetration_testing"
    DIGITAL_FORENSICS = "digital_forensics"
    THREAT_HUNTING = "threat_hunting"
    SECURITY_OPERATIONS = "security_operations"
    MALWARE_ANALYSIS = "malware_analysis"
    NETWORK_SECURITY = "network_security"
    WEB_APPLICATION_SECURITY = "web_application_security"


@dataclass
class LearningObjective:
    """Individual learning objective definition"""
    id: str
    description: str
    domain: TrainingDomain
    skill_level: SkillLevel
    estimated_time: int  # minutes
    prerequisites: List[str] = field(default_factory=list)
    assessment_criteria: List[str] = field(default_factory=list)


@dataclass
class TrainingModule:
    """Complete training module with multiple objectives"""
    name: str
    description: str
    objectives: List[LearningObjective]
    duration: timedelta
    difficulty_progression: str  # "linear", "adaptive", "branching"
    hands_on_percentage: float  # 0.0 - 1.0
    team_based: bool = False
    certification_eligible: bool = False


@dataclass
class TrainingProgram:
    """Complete training program with multiple modules"""
    name: str
    description: str
    modules: List[TrainingModule]
    target_audience: str
    total_duration: timedelta
    completion_criteria: Dict[str, Any]
    certification_info: Optional[Dict[str, Any]] = None


class TrainingFactory:
    """Factory for creating comprehensive cybersecurity training programs"""
    
    # Pre-defined learning objectives library
    LEARNING_OBJECTIVES = {
        "ir_basic_triage": LearningObjective(
            id="ir_basic_triage",
            description="Perform initial incident triage and classification",
            domain=TrainingDomain.INCIDENT_RESPONSE,
            skill_level=SkillLevel.BEGINNER,
            estimated_time=30,
            assessment_criteria=[
                "Correctly classify incident severity",
                "Identify affected systems",
                "Document initial findings"
            ]
        ),
        "ir_containment": LearningObjective(
            id="ir_containment", 
            description="Implement effective incident containment strategies",
            domain=TrainingDomain.INCIDENT_RESPONSE,
            skill_level=SkillLevel.INTERMEDIATE,
            estimated_time=45,
            prerequisites=["ir_basic_triage"],
            assessment_criteria=[
                "Select appropriate containment method",
                "Execute containment without data loss",
                "Document containment actions"
            ]
        ),
        "pt_reconnaissance": LearningObjective(
            id="pt_reconnaissance",
            description="Conduct comprehensive target reconnaissance", 
            domain=TrainingDomain.PENETRATION_TESTING,
            skill_level=SkillLevel.INTERMEDIATE,
            estimated_time=60,
            assessment_criteria=[
                "Gather intelligence using multiple methods",
                "Document findings systematically",
                "Identify potential attack vectors"
            ]
        ),
        "df_evidence_collection": LearningObjective(
            id="df_evidence_collection",
            description="Collect and preserve digital evidence",
            domain=TrainingDomain.DIGITAL_FORENSICS,
            skill_level=SkillLevel.INTERMEDIATE,
            estimated_time=90,
            assessment_criteria=[
                "Follow chain of custody procedures",
                "Use appropriate collection tools",
                "Maintain evidence integrity"
            ]
        ),
        "th_hypothesis_development": LearningObjective(
            id="th_hypothesis_development", 
            description="Develop and test threat hunting hypotheses",
            domain=TrainingDomain.THREAT_HUNTING,
            skill_level=SkillLevel.ADVANCED,
            estimated_time=120,
            prerequisites=["basic_log_analysis"],
            assessment_criteria=[
                "Formulate testable hypotheses",
                "Design appropriate hunt queries",
                "Validate findings effectively"
            ]
        )
        # Add more objectives as needed...
    }
    
    # Pre-defined training modules
    TRAINING_MODULES = {
        "incident_response_fundamentals": TrainingModule(
            name="Incident Response Fundamentals",
            description="Core incident response skills and procedures",
            objectives=["ir_basic_triage", "ir_containment"],
            duration=timedelta(hours=4),
            difficulty_progression="linear",
            hands_on_percentage=0.7,
            team_based=True
        ),
        "advanced_threat_hunting": TrainingModule(
            name="Advanced Threat Hunting",
            description="Proactive threat detection and analysis",
            objectives=["th_hypothesis_development"],
            duration=timedelta(hours=8),
            difficulty_progression="adaptive", 
            hands_on_percentage=0.8,
            certification_eligible=True
        ),
        "digital_forensics_basics": TrainingModule(
            name="Digital Forensics Fundamentals",
            description="Introduction to digital forensics methods",
            objectives=["df_evidence_collection"],
            duration=timedelta(hours=6),
            difficulty_progression="linear",
            hands_on_percentage=0.6
        )
        # Add more modules...
    }
    
    def __init__(self, cyber_range_factory=None, attack_factory=None):
        self.cyber_range_factory = cyber_range_factory
        self.attack_factory = attack_factory
        self._custom_objectives: Dict[str, LearningObjective] = {}
        self._custom_modules: Dict[str, TrainingModule] = {}
        
    def create_comprehensive_program(self,
                                   program_name: str,
                                   target_audience: str,
                                   domains: List[TrainingDomain],
                                   skill_level: SkillLevel,
                                   duration_weeks: int = 4) -> TrainingProgram:
        """Create a comprehensive training program"""
        
        logger.info(f"Creating comprehensive training program: {program_name}")
        
        # Select relevant modules based on domains and skill level
        selected_modules = self._select_modules_for_program(domains, skill_level)
        
        # Sequence modules for optimal learning progression
        sequenced_modules = self._sequence_modules(selected_modules)
        
        # Calculate total duration and adjust if needed
        total_duration = sum([module.duration for module in sequenced_modules], timedelta())
        target_duration = timedelta(weeks=duration_weeks)
        
        if total_duration > target_duration:
            # Adjust module content to fit timeframe
            sequenced_modules = self._adjust_modules_for_duration(sequenced_modules, target_duration)
            total_duration = target_duration
            
        # Define completion criteria
        completion_criteria = {
            "min_modules_completed": len(sequenced_modules),
            "min_hands_on_score": 80,  # Percentage
            "final_assessment_required": True,
            "peer_evaluation": target_audience in ["professionals", "teams"]
        }
        
        # Create program
        program = TrainingProgram(
            name=program_name,
            description=f"Comprehensive {skill_level.value} training program covering {len(domains)} domains",
            modules=sequenced_modules,
            target_audience=target_audience,
            total_duration=total_duration,
            completion_criteria=completion_criteria
        )
        
        return program
        
    def create_custom_module(self,
                           module_name: str,
                           objective_ids: List[str],
                           hands_on_percentage: float = 0.7,
                           team_based: bool = False) -> TrainingModule:
        """Create a custom training module from learning objectives"""
        
        # Validate objective IDs
        available_objectives = {**self.LEARNING_OBJECTIVES, **self._custom_objectives}
        objectives = []
        
        for obj_id in objective_ids:
            if obj_id not in available_objectives:
                raise TrainingConfigurationError(f"Unknown learning objective: {obj_id}")
            objectives.append(available_objectives[obj_id])
            
        # Calculate module duration
        total_time = sum(obj.estimated_time for obj in objectives)
        duration = timedelta(minutes=int(total_time / hands_on_percentage))  # Account for instruction time
        
        # Determine difficulty progression
        skill_levels = [obj.skill_level for obj in objectives]
        if len(set(skill_levels)) == 1:
            progression = "linear"
        else:
            progression = "adaptive"
            
        module = TrainingModule(
            name=module_name,
            description=f"Custom module with {len(objectives)} learning objectives",
            objectives=objective_ids,
            duration=duration,
            difficulty_progression=progression,
            hands_on_percentage=hands_on_percentage,
            team_based=team_based
        )
        
        # Cache for reuse
        self._custom_modules[module_name] = module
        
        return module
        
    def create_scenario_based_training(self,
                                     scenario_type: str,
                                     skill_level: SkillLevel,
                                     team_size: int = 4,
                                     include_debrief: bool = True) -> Dict[str, Any]:
        """Create scenario-based training configuration"""
        
        scenario_templates = {
            "apt_campaign": {
                "name": "Advanced Persistent Threat Campaign",
                "description": "Multi-stage APT attack simulation",
                "duration": timedelta(hours=6),
                "roles": ["incident_commander", "analyst", "forensics_specialist", "communications"],
                "phases": ["initial_compromise", "persistence", "lateral_movement", "exfiltration"],
                "learning_focus": [TrainingDomain.INCIDENT_RESPONSE, TrainingDomain.THREAT_HUNTING]
            },
            "ransomware_outbreak": {
                "name": "Ransomware Incident Response",
                "description": "Rapid response to ransomware infection",
                "duration": timedelta(hours=4),
                "roles": ["incident_commander", "technical_lead", "communications", "legal_liaison"],
                "phases": ["detection", "containment", "assessment", "recovery"],
                "learning_focus": [TrainingDomain.INCIDENT_RESPONSE, TrainingDomain.DIGITAL_FORENSICS]
            },
            "insider_threat": {
                "name": "Insider Threat Investigation",
                "description": "Investigation of malicious insider activity",
                "duration": timedelta(hours=8),
                "roles": ["investigator", "analyst", "hr_liaison", "legal_counsel"],
                "phases": ["initial_report", "evidence_gathering", "analysis", "response"],
                "learning_focus": [TrainingDomain.DIGITAL_FORENSICS, TrainingDomain.THREAT_HUNTING]
            }
        }
        
        if scenario_type not in scenario_templates:
            raise TrainingConfigurationError(f"Unknown scenario type: {scenario_type}")
            
        template = scenario_templates[scenario_type]
        
        # Adjust scenario based on skill level
        scenario_config = self._adjust_scenario_for_skill_level(template, skill_level)
        
        # Configure team roles
        if team_size < len(template["roles"]):
            # Combine roles for smaller teams
            scenario_config["roles"] = self._combine_roles_for_team_size(template["roles"], team_size)
        elif team_size > len(template["roles"]):
            # Add observer/trainee roles
            scenario_config["roles"].extend([f"trainee_{i}" for i in range(team_size - len(template["roles"]))])
            
        # Add assessment criteria
        scenario_config["assessment_criteria"] = self._generate_scenario_assessment_criteria(scenario_config)
        
        # Configure environment requirements
        scenario_config["environment_requirements"] = self._generate_environment_requirements(scenario_type)
        
        if include_debrief:
            scenario_config["debrief_structure"] = self._create_debrief_structure(scenario_config)
            
        return scenario_config
        
    def create_certification_track(self,
                                 certification_name: str,
                                 industry_focus: Optional[str] = None) -> TrainingProgram:
        """Create a training track for professional certification"""
        
        certification_requirements = {
            "cissp_associate": {
                "domains": [
                    TrainingDomain.SECURITY_OPERATIONS,
                    TrainingDomain.NETWORK_SECURITY,
                    TrainingDomain.INCIDENT_RESPONSE
                ],
                "duration_weeks": 12,
                "skill_level": SkillLevel.INTERMEDIATE
            },
            "gcih": {
                "domains": [
                    TrainingDomain.INCIDENT_RESPONSE,
                    TrainingDomain.DIGITAL_FORENSICS,
                    TrainingDomain.THREAT_HUNTING
                ],
                "duration_weeks": 8,
                "skill_level": SkillLevel.INTERMEDIATE
            },
            "oscp": {
                "domains": [
                    TrainingDomain.PENETRATION_TESTING,
                    TrainingDomain.WEB_APPLICATION_SECURITY
                ],
                "duration_weeks": 16,
                "skill_level": SkillLevel.ADVANCED
            }
        }
        
        if certification_name not in certification_requirements:
            raise TrainingConfigurationError(f"Unknown certification: {certification_name}")
            
        req = certification_requirements[certification_name]
        
        # Create comprehensive program
        program = self.create_comprehensive_program(
            program_name=f"{certification_name.upper()} Preparation Track",
            target_audience="certification_candidates",
            domains=req["domains"],
            skill_level=req["skill_level"],
            duration_weeks=req["duration_weeks"]
        )
        
        # Add certification-specific enhancements
        program.certification_info = {
            "certification_name": certification_name,
            "exam_preparation": True,
            "practice_exams": True,
            "industry_focus": industry_focus
        }
        
        # Mark eligible modules
        for module in program.modules:
            module.certification_eligible = True
            
        return program
        
    def create_adaptive_learning_path(self,
                                    learner_profile: Dict[str, Any],
                                    learning_goals: List[str]) -> Dict[str, Any]:
        """Create personalized adaptive learning path"""
        
        # Analyze learner profile
        current_skill_level = SkillLevel(learner_profile.get("skill_level", "beginner"))
        preferred_domains = learner_profile.get("domains", [])
        available_time = learner_profile.get("weekly_hours", 10)
        learning_style = learner_profile.get("learning_style", "mixed")  # "visual", "hands_on", "theoretical", "mixed"
        
        # Generate personalized objectives
        recommended_objectives = self._recommend_objectives_for_learner(
            current_skill_level, 
            preferred_domains, 
            learning_goals
        )
        
        # Create adaptive module sequence
        learning_modules = self._create_adaptive_module_sequence(
            recommended_objectives,
            available_time,
            learning_style
        )
        
        # Calculate learning timeline
        timeline = self._calculate_adaptive_timeline(learning_modules, available_time)
        
        adaptive_path = {
            "learner_id": learner_profile.get("id", "anonymous"),
            "learning_modules": learning_modules,
            "timeline": timeline,
            "adaptation_triggers": self._define_adaptation_triggers(),
            "progress_checkpoints": self._define_progress_checkpoints(learning_modules),
            "remediation_strategies": self._define_remediation_strategies()
        }
        
        return adaptive_path
        
    def get_training_analytics(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analytics for training effectiveness"""
        
        analytics = {
            "completion_rates": {},
            "skill_improvement": {},
            "time_efficiency": {},
            "engagement_metrics": {},
            "recommendations": []
        }
        
        # Analyze completion rates by module
        for module_name, completion_data in training_data.get("completions", {}).items():
            total_attempts = len(completion_data)
            successful = sum(1 for attempt in completion_data if attempt.get("passed", False))
            analytics["completion_rates"][module_name] = successful / total_attempts if total_attempts > 0 else 0
            
        # Analyze skill improvement
        pre_scores = training_data.get("pre_assessment_scores", {})
        post_scores = training_data.get("post_assessment_scores", {})
        
        for domain in pre_scores:
            if domain in post_scores:
                improvement = post_scores[domain] - pre_scores[domain]
                analytics["skill_improvement"][domain] = improvement
                
        # Generate recommendations
        analytics["recommendations"] = self._generate_training_recommendations(analytics)
        
        return analytics
        
    def _select_modules_for_program(self, domains: List[TrainingDomain], skill_level: SkillLevel) -> List[TrainingModule]:
        """Select appropriate modules for training program"""
        
        available_modules = {**self.TRAINING_MODULES, **self._custom_modules}
        selected = []
        
        for module_name, module in available_modules.items():
            # Check if module objectives match target domains and skill level
            module_objectives = [self.LEARNING_OBJECTIVES.get(obj_id) for obj_id in module.objectives]
            module_objectives = [obj for obj in module_objectives if obj is not None]
            
            if not module_objectives:
                continue
                
            # Check domain alignment
            module_domains = [obj.domain for obj in module_objectives]
            if any(domain in domains for domain in module_domains):
                # Check skill level appropriateness
                module_skill_levels = [obj.skill_level for obj in module_objectives]
                if skill_level in module_skill_levels or any(
                    self._skill_level_compatible(skill_level, level) for level in module_skill_levels
                ):
                    selected.append(module)
                    
        return selected
        
    def _sequence_modules(self, modules: List[TrainingModule]) -> List[TrainingModule]:
        """Sequence modules for optimal learning progression"""
        
        # Simple implementation - sort by average skill level required
        def get_avg_skill_level(module):
            objectives = [self.LEARNING_OBJECTIVES.get(obj_id) for obj_id in module.objectives]
            skill_values = [self._skill_level_to_int(obj.skill_level) for obj in objectives if obj]
            return sum(skill_values) / len(skill_values) if skill_values else 0
            
        return sorted(modules, key=get_avg_skill_level)
        
    def _skill_level_to_int(self, skill_level: SkillLevel) -> int:
        """Convert skill level to integer for comparison"""
        mapping = {
            SkillLevel.NOVICE: 1,
            SkillLevel.BEGINNER: 2,
            SkillLevel.INTERMEDIATE: 3,
            SkillLevel.ADVANCED: 4,
            SkillLevel.EXPERT: 5
        }
        return mapping.get(skill_level, 2)
        
    def _skill_level_compatible(self, target: SkillLevel, module_level: SkillLevel) -> bool:
        """Check if skill levels are compatible"""
        target_int = self._skill_level_to_int(target)
        module_int = self._skill_level_to_int(module_level)
        
        # Allow some flexibility (±1 level)
        return abs(target_int - module_int) <= 1
        
    def _adjust_modules_for_duration(self, modules: List[TrainingModule], target_duration: timedelta) -> List[TrainingModule]:
        """Adjust module content to fit target duration"""
        # Simplified implementation - would involve more sophisticated content adjustment
        current_duration = sum([module.duration for module in modules], timedelta())
        scale_factor = target_duration.total_seconds() / current_duration.total_seconds()
        
        adjusted_modules = []
        for module in modules:
            adjusted_module = TrainingModule(
                name=module.name,
                description=module.description,
                objectives=module.objectives,
                duration=timedelta(seconds=module.duration.total_seconds() * scale_factor),
                difficulty_progression=module.difficulty_progression,
                hands_on_percentage=module.hands_on_percentage,
                team_based=module.team_based,
                certification_eligible=module.certification_eligible
            )
            adjusted_modules.append(adjusted_module)
            
        return adjusted_modules
        
    def _adjust_scenario_for_skill_level(self, template: Dict[str, Any], skill_level: SkillLevel) -> Dict[str, Any]:
        """Adjust scenario difficulty based on skill level"""
        
        scenario = template.copy()
        
        if skill_level == SkillLevel.NOVICE:
            # Simplify scenario, add more guidance
            scenario["guidance_level"] = "high"
            scenario["time_pressure"] = "low"
            scenario["complexity_reduction"] = 0.3
        elif skill_level == SkillLevel.EXPERT:
            # Increase complexity, reduce guidance
            scenario["guidance_level"] = "minimal"
            scenario["time_pressure"] = "high" 
            scenario["additional_challenges"] = True
            
        return scenario
        
    def _combine_roles_for_team_size(self, roles: List[str], team_size: int) -> List[str]:
        """Combine roles for smaller team sizes"""
        
        if team_size >= len(roles):
            return roles
            
        # Combine similar roles
        role_combinations = {
            ("incident_commander", "communications"): "incident_commander_comms",
            ("analyst", "technical_lead"): "senior_analyst", 
            ("forensics_specialist", "investigator"): "forensics_investigator"
        }
        
        combined_roles = roles[:team_size]  # Simplified - would be more sophisticated
        return combined_roles
        
    def _generate_scenario_assessment_criteria(self, scenario_config: Dict[str, Any]) -> List[str]:
        """Generate assessment criteria for scenario"""
        
        base_criteria = [
            "Demonstrates effective team communication",
            "Follows established procedures",
            "Documents actions appropriately",
            "Makes timely decisions"
        ]
        
        # Add phase-specific criteria
        for phase in scenario_config.get("phases", []):
            if phase == "containment":
                base_criteria.append("Implements effective containment strategies")
            elif phase == "analysis":
                base_criteria.append("Conducts thorough technical analysis")
                
        return base_criteria
        
    def _generate_environment_requirements(self, scenario_type: str) -> Dict[str, Any]:
        """Generate environment requirements for scenario"""
        
        base_requirements = {
            "network_complexity": "medium",
            "monitoring_tools": ["siem", "network_monitor"],
            "victim_systems": 5,
            "simulated_users": 20
        }
        
        scenario_specific = {
            "apt_campaign": {
                "network_complexity": "high",
                "attack_sophistication": "advanced",
                "duration": "extended"
            },
            "ransomware_outbreak": {
                "attack_speed": "rapid",
                "affected_systems_percentage": 0.3
            }
        }
        
        requirements = base_requirements.copy()
        requirements.update(scenario_specific.get(scenario_type, {}))
        
        return requirements
        
    def _create_debrief_structure(self, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured debrief for scenario training"""
        
        debrief = {
            "duration": timedelta(minutes=60),
            "facilitator_required": True,
            "sections": [
                {
                    "name": "Initial Reactions",
                    "duration": timedelta(minutes=10),
                    "type": "open_discussion"
                },
                {
                    "name": "Timeline Review",
                    "duration": timedelta(minutes=15),
                    "type": "structured_review"
                },
                {
                    "name": "Decision Analysis", 
                    "duration": timedelta(minutes=20),
                    "type": "critical_analysis"
                },
                {
                    "name": "Lessons Learned",
                    "duration": timedelta(minutes=15),
                    "type": "action_planning"
                }
            ],
            "deliverables": [
                "Individual reflection notes",
                "Team lessons learned document",
                "Improvement action plan"
            ]
        }
        
        return debrief
        
    def _recommend_objectives_for_learner(self, 
                                        skill_level: SkillLevel, 
                                        domains: List[TrainingDomain], 
                                        goals: List[str]) -> List[str]:
        """Recommend learning objectives for individual learner"""
        
        # Filter objectives by skill level and domains
        suitable_objectives = []
        
        for obj_id, objective in self.LEARNING_OBJECTIVES.items():
            if objective.skill_level == skill_level and objective.domain in domains:
                suitable_objectives.append(obj_id)
                
        # Prioritize based on goals (simplified implementation)
        prioritized = suitable_objectives[:10]  # Limit to manageable number
        
        return prioritized
        
    def _create_adaptive_module_sequence(self, 
                                       objectives: List[str], 
                                       weekly_hours: int, 
                                       learning_style: str) -> List[Dict[str, Any]]:
        """Create adaptive sequence of learning modules"""
        
        modules = []
        for i, obj_id in enumerate(objectives):
            objective = self.LEARNING_OBJECTIVES.get(obj_id)
            if not objective:
                continue
                
            module_config = {
                "objective_id": obj_id,
                "estimated_hours": objective.estimated_time / 60,
                "learning_style_adaptations": self._get_style_adaptations(learning_style),
                "prerequisites_check": True,
                "adaptive_difficulty": True
            }
            
            modules.append(module_config)
            
        return modules
        
    def _get_style_adaptations(self, learning_style: str) -> Dict[str, Any]:
        """Get adaptations for different learning styles"""
        
        adaptations = {
            "visual": {
                "include_diagrams": True,
                "video_content": True,
                "infographics": True
            },
            "hands_on": {
                "lab_percentage": 0.8,
                "interactive_simulations": True,
                "practical_exercises": True
            },
            "theoretical": {
                "reading_materials": True,
                "case_studies": True,
                "research_assignments": True
            },
            "mixed": {
                "balanced_content": True,
                "multiple_formats": True
            }
        }
        
        return adaptations.get(learning_style, adaptations["mixed"])
        
    def _calculate_adaptive_timeline(self, modules: List[Dict[str, Any]], weekly_hours: int) -> Dict[str, Any]:
        """Calculate realistic timeline for adaptive learning"""
        
        total_hours = sum(module["estimated_hours"] for module in modules)
        weeks_needed = total_hours / weekly_hours
        
        timeline = {
            "total_estimated_hours": total_hours,
            "weeks_needed": weeks_needed,
            "weekly_commitment": weekly_hours,
            "milestone_schedule": self._create_milestone_schedule(modules, weekly_hours)
        }
        
        return timeline
        
    def _create_milestone_schedule(self, modules: List[Dict[str, Any]], weekly_hours: int) -> List[Dict[str, Any]]:
        """Create milestone schedule for learning path"""
        
        milestones = []
        cumulative_hours = 0
        
        for i, module in enumerate(modules):
            cumulative_hours += module["estimated_hours"]
            week_number = int(cumulative_hours / weekly_hours) + 1
            
            milestone = {
                "week": week_number,
                "module": module["objective_id"],
                "cumulative_hours": cumulative_hours,
                "completion_percentage": (i + 1) / len(modules)
            }
            
            milestones.append(milestone)
            
        return milestones
        
    def _define_adaptation_triggers(self) -> Dict[str, Any]:
        """Define triggers for learning path adaptation"""
        
        return {
            "performance_threshold": 70,  # Percentage
            "time_deviation_threshold": 1.5,  # Multiplier
            "engagement_threshold": 0.8,
            "adaptation_frequency": "weekly"
        }
        
    def _define_progress_checkpoints(self, modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Define progress checkpoints for learning path"""
        
        checkpoints = []
        
        # Create checkpoint every 3 modules or 25% progress
        checkpoint_frequency = max(3, len(modules) // 4)
        
        for i in range(0, len(modules), checkpoint_frequency):
            checkpoint = {
                "checkpoint_id": f"checkpoint_{i // checkpoint_frequency + 1}",
                "modules_completed": i + checkpoint_frequency,
                "assessment_required": True,
                "adaptation_opportunity": True
            }
            checkpoints.append(checkpoint)
            
        return checkpoints
        
    def _define_remediation_strategies(self) -> Dict[str, Any]:
        """Define remediation strategies for struggling learners"""
        
        return {
            "additional_practice": {
                "trigger": "score_below_70",
                "action": "provide_additional_exercises"
            },
            "pace_adjustment": {
                "trigger": "time_exceeded_150_percent",
                "action": "reduce_weekly_load"
            },
            "learning_style_adjustment": {
                "trigger": "low_engagement", 
                "action": "adapt_content_format"
            },
            "mentor_intervention": {
                "trigger": "multiple_checkpoint_failures",
                "action": "assign_mentor"
            }
        }
        
    def _generate_training_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on training analytics"""
        
        recommendations = []
        
        # Analyze completion rates
        avg_completion = sum(analytics["completion_rates"].values()) / len(analytics["completion_rates"]) if analytics["completion_rates"] else 0
        
        if avg_completion < 0.7:
            recommendations.append("Consider reducing module complexity or increasing support")
            
        # Analyze skill improvement
        avg_improvement = sum(analytics["skill_improvement"].values()) / len(analytics["skill_improvement"]) if analytics["skill_improvement"] else 0
        
        if avg_improvement < 10:  # Less than 10 point improvement
            recommendations.append("Review training effectiveness and consider curriculum updates")
            
        return recommendations