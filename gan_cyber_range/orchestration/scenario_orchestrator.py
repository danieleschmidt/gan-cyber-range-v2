"""
Scenario Orchestrator for managing complex training scenarios.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .workflow_engine import WorkflowEngine, Workflow, WorkflowStep
from ..factories.attack_factory import AttackFactory
from ..factories.range_factory import CyberRangeFactory
from ..utils.error_handling import ScenarioOrchestrationError
from ..utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of training scenarios"""
    INCIDENT_RESPONSE = "incident_response"
    PENETRATION_TESTING = "penetration_testing" 
    THREAT_HUNTING = "threat_hunting"
    DIGITAL_FORENSICS = "digital_forensics"
    RED_TEAM_EXERCISE = "red_team_exercise"
    BLUE_TEAM_DEFENSE = "blue_team_defense"
    PURPLE_TEAM_COLLAB = "purple_team_collab"
    COMPLIANCE_AUDIT = "compliance_audit"


class ScenarioPhase(Enum):
    """Phases of scenario execution"""
    PREPARATION = "preparation"
    BRIEFING = "briefing"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    EVALUATION = "evaluation"
    DEBRIEF = "debrief"
    CLEANUP = "cleanup"


@dataclass
class TrainingScenario:
    """Complete training scenario definition"""
    id: str
    name: str
    description: str
    scenario_type: ScenarioType
    difficulty_level: str  # "novice", "intermediate", "advanced", "expert"
    estimated_duration: timedelta
    max_participants: int
    learning_objectives: List[str]
    success_criteria: List[str]
    required_roles: List[str]
    environment_requirements: Dict[str, Any]
    attack_patterns: List[Dict[str, Any]]
    monitoring_requirements: Dict[str, Any]
    evaluation_criteria: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class ScenarioExecution:
    """Active scenario execution context"""
    scenario_id: str
    execution_id: str
    participants: List[Dict[str, Any]]
    start_time: datetime
    current_phase: ScenarioPhase
    phase_start_time: datetime
    environment_id: Optional[str] = None
    workflow_execution_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    participant_actions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    real_time_feedback: Dict[str, Any] = field(default_factory=dict)


class ScenarioOrchestrator:
    """Orchestrator for complex training scenarios with real-time coordination"""
    
    def __init__(self,
                 workflow_engine: WorkflowEngine,
                 attack_factory: AttackFactory,
                 range_factory: CyberRangeFactory,
                 metrics_collector: Optional[MetricsCollector] = None):
        self.workflow_engine = workflow_engine
        self.attack_factory = attack_factory
        self.range_factory = range_factory
        self.metrics_collector = metrics_collector or MetricsCollector()
        
        self.scenario_library: Dict[str, TrainingScenario] = {}
        self.active_executions: Dict[str, ScenarioExecution] = {}
        self.scenario_templates: Dict[ScenarioType, Dict[str, Any]] = self._initialize_scenario_templates()
        
        # Event handlers for real-time coordination
        self.event_handlers: Dict[str, List[callable]] = {}
        
    def _initialize_scenario_templates(self) -> Dict[ScenarioType, Dict[str, Any]]:
        """Initialize scenario templates for different types"""
        
        templates = {
            ScenarioType.INCIDENT_RESPONSE: {
                "phases": [
                    ScenarioPhase.PREPARATION,
                    ScenarioPhase.BRIEFING,
                    ScenarioPhase.EXECUTION,
                    ScenarioPhase.EVALUATION,
                    ScenarioPhase.DEBRIEF
                ],
                "required_roles": ["incident_commander", "analyst", "technical_lead", "communications"],
                "typical_duration": timedelta(hours=4),
                "environment_complexity": "medium",
                "attack_sophistication": "intermediate"
            },
            
            ScenarioType.RED_TEAM_EXERCISE: {
                "phases": [
                    ScenarioPhase.PREPARATION,
                    ScenarioPhase.BRIEFING,
                    ScenarioPhase.EXECUTION,
                    ScenarioPhase.MONITORING,
                    ScenarioPhase.EVALUATION,
                    ScenarioPhase.DEBRIEF
                ],
                "required_roles": ["red_team_leader", "penetration_tester", "social_engineer", "payload_developer"],
                "typical_duration": timedelta(days=2),
                "environment_complexity": "high",
                "attack_sophistication": "advanced"
            },
            
            ScenarioType.THREAT_HUNTING: {
                "phases": [
                    ScenarioPhase.PREPARATION,
                    ScenarioPhase.BRIEFING,
                    ScenarioPhase.EXECUTION,
                    ScenarioPhase.EVALUATION,
                    ScenarioPhase.DEBRIEF
                ],
                "required_roles": ["hunt_leader", "data_analyst", "threat_intelligence", "forensics"],
                "typical_duration": timedelta(hours=6),
                "environment_complexity": "high",
                "attack_sophistication": "advanced"
            }
        }
        
        return templates
        
    def create_scenario(self,
                       scenario_type: ScenarioType,
                       name: str,
                       custom_parameters: Optional[Dict[str, Any]] = None) -> TrainingScenario:
        """Create a new training scenario from template"""
        
        if scenario_type not in self.scenario_templates:
            raise ScenarioOrchestrationError(f"Unknown scenario type: {scenario_type}")
            
        template = self.scenario_templates[scenario_type]
        scenario_id = str(uuid.uuid4())
        
        # Apply custom parameters
        params = custom_parameters or {}
        
        scenario = TrainingScenario(
            id=scenario_id,
            name=name,
            description=params.get("description", f"{scenario_type.value.replace('_', ' ').title()} Training Scenario"),
            scenario_type=scenario_type,
            difficulty_level=params.get("difficulty_level", "intermediate"),
            estimated_duration=params.get("duration", template["typical_duration"]),
            max_participants=params.get("max_participants", 8),
            learning_objectives=params.get("learning_objectives", self._get_default_objectives(scenario_type)),
            success_criteria=params.get("success_criteria", self._get_default_success_criteria(scenario_type)),
            required_roles=params.get("required_roles", template["required_roles"]),
            environment_requirements=params.get("environment_requirements", {
                "complexity": template["environment_complexity"],
                "network_topology": "enterprise",
                "monitoring_enabled": True
            }),
            attack_patterns=params.get("attack_patterns", self._get_default_attack_patterns(scenario_type)),
            monitoring_requirements=params.get("monitoring_requirements", {
                "real_time_metrics": True,
                "participant_tracking": True,
                "performance_analysis": True
            }),
            evaluation_criteria=params.get("evaluation_criteria", self._get_default_evaluation_criteria(scenario_type))
        )
        
        # Store in library
        self.scenario_library[scenario_id] = scenario
        
        logger.info(f"Created scenario: {name} ({scenario_id}) of type {scenario_type.value}")
        
        return scenario
        
    async def execute_scenario(self,
                             scenario_id: str,
                             participants: List[Dict[str, Any]],
                             execution_options: Optional[Dict[str, Any]] = None) -> str:
        """Execute a training scenario with full orchestration"""
        
        if scenario_id not in self.scenario_library:
            raise ScenarioOrchestrationError(f"Scenario {scenario_id} not found")
            
        scenario = self.scenario_library[scenario_id]
        execution_id = str(uuid.uuid4())
        
        # Validate participants
        self._validate_participants(scenario, participants)
        
        # Create execution context
        execution = ScenarioExecution(
            scenario_id=scenario_id,
            execution_id=execution_id,
            participants=participants,
            start_time=datetime.now(),
            current_phase=ScenarioPhase.PREPARATION,
            phase_start_time=datetime.now()
        )
        
        self.active_executions[execution_id] = execution
        
        try:
            logger.info(f"Starting scenario execution: {scenario.name} ({execution_id})")
            
            # Create and execute workflow
            workflow = await self._create_scenario_workflow(scenario, execution, execution_options or {})
            workflow_execution_id = self.workflow_engine.register_workflow(workflow)
            execution.workflow_execution_id = workflow_execution_id
            
            # Start workflow execution
            workflow_result = await self.workflow_engine.execute_workflow(
                workflow_execution_id,
                initial_variables={
                    "scenario_id": scenario_id,
                    "execution_id": execution_id,
                    "participants": participants,
                    "scenario_config": scenario.__dict__
                }
            )
            
            # Update execution status based on workflow result
            execution.metrics["workflow_status"] = workflow_result.status.value
            execution.metrics["workflow_duration"] = (workflow_result.end_time - workflow_result.start_time).total_seconds()
            
            logger.info(f"Scenario execution completed: {execution_id}")
            
            return execution_id
            
        except Exception as e:
            execution.events.append({
                "timestamp": datetime.now(),
                "event_type": "execution_error",
                "details": str(e)
            })
            
            logger.error(f"Scenario execution failed: {str(e)}")
            raise
            
    async def monitor_scenario_execution(self, execution_id: str) -> Dict[str, Any]:
        """Monitor real-time scenario execution"""
        
        if execution_id not in self.active_executions:
            raise ScenarioOrchestrationError(f"Execution {execution_id} not found")
            
        execution = self.active_executions[execution_id]
        scenario = self.scenario_library[execution.scenario_id]
        
        # Collect real-time metrics
        monitoring_data = {
            "execution_id": execution_id,
            "scenario_name": scenario.name,
            "current_phase": execution.current_phase.value,
            "phase_duration": (datetime.now() - execution.phase_start_time).total_seconds(),
            "total_duration": (datetime.now() - execution.start_time).total_seconds(),
            "participants_active": len([p for p in execution.participants if p.get("status") == "active"]),
            "recent_events": execution.events[-10:],  # Last 10 events
            "performance_metrics": await self._collect_performance_metrics(execution),
            "learning_progress": await self._assess_learning_progress(execution),
            "environment_status": await self._get_environment_status(execution)
        }
        
        return monitoring_data
        
    async def inject_scenario_event(self,
                                  execution_id: str,
                                  event_type: str,
                                  event_data: Dict[str, Any]) -> bool:
        """Inject real-time events into running scenario"""
        
        if execution_id not in self.active_executions:
            return False
            
        execution = self.active_executions[execution_id]
        
        # Create event record
        event_record = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "data": event_data,
            "injected": True
        }
        
        execution.events.append(event_record)
        
        # Process event based on type
        await self._process_injected_event(execution, event_record)
        
        logger.info(f"Injected event {event_type} into scenario execution {execution_id}")
        
        return True
        
    async def adapt_scenario_difficulty(self,
                                      execution_id: str,
                                      adaptation_type: str,
                                      parameters: Dict[str, Any]) -> bool:
        """Dynamically adapt scenario difficulty during execution"""
        
        if execution_id not in self.active_executions:
            return False
            
        execution = self.active_executions[execution_id]
        
        adaptation_record = {
            "timestamp": datetime.now(),
            "adaptation_type": adaptation_type,
            "parameters": parameters,
            "previous_state": execution.real_time_feedback.copy()
        }
        
        # Apply adaptation
        if adaptation_type == "increase_difficulty":
            await self._increase_scenario_difficulty(execution, parameters)
        elif adaptation_type == "decrease_difficulty":
            await self._decrease_scenario_difficulty(execution, parameters)
        elif adaptation_type == "adjust_pacing":
            await self._adjust_scenario_pacing(execution, parameters)
        elif adaptation_type == "inject_hint":
            await self._inject_learning_hint(execution, parameters)
            
        # Record adaptation
        execution.events.append({
            "timestamp": datetime.now(),
            "event_type": "scenario_adaptation",
            "details": adaptation_record
        })
        
        logger.info(f"Applied scenario adaptation {adaptation_type} to execution {execution_id}")
        
        return True
        
    def get_scenario_library(self) -> Dict[str, Dict[str, Any]]:
        """Get library of available scenarios"""
        
        library = {}
        
        for scenario_id, scenario in self.scenario_library.items():
            library[scenario_id] = {
                "id": scenario_id,
                "name": scenario.name,
                "description": scenario.description,
                "type": scenario.scenario_type.value,
                "difficulty_level": scenario.difficulty_level,
                "estimated_duration": str(scenario.estimated_duration),
                "max_participants": scenario.max_participants,
                "learning_objectives": scenario.learning_objectives
            }
            
        return library
        
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics across all scenario executions"""
        
        analytics = {
            "total_scenarios": len(self.scenario_library),
            "active_executions": len(self.active_executions),
            "scenario_type_distribution": {},
            "difficulty_distribution": {},
            "average_participation": 0,
            "completion_rates": {},
            "learning_effectiveness": {}
        }
        
        # Analyze scenario types
        for scenario in self.scenario_library.values():
            scenario_type = scenario.scenario_type.value
            analytics["scenario_type_distribution"][scenario_type] = \
                analytics["scenario_type_distribution"].get(scenario_type, 0) + 1
                
            difficulty = scenario.difficulty_level
            analytics["difficulty_distribution"][difficulty] = \
                analytics["difficulty_distribution"].get(difficulty, 0) + 1
                
        # Analyze active executions
        total_participants = 0
        for execution in self.active_executions.values():
            total_participants += len(execution.participants)
            
        if self.active_executions:
            analytics["average_participation"] = total_participants / len(self.active_executions)
            
        return analytics
        
    async def _create_scenario_workflow(self,
                                      scenario: TrainingScenario,
                                      execution: ScenarioExecution,
                                      options: Dict[str, Any]) -> Workflow:
        """Create workflow for scenario execution"""
        
        workflow = Workflow(
            name=f"Scenario: {scenario.name}",
            description=f"Execution workflow for {scenario.name} scenario"
        )
        
        # Add preparation phase
        workflow.add_step(WorkflowStep(
            id="prepare_environment",
            name="Prepare Training Environment",
            description="Set up cyber range and attack vectors",
            handler=self._prepare_scenario_environment,
            parameters={
                "scenario": scenario.__dict__,
                "execution": execution.__dict__,
                "options": options
            }
        ))
        
        # Add briefing phase
        workflow.add_step(WorkflowStep(
            id="conduct_briefing",
            name="Conduct Participant Briefing",
            description="Brief participants on scenario objectives and roles",
            handler=self._conduct_scenario_briefing,
            dependencies=["prepare_environment"],
            parameters={"briefing_duration": 15}
        ))
        
        # Add execution phase  
        workflow.add_step(WorkflowStep(
            id="execute_scenario",
            name="Execute Main Scenario",
            description="Run the main scenario with real-time monitoring",
            handler=self._execute_main_scenario,
            dependencies=["conduct_briefing"],
            timeout=scenario.estimated_duration,
            parameters={"monitoring_enabled": True}
        ))
        
        # Add evaluation phase
        workflow.add_step(WorkflowStep(
            id="evaluate_performance",
            name="Evaluate Participant Performance",
            description="Assess participant performance against criteria",
            handler=self._evaluate_scenario_performance,
            dependencies=["execute_scenario"],
            parameters={"evaluation_criteria": scenario.evaluation_criteria}
        ))
        
        # Add debrief phase
        workflow.add_step(WorkflowStep(
            id="conduct_debrief",
            name="Conduct Scenario Debrief",
            description="Facilitate learning debrief session",
            handler=self._conduct_scenario_debrief,
            dependencies=["evaluate_performance"],
            parameters={"debrief_duration": 30}
        ))
        
        # Add cleanup phase
        workflow.add_step(WorkflowStep(
            id="cleanup_resources",
            name="Clean Up Resources",
            description="Clean up scenario environment and resources",
            handler=self._cleanup_scenario_resources,
            dependencies=["conduct_debrief"],
            parameters={}
        ))
        
        return workflow
        
    def _validate_participants(self, scenario: TrainingScenario, participants: List[Dict[str, Any]]):
        """Validate participants meet scenario requirements"""
        
        if len(participants) > scenario.max_participants:
            raise ScenarioOrchestrationError(f"Too many participants: {len(participants)} > {scenario.max_participants}")
            
        # Check role coverage
        participant_roles = [p.get("role") for p in participants]
        missing_roles = []
        
        for required_role in scenario.required_roles:
            if required_role not in participant_roles:
                missing_roles.append(required_role)
                
        if missing_roles:
            logger.warning(f"Missing required roles: {missing_roles}")
            
        # Validate participant skill levels
        for participant in participants:
            skill_level = participant.get("skill_level", "novice")
            if not self._skill_level_appropriate(skill_level, scenario.difficulty_level):
                logger.warning(f"Participant skill level {skill_level} may not match scenario difficulty {scenario.difficulty_level}")
                
    def _skill_level_appropriate(self, participant_skill: str, scenario_difficulty: str) -> bool:
        """Check if participant skill level is appropriate for scenario"""
        
        skill_levels = ["novice", "beginner", "intermediate", "advanced", "expert"]
        
        try:
            participant_idx = skill_levels.index(participant_skill.lower())
            scenario_idx = skill_levels.index(scenario_difficulty.lower())
            
            # Allow ±1 level difference
            return abs(participant_idx - scenario_idx) <= 1
            
        except ValueError:
            return False
            
    def _get_default_objectives(self, scenario_type: ScenarioType) -> List[str]:
        """Get default learning objectives for scenario type"""
        
        objectives_map = {
            ScenarioType.INCIDENT_RESPONSE: [
                "Demonstrate effective incident triage",
                "Execute containment procedures",
                "Coordinate team communication",
                "Document incident timeline"
            ],
            ScenarioType.THREAT_HUNTING: [
                "Develop threat hypotheses", 
                "Execute hunting queries",
                "Analyze hunt results",
                "Present findings effectively"
            ],
            ScenarioType.RED_TEAM_EXERCISE: [
                "Execute multi-stage attack campaign",
                "Maintain operational security",
                "Document attack methodology",
                "Provide defensive recommendations"
            ]
        }
        
        return objectives_map.get(scenario_type, ["Complete scenario objectives"])
        
    def _get_default_success_criteria(self, scenario_type: ScenarioType) -> List[str]:
        """Get default success criteria for scenario type"""
        
        criteria_map = {
            ScenarioType.INCIDENT_RESPONSE: [
                "Incident properly classified within 15 minutes",
                "Containment achieved within 1 hour", 
                "All team members contribute effectively",
                "Complete documentation maintained"
            ],
            ScenarioType.THREAT_HUNTING: [
                "Threat hypothesis properly formulated",
                "Hunt queries executed successfully",
                "Threat indicators identified",
                "Actionable intelligence produced"
            ],
            ScenarioType.RED_TEAM_EXERCISE: [
                "Objective achieved within time limit",
                "Stealth maintained throughout",
                "Complete attack documentation",
                "Defensive gaps identified"
            ]
        }
        
        return criteria_map.get(scenario_type, ["Scenario completed successfully"])
        
    def _get_default_attack_patterns(self, scenario_type: ScenarioType) -> List[Dict[str, Any]]:
        """Get default attack patterns for scenario type"""
        
        patterns_map = {
            ScenarioType.INCIDENT_RESPONSE: [
                {"type": "malware_infection", "sophistication": "medium", "stealth": "low"},
                {"type": "data_exfiltration", "sophistication": "medium", "stealth": "medium"}
            ],
            ScenarioType.THREAT_HUNTING: [
                {"type": "apt_campaign", "sophistication": "high", "stealth": "high"},
                {"type": "living_off_land", "sophistication": "high", "stealth": "high"}
            ],
            ScenarioType.RED_TEAM_EXERCISE: [
                {"type": "phishing_campaign", "sophistication": "high", "stealth": "medium"},
                {"type": "lateral_movement", "sophistication": "high", "stealth": "high"},
                {"type": "privilege_escalation", "sophistication": "high", "stealth": "high"}
            ]
        }
        
        return patterns_map.get(scenario_type, [{"type": "basic_attack", "sophistication": "low", "stealth": "low"}])
        
    def _get_default_evaluation_criteria(self, scenario_type: ScenarioType) -> Dict[str, Any]:
        """Get default evaluation criteria for scenario type"""
        
        criteria_map = {
            ScenarioType.INCIDENT_RESPONSE: {
                "technical_skills": 40,
                "communication": 30,
                "decision_making": 20,
                "documentation": 10
            },
            ScenarioType.THREAT_HUNTING: {
                "analytical_skills": 50,
                "technical_execution": 30,
                "hypothesis_formation": 20
            },
            ScenarioType.RED_TEAM_EXERCISE: {
                "attack_execution": 40,
                "operational_security": 30,
                "documentation": 20,
                "defensive_insights": 10
            }
        }
        
        return criteria_map.get(scenario_type, {"overall_performance": 100})
        
    # Workflow step handlers
    async def _prepare_scenario_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare environment for scenario execution"""
        # Implementation would set up cyber range, deploy attacks, etc.
        return {"environment_prepared": True, "environment_id": "env_" + str(uuid.uuid4())[:8]}
        
    async def _conduct_scenario_briefing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct participant briefing"""
        # Implementation would deliver briefing materials and instructions
        return {"briefing_completed": True, "participants_briefed": len(context.get("participants", []))}
        
    async def _execute_main_scenario(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the main scenario"""
        # Implementation would orchestrate the full scenario execution
        return {"scenario_executed": True, "events_processed": 25, "participant_actions": 47}
        
    async def _evaluate_scenario_performance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate participant performance"""
        # Implementation would assess performance against criteria
        return {"evaluation_completed": True, "performance_scores": {"team": 85, "individual_avg": 82}}
        
    async def _conduct_scenario_debrief(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct scenario debrief"""
        # Implementation would facilitate learning discussion
        return {"debrief_completed": True, "lessons_learned": 8, "action_items": 3}
        
    async def _cleanup_scenario_resources(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up scenario resources"""
        # Implementation would clean up environment and resources
        return {"cleanup_completed": True, "resources_freed": True}
        
    # Real-time monitoring and adaptation methods
    async def _collect_performance_metrics(self, execution: ScenarioExecution) -> Dict[str, Any]:
        """Collect real-time performance metrics"""
        # Implementation would collect comprehensive performance data
        return {"metrics_collected": True}
        
    async def _assess_learning_progress(self, execution: ScenarioExecution) -> Dict[str, Any]:
        """Assess current learning progress"""
        # Implementation would assess learning against objectives
        return {"progress_assessed": True}
        
    async def _get_environment_status(self, execution: ScenarioExecution) -> Dict[str, Any]:
        """Get current environment status"""
        # Implementation would check environment health and status
        return {"environment_healthy": True}
        
    async def _process_injected_event(self, execution: ScenarioExecution, event: Dict[str, Any]):
        """Process injected scenario event"""
        # Implementation would handle dynamic event injection
        pass
        
    async def _increase_scenario_difficulty(self, execution: ScenarioExecution, parameters: Dict[str, Any]):
        """Increase scenario difficulty dynamically"""
        # Implementation would ramp up difficulty
        pass
        
    async def _decrease_scenario_difficulty(self, execution: ScenarioExecution, parameters: Dict[str, Any]):
        """Decrease scenario difficulty dynamically"""
        # Implementation would reduce difficulty
        pass
        
    async def _adjust_scenario_pacing(self, execution: ScenarioExecution, parameters: Dict[str, Any]):
        """Adjust scenario pacing"""
        # Implementation would modify scenario timing
        pass
        
    async def _inject_learning_hint(self, execution: ScenarioExecution, parameters: Dict[str, Any]):
        """Inject learning hint for struggling participants"""
        # Implementation would provide contextual guidance
        pass