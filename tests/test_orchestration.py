"""
Tests for orchestration components and workflow management.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from gan_cyber_range.orchestration.workflow_engine import (
    WorkflowEngine, Workflow, WorkflowStep, WorkflowStatus, StepStatus
)
from gan_cyber_range.orchestration.scenario_orchestrator import (
    ScenarioOrchestrator, TrainingScenario, ScenarioType, ScenarioPhase
)
from gan_cyber_range.orchestration.pipeline_manager import (
    PipelineManager, Pipeline, PipelineStage, PipelineType, StageType
)
from gan_cyber_range.utils.monitoring import MetricsCollector


class TestWorkflowEngine:
    """Test suite for WorkflowEngine"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.metrics_collector = Mock(spec=MetricsCollector)
        self.workflow_engine = WorkflowEngine(self.metrics_collector)
        
    def test_workflow_creation_and_validation(self):
        """Test workflow creation and validation"""
        workflow = Workflow("Test Workflow", "Test workflow description")
        
        # Add steps
        step1 = WorkflowStep("step1", "Step 1", "First step", Mock())
        step2 = WorkflowStep("step2", "Step 2", "Second step", Mock(), dependencies=["step1"])
        
        workflow.add_step(step1).add_step(step2)
        
        # Validate
        issues = workflow.validate()
        assert len(issues) == 0
        
        # Test execution order
        order = workflow.get_execution_order()
        assert order == ["step1", "step2"]
        
    def test_workflow_circular_dependency_detection(self):
        """Test detection of circular dependencies"""
        workflow = Workflow("Circular Test", "Test circular dependencies")
        
        step1 = WorkflowStep("step1", "Step 1", "First step", Mock(), dependencies=["step2"])
        step2 = WorkflowStep("step2", "Step 2", "Second step", Mock(), dependencies=["step1"])
        
        workflow.add_step(step1).add_step(step2)
        
        issues = workflow.validate()
        assert any("circular" in issue.lower() for issue in issues)
        
    def test_workflow_registration(self):
        """Test workflow registration"""
        workflow = Workflow("Test Workflow", "Test workflow")
        workflow.add_step(WorkflowStep("step1", "Step 1", "Test step", Mock()))
        
        workflow_id = self.workflow_engine.register_workflow(workflow)
        
        assert workflow_id in self.workflow_engine.workflow_registry
        assert self.workflow_engine.workflow_registry[workflow_id] == workflow
        
    @pytest.mark.asyncio
    async def test_workflow_execution_success(self):
        """Test successful workflow execution"""
        # Create mock handler
        async def mock_handler(context):
            return {"result": "success", "data": context.get("test_data", "default")}
            
        # Create workflow
        workflow = Workflow("Test Execution", "Test workflow execution")
        step = WorkflowStep("test_step", "Test Step", "Test step execution", mock_handler)
        workflow.add_step(step)
        
        # Register and execute
        workflow_id = self.workflow_engine.register_workflow(workflow)
        
        execution = await self.workflow_engine.execute_workflow(
            workflow_id,
            initial_variables={"test_data": "custom_value"}
        )
        
        # Verify execution
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.steps_completed == 1
        assert execution.results["test_step"]["result"] == "success"
        assert execution.results["test_step"]["data"] == "custom_value"
        
    @pytest.mark.asyncio
    async def test_workflow_execution_failure(self):
        """Test workflow execution with step failure"""
        # Create failing handler
        async def failing_handler(context):
            raise Exception("Step failed intentionally")
            
        # Create workflow
        workflow = Workflow("Failing Workflow", "Test workflow failure")
        step = WorkflowStep("failing_step", "Failing Step", "Step that fails", failing_handler)
        workflow.add_step(step)
        
        # Register and execute
        workflow_id = self.workflow_engine.register_workflow(workflow)
        
        execution = await self.workflow_engine.execute_workflow(workflow_id)
        
        # Verify failure handling
        assert execution.status == WorkflowStatus.FAILED
        assert execution.steps_completed == 0
        assert len(execution.error_log) > 0
        assert "Step failed intentionally" in str(execution.error_log)
        
    @pytest.mark.asyncio
    async def test_workflow_step_retry(self):
        """Test workflow step retry functionality"""
        call_count = 0
        
        async def flaky_handler(context):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Failure {call_count}")
            return {"result": "success_after_retries", "attempts": call_count}
            
        # Create workflow with retry
        workflow = Workflow("Retry Test", "Test step retries")
        step = WorkflowStep(
            "retry_step", 
            "Retry Step", 
            "Step with retries", 
            flaky_handler,
            max_retries=3
        )
        workflow.add_step(step)
        
        # Execute
        workflow_id = self.workflow_engine.register_workflow(workflow)
        execution = await self.workflow_engine.execute_workflow(workflow_id)
        
        # Verify retry behavior
        assert execution.status == WorkflowStatus.COMPLETED
        assert call_count == 3  # Should succeed on third attempt
        assert execution.results["retry_step"]["attempts"] == 3
        
    @pytest.mark.asyncio
    async def test_workflow_pause_resume(self):
        """Test workflow pause and resume functionality"""
        # Create long-running handler
        async def long_running_handler(context):
            await asyncio.sleep(0.1)
            return {"result": "completed"}
            
        workflow = Workflow("Pausable Workflow", "Test pause/resume")
        step = WorkflowStep("long_step", "Long Step", "Long running step", long_running_handler)
        workflow.add_step(step)
        
        workflow_id = self.workflow_engine.register_workflow(workflow)
        
        # Start execution (don't await)
        execution_task = asyncio.create_task(
            self.workflow_engine.execute_workflow(workflow_id)
        )
        
        # Give it a moment to start
        await asyncio.sleep(0.01)
        
        # Get execution ID (simplified for test)
        active_executions = list(self.workflow_engine.active_executions.keys())
        if active_executions:
            execution_id = active_executions[0]
            
            # Test pause
            paused = await self.workflow_engine.pause_workflow(execution_id)
            assert paused is True
            
            # Test resume  
            resumed = await self.workflow_engine.resume_workflow(execution_id)
            assert resumed is True
            
        # Wait for completion
        execution = await execution_task
        assert execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
        
    def test_workflow_library(self):
        """Test workflow library functionality"""
        # Create and register workflows
        workflow1 = Workflow("Workflow 1", "First workflow")
        workflow1.add_step(WorkflowStep("step1", "Step 1", "Test", Mock()))
        workflow1.tags = ["test", "demo"]
        
        workflow2 = Workflow("Workflow 2", "Second workflow")  
        workflow2.add_step(WorkflowStep("step2", "Step 2", "Test", Mock()))
        
        id1 = self.workflow_engine.register_workflow(workflow1)
        id2 = self.workflow_engine.register_workflow(workflow2)
        
        # Get library
        library = self.workflow_engine.get_workflow_library()
        
        assert len(library) == 2
        assert id1 in library
        assert id2 in library
        assert library[id1]["name"] == "Workflow 1"
        assert library[id1]["step_count"] == 1
        assert library[id1]["tags"] == ["test", "demo"]
        
    def test_training_workflow_creation(self):
        """Test creation of training workflows"""
        components = {
            "environment": {"topology": "enterprise"},
            "attacks": {"types": ["phishing", "lateral_movement"]},
            "monitoring": {"real_time": True}
        }
        
        workflow = self.workflow_engine.create_training_workflow("APT Simulation", components)
        
        assert workflow.name == "Training: APT Simulation"
        assert len(workflow.steps) == 5  # Standard training workflow steps
        
        expected_steps = [
            "setup_environment", 
            "deploy_attacks", 
            "monitor_training", 
            "collect_results", 
            "cleanup_environment"
        ]
        
        for step_id in expected_steps:
            assert step_id in workflow.steps


class TestScenarioOrchestrator:
    """Test suite for ScenarioOrchestrator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.workflow_engine = Mock(spec=WorkflowEngine)
        self.attack_factory = Mock()
        self.range_factory = Mock()
        self.orchestrator = ScenarioOrchestrator(
            self.workflow_engine,
            self.attack_factory, 
            self.range_factory
        )
        
    def test_scenario_creation(self):
        """Test scenario creation from template"""
        scenario = self.orchestrator.create_scenario(
            ScenarioType.INCIDENT_RESPONSE,
            "IR Training Scenario",
            custom_parameters={
                "difficulty_level": "advanced",
                "max_participants": 6
            }
        )
        
        assert isinstance(scenario, TrainingScenario)
        assert scenario.name == "IR Training Scenario"
        assert scenario.scenario_type == ScenarioType.INCIDENT_RESPONSE
        assert scenario.difficulty_level == "advanced"
        assert scenario.max_participants == 6
        assert len(scenario.learning_objectives) > 0
        assert len(scenario.success_criteria) > 0
        
    @pytest.mark.asyncio
    async def test_scenario_execution(self):
        """Test scenario execution orchestration"""
        # Create scenario
        scenario = self.orchestrator.create_scenario(
            ScenarioType.THREAT_HUNTING,
            "Threat Hunt Exercise"
        )
        
        participants = [
            {"id": "user1", "role": "hunt_leader", "skill_level": "advanced"},
            {"id": "user2", "role": "data_analyst", "skill_level": "intermediate"}
        ]
        
        # Mock workflow engine behavior
        self.workflow_engine.register_workflow.return_value = "workflow_123"
        
        mock_workflow_result = Mock()
        mock_workflow_result.status.value = "completed"
        mock_workflow_result.start_time = datetime.now()
        mock_workflow_result.end_time = datetime.now() + timedelta(minutes=30)
        
        self.workflow_engine.execute_workflow = AsyncMock(return_value=mock_workflow_result)
        
        # Execute scenario
        execution_id = await self.orchestrator.execute_scenario(
            scenario.id,
            participants
        )
        
        assert execution_id is not None
        assert execution_id in self.orchestrator.active_executions
        
        execution = self.orchestrator.active_executions[execution_id]
        assert execution.scenario_id == scenario.id
        assert len(execution.participants) == 2
        assert execution.workflow_execution_id == "workflow_123"
        
    @pytest.mark.asyncio
    async def test_scenario_monitoring(self):
        """Test real-time scenario monitoring"""
        # Create and execute scenario
        scenario = self.orchestrator.create_scenario(
            ScenarioType.RED_TEAM_EXERCISE,
            "Red Team Exercise"
        )
        
        participants = [{"id": "user1", "role": "red_team_leader"}]
        
        # Mock execution
        execution_id = "test_execution"
        execution = Mock()
        execution.scenario_id = scenario.id
        execution.current_phase = ScenarioPhase.EXECUTION
        execution.phase_start_time = datetime.now() - timedelta(minutes=15)
        execution.start_time = datetime.now() - timedelta(minutes=30)
        execution.participants = participants
        execution.events = []
        
        self.orchestrator.active_executions[execution_id] = execution
        
        # Mock monitoring methods
        self.orchestrator._collect_performance_metrics = AsyncMock(return_value={"metrics_collected": True})
        self.orchestrator._assess_learning_progress = AsyncMock(return_value={"progress_assessed": True})
        self.orchestrator._get_environment_status = AsyncMock(return_value={"environment_healthy": True})
        
        # Monitor scenario
        monitoring_data = await self.orchestrator.monitor_scenario_execution(execution_id)
        
        assert monitoring_data["execution_id"] == execution_id
        assert monitoring_data["current_phase"] == ScenarioPhase.EXECUTION.value
        assert monitoring_data["phase_duration"] > 0
        assert monitoring_data["total_duration"] > 0
        assert monitoring_data["participants_active"] == 0  # No active participants in mock
        
    @pytest.mark.asyncio
    async def test_scenario_event_injection(self):
        """Test dynamic event injection"""
        # Create mock execution
        execution_id = "test_execution"
        execution = Mock()
        execution.events = []
        
        self.orchestrator.active_executions[execution_id] = execution
        self.orchestrator._process_injected_event = AsyncMock()
        
        # Inject event
        event_data = {"type": "network_compromise", "severity": "high"}
        
        success = await self.orchestrator.inject_scenario_event(
            execution_id,
            "security_incident",
            event_data
        )
        
        assert success is True
        assert len(execution.events) == 1
        
        injected_event = execution.events[0]
        assert injected_event["event_type"] == "security_incident"
        assert injected_event["data"] == event_data
        assert injected_event["injected"] is True
        
    @pytest.mark.asyncio
    async def test_scenario_difficulty_adaptation(self):
        """Test dynamic difficulty adaptation"""
        # Create mock execution
        execution_id = "test_execution"
        execution = Mock()
        execution.events = []
        execution.real_time_feedback = {"difficulty": "medium"}
        
        self.orchestrator.active_executions[execution_id] = execution
        self.orchestrator._increase_scenario_difficulty = AsyncMock()
        
        # Adapt difficulty
        adaptation_params = {"difficulty_increase": 0.2, "add_complexity": True}
        
        success = await self.orchestrator.adapt_scenario_difficulty(
            execution_id,
            "increase_difficulty",
            adaptation_params
        )
        
        assert success is True
        assert len(execution.events) == 1
        
        adaptation_event = execution.events[0]
        assert adaptation_event["event_type"] == "scenario_adaptation"
        assert adaptation_params in str(adaptation_event["details"])
        
    def test_scenario_library(self):
        """Test scenario library functionality"""
        # Create scenarios
        scenario1 = self.orchestrator.create_scenario(
            ScenarioType.INCIDENT_RESPONSE,
            "IR Scenario 1"
        )
        
        scenario2 = self.orchestrator.create_scenario(
            ScenarioType.DIGITAL_FORENSICS,
            "Forensics Scenario 1"
        )
        
        # Get library
        library = self.orchestrator.get_scenario_library()
        
        assert len(library) == 2
        assert scenario1.id in library
        assert scenario2.id in library
        assert library[scenario1.id]["name"] == "IR Scenario 1"
        assert library[scenario1.id]["type"] == ScenarioType.INCIDENT_RESPONSE.value
        
    def test_execution_analytics(self):
        """Test execution analytics"""
        # Create some scenarios
        self.orchestrator.create_scenario(ScenarioType.INCIDENT_RESPONSE, "IR 1")
        self.orchestrator.create_scenario(ScenarioType.INCIDENT_RESPONSE, "IR 2")
        self.orchestrator.create_scenario(ScenarioType.THREAT_HUNTING, "TH 1")
        
        # Create mock executions
        execution1 = Mock()
        execution1.participants = [{"id": "u1"}, {"id": "u2"}]
        execution2 = Mock()
        execution2.participants = [{"id": "u3"}]
        
        self.orchestrator.active_executions = {
            "exec1": execution1,
            "exec2": execution2
        }
        
        # Get analytics
        analytics = self.orchestrator.get_execution_analytics()
        
        expected_keys = [
            "total_scenarios", "active_executions", "scenario_type_distribution", 
            "difficulty_distribution", "average_participation"
        ]
        assert all(key in analytics for key in expected_keys)
        assert analytics["total_scenarios"] == 3
        assert analytics["active_executions"] == 2
        assert analytics["average_participation"] == 1.5  # (2 + 1) / 2
        
    def test_participant_validation(self):
        """Test participant validation"""
        scenario = self.orchestrator.create_scenario(
            ScenarioType.INCIDENT_RESPONSE,
            "IR Scenario",
            custom_parameters={"max_participants": 3}
        )
        
        # Too many participants
        too_many_participants = [{"id": f"user{i}"} for i in range(5)]
        
        with pytest.raises(Exception) as exc_info:
            self.orchestrator._validate_participants(scenario, too_many_participants)
            
        assert "Too many participants" in str(exc_info.value)


class TestPipelineManager:
    """Test suite for PipelineManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.workflow_engine = Mock(spec=WorkflowEngine)
        self.pipeline_manager = PipelineManager(self.workflow_engine)
        
    def test_pipeline_creation(self):
        """Test pipeline creation from template"""
        pipeline_id = self.pipeline_manager.create_pipeline(
            PipelineType.ATTACK_GENERATION,
            "Test Attack Generation Pipeline"
        )
        
        assert pipeline_id in self.pipeline_manager.pipeline_registry
        
        pipeline = self.pipeline_manager.pipeline_registry[pipeline_id]
        assert pipeline.name == "Test Attack Generation Pipeline"
        assert pipeline.pipeline_type == PipelineType.ATTACK_GENERATION
        assert len(pipeline.stages) > 0
        
    def test_custom_pipeline_creation(self):
        """Test creation of pipeline with custom stages"""
        custom_stages = [
            {
                "type": StageType.DATA_INGESTION.value,
                "name": "Custom Data Load",
                "parameters": {"source": "custom_dataset"}
            },
            {
                "type": StageType.DATA_VALIDATION.value,
                "name": "Custom Validation",
                "dependencies": ["stage_01"]
            }
        ]
        
        pipeline_id = self.pipeline_manager.create_pipeline(
            PipelineType.DATA_PROCESSING,
            "Custom Pipeline",
            custom_stages=custom_stages
        )
        
        pipeline = self.pipeline_manager.pipeline_registry[pipeline_id]
        assert len(pipeline.stages) == 2
        assert pipeline.stages[0].name == "Custom Data Load"
        assert pipeline.stages[1].dependencies == ["stage_01"]
        
    @pytest.mark.asyncio
    async def test_pipeline_execution(self):
        """Test pipeline execution"""
        # Create pipeline
        pipeline_id = self.pipeline_manager.create_pipeline(
            PipelineType.THREAT_ANALYSIS,
            "Test Threat Analysis"
        )
        
        # Mock resource availability
        self.pipeline_manager._check_resource_availability = AsyncMock(return_value=True)
        self.pipeline_manager._allocate_pipeline_resources = AsyncMock()
        self.pipeline_manager._release_pipeline_resources = AsyncMock()
        
        # Mock workflow execution
        self.workflow_engine.register_workflow.return_value = "workflow_123"
        
        mock_workflow_result = Mock()
        mock_workflow_result.status.value = "completed"
        mock_workflow_result.results = {"stage_01": {"result": "success"}}
        
        self.workflow_engine.execute_workflow = AsyncMock(return_value=mock_workflow_result)
        
        # Execute pipeline
        input_data = {"threat_data": "sample_data"}
        
        execution_id = await self.pipeline_manager.execute_pipeline(
            pipeline_id,
            input_data
        )
        
        assert execution_id is not None
        assert execution_id in self.pipeline_manager.active_executions
        
        execution = self.pipeline_manager.active_executions[execution_id]
        assert execution.pipeline_id == pipeline_id
        assert execution.input_data == input_data
        assert execution.status == "completed"
        
    @pytest.mark.asyncio
    async def test_pipeline_resource_management(self):
        """Test pipeline resource management"""
        # Create pipeline with resource requirements
        pipeline_id = self.pipeline_manager.create_pipeline(
            PipelineType.MODEL_TRAINING,
            "Resource Test Pipeline",
            parameters={"resource_intensive": True}
        )
        
        pipeline = self.pipeline_manager.pipeline_registry[pipeline_id]
        
        # Test resource availability check
        available = await self.pipeline_manager._check_resource_availability(pipeline)
        assert isinstance(available, bool)
        
        # Test resource allocation
        execution_id = "test_exec"
        await self.pipeline_manager._allocate_pipeline_resources(execution_id, pipeline)
        
        assert execution_id in self.pipeline_manager.allocated_resources
        allocated = self.pipeline_manager.allocated_resources[execution_id]
        assert "cpu_cores" in allocated
        assert "memory_gb" in allocated
        
        # Test resource release
        await self.pipeline_manager._release_pipeline_resources(execution_id)
        assert execution_id not in self.pipeline_manager.allocated_resources
        
    @pytest.mark.asyncio
    async def test_pipeline_monitoring(self):
        """Test pipeline execution monitoring"""
        # Create and start mock execution
        pipeline_id = self.pipeline_manager.create_pipeline(
            PipelineType.PERFORMANCE_EVALUATION,
            "Monitoring Test Pipeline"
        )
        
        execution_id = "test_execution"
        execution = Mock()
        execution.pipeline_id = pipeline_id
        execution.status = "running"
        execution.start_time = datetime.now() - timedelta(minutes=5)
        execution.stage_results = {}
        execution.execution_metrics = {}
        
        self.pipeline_manager.active_executions[execution_id] = execution
        
        # Mock workflow status
        mock_workflow_status = Mock()
        mock_workflow_status.total_steps = 5
        mock_workflow_status.steps_completed = 2
        mock_workflow_status.current_step = "stage_03"
        
        self.workflow_engine.get_execution_status.return_value = mock_workflow_status
        
        # Monitor execution
        monitoring_data = await self.pipeline_manager.monitor_pipeline_execution(execution_id)
        
        expected_keys = [
            "execution_id", "pipeline_name", "status", "progress_percentage",
            "current_stage", "elapsed_time", "estimated_remaining", "resource_usage"
        ]
        
        assert all(key in monitoring_data for key in expected_keys)
        assert monitoring_data["execution_id"] == execution_id
        assert monitoring_data["progress_percentage"] == 40.0  # 2/5 * 100
        assert monitoring_data["current_stage"] == "stage_03"
        assert monitoring_data["elapsed_time"] > 0
        
    def test_research_pipeline_creation(self):
        """Test creation of specialized research pipeline"""
        research_config = {
            "dataset_path": "/data/research_dataset.csv",
            "model_types": ["random_forest", "neural_network", "svm"],
            "cross_validation": True,
            "significance_level": 0.01
        }
        
        pipeline_id = self.pipeline_manager.create_research_pipeline(
            "Cybersecurity ML Research",
            research_config
        )
        
        pipeline = self.pipeline_manager.pipeline_registry[pipeline_id]
        assert "Research:" in pipeline.name
        assert pipeline.pipeline_type == PipelineType.RESEARCH_EXPERIMENT
        assert len(pipeline.stages) == 7  # Research pipeline stages
        
        # Check research metadata
        assert pipeline.metadata["research_type"] == "cybersecurity"
        assert "experimental_design" in pipeline.metadata
        
    def test_custom_stage_creation(self):
        """Test creation of custom pipeline stages"""
        def custom_processor(context):
            return {"custom_result": "processed", "input_size": len(context.get("input_data", {}))}
            
        stage = self.pipeline_manager.create_custom_stage(
            stage_name="Custom Processing Stage",
            stage_type=StageType.DATA_TRANSFORMATION,
            processor_function=custom_processor,
            input_requirements=["raw_data"],
            output_schema={"required": ["custom_result"]},
            parameters={"custom_param": "value"}
        )
        
        assert stage.name == "Custom Processing Stage"
        assert stage.stage_type == StageType.DATA_TRANSFORMATION
        assert stage.processor == custom_processor
        assert stage.input_requirements == ["raw_data"]
        assert stage.parameters == {"custom_param": "value"}
        
    def test_pipeline_library(self):
        """Test pipeline library functionality"""
        # Create pipelines
        pipeline1_id = self.pipeline_manager.create_pipeline(
            PipelineType.ATTACK_GENERATION,
            "Attack Gen Pipeline"
        )
        
        pipeline2_id = self.pipeline_manager.create_pipeline(
            PipelineType.THREAT_ANALYSIS,
            "Threat Analysis Pipeline"
        )
        
        # Get library
        library = self.pipeline_manager.get_pipeline_library()
        
        assert len(library) == 2
        assert pipeline1_id in library
        assert pipeline2_id in library
        
        assert library[pipeline1_id]["name"] == "Attack Gen Pipeline"
        assert library[pipeline1_id]["type"] == PipelineType.ATTACK_GENERATION.value
        assert "stage_count" in library[pipeline1_id]
        assert "resource_requirements" in library[pipeline1_id]
        
    def test_execution_analytics(self):
        """Test pipeline execution analytics"""
        # Create pipelines
        self.pipeline_manager.create_pipeline(PipelineType.ATTACK_GENERATION, "Pipeline 1")
        self.pipeline_manager.create_pipeline(PipelineType.THREAT_ANALYSIS, "Pipeline 2")
        
        # Create mock executions
        execution1 = Mock()
        execution1.status = "completed"
        execution1.start_time = datetime.now() - timedelta(minutes=30)
        execution1.end_time = datetime.now()
        
        execution2 = Mock()
        execution2.status = "running"
        execution2.start_time = datetime.now() - timedelta(minutes=10)
        execution2.end_time = None
        
        self.pipeline_manager.active_executions = {
            "exec1": execution1,
            "exec2": execution2
        }
        
        # Mock resource allocation
        self.pipeline_manager.allocated_resources = {
            "exec2": {"cpu_cores": 4, "memory_gb": 8}
        }
        
        # Get analytics
        analytics = self.pipeline_manager.get_execution_analytics()
        
        expected_keys = [
            "total_pipelines", "active_executions", "pipeline_type_distribution",
            "average_execution_time", "success_rate", "resource_utilization"
        ]
        
        assert all(key in analytics for key in expected_keys)
        assert analytics["total_pipelines"] == 2
        assert analytics["active_executions"] == 2
        assert analytics["success_rate"] == 1.0  # Only completed execution counted
        assert analytics["resource_utilization"]["cpu_utilization"] > 0