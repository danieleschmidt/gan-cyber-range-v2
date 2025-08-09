"""
Workflow Engine for orchestrating complex cybersecurity training workflows.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from ..utils.error_handling import WorkflowError
from ..utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Individual step status"""
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    id: str
    name: str
    description: str
    handler: Callable
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3
    status: StepStatus = StepStatus.WAITING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Workflow execution context"""
    workflow_id: str
    execution_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    steps_completed: int = 0
    total_steps: int = 0
    current_step: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)


class Workflow:
    """Workflow definition and execution container"""
    
    def __init__(self, 
                 name: str,
                 description: str = "",
                 version: str = "1.0.0"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.version = version
        self.steps: Dict[str, WorkflowStep] = {}
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.tags: List[str] = []
        
    def add_step(self, step: WorkflowStep) -> 'Workflow':
        """Add a step to the workflow"""
        if step.id in self.steps:
            raise WorkflowError(f"Step {step.id} already exists in workflow")
            
        self.steps[step.id] = step
        return self
        
    def add_dependency(self, step_id: str, depends_on: str) -> 'Workflow':
        """Add dependency between steps"""
        if step_id not in self.steps:
            raise WorkflowError(f"Step {step_id} not found")
        if depends_on not in self.steps:
            raise WorkflowError(f"Dependency step {depends_on} not found")
            
        if depends_on not in self.steps[step_id].dependencies:
            self.steps[step_id].dependencies.append(depends_on)
            
        return self
        
    def validate(self) -> List[str]:
        """Validate workflow for circular dependencies and other issues"""
        issues = []
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            issues.append("Circular dependency detected")
            
        # Check that all dependencies exist
        for step_id, step in self.steps.items():
            for dep in step.dependencies:
                if dep not in self.steps:
                    issues.append(f"Step {step_id} depends on non-existent step {dep}")
                    
        # Check for isolated steps (no dependencies and no dependents)
        dependents = set()
        for step in self.steps.values():
            dependents.update(step.dependencies)
            
        isolated = []
        for step_id in self.steps:
            if step_id not in dependents and not self.steps[step_id].dependencies:
                isolated.append(step_id)
                
        if len(isolated) > 1:
            issues.append(f"Multiple isolated steps found: {isolated}")
            
        return issues
        
    def get_execution_order(self) -> List[str]:
        """Get topologically sorted execution order"""
        # Kahn's algorithm for topological sorting
        in_degree = {step_id: 0 for step_id in self.steps}
        
        # Calculate in-degrees
        for step in self.steps.values():
            for dep in step.dependencies:
                in_degree[step.id] += 1
                
        # Find steps with no dependencies
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees of dependent steps
            for step_id, step in self.steps.items():
                if current in step.dependencies:
                    in_degree[step_id] -= 1
                    if in_degree[step_id] == 0:
                        queue.append(step_id)
                        
        if len(result) != len(self.steps):
            raise WorkflowError("Cannot create execution order - circular dependency exists")
            
        return result
        
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS"""
        visited = set()
        rec_stack = set()
        
        def dfs(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            
            for dep in self.steps[step_id].dependencies:
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True
                    
            rec_stack.remove(step_id)
            return False
            
        for step_id in self.steps:
            if step_id not in visited:
                if dfs(step_id):
                    return True
                    
        return False


class WorkflowEngine:
    """Engine for executing workflows with orchestration capabilities"""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.workflow_registry: Dict[str, Workflow] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.global_variables: Dict[str, Any] = {}
        
    def register_workflow(self, workflow: Workflow) -> str:
        """Register a workflow for execution"""
        issues = workflow.validate()
        if issues:
            raise WorkflowError(f"Workflow validation failed: {issues}")
            
        self.workflow_registry[workflow.id] = workflow
        logger.info(f"Registered workflow: {workflow.name} ({workflow.id})")
        
        return workflow.id
        
    async def execute_workflow(self, 
                             workflow_id: str,
                             initial_variables: Optional[Dict[str, Any]] = None,
                             execution_options: Optional[Dict[str, Any]] = None) -> WorkflowExecution:
        """Execute a workflow asynchronously"""
        
        if workflow_id not in self.workflow_registry:
            raise WorkflowError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflow_registry[workflow_id]
        execution_id = str(uuid.uuid4())
        
        # Create execution context
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now(),
            total_steps=len(workflow.steps),
            variables=initial_variables or {}
        )
        
        # Merge global variables
        execution.variables.update(self.global_variables)
        
        self.active_executions[execution_id] = execution
        
        try:
            logger.info(f"Starting workflow execution: {workflow.name} ({execution_id})")
            
            # Emit start event
            await self._emit_event("workflow_started", {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "workflow_name": workflow.name
            })
            
            # Get execution order
            execution_order = workflow.get_execution_order()
            
            # Execute steps in order
            for step_id in execution_order:
                if execution.status != WorkflowStatus.RUNNING:
                    break
                    
                step = workflow.steps[step_id]
                execution.current_step = step_id
                
                # Check if dependencies are satisfied
                if not await self._dependencies_satisfied(step, execution):
                    step.status = StepStatus.FAILED
                    step.error = "Dependencies not satisfied"
                    execution.error_log.append(f"Step {step_id}: Dependencies not satisfied")
                    break
                    
                # Execute step with retries
                await self._execute_step_with_retries(step, execution, workflow)
                
                if step.status == StepStatus.COMPLETED:
                    execution.steps_completed += 1
                    execution.results[step_id] = step.result
                elif step.status == StepStatus.FAILED:
                    execution.status = WorkflowStatus.FAILED
                    break
                    
            # Determine final status
            if execution.status == WorkflowStatus.RUNNING:
                if execution.steps_completed == execution.total_steps:
                    execution.status = WorkflowStatus.COMPLETED
                else:
                    execution.status = WorkflowStatus.FAILED
                    
            execution.end_time = datetime.now()
            
            # Emit completion event
            await self._emit_event("workflow_completed", {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": execution.status.value,
                "duration": (execution.end_time - execution.start_time).total_seconds()
            })
            
            logger.info(f"Workflow execution completed: {execution.status.value}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            execution.error_log.append(f"Workflow execution failed: {str(e)}")
            
            await self._emit_event("workflow_failed", {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "error": str(e)
            })
            
            logger.error(f"Workflow execution failed: {str(e)}")
            
        finally:
            # Record metrics
            self.metrics_collector.record_workflow_execution(
                workflow_id=workflow_id,
                execution_id=execution_id,
                duration=(execution.end_time - execution.start_time).total_seconds() if execution.end_time else 0,
                status=execution.status.value,
                steps_completed=execution.steps_completed
            )
            
        return execution
        
    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause a running workflow"""
        if execution_id not in self.active_executions:
            return False
            
        execution = self.active_executions[execution_id]
        if execution.status == WorkflowStatus.RUNNING:
            execution.status = WorkflowStatus.PAUSED
            
            await self._emit_event("workflow_paused", {
                "execution_id": execution_id,
                "current_step": execution.current_step
            })
            
            return True
            
        return False
        
    async def resume_workflow(self, execution_id: str) -> bool:
        """Resume a paused workflow"""
        if execution_id not in self.active_executions:
            return False
            
        execution = self.active_executions[execution_id]
        if execution.status == WorkflowStatus.PAUSED:
            execution.status = WorkflowStatus.RUNNING
            
            await self._emit_event("workflow_resumed", {
                "execution_id": execution_id,
                "current_step": execution.current_step
            })
            
            return True
            
        return False
        
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a workflow execution"""
        if execution_id not in self.active_executions:
            return False
            
        execution = self.active_executions[execution_id]
        execution.status = WorkflowStatus.CANCELLED
        execution.end_time = datetime.now()
        
        await self._emit_event("workflow_cancelled", {
            "execution_id": execution_id,
            "cancelled_at_step": execution.current_step
        })
        
        return True
        
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get current execution status"""
        return self.active_executions.get(execution_id)
        
    def get_workflow_library(self) -> Dict[str, Dict[str, Any]]:
        """Get library of available workflows"""
        library = {}
        
        for workflow_id, workflow in self.workflow_registry.items():
            library[workflow_id] = {
                "name": workflow.name,
                "description": workflow.description,
                "version": workflow.version,
                "step_count": len(workflow.steps),
                "tags": workflow.tags,
                "created_at": workflow.created_at.isoformat()
            }
            
        return library
        
    def on_event(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
            
        self.event_handlers[event_type].append(handler)
        
    async def _execute_step_with_retries(self, 
                                       step: WorkflowStep,
                                       execution: WorkflowExecution,
                                       workflow: Workflow):
        """Execute a step with retry logic"""
        
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()
        
        await self._emit_event("step_started", {
            "workflow_id": workflow.id,
            "execution_id": execution.execution_id,
            "step_id": step.id,
            "step_name": step.name
        })
        
        for attempt in range(step.max_retries + 1):
            try:
                # Prepare step context
                step_context = {
                    "workflow_id": workflow.id,
                    "execution_id": execution.execution_id,
                    "step_id": step.id,
                    "attempt": attempt + 1,
                    "execution_variables": execution.variables,
                    "step_parameters": step.parameters,
                    "previous_results": execution.results
                }
                
                step.execution_context = step_context
                
                # Execute step handler
                if asyncio.iscoroutinefunction(step.handler):
                    result = await step.handler(step_context)
                else:
                    result = step.handler(step_context)
                    
                # Step completed successfully
                step.result = result
                step.status = StepStatus.COMPLETED
                step.end_time = datetime.now()
                
                await self._emit_event("step_completed", {
                    "workflow_id": workflow.id,
                    "execution_id": execution.execution_id,
                    "step_id": step.id,
                    "duration": (step.end_time - step.start_time).total_seconds(),
                    "attempt": attempt + 1
                })
                
                logger.info(f"Step {step.id} completed successfully on attempt {attempt + 1}")
                break
                
            except Exception as e:
                step.retry_count = attempt + 1
                error_msg = f"Step {step.id} failed on attempt {attempt + 1}: {str(e)}"
                
                if attempt < step.max_retries:
                    logger.warning(f"{error_msg}. Retrying...")
                    execution.error_log.append(error_msg)
                    
                    await self._emit_event("step_retry", {
                        "workflow_id": workflow.id,
                        "execution_id": execution.execution_id,
                        "step_id": step.id,
                        "attempt": attempt + 1,
                        "error": str(e)
                    })
                    
                    # Brief delay before retry
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
                else:
                    # Final failure
                    step.status = StepStatus.FAILED
                    step.error = str(e)
                    step.end_time = datetime.now()
                    execution.error_log.append(f"{error_msg}. Max retries exceeded.")
                    
                    await self._emit_event("step_failed", {
                        "workflow_id": workflow.id,
                        "execution_id": execution.execution_id,
                        "step_id": step.id,
                        "final_error": str(e),
                        "total_attempts": step.retry_count
                    })
                    
                    logger.error(f"Step {step.id} failed permanently after {step.retry_count} attempts")
                    break
                    
    async def _dependencies_satisfied(self, 
                                    step: WorkflowStep,
                                    execution: WorkflowExecution) -> bool:
        """Check if step dependencies are satisfied"""
        
        workflow = self.workflow_registry[execution.workflow_id]
        
        for dep_id in step.dependencies:
            dep_step = workflow.steps.get(dep_id)
            if not dep_step or dep_step.status != StepStatus.COMPLETED:
                return False
                
        return True
        
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit workflow event to registered handlers"""
        
        handlers = self.event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data)
                else:
                    handler(event_type, data)
            except Exception as e:
                logger.error(f"Event handler failed for {event_type}: {str(e)}")
                
    def create_training_workflow(self, 
                               scenario_name: str,
                               components: Dict[str, Any]) -> Workflow:
        """Create a training workflow from components"""
        
        workflow = Workflow(
            name=f"Training: {scenario_name}",
            description=f"Automated training workflow for {scenario_name} scenario"
        )
        
        # Add standard training steps
        workflow.add_step(WorkflowStep(
            id="setup_environment",
            name="Setup Training Environment",
            description="Initialize cyber range and network topology",
            handler=self._setup_training_environment,
            parameters=components.get("environment", {})
        ))
        
        workflow.add_step(WorkflowStep(
            id="deploy_attacks",
            name="Deploy Attack Scenarios", 
            description="Deploy synthetic attacks and red team scenarios",
            handler=self._deploy_attack_scenarios,
            dependencies=["setup_environment"],
            parameters=components.get("attacks", {})
        ))
        
        workflow.add_step(WorkflowStep(
            id="monitor_training",
            name="Monitor Training Progress",
            description="Monitor trainee progress and collect metrics",
            handler=self._monitor_training_progress,
            dependencies=["deploy_attacks"],
            parameters=components.get("monitoring", {})
        ))
        
        workflow.add_step(WorkflowStep(
            id="collect_results",
            name="Collect Training Results",
            description="Collect and analyze training results",
            handler=self._collect_training_results,
            dependencies=["monitor_training"],
            parameters=components.get("collection", {})
        ))
        
        workflow.add_step(WorkflowStep(
            id="cleanup_environment", 
            name="Cleanup Environment",
            description="Clean up training environment and resources",
            handler=self._cleanup_training_environment,
            dependencies=["collect_results"],
            parameters={}
        ))
        
        return workflow
        
    async def _setup_training_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Setup training environment step handler"""
        # This would integrate with CyberRangeFactory
        return {"environment_id": "env_" + str(uuid.uuid4())[:8]}
        
    async def _deploy_attack_scenarios(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy attack scenarios step handler"""
        # This would integrate with AttackFactory
        return {"scenarios_deployed": 3, "attack_vectors": ["phishing", "lateral_movement", "exfiltration"]}
        
    async def _monitor_training_progress(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor training progress step handler"""
        # This would collect real-time metrics
        return {"trainees_active": 5, "completion_rate": 0.75}
        
    async def _collect_training_results(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect training results step handler"""
        # This would gather comprehensive results
        return {"results_collected": True, "performance_scores": [85, 92, 78, 88, 90]}
        
    async def _cleanup_training_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Cleanup environment step handler"""
        # This would clean up resources
        return {"cleanup_completed": True}
        
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics across all workflows"""
        
        stats = {
            "total_workflows": len(self.workflow_registry),
            "active_executions": len(self.active_executions),
            "execution_status_counts": {},
            "average_execution_time": 0,
            "success_rate": 0
        }
        
        # Analyze execution statuses
        status_counts = {}
        total_duration = 0
        successful_executions = 0
        total_executions = len(self.active_executions)
        
        for execution in self.active_executions.values():
            status = execution.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if execution.end_time and execution.start_time:
                total_duration += (execution.end_time - execution.start_time).total_seconds()
                
            if execution.status == WorkflowStatus.COMPLETED:
                successful_executions += 1
                
        stats["execution_status_counts"] = status_counts
        
        if total_executions > 0:
            stats["average_execution_time"] = total_duration / total_executions
            stats["success_rate"] = successful_executions / total_executions
            
        return stats