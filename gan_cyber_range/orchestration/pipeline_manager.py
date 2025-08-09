"""
Pipeline Manager for orchestrating complex data processing and ML pipelines.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from .workflow_engine import WorkflowEngine, WorkflowStep, Workflow
from ..utils.error_handling import PipelineError
from ..utils.monitoring import MetricsCollector
from ..utils.caching import CacheManager

logger = logging.getLogger(__name__)


class PipelineType(Enum):
    """Types of processing pipelines"""
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    ATTACK_GENERATION = "attack_generation"
    THREAT_ANALYSIS = "threat_analysis"
    PERFORMANCE_EVALUATION = "performance_evaluation"
    RESEARCH_EXPERIMENT = "research_experiment"


class StageType(Enum):
    """Types of pipeline stages"""
    DATA_INGESTION = "data_ingestion"
    DATA_VALIDATION = "data_validation"
    DATA_TRANSFORMATION = "data_transformation"
    FEATURE_EXTRACTION = "feature_extraction"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    RESULT_ANALYSIS = "result_analysis"
    OUTPUT_GENERATION = "output_generation"


@dataclass
class PipelineStage:
    """Individual pipeline stage definition"""
    id: str
    name: str
    stage_type: StageType
    processor: Callable
    input_requirements: List[str]
    output_schema: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    caching_enabled: bool = True
    timeout: Optional[timedelta] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Pipeline:
    """Complete processing pipeline definition"""
    id: str
    name: str
    description: str
    pipeline_type: PipelineType
    version: str
    stages: List[PipelineStage]
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class PipelineExecution:
    """Pipeline execution context and state"""
    pipeline_id: str
    execution_id: str
    workflow_execution_id: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"
    error_log: List[str] = field(default_factory=list)


class PipelineManager:
    """Manager for complex data processing and ML pipelines"""
    
    def __init__(self,
                 workflow_engine: WorkflowEngine,
                 cache_manager: Optional[CacheManager] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        self.workflow_engine = workflow_engine
        self.cache_manager = cache_manager or CacheManager()
        self.metrics_collector = metrics_collector or MetricsCollector()
        
        self.pipeline_registry: Dict[str, Pipeline] = {}
        self.active_executions: Dict[str, PipelineExecution] = {}
        self.pipeline_templates: Dict[PipelineType, Dict[str, Any]] = self._initialize_pipeline_templates()
        
        # Resource management
        self.resource_pool = {
            "cpu_cores": 8,
            "memory_gb": 16,
            "gpu_count": 1,
            "storage_gb": 100
        }
        self.allocated_resources: Dict[str, Dict[str, Any]] = {}
        
    def _initialize_pipeline_templates(self) -> Dict[PipelineType, Dict[str, Any]]:
        """Initialize pipeline templates for different types"""
        
        templates = {
            PipelineType.ATTACK_GENERATION: {
                "stages": [
                    {"type": StageType.DATA_INGESTION, "name": "Load Training Data"},
                    {"type": StageType.DATA_VALIDATION, "name": "Validate Attack Patterns"},
                    {"type": StageType.FEATURE_EXTRACTION, "name": "Extract Attack Features"},
                    {"type": StageType.MODEL_TRAINING, "name": "Train Attack GAN"},
                    {"type": StageType.MODEL_EVALUATION, "name": "Evaluate Generated Attacks"},
                    {"type": StageType.OUTPUT_GENERATION, "name": "Generate Attack Variants"}
                ],
                "resource_requirements": {"cpu_cores": 4, "memory_gb": 8, "gpu_count": 1},
                "typical_duration": timedelta(hours=2)
            },
            
            PipelineType.THREAT_ANALYSIS: {
                "stages": [
                    {"type": StageType.DATA_INGESTION, "name": "Ingest Threat Intelligence"},
                    {"type": StageType.DATA_TRANSFORMATION, "name": "Normalize Threat Data"},
                    {"type": StageType.FEATURE_EXTRACTION, "name": "Extract IOCs and TTPs"},
                    {"type": StageType.RESULT_ANALYSIS, "name": "Analyze Threat Patterns"},
                    {"type": StageType.OUTPUT_GENERATION, "name": "Generate Threat Report"}
                ],
                "resource_requirements": {"cpu_cores": 2, "memory_gb": 4},
                "typical_duration": timedelta(minutes=30)
            },
            
            PipelineType.RESEARCH_EXPERIMENT: {
                "stages": [
                    {"type": StageType.DATA_INGESTION, "name": "Load Experimental Data"},
                    {"type": StageType.DATA_VALIDATION, "name": "Validate Data Quality"},
                    {"type": StageType.DATA_TRANSFORMATION, "name": "Preprocess Data"},
                    {"type": StageType.MODEL_TRAINING, "name": "Train Models"},
                    {"type": StageType.MODEL_EVALUATION, "name": "Evaluate Results"},
                    {"type": StageType.RESULT_ANALYSIS, "name": "Statistical Analysis"},
                    {"type": StageType.OUTPUT_GENERATION, "name": "Generate Research Report"}
                ],
                "resource_requirements": {"cpu_cores": 6, "memory_gb": 12, "gpu_count": 1},
                "typical_duration": timedelta(hours=4)
            }
        }
        
        return templates
        
    def create_pipeline(self,
                       pipeline_type: PipelineType,
                       name: str,
                       custom_stages: Optional[List[Dict[str, Any]]] = None,
                       parameters: Optional[Dict[str, Any]] = None) -> str:
        """Create a new processing pipeline"""
        
        pipeline_id = str(uuid.uuid4())
        
        if pipeline_type not in self.pipeline_templates:
            raise PipelineError(f"Unknown pipeline type: {pipeline_type}")
            
        template = self.pipeline_templates[pipeline_type]
        
        # Create pipeline stages
        stages = []
        stage_definitions = custom_stages or template["stages"]
        
        for i, stage_def in enumerate(stage_definitions):
            stage = PipelineStage(
                id=f"stage_{i+1:02d}",
                name=stage_def["name"],
                stage_type=StageType(stage_def["type"]),
                processor=self._get_stage_processor(stage_def["type"]),
                input_requirements=stage_def.get("input_requirements", []),
                output_schema=stage_def.get("output_schema", {}),
                dependencies=stage_def.get("dependencies", [f"stage_{i:02d}"] if i > 0 else []),
                parameters=stage_def.get("parameters", {}),
                resource_requirements=stage_def.get("resource_requirements", {}),
                timeout=stage_def.get("timeout")
            )
            stages.append(stage)
            
        # Create pipeline
        pipeline = Pipeline(
            id=pipeline_id,
            name=name,
            description=f"{pipeline_type.value.replace('_', ' ').title()} Pipeline",
            pipeline_type=pipeline_type,
            version="1.0.0",
            stages=stages,
            global_parameters=parameters or {},
            resource_limits=template["resource_requirements"],
            monitoring_config={
                "metrics_enabled": True,
                "progress_tracking": True,
                "resource_monitoring": True
            }
        )
        
        # Register pipeline
        self.pipeline_registry[pipeline_id] = pipeline
        
        logger.info(f"Created pipeline: {name} ({pipeline_id}) with {len(stages)} stages")
        
        return pipeline_id
        
    async def execute_pipeline(self,
                             pipeline_id: str,
                             input_data: Dict[str, Any],
                             execution_options: Optional[Dict[str, Any]] = None) -> str:
        """Execute a processing pipeline"""
        
        if pipeline_id not in self.pipeline_registry:
            raise PipelineError(f"Pipeline {pipeline_id} not found")
            
        pipeline = self.pipeline_registry[pipeline_id]
        execution_id = str(uuid.uuid4())
        
        # Check resource availability
        if not await self._check_resource_availability(pipeline):
            raise PipelineError("Insufficient resources to execute pipeline")
            
        # Allocate resources
        await self._allocate_pipeline_resources(execution_id, pipeline)
        
        # Create execution context
        execution = PipelineExecution(
            pipeline_id=pipeline_id,
            execution_id=execution_id,
            input_data=input_data,
            start_time=datetime.now(),
            status="running"
        )
        
        self.active_executions[execution_id] = execution
        
        try:
            logger.info(f"Starting pipeline execution: {pipeline.name} ({execution_id})")
            
            # Convert pipeline to workflow
            workflow = await self._create_pipeline_workflow(pipeline, execution, execution_options or {})
            workflow_id = self.workflow_engine.register_workflow(workflow)
            execution.workflow_execution_id = workflow_id
            
            # Execute workflow
            workflow_result = await self.workflow_engine.execute_workflow(
                workflow_id,
                initial_variables={
                    "pipeline_id": pipeline_id,
                    "execution_id": execution_id,
                    "input_data": input_data,
                    "pipeline_config": pipeline.__dict__
                }
            )
            
            # Update execution status
            execution.status = workflow_result.status.value
            execution.end_time = datetime.now()
            execution.stage_results = workflow_result.results
            
            # Calculate execution metrics
            execution.execution_metrics = {
                "total_duration": (execution.end_time - execution.start_time).total_seconds(),
                "stages_completed": len([r for r in workflow_result.results.values() if r]),
                "success_rate": 1.0 if workflow_result.status.value == "completed" else 0.0
            }
            
            logger.info(f"Pipeline execution completed: {execution.status}")
            
            return execution_id
            
        except Exception as e:
            execution.status = "failed"
            execution.end_time = datetime.now()
            execution.error_log.append(str(e))
            
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
            
        finally:
            # Release resources
            await self._release_pipeline_resources(execution_id)
            
    async def monitor_pipeline_execution(self, execution_id: str) -> Dict[str, Any]:
        """Monitor pipeline execution progress"""
        
        if execution_id not in self.active_executions:
            raise PipelineError(f"Execution {execution_id} not found")
            
        execution = self.active_executions[execution_id]
        pipeline = self.pipeline_registry[execution.pipeline_id]
        
        # Get workflow status if available
        workflow_status = None
        if execution.workflow_execution_id:
            workflow_status = self.workflow_engine.get_execution_status(execution.workflow_execution_id)
            
        monitoring_data = {
            "execution_id": execution_id,
            "pipeline_name": pipeline.name,
            "status": execution.status,
            "progress_percentage": self._calculate_progress_percentage(execution, workflow_status),
            "current_stage": self._get_current_stage(execution, workflow_status),
            "elapsed_time": (datetime.now() - execution.start_time).total_seconds() if execution.start_time else 0,
            "estimated_remaining": self._estimate_remaining_time(execution, pipeline),
            "resource_usage": self._get_resource_usage(execution_id),
            "stage_metrics": self._get_stage_metrics(execution),
            "performance_metrics": self._get_performance_metrics(execution)
        }
        
        return monitoring_data
        
    def create_custom_stage(self,
                           stage_name: str,
                           stage_type: StageType,
                           processor_function: Callable,
                           input_requirements: List[str],
                           output_schema: Dict[str, Any],
                           parameters: Optional[Dict[str, Any]] = None) -> PipelineStage:
        """Create a custom pipeline stage"""
        
        stage = PipelineStage(
            id=str(uuid.uuid4()),
            name=stage_name,
            stage_type=stage_type,
            processor=processor_function,
            input_requirements=input_requirements,
            output_schema=output_schema,
            parameters=parameters or {}
        )
        
        return stage
        
    def create_research_pipeline(self,
                               experiment_name: str,
                               research_config: Dict[str, Any]) -> str:
        """Create a specialized research pipeline"""
        
        # Custom stages for research pipelines
        research_stages = [
            {
                "type": StageType.DATA_INGESTION.value,
                "name": "Load Research Dataset",
                "parameters": {
                    "dataset_path": research_config.get("dataset_path"),
                    "data_format": research_config.get("data_format", "csv")
                }
            },
            {
                "type": StageType.DATA_VALIDATION.value,
                "name": "Validate Research Data",
                "parameters": {
                    "validation_rules": research_config.get("validation_rules", []),
                    "quality_threshold": research_config.get("quality_threshold", 0.95)
                }
            },
            {
                "type": StageType.FEATURE_EXTRACTION.value,
                "name": "Extract Research Features",
                "parameters": {
                    "feature_config": research_config.get("feature_config", {}),
                    "feature_selection": research_config.get("feature_selection", "auto")
                }
            },
            {
                "type": StageType.MODEL_TRAINING.value,
                "name": "Train Research Models",
                "parameters": {
                    "model_types": research_config.get("model_types", ["random_forest", "neural_network"]),
                    "cross_validation": research_config.get("cross_validation", True),
                    "hyperparameter_tuning": research_config.get("hyperparameter_tuning", True)
                }
            },
            {
                "type": StageType.MODEL_EVALUATION.value,
                "name": "Evaluate Model Performance",
                "parameters": {
                    "evaluation_metrics": research_config.get("evaluation_metrics", ["accuracy", "precision", "recall", "f1"]),
                    "statistical_tests": research_config.get("statistical_tests", ["t_test", "wilcoxon"])
                }
            },
            {
                "type": StageType.RESULT_ANALYSIS.value,
                "name": "Analyze Research Results",
                "parameters": {
                    "significance_level": research_config.get("significance_level", 0.05),
                    "confidence_interval": research_config.get("confidence_interval", 0.95),
                    "effect_size_calculation": research_config.get("effect_size_calculation", True)
                }
            },
            {
                "type": StageType.OUTPUT_GENERATION.value,
                "name": "Generate Research Report",
                "parameters": {
                    "report_format": research_config.get("report_format", "pdf"),
                    "include_visualizations": research_config.get("include_visualizations", True),
                    "statistical_appendix": research_config.get("statistical_appendix", True)
                }
            }
        ]
        
        pipeline_id = self.create_pipeline(
            PipelineType.RESEARCH_EXPERIMENT,
            f"Research: {experiment_name}",
            custom_stages=research_stages,
            parameters=research_config
        )
        
        # Add research-specific metadata
        pipeline = self.pipeline_registry[pipeline_id]
        pipeline.metadata.update({
            "research_type": "cybersecurity",
            "experimental_design": research_config.get("experimental_design", "comparative"),
            "hypothesis": research_config.get("hypothesis", ""),
            "expected_outcome": research_config.get("expected_outcome", "")
        })
        
        return pipeline_id
        
    def get_pipeline_library(self) -> Dict[str, Dict[str, Any]]:
        """Get library of available pipelines"""
        
        library = {}
        
        for pipeline_id, pipeline in self.pipeline_registry.items():
            library[pipeline_id] = {
                "id": pipeline_id,
                "name": pipeline.name,
                "description": pipeline.description,
                "type": pipeline.pipeline_type.value,
                "version": pipeline.version,
                "stage_count": len(pipeline.stages),
                "estimated_duration": self._estimate_pipeline_duration(pipeline),
                "resource_requirements": pipeline.resource_limits,
                "tags": pipeline.tags
            }
            
        return library
        
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics for pipeline executions"""
        
        analytics = {
            "total_pipelines": len(self.pipeline_registry),
            "active_executions": len(self.active_executions),
            "pipeline_type_distribution": {},
            "average_execution_time": 0,
            "success_rate": 0,
            "resource_utilization": {},
            "stage_performance": {}
        }
        
        # Analyze pipeline types
        for pipeline in self.pipeline_registry.values():
            pipeline_type = pipeline.pipeline_type.value
            analytics["pipeline_type_distribution"][pipeline_type] = \
                analytics["pipeline_type_distribution"].get(pipeline_type, 0) + 1
                
        # Analyze executions
        total_duration = 0
        successful_executions = 0
        completed_executions = 0
        
        for execution in self.active_executions.values():
            if execution.end_time and execution.start_time:
                duration = (execution.end_time - execution.start_time).total_seconds()
                total_duration += duration
                completed_executions += 1
                
                if execution.status == "completed":
                    successful_executions += 1
                    
        if completed_executions > 0:
            analytics["average_execution_time"] = total_duration / completed_executions
            analytics["success_rate"] = successful_executions / completed_executions
            
        # Resource utilization
        total_cpu = self.resource_pool["cpu_cores"]
        total_memory = self.resource_pool["memory_gb"]
        
        allocated_cpu = sum(res.get("cpu_cores", 0) for res in self.allocated_resources.values())
        allocated_memory = sum(res.get("memory_gb", 0) for res in self.allocated_resources.values())
        
        analytics["resource_utilization"] = {
            "cpu_utilization": allocated_cpu / total_cpu if total_cpu > 0 else 0,
            "memory_utilization": allocated_memory / total_memory if total_memory > 0 else 0
        }
        
        return analytics
        
    async def _create_pipeline_workflow(self,
                                      pipeline: Pipeline,
                                      execution: PipelineExecution,
                                      options: Dict[str, Any]) -> Workflow:
        """Convert pipeline to workflow for execution"""
        
        workflow = Workflow(
            name=f"Pipeline: {pipeline.name}",
            description=f"Execution workflow for {pipeline.name} pipeline"
        )
        
        # Create workflow steps from pipeline stages
        for stage in pipeline.stages:
            step = WorkflowStep(
                id=stage.id,
                name=stage.name,
                description=f"Execute {stage.name} stage",
                handler=self._create_stage_handler(stage),
                dependencies=stage.dependencies,
                parameters={
                    "stage_config": stage.__dict__,
                    "pipeline_config": pipeline.__dict__,
                    "execution_context": execution.__dict__
                },
                timeout=stage.timeout,
                max_retries=stage.retry_policy.get("max_retries", 3)
            )
            
            workflow.add_step(step)
            
        return workflow
        
    def _create_stage_handler(self, stage: PipelineStage) -> Callable:
        """Create handler function for pipeline stage"""
        
        async def stage_handler(context: Dict[str, Any]) -> Dict[str, Any]:
            """Execute pipeline stage"""
            
            stage_context = context.get("stage_config", {})
            pipeline_context = context.get("pipeline_config", {})
            execution_context = context.get("execution_context", {})
            
            # Get input data
            input_data = context.get("previous_results", {})
            
            # Check cache if enabled
            cache_key = None
            if stage.caching_enabled:
                cache_key = self._generate_cache_key(stage, input_data, stage_context)
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    logger.info(f"Using cached result for stage {stage.id}")
                    return cached_result
                    
            # Execute stage processor
            logger.info(f"Executing stage: {stage.name}")
            
            stage_input = {
                "input_data": input_data,
                "stage_parameters": stage.parameters,
                "global_parameters": pipeline_context.get("global_parameters", {}),
                "execution_id": execution_context.get("execution_id")
            }
            
            # Call the stage processor
            if asyncio.iscoroutinefunction(stage.processor):
                result = await stage.processor(stage_input)
            else:
                result = stage.processor(stage_input)
                
            # Validate result against schema
            if stage.output_schema:
                self._validate_stage_output(result, stage.output_schema, stage.id)
                
            # Cache result if enabled
            if stage.caching_enabled and cache_key:
                await self.cache_manager.set(cache_key, result, ttl=3600)  # 1 hour TTL
                
            logger.info(f"Stage {stage.name} completed successfully")
            
            return result
            
        return stage_handler
        
    def _get_stage_processor(self, stage_type: str) -> Callable:
        """Get appropriate processor function for stage type"""
        
        processors = {
            StageType.DATA_INGESTION.value: self._process_data_ingestion,
            StageType.DATA_VALIDATION.value: self._process_data_validation,
            StageType.DATA_TRANSFORMATION.value: self._process_data_transformation,
            StageType.FEATURE_EXTRACTION.value: self._process_feature_extraction,
            StageType.MODEL_TRAINING.value: self._process_model_training,
            StageType.MODEL_EVALUATION.value: self._process_model_evaluation,
            StageType.RESULT_ANALYSIS.value: self._process_result_analysis,
            StageType.OUTPUT_GENERATION.value: self._process_output_generation
        }
        
        return processors.get(stage_type, self._default_processor)
        
    # Stage processor implementations
    async def _process_data_ingestion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process data ingestion stage"""
        # Implementation would load and prepare data
        return {"data_loaded": True, "record_count": 1000, "data_quality": 0.95}
        
    async def _process_data_validation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process data validation stage"""
        # Implementation would validate data quality and completeness
        return {"validation_passed": True, "quality_score": 0.96, "issues_found": 2}
        
    async def _process_data_transformation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process data transformation stage"""
        # Implementation would transform and clean data
        return {"transformation_completed": True, "features_extracted": 25, "data_size_mb": 15.7}
        
    async def _process_feature_extraction(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process feature extraction stage"""
        # Implementation would extract relevant features
        return {"features_extracted": 50, "feature_importance": {"feat1": 0.85, "feat2": 0.72}}
        
    async def _process_model_training(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process model training stage"""
        # Implementation would train ML models
        return {"model_trained": True, "training_accuracy": 0.94, "validation_accuracy": 0.91}
        
    async def _process_model_evaluation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process model evaluation stage"""
        # Implementation would evaluate model performance
        return {"evaluation_completed": True, "test_accuracy": 0.89, "precision": 0.92, "recall": 0.87}
        
    async def _process_result_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process result analysis stage"""
        # Implementation would analyze results
        return {"analysis_completed": True, "statistical_significance": True, "p_value": 0.023}
        
    async def _process_output_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process output generation stage"""
        # Implementation would generate final outputs
        return {"output_generated": True, "report_pages": 15, "visualizations": 8}
        
    async def _default_processor(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default processor for unknown stage types"""
        return {"processed": True, "stage_completed": True}
        
    # Helper methods
    async def _check_resource_availability(self, pipeline: Pipeline) -> bool:
        """Check if sufficient resources are available for pipeline"""
        
        required_cpu = pipeline.resource_limits.get("cpu_cores", 1)
        required_memory = pipeline.resource_limits.get("memory_gb", 1)
        required_gpu = pipeline.resource_limits.get("gpu_count", 0)
        
        allocated_cpu = sum(res.get("cpu_cores", 0) for res in self.allocated_resources.values())
        allocated_memory = sum(res.get("memory_gb", 0) for res in self.allocated_resources.values())
        allocated_gpu = sum(res.get("gpu_count", 0) for res in self.allocated_resources.values())
        
        available_cpu = self.resource_pool["cpu_cores"] - allocated_cpu
        available_memory = self.resource_pool["memory_gb"] - allocated_memory
        available_gpu = self.resource_pool["gpu_count"] - allocated_gpu
        
        return (available_cpu >= required_cpu and 
                available_memory >= required_memory and
                available_gpu >= required_gpu)
                
    async def _allocate_pipeline_resources(self, execution_id: str, pipeline: Pipeline):
        """Allocate resources for pipeline execution"""
        
        self.allocated_resources[execution_id] = {
            "cpu_cores": pipeline.resource_limits.get("cpu_cores", 1),
            "memory_gb": pipeline.resource_limits.get("memory_gb", 1), 
            "gpu_count": pipeline.resource_limits.get("gpu_count", 0),
            "allocated_at": datetime.now()
        }
        
        logger.info(f"Allocated resources for execution {execution_id}: {self.allocated_resources[execution_id]}")
        
    async def _release_pipeline_resources(self, execution_id: str):
        """Release resources after pipeline execution"""
        
        if execution_id in self.allocated_resources:
            released = self.allocated_resources.pop(execution_id)
            logger.info(f"Released resources for execution {execution_id}: {released}")
            
    def _generate_cache_key(self, stage: PipelineStage, input_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate cache key for stage result"""
        
        key_components = [
            stage.id,
            stage.name,
            str(hash(str(sorted(input_data.items())))),
            str(hash(str(sorted(stage.parameters.items()))))
        ]
        
        return "pipeline_stage:" + ":".join(key_components)
        
    def _validate_stage_output(self, result: Dict[str, Any], schema: Dict[str, Any], stage_id: str):
        """Validate stage output against schema"""
        
        # Simple validation - could be more sophisticated
        required_fields = schema.get("required", [])
        
        for field in required_fields:
            if field not in result:
                raise PipelineError(f"Stage {stage_id} missing required output field: {field}")
                
    def _calculate_progress_percentage(self, execution: PipelineExecution, workflow_status) -> float:
        """Calculate execution progress percentage"""
        
        if workflow_status:
            total_steps = workflow_status.total_steps
            completed_steps = workflow_status.steps_completed
            
            if total_steps > 0:
                return (completed_steps / total_steps) * 100
                
        return 0.0
        
    def _get_current_stage(self, execution: PipelineExecution, workflow_status) -> Optional[str]:
        """Get currently executing stage"""
        
        if workflow_status:
            return workflow_status.current_step
            
        return None
        
    def _estimate_remaining_time(self, execution: PipelineExecution, pipeline: Pipeline) -> float:
        """Estimate remaining execution time"""
        
        if not execution.start_time:
            return 0
            
        elapsed = (datetime.now() - execution.start_time).total_seconds()
        
        # Simple estimation based on pipeline template
        template = self.pipeline_templates.get(pipeline.pipeline_type)
        if template and "typical_duration" in template:
            typical_duration = template["typical_duration"].total_seconds()
            return max(0, typical_duration - elapsed)
            
        return 0
        
    def _get_resource_usage(self, execution_id: str) -> Dict[str, Any]:
        """Get current resource usage for execution"""
        
        allocated = self.allocated_resources.get(execution_id, {})
        
        return {
            "cpu_cores_allocated": allocated.get("cpu_cores", 0),
            "memory_gb_allocated": allocated.get("memory_gb", 0),
            "gpu_count_allocated": allocated.get("gpu_count", 0)
        }
        
    def _get_stage_metrics(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Get metrics for individual stages"""
        
        # This would collect detailed stage performance metrics
        return {"stages_analyzed": len(execution.stage_results)}
        
    def _get_performance_metrics(self, execution: PipelineExecution) -> Dict[str, Any]:
        """Get overall performance metrics"""
        
        return execution.execution_metrics
        
    def _estimate_pipeline_duration(self, pipeline: Pipeline) -> str:
        """Estimate pipeline execution duration"""
        
        template = self.pipeline_templates.get(pipeline.pipeline_type)
        if template and "typical_duration" in template:
            duration = template["typical_duration"]
            return str(duration)
            
        return "Unknown"