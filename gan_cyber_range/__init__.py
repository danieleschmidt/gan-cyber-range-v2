"""
GAN-Cyber-Range-v2: Advanced Cybersecurity Training Platform

A second-generation adversarial cyber range that combines GAN-based attack generation 
with LLM-driven red team curricula for comprehensive cybersecurity training and research.
"""

__version__ = "2.0.0"
__author__ = "Daniel Schmidt"
__license__ = "MIT"

# Core imports (only import what exists)
from .core.attack_gan import AttackGAN
from .core.cyber_range import CyberRange  
from .core.network_sim import NetworkTopology
from .core.attack_engine import AttackSimulator

# Red team imports (only import what exists)
from .red_team.llm_adversary import RedTeamLLM

# Factory imports
from .factories.attack_factory import AttackFactory, AttackConfig
from .factories.range_factory import CyberRangeFactory, RangeTemplateConfig
from .factories.network_factory import NetworkFactory, NetworkTemplate
from .factories.training_factory import TrainingFactory, TrainingProgram, TrainingModule

# Orchestration imports
from .orchestration.workflow_engine import WorkflowEngine, Workflow, WorkflowStep
from .orchestration.scenario_orchestrator import ScenarioOrchestrator, TrainingScenario
from .orchestration.pipeline_manager import PipelineManager, Pipeline, PipelineStage

# Optimization imports
from .optimization.cache_optimizer import CacheOptimizer, CacheStrategy
from .optimization.query_optimizer import QueryOptimizer, QueryPlan
from .optimization.resource_pool import ResourcePool, ResourceManager
from .optimization.performance_monitor import PerformanceMonitor, PerformanceProfiler

__all__ = [
    # Core
    "AttackGAN",
    "CyberRange", 
    "NetworkTopology",
    "AttackSimulator",
    
    # Red Team
    "RedTeamLLM",
    
    # Factories
    "AttackFactory",
    "AttackConfig", 
    "CyberRangeFactory",
    "RangeTemplateConfig",
    "NetworkFactory",
    "NetworkTemplate",
    "TrainingFactory",
    "TrainingProgram",
    "TrainingModule",
    
    # Orchestration
    "WorkflowEngine",
    "Workflow",
    "WorkflowStep",
    "ScenarioOrchestrator",
    "TrainingScenario",
    "PipelineManager",
    "Pipeline",
    "PipelineStage",
    
    # Optimization
    "CacheOptimizer",
    "CacheStrategy",
    "QueryOptimizer",
    "QueryPlan",
    "ResourcePool",
    "ResourceManager",
    "PerformanceMonitor",
    "PerformanceProfiler",
]