"""
GAN-Cyber-Range-v2: Advanced Cybersecurity Training Platform

A second-generation adversarial cyber range that combines GAN-based attack generation 
with LLM-driven red team curricula for comprehensive cybersecurity training and research.
"""

__version__ = "2.0.0"
__author__ = "Daniel Schmidt"
__license__ = "MIT"

# Core imports with optional dependency handling
try:
    from .core.attack_gan import AttackGAN
except ImportError:
    AttackGAN = None

try:
    from .core.cyber_range import CyberRange
except ImportError:
    CyberRange = None

try:
    from .core.network_sim import NetworkTopology
except ImportError:
    NetworkTopology = None

try:
    from .core.attack_engine import AttackSimulator
except ImportError:
    AttackSimulator = None

# Red team imports
try:
    from .red_team.llm_adversary import RedTeamLLM
except ImportError:
    RedTeamLLM = None

# Factory imports
try:
    from .factories.attack_factory import AttackFactory, AttackConfig
except ImportError:
    AttackFactory = None
    AttackConfig = None

try:
    from .factories.range_factory import CyberRangeFactory, RangeTemplateConfig
except ImportError:
    CyberRangeFactory = None
    RangeTemplateConfig = None

try:
    from .factories.network_factory import NetworkFactory, NetworkTemplate
except ImportError:
    NetworkFactory = None
    NetworkTemplate = None

try:
    from .factories.training_factory import TrainingFactory, TrainingProgram, TrainingModule
except ImportError:
    TrainingFactory = None
    TrainingProgram = None
    TrainingModule = None

# Orchestration imports
try:
    from .orchestration.workflow_engine import WorkflowEngine, Workflow, WorkflowStep
except ImportError:
    WorkflowEngine = None
    Workflow = None
    WorkflowStep = None

try:
    from .orchestration.scenario_orchestrator import ScenarioOrchestrator, TrainingScenario
except ImportError:
    ScenarioOrchestrator = None
    TrainingScenario = None

try:
    from .orchestration.pipeline_manager import PipelineManager, Pipeline, PipelineStage
except ImportError:
    PipelineManager = None
    Pipeline = None
    PipelineStage = None

# Optimization imports
try:
    from .optimization.cache_optimizer import CacheOptimizer, CacheStrategy
except ImportError:
    CacheOptimizer = None
    CacheStrategy = None

try:
    from .optimization.query_optimizer import QueryOptimizer, QueryPlan
except ImportError:
    QueryOptimizer = None
    QueryPlan = None

try:
    from .optimization.resource_pool import ResourcePool, ResourceManager
except ImportError:
    ResourcePool = None
    ResourceManager = None

try:
    from .optimization.performance_monitor import PerformanceMonitor, PerformanceProfiler
except ImportError:
    PerformanceMonitor = None
    PerformanceProfiler = None

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