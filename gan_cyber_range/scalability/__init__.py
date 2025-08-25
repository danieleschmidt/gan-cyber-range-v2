"""
Advanced scalability module for high-performance cyber range deployment.

This module provides horizontal scaling, load balancing, distributed computing,
and performance optimization features for enterprise-scale cyber ranges.
"""

# Import new intelligent scaling components (no external dependencies)
try:
    from .intelligent_scaling import IntelligentAutoScaler, PredictiveScaler, LoadBalancer
    INTELLIGENT_SCALING_AVAILABLE = True
except ImportError:
    INTELLIGENT_SCALING_AVAILABLE = False

# Import original components with graceful fallback
try:
    from .auto_scaler import AutoScaler, ScalingPolicy, ScalingMetrics
except ImportError:
    AutoScaler = None
    ScalingPolicy = None
    ScalingMetrics = None

try:
    from .load_balancer import LoadBalancer as OriginalLoadBalancer, LoadBalancingStrategy
except ImportError:
    OriginalLoadBalancer = None
    LoadBalancingStrategy = None

try:
    from .distributed_computing import DistributedCompute, ComputeCluster
except ImportError:
    DistributedCompute = None
    ComputeCluster = None

try:
    from .performance_optimizer import PerformanceOptimizer as OriginalPerformanceOptimizer, OptimizationStrategy
except ImportError:
    OriginalPerformanceOptimizer = None
    OptimizationStrategy = None

try:
    from .cache_manager import CacheManager, CacheStrategy, CachePolicy
except ImportError:
    CacheManager = None
    CacheStrategy = None
    CachePolicy = None

try:
    from .resource_manager import ResourceManager, ResourcePool, ResourceAllocation
except ImportError:
    ResourceManager = None
    ResourcePool = None
    ResourceAllocation = None

__all__ = [
    "IntelligentAutoScaler",
    "PredictiveScaler", 
    "LoadBalancer",
    "AutoScaler",
    "ScalingPolicy", 
    "ScalingMetrics",
    "LoadBalancingStrategy",
    "DistributedCompute",
    "ComputeCluster",
    "OptimizationStrategy",
    "CacheManager", 
    "CacheStrategy",
    "CachePolicy",
    "ResourceManager",
    "ResourcePool",
    "ResourceAllocation",
    "INTELLIGENT_SCALING_AVAILABLE"
]