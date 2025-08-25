"""
Advanced optimization and performance enhancement components.

Provides intelligent caching, query optimization, resource pooling,
and performance monitoring capabilities.
"""

# Import new adaptive performance components (no external dependencies)
try:
    from .adaptive_performance import AdaptiveResourcePool, PerformanceOptimizer, DefensiveWorkloadManager
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

# Import original components with graceful fallback for missing dependencies
try:
    from .cache_optimizer import CacheOptimizer, CacheStrategy
except ImportError:
    CacheOptimizer = None
    CacheStrategy = None

try:
    from .query_optimizer import QueryOptimizer, QueryPlan
except ImportError:
    QueryOptimizer = None
    QueryPlan = None

try:
    from .resource_pool import ResourcePool, ResourceManager
except ImportError:
    ResourcePool = None
    ResourceManager = None

try:
    from .performance_monitor import PerformanceMonitor, PerformanceProfiler
except ImportError:
    PerformanceMonitor = None
    PerformanceProfiler = None

__all__ = [
    "AdaptiveResourcePool",
    "PerformanceOptimizer",
    "DefensiveWorkloadManager",
    "CacheOptimizer",
    "CacheStrategy",
    "QueryOptimizer", 
    "QueryPlan",
    "ResourcePool",
    "ResourceManager",
    "PerformanceMonitor",
    "PerformanceProfiler",
    "ADAPTIVE_AVAILABLE"
]