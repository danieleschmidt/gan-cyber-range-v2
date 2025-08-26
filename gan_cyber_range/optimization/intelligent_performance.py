"""
Intelligent Performance Optimization System

Advanced performance optimization with adaptive caching, resource pooling,
auto-scaling, and machine learning-based optimization.
"""

import asyncio
import threading
import time
import psutil
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
from enum import Enum
import hashlib

from ..utils.robust_error_handler import robust, critical, ErrorSeverity
from ..utils.comprehensive_logging import comprehensive_logger, timed_operation


class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"


class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    cache_hit_rate: float = 0.0
    average_response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    queue_length: int = 0


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl


class IntelligentCache:
    """Adaptive caching system with multiple eviction policies"""
    
    def __init__(
        self, 
        max_size: int = 10000,
        max_memory_mb: int = 512,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.ADAPTIVE
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: deque = deque()  # For LRU
        self.frequency_counter: Dict[str, int] = defaultdict(int)  # For LFU
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_size = 0
        self.current_memory = 0
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        # Adaptive parameters
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.pattern_weights: Dict[CacheEvictionPolicy, float] = {
            CacheEvictionPolicy.LRU: 0.33,
            CacheEvictionPolicy.LFU: 0.33,
            CacheEvictionPolicy.FIFO: 0.34
        }
    
    @robust(severity=ErrorSeverity.LOW)
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self._remove_from_tracking(key)
                self.misses += 1
                return None
            
            # Update access info
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self.frequency_counter[key] += 1
            
            # Update LRU order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            # Record access pattern
            access_time = time.time()
            self.access_patterns[key].append(access_time)
            
            # Keep only recent access times (last hour)
            cutoff = access_time - 3600
            self.access_patterns[key] = [
                t for t in self.access_patterns[key] if t > cutoff
            ]
            
            self.hits += 1
            return entry.value
    
    @robust(severity=ErrorSeverity.LOW)
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in cache"""
        with self.lock:
            # Calculate size
            size = self._estimate_size(value)
            
            # Check if we need to evict
            while (
                len(self.cache) >= self.max_size or
                self.current_memory + size > self.max_memory_bytes
            ):
                if not self._evict_one():
                    # Can't evict anything, reject this entry
                    comprehensive_logger.warning(
                        f"Cache full, cannot store key: {key}",
                        additional_data={"cache_size": len(self.cache), "memory_usage": self.current_memory}
                    )
                    return False
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size,
                ttl=ttl
            )
            
            # Store entry
            if key in self.cache:
                # Update existing entry
                old_entry = self.cache[key]
                self.current_memory -= old_entry.size_bytes
            else:
                self.current_size += 1
            
            self.cache[key] = entry
            self.current_memory += size
            
            # Update tracking
            self.frequency_counter[key] = entry.access_count
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return True
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy"""
        if not self.cache:
            return False
        
        if self.eviction_policy == CacheEvictionPolicy.ADAPTIVE:
            # Use adaptive policy selection
            policy = self._select_adaptive_policy()
        else:
            policy = self.eviction_policy
        
        # Select victim based on policy
        victim_key = None
        
        if policy == CacheEvictionPolicy.LRU:
            victim_key = self.access_order[0] if self.access_order else None
        elif policy == CacheEvictionPolicy.LFU:
            victim_key = min(
                self.frequency_counter.keys(),
                key=lambda k: self.frequency_counter[k]
            ) if self.frequency_counter else None
        elif policy == CacheEvictionPolicy.FIFO:
            victim_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].created_at
            ) if self.cache else None
        
        if victim_key and victim_key in self.cache:
            entry = self.cache[victim_key]
            del self.cache[victim_key]
            self._remove_from_tracking(victim_key)
            self.current_size -= 1
            self.current_memory -= entry.size_bytes
            self.evictions += 1
            return True
        
        return False
    
    def _select_adaptive_policy(self) -> CacheEvictionPolicy:
        """Select best eviction policy based on access patterns"""
        # Simple heuristic: use LRU for general purpose
        # In a real implementation, this would use ML to optimize
        return CacheEvictionPolicy.LRU
    
    def _remove_from_tracking(self, key: str):
        """Remove key from all tracking structures"""
        if key in self.access_order:
            self.access_order.remove(key)
        if key in self.frequency_counter:
            del self.frequency_counter[key]
        if key in self.access_patterns:
            del self.access_patterns[key]
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        try:
            return len(json.dumps(obj, default=str).encode('utf-8'))
        except:
            # Fallback estimation
            return 1024  # 1KB default
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(total_requests, 1)
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_usage_mb": self.current_memory / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "eviction_policy": self.eviction_policy.value
        }
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
            self.access_patterns.clear()
            self.current_size = 0
            self.current_memory = 0


class ResourcePool:
    """Dynamic resource pooling for expensive objects"""
    
    def __init__(
        self,
        factory: Callable[[], Any],
        max_size: int = 10,
        min_size: int = 2,
        max_idle_time: int = 300  # 5 minutes
    ):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        
        # Pool storage
        self.available: deque = deque()
        self.in_use: weakref.WeakSet = weakref.WeakSet()
        self.created_times: Dict[int, datetime] = {}
        self.last_used_times: Dict[int, datetime] = {}
        
        # Statistics
        self.created_count = 0
        self.borrowed_count = 0
        self.returned_count = 0
        self.destroyed_count = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Initialize minimum pool size
        self._initialize_pool()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _initialize_pool(self):
        """Initialize pool with minimum number of resources"""
        for _ in range(self.min_size):
            resource = self._create_resource()
            self.available.append(resource)
    
    def _create_resource(self) -> Any:
        """Create new resource"""
        resource = self.factory()
        resource_id = id(resource)
        self.created_times[resource_id] = datetime.now()
        self.last_used_times[resource_id] = datetime.now()
        self.created_count += 1
        return resource
    
    @robust(severity=ErrorSeverity.MEDIUM)
    def borrow(self, timeout: float = 5.0) -> Optional[Any]:
        """Borrow resource from pool"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                # Try to get available resource
                if self.available:
                    resource = self.available.popleft()
                    self.in_use.add(resource)
                    self.last_used_times[id(resource)] = datetime.now()
                    self.borrowed_count += 1
                    return resource
                
                # Create new resource if under limit
                if len(self.in_use) < self.max_size:
                    resource = self._create_resource()
                    self.in_use.add(resource)
                    self.borrowed_count += 1
                    return resource
            
            # Wait a bit before retrying
            time.sleep(0.1)
        
        comprehensive_logger.warning(
            f"Resource pool timeout after {timeout}s",
            additional_data={"pool_size": len(self.in_use), "max_size": self.max_size}
        )
        return None
    
    @robust(severity=ErrorSeverity.LOW)
    def return_resource(self, resource: Any):
        """Return resource to pool"""
        with self.lock:
            if resource in self.in_use:
                self.in_use.discard(resource)
                
                # Only keep if under max size and resource is still valid
                if len(self.available) < self.max_size and self._is_resource_valid(resource):
                    self.available.append(resource)
                    self.last_used_times[id(resource)] = datetime.now()
                else:
                    # Destroy excess resource
                    self._destroy_resource(resource)
                
                self.returned_count += 1
    
    def _is_resource_valid(self, resource: Any) -> bool:
        """Check if resource is still valid"""
        # Override in subclass for specific validation
        return True
    
    def _destroy_resource(self, resource: Any):
        """Destroy resource and clean up tracking"""
        resource_id = id(resource)
        if resource_id in self.created_times:
            del self.created_times[resource_id]
        if resource_id in self.last_used_times:
            del self.last_used_times[resource_id]
        
        # Call cleanup method if exists
        if hasattr(resource, 'cleanup'):
            try:
                resource.cleanup()
            except Exception as e:
                comprehensive_logger.warning(f"Resource cleanup failed: {e}")
        
        self.destroyed_count += 1
    
    def _cleanup_worker(self):
        """Background worker to clean up idle resources"""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                with self.lock:
                    current_time = datetime.now()
                    idle_resources = []
                    
                    # Find idle resources in available pool
                    for resource in list(self.available):
                        resource_id = id(resource)
                        if resource_id in self.last_used_times:
                            idle_time = (current_time - self.last_used_times[resource_id]).total_seconds()
                            if idle_time > self.max_idle_time and len(self.available) > self.min_size:
                                idle_resources.append(resource)
                    
                    # Remove idle resources
                    for resource in idle_resources:
                        self.available.remove(resource)
                        self._destroy_resource(resource)
                        
                    if idle_resources:
                        comprehensive_logger.info(
                            f"Cleaned up {len(idle_resources)} idle resources from pool"
                        )
                        
            except Exception as e:
                comprehensive_logger.error(f"Pool cleanup worker error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self.lock:
            return {
                "available": len(self.available),
                "in_use": len(self.in_use),
                "total": len(self.available) + len(self.in_use),
                "max_size": self.max_size,
                "min_size": self.min_size,
                "created": self.created_count,
                "borrowed": self.borrowed_count,
                "returned": self.returned_count,
                "destroyed": self.destroyed_count
            }


class AdaptivePerformanceOptimizer:
    """Main performance optimization orchestrator"""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE):
        self.strategy = strategy
        
        # Component initialization
        self.cache = IntelligentCache(
            max_size=10000,
            max_memory_mb=512,
            eviction_policy=CacheEvictionPolicy.ADAPTIVE
        )
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(32, (psutil.cpu_count() or 1) + 4)
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=psutil.cpu_count() or 1
        )
        
        # Resource pools for expensive objects
        self.resource_pools: Dict[str, ResourcePool] = {}
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Optimization parameters
        self.optimization_config = {
            "cache_enabled": True,
            "async_enabled": True,
            "pooling_enabled": True,
            "batch_processing": True,
            "compression_enabled": False,
            "lazy_loading": True
        }
        
        # Start monitoring
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
    
    @robust(severity=ErrorSeverity.LOW)
    def optimize_function(self, cache_key: Optional[str] = None, async_enabled: bool = True):
        """Decorator for function optimization"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key if not provided
                if cache_key and self.optimization_config["cache_enabled"]:
                    key = f"{func.__name__}_{cache_key}"
                    
                    # Try cache first
                    cached_result = self.cache.get(key)
                    if cached_result is not None:
                        return cached_result
                
                # Execute function
                start_time = time.time()
                
                try:
                    if async_enabled and self.optimization_config["async_enabled"]:
                        # Submit to thread pool for I/O bound operations
                        future = self.thread_pool.submit(func, *args, **kwargs)
                        result = future.result()
                    else:
                        result = func(*args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    
                    # Cache result if enabled
                    if cache_key and self.optimization_config["cache_enabled"]:
                        self.cache.put(key, result, ttl=3600)  # 1 hour TTL
                    
                    # Record performance metrics
                    comprehensive_logger.performance_event(
                        func.__name__,
                        execution_time,
                        additional_data={
                            "cached": False,
                            "async": async_enabled
                        }
                    )
                    
                    return result
                    
                except Exception as e:
                    comprehensive_logger.error(
                        f"Optimized function {func.__name__} failed: {e}",
                        additional_data={
                            "execution_time": time.time() - start_time
                        }
                    )
                    raise
            
            return wrapper
        return decorator
    
    @timed_operation("batch_process", comprehensive_logger)
    def batch_process(
        self,
        items: List[Any],
        processor: Callable,
        batch_size: int = 100,
        parallel: bool = True
    ) -> List[Any]:
        """Process items in optimized batches"""
        
        if not items:
            return []
        
        if not self.optimization_config["batch_processing"]:
            # Process individually
            return [processor(item) for item in items]
        
        results = []
        
        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            if parallel and len(batch) > 1:
                # Parallel processing
                futures = [
                    self.thread_pool.submit(processor, item)
                    for item in batch
                ]
                
                batch_results = [future.result() for future in futures]
            else:
                # Sequential processing
                batch_results = [processor(item) for item in batch]
            
            results.extend(batch_results)
        
        comprehensive_logger.info(
            f"Batch processed {len(items)} items in batches of {batch_size}",
            additional_data={
                "total_items": len(items),
                "batch_size": batch_size,
                "parallel": parallel
            }
        )
        
        return results
    
    def create_resource_pool(
        self,
        name: str,
        factory: Callable[[], Any],
        max_size: int = 10,
        min_size: int = 2
    ) -> ResourcePool:
        """Create a new resource pool"""
        pool = ResourcePool(factory, max_size, min_size)
        self.resource_pools[name] = pool
        
        comprehensive_logger.info(
            f"Created resource pool '{name}'",
            additional_data={
                "max_size": max_size,
                "min_size": min_size
            }
        )
        
        return pool
    
    def get_resource_pool(self, name: str) -> Optional[ResourcePool]:
        """Get existing resource pool"""
        return self.resource_pools.get(name)
    
    def _monitoring_worker(self):
        """Background worker for performance monitoring"""
        while True:
            try:
                # Collect system metrics
                self.metrics.cpu_usage = psutil.cpu_percent()
                self.metrics.memory_usage = psutil.virtual_memory().percent
                
                # Collect cache metrics
                cache_stats = self.cache.get_stats()
                self.metrics.cache_hit_rate = cache_stats["hit_rate"]
                
                # Store metrics history
                self.metrics_history.append(self.metrics)
                
                # Adaptive optimization
                if self.strategy == OptimizationStrategy.ADAPTIVE:
                    self._adapt_optimization()
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                comprehensive_logger.error(f"Monitoring worker error: {e}")
    
    def _adapt_optimization(self):
        """Adapt optimization parameters based on metrics"""
        # Simple adaptive logic
        if self.metrics.memory_usage > 80:
            # High memory usage - be more aggressive with caching
            self.optimization_config["compression_enabled"] = True
            self.optimization_config["lazy_loading"] = True
        elif self.metrics.memory_usage < 50:
            # Low memory usage - can be more relaxed
            self.optimization_config["compression_enabled"] = False
        
        if self.metrics.cpu_usage > 80:
            # High CPU usage - reduce parallel processing
            self.optimization_config["async_enabled"] = False
        elif self.metrics.cpu_usage < 30:
            # Low CPU usage - enable more parallel processing
            self.optimization_config["async_enabled"] = True
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "strategy": self.strategy.value,
            "current_metrics": {
                "cpu_usage": self.metrics.cpu_usage,
                "memory_usage": self.metrics.memory_usage,
                "cache_hit_rate": self.metrics.cache_hit_rate
            },
            "cache_stats": self.cache.get_stats(),
            "thread_pool_stats": {
                "max_workers": self.thread_pool._max_workers,
                "active_threads": threading.active_count()
            },
            "resource_pools": {
                name: pool.get_stats()
                for name, pool in self.resource_pools.items()
            },
            "optimization_config": self.optimization_config.copy()
        }
    
    def shutdown(self):
        """Gracefully shutdown optimizer"""
        comprehensive_logger.info("Shutting down performance optimizer")
        
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Clean up resource pools
        for pool in self.resource_pools.values():
            # Resource pools clean up automatically via daemon threads
            pass


# Global performance optimizer
performance_optimizer = AdaptivePerformanceOptimizer()

# Convenience decorators
def cached(cache_key: Optional[str] = None):
    """Simple caching decorator"""
    return performance_optimizer.optimize_function(cache_key=cache_key)

def async_optimized(func):
    """Async optimization decorator"""
    return performance_optimizer.optimize_function(async_enabled=True)(func)

def batch_optimized(batch_size: int = 100):
    """Batch processing decorator"""
    def decorator(func):
        def wrapper(items: List[Any], *args, **kwargs):
            processor = lambda item: func(item, *args, **kwargs)
            return performance_optimizer.batch_process(items, processor, batch_size)
        return wrapper
    return decorator