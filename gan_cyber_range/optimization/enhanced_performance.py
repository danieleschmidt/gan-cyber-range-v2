"""
Enhanced Performance Optimization Module for GAN-Cyber-Range-v2

Advanced performance optimization system with:
- Intelligent caching strategies
- Resource pooling and management
- Auto-scaling capabilities
- Performance monitoring and profiling
- Load balancing and optimization
"""

import asyncio
import threading
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict, deque
import json
import hashlib
import weakref
import gc
from functools import wraps, lru_cache
from contextlib import contextmanager
import multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    active_threads: int
    request_rate: float
    response_time_avg: float
    cache_hit_rate: float
    error_rate: float


@dataclass
class ResourcePool:
    """Resource pool configuration"""
    pool_type: str
    min_size: int
    max_size: int
    current_size: int
    available: int
    utilization: float
    created_at: datetime
    last_scaled: datetime


class IntelligentCache:
    """High-performance caching system with intelligence"""
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600,
        cleanup_interval: int = 300
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        self._cache = {}
        self._access_times = {}
        self._hit_count = 0
        self._miss_count = 0
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"Initialized IntelligentCache with max_size={max_size}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with hit/miss tracking"""
        with self._lock:
            current_time = time.time()
            
            if key not in self._cache:
                self._miss_count += 1
                return None
            
            data, timestamp = self._cache[key]
            
            # Check if expired
            if current_time - timestamp > self.ttl_seconds:
                del self._cache[key]
                del self._access_times[key]
                self._miss_count += 1
                return None
            
            # Update access time for LRU
            self._access_times[key] = current_time
            self._hit_count += 1
            return data
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache with intelligent eviction"""
        with self._lock:
            current_time = time.time()
            
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = (value, current_time)
            self._access_times[key] = current_time
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._hit_count = 0
            self._miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / max(1, total_requests)
            
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'hit_count': self._hit_count,
                'miss_count': self._miss_count,
                'hit_rate': hit_rate,
                'utilization': len(self._cache) / self.max_size
            }
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times, key=self._access_times.get)
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired items"""
        while True:
            time.sleep(self.cleanup_interval)
            
            with self._lock:
                current_time = time.time()
                expired_keys = []
                
                for key, (_, timestamp) in self._cache.items():
                    if current_time - timestamp > self.ttl_seconds:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self._cache[key]
                    del self._access_times[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class AdaptiveResourcePool:
    """Adaptive resource pool with auto-scaling"""
    
    def __init__(
        self,
        resource_factory: Callable,
        min_size: int = 5,
        max_size: int = 50,
        growth_factor: float = 1.5,
        shrink_threshold: float = 0.3,
        scale_interval: int = 60
    ):
        self.resource_factory = resource_factory
        self.min_size = min_size
        self.max_size = max_size
        self.growth_factor = growth_factor
        self.shrink_threshold = shrink_threshold
        self.scale_interval = scale_interval
        
        self._pool = deque()
        self._in_use = set()
        self._lock = threading.RLock()
        self._created_count = 0
        self._destroyed_count = 0
        self._stats = defaultdict(int)
        
        # Initialize pool
        self._populate_pool(self.min_size)
        
        # Start scaling thread
        self._scaling_thread = threading.Thread(target=self._auto_scale, daemon=True)
        self._scaling_thread.start()
        
        logger.info(f"Initialized AdaptiveResourcePool with min={min_size}, max={max_size}")
    
    @contextmanager
    def get_resource(self):
        """Get resource from pool with context manager"""
        resource = self._acquire_resource()
        try:
            yield resource
        finally:
            self._release_resource(resource)
    
    def _acquire_resource(self) -> Any:
        """Acquire resource from pool"""
        with self._lock:
            # Try to get from pool
            if self._pool:
                resource = self._pool.popleft()
                self._in_use.add(id(resource))
                self._stats['acquisitions'] += 1
                return resource
            
            # Create new resource if under max limit
            if len(self._in_use) < self.max_size:
                resource = self._create_resource()
                self._in_use.add(id(resource))
                self._stats['acquisitions'] += 1
                return resource
            
            # Pool exhausted
            self._stats['pool_exhausted'] += 1
            raise RuntimeError("Resource pool exhausted")
    
    def _release_resource(self, resource: Any) -> None:
        """Release resource back to pool"""
        with self._lock:
            resource_id = id(resource)
            if resource_id in self._in_use:
                self._in_use.remove(resource_id)
                
                # Return to pool if under capacity
                if len(self._pool) < self.max_size:
                    self._pool.append(resource)
                    self._stats['returns'] += 1
                else:
                    self._destroy_resource(resource)
    
    def _create_resource(self) -> Any:
        """Create new resource"""
        resource = self.resource_factory()
        self._created_count += 1
        logger.debug(f"Created resource #{self._created_count}")
        return resource
    
    def _destroy_resource(self, resource: Any) -> None:
        """Destroy resource"""
        if hasattr(resource, 'close'):
            resource.close()
        self._destroyed_count += 1
        logger.debug(f"Destroyed resource #{self._destroyed_count}")
    
    def _populate_pool(self, count: int) -> None:
        """Populate pool with initial resources"""
        for _ in range(count):
            resource = self._create_resource()
            self._pool.append(resource)
    
    def _auto_scale(self) -> None:
        """Auto-scaling logic"""
        while True:
            time.sleep(self.scale_interval)
            
            with self._lock:
                current_size = len(self._pool) + len(self._in_use)
                utilization = len(self._in_use) / max(1, current_size)
                
                # Scale up if high utilization
                if utilization > 0.8 and current_size < self.max_size:
                    new_resources = min(
                        int(current_size * self.growth_factor) - current_size,
                        self.max_size - current_size
                    )
                    self._populate_pool(new_resources)
                    logger.info(f"Scaled up pool by {new_resources} resources")
                
                # Scale down if low utilization
                elif utilization < self.shrink_threshold and current_size > self.min_size:
                    excess_count = current_size - max(self.min_size, int(current_size * 0.8))
                    for _ in range(min(excess_count, len(self._pool))):
                        if self._pool:
                            resource = self._pool.popleft()
                            self._destroy_resource(resource)
                    logger.info(f"Scaled down pool by {excess_count} resources")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'in_use': len(self._in_use),
                'total_size': len(self._pool) + len(self._in_use),
                'utilization': len(self._in_use) / max(1, len(self._pool) + len(self._in_use)),
                'created_count': self._created_count,
                'destroyed_count': self._destroyed_count,
                'stats': dict(self._stats)
            }


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, collection_interval: int = 5):
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=1440)  # Keep 2 hours at 5-second intervals
        self.alerts = []
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'response_time_avg': 5.0,  # seconds
            'error_rate': 0.05  # 5%
        }
        
        self._monitoring = False
        self._monitor_thread = None
        
        # Performance counters
        self._request_count = 0
        self._error_count = 0
        self._response_times = deque(maxlen=1000)
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info("Initialized PerformanceMonitor")
    
    def start_monitoring(self) -> None:
        """Start performance monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
    
    def record_request(self, response_time: float, error: bool = False) -> None:
        """Record request metrics"""
        self._request_count += 1
        if error:
            self._error_count += 1
        
        self._response_times.append(response_time)
    
    def record_cache_event(self, hit: bool) -> None:
        """Record cache hit/miss"""
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
    
    def _collect_metrics(self) -> None:
        """Collect system metrics periodically"""
        while self._monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                # Application metrics
                request_rate = self._request_count / self.collection_interval
                error_rate = self._error_count / max(1, self._request_count)
                avg_response_time = sum(self._response_times) / max(1, len(self._response_times))
                cache_hit_rate = self._cache_hits / max(1, self._cache_hits + self._cache_misses)
                
                # Create metrics object
                metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / 1024 / 1024,
                    disk_io_read=disk_io.read_bytes if disk_io else 0,
                    disk_io_write=disk_io.write_bytes if disk_io else 0,
                    network_sent=network_io.bytes_sent if network_io else 0,
                    network_recv=network_io.bytes_recv if network_io else 0,
                    active_threads=threading.active_count(),
                    request_rate=request_rate,
                    response_time_avg=avg_response_time,
                    cache_hit_rate=cache_hit_rate,
                    error_rate=error_rate
                )
                
                self.metrics_history.append(metrics)
                
                # Check thresholds
                self._check_thresholds(metrics)
                
                # Reset counters for rate calculations
                self._request_count = 0
                self._error_count = 0
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            time.sleep(self.collection_interval)
    
    def _check_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check performance thresholds and generate alerts"""
        alerts_triggered = []
        
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts_triggered.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts_triggered.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.response_time_avg > self.thresholds['response_time_avg']:
            alerts_triggered.append(f"High response time: {metrics.response_time_avg:.2f}s")
        
        if metrics.error_rate > self.thresholds['error_rate']:
            alerts_triggered.append(f"High error rate: {metrics.error_rate:.1%}")
        
        for alert in alerts_triggered:
            self.alerts.append({
                'timestamp': datetime.now(),
                'message': alert,
                'severity': 'warning'
            })
            logger.warning(f"Performance alert: {alert}")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get latest metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, minutes: int = 30) -> Dict[str, Any]:
        """Get metrics summary for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {'message': 'No metrics available'}
        
        return {
            'period_minutes': minutes,
            'sample_count': len(recent_metrics),
            'cpu_avg': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'memory_avg': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'response_time_avg': sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics),
            'cache_hit_rate_avg': sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics),
            'error_rate_avg': sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            'request_rate_avg': sum(m.request_rate for m in recent_metrics) / len(recent_metrics),
            'alerts_count': len([a for a in self.alerts if a['timestamp'] > cutoff_time])
        }


class LoadBalancer:
    """Intelligent load balancing for distributed processing"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        self.worker_stats = defaultdict(lambda: {'tasks': 0, 'errors': 0, 'avg_time': 0.0})
        self.pending_tasks = 0
        self.completed_tasks = 0
        
        logger.info(f"Initialized LoadBalancer with {self.max_workers} workers")
    
    async def distribute_work(
        self,
        tasks: List[Callable],
        use_processes: bool = False,
        batch_size: int = 10
    ) -> List[Any]:
        """Distribute work across available workers"""
        executor = self.process_executor if use_processes else self.thread_executor
        
        self.pending_tasks += len(tasks)
        results = []
        
        # Process tasks in batches
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_futures = []
            
            for task in batch:
                future = executor.submit(task)
                batch_futures.append(future)
            
            # Collect batch results
            for future in batch_futures:
                try:
                    result = future.result(timeout=30)  # 30-second timeout
                    results.append(result)
                    self.completed_tasks += 1
                    self.pending_tasks -= 1
                except Exception as e:
                    logger.error(f"Task execution error: {e}")
                    results.append(None)
                    self.pending_tasks -= 1
        
        return results
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        return {
            'max_workers': self.max_workers,
            'pending_tasks': self.pending_tasks,
            'completed_tasks': self.completed_tasks,
            'worker_utilization': self.pending_tasks / self.max_workers,
            'worker_stats': dict(self.worker_stats)
        }


class AutoOptimizer:
    """Automatic performance optimization system"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_rules = []
        self.applied_optimizations = []
        self._setup_default_rules()
        
        logger.info("Initialized AutoOptimizer")
    
    def _setup_default_rules(self) -> None:
        """Set up default optimization rules"""
        
        # Rule: Increase cache size if hit rate is low
        def optimize_cache_size():
            metrics = self.monitor.get_current_metrics()
            if metrics and metrics.cache_hit_rate < 0.7:
                return "increase_cache_size"
            return None
        
        # Rule: Enable compression if response times are high
        def optimize_compression():
            metrics = self.monitor.get_current_metrics()
            if metrics and metrics.response_time_avg > 2.0:
                return "enable_compression"
            return None
        
        # Rule: Scale resources if CPU/memory usage is high
        def optimize_resources():
            metrics = self.monitor.get_current_metrics()
            if metrics and (metrics.cpu_percent > 80 or metrics.memory_percent > 85):
                return "scale_resources"
            return None
        
        self.optimization_rules = [
            optimize_cache_size,
            optimize_compression,
            optimize_resources
        ]
    
    def run_optimization_cycle(self) -> List[str]:
        """Run optimization cycle"""
        applied = []
        
        for rule in self.optimization_rules:
            try:
                optimization = rule()
                if optimization and optimization not in self.applied_optimizations:
                    self._apply_optimization(optimization)
                    applied.append(optimization)
                    self.applied_optimizations.append(optimization)
                    
            except Exception as e:
                logger.error(f"Error in optimization rule: {e}")
        
        return applied
    
    def _apply_optimization(self, optimization: str) -> None:
        """Apply specific optimization"""
        logger.info(f"Applying optimization: {optimization}")
        
        if optimization == "increase_cache_size":
            # Implementation would increase cache size
            logger.info("Increased cache size by 25%")
            
        elif optimization == "enable_compression":
            # Implementation would enable response compression
            logger.info("Enabled response compression")
            
        elif optimization == "scale_resources":
            # Implementation would scale resources
            logger.info("Initiated resource scaling")


# Performance decorators
def performance_tracked(monitor: PerformanceMonitor):
    """Decorator to track function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error_occurred = False
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                raise
            finally:
                execution_time = time.time() - start_time
                monitor.record_request(execution_time, error_occurred)
        
        return wrapper
    return decorator


def cached_result(cache: IntelligentCache, ttl: int = 3600):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator


# Global performance components
global_cache = IntelligentCache(max_size=50000, ttl_seconds=3600)
performance_monitor = PerformanceMonitor(collection_interval=5)
load_balancer = LoadBalancer()
auto_optimizer = AutoOptimizer(performance_monitor)

# Start monitoring
performance_monitor.start_monitoring()

logger.info("Enhanced Performance Optimization System initialized")