#!/usr/bin/env python3
"""
Advanced performance optimization system for defensive cybersecurity operations

This module provides comprehensive performance optimization including caching,
resource pooling, concurrent processing, and adaptive scaling for defensive systems.
"""

import time
import threading
import asyncio
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps, lru_cache
import queue
import weakref

# Setup performance logging
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache replacement strategies"""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    FIFO = "fifo"        # First In First Out
    ADAPTIVE = "adaptive" # Adaptive based on access patterns

class ResourceType(Enum):
    """Types of resources that can be pooled"""
    THREAD = "thread"
    PROCESS = "process"
    CONNECTION = "connection"
    COMPUTATION = "computation"

@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    throughput_ops_per_sec: float
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat()
        }

class IntelligentCache:
    """High-performance intelligent cache with multiple strategies"""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.insertion_order = []
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        
        logger.info(f"Intelligent cache initialized - Size: {max_size}, Strategy: {strategy.value}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with intelligent tracking"""
        
        with self.lock:
            if key in self.cache:
                # Update access statistics
                self.access_times[key] = datetime.now()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                self.hit_count += 1
                
                logger.debug(f"Cache HIT for key: {key}")
                return self.cache[key]
            else:
                self.miss_count += 1
                logger.debug(f"Cache MISS for key: {key}")
                return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Put item in cache with intelligent eviction"""
        
        with self.lock:
            now = datetime.now()
            
            # If cache is full, evict based on strategy
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_item()
            
            # Store item
            self.cache[key] = {
                'value': value,
                'created_at': now,
                'ttl_seconds': ttl_seconds,
                'size_bytes': self._estimate_size(value)
            }
            
            self.access_times[key] = now
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            if key not in self.insertion_order:
                self.insertion_order.append(key)
            
            logger.debug(f"Cache PUT for key: {key}")
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of cached value"""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, dict)):
                return len(json.dumps(value, default=str).encode('utf-8'))
            else:
                return 1024  # Default estimate
        except:
            return 1024
    
    def _evict_item(self):
        """Evict item based on cache strategy"""
        
        if not self.cache:
            return
        
        key_to_evict = None
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently accessed
            key_to_evict = min(self.access_times.keys(), 
                             key=lambda k: self.access_times[k])
            
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently accessed
            key_to_evict = min(self.access_counts.keys(),
                             key=lambda k: self.access_counts[k])
            
        elif self.strategy == CacheStrategy.FIFO:
            # Evict first inserted
            key_to_evict = self.insertion_order[0]
            
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on access patterns
            key_to_evict = self._adaptive_eviction()
        
        if key_to_evict:
            self.remove(key_to_evict)
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction based on access patterns"""
        
        now = datetime.now()
        candidates = []
        
        for key in self.cache.keys():
            last_access = self.access_times.get(key, now)
            access_count = self.access_counts.get(key, 0)
            age = (now - last_access).total_seconds()
            
            # Score based on recency, frequency, and age
            score = (age / 3600) - (access_count / 10)  # Higher score = better eviction candidate
            candidates.append((key, score))
        
        if candidates:
            # Sort by score (descending) and return highest score (best eviction candidate)
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def remove(self, key: str):
        """Remove item from cache"""
        
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_times.pop(key, None)
                self.access_counts.pop(key, None)
                
                if key in self.insertion_order:
                    self.insertion_order.remove(key)
                
                logger.debug(f"Cache EVICT for key: {key}")
    
    def clear(self):
        """Clear entire cache"""
        
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.insertion_order.clear()
            self.hit_count = 0
            self.miss_count = 0
            
            logger.info("Cache cleared")
    
    def get_statistics(self) -> Dict:
        """Get cache performance statistics"""
        
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / max(total_requests, 1)) * 100
            
            total_size_bytes = sum(
                item['size_bytes'] for item in self.cache.values()
            )
            
            return {
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate_percent': round(hit_rate, 2),
                'total_items': len(self.cache),
                'max_size': self.max_size,
                'utilization_percent': round((len(self.cache) / self.max_size) * 100, 2),
                'total_size_bytes': total_size_bytes,
                'average_access_count': round(
                    sum(self.access_counts.values()) / max(len(self.access_counts), 1), 2
                )
            }

class ResourcePool:
    """High-performance resource pool for expensive operations"""
    
    def __init__(self, resource_type: ResourceType, min_size: int = 2, 
                 max_size: int = 10, resource_factory: Callable = None):
        self.resource_type = resource_type
        self.min_size = min_size
        self.max_size = max_size
        self.resource_factory = resource_factory
        
        self.available_resources = queue.Queue()
        self.in_use_resources = set()
        self.total_created = 0
        self.lock = threading.Lock()
        
        # Initialize minimum resources
        self._initialize_pool()
        
        logger.info(f"Resource pool initialized - Type: {resource_type.value}, "
                   f"Min: {min_size}, Max: {max_size}")
    
    def _initialize_pool(self):
        """Initialize pool with minimum resources"""
        
        for _ in range(self.min_size):
            resource = self._create_resource()
            if resource:
                self.available_resources.put(resource)
    
    def _create_resource(self):
        """Create a new resource"""
        
        try:
            if self.resource_factory:
                resource = self.resource_factory()
            else:
                # Default resource creation based on type
                if self.resource_type == ResourceType.THREAD:
                    resource = ThreadPoolExecutor(max_workers=1)
                elif self.resource_type == ResourceType.PROCESS:
                    resource = ProcessPoolExecutor(max_workers=1)
                else:
                    resource = f"resource_{self.total_created}"
            
            self.total_created += 1
            logger.debug(f"Created new resource: {self.resource_type.value}_{self.total_created}")
            return resource
            
        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
            return None
    
    def acquire(self, timeout: float = 5.0):
        """Acquire a resource from the pool"""
        
        try:
            # Try to get available resource
            resource = self.available_resources.get(timeout=timeout)
            
            with self.lock:
                self.in_use_resources.add(id(resource))
            
            logger.debug(f"Acquired resource: {self.resource_type.value}")
            return resource
            
        except queue.Empty:
            # No available resources, try to create new one if under max
            with self.lock:
                if self.total_created < self.max_size:
                    resource = self._create_resource()
                    if resource:
                        self.in_use_resources.add(id(resource))
                        return resource
            
            logger.warning(f"Resource pool exhausted for {self.resource_type.value}")
            raise ResourceExhaustedException(f"No available {self.resource_type.value} resources")
    
    def release(self, resource):
        """Release a resource back to the pool"""
        
        resource_id = id(resource)
        
        with self.lock:
            if resource_id in self.in_use_resources:
                self.in_use_resources.remove(resource_id)
                
                # Return to available pool if under min size
                if self.available_resources.qsize() < self.min_size:
                    self.available_resources.put(resource)
                    logger.debug(f"Returned resource to pool: {self.resource_type.value}")
                else:
                    # Cleanup resource if pool is full
                    self._cleanup_resource(resource)
                    logger.debug(f"Cleaned up excess resource: {self.resource_type.value}")
    
    def _cleanup_resource(self, resource):
        """Cleanup a resource when no longer needed"""
        
        try:
            if hasattr(resource, 'shutdown'):
                resource.shutdown(wait=False)
            elif hasattr(resource, 'close'):
                resource.close()
        except Exception as e:
            logger.warning(f"Error cleaning up resource: {e}")
    
    def get_statistics(self) -> Dict:
        """Get resource pool statistics"""
        
        with self.lock:
            return {
                'resource_type': self.resource_type.value,
                'total_created': self.total_created,
                'available': self.available_resources.qsize(),
                'in_use': len(self.in_use_resources),
                'min_size': self.min_size,
                'max_size': self.max_size,
                'utilization_percent': round(
                    (len(self.in_use_resources) / self.max_size) * 100, 2
                )
            }

class ResourceExhaustedException(Exception):
    """Exception raised when resource pool is exhausted"""
    pass

class PerformanceOptimizer:
    """Comprehensive performance optimization system"""
    
    def __init__(self):
        self.caches = {}
        self.resource_pools = {}
        self.performance_metrics = []
        self.optimization_enabled = True
        
        # Performance configuration
        self.config = {
            'default_cache_size': 1000,
            'default_cache_strategy': CacheStrategy.ADAPTIVE,
            'thread_pool_size': 10,
            'process_pool_size': 4,
            'metrics_retention_hours': 24
        }
        
        logger.info("Performance optimizer initialized")
    
    def get_cache(self, cache_name: str, max_size: int = None, 
                  strategy: CacheStrategy = None) -> IntelligentCache:
        """Get or create a named cache"""
        
        if cache_name not in self.caches:
            cache_size = max_size or self.config['default_cache_size']
            cache_strategy = strategy or self.config['default_cache_strategy']
            
            self.caches[cache_name] = IntelligentCache(cache_size, cache_strategy)
            logger.info(f"Created cache: {cache_name}")
        
        return self.caches[cache_name]
    
    def get_resource_pool(self, pool_name: str, resource_type: ResourceType,
                         min_size: int = 2, max_size: int = 10,
                         resource_factory: Callable = None) -> ResourcePool:
        """Get or create a named resource pool"""
        
        if pool_name not in self.resource_pools:
            self.resource_pools[pool_name] = ResourcePool(
                resource_type, min_size, max_size, resource_factory
            )
            logger.info(f"Created resource pool: {pool_name}")
        
        return self.resource_pools[pool_name]
    
    def cached(self, cache_name: str = "default", ttl_seconds: Optional[int] = None):
        """Decorator for caching function results"""
        
        def decorator(func: Callable) -> Callable:
            cache = self.get_cache(cache_name)
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                key_data = {
                    'function': func.__name__,
                    'args': args,
                    'kwargs': sorted(kwargs.items())
                }
                cache_key = hashlib.md5(
                    json.dumps(key_data, default=str, sort_keys=True).encode()
                ).hexdigest()
                
                # Try cache first
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    return cached_result['value']
                
                # Execute function and cache result
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                cache.put(cache_key, result, ttl_seconds)
                
                logger.debug(f"Cached result for {func.__name__} (execution: {execution_time:.3f}s)")
                return result
            
            return wrapper
        return decorator
    
    def parallel_execution(self, pool_name: str = "default_threads"):
        """Decorator for parallel execution of functions"""
        
        def decorator(func: Callable) -> Callable:
            pool = self.get_resource_pool(
                pool_name, ResourceType.THREAD, 
                max_size=self.config['thread_pool_size']
            )
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                executor = pool.acquire()
                try:
                    future = executor.submit(func, *args, **kwargs)
                    result = future.result(timeout=30)  # 30 second timeout
                    return result
                finally:
                    pool.release(executor)
            
            return wrapper
        return decorator
    
    def batch_processor(self, batch_size: int = 100, parallel: bool = True):
        """Decorator for optimized batch processing"""
        
        def decorator(func: Callable) -> Callable:
            
            @wraps(func)
            def wrapper(items: List[Any], *args, **kwargs):
                if not items:
                    return []
                
                results = []
                batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
                
                if parallel and len(batches) > 1:
                    # Parallel batch processing
                    pool = self.get_resource_pool(
                        "batch_processor", ResourceType.THREAD,
                        max_size=min(len(batches), self.config['thread_pool_size'])
                    )
                    
                    def process_batch(batch):
                        executor = pool.acquire()
                        try:
                            return [func(item, *args, **kwargs) for item in batch]
                        finally:
                            pool.release(executor)
                    
                    with ThreadPoolExecutor(max_workers=min(len(batches), 5)) as executor:
                        batch_futures = [executor.submit(process_batch, batch) for batch in batches]
                        
                        for future in as_completed(batch_futures):
                            results.extend(future.result())
                else:
                    # Sequential batch processing
                    for batch in batches:
                        batch_results = [func(item, *args, **kwargs) for item in batch]
                        results.extend(batch_results)
                
                return results
            
            return wrapper
        return decorator
    
    def performance_monitor(self, operation_name: str = None):
        """Decorator for monitoring function performance"""
        
        def decorator(func: Callable) -> Callable:
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                start_time = datetime.now()
                start_timestamp = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = e
                    success = False
                
                end_time = datetime.now()
                duration = time.time() - start_timestamp
                
                # Collect performance metrics
                try:
                    import os
                    import resource
                    
                    # Get memory usage (approximate)
                    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
                    
                    metrics = PerformanceMetrics(
                        operation_name=op_name,
                        start_time=start_time,
                        end_time=end_time,
                        duration_seconds=duration,
                        memory_usage_mb=memory_usage,
                        cpu_usage_percent=0.0,  # Would need psutil for accurate CPU usage
                        cache_hit_rate=0.0,     # Calculate if using cache
                        throughput_ops_per_sec=1.0 / duration if duration > 0 else 0.0
                    )
                    
                    self._record_metrics(metrics)
                    
                except Exception as e:
                    logger.warning(f"Failed to collect performance metrics: {e}")
                
                if success:
                    return result
                else:
                    raise result
            
            return wrapper
        return decorator
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        
        self.performance_metrics.append(metrics)
        
        # Limit metrics retention
        retention_cutoff = datetime.now() - timedelta(
            hours=self.config['metrics_retention_hours']
        )
        
        self.performance_metrics = [
            m for m in self.performance_metrics 
            if m.start_time > retention_cutoff
        ]
        
        logger.debug(f"Recorded performance metrics for {metrics.operation_name}: "
                    f"{metrics.duration_seconds:.3f}s")
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        
        if not self.performance_metrics:
            return {"message": "No performance data available"}
        
        # Aggregate metrics by operation
        operation_stats = {}
        
        for metric in self.performance_metrics:
            op_name = metric.operation_name
            
            if op_name not in operation_stats:
                operation_stats[op_name] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'min_duration': float('inf'),
                    'max_duration': 0.0,
                    'total_memory': 0.0,
                    'total_throughput': 0.0
                }
            
            stats = operation_stats[op_name]
            stats['count'] += 1
            stats['total_duration'] += metric.duration_seconds
            stats['min_duration'] = min(stats['min_duration'], metric.duration_seconds)
            stats['max_duration'] = max(stats['max_duration'], metric.duration_seconds)
            stats['total_memory'] += metric.memory_usage_mb
            stats['total_throughput'] += metric.throughput_ops_per_sec
        
        # Calculate averages
        for op_name, stats in operation_stats.items():
            count = stats['count']
            stats['avg_duration'] = stats['total_duration'] / count
            stats['avg_memory'] = stats['total_memory'] / count
            stats['avg_throughput'] = stats['total_throughput'] / count
        
        # Cache statistics
        cache_stats = {}
        for cache_name, cache in self.caches.items():
            cache_stats[cache_name] = cache.get_statistics()
        
        # Resource pool statistics
        pool_stats = {}
        for pool_name, pool in self.resource_pools.items():
            pool_stats[pool_name] = pool.get_statistics()
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'total_operations': len(self.performance_metrics),
            'operation_statistics': operation_stats,
            'cache_statistics': cache_stats,
            'resource_pool_statistics': pool_stats,
            'optimization_recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        
        recommendations = []
        
        # Cache recommendations
        for cache_name, cache in self.caches.items():
            stats = cache.get_statistics()
            if stats['hit_rate_percent'] < 70:
                recommendations.append(
                    f"Consider increasing cache size for '{cache_name}' "
                    f"(current hit rate: {stats['hit_rate_percent']}%)"
                )
        
        # Resource pool recommendations
        for pool_name, pool in self.resource_pools.items():
            stats = pool.get_statistics()
            if stats['utilization_percent'] > 80:
                recommendations.append(
                    f"Consider increasing pool size for '{pool_name}' "
                    f"(current utilization: {stats['utilization_percent']}%)"
                )
        
        # Performance recommendations
        if self.performance_metrics:
            avg_duration = sum(m.duration_seconds for m in self.performance_metrics) / len(self.performance_metrics)
            if avg_duration > 1.0:
                recommendations.append(
                    f"Average operation duration is high ({avg_duration:.2f}s). "
                    "Consider using caching or parallel processing."
                )
        
        return recommendations if recommendations else ["Performance is optimal"]

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Convenience decorators using global optimizer
cached = performance_optimizer.cached
parallel_execution = performance_optimizer.parallel_execution  
batch_processor = performance_optimizer.batch_processor
performance_monitor = performance_optimizer.performance_monitor

def main():
    """Demonstrate performance optimization capabilities"""
    
    print("🛡️  Advanced Performance Optimization System")
    print("=" * 55)
    
    # Demonstrate caching
    @cached("demo_cache")
    @performance_monitor("expensive_operation")
    def expensive_defensive_calculation(n: int) -> int:
        """Simulate expensive defensive computation"""
        time.sleep(0.1)  # Simulate work
        return sum(i * i for i in range(n))
    
    print("\n⚡ TESTING CACHING PERFORMANCE")
    print("-" * 35)
    
    # First call (cache miss)
    start_time = time.time()
    result1 = expensive_defensive_calculation(1000)
    first_call_time = time.time() - start_time
    print(f"First call (cache miss): {first_call_time:.3f}s - Result: {result1}")
    
    # Second call (cache hit)
    start_time = time.time()
    result2 = expensive_defensive_calculation(1000)
    second_call_time = time.time() - start_time
    print(f"Second call (cache hit): {second_call_time:.3f}s - Result: {result2}")
    
    speedup = first_call_time / max(second_call_time, 0.001)
    print(f"Cache speedup: {speedup:.1f}x")
    
    # Demonstrate batch processing
    @batch_processor(batch_size=50, parallel=True)
    @performance_monitor("batch_processing")
    def process_defensive_item(item: int) -> int:
        """Process individual defensive item"""
        time.sleep(0.01)  # Simulate processing
        return item * 2
    
    print(f"\n🔄 TESTING BATCH PROCESSING")
    print("-" * 30)
    
    items = list(range(200))
    
    start_time = time.time()
    batch_results = process_defensive_item(items)
    batch_time = time.time() - start_time
    
    print(f"Batch processed {len(items)} items in {batch_time:.3f}s")
    print(f"Throughput: {len(items)/batch_time:.1f} items/sec")
    
    # Demonstrate resource pooling
    resource_pool = performance_optimizer.get_resource_pool(
        "demo_pool", ResourceType.THREAD, min_size=3, max_size=8
    )
    
    print(f"\n🏊 TESTING RESOURCE POOLING")
    print("-" * 30)
    
    def pool_demonstration():
        results = []
        for i in range(15):  # More requests than pool size
            try:
                resource = resource_pool.acquire(timeout=1.0)
                results.append(f"Task {i}: Acquired resource")
                resource_pool.release(resource)
            except ResourceExhaustedException:
                results.append(f"Task {i}: Resource pool exhausted")
        return results
    
    pool_results = pool_demonstration()
    successful_acquisitions = sum(1 for r in pool_results if "Acquired" in r)
    print(f"Successful resource acquisitions: {successful_acquisitions}/15")
    
    # Generate performance report
    print(f"\n📊 PERFORMANCE REPORT")
    print("-" * 25)
    
    report = performance_optimizer.get_performance_report()
    
    print(f"Total Operations Monitored: {report['total_operations']}")
    
    if 'operation_statistics' in report:
        print(f"\nOperation Statistics:")
        for op_name, stats in report['operation_statistics'].items():
            print(f"  • {op_name}:")
            print(f"    - Count: {stats['count']}")
            print(f"    - Avg Duration: {stats['avg_duration']:.3f}s")
            print(f"    - Avg Throughput: {stats['avg_throughput']:.1f} ops/sec")
    
    if 'cache_statistics' in report:
        print(f"\nCache Statistics:")
        for cache_name, stats in report['cache_statistics'].items():
            print(f"  • {cache_name}: {stats['hit_rate_percent']}% hit rate, "
                  f"{stats['utilization_percent']}% utilization")
    
    if 'resource_pool_statistics' in report:
        print(f"\nResource Pool Statistics:")
        for pool_name, stats in report['resource_pool_statistics'].items():
            print(f"  • {pool_name}: {stats['utilization_percent']}% utilization, "
                  f"{stats['available']} available, {stats['in_use']} in use")
    
    print(f"\nOptimization Recommendations:")
    for rec in report['optimization_recommendations']:
        print(f"  • {rec}")
    
    # Export performance data
    export_file = f"logs/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("logs").mkdir(exist_ok=True)
    
    with open(export_file, 'w') as f:
        # Convert metrics to dict for JSON serialization
        report_copy = report.copy()
        if 'performance_metrics' not in report_copy:
            report_copy['performance_metrics'] = [m.to_dict() for m in performance_optimizer.performance_metrics]
        
        json.dump(report_copy, f, indent=2)
    
    print(f"\n💾 Performance report exported to: {export_file}")
    print("✅ Performance optimization demonstration completed successfully!")

if __name__ == "__main__":
    main()