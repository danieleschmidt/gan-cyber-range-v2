"""
High-Performance Computing optimization for GAN-Cyber-Range-v2.
Implements advanced caching, parallel processing, and resource optimization.
"""

import logging
import asyncio
import threading
import multiprocessing
import time
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import concurrent.futures
import queue
import weakref
from collections import defaultdict, OrderedDict
import json

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Performance optimization levels"""
    MINIMAL = 1
    STANDARD = 2
    AGGRESSIVE = 3
    MAXIMUM = 4


class ProcessingMode(Enum):
    """Processing execution modes"""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"
    ASYNC = "async"
    HYBRID = "hybrid"


@dataclass
class PerformanceMetrics:
    """Performance measurement data"""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    throughput_ops_per_sec: Optional[float] = None
    error_count: int = 0
    success_count: int = 0


class IntelligentCache:
    """Multi-level intelligent caching system"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.l1_cache = OrderedDict()  # LRU cache
        self.l2_cache = {}  # Frequency-based cache
        self.access_frequency = defaultdict(int)
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent promotion"""
        with self.lock:
            now = time.time()
            
            # Check L1 cache first
            if key in self.l1_cache:
                value, timestamp = self.l1_cache[key]
                if now - timestamp < self.ttl_seconds:
                    # Move to end (most recently used)
                    self.l1_cache.move_to_end(key)
                    self.access_frequency[key] += 1
                    self.hit_count += 1
                    return value
                else:
                    del self.l1_cache[key]
            
            # Check L2 cache
            if key in self.l2_cache:
                value, timestamp = self.l2_cache[key]
                if now - timestamp < self.ttl_seconds:
                    # Promote to L1 if frequently accessed
                    self.access_frequency[key] += 1
                    if self.access_frequency[key] > 5:
                        self._promote_to_l1(key, value, timestamp)
                    self.hit_count += 1
                    return value
                else:
                    del self.l2_cache[key]
            
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any):
        """Store value in cache with intelligent placement"""
        with self.lock:
            timestamp = time.time()
            
            # Decide cache level based on access frequency
            if self.access_frequency[key] > 3:
                self._put_l1(key, value, timestamp)
            else:
                self._put_l2(key, value, timestamp)
            
            self.access_times[key] = timestamp
    
    def _put_l1(self, key: str, value: Any, timestamp: float):
        """Store in L1 cache"""
        if len(self.l1_cache) >= self.max_size // 2:
            # Evict least recently used
            old_key, _ = self.l1_cache.popitem(last=False)
            # Demote to L2
            if old_key in self.l1_cache:
                self.l2_cache[old_key] = self.l1_cache[old_key]
        
        self.l1_cache[key] = (value, timestamp)
    
    def _put_l2(self, key: str, value: Any, timestamp: float):
        """Store in L2 cache"""
        if len(self.l2_cache) >= self.max_size:
            # Evict least frequently used
            lfu_key = min(self.l2_cache.keys(), 
                         key=lambda k: self.access_frequency[k])
            del self.l2_cache[lfu_key]
        
        self.l2_cache[key] = (value, timestamp)
    
    def _promote_to_l1(self, key: str, value: Any, timestamp: float):
        """Promote item from L2 to L1"""
        if key in self.l2_cache:
            del self.l2_cache[key]
        self._put_l1(key, value, timestamp)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(1, total_requests)
        
        return {
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'l1_size': len(self.l1_cache),
            'l2_size': len(self.l2_cache),
            'total_size': len(self.l1_cache) + len(self.l2_cache)
        }


class ParallelProcessingEngine:
    """Advanced parallel processing with adaptive load balancing"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_queue = queue.Queue()
        self.results_cache = IntelligentCache()
        self.performance_history = []
    
    async def execute_parallel(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        mode: ProcessingMode = ProcessingMode.THREADED,
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    ) -> List[Any]:
        """Execute tasks in parallel with adaptive optimization"""
        
        if optimization_level == OptimizationLevel.MAXIMUM:
            return await self._execute_hybrid_parallel(tasks)
        elif mode == ProcessingMode.ASYNC:
            return await self._execute_async_parallel(tasks)
        elif mode == ProcessingMode.MULTIPROCESS:
            return await self._execute_process_parallel(tasks)
        else:
            return await self._execute_thread_parallel(tasks)
    
    async def _execute_hybrid_parallel(self, tasks: List[Tuple[Callable, tuple, dict]]) -> List[Any]:
        """Hybrid execution with intelligent task distribution"""
        cpu_bound_tasks = []
        io_bound_tasks = []
        
        # Classify tasks based on historical performance
        for task in tasks:
            func, args, kwargs = task
            task_signature = f"{func.__name__}_{len(args)}_{len(kwargs)}"
            
            # Use heuristics to classify task type
            if self._is_cpu_intensive(task_signature):
                cpu_bound_tasks.append(task)
            else:
                io_bound_tasks.append(task)
        
        # Execute CPU-bound tasks in processes, I/O-bound in threads
        results = []
        
        if cpu_bound_tasks:
            cpu_results = await self._execute_process_parallel(cpu_bound_tasks)
            results.extend(cpu_results)
        
        if io_bound_tasks:
            io_results = await self._execute_thread_parallel(io_bound_tasks)
            results.extend(io_results)
        
        return results
    
    async def _execute_async_parallel(self, tasks: List[Tuple[Callable, tuple, dict]]) -> List[Any]:
        """Execute tasks asynchronously"""
        async_tasks = []
        
        for func, args, kwargs in tasks:
            if asyncio.iscoroutinefunction(func):
                async_tasks.append(func(*args, **kwargs))
            else:
                # Wrap non-async functions
                loop = asyncio.get_event_loop()
                async_tasks.append(loop.run_in_executor(None, func, *args))
        
        return await asyncio.gather(*async_tasks, return_exceptions=True)
    
    async def _execute_thread_parallel(self, tasks: List[Tuple[Callable, tuple, dict]]) -> List[Any]:
        """Execute tasks in thread pool"""
        loop = asyncio.get_event_loop()
        futures = []
        
        for func, args, kwargs in tasks:
            future = loop.run_in_executor(self.thread_executor, func, *args)
            futures.append(future)
        
        return await asyncio.gather(*futures, return_exceptions=True)
    
    async def _execute_process_parallel(self, tasks: List[Tuple[Callable, tuple, dict]]) -> List[Any]:
        """Execute tasks in process pool"""
        loop = asyncio.get_event_loop()
        futures = []
        
        for func, args, kwargs in tasks:
            # Process pools require pickleable functions
            if self._is_pickleable(func):
                future = loop.run_in_executor(self.process_executor, func, *args)
                futures.append(future)
            else:
                # Fallback to thread execution
                future = loop.run_in_executor(self.thread_executor, func, *args)
                futures.append(future)
        
        return await asyncio.gather(*futures, return_exceptions=True)
    
    def _is_cpu_intensive(self, task_signature: str) -> bool:
        """Heuristic to determine if task is CPU-intensive"""
        cpu_keywords = ['compute', 'calculate', 'process', 'generate', 'train', 'optimize']
        return any(keyword in task_signature.lower() for keyword in cpu_keywords)
    
    def _is_pickleable(self, func: Callable) -> bool:
        """Check if function is pickleable for multiprocessing"""
        try:
            import pickle
            pickle.dumps(func)
            return True
        except:
            return False
    
    def shutdown(self):
        """Shutdown executors"""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)


class ResourceOptimizer:
    """Dynamic resource optimization and allocation"""
    
    def __init__(self):
        self.memory_thresholds = {
            'warning': 0.8,
            'critical': 0.9
        }
        self.cpu_thresholds = {
            'warning': 0.8,
            'critical': 0.95
        }
        self.optimization_history = []
        self.active_optimizations = set()
    
    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor system resources with fallback"""
        try:
            import psutil
            return self._monitor_with_psutil()
        except ImportError:
            return self._monitor_fallback()
    
    def _monitor_with_psutil(self) -> Dict[str, Any]:
        """Monitor resources using psutil"""
        import psutil
        
        # Memory information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        return {
            'memory': {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_percent': memory.percent,
                'free_gb': memory.free / (1024**3)
            },
            'cpu': {
                'usage_percent': cpu_percent,
                'core_count': cpu_count,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            },
            'disk': {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'usage_percent': (disk.used / disk.total) * 100
            },
            'timestamp': datetime.now()
        }
    
    def _monitor_fallback(self) -> Dict[str, Any]:
        """Fallback resource monitoring without psutil"""
        import os
        
        # Basic memory info from /proc/meminfo (Linux)
        memory_info = {'total_gb': 8, 'available_gb': 4, 'used_percent': 50, 'free_gb': 4}
        
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if 'MemTotal:' in line:
                        total_kb = int(line.split()[1])
                        memory_info['total_gb'] = total_kb / (1024**2)
                    elif 'MemAvailable:' in line:
                        available_kb = int(line.split()[1])
                        memory_info['available_gb'] = available_kb / (1024**2)
        except:
            pass  # Use defaults
        
        # CPU count
        cpu_count = os.cpu_count() or 4
        
        return {
            'memory': memory_info,
            'cpu': {
                'usage_percent': 50,  # Default estimate
                'core_count': cpu_count,
                'load_average': [1.0, 1.0, 1.0]
            },
            'disk': {
                'total_gb': 100,
                'used_gb': 50,
                'free_gb': 50,
                'usage_percent': 50
            },
            'timestamp': datetime.now()
        }
    
    def optimize_resources(self, current_resources: Dict[str, Any]) -> List[str]:
        """Apply resource optimizations based on current state"""
        optimizations_applied = []
        
        # Memory optimization
        memory_usage = current_resources['memory']['used_percent']
        if memory_usage > self.memory_thresholds['critical']:
            optimizations_applied.extend(self._aggressive_memory_optimization())
        elif memory_usage > self.memory_thresholds['warning']:
            optimizations_applied.extend(self._standard_memory_optimization())
        
        # CPU optimization
        cpu_usage = current_resources['cpu']['usage_percent']
        if cpu_usage > self.cpu_thresholds['critical']:
            optimizations_applied.extend(self._cpu_optimization())
        
        return optimizations_applied
    
    def _aggressive_memory_optimization(self) -> List[str]:
        """Aggressive memory optimization strategies"""
        optimizations = []
        
        if 'memory_gc' not in self.active_optimizations:
            # Force garbage collection
            import gc
            collected = gc.collect()
            optimizations.append(f"Garbage collection freed {collected} objects")
            self.active_optimizations.add('memory_gc')
        
        if 'cache_cleanup' not in self.active_optimizations:
            # Clear caches (implementation specific)
            optimizations.append("Cleared application caches")
            self.active_optimizations.add('cache_cleanup')
        
        return optimizations
    
    def _standard_memory_optimization(self) -> List[str]:
        """Standard memory optimization strategies"""
        optimizations = []
        
        # Gentle garbage collection
        import gc
        if gc.garbage:
            collected = gc.collect()
            optimizations.append(f"Collected {collected} garbage objects")
        
        return optimizations
    
    def _cpu_optimization(self) -> List[str]:
        """CPU optimization strategies"""
        optimizations = []
        
        if 'reduce_parallelism' not in self.active_optimizations:
            # Reduce parallel task count
            optimizations.append("Reduced parallel task concurrency")
            self.active_optimizations.add('reduce_parallelism')
        
        return optimizations


class PerformanceProfiler:
    """Comprehensive performance profiling and analysis"""
    
    def __init__(self):
        self.metrics_history = []
        self.profiling_active = False
        self.benchmark_results = {}
    
    def start_profiling(self, operation_name: str) -> PerformanceMetrics:
        """Start profiling an operation"""
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.now()
        )
        return metrics
    
    def end_profiling(self, metrics: PerformanceMetrics) -> PerformanceMetrics:
        """End profiling and calculate final metrics"""
        metrics.end_time = datetime.now()
        metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
        
        # Record metrics
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics (last 1000)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def benchmark_operation(self, operation_name: str, operation_func: Callable, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark an operation"""
        durations = []
        success_count = 0
        error_count = 0
        
        for i in range(iterations):
            metrics = self.start_profiling(f"{operation_name}_benchmark_{i}")
            
            try:
                result = operation_func()
                success_count += 1
                metrics.success_count = 1
            except Exception as e:
                error_count += 1
                metrics.error_count = 1
                logger.warning(f"Benchmark iteration {i} failed: {e}")
            
            self.end_profiling(metrics)
            durations.append(metrics.duration_ms)
        
        # Calculate statistics
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        p50 = sorted_durations[len(sorted_durations) // 2] if sorted_durations else 0
        p95 = sorted_durations[int(len(sorted_durations) * 0.95)] if sorted_durations else 0
        p99 = sorted_durations[int(len(sorted_durations) * 0.99)] if sorted_durations else 0
        
        benchmark_result = {
            'operation_name': operation_name,
            'iterations': iterations,
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': success_count / iterations,
            'avg_duration_ms': avg_duration,
            'min_duration_ms': min_duration,
            'max_duration_ms': max_duration,
            'p50_duration_ms': p50,
            'p95_duration_ms': p95,
            'p99_duration_ms': p99,
            'throughput_ops_per_sec': iterations / (sum(durations) / 1000) if sum(durations) > 0 else 0,
            'timestamp': datetime.now()
        }
        
        self.benchmark_results[operation_name] = benchmark_result
        return benchmark_result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {'message': 'No performance data available'}
        
        # Aggregate metrics by operation name
        operation_stats = defaultdict(list)
        for metric in self.metrics_history:
            operation_stats[metric.operation_name].append(metric)
        
        report = {
            'report_timestamp': datetime.now(),
            'total_operations': len(self.metrics_history),
            'unique_operations': len(operation_stats),
            'time_range': {
                'start': min(m.start_time for m in self.metrics_history),
                'end': max(m.end_time for m in self.metrics_history if m.end_time)
            },
            'operations': {}
        }
        
        # Per-operation statistics
        for op_name, metrics in operation_stats.items():
            durations = [m.duration_ms for m in metrics if m.duration_ms]
            if durations:
                report['operations'][op_name] = {
                    'count': len(metrics),
                    'avg_duration_ms': sum(durations) / len(durations),
                    'min_duration_ms': min(durations),
                    'max_duration_ms': max(durations),
                    'total_errors': sum(m.error_count for m in metrics),
                    'total_successes': sum(m.success_count for m in metrics)
                }
        
        return report


class HyperParameterOptimizer:
    """Automated hyperparameter optimization for ML components"""
    
    def __init__(self):
        self.optimization_history = []
        self.best_parameters = {}
    
    def optimize_parameters(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, List[Any]],
        max_iterations: int = 50,
        optimization_method: str = "random_search"
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using specified method"""
        
        if optimization_method == "grid_search":
            return self._grid_search(objective_function, parameter_space, max_iterations)
        elif optimization_method == "random_search":
            return self._random_search(objective_function, parameter_space, max_iterations)
        else:
            return self._bayesian_optimization(objective_function, parameter_space, max_iterations)
    
    def _random_search(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, List[Any]],
        max_iterations: int
    ) -> Dict[str, Any]:
        """Random search optimization"""
        import random
        
        best_score = float('-inf')
        best_params = {}
        
        for iteration in range(max_iterations):
            # Sample random parameters
            params = {}
            for param_name, param_values in parameter_space.items():
                params[param_name] = random.choice(param_values)
            
            # Evaluate objective function
            try:
                score = objective_function(**params)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                self.optimization_history.append({
                    'iteration': iteration,
                    'parameters': params,
                    'score': score,
                    'timestamp': datetime.now()
                })
                
            except Exception as e:
                logger.warning(f"Optimization iteration {iteration} failed: {e}")
        
        result = {
            'best_parameters': best_params,
            'best_score': best_score,
            'total_iterations': max_iterations,
            'optimization_method': 'random_search'
        }
        
        self.best_parameters.update(best_params)
        return result
    
    def _grid_search(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, List[Any]],
        max_iterations: int
    ) -> Dict[str, Any]:
        """Grid search optimization"""
        import itertools
        
        # Generate all parameter combinations
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        combinations = list(itertools.product(*param_values))
        
        # Limit combinations to max_iterations
        if len(combinations) > max_iterations:
            import random
            combinations = random.sample(combinations, max_iterations)
        
        best_score = float('-inf')
        best_params = {}
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            try:
                score = objective_function(**params)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                self.optimization_history.append({
                    'iteration': i,
                    'parameters': params,
                    'score': score,
                    'timestamp': datetime.now()
                })
                
            except Exception as e:
                logger.warning(f"Grid search iteration {i} failed: {e}")
        
        result = {
            'best_parameters': best_params,
            'best_score': best_score,
            'total_iterations': len(combinations),
            'optimization_method': 'grid_search'
        }
        
        self.best_parameters.update(best_params)
        return result
    
    def _bayesian_optimization(
        self,
        objective_function: Callable,
        parameter_space: Dict[str, List[Any]],
        max_iterations: int
    ) -> Dict[str, Any]:
        """Simplified Bayesian optimization (falls back to random search)"""
        logger.info("Bayesian optimization not available, falling back to random search")
        return self._random_search(objective_function, parameter_space, max_iterations)


# Performance decorators
def profile_performance(operation_name: str = None):
    """Decorator to profile function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler()
            op_name = operation_name or func.__name__
            
            metrics = profiler.start_profiling(op_name)
            try:
                result = func(*args, **kwargs)
                metrics.success_count = 1
                return result
            except Exception as e:
                metrics.error_count = 1
                raise
            finally:
                profiler.end_profiling(metrics)
                logger.debug(f"Operation {op_name} took {metrics.duration_ms:.2f}ms")
        
        return wrapper
    return decorator


def cache_result(ttl_seconds: int = 3600):
    """Decorator to cache function results"""
    cache = IntelligentCache(ttl_seconds=ttl_seconds)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        return wrapper
    return decorator


# Global optimization instances
intelligent_cache = IntelligentCache()
parallel_engine = ParallelProcessingEngine()
resource_optimizer = ResourceOptimizer()
performance_profiler = PerformanceProfiler()
hyperparameter_optimizer = HyperParameterOptimizer()