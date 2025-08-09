"""
Performance Optimization Module for GAN-Cyber-Range-v2
Advanced performance monitoring, optimization, and scaling utilities
"""

import asyncio
import time
import psutil
import threading
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache
import weakref
import gc
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import torch

from .logging_config import get_logger
from .monitoring import MetricsCollector
from .caching import CacheManager

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_percent: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    process_count: int = 0
    thread_count: int = 0
    response_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    throughput: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class PerformanceProfiler:
    """Advanced performance profiler for function and method calls"""
    
    def __init__(self):
        self.profiles = defaultdict(list)
        self.active_profiles = {}
        self.lock = threading.Lock()
    
    def profile(self, name: Optional[str] = None):
        """Decorator for profiling function execution"""
        def decorator(func: Callable) -> Callable:
            profile_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    self._record_profile(profile_name, {
                        'execution_time': execution_time,
                        'memory_delta': memory_delta,
                        'success': success,
                        'error': error,
                        'timestamp': time.time()
                    })
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    self._record_profile(profile_name, {
                        'execution_time': execution_time,
                        'memory_delta': memory_delta,
                        'success': success,
                        'error': error,
                        'timestamp': time.time()
                    })
                
                return result
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def _record_profile(self, name: str, data: Dict[str, Any]):
        """Record profiling data"""
        with self.lock:
            self.profiles[name].append(data)
            
            # Keep only last 1000 entries per function
            if len(self.profiles[name]) > 1000:
                self.profiles[name] = self.profiles[name][-1000:]
    
    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get profiling statistics"""
        with self.lock:
            if name:
                if name not in self.profiles:
                    return {}
                
                data = self.profiles[name]
                return self._calculate_stats(data)
            else:
                stats = {}
                for profile_name, data in self.profiles.items():
                    stats[profile_name] = self._calculate_stats(data)
                return stats
    
    def _calculate_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from profiling data"""
        if not data:
            return {}
        
        execution_times = [d['execution_time'] for d in data]
        memory_deltas = [d['memory_delta'] for d in data]
        success_count = sum(1 for d in data if d['success'])
        
        return {
            'call_count': len(data),
            'success_rate': success_count / len(data),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'avg_memory_delta': sum(memory_deltas) / len(memory_deltas),
            'total_memory_delta': sum(memory_deltas),
            'last_call': max(d['timestamp'] for d in data)
        }
    
    def reset_stats(self, name: Optional[str] = None):
        """Reset profiling statistics"""
        with self.lock:
            if name:
                if name in self.profiles:
                    self.profiles[name].clear()
            else:
                self.profiles.clear()


class ResourcePool:
    """Generic resource pool for managing expensive resources"""
    
    def __init__(self, factory: Callable, max_size: int = 10, min_size: int = 2):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.pool = deque()
        self.in_use = set()
        self.lock = asyncio.Lock()
        self.created_count = 0
        self.metrics = {
            'total_created': 0,
            'currently_in_use': 0,
            'pool_size': 0,
            'wait_times': deque(maxlen=1000)
        }
    
    async def get_resource(self) -> Any:
        """Get resource from pool"""
        start_time = time.perf_counter()
        
        async with self.lock:
            # Try to get from pool
            if self.pool:
                resource = self.pool.popleft()
                self.in_use.add(resource)
                self.metrics['currently_in_use'] = len(self.in_use)
                self.metrics['pool_size'] = len(self.pool)
                return resource
            
            # Create new resource if under limit
            if self.created_count < self.max_size:
                resource = await self._create_resource()
                self.in_use.add(resource)
                self.created_count += 1
                self.metrics['total_created'] += 1
                self.metrics['currently_in_use'] = len(self.in_use)
                return resource
            
        # Wait for resource to become available
        while True:
            await asyncio.sleep(0.1)
            async with self.lock:
                if self.pool:
                    resource = self.pool.popleft()
                    self.in_use.add(resource)
                    wait_time = time.perf_counter() - start_time
                    self.metrics['wait_times'].append(wait_time)
                    self.metrics['currently_in_use'] = len(self.in_use)
                    self.metrics['pool_size'] = len(self.pool)
                    return resource
    
    async def return_resource(self, resource: Any):
        """Return resource to pool"""
        async with self.lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                
                # Return to pool if under max pool size
                if len(self.pool) < self.max_size:
                    self.pool.append(resource)
                else:
                    # Destroy excess resource
                    await self._destroy_resource(resource)
                    self.created_count -= 1
                
                self.metrics['currently_in_use'] = len(self.in_use)
                self.metrics['pool_size'] = len(self.pool)
    
    async def _create_resource(self) -> Any:
        """Create new resource"""
        if asyncio.iscoroutinefunction(self.factory):
            return await self.factory()
        else:
            return self.factory()
    
    async def _destroy_resource(self, resource: Any):
        """Destroy resource"""
        if hasattr(resource, 'close'):
            if asyncio.iscoroutinefunction(resource.close):
                await resource.close()
            else:
                resource.close()
    
    @asynccontextmanager
    async def acquire(self):
        """Context manager for resource acquisition"""
        resource = await self.get_resource()
        try:
            yield resource
        finally:
            await self.return_resource(resource)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics"""
        metrics = self.metrics.copy()
        if metrics['wait_times']:
            metrics['avg_wait_time'] = sum(metrics['wait_times']) / len(metrics['wait_times'])
            metrics['max_wait_time'] = max(metrics['wait_times'])
        return metrics


class ModelOptimizer:
    """AI/ML model performance optimizer"""
    
    def __init__(self):
        self.optimized_models = weakref.WeakKeyDictionary()
        self.optimization_cache = {}
    
    def optimize_pytorch_model(self, model: torch.nn.Module, 
                             input_shape: tuple, 
                             enable_jit: bool = True,
                             enable_quantization: bool = False,
                             device: str = "auto") -> torch.nn.Module:
        """Optimize PyTorch model for inference"""
        try:
            # Determine optimal device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model = model.to(device)
            model.eval()
            
            # Create dummy input for optimization
            dummy_input = torch.randn(input_shape).to(device)
            
            # Apply JIT compilation if enabled
            if enable_jit:
                try:
                    model = torch.jit.trace(model, dummy_input)
                    logger.info("Applied JIT compilation to model")
                except Exception as e:
                    logger.warning(f"JIT compilation failed: {e}")
            
            # Apply quantization if enabled
            if enable_quantization and device == "cpu":
                try:
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("Applied dynamic quantization to model")
                except Exception as e:
                    logger.warning(f"Quantization failed: {e}")
            
            # Enable inference optimizations
            torch.set_grad_enabled(False)
            
            # Cache optimized model
            self.optimized_models[model] = {
                'device': device,
                'input_shape': input_shape,
                'jit_enabled': enable_jit,
                'quantized': enable_quantization
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def optimize_inference_batch_size(self, model: torch.nn.Module,
                                    input_shape: tuple,
                                    max_memory_gb: float = 4.0) -> int:
        """Find optimal batch size for inference"""
        if id(model) in self.optimization_cache:
            return self.optimization_cache[id(model)]['optimal_batch_size']
        
        device = next(model.parameters()).device
        
        # Start with batch size 1 and increase
        batch_size = 1
        optimal_batch_size = 1
        
        try:
            while batch_size <= 256:  # Reasonable upper limit
                try:
                    # Create test input
                    test_input = torch.randn(batch_size, *input_shape[1:]).to(device)
                    
                    # Measure memory usage
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                        
                    with torch.no_grad():
                        _ = model(test_input)
                    
                    if device.type == 'cuda':
                        memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                        if memory_used > max_memory_gb:
                            break
                    
                    optimal_batch_size = batch_size
                    batch_size *= 2
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        break
                    else:
                        raise
        
        except Exception as e:
            logger.warning(f"Batch size optimization failed: {e}")
        
        self.optimization_cache[id(model)] = {'optimal_batch_size': optimal_batch_size}
        return optimal_batch_size
    
    def benchmark_model(self, model: torch.nn.Module, 
                       input_shape: tuple,
                       num_runs: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Synchronize device
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        return {
            'avg_inference_time': avg_time,
            'throughput_per_second': throughput,
            'total_benchmark_time': total_time
        }


class AsyncTaskScheduler:
    """Advanced async task scheduler with load balancing"""
    
    def __init__(self, max_concurrent_tasks: int = 100):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue = asyncio.Queue()
        self.running_tasks = set()
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.task_metrics = defaultdict(list)
        self.is_running = False
        self._scheduler_task = None
    
    async def start(self):
        """Start the task scheduler"""
        if self.is_running:
            return
        
        self.is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("AsyncTaskScheduler started")
    
    async def stop(self):
        """Stop the task scheduler"""
        self.is_running = False
        
        if self._scheduler_task:
            await self._scheduler_task
        
        # Wait for running tasks to complete
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks, return_exceptions=True)
        
        logger.info("AsyncTaskScheduler stopped")
    
    async def submit_task(self, coro, priority: int = 0, task_id: Optional[str] = None):
        """Submit a task for execution"""
        task_info = {
            'coro': coro,
            'priority': priority,
            'task_id': task_id or str(id(coro)),
            'submitted_at': time.time()
        }
        
        await self.task_queue.put(task_info)
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                # Check if we can start more tasks
                if len(self.running_tasks) < self.max_concurrent_tasks:
                    try:
                        # Get next task (with timeout to allow checking is_running)
                        task_info = await asyncio.wait_for(
                            self.task_queue.get(), 
                            timeout=1.0
                        )
                        
                        # Start the task
                        task = asyncio.create_task(
                            self._execute_task(task_info)
                        )
                        self.running_tasks.add(task)
                        
                    except asyncio.TimeoutError:
                        continue
                else:
                    # Wait for some tasks to complete
                    await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_task(self, task_info: Dict[str, Any]):
        """Execute a single task"""
        task_id = task_info['task_id']
        start_time = time.time()
        
        try:
            result = await task_info['coro']
            
            execution_time = time.time() - start_time
            self.completed_tasks += 1
            
            self.task_metrics[task_id].append({
                'execution_time': execution_time,
                'success': True,
                'timestamp': start_time
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.failed_tasks += 1
            
            self.task_metrics[task_id].append({
                'execution_time': execution_time,
                'success': False,
                'error': str(e),
                'timestamp': start_time
            })
            
            logger.error(f"Task {task_id} failed: {e}")
            raise
            
        finally:
            self.running_tasks.discard(asyncio.current_task())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            'is_running': self.is_running,
            'queue_size': self.task_queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'max_concurrent_tasks': self.max_concurrent_tasks
        }


class LoadBalancer:
    """Load balancer for distributing requests across resources"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.resources = []
        self.current_index = 0
        self.resource_stats = defaultdict(lambda: {
            'requests': 0,
            'failures': 0,
            'avg_response_time': 0.0,
            'last_used': 0.0
        })
        self.lock = threading.Lock()
    
    def add_resource(self, resource: Any):
        """Add a resource to the load balancer"""
        with self.lock:
            if resource not in self.resources:
                self.resources.append(resource)
                logger.info(f"Added resource to load balancer: {resource}")
    
    def remove_resource(self, resource: Any):
        """Remove a resource from the load balancer"""
        with self.lock:
            if resource in self.resources:
                self.resources.remove(resource)
                if resource in self.resource_stats:
                    del self.resource_stats[resource]
                logger.info(f"Removed resource from load balancer: {resource}")
    
    def get_resource(self) -> Optional[Any]:
        """Get next resource based on load balancing strategy"""
        with self.lock:
            if not self.resources:
                return None
            
            if self.strategy == "round_robin":
                resource = self.resources[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.resources)
                return resource
            
            elif self.strategy == "least_connections":
                # Return resource with fewest active requests
                return min(self.resources, key=lambda r: self.resource_stats[r]['requests'])
            
            elif self.strategy == "fastest_response":
                # Return resource with best average response time
                return min(self.resources, 
                         key=lambda r: self.resource_stats[r]['avg_response_time'] or float('inf'))
            
            else:
                # Default to round robin
                return self.resources[self.current_index % len(self.resources)]
    
    def record_request(self, resource: Any, response_time: float, success: bool = True):
        """Record request statistics for a resource"""
        with self.lock:
            stats = self.resource_stats[resource]
            
            stats['requests'] += 1
            if not success:
                stats['failures'] += 1
            
            # Update average response time
            if stats['avg_response_time'] == 0:
                stats['avg_response_time'] = response_time
            else:
                stats['avg_response_time'] = (
                    stats['avg_response_time'] * 0.9 + response_time * 0.1
                )
            
            stats['last_used'] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        with self.lock:
            return {
                'strategy': self.strategy,
                'total_resources': len(self.resources),
                'resource_stats': dict(self.resource_stats)
            }


class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.model_optimizer = ModelOptimizer()
        self.task_scheduler = AsyncTaskScheduler()
        self.load_balancer = LoadBalancer()
        self.resource_pools = {}
        self.metrics_collector = MetricsCollector()
        self.is_monitoring = False
        self._monitoring_task = None
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        await self.task_scheduler.start()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        
        if self._monitoring_task:
            await self._monitoring_task
        
        await self.task_scheduler.stop()
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = self._collect_performance_metrics()
                
                # Log metrics
                self.metrics_collector.record_metrics(metrics.__dict__)
                
                # Check for performance issues
                await self._check_performance_issues(metrics)
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        process = psutil.Process()
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process_memory = process.memory_info()
        
        # Network I/O
        try:
            network_io = psutil.net_io_counters()
            network_data = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }
        except:
            network_data = {}
        
        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=process_memory.rss / 1024**3,  # GB
            memory_percent=memory.percent,
            disk_usage=disk.percent,
            network_io=network_data,
            process_count=len(psutil.pids()),
            thread_count=process.num_threads()
        )
    
    async def _check_performance_issues(self, metrics: PerformanceMetrics):
        """Check for performance issues and take corrective actions"""
        issues = []
        
        # High CPU usage
        if metrics.cpu_usage > 80:
            issues.append("High CPU usage detected")
            # Could implement CPU throttling or load shedding here
        
        # High memory usage
        if metrics.memory_percent > 85:
            issues.append("High memory usage detected")
            # Trigger garbage collection
            gc.collect()
        
        # High disk usage
        if metrics.disk_usage > 90:
            issues.append("High disk usage detected")
            # Could implement log rotation or data cleanup
        
        if issues:
            logger.warning(f"Performance issues detected: {', '.join(issues)}")
    
    def create_resource_pool(self, name: str, factory: Callable, 
                           max_size: int = 10) -> ResourcePool:
        """Create a new resource pool"""
        pool = ResourcePool(factory, max_size=max_size)
        self.resource_pools[name] = pool
        return pool
    
    def get_resource_pool(self, name: str) -> Optional[ResourcePool]:
        """Get resource pool by name"""
        return self.resource_pools.get(name)
    
    def optimize_for_gpu(self):
        """Optimize system for GPU usage"""
        if torch.cuda.is_available():
            # Set GPU memory growth
            torch.cuda.empty_cache()
            
            # Enable TensorFloat-32 (TF32) on Ampere GPUs
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN auto-tuner
            torch.backends.cudnn.benchmark = True
            
            logger.info("GPU optimizations applied")
        else:
            logger.warning("No GPU available for optimization")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'profiler_stats': self.profiler.get_stats(),
            'task_scheduler_stats': self.task_scheduler.get_stats(),
            'load_balancer_stats': self.load_balancer.get_stats(),
            'resource_pools': {
                name: pool.get_metrics() 
                for name, pool in self.resource_pools.items()
            },
            'system_metrics': self._collect_performance_metrics().__dict__,
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


# Global performance optimizer instance
global_optimizer = PerformanceOptimizer()


# Decorators for easy performance monitoring
def performance_monitor(name: Optional[str] = None):
    """Decorator for performance monitoring"""
    return global_optimizer.profiler.profile(name)


@lru_cache(maxsize=1000)
def cached_computation(func: Callable) -> Callable:
    """Decorator for caching expensive computations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # This is a placeholder - actual implementation would use proper cache key generation
        return func(*args, **kwargs)
    return wrapper


# Context managers for performance optimization
@asynccontextmanager
async def optimized_execution():
    """Context manager for optimized execution environment"""
    # Set optimal thread count
    original_threads = torch.get_num_threads()
    optimal_threads = min(mp.cpu_count(), 8)
    torch.set_num_threads(optimal_threads)
    
    try:
        yield
    finally:
        # Restore original settings
        torch.set_num_threads(original_threads)