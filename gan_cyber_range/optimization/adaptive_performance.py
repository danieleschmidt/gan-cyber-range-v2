#!/usr/bin/env python3
"""
Adaptive performance optimization for defensive cybersecurity platform
Implements auto-scaling, resource pooling, and performance monitoring
"""

import time
import threading
import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json
import uuid
from pathlib import Path
import queue
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for system monitoring"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    network_io: float
    disk_io: float
    active_tasks: int
    completed_tasks: int
    error_rate: float
    response_time: float
    throughput: float


@dataclass
class ResourcePool:
    """Resource pool configuration"""
    pool_type: str
    min_size: int
    max_size: int
    current_size: int = 0
    active_resources: int = 0
    queue_size: int = 0
    creation_time: str = field(default_factory=lambda: datetime.now().isoformat())


class AdaptiveResourcePool:
    """Adaptive resource pool with auto-scaling"""
    
    def __init__(self, 
                 pool_type: str = "thread",
                 min_workers: int = 2,
                 max_workers: int = 20,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3):
        
        self.pool_type = pool_type
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_workers = min_workers
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        self.task_queue = queue.Queue()
        self.results = {}
        
        # Performance tracking
        self.performance_history = []
        self.last_scale_time = datetime.now()
        self.scale_cooldown = timedelta(seconds=30)
        
        # Initialize executor
        self._initialize_executor()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Initialized adaptive resource pool: {pool_type} (min={min_workers}, max={max_workers})")
    
    def _initialize_executor(self) -> None:
        """Initialize the appropriate executor"""
        if self.pool_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
        elif self.pool_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.current_workers)
        else:
            raise ValueError(f"Unsupported pool type: {self.pool_type}")
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit task to the resource pool"""
        task_id = str(uuid.uuid4())
        
        try:
            # Submit to executor
            future = self.executor.submit(func, *args, **kwargs)
            
            # Track task
            self.results[task_id] = {
                "future": future,
                "submitted_at": datetime.now().isoformat(),
                "status": "running"
            }
            
            self.active_tasks += 1
            
            # Callback for completion
            future.add_done_callback(lambda f: self._task_completed(task_id, f))
            
            logger.debug(f"Task {task_id} submitted to {self.pool_type} pool")
            return task_id
            
        except Exception as e:
            logger.error(f"Error submitting task: {str(e)}")
            self.failed_tasks += 1
            return task_id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result of submitted task"""
        if task_id not in self.results:
            raise ValueError(f"Task {task_id} not found")
        
        task_info = self.results[task_id]
        future = task_info["future"]
        
        try:
            result = future.result(timeout=timeout)
            task_info["status"] = "completed"
            return result
        except Exception as e:
            task_info["status"] = "failed"
            task_info["error"] = str(e)
            raise e
    
    def _task_completed(self, task_id: str, future) -> None:
        """Handle task completion"""
        self.active_tasks = max(0, self.active_tasks - 1)
        
        if future.exception():
            self.failed_tasks += 1
            logger.debug(f"Task {task_id} failed: {future.exception()}")
        else:
            self.completed_tasks += 1
            logger.debug(f"Task {task_id} completed successfully")
    
    def _monitor_performance(self) -> None:
        """Monitor performance and auto-scale"""
        while self.monitoring:
            try:
                utilization = self.active_tasks / self.current_workers if self.current_workers > 0 else 0
                
                # Record performance
                metrics = PerformanceMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_usage=0.0,  # Placeholder
                    memory_usage=0.0,  # Placeholder
                    network_io=0.0,  # Placeholder
                    disk_io=0.0,  # Placeholder
                    active_tasks=self.active_tasks,
                    completed_tasks=self.completed_tasks,
                    error_rate=self.failed_tasks / max(1, self.completed_tasks + self.failed_tasks),
                    response_time=0.0,  # Placeholder
                    throughput=utilization
                )
                
                self.performance_history.append(metrics)
                
                # Keep only last 100 metrics
                if len(self.performance_history) > 100:
                    self.performance_history = self.performance_history[-100:]
                
                # Check if we need to scale
                self._check_scaling(utilization)
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                time.sleep(10)
    
    def _check_scaling(self, utilization: float) -> None:
        """Check if scaling is needed"""
        now = datetime.now()
        
        # Ensure cooldown period
        if now - self.last_scale_time < self.scale_cooldown:
            return
        
        # Scale up if high utilization
        if utilization >= self.scale_up_threshold and self.current_workers < self.max_workers:
            self._scale_up()
            self.last_scale_time = now
        
        # Scale down if low utilization
        elif utilization <= self.scale_down_threshold and self.current_workers > self.min_workers:
            self._scale_down()
            self.last_scale_time = now
    
    def _scale_up(self) -> None:
        """Scale up the resource pool"""
        new_size = min(self.current_workers + 2, self.max_workers)
        
        if new_size > self.current_workers:
            logger.info(f"Scaling up {self.pool_type} pool: {self.current_workers} -> {new_size}")
            
            # Recreate executor with new size
            self.executor.shutdown(wait=False)
            self.current_workers = new_size
            self._initialize_executor()
    
    def _scale_down(self) -> None:
        """Scale down the resource pool"""
        new_size = max(self.current_workers - 1, self.min_workers)
        
        if new_size < self.current_workers:
            logger.info(f"Scaling down {self.pool_type} pool: {self.current_workers} -> {new_size}")
            
            # Recreate executor with new size
            self.executor.shutdown(wait=False)
            self.current_workers = new_size
            self._initialize_executor()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics"""
        recent_metrics = self.performance_history[-10:] if self.performance_history else []
        
        return {
            "pool_type": self.pool_type,
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.completed_tasks / max(1, self.completed_tasks + self.failed_tasks),
            "current_utilization": self.active_tasks / max(1, self.current_workers),
            "recent_performance": [vars(m) for m in recent_metrics]
        }
    
    def shutdown(self) -> None:
        """Shutdown the resource pool"""
        self.monitoring = False
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        logger.info(f"Resource pool {self.pool_type} shutdown completed")


class PerformanceOptimizer:
    """Comprehensive performance optimization system"""
    
    def __init__(self):
        self.resource_pools = {}
        self.cache = {}
        self.performance_targets = {
            "max_response_time": 5.0,  # seconds
            "min_throughput": 10.0,     # tasks per second
            "max_error_rate": 0.05,     # 5%
            "max_cpu_usage": 0.8,       # 80%
            "max_memory_usage": 0.8     # 80%
        }
        
        self.optimization_history = []
        self.is_optimizing = False
        
        logger.info("Performance optimizer initialized")
    
    def create_resource_pool(self, 
                           name: str,
                           pool_type: str = "thread",
                           min_workers: int = 2,
                           max_workers: int = 20) -> str:
        """Create a new adaptive resource pool"""
        
        if name in self.resource_pools:
            logger.warning(f"Resource pool {name} already exists")
            return name
        
        pool = AdaptiveResourcePool(
            pool_type=pool_type,
            min_workers=min_workers,
            max_workers=max_workers
        )
        
        self.resource_pools[name] = pool
        logger.info(f"Created resource pool: {name}")
        return name
    
    def submit_to_pool(self, pool_name: str, func: Callable, *args, **kwargs) -> str:
        """Submit task to specific resource pool"""
        if pool_name not in self.resource_pools:
            raise ValueError(f"Resource pool {pool_name} not found")
        
        return self.resource_pools[pool_name].submit_task(func, *args, **kwargs)
    
    def get_from_pool(self, pool_name: str, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result from specific resource pool"""
        if pool_name not in self.resource_pools:
            raise ValueError(f"Resource pool {pool_name} not found")
        
        return self.resource_pools[pool_name].get_task_result(task_id, timeout)
    
    def optimize_batch_processing(self, 
                                tasks: List[Tuple[Callable, tuple, dict]],
                                pool_name: str = "batch_pool",
                                chunk_size: Optional[int] = None) -> List[Any]:
        """Optimize batch processing of tasks"""
        
        if pool_name not in self.resource_pools:
            self.create_resource_pool(pool_name, "thread", 4, 16)
        
        pool = self.resource_pools[pool_name]
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(tasks) // (pool.current_workers * 2))
        
        logger.info(f"Processing {len(tasks)} tasks in chunks of {chunk_size}")
        
        # Submit tasks in chunks
        task_ids = []
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            chunk_func = lambda c=chunk: [func(*args, **kwargs) for func, args, kwargs in c]
            task_id = pool.submit_task(chunk_func)
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            try:
                chunk_results = pool.get_task_result(task_id, timeout=30.0)
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Chunk processing failed: {str(e)}")
                results.extend([None] * chunk_size)
        
        return results
    
    def cache_result(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Cache result with optional TTL"""
        cache_entry = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "ttl": ttl,
            "hits": 0
        }
        
        self.cache[key] = cache_entry
        logger.debug(f"Cached result for key: {key}")
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if valid"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if entry.get("ttl"):
            cache_time = datetime.fromisoformat(entry["timestamp"])
            if datetime.now() - cache_time > timedelta(seconds=entry["ttl"]):
                del self.cache[key]
                return None
        
        # Update hit count
        entry["hits"] += 1
        logger.debug(f"Cache hit for key: {key}")
        return entry["value"]
    
    def clear_cache(self) -> int:
        """Clear cache and return number of cleared entries"""
        cleared = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {cleared} cache entries")
        return cleared
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Resource pool statistics
        pool_stats = {}
        for name, pool in self.resource_pools.items():
            pool_stats[name] = pool.get_stats()
        
        # Cache statistics
        cache_stats = {
            "total_entries": len(self.cache),
            "total_hits": sum(entry.get("hits", 0) for entry in self.cache.values()),
            "hit_rate": 0.0
        }
        
        if cache_stats["total_entries"] > 0:
            total_requests = cache_stats["total_hits"] + cache_stats["total_entries"]
            cache_stats["hit_rate"] = cache_stats["total_hits"] / total_requests
        
        # System utilization
        total_workers = sum(pool.current_workers for pool in self.resource_pools.values())
        total_active = sum(pool.active_tasks for pool in self.resource_pools.values())
        system_utilization = total_active / max(1, total_workers)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "resource_pools": pool_stats,
            "cache_statistics": cache_stats,
            "system_utilization": system_utilization,
            "performance_targets": self.performance_targets,
            "optimization_suggestions": self._generate_optimization_suggestions()
        }
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on current performance"""
        suggestions = []
        
        # Analyze resource pools
        for name, pool in self.resource_pools.items():
            stats = pool.get_stats()
            utilization = stats["current_utilization"]
            
            if utilization > 0.9:
                suggestions.append(f"Consider increasing max workers for pool '{name}' (utilization: {utilization:.1%})")
            elif utilization < 0.1 and stats["current_workers"] > stats["min_workers"]:
                suggestions.append(f"Consider reducing min workers for pool '{name}' (utilization: {utilization:.1%})")
            
            if stats["success_rate"] < 0.95:
                suggestions.append(f"Investigate errors in pool '{name}' (success rate: {stats['success_rate']:.1%})")
        
        # Cache analysis
        if len(self.cache) == 0:
            suggestions.append("Consider implementing caching for frequently accessed data")
        elif len(self.cache) > 1000:
            suggestions.append("Consider implementing cache size limits and cleanup policies")
        
        return suggestions
    
    def shutdown(self) -> None:
        """Shutdown all resource pools"""
        logger.info("Shutting down performance optimizer...")
        
        for name, pool in self.resource_pools.items():
            try:
                pool.shutdown()
                logger.info(f"Shutdown pool: {name}")
            except Exception as e:
                logger.error(f"Error shutting down pool {name}: {str(e)}")
        
        self.resource_pools.clear()
        logger.info("Performance optimizer shutdown completed")


class DefensiveWorkloadManager:
    """Specialized workload manager for defensive operations"""
    
    def __init__(self):
        self.optimizer = PerformanceOptimizer()
        
        # Create specialized pools for defensive tasks
        self.optimizer.create_resource_pool("attack_generation", "process", 2, 8)
        self.optimizer.create_resource_pool("threat_analysis", "thread", 4, 16)
        self.optimizer.create_resource_pool("monitoring", "thread", 2, 6)
        self.optimizer.create_resource_pool("training", "thread", 3, 12)
        
        logger.info("Defensive workload manager initialized")
    
    def generate_attacks_batch(self, 
                             attack_configs: List[Dict],
                             cache_results: bool = True) -> List[Any]:
        """Generate attacks in optimized batches"""
        
        # Check cache first
        if cache_results:
            cache_key = f"attacks_batch_{hash(str(sorted(str(c) for c in attack_configs)))}"
            cached_result = self.optimizer.get_cached_result(cache_key)
            if cached_result:
                logger.info("Returned cached attack generation results")
                return cached_result
        
        # Create generation tasks
        def generate_attack(config):
            # Simulate attack generation
            time.sleep(0.1 + len(config.get("payload", "")) * 0.001)
            return {
                "attack_id": str(uuid.uuid4()),
                "config": config,
                "generated_at": datetime.now().isoformat()
            }
        
        tasks = [(generate_attack, (config,), {}) for config in attack_configs]
        
        # Process with optimization
        results = self.optimizer.optimize_batch_processing(tasks, "attack_generation")
        
        # Cache results
        if cache_results and results:
            self.optimizer.cache_result(cache_key, results, ttl=300)  # 5 minute TTL
        
        logger.info(f"Generated {len(results)} attacks using optimized batch processing")
        return results
    
    def analyze_threats_parallel(self, 
                               threat_data: List[Dict],
                               analysis_depth: str = "standard") -> List[Dict]:
        """Analyze threats in parallel for faster processing"""
        
        def analyze_threat(threat):
            # Simulate threat analysis
            complexity = {"basic": 0.05, "standard": 0.1, "deep": 0.2}
            time.sleep(complexity.get(analysis_depth, 0.1))
            
            return {
                "threat_id": threat.get("id", str(uuid.uuid4())),
                "risk_score": hash(str(threat)) % 100 / 100.0,
                "analysis_depth": analysis_depth,
                "recommendations": ["Monitor", "Alert", "Block"][:hash(str(threat)) % 3 + 1],
                "analyzed_at": datetime.now().isoformat()
            }
        
        tasks = [(analyze_threat, (threat,), {}) for threat in threat_data]
        
        results = self.optimizer.optimize_batch_processing(tasks, "threat_analysis", chunk_size=5)
        
        logger.info(f"Analyzed {len(results)} threats with {analysis_depth} depth")
        return results
    
    def run_training_scenarios(self, 
                             scenarios: List[Dict],
                             parallel_execution: bool = True) -> List[Dict]:
        """Run training scenarios with performance optimization"""
        
        def execute_scenario(scenario):
            # Simulate scenario execution
            duration = scenario.get("duration", 30)
            time.sleep(min(duration / 100, 2.0))  # Scaled simulation time
            
            return {
                "scenario_id": scenario.get("id", str(uuid.uuid4())),
                "status": "completed",
                "score": hash(str(scenario)) % 100,
                "completion_time": duration,
                "executed_at": datetime.now().isoformat()
            }
        
        if parallel_execution:
            tasks = [(execute_scenario, (scenario,), {}) for scenario in scenarios]
            results = self.optimizer.optimize_batch_processing(tasks, "training", chunk_size=3)
        else:
            # Sequential execution for comparison
            results = [execute_scenario(scenario) for scenario in scenarios]
        
        logger.info(f"Executed {len(results)} training scenarios ({'parallel' if parallel_execution else 'sequential'})")
        return results
    
    def get_workload_stats(self) -> Dict[str, Any]:
        """Get comprehensive workload statistics"""
        base_report = self.optimizer.get_performance_report()
        
        # Add defensive-specific metrics
        defensive_metrics = {
            "specialized_pools": len(self.optimizer.resource_pools),
            "total_workers": sum(pool.current_workers for pool in self.optimizer.resource_pools.values()),
            "total_active_tasks": sum(pool.active_tasks for pool in self.optimizer.resource_pools.values()),
            "cache_efficiency": len(self.optimizer.cache)
        }
        
        base_report["defensive_metrics"] = defensive_metrics
        return base_report
    
    def shutdown(self) -> None:
        """Shutdown workload manager"""
        logger.info("Shutting down defensive workload manager...")
        self.optimizer.shutdown()


if __name__ == "__main__":
    # Test adaptive performance optimization
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    workload_manager = DefensiveWorkloadManager()
    
    # Test attack generation
    attack_configs = [
        {"type": "malware", "payload": "test1"},
        {"type": "network", "payload": "test2"},
        {"type": "web", "payload": "test3"},
        {"type": "social", "payload": "test4"},
    ]
    
    print("Testing optimized attack generation...")
    start_time = time.time()
    attacks = workload_manager.generate_attacks_batch(attack_configs)
    generation_time = time.time() - start_time
    print(f"Generated {len(attacks)} attacks in {generation_time:.2f}s")
    
    # Test threat analysis
    threat_data = [{"id": f"threat_{i}", "data": f"threat_data_{i}"} for i in range(10)]
    
    print("Testing parallel threat analysis...")
    start_time = time.time()
    analyses = workload_manager.analyze_threats_parallel(threat_data, "standard")
    analysis_time = time.time() - start_time
    print(f"Analyzed {len(analyses)} threats in {analysis_time:.2f}s")
    
    # Test training scenarios
    scenarios = [{"id": f"scenario_{i}", "duration": 30} for i in range(6)]
    
    print("Testing parallel training execution...")
    start_time = time.time()
    results = workload_manager.run_training_scenarios(scenarios, parallel_execution=True)
    parallel_time = time.time() - start_time
    
    print("Testing sequential training execution...")
    start_time = time.time()
    sequential_results = workload_manager.run_training_scenarios(scenarios, parallel_execution=False)
    sequential_time = time.time() - start_time
    
    print(f"Parallel execution: {parallel_time:.2f}s")
    print(f"Sequential execution: {sequential_time:.2f}s")
    print(f"Performance improvement: {sequential_time/parallel_time:.1f}x")
    
    # Get performance report
    report = workload_manager.get_workload_stats()
    print(f"\nPerformance Report:")
    print(f"System utilization: {report['system_utilization']:.1%}")
    print(f"Cache entries: {report['cache_statistics']['total_entries']}")
    print(f"Optimization suggestions: {len(report['optimization_suggestions'])}")
    
    workload_manager.shutdown()
    print("\nAdaptive performance optimization test completed ✅")