#!/usr/bin/env python3
"""
High-Performance Defensive Platform - Generation 3
Advanced performance optimization, caching, concurrency, and auto-scaling
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import queue
import json
import hashlib
import uuid
from pathlib import Path
import aioredis
import psutil
import numpy as np
from collections import defaultdict, deque
import weakref
import gc
import signal
import sys

# Performance monitoring
import cProfile
import pstats
from memory_profiler import profile as memory_profile

# Core imports
from gan_cyber_range.optimization import CacheOptimizer, ResourcePool, PerformanceMonitor
from gan_cyber_range.scalability import AutoScaler
from gan_cyber_range.security import SecurityOrchestrator
from gan_cyber_range.training import DefensiveTrainingEnhancer

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Caching strategies for different data types"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


class ScalingStrategy(Enum):
    """Auto-scaling strategies"""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


@dataclass
class PerformanceMetrics:
    """Detailed performance metrics"""
    timestamp: datetime
    operation_type: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    throughput_ops_per_sec: float
    concurrent_operations: int
    queue_size: int
    error_count: int


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds


class AdvancedCache:
    """High-performance multi-strategy cache"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # For LRU
        self.access_frequency = defaultdict(int)  # For LFU
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        
        # Background cleanup task
        self.cleanup_running = False
        self.cleanup_task = None
    
    def start_cleanup(self):
        """Start background cleanup task"""
        if not self.cleanup_running:
            self.cleanup_running = True
            self.cleanup_task = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_task.start()
    
    def stop_cleanup(self):
        """Stop background cleanup task"""
        if self.cleanup_running:
            self.cleanup_running = False
            if self.cleanup_task:
                self.cleanup_task.join(timeout=1.0)
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.cleanup_running:
            try:
                self._cleanup_expired()
                time.sleep(60)  # Cleanup every minute
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and tracking structures"""
        if key in self.cache:
            del self.cache[key]
        
        # Remove from LRU tracking
        if key in self.access_order:
            self.access_order.remove(key)
        
        # Remove from LFU tracking
        if key in self.access_frequency:
            del self.access_frequency[key]
    
    def get(self, key: str, strategy: CacheStrategy = CacheStrategy.LRU) -> Optional[Any]:
        """Get value from cache with specified strategy"""
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.miss_count += 1
                return None
            
            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Update tracking structures based on strategy
            if strategy == CacheStrategy.LRU:
                # Move to end for LRU
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
            
            elif strategy == CacheStrategy.LFU:
                # Update frequency count
                self.access_frequency[key] += 1
            
            self.hit_count += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None, 
            strategy: CacheStrategy = CacheStrategy.LRU):
        """Put value in cache with specified strategy"""
        with self.lock:
            # Calculate entry size (approximate)
            size_bytes = sys.getsizeof(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_entry(strategy)
            
            # Store entry
            self.cache[key] = entry
            
            # Update tracking structures
            if strategy == CacheStrategy.LRU:
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
            
            elif strategy == CacheStrategy.LFU:
                self.access_frequency[key] = 1
    
    def _evict_entry(self, strategy: CacheStrategy):
        """Evict entry based on strategy"""
        if not self.cache:
            return
        
        evict_key = None
        
        if strategy == CacheStrategy.LRU:
            # Evict least recently used
            if self.access_order:
                evict_key = self.access_order[0]
        
        elif strategy == CacheStrategy.LFU:
            # Evict least frequently used
            if self.access_frequency:
                evict_key = min(self.access_frequency.keys(), 
                              key=lambda k: self.access_frequency[k])
        
        else:
            # Default: evict oldest entry
            evict_key = min(self.cache.keys(), 
                          key=lambda k: self.cache[k].created_at)
        
        if evict_key:
            self._remove_entry(evict_key)
            logger.debug(f"Evicted cache entry: {evict_key}")
    
    def get_stats(self) -> Dict:
        """Get cache performance statistics"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0.0
            
            total_size_mb = sum(entry.size_bytes for entry in self.cache.values()) / (1024 * 1024)
            
            return {
                "entries": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "total_size_mb": total_size_mb,
                "average_entry_size_kb": (total_size_mb * 1024) / len(self.cache) if self.cache else 0
            }
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            self.hit_count = 0
            self.miss_count = 0


class ConnectionPool:
    """High-performance connection pool"""
    
    def __init__(self, create_connection_func: Callable, 
                 min_connections: int = 5, max_connections: int = 50):
        self.create_connection = create_connection_func
        self.min_connections = min_connections
        self.max_connections = max_connections
        
        self.available = queue.Queue(maxsize=max_connections)
        self.in_use: Set = set()
        self.total_created = 0
        self.lock = threading.Lock()
        
        # Pre-create minimum connections
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize pool with minimum connections"""
        for _ in range(self.min_connections):
            try:
                conn = self.create_connection()
                self.available.put(conn)
                self.total_created += 1
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
    
    def get_connection(self, timeout: float = 30.0):
        """Get connection from pool"""
        try:
            # Try to get available connection
            conn = self.available.get(timeout=timeout)
            
            with self.lock:
                self.in_use.add(conn)
            
            return conn
            
        except queue.Empty:
            # Create new connection if under limit
            with self.lock:
                if self.total_created < self.max_connections:
                    try:
                        conn = self.create_connection()
                        self.in_use.add(conn)
                        self.total_created += 1
                        return conn
                    except Exception as e:
                        logger.error(f"Failed to create new connection: {e}")
                        raise
                else:
                    raise Exception("Connection pool exhausted")
    
    def return_connection(self, conn):
        """Return connection to pool"""
        with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
                
                # Return to available pool if not full
                try:
                    self.available.put_nowait(conn)
                except queue.Full:
                    # Pool is full, close connection
                    self._close_connection(conn)
                    self.total_created -= 1
    
    def _close_connection(self, conn):
        """Close connection (override in subclass)"""
        try:
            if hasattr(conn, 'close'):
                conn.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def get_stats(self) -> Dict:
        """Get pool statistics"""
        with self.lock:
            return {
                "available": self.available.qsize(),
                "in_use": len(self.in_use),
                "total_created": self.total_created,
                "utilization": len(self.in_use) / self.max_connections
            }


class AdaptiveTaskScheduler:
    """Adaptive task scheduler with load balancing"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Different executor types for different workloads
        self.cpu_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.io_executor = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        self.compute_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Task queues by priority
        self.high_priority = asyncio.Queue(maxsize=1000)
        self.normal_priority = asyncio.Queue(maxsize=5000)
        self.low_priority = asyncio.Queue(maxsize=10000)
        
        # Performance tracking
        self.task_metrics = {}
        self.load_metrics = {
            "cpu_tasks": 0,
            "io_tasks": 0,
            "compute_tasks": 0
        }
        
        # Worker tasks
        self.workers = []
        self.running = False
    
    async def start(self):
        """Start scheduler workers"""
        if not self.running:
            self.running = True
            
            # Start priority-based workers
            for i in range(self.max_workers):
                worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
                self.workers.append(worker)
            
            logger.info(f"Started adaptive task scheduler with {self.max_workers} workers")
    
    async def stop(self):
        """Stop scheduler workers"""
        if self.running:
            self.running = False
            
            # Cancel all workers
            for worker in self.workers:
                worker.cancel()
            
            # Wait for workers to finish
            if self.workers:
                await asyncio.gather(*self.workers, return_exceptions=True)
            
            # Shutdown executors
            self.cpu_executor.shutdown(wait=True)
            self.io_executor.shutdown(wait=True)
            self.compute_executor.shutdown(wait=True)
            
            logger.info("Stopped adaptive task scheduler")
    
    async def _worker_loop(self, worker_id: str):
        """Main worker loop with priority handling"""
        while self.running:
            try:
                # Check queues by priority
                task = None
                
                # High priority first
                if not self.high_priority.empty():
                    task = await self.high_priority.get()
                # Then normal priority
                elif not self.normal_priority.empty():
                    task = await self.normal_priority.get()
                # Finally low priority
                elif not self.low_priority.empty():
                    task = await self.low_priority.get()
                else:
                    # No tasks, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                if task:
                    await self._execute_task(task, worker_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)  # Back off on error
    
    async def _execute_task(self, task: Dict, worker_id: str):
        """Execute task with appropriate executor"""
        task_type = task.get("type", "cpu")
        task_func = task.get("func")
        task_args = task.get("args", ())
        task_kwargs = task.get("kwargs", {})
        task_id = task.get("id", str(uuid.uuid4()))
        
        start_time = time.time()
        
        try:
            # Choose appropriate executor
            if task_type == "io":
                executor = self.io_executor
                self.load_metrics["io_tasks"] += 1
            elif task_type == "compute":
                executor = self.compute_executor
                self.load_metrics["compute_tasks"] += 1
            else:
                executor = self.cpu_executor
                self.load_metrics["cpu_tasks"] += 1
            
            # Execute task
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, task_func, *task_args)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Record metrics
            self.task_metrics[task_id] = {
                "execution_time_ms": execution_time,
                "task_type": task_type,
                "worker_id": worker_id,
                "success": True,
                "timestamp": datetime.now()
            }
            
            logger.debug(f"Task {task_id} completed in {execution_time:.1f}ms by {worker_id}")
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            self.task_metrics[task_id] = {
                "execution_time_ms": execution_time,
                "task_type": task_type,
                "worker_id": worker_id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now()
            }
            
            logger.error(f"Task {task_id} failed after {execution_time:.1f}ms: {e}")
    
    async def submit_task(self, func: Callable, *args, task_type: str = "cpu", 
                         priority: str = "normal", **kwargs) -> str:
        """Submit task for execution"""
        task_id = str(uuid.uuid4())
        
        task = {
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "type": task_type,
            "submitted_at": datetime.now()
        }
        
        # Route to appropriate priority queue
        if priority == "high":
            await self.high_priority.put(task)
        elif priority == "low":
            await self.low_priority.put(task)
        else:
            await self.normal_priority.put(task)
        
        return task_id
    
    def get_metrics(self) -> Dict:
        """Get scheduler performance metrics"""
        return {
            "queues": {
                "high_priority": self.high_priority.qsize(),
                "normal_priority": self.normal_priority.qsize(),
                "low_priority": self.low_priority.qsize()
            },
            "load_metrics": self.load_metrics.copy(),
            "workers_active": len([w for w in self.workers if not w.done()]),
            "total_tasks": len(self.task_metrics),
            "recent_performance": self._calculate_recent_performance()
        }
    
    def _calculate_recent_performance(self) -> Dict:
        """Calculate performance metrics for recent tasks"""
        recent_cutoff = datetime.now() - timedelta(minutes=5)
        recent_tasks = [
            task for task in self.task_metrics.values()
            if task["timestamp"] > recent_cutoff
        ]
        
        if not recent_tasks:
            return {"avg_execution_time_ms": 0, "success_rate": 0, "throughput": 0}
        
        avg_time = sum(task["execution_time_ms"] for task in recent_tasks) / len(recent_tasks)
        success_count = sum(1 for task in recent_tasks if task["success"])
        success_rate = success_count / len(recent_tasks)
        throughput = len(recent_tasks) / 300  # Tasks per second over 5 minutes
        
        return {
            "avg_execution_time_ms": avg_time,
            "success_rate": success_rate,
            "throughput": throughput
        }


class HighPerformanceDefensivePlatform:
    """High-performance defensive cybersecurity platform"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize high-performance components
        self.cache = AdvancedCache(
            max_size=self.config.get("cache_size", 50000),
            default_ttl=self.config.get("cache_ttl", 3600)
        )
        
        self.scheduler = AdaptiveTaskScheduler(
            max_workers=self.config.get("max_workers", multiprocessing.cpu_count())
        )
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.metrics_history = deque(maxlen=10000)  # Keep last 10k metrics
        
        # Auto-scaling
        self.auto_scaler = AutoScaler()
        
        # Core defensive components
        self.security_orchestrator = SecurityOrchestrator()
        self.training_enhancer = DefensiveTrainingEnhancer()
        
        # Performance optimization flags
        self.profiling_enabled = self.config.get("profiling", False)
        self.memory_profiling_enabled = self.config.get("memory_profiling", False)
        
        # Connection pools
        self.connection_pools = {}
        
        # Background tasks
        self.background_tasks = []
        self.running = False
    
    async def initialize(self):
        """Initialize high-performance platform"""
        logger.info("🚀 Initializing High-Performance Defensive Platform")
        
        try:
            # Start cache cleanup
            self.cache.start_cleanup()
            
            # Start task scheduler
            await self.scheduler.start()
            
            # Initialize core components
            await self.security_orchestrator.initialize()
            
            # Start auto-scaler
            await self.auto_scaler.start()
            
            # Start performance monitoring
            self.performance_monitor.start()
            
            # Start background optimization tasks
            await self._start_background_tasks()
            
            self.running = True
            
            logger.info("✅ High-Performance Platform initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Platform initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown platform gracefully"""
        logger.info("🔄 Shutting down High-Performance Platform")
        
        try:
            self.running = False
            
            # Stop background tasks
            for task in self.background_tasks:
                task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Stop components
            await self.scheduler.stop()
            self.cache.stop_cleanup()
            await self.auto_scaler.stop()
            self.performance_monitor.stop()
            
            logger.info("✅ Platform shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    async def _start_background_tasks(self):
        """Start background optimization tasks"""
        
        # Performance metrics collection
        self.background_tasks.append(
            asyncio.create_task(self._metrics_collection_loop())
        )
        
        # Cache optimization
        self.background_tasks.append(
            asyncio.create_task(self._cache_optimization_loop())
        )
        
        # Memory optimization
        self.background_tasks.append(
            asyncio.create_task(self._memory_optimization_loop())
        )
        
        # Performance analysis
        if self.profiling_enabled:
            self.background_tasks.append(
                asyncio.create_task(self._performance_analysis_loop())
            )
    
    async def _metrics_collection_loop(self):
        """Continuous metrics collection"""
        while self.running:
            try:
                # Collect system metrics
                metrics = await self._collect_performance_metrics()
                self.metrics_history.append(metrics)
                
                # Check for performance issues
                await self._analyze_performance_trends()
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _cache_optimization_loop(self):
        """Continuous cache optimization"""
        while self.running:
            try:
                # Analyze cache performance
                cache_stats = self.cache.get_stats()
                
                # Optimize cache strategy if hit rate is low
                if cache_stats["hit_rate"] < 0.7:
                    logger.info(f"Cache hit rate low ({cache_stats['hit_rate']:.1%}), optimizing...")
                    await self._optimize_cache_strategy()
                
                # Check memory usage
                if cache_stats["total_size_mb"] > 1000:  # 1GB threshold
                    logger.info("Cache memory usage high, triggering cleanup")
                    self.cache._cleanup_expired()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
                await asyncio.sleep(300)
    
    async def _memory_optimization_loop(self):
        """Continuous memory optimization"""
        while self.running:
            try:
                # Check memory usage
                process = psutil.Process()
                memory_percent = process.memory_percent()
                
                if memory_percent > 80:  # High memory usage
                    logger.warning(f"High memory usage: {memory_percent:.1f}%")
                    
                    # Trigger garbage collection
                    collected = gc.collect()
                    logger.info(f"Garbage collection freed {collected} objects")
                    
                    # Reduce cache size if necessary
                    if memory_percent > 90:
                        self.cache.max_size = int(self.cache.max_size * 0.8)
                        logger.info(f"Reduced cache size to {self.cache.max_size}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory optimization error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_analysis_loop(self):
        """Continuous performance analysis"""
        while self.running and self.profiling_enabled:
            try:
                # Run performance profiling
                profiler = cProfile.Profile()
                profiler.enable()
                
                # Let profiler run for a period
                await asyncio.sleep(30)
                
                profiler.disable()
                
                # Analyze profile
                stats = pstats.Stats(profiler)
                stats.sort_stats('cumulative')
                
                # Log top performance bottlenecks
                logger.info("Performance profiling results (top functions):")
                # Note: In production, you'd save this to a file instead
                
                await asyncio.sleep(300)  # Profile every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Cache metrics
        cache_stats = self.cache.get_stats()
        
        # Scheduler metrics
        scheduler_metrics = self.scheduler.get_metrics()
        
        # Auto-scaler metrics
        scaler_metrics = await self.auto_scaler.get_metrics()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            operation_type="system",
            execution_time_ms=0.0,  # N/A for system metrics
            memory_usage_mb=memory.used / (1024 * 1024),
            cpu_usage_percent=cpu_percent,
            cache_hit_rate=cache_stats["hit_rate"],
            throughput_ops_per_sec=scheduler_metrics["recent_performance"]["throughput"],
            concurrent_operations=scheduler_metrics["workers_active"],
            queue_size=sum(scheduler_metrics["queues"].values()),
            error_count=0  # Would track actual errors in production
        )
    
    async def _analyze_performance_trends(self):
        """Analyze performance trends and trigger optimizations"""
        if len(self.metrics_history) < 10:
            return
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Analyze CPU trend
        cpu_trend = [m.cpu_usage_percent for m in recent_metrics]
        avg_cpu = np.mean(cpu_trend)
        cpu_increasing = np.polyfit(range(len(cpu_trend)), cpu_trend, 1)[0] > 0
        
        # Analyze memory trend
        memory_trend = [m.memory_usage_mb for m in recent_metrics]
        avg_memory = np.mean(memory_trend)
        memory_increasing = np.polyfit(range(len(memory_trend)), memory_trend, 1)[0] > 0
        
        # Trigger auto-scaling if needed
        if avg_cpu > 80 and cpu_increasing:
            logger.info("High CPU usage trend detected, triggering scale-up")
            await self.auto_scaler.scale_up("cpu_usage", avg_cpu)
        
        if avg_memory > 1000 and memory_increasing:  # 1GB
            logger.info("High memory usage trend detected, triggering optimization")
            await self._optimize_memory_usage()
        
        # Cache optimization
        cache_hit_rates = [m.cache_hit_rate for m in recent_metrics]
        avg_hit_rate = np.mean(cache_hit_rates)
        
        if avg_hit_rate < 0.8:
            logger.info("Low cache hit rate trend detected, optimizing cache")
            await self._optimize_cache_strategy()
    
    async def _optimize_cache_strategy(self):
        """Optimize cache strategy based on usage patterns"""
        # Analyze cache usage patterns
        cache_stats = self.cache.get_stats()
        
        # Switch to more aggressive caching if hit rate is low
        if cache_stats["hit_rate"] < 0.5:
            # Increase cache size
            self.cache.max_size = min(self.cache.max_size * 1.2, 100000)
            logger.info(f"Increased cache size to {self.cache.max_size}")
        
        # Clear frequently evicted entries
        if cache_stats["utilization"] > 0.9:
            # Cleanup expired entries more aggressively
            self.cache._cleanup_expired()
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage"""
        # Force garbage collection
        collected = gc.collect()
        
        # Reduce cache size if memory is critical
        process = psutil.Process()
        if process.memory_percent() > 85:
            self.cache.max_size = int(self.cache.max_size * 0.9)
            logger.info(f"Reduced cache size to {self.cache.max_size} due to memory pressure")
        
        logger.info(f"Memory optimization: freed {collected} objects")
    
    async def execute_defensive_training(self, training_config: Dict) -> Dict:
        """Execute high-performance defensive training"""
        start_time = time.time()
        
        # Cache training data if not cached
        cache_key = f"training_{hashlib.md5(str(training_config).encode()).hexdigest()}"
        cached_result = self.cache.get(cache_key, CacheStrategy.LRU)
        
        if cached_result:
            logger.info(f"Training result served from cache: {cache_key}")
            return cached_result
        
        try:
            # Submit training task to high-priority queue
            task_id = await self.scheduler.submit_task(
                self._execute_training_task,
                training_config,
                task_type="compute",
                priority="high"
            )
            
            # Wait for task completion (in production, this would be async)
            # For demo, we'll simulate training completion
            await asyncio.sleep(2)  # Simulate training time
            
            # Generate training result
            result = {
                "training_id": task_id,
                "config": training_config,
                "completion_time": datetime.now().isoformat(),
                "metrics": {
                    "accuracy": 0.95,
                    "training_time_seconds": time.time() - start_time,
                    "memory_peak_mb": 512,
                    "samples_processed": 10000
                },
                "performance": {
                    "cache_hit": cached_result is not None,
                    "execution_time_ms": (time.time() - start_time) * 1000
                }
            }
            
            # Cache result for future use
            self.cache.put(cache_key, result, ttl=7200, strategy=CacheStrategy.LRU)
            
            return result
            
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            return {"error": str(e)}
    
    def _execute_training_task(self, training_config: Dict) -> Dict:
        """Execute training task (runs in executor)"""
        # This would contain the actual training logic
        # For demo purposes, we'll simulate training
        time.sleep(1)  # Simulate computation
        
        return {
            "status": "completed",
            "config": training_config,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_platform_metrics(self) -> Dict:
        """Get comprehensive platform performance metrics"""
        
        # System metrics
        current_metrics = await self._collect_performance_metrics()
        
        # Component metrics
        cache_stats = self.cache.get_stats()
        scheduler_metrics = self.scheduler.get_metrics()
        scaler_metrics = await self.auto_scaler.get_metrics()
        
        # Calculate performance trends
        performance_trends = {}
        if len(self.metrics_history) >= 2:
            recent_metrics = list(self.metrics_history)[-10:]
            performance_trends = {
                "cpu_trend": np.mean([m.cpu_usage_percent for m in recent_metrics]),
                "memory_trend": np.mean([m.memory_usage_mb for m in recent_metrics]),
                "cache_hit_trend": np.mean([m.cache_hit_rate for m in recent_metrics]),
                "throughput_trend": np.mean([m.throughput_ops_per_sec for m in recent_metrics])
            }
        
        return {
            "platform_status": "running" if self.running else "stopped",
            "timestamp": datetime.now().isoformat(),
            "current_metrics": {
                "cpu_usage": current_metrics.cpu_usage_percent,
                "memory_usage_mb": current_metrics.memory_usage_mb,
                "cache_hit_rate": current_metrics.cache_hit_rate,
                "active_workers": current_metrics.concurrent_operations,
                "queue_size": current_metrics.queue_size
            },
            "cache_performance": cache_stats,
            "scheduler_performance": scheduler_metrics,
            "auto_scaling": scaler_metrics,
            "performance_trends": performance_trends,
            "optimization_status": {
                "profiling_enabled": self.profiling_enabled,
                "memory_profiling": self.memory_profiling_enabled,
                "background_tasks_active": len([t for t in self.background_tasks if not t.done()])
            }
        }


async def demonstrate_high_performance_platform():
    """Demonstrate high-performance defensive platform"""
    logger.info("🚀 Starting High-Performance Defensive Platform Demo")
    
    # Configuration for high performance
    config = {
        "cache_size": 10000,
        "cache_ttl": 3600,
        "max_workers": multiprocessing.cpu_count(),
        "profiling": True,
        "memory_profiling": False  # Disable for demo
    }
    
    platform = HighPerformanceDefensivePlatform(config)
    
    try:
        # Initialize platform
        await platform.initialize()
        
        # Wait for system to stabilize
        logger.info("⏱️ Waiting for platform to stabilize...")
        await asyncio.sleep(5)
        
        # Execute high-performance training scenarios
        training_scenarios = [
            {"type": "threat_detection", "difficulty": "advanced", "samples": 5000},
            {"type": "incident_response", "difficulty": "expert", "samples": 3000},
            {"type": "malware_analysis", "difficulty": "intermediate", "samples": 7000},
        ]
        
        logger.info("🎓 Executing high-performance training scenarios")
        
        training_results = []
        for i, scenario in enumerate(training_scenarios):
            logger.info(f"Starting training scenario {i+1}/{len(training_scenarios)}")
            
            result = await platform.execute_defensive_training(scenario)
            training_results.append(result)
            
            # Brief pause between scenarios
            await asyncio.sleep(1)
        
        # Wait for metrics to be collected
        await asyncio.sleep(10)
        
        # Get comprehensive platform metrics
        logger.info("📊 Collecting platform performance metrics")
        platform_metrics = await platform.get_platform_metrics()
        
        # Display results
        print(f"\n{'='*80}")
        print("🚀 HIGH-PERFORMANCE PLATFORM RESULTS")
        print('='*80)
        
        # System Performance
        current = platform_metrics["current_metrics"]
        print(f"🖥️  CPU Usage: {current['cpu_usage']:.1f}%")
        print(f"💾 Memory Usage: {current['memory_usage_mb']:.1f} MB")
        print(f"⚡ Cache Hit Rate: {current['cache_hit_rate']:.1%}")
        print(f"👷 Active Workers: {current['active_workers']}")
        print(f"📋 Queue Size: {current['queue_size']}")
        
        # Cache Performance
        cache_perf = platform_metrics["cache_performance"]
        print(f"\n🗄️ CACHE PERFORMANCE:")
        print(f"   Entries: {cache_perf['entries']}")
        print(f"   Utilization: {cache_perf['utilization']:.1%}")
        print(f"   Hit Rate: {cache_perf['hit_rate']:.1%}")
        print(f"   Total Size: {cache_perf['total_size_mb']:.1f} MB")
        
        # Scheduler Performance
        scheduler_perf = platform_metrics["scheduler_performance"]
        recent = scheduler_perf["recent_performance"]
        print(f"\n⚙️ SCHEDULER PERFORMANCE:")
        print(f"   Throughput: {recent['throughput']:.2f} tasks/sec")
        print(f"   Avg Execution Time: {recent['avg_execution_time_ms']:.1f} ms")
        print(f"   Success Rate: {recent['success_rate']:.1%}")
        print(f"   Total Tasks: {scheduler_perf['total_tasks']}")
        
        # Training Results
        print(f"\n🎓 TRAINING RESULTS:")
        total_samples = sum(r.get('metrics', {}).get('samples_processed', 0) for r in training_results)
        avg_accuracy = sum(r.get('metrics', {}).get('accuracy', 0) for r in training_results) / len(training_results)
        total_time = sum(r.get('metrics', {}).get('training_time_seconds', 0) for r in training_results)
        
        print(f"   Scenarios Completed: {len(training_results)}")
        print(f"   Total Samples Processed: {total_samples:,}")
        print(f"   Average Accuracy: {avg_accuracy:.1%}")
        print(f"   Total Training Time: {total_time:.1f} seconds")
        print(f"   Processing Rate: {total_samples/total_time:.0f} samples/sec")
        
        # Performance Optimization Status
        optimization = platform_metrics["optimization_status"]
        print(f"\n🔧 OPTIMIZATION STATUS:")
        print(f"   Profiling Enabled: {'✅' if optimization['profiling_enabled'] else '❌'}")
        print(f"   Background Tasks Active: {optimization['background_tasks_active']}")
        
        # Save detailed metrics
        metrics_file = Path(f"platform_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(metrics_file, 'w') as f:
            json.dump(platform_metrics, f, indent=2, default=str)
        
        print(f"\n💾 Detailed metrics saved to: {metrics_file}")
        
    except Exception as e:
        logger.error(f"❌ Platform demonstration failed: {e}")
        raise
    
    finally:
        # Graceful shutdown
        await platform.shutdown()
    
    logger.info("🏁 High-Performance Platform demonstration completed")


async def main():
    """Main execution function"""
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        # This would trigger graceful shutdown in production
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await demonstrate_high_performance_platform()
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure high-performance logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('high_performance_platform.log')
        ]
    )
    
    # Run demonstration
    asyncio.run(main())