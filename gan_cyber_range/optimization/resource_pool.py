"""
Advanced resource pooling and management for optimal resource utilization.
"""

import logging
import asyncio
import threading
import queue
import time
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')

if TYPE_CHECKING:
    from typing import TYPE_CHECKING


class ResourceState(Enum):
    """Resource state in the pool"""
    AVAILABLE = "available"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    total_requests: int = 0
    active_connections: int = 0
    failed_requests: int = 0
    avg_wait_time: float = 0.0
    avg_usage_time: float = 0.0
    peak_usage: int = 0
    resource_efficiency: float = 0.0


@dataclass
class PooledResource:
    """Wrapper for pooled resources"""
    resource_id: str
    resource: Any
    created_at: datetime
    last_used: datetime
    usage_count: int = 0
    state: ResourceState = ResourceState.AVAILABLE
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_score: float = 1.0
    

class ResourcePool(Generic[T]):
    """Generic resource pool with intelligent management"""
    
    def __init__(self,
                 factory: Callable[[], T],
                 min_size: int = 5,
                 max_size: int = 20,
                 max_idle_time: int = 300,  # seconds
                 health_check: Optional[Callable[[T], bool]] = None,
                 cleanup: Optional[Callable[[T], None]] = None):
        
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check = health_check
        self.cleanup = cleanup
        
        self._pool: Dict[str, 'PooledResource[T]'] = {}
        self._available = queue.Queue()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        self.metrics = ResourceMetrics()
        self._running = True
        self._maintenance_thread = None
        
        # Initialize pool
        self._initialize_pool()
        self._start_maintenance()
        
    def _initialize_pool(self):
        """Initialize pool with minimum resources"""
        for i in range(self.min_size):
            resource = self._create_resource()
            if resource:
                self._pool[resource.resource_id] = resource
                self._available.put(resource.resource_id)
                
        logger.info(f"Initialized resource pool with {len(self._pool)} resources")
        
    def _create_resource(self) -> Optional['PooledResource[T]']:
        """Create a new resource"""
        try:
            resource = self.factory()
            resource_id = f"resource_{id(resource)}_{len(self._pool)}"
            
            pooled_resource = PooledResource(
                resource_id=resource_id,
                resource=resource,
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            
            return pooled_resource
            
        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
            return None
            
    @contextmanager
    def acquire(self, timeout: float = 30.0):
        """Acquire a resource from the pool"""
        start_time = time.perf_counter()
        resource_id = None
        
        try:
            with self._condition:
                # Wait for available resource
                while self._running:
                    try:
                        resource_id = self._available.get_nowait()
                        break
                    except queue.Empty:
                        # No available resource, try to create new one
                        if len(self._pool) < self.max_size:
                            new_resource = self._create_resource()
                            if new_resource:
                                self._pool[new_resource.resource_id] = new_resource
                                resource_id = new_resource.resource_id
                                break
                                
                        # Wait for resource to become available
                        if not self._condition.wait(timeout=min(1.0, timeout)):
                            # Timeout waiting
                            break
                            
                        timeout -= 1.0
                        if timeout <= 0:
                            break
                            
                if not resource_id:
                    raise TimeoutError("Timeout waiting for resource")
                    
                # Mark resource as in use
                pooled_resource = self._pool[resource_id]
                pooled_resource.state = ResourceState.IN_USE
                pooled_resource.usage_count += 1
                pooled_resource.last_used = datetime.now()
                
                # Update metrics
                wait_time = time.perf_counter() - start_time
                self.metrics.total_requests += 1
                self.metrics.active_connections += 1
                self.metrics.peak_usage = max(self.metrics.peak_usage, self.metrics.active_connections)
                
                if self.metrics.total_requests > 1:
                    self.metrics.avg_wait_time = (
                        (self.metrics.avg_wait_time * (self.metrics.total_requests - 1) + wait_time)
                        / self.metrics.total_requests
                    )
                else:
                    self.metrics.avg_wait_time = wait_time
                    
            usage_start_time = time.perf_counter()
            
            yield pooled_resource.resource
            
        except Exception as e:
            self.metrics.failed_requests += 1
            logger.error(f"Error using resource {resource_id}: {e}")
            
            # Mark resource as potentially failed
            if resource_id and resource_id in self._pool:
                self._pool[resource_id].health_score *= 0.8  # Reduce health score
                
            raise
            
        finally:
            # Return resource to pool
            if resource_id and resource_id in self._pool:
                with self._condition:
                    pooled_resource = self._pool[resource_id]
                    
                    # Update usage metrics
                    usage_time = time.perf_counter() - usage_start_time
                    if self.metrics.total_requests > 1:
                        self.metrics.avg_usage_time = (
                            (self.metrics.avg_usage_time * (self.metrics.total_requests - 1) + usage_time)
                            / self.metrics.total_requests
                        )
                    else:
                        self.metrics.avg_usage_time = usage_time
                        
                    # Health check before returning to pool
                    if self._perform_health_check(pooled_resource):
                        pooled_resource.state = ResourceState.AVAILABLE
                        self._available.put(resource_id)
                    else:
                        pooled_resource.state = ResourceState.FAILED
                        self._remove_resource(resource_id)
                        
                    self.metrics.active_connections -= 1
                    self._condition.notify()
                    
    def _perform_health_check(self, pooled_resource: 'PooledResource[T]') -> bool:
        """Perform health check on resource"""
        if not self.health_check:
            return True
            
        try:
            return self.health_check(pooled_resource.resource)
        except Exception as e:
            logger.warning(f"Health check failed for resource {pooled_resource.resource_id}: {e}")
            return False
            
    def _remove_resource(self, resource_id: str):
        """Remove resource from pool"""
        if resource_id in self._pool:
            pooled_resource = self._pool[resource_id]
            
            # Cleanup resource
            if self.cleanup:
                try:
                    self.cleanup(pooled_resource.resource)
                except Exception as e:
                    logger.error(f"Error cleaning up resource {resource_id}: {e}")
                    
            del self._pool[resource_id]
            logger.debug(f"Removed resource {resource_id} from pool")
            
    def _start_maintenance(self):
        """Start background maintenance thread"""
        self._maintenance_thread = threading.Thread(target=self._maintenance_loop)
        self._maintenance_thread.daemon = True
        self._maintenance_thread.start()
        
    def _maintenance_loop(self):
        """Background maintenance loop"""
        while self._running:
            try:
                self._perform_maintenance()
                time.sleep(30)  # Run maintenance every 30 seconds
            except Exception as e:
                logger.error(f"Error in resource pool maintenance: {e}")
                time.sleep(5)
                
    def _perform_maintenance(self):
        """Perform pool maintenance"""
        current_time = datetime.now()
        resources_to_remove = []
        
        with self._lock:
            # Check for idle resources
            for resource_id, pooled_resource in self._pool.items():
                if pooled_resource.state == ResourceState.AVAILABLE:
                    idle_time = (current_time - pooled_resource.last_used).total_seconds()
                    
                    # Remove idle resources (but maintain minimum)
                    if (idle_time > self.max_idle_time and 
                        len(self._pool) > self.min_size):
                        resources_to_remove.append(resource_id)
                        
                    # Remove unhealthy resources
                    elif pooled_resource.health_score < 0.5:
                        resources_to_remove.append(resource_id)
                        
            # Remove identified resources
            for resource_id in resources_to_remove:
                try:
                    # Remove from available queue
                    temp_queue = queue.Queue()
                    while True:
                        try:
                            item = self._available.get_nowait()
                            if item != resource_id:
                                temp_queue.put(item)
                        except queue.Empty:
                            break
                            
                    # Put back non-removed items
                    while True:
                        try:
                            self._available.put(temp_queue.get_nowait())
                        except queue.Empty:
                            break
                            
                    self._remove_resource(resource_id)
                except Exception as e:
                    logger.error(f"Error removing resource {resource_id}: {e}")
                    
            # Ensure minimum pool size
            while len(self._pool) < self.min_size:
                new_resource = self._create_resource()
                if new_resource:
                    self._pool[new_resource.resource_id] = new_resource
                    self._available.put(new_resource.resource_id)
                else:
                    break
                    
            # Update efficiency metric
            if self.metrics.peak_usage > 0:
                self.metrics.resource_efficiency = len(self._pool) / self.metrics.peak_usage
                
    def get_stats(self) -> ResourceMetrics:
        """Get pool statistics"""
        with self._lock:
            stats = ResourceMetrics(
                total_requests=self.metrics.total_requests,
                active_connections=self.metrics.active_connections,
                failed_requests=self.metrics.failed_requests,
                avg_wait_time=self.metrics.avg_wait_time,
                avg_usage_time=self.metrics.avg_usage_time,
                peak_usage=self.metrics.peak_usage,
                resource_efficiency=self.metrics.resource_efficiency
            )
            return stats
            
    def shutdown(self):
        """Shutdown the resource pool"""
        self._running = False
        
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5)
            
        # Cleanup all resources
        with self._lock:
            for resource_id in list(self._pool.keys()):
                self._remove_resource(resource_id)
                
        logger.info("Resource pool shutdown complete")


class ResourceManager:
    """Centralized resource manager for multiple pools"""
    
    def __init__(self):
        self._pools: Dict[str, ResourcePool] = {}
        self._global_stats = {
            "total_pools": 0,
            "total_resources": 0,
            "total_requests": 0,
            "global_efficiency": 0.0
        }
        
    def create_pool(self,
                   name: str,
                   factory: Callable,
                   **pool_config) -> ResourcePool:
        """Create a new resource pool"""
        
        if name in self._pools:
            raise ValueError(f"Pool '{name}' already exists")
            
        pool = ResourcePool(factory=factory, **pool_config)
        self._pools[name] = pool
        
        self._global_stats["total_pools"] += 1
        
        logger.info(f"Created resource pool '{name}'")
        return pool
        
    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get resource pool by name"""
        return self._pools.get(name)
        
    def remove_pool(self, name: str) -> bool:
        """Remove and shutdown resource pool"""
        if name in self._pools:
            pool = self._pools[name]
            pool.shutdown()
            del self._pools[name]
            
            self._global_stats["total_pools"] -= 1
            
            logger.info(f"Removed resource pool '{name}'")
            return True
            
        return False
        
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global resource statistics"""
        
        total_resources = 0
        total_requests = 0
        total_efficiency = 0.0
        pool_details = {}
        
        for name, pool in self._pools.items():
            stats = pool.get_stats()
            pool_size = len(pool._pool)
            
            total_resources += pool_size
            total_requests += stats.total_requests
            total_efficiency += stats.resource_efficiency
            
            pool_details[name] = {
                "pool_size": pool_size,
                "active_connections": stats.active_connections,
                "total_requests": stats.total_requests,
                "failed_requests": stats.failed_requests,
                "avg_wait_time": stats.avg_wait_time,
                "resource_efficiency": stats.resource_efficiency
            }
            
        avg_efficiency = total_efficiency / len(self._pools) if self._pools else 0.0
        
        return {
            "total_pools": len(self._pools),
            "total_resources": total_resources,
            "total_requests": total_requests,
            "global_efficiency": avg_efficiency,
            "pool_details": pool_details
        }
        
    def optimize_all_pools(self):
        """Trigger optimization for all pools"""
        for name, pool in self._pools.items():
            try:
                pool._perform_maintenance()
                logger.debug(f"Optimized pool '{name}'")
            except Exception as e:
                logger.error(f"Error optimizing pool '{name}': {e}")
                
    def shutdown_all(self):
        """Shutdown all resource pools"""
        for name in list(self._pools.keys()):
            self.remove_pool(name)
            
        logger.info("All resource pools shutdown")


class AsyncResourcePool:
    """Asynchronous resource pool implementation"""
    
    def __init__(self,
                 factory: Callable,
                 min_size: int = 5,
                 max_size: int = 20,
                 max_idle_time: int = 300):
        
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        
        self._pool = asyncio.Queue(maxsize=max_size)
        self._created_resources = 0
        self._lock = asyncio.Lock()
        self._initialized = False
        
    async def initialize(self):
        """Initialize the async resource pool"""
        if self._initialized:
            return
            
        async with self._lock:
            for _ in range(self.min_size):
                resource = await self._create_resource()
                if resource:
                    await self._pool.put(resource)
                    self._created_resources += 1
                    
            self._initialized = True
            
        logger.info(f"Initialized async resource pool with {self._created_resources} resources")
        
    async def _create_resource(self):
        """Create a new async resource"""
        try:
            if asyncio.iscoroutinefunction(self.factory):
                return await self.factory()
            else:
                return self.factory()
        except Exception as e:
            logger.error(f"Failed to create async resource: {e}")
            return None
            
    @contextmanager
    async def acquire(self, timeout: float = 30.0):
        """Acquire resource from async pool"""
        if not self._initialized:
            await self.initialize()
            
        resource = None
        try:
            # Try to get existing resource
            try:
                resource = await asyncio.wait_for(self._pool.get(), timeout=timeout)
            except asyncio.TimeoutError:
                # Try to create new resource if under limit
                if self._created_resources < self.max_size:
                    async with self._lock:
                        if self._created_resources < self.max_size:
                            resource = await self._create_resource()
                            if resource:
                                self._created_resources += 1
                                
            if not resource:
                raise TimeoutError("Could not acquire resource")
                
            yield resource
            
        finally:
            if resource:
                # Return resource to pool
                try:
                    self._pool.put_nowait(resource)
                except asyncio.QueueFull:
                    # Pool is full, cleanup resource
                    if hasattr(resource, 'close'):
                        try:
                            if asyncio.iscoroutinefunction(resource.close):
                                await resource.close()
                            else:
                                resource.close()
                        except Exception as e:
                            logger.error(f"Error closing resource: {e}")
                            
                    self._created_resources -= 1