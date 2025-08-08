"""
Advanced caching system for GAN-Cyber-Range-v2.

This module provides multi-level caching, cache warming, intelligent eviction
policies, and distributed caching capabilities for optimal performance.
"""

import hashlib
import pickle
import json
import time
import threading
from typing import Any, Dict, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from abc import ABC, abstractmethod
import logging
import redis
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[timedelta] = None
    size: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return datetime.now() > self.created_at + self.ttl
    
    @property
    def age(self) -> timedelta:
        """Get age of cache entry"""
        return datetime.now() - self.created_at


class CacheBackend(ABC):
    """Abstract cache backend interface"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired:
                    self._remove_entry(key)
                    self.misses += 1
                    return None
                
                # Update access info
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                self.hits += 1
                return entry.value
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            try:
                # Calculate size
                serialized = pickle.dumps(value)
                size = len(serialized)
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    ttl=timedelta(seconds=ttl) if ttl else None,
                    size=size
                )
                
                # Remove existing entry if present
                if key in self.cache:
                    self._remove_entry(key)
                
                # Check if we need to evict entries
                while (len(self.cache) >= self.max_size or 
                       self.current_memory + size > self.max_memory_bytes):
                    if not self._evict_lru():
                        # Can't evict anymore, reject the entry
                        logger.warning(f"Cache full, cannot store key: {key}")
                        return False
                
                # Add new entry
                self.cache[key] = entry
                self.current_memory += size
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache key {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0
            return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired:
                    self._remove_entry(key)
                    return False
                return True
            return False
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        with self.lock:
            import fnmatch
            return [key for key in self.cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'entries': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'evictions': self.evictions
            }
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update memory usage"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_memory -= entry.size
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry"""
        if not self.cache:
            return False
        
        # Get least recently used key (first in OrderedDict)
        lru_key = next(iter(self.cache))
        self._remove_entry(lru_key)
        self.evictions += 1
        
        return True


class RedisCache(CacheBackend):
    """Redis-based distributed cache"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Fallback to memory cache
            self.redis_client = None
            self.fallback = MemoryCache()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return pickle.loads(data)
            else:
                return self.fallback.get(key)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            if self.redis_client:
                data = pickle.dumps(value)
                return self.redis_client.set(key, data, ex=ttl)
            else:
                return self.fallback.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                return self.fallback.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            if self.redis_client:
                return self.redis_client.flushdb()
            else:
                return self.fallback.clear()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            if self.redis_client:
                return bool(self.redis_client.exists(key))
            else:
                return self.fallback.exists(key)
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        try:
            if self.redis_client:
                return [key.decode() for key in self.redis_client.keys(pattern)]
            else:
                return self.fallback.keys(pattern)
        except Exception as e:
            logger.error(f"Cache keys error: {e}")
            return []


class TieredCache:
    """Multi-level cache with L1 (memory) and L2 (Redis) tiers"""
    
    def __init__(
        self,
        l1_max_size: int = 500,
        l1_max_memory_mb: int = 50,
        redis_host: str = "localhost",
        redis_port: int = 6379
    ):
        self.l1_cache = MemoryCache(l1_max_size, l1_max_memory_mb)
        self.l2_cache = RedisCache(redis_host, redis_port)
        
        # Cache promotion/demotion policies
        self.l1_hit_threshold = 3  # Promote to L1 after this many hits
        self.access_counters: Dict[str, int] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from tiered cache"""
        
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Track access for potential promotion
            self.access_counters[key] = self.access_counters.get(key, 0) + 1
            
            # Promote to L1 if accessed frequently
            if self.access_counters[key] >= self.l1_hit_threshold:
                self.l1_cache.set(key, value)
                self.access_counters.pop(key, None)
            
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, tier: str = "auto") -> bool:
        """Set value in tiered cache"""
        
        if tier == "l1" or tier == "auto":
            # Always try to cache in L1 for fast access
            l1_success = self.l1_cache.set(key, value, ttl)
        
        if tier == "l2" or tier == "auto":
            # Cache in L2 for persistence
            l2_success = self.l2_cache.set(key, value, ttl)
        
        # Reset access counter
        self.access_counters.pop(key, None)
        
        return True  # Success if stored in any tier
    
    def delete(self, key: str) -> bool:
        """Delete from all tiers"""
        l1_deleted = self.l1_cache.delete(key)
        l2_deleted = self.l2_cache.delete(key)
        self.access_counters.pop(key, None)
        
        return l1_deleted or l2_deleted
    
    def clear(self) -> bool:
        """Clear all tiers"""
        l1_cleared = self.l1_cache.clear()
        l2_cleared = self.l2_cache.clear()
        self.access_counters.clear()
        
        return l1_cleared and l2_cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all tiers"""
        return {
            'l1_cache': self.l1_cache.get_stats(),
            'l2_cache': 'redis_connected' if hasattr(self.l2_cache, 'redis_client') and self.l2_cache.redis_client else 'redis_unavailable',
            'promotion_candidates': len(self.access_counters)
        }


class CacheManager:
    """High-level cache manager with intelligent caching policies"""
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        self.backend = backend or TieredCache()
        self.cache_policies: Dict[str, Dict[str, Any]] = {}
        self.warming_tasks: Dict[str, threading.Thread] = {}
        
        # Default cache policies
        self._setup_default_policies()
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache with namespace"""
        full_key = self._make_key(key, namespace)
        return self.backend.get(full_key)
    
    def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[int] = None,
        **kwargs
    ) -> bool:
        """Set value in cache with intelligent policies"""
        
        full_key = self._make_key(key, namespace)
        
        # Apply cache policy if exists
        policy = self.cache_policies.get(namespace, {})
        
        # Use policy TTL if not specified
        if ttl is None:
            ttl = policy.get('default_ttl')
        
        # Check size limits
        max_size = policy.get('max_value_size')
        if max_size:
            try:
                value_size = len(pickle.dumps(value))
                if value_size > max_size:
                    logger.warning(f"Value too large for cache: {value_size} > {max_size}")
                    return False
            except Exception:
                pass
        
        return self.backend.set(full_key, value, ttl, **kwargs)
    
    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete from cache"""
        full_key = self._make_key(key, namespace)
        return self.backend.delete(full_key)
    
    def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace"""
        pattern = f"{namespace}:*"
        keys = self.backend.keys(pattern)
        
        deleted = 0
        for key in keys:
            if self.backend.delete(key):
                deleted += 1
        
        return deleted
    
    def warm_cache(self, namespace: str, data_loader: Callable[[], Dict[str, Any]]) -> None:
        """Warm cache with data from loader function"""
        
        def warming_task():
            try:
                logger.info(f"Starting cache warming for namespace: {namespace}")
                data = data_loader()
                
                for key, value in data.items():
                    self.set(key, value, namespace)
                
                logger.info(f"Cache warming completed for namespace: {namespace}")
                
            except Exception as e:
                logger.error(f"Cache warming failed for namespace {namespace}: {e}")
            finally:
                # Remove from active warming tasks
                self.warming_tasks.pop(namespace, None)
        
        # Don't start if already warming
        if namespace in self.warming_tasks:
            logger.info(f"Cache warming already in progress for namespace: {namespace}")
            return
        
        # Start warming in background
        thread = threading.Thread(target=warming_task, daemon=True)
        self.warming_tasks[namespace] = thread
        thread.start()
    
    def configure_policy(
        self,
        namespace: str,
        default_ttl: Optional[int] = None,
        max_value_size: Optional[int] = None,
        auto_warm: bool = False,
        warm_loader: Optional[Callable[[], Dict[str, Any]]] = None
    ) -> None:
        """Configure caching policy for a namespace"""
        
        policy = {
            'default_ttl': default_ttl,
            'max_value_size': max_value_size,
            'auto_warm': auto_warm,
            'warm_loader': warm_loader
        }
        
        self.cache_policies[namespace] = policy
        
        # Auto-warm if configured
        if auto_warm and warm_loader:
            self.warm_cache(namespace, warm_loader)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        
        info = {
            'backend_type': type(self.backend).__name__,
            'policies': list(self.cache_policies.keys()),
            'warming_tasks': list(self.warming_tasks.keys())
        }
        
        # Add backend-specific stats
        if hasattr(self.backend, 'get_stats'):
            info['stats'] = self.backend.get_stats()
        
        return info
    
    def _make_key(self, key: str, namespace: str) -> str:
        """Create namespaced cache key"""
        return f"{namespace}:{key}"
    
    def _setup_default_policies(self) -> None:
        """Setup default cache policies"""
        
        # GAN model cache - long TTL, large values allowed
        self.configure_policy(
            'gan_models',
            default_ttl=3600 * 24,  # 24 hours
            max_value_size=100 * 1024 * 1024  # 100MB
        )
        
        # Attack data cache - medium TTL
        self.configure_policy(
            'attack_data',
            default_ttl=3600,  # 1 hour
            max_value_size=10 * 1024 * 1024  # 10MB
        )
        
        # Network topology cache - long TTL
        self.configure_policy(
            'network_topology',
            default_ttl=3600 * 6,  # 6 hours
            max_value_size=5 * 1024 * 1024  # 5MB
        )
        
        # Metrics cache - short TTL
        self.configure_policy(
            'metrics',
            default_ttl=300,  # 5 minutes
            max_value_size=1024 * 1024  # 1MB
        )


# Decorators for caching
def cached(
    namespace: str = "default",
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
    cache_manager: Optional[CacheManager] = None
):
    """Decorator for caching function results"""
    
    if cache_manager is None:
        cache_manager = CacheManager()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key, namespace)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, namespace, ttl)
            
            return result
        
        # Add cache management methods to function
        wrapper.cache_clear = lambda: cache_manager.clear_namespace(namespace)
        wrapper.cache_info = lambda: cache_manager.get_cache_info()
        
        return wrapper
    
    return decorator


def cache_on_attribute(
    attr_name: str,
    namespace: str = "default",
    ttl: Optional[int] = None,
    cache_manager: Optional[CacheManager] = None
):
    """Decorator for caching based on object attribute"""
    
    if cache_manager is None:
        cache_manager = CacheManager()
    
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Use attribute value as cache key
            attr_value = getattr(self, attr_name, None)
            if attr_value is None:
                # No caching if attribute is None
                return func(self, *args, **kwargs)
            
            cache_key = f"{func.__name__}:{attr_value}"
            
            # Try cache first
            cached_result = cache_manager.get(cache_key, namespace)
            if cached_result is not None:
                return cached_result
            
            # Execute and cache
            result = func(self, *args, **kwargs)
            cache_manager.set(cache_key, result, namespace, ttl)
            
            return result
        
        return wrapper
    
    return decorator


# Global cache manager instance
default_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the default cache manager"""
    return default_cache_manager


def configure_global_cache(backend: CacheBackend) -> None:
    """Configure the global cache backend"""
    global default_cache_manager
    default_cache_manager = CacheManager(backend)