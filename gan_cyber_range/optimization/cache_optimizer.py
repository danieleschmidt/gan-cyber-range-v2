"""
Advanced cache optimization with intelligent strategies and adaptive algorithms.
"""

import logging
import asyncio
import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import weakref
from collections import OrderedDict, defaultdict
import threading

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache replacement strategies"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    ADAPTIVE = "adaptive"
    INTELLIGENT = "intelligent"
        
    @classmethod
    def create(cls, strategy_name: str, **kwargs):
        """Create cache strategy with parameters"""
        if strategy_name.upper() in [s.name for s in cls]:
            return cls[strategy_name.upper()]
        return cls.LRU


class CacheLevel(Enum):
    """Cache hierarchy levels"""
    L1_MEMORY = "l1_memory"
    L2_DISK = "l2_disk"
    L3_DISTRIBUTED = "l3_distributed"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    size: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    cost_to_compute: float = 0.0
    priority: int = 1
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    avg_access_time: float = 0.0
    memory_pressure: float = 0.0


class IntelligentCache:
    """Intelligent cache with adaptive algorithms"""
    
    def __init__(self,
                 max_size: int = 1000,
                 max_memory_mb: int = 100,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 ttl_seconds: Optional[int] = None):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.strategy = strategy
        self.default_ttl = ttl_seconds
        
        self._entries: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # For LRU
        self._frequency_counter = defaultdict(int)  # For LFU
        self._size_tracker = 0
        self._lock = threading.RLock()
        
        # Adaptive strategy parameters
        self._strategy_performance = {
            CacheStrategy.LRU: {"hits": 0, "misses": 0},
            CacheStrategy.LFU: {"hits": 0, "misses": 0},
            CacheStrategy.FIFO: {"hits": 0, "misses": 0}
        }
        self._current_adaptive_strategy = CacheStrategy.LRU
        self._strategy_switch_threshold = 100
        self._access_pattern_history = []
        
        # Performance monitoring
        self.stats = CacheStats()
        self._access_times = []
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent access tracking"""
        start_time = time.perf_counter()
        
        with self._lock:
            if key not in self._entries:
                self.stats.misses += 1
                self._update_strategy_performance(False)
                return None
                
            entry = self._entries[key]
            
            # Check TTL expiration
            if self._is_expired(entry):
                self._remove_entry(key)
                self.stats.misses += 1
                self._update_strategy_performance(False)
                return None
                
            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self._frequency_counter[key] += 1
            
            # Update access order for LRU
            if key in self._access_order:
                del self._access_order[key]
            self._access_order[key] = True
            
            # Track access patterns for adaptive strategy
            self._track_access_pattern(key)
            
            self.stats.hits += 1
            self._update_strategy_performance(True)
            
            # Update access time statistics
            access_time = time.perf_counter() - start_time
            self._access_times.append(access_time)
            if len(self._access_times) > 1000:
                self._access_times = self._access_times[-500:]  # Keep last 500
                
            return entry.value
            
    def put(self, 
            key: str, 
            value: Any, 
            ttl_seconds: Optional[int] = None,
            priority: int = 1,
            tags: Optional[List[str]] = None,
            cost_to_compute: float = 0.0) -> bool:
        """Put value in cache with intelligent eviction"""
        
        with self._lock:
            # Calculate size
            try:
                size = self._calculate_size(value)
            except Exception:
                size = 1024  # Default size estimate
                
            # Check if we need to evict
            if key not in self._entries:
                while (len(self._entries) >= self.max_size or 
                       self._size_tracker + size > self.max_memory_bytes):
                    if not self._evict_entry():
                        return False  # Cannot evict anything
                        
            # Create or update entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                priority=priority,
                tags=tags or [],
                cost_to_compute=cost_to_compute,
                metadata={
                    "ttl_seconds": ttl_seconds or self.default_ttl
                }
            )
            
            # Update size tracking
            if key in self._entries:
                self._size_tracker -= self._entries[key].size
            
            self._entries[key] = entry
            self._size_tracker += size
            self._access_order[key] = True
            
            return True
            
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry"""
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                return True
            return False
            
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all entries with specified tags"""
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._entries.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                self._remove_entry(key)
                
            return len(keys_to_remove)
            
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._entries.clear()
            self._access_order.clear()
            self._frequency_counter.clear()
            self._size_tracker = 0
            
    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        with self._lock:
            total_accesses = self.stats.hits + self.stats.misses
            
            if total_accesses > 0:
                self.stats.hit_rate = self.stats.hits / total_accesses
                self.stats.miss_rate = self.stats.misses / total_accesses
            else:
                self.stats.hit_rate = 0.0
                self.stats.miss_rate = 0.0
                
            if self._access_times:
                self.stats.avg_access_time = sum(self._access_times) / len(self._access_times)
            else:
                self.stats.avg_access_time = 0.0
                
            self.stats.size_bytes = self._size_tracker
            self.stats.entry_count = len(self._entries)
            self.stats.memory_pressure = self._size_tracker / self.max_memory_bytes
            
            return self.stats
            
    def optimize(self):
        """Perform cache optimization"""
        with self._lock:
            # Adaptive strategy optimization
            if self.strategy == CacheStrategy.ADAPTIVE:
                self._optimize_adaptive_strategy()
                
            # Memory optimization
            self._optimize_memory_usage()
            
            # Remove expired entries
            self._cleanup_expired_entries()
            
    def _evict_entry(self) -> bool:
        """Evict entry based on current strategy"""
        if not self._entries:
            return False
            
        if self.strategy == CacheStrategy.ADAPTIVE:
            strategy = self._current_adaptive_strategy
        else:
            strategy = self.strategy
            
        if strategy == CacheStrategy.LRU:
            return self._evict_lru()
        elif strategy == CacheStrategy.LFU:
            return self._evict_lfu()
        elif strategy == CacheStrategy.FIFO:
            return self._evict_fifo()
        elif strategy == CacheStrategy.INTELLIGENT:
            return self._evict_intelligent()
        else:
            return self._evict_lru()  # Default fallback
            
    def _evict_lru(self) -> bool:
        """Evict least recently used entry"""
        if not self._access_order:
            return False
            
        key_to_evict = next(iter(self._access_order))
        self._remove_entry(key_to_evict)
        return True
        
    def _evict_lfu(self) -> bool:
        """Evict least frequently used entry"""
        if not self._entries:
            return False
            
        # Find entry with lowest access count
        min_access_count = float('inf')
        key_to_evict = None
        
        for key, entry in self._entries.items():
            if entry.access_count < min_access_count:
                min_access_count = entry.access_count
                key_to_evict = key
                
        if key_to_evict:
            self._remove_entry(key_to_evict)
            return True
            
        return False
        
    def _evict_fifo(self) -> bool:
        """Evict first in, first out"""
        if not self._entries:
            return False
            
        # Find oldest entry
        oldest_time = datetime.max
        key_to_evict = None
        
        for key, entry in self._entries.items():
            if entry.created_at < oldest_time:
                oldest_time = entry.created_at
                key_to_evict = key
                
        if key_to_evict:
            self._remove_entry(key_to_evict)
            return True
            
        return False
        
    def _evict_intelligent(self) -> bool:
        """Intelligent eviction based on multiple factors"""
        if not self._entries:
            return False
            
        # Score entries based on multiple factors
        scores = {}
        current_time = datetime.now()
        
        for key, entry in self._entries.items():
            age_seconds = (current_time - entry.last_accessed).total_seconds()
            
            # Factors: age, frequency, size, priority, cost_to_compute
            age_score = age_seconds / 3600  # Normalize to hours
            frequency_score = 1.0 / (entry.access_count + 1)
            size_score = entry.size / (1024 * 1024)  # Normalize to MB
            priority_score = 1.0 / entry.priority
            cost_score = 1.0 / (entry.cost_to_compute + 1)
            
            # Combined score (higher = more likely to evict)
            total_score = (age_score * 0.3 + 
                          frequency_score * 0.3 + 
                          size_score * 0.2 + 
                          priority_score * 0.1 + 
                          cost_score * 0.1)
            
            scores[key] = total_score
            
        # Evict entry with highest score
        key_to_evict = max(scores, key=scores.get)
        self._remove_entry(key_to_evict)
        return True
        
    def _remove_entry(self, key: str):
        """Remove entry and update tracking"""
        if key in self._entries:
            entry = self._entries[key]
            self._size_tracker -= entry.size
            del self._entries[key]
            self.stats.evictions += 1
            
        if key in self._access_order:
            del self._access_order[key]
            
        if key in self._frequency_counter:
            del self._frequency_counter[key]
            
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired"""
        ttl = entry.metadata.get("ttl_seconds")
        if ttl is None:
            return False
            
        age_seconds = (datetime.now() - entry.created_at).total_seconds()
        return age_seconds > ttl
        
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of cached value"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (dict, list, tuple)):
                return len(str(value))  # Approximation
            else:
                return 1024  # Default estimate
        except Exception:
            return 1024  # Fallback
            
    def _update_strategy_performance(self, hit: bool):
        """Update strategy performance for adaptive optimization"""
        if self.strategy == CacheStrategy.ADAPTIVE:
            current_strategy = self._current_adaptive_strategy
            if hit:
                self._strategy_performance[current_strategy]["hits"] += 1
            else:
                self._strategy_performance[current_strategy]["misses"] += 1
                
    def _track_access_pattern(self, key: str):
        """Track access patterns for intelligent optimization"""
        self._access_pattern_history.append({
            "key": key,
            "timestamp": time.time(),
            "strategy": self._current_adaptive_strategy
        })
        
        # Keep only recent history
        if len(self._access_pattern_history) > 10000:
            self._access_pattern_history = self._access_pattern_history[-5000:]
            
    def _optimize_adaptive_strategy(self):
        """Optimize adaptive strategy based on performance"""
        total_accesses = sum(
            perf["hits"] + perf["misses"] 
            for perf in self._strategy_performance.values()
        )
        
        if total_accesses < self._strategy_switch_threshold:
            return
            
        # Calculate hit rates for each strategy
        hit_rates = {}
        for strategy, perf in self._strategy_performance.items():
            total = perf["hits"] + perf["misses"]
            if total > 0:
                hit_rates[strategy] = perf["hits"] / total
            else:
                hit_rates[strategy] = 0.0
                
        # Switch to best performing strategy
        best_strategy = max(hit_rates, key=hit_rates.get)
        if best_strategy != self._current_adaptive_strategy:
            logger.info(f"Switching adaptive cache strategy from {self._current_adaptive_strategy} to {best_strategy}")
            self._current_adaptive_strategy = best_strategy
            
        # Reset counters for fresh evaluation
        for perf in self._strategy_performance.values():
            perf["hits"] = 0
            perf["misses"] = 0
            
    def _optimize_memory_usage(self):
        """Optimize memory usage by preemptive eviction"""
        if self.stats.memory_pressure > 0.8:  # 80% memory usage
            # Preemptively evict low-value entries
            entries_to_evict = max(1, len(self._entries) // 10)  # Evict 10%
            
            for _ in range(entries_to_evict):
                if not self._evict_entry():
                    break
                    
    def _cleanup_expired_entries(self):
        """Remove expired entries"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, entry in self._entries.items():
            if self._is_expired(entry):
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            self._remove_entry(key)


class CacheOptimizer:
    """Advanced cache optimization manager"""
    
    def __init__(self):
        self.caches: Dict[str, IntelligentCache] = {}
        self._optimization_thread = None
        self._optimization_interval = 60  # seconds
        self._running = False
        
        # Global cache statistics
        self.global_stats = {
            "total_hits": 0,
            "total_misses": 0,
            "total_size": 0,
            "cache_count": 0
        }
        
    def create_cache(self,
                    name: str,
                    max_size: int = 1000,
                    max_memory_mb: int = 100,
                    strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                    ttl_seconds: Optional[int] = None) -> IntelligentCache:
        """Create a new intelligent cache"""
        
        cache = IntelligentCache(
            max_size=max_size,
            max_memory_mb=max_memory_mb,
            strategy=strategy,
            ttl_seconds=ttl_seconds
        )
        
        self.caches[name] = cache
        logger.info(f"Created intelligent cache '{name}' with strategy {strategy.value}")
        
        return cache
        
    def get_cache(self, name: str) -> Optional[IntelligentCache]:
        """Get cache by name"""
        return self.caches.get(name)
        
    def start_optimization(self):
        """Start background optimization"""
        if self._running:
            return
            
        self._running = True
        self._optimization_thread = threading.Thread(target=self._optimization_loop)
        self._optimization_thread.daemon = True
        self._optimization_thread.start()
        
        logger.info("Started cache optimization background thread")
        
    def stop_optimization(self):
        """Stop background optimization"""
        self._running = False
        if self._optimization_thread:
            self._optimization_thread.join()
            
        logger.info("Stopped cache optimization")
        
    def optimize_all_caches(self):
        """Manually trigger optimization for all caches"""
        for name, cache in self.caches.items():
            try:
                cache.optimize()
                logger.debug(f"Optimized cache '{name}'")
            except Exception as e:
                logger.error(f"Error optimizing cache '{name}': {e}")
                
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global cache statistics"""
        total_hits = 0
        total_misses = 0
        total_size = 0
        cache_details = {}
        
        for name, cache in self.caches.items():
            stats = cache.get_stats()
            total_hits += stats.hits
            total_misses += stats.misses
            total_size += stats.size_bytes
            
            cache_details[name] = {
                "hit_rate": stats.hit_rate,
                "entries": stats.entry_count,
                "size_mb": stats.size_bytes / (1024 * 1024),
                "memory_pressure": stats.memory_pressure
            }
            
        total_accesses = total_hits + total_misses
        global_hit_rate = total_hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            "global_hit_rate": global_hit_rate,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_count": len(self.caches),
            "cache_details": cache_details
        }
        
    def create_tiered_cache(self,
                          name: str,
                          l1_config: Dict[str, Any],
                          l2_config: Optional[Dict[str, Any]] = None) -> 'TieredCache':
        """Create a tiered cache system"""
        
        return TieredCache(
            name=name,
            l1_config=l1_config,
            l2_config=l2_config or {},
            optimizer=self
        )
        
    def _optimization_loop(self):
        """Background optimization loop"""
        while self._running:
            try:
                self.optimize_all_caches()
                time.sleep(self._optimization_interval)
            except Exception as e:
                logger.error(f"Error in cache optimization loop: {e}")
                time.sleep(5)  # Brief pause before retrying


class TieredCache:
    """Multi-level tiered cache system"""
    
    def __init__(self,
                 name: str,
                 l1_config: Dict[str, Any],
                 l2_config: Dict[str, Any],
                 optimizer: CacheOptimizer):
        self.name = name
        self.optimizer = optimizer
        
        # Create L1 cache (fast, small)
        self.l1_cache = IntelligentCache(
            max_size=l1_config.get("max_size", 100),
            max_memory_mb=l1_config.get("max_memory_mb", 10),
            strategy=CacheStrategy(l1_config.get("strategy", "lru")),
            ttl_seconds=l1_config.get("ttl_seconds")
        )
        
        # Create L2 cache (larger, slower)  
        self.l2_cache = IntelligentCache(
            max_size=l2_config.get("max_size", 1000),
            max_memory_mb=l2_config.get("max_memory_mb", 100),
            strategy=CacheStrategy(l2_config.get("strategy", "adaptive")),
            ttl_seconds=l2_config.get("ttl_seconds")
        )
        
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0,
            "promotions": 0,  # L2 -> L1
            "demotions": 0    # L1 -> L2
        }
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from tiered cache"""
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats["l1_hits"] += 1
            return value
            
        # Try L2 
        value = self.l2_cache.get(key)
        if value is not None:
            self.stats["l2_hits"] += 1
            
            # Promote to L1 if frequently accessed
            entry = self.l2_cache._entries.get(key)
            if entry and entry.access_count > 2:
                self.l1_cache.put(key, value, priority=2)  # Higher priority
                self.stats["promotions"] += 1
                
            return value
            
        self.stats["misses"] += 1
        return None
        
    def put(self, key: str, value: Any, **kwargs) -> bool:
        """Put value in tiered cache"""
        # Always put in L1 first
        success = self.l1_cache.put(key, value, **kwargs)
        
        if not success:
            # If L1 is full, try L2
            return self.l2_cache.put(key, value, **kwargs)
            
        return success
        
    def invalidate(self, key: str) -> bool:
        """Invalidate from both levels"""
        l1_result = self.l1_cache.invalidate(key)
        l2_result = self.l2_cache.invalidate(key)
        return l1_result or l2_result
        
    def get_stats(self) -> Dict[str, Any]:
        """Get tiered cache statistics"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        total_accesses = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["misses"]
        
        return {
            "l1_hit_rate": self.stats["l1_hits"] / total_accesses if total_accesses > 0 else 0.0,
            "l2_hit_rate": self.stats["l2_hits"] / total_accesses if total_accesses > 0 else 0.0,
            "overall_hit_rate": (self.stats["l1_hits"] + self.stats["l2_hits"]) / total_accesses if total_accesses > 0 else 0.0,
            "promotions": self.stats["promotions"],
            "demotions": self.stats["demotions"],
            "l1_stats": {
                "entries": l1_stats.entry_count,
                "size_mb": l1_stats.size_bytes / (1024 * 1024),
                "memory_pressure": l1_stats.memory_pressure
            },
            "l2_stats": {
                "entries": l2_stats.entry_count,
                "size_mb": l2_stats.size_bytes / (1024 * 1024),
                "memory_pressure": l2_stats.memory_pressure
            }
        }