"""
Performance optimization utilities for GAN-Cyber-Range-v2.

This module provides optimization techniques including model optimization,
computation acceleration, memory management, and parallel processing.
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
import threading
import multiprocessing
import concurrent.futures
import functools
import time
import psutil
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
import gc
import weakref

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization settings"""
    enable_gpu: bool = True
    enable_mixed_precision: bool = True
    enable_model_parallel: bool = False
    enable_data_parallel: bool = True
    max_workers: int = None
    memory_optimization: bool = True
    compilation_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    
    def __post_init__(self):
        if self.max_workers is None:
            self.max_workers = min(32, (psutil.cpu_count() or 1) + 4)


class DeviceManager:
    """Manages compute devices and memory"""
    
    def __init__(self):
        self.devices = self._detect_devices()
        self.current_device = self._select_best_device()
        self.memory_pools = {}
        
    def _detect_devices(self) -> Dict[str, Any]:
        """Detect available compute devices"""
        devices = {
            'cpu': {
                'available': True,
                'cores': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3)
            }
        }
        
        # Check for CUDA
        if torch.cuda.is_available():
            devices['cuda'] = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'devices': []
            }
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices['cuda']['devices'].append({
                    'id': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}"
                })
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices['mps'] = {
                'available': True,
                'memory_gb': 'unified'
            }
        
        return devices
    
    def _select_best_device(self) -> torch.device:
        """Select the best available device"""
        if 'cuda' in self.devices and self.devices['cuda']['available']:
            return torch.device('cuda')
        elif 'mps' in self.devices and self.devices['mps']['available']:
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def get_optimal_batch_size(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Determine optimal batch size for a model"""
        if self.current_device.type == 'cpu':
            return 32  # Conservative for CPU
        
        # Start with a reasonable batch size and test
        batch_size = 64
        max_batch_size = 1024
        
        while batch_size <= max_batch_size:
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape, device=self.current_device)
                
                # Test forward pass
                model.eval()
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # If successful, try larger batch size
                batch_size *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Return previous working batch size
                    return max(1, batch_size // 2)
                else:
                    raise
        
        return batch_size // 2
    
    def optimize_memory_usage(self) -> None:
        """Optimize memory usage"""
        if self.current_device.type == 'cuda':
            torch.cuda.empty_cache()
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Force garbage collection
        gc.collect()


class ModelOptimizer:
    """Optimizes neural network models for performance"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device_manager = DeviceManager()
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive model optimizations"""
        
        logger.info("Applying model optimizations...")
        
        # Move to optimal device
        model = model.to(self.device_manager.current_device)
        
        # Apply quantization if appropriate
        if self.config.enable_mixed_precision and self.device_manager.current_device.type == 'cuda':
            model = self._apply_mixed_precision(model)
        
        # Apply model parallelism if multiple GPUs
        if self.config.enable_data_parallel and torch.cuda.device_count() > 1:
            model = self._apply_data_parallel(model)
        
        # Apply model compilation (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = self._apply_compilation(model)
        
        # Optimize for inference if not training
        model = self._optimize_for_inference(model)
        
        logger.info("Model optimization completed")
        return model
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Apply automatic mixed precision"""
        logger.info("Applying mixed precision optimization")
        
        # Convert to half precision where appropriate
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                module.half()
        
        return model
    
    def _apply_data_parallel(self, model: nn.Module) -> nn.Module:
        """Apply data parallelism"""
        logger.info(f"Applying data parallel across {torch.cuda.device_count()} GPUs")
        return DataParallel(model)
    
    def _apply_compilation(self, model: nn.Module) -> nn.Module:
        """Apply torch.compile optimization"""
        try:
            logger.info(f"Compiling model with mode: {self.config.compilation_mode}")
            return torch.compile(model, mode=self.config.compilation_mode)
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
            return model
    
    def _optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference"""
        model.eval()
        
        # Fuse operations where possible
        if hasattr(torch.quantization, 'fuse_modules'):
            try:
                # Common fusion patterns
                for module in model.modules():
                    if hasattr(module, 'conv') and hasattr(module, 'bn'):
                        torch.quantization.fuse_modules(
                            module, ['conv', 'bn'], inplace=True
                        )
            except Exception as e:
                logger.warning(f"Module fusion failed: {e}")
        
        return model


class ComputationAccelerator:
    """Accelerates common computational tasks"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_workers
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(config.max_workers, psutil.cpu_count())
        )
    
    def parallelize_function(
        self,
        func: Callable,
        inputs: List[Any],
        use_processes: bool = False,
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Parallelize function execution over inputs"""
        
        if len(inputs) == 1:
            # No need to parallelize single input
            return [func(inputs[0])]
        
        if chunk_size is None:
            chunk_size = max(1, len(inputs) // self.config.max_workers)
        
        executor = self.process_pool if use_processes else self.thread_pool
        
        try:
            futures = []
            for i in range(0, len(inputs), chunk_size):
                chunk = inputs[i:i + chunk_size]
                if len(chunk) == 1:
                    future = executor.submit(func, chunk[0])
                else:
                    future = executor.submit(self._process_chunk, func, chunk)
                futures.append(future)
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            # Fallback to sequential execution
            return [func(inp) for inp in inputs]
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of inputs"""
        return [func(item) for item in chunk]
    
    def accelerate_numpy_ops(self) -> None:
        """Configure NumPy for optimal performance"""
        # Set optimal thread count for NumPy
        import os
        os.environ['OMP_NUM_THREADS'] = str(min(4, psutil.cpu_count()))
        os.environ['MKL_NUM_THREADS'] = str(min(4, psutil.cpu_count()))
        os.environ['NUMEXPR_NUM_THREADS'] = str(min(4, psutil.cpu_count()))
        
        logger.info("Configured NumPy threading for optimal performance")
    
    def optimize_pytorch_settings(self) -> None:
        """Configure PyTorch for optimal performance"""
        # Set number of threads
        torch.set_num_threads(min(4, psutil.cpu_count()))
        
        # Enable cudnn benchmark for consistent input sizes
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        
        # Enable JIT optimizations
        torch.jit.set_fusion_strategy([("STATIC", 2), ("DYNAMIC", 2)])
        
        logger.info("Configured PyTorch for optimal performance")


class MemoryOptimizer:
    """Optimizes memory usage and prevents memory leaks"""
    
    def __init__(self):
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.weak_refs = []
        
    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage"""
        memory = psutil.virtual_memory()
        
        stats = {
            'ram_percent': memory.percent,
            'ram_available_gb': memory.available / (1024**3),
            'ram_used_gb': memory.used / (1024**3)
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.memory_stats(i)
                stats[f'gpu_{i}_allocated_gb'] = gpu_memory.get('allocated_bytes.all.current', 0) / (1024**3)
                stats[f'gpu_{i}_reserved_gb'] = gpu_memory.get('reserved_bytes.all.current', 0) / (1024**3)
        
        return stats
    
    def cleanup_memory(self) -> None:
        """Perform memory cleanup"""
        # Python garbage collection
        collected = gc.collect()
        
        # PyTorch memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clean weak references
        self.weak_refs = [ref for ref in self.weak_refs if ref() is not None]
        
        logger.info(f"Memory cleanup completed. Collected {collected} objects.")
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        memory_stats = self.monitor_memory_usage()
        return memory_stats['ram_percent'] > (self.memory_threshold * 100)
    
    def optimize_tensor_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory usage"""
        # Convert to appropriate dtype if possible
        if tensor.dtype == torch.float64:
            tensor = tensor.float()  # Convert to float32
        
        # Make contiguous for better memory access
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        return tensor
    
    def create_memory_efficient_dataloader(
        self,
        dataset,
        batch_size: int,
        num_workers: Optional[int] = None
    ):
        """Create memory-efficient data loader"""
        if num_workers is None:
            num_workers = min(4, psutil.cpu_count())
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )


class CacheOptimizer:
    """Optimizes caching strategies for better performance"""
    
    def __init__(self):
        self.access_patterns = {}
        self.cache_hit_rates = {}
        
    def analyze_access_pattern(self, key: str) -> None:
        """Analyze access patterns for cache optimization"""
        current_time = time.time()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(current_time)
        
        # Keep only recent accesses (last hour)
        cutoff_time = current_time - 3600
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff_time
        ]
    
    def get_optimal_ttl(self, key: str) -> Optional[int]:
        """Get optimal TTL based on access patterns"""
        if key not in self.access_patterns:
            return None
        
        accesses = self.access_patterns[key]
        if len(accesses) < 2:
            return 3600  # Default 1 hour
        
        # Calculate average access interval
        intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
        avg_interval = sum(intervals) / len(intervals)
        
        # Set TTL to 2x average interval (with bounds)
        optimal_ttl = int(avg_interval * 2)
        return max(300, min(optimal_ttl, 86400))  # 5 minutes to 24 hours
    
    def should_cache(self, key: str, computation_cost: float) -> bool:
        """Determine if an item should be cached based on cost/benefit"""
        access_frequency = len(self.access_patterns.get(key, []))
        
        # Cache if accessed frequently and computation is expensive
        return access_frequency >= 2 and computation_cost > 0.1  # 100ms threshold


class ProfiledFunction:
    """Wrapper for profiling function performance"""
    
    def __init__(self, func: Callable, name: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        
    def __call__(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = self.func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            self.call_count += 1
            self.total_time += duration
            self.min_time = min(self.min_time, duration)
            self.max_time = max(self.max_time, duration)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if self.call_count == 0:
            return {'calls': 0}
        
        return {
            'calls': self.call_count,
            'total_time': self.total_time,
            'avg_time': self.total_time / self.call_count,
            'min_time': self.min_time,
            'max_time': self.max_time
        }


def optimize_for_performance(config: OptimizationConfig) -> Dict[str, Any]:
    """Apply global performance optimizations"""
    
    logger.info("Applying global performance optimizations...")
    
    results = {}
    
    # Initialize optimizers
    device_manager = DeviceManager()
    accelerator = ComputationAccelerator(config)
    memory_optimizer = MemoryOptimizer()
    
    # Apply optimizations
    device_manager.optimize_memory_usage()
    accelerator.accelerate_numpy_ops()
    accelerator.optimize_pytorch_settings()
    
    # Collect optimization results
    results['devices'] = device_manager.devices
    results['current_device'] = str(device_manager.current_device)
    results['memory_stats'] = memory_optimizer.monitor_memory_usage()
    results['optimization_config'] = config
    
    logger.info("Global performance optimizations completed")
    return results


def profile_function(func: Callable) -> ProfiledFunction:
    """Decorator to profile function performance"""
    return ProfiledFunction(func)


def batch_processor(batch_size: int = 32, use_processes: bool = False):
    """Decorator for batch processing optimization"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(inputs: List[Any], *args, **kwargs):
            if len(inputs) <= batch_size:
                return func(inputs, *args, **kwargs)
            
            # Process in batches
            results = []
            config = OptimizationConfig()
            accelerator = ComputationAccelerator(config)
            
            # Create batches
            batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
            
            # Process batches in parallel
            batch_results = accelerator.parallelize_function(
                lambda batch: func(batch, *args, **kwargs),
                batches,
                use_processes=use_processes
            )
            
            # Flatten results
            for batch_result in batch_results:
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
            
            return results
        
        return wrapper
    return decorator


def memory_efficient(auto_cleanup: bool = True):
    """Decorator for memory-efficient function execution"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            memory_optimizer = MemoryOptimizer()
            
            # Check initial memory
            initial_memory = memory_optimizer.monitor_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                if auto_cleanup:
                    # Cleanup if under memory pressure
                    if memory_optimizer.check_memory_pressure():
                        memory_optimizer.cleanup_memory()
        
        return wrapper
    return decorator


class PerformanceManager:
    """Centralized performance management"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.optimizers = {
            'model': ModelOptimizer(self.config),
            'memory': MemoryOptimizer(),
            'computation': ComputationAccelerator(self.config),
            'cache': CacheOptimizer()
        }
        self.profiled_functions = {}
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimize a model for performance"""
        return self.optimizers['model'].optimize_model(model)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'timestamp': time.time(),
            'config': self.config,
            'memory_stats': self.optimizers['memory'].monitor_memory_usage(),
            'function_profiles': {
                name: func.get_stats() 
                for name, func in self.profiled_functions.items()
            }
        }
        
        return report
    
    def register_profiled_function(self, name: str, func: Callable) -> ProfiledFunction:
        """Register a function for profiling"""
        profiled_func = ProfiledFunction(func, name)
        self.profiled_functions[name] = profiled_func
        return profiled_func
    
    def cleanup_resources(self) -> None:
        """Cleanup all managed resources"""
        self.optimizers['memory'].cleanup_memory()
        
        # Close thread pools
        if hasattr(self.optimizers['computation'], 'thread_pool'):
            self.optimizers['computation'].thread_pool.shutdown(wait=False)
        if hasattr(self.optimizers['computation'], 'process_pool'):
            self.optimizers['computation'].process_pool.shutdown(wait=False)


# Global performance manager
global_performance_manager = PerformanceManager()


def get_performance_manager() -> PerformanceManager:
    """Get the global performance manager"""
    return global_performance_manager