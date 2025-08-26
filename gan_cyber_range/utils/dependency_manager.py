"""
Dependency Management and Graceful Fallbacks

This module provides intelligent dependency management with graceful
fallbacks when optional dependencies are not available.
"""

import logging
import importlib
import warnings
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class DependencyStatus:
    """Status of a dependency"""
    name: str
    available: bool
    version: Optional[str] = None
    fallback_available: bool = False
    error_message: Optional[str] = None


class DependencyManager:
    """Manages dependencies with intelligent fallbacks"""
    
    def __init__(self):
        self.dependencies: Dict[str, DependencyStatus] = {}
        self.fallbacks: Dict[str, Any] = {}
        self._check_core_dependencies()
    
    def check_dependency(self, name: str, min_version: Optional[str] = None) -> bool:
        """Check if a dependency is available"""
        if name in self.dependencies:
            return self.dependencies[name].available
            
        try:
            module = importlib.import_module(name)
            version = getattr(module, '__version__', 'unknown')
            
            status = DependencyStatus(
                name=name,
                available=True,
                version=version
            )
            
            logger.info(f"✅ Dependency {name} ({version}) available")
            
        except ImportError as e:
            status = DependencyStatus(
                name=name,
                available=False,
                error_message=str(e)
            )
            
            logger.warning(f"❌ Dependency {name} not available: {e}")
        
        self.dependencies[name] = status
        return status.available
    
    def require_dependency(self, name: str, fallback: Optional[Any] = None):
        """Decorator to require a dependency with optional fallback"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self.check_dependency(name):
                    return func(*args, **kwargs)
                elif fallback is not None:
                    logger.info(f"Using fallback for {name}")
                    return fallback(*args, **kwargs)
                else:
                    raise ImportError(
                        f"Required dependency '{name}' not available. "
                        f"Install with: pip install {name}"
                    )
            return wrapper
        return decorator
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get dependency status report"""
        return {
            "core_available": all(
                dep.available for dep in self.dependencies.values() 
                if dep.name in ['json', 'logging', 'uuid']
            ),
            "ml_available": all(
                self.check_dependency(name) 
                for name in ['numpy', 'torch', 'transformers']
            ),
            "network_available": all(
                self.check_dependency(name)
                for name in ['docker', 'paramiko', 'scapy']
            ),
            "dependencies": {
                name: {
                    "available": dep.available,
                    "version": dep.version,
                    "error": dep.error_message
                }
                for name, dep in self.dependencies.items()
            }
        }
    
    def _check_core_dependencies(self):
        """Check core Python dependencies"""
        core_deps = ['json', 'logging', 'uuid', 'datetime', 'random']
        for dep in core_deps:
            self.check_dependency(dep)


# Global dependency manager instance
dep_manager = DependencyManager()


def safe_import(module_name: str, fallback=None):
    """Safely import a module with fallback"""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        logger.warning(f"Failed to import {module_name}: {e}")
        if fallback:
            logger.info(f"Using fallback for {module_name}")
            return fallback
        return None


# Fallback implementations for common dependencies
class NumpyFallback:
    """Fallback for numpy operations"""
    
    @staticmethod
    def array(data):
        """Convert to list (fallback for numpy.array)"""
        return list(data) if hasattr(data, '__iter__') else [data]
    
    @staticmethod
    def random(size=None):
        """Random number generation fallback"""
        import random
        if size is None:
            return random.random()
        elif isinstance(size, int):
            return [random.random() for _ in range(size)]
        else:
            # For tuple sizes, generate nested lists
            if len(size) == 1:
                return [random.random() for _ in range(size[0])]
            else:
                return [[random.random() for _ in range(size[1])] 
                        for _ in range(size[0])]
    
    @staticmethod
    def mean(data):
        """Calculate mean (fallback for numpy.mean)"""
        return sum(data) / len(data) if data else 0
    
    @staticmethod
    def std(data):
        """Calculate standard deviation (fallback for numpy.std)"""
        if not data:
            return 0
        mean_val = NumpyFallback.mean(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        return variance ** 0.5


class TorchFallback:
    """Fallback for PyTorch operations"""
    
    class Tensor:
        """Minimal tensor-like object"""
        def __init__(self, data):
            self.data = data if isinstance(data, list) else [data]
        
        def __repr__(self):
            return f"FallbackTensor({self.data})"
        
        def size(self):
            return len(self.data)
        
        def item(self):
            return self.data[0] if self.data else 0
    
    @staticmethod
    def tensor(data):
        return TorchFallback.Tensor(data)
    
    @staticmethod
    def randn(*size):
        import random
        if len(size) == 1:
            return TorchFallback.Tensor([random.gauss(0, 1) for _ in range(size[0])])
        return TorchFallback.Tensor([0.0])  # Simplified


# Register fallbacks
dep_manager.fallbacks.update({
    'numpy': NumpyFallback,
    'torch': TorchFallback
})


def get_numpy(fallback=True):
    """Get numpy module or fallback"""
    try:
        import numpy as np
        return np
    except ImportError:
        if fallback:
            logger.info("Using numpy fallback implementation")
            return NumpyFallback
        raise


def get_torch(fallback=True):
    """Get torch module or fallback"""
    try:
        import torch
        return torch
    except ImportError:
        if fallback:
            logger.info("Using torch fallback implementation")
            return TorchFallback
        raise


def optional_dependency(func):
    """Decorator for functions that use optional dependencies"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            logger.warning(f"Optional dependency missing in {func.__name__}: {e}")
            logger.info("Consider installing missing dependencies for full functionality")
            return None
    return wrapper