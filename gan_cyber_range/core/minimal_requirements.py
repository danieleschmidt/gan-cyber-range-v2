"""
Minimal requirements implementation to handle missing dependencies gracefully.
This module provides fallback implementations when optional dependencies are not available.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class MockTensor:
    """Mock tensor implementation when PyTorch is not available"""
    
    def __init__(self, data):
        self.data = data
        self.shape = getattr(data, 'shape', (1,))
    
    def numpy(self):
        return self.data if hasattr(self.data, 'numpy') else self.data
    
    def __str__(self):
        return f"MockTensor({self.data})"


class MockModule:
    """Mock neural network module when PyTorch is not available"""
    
    def __init__(self):
        self.training = True
        
    def forward(self, x):
        return x
        
    def train(self, mode=True):
        self.training = mode
        return self
        
    def eval(self):
        return self.train(False)
        
    def parameters(self):
        return []


def safe_import(module_name: str, fallback=None):
    """Safely import a module with fallback"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        logger.warning(f"Module {module_name} not available, using fallback")
        return False


# Check for numpy availability
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Create a minimal numpy-like interface
    class MockNumpy:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def random():
            import random
            class MockRandom:
                @staticmethod
                def normal(loc=0, scale=1, size=None):
                    if size is None:
                        return random.gauss(loc, scale)
                    return [random.gauss(loc, scale) for _ in range(size)]
            return MockRandom()
    
    np = MockNumpy()


# Check for torch availability  
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Create minimal torch-like interface
    class MockTorch:
        @staticmethod
        def tensor(data):
            return MockTensor(data)
        
        @staticmethod
        def randn(*args):
            import random
            if len(args) == 1:
                return MockTensor([random.gauss(0, 1) for _ in range(args[0])])
            return MockTensor([random.gauss(0, 1)])
    
    class MockNN:
        Linear = MockModule
        ReLU = MockModule
        Dropout = MockModule
        BatchNorm1d = MockModule
        Module = MockModule
    
    torch = MockTorch()
    nn = MockNN()


@dataclass
class MinimalConfig:
    """Minimal configuration for basic functionality"""
    name: str = "minimal_range"
    mode: str = "defensive_only"
    security_level: str = "high"
    resource_limits: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = {
                'cpu_cores': 2,
                'memory_gb': 4,
                'storage_gb': 10
            }


class MinimalAttackVector:
    """Minimal attack vector for testing purposes"""
    
    def __init__(self, attack_type: str = "test", payload: str = "harmless_test"):
        self.attack_type = attack_type
        self.payload = payload
        self.techniques = ["test_technique"]
        self.severity = 0.1  # Very low severity for testing
        self.stealth_level = 0.0
        self.target_systems = ["test_system"]
        self.timestamp = datetime.now().isoformat()
        self.metadata = {"purpose": "testing", "safe": True}


class MinimalGenerator:
    """Minimal generator when GAN components are not available"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        logger.info("Using minimal generator (GAN components not available)")
    
    def generate(self, num_samples: int = 1) -> List[MinimalAttackVector]:
        """Generate minimal test attack vectors"""
        return [
            MinimalAttackVector(
                attack_type="test_attack",
                payload=f"test_payload_{i}"
            ) for i in range(num_samples)
        ]
    
    def train(self, *args, **kwargs):
        """Mock training method"""
        logger.info("Mock training completed (no actual training performed)")
        return {"status": "completed", "mode": "mock"}


class MinimalCyberRange:
    """Minimal cyber range for basic functionality testing"""
    
    def __init__(self, config: Optional[MinimalConfig] = None):
        self.config = config or MinimalConfig()
        self.status = "initialized"
        self.attack_log = []
        logger.info(f"Initialized minimal cyber range: {self.config.name}")
    
    def deploy(self, **kwargs):
        """Mock deployment"""
        self.status = "deployed"
        logger.info("Mock deployment completed")
        return {"status": "deployed", "mode": "mock"}
    
    def execute_attack(self, attack_vector: MinimalAttackVector):
        """Mock attack execution"""
        self.attack_log.append({
            'attack': attack_vector,
            'timestamp': datetime.now(),
            'result': 'simulated_success'
        })
        logger.info(f"Mock attack executed: {attack_vector.attack_type}")
        return {"status": "executed", "mode": "mock"}
    
    def get_status(self):
        """Get range status"""
        return {
            "status": self.status,
            "config": self.config,
            "attacks_executed": len(self.attack_log),
            "mode": "minimal"
        }


def create_minimal_components():
    """Create minimal components for testing"""
    return {
        "generator": MinimalGenerator(),
        "cyber_range": MinimalCyberRange(),
        "config": MinimalConfig(),
        "has_numpy": HAS_NUMPY,
        "has_torch": HAS_TORCH
    }


def check_dependencies():
    """Check which dependencies are available"""
    deps = {
        "numpy": HAS_NUMPY,
        "torch": HAS_TORCH,
        "docker": safe_import("docker"),
        "redis": safe_import("redis"),
        "sqlalchemy": safe_import("sqlalchemy"),
        "fastapi": safe_import("fastapi"),
        "cryptography": safe_import("cryptography")
    }
    
    logger.info(f"Dependency check: {deps}")
    return deps