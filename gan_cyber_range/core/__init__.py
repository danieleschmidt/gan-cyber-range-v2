"""Core components for GAN-Cyber-Range-v2"""

# Try to import full components, fall back to minimal if dependencies missing
try:
    from .attack_gan import AttackGAN
except ImportError:
    from .minimal_requirements import MinimalGenerator as AttackGAN

try:
    from .cyber_range import CyberRange
except ImportError:
    from .minimal_requirements import MinimalCyberRange as CyberRange

try:  
    from .network_sim import NetworkTopology, NetworkSimulator
except ImportError:
    # Minimal network simulation components
    class NetworkTopology:
        def __init__(self, **kwargs):
            self.config = kwargs
            
    class NetworkSimulator:
        def __init__(self, **kwargs):
            self.config = kwargs

try:
    from .attack_engine import AttackEngine, AttackSimulator
except ImportError:
    # Minimal attack engine components
    class AttackEngine:
        def __init__(self, **kwargs):
            self.config = kwargs
            
    class AttackSimulator:
        def __init__(self, **kwargs):
            self.config = kwargs

__all__ = [
    "AttackGAN",
    "CyberRange", 
    "NetworkTopology",
    "NetworkSimulator",
    "AttackEngine",
    "AttackSimulator"
]