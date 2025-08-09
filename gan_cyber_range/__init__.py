"""
GAN-Cyber-Range-v2: Advanced Cybersecurity Training Platform

A second-generation adversarial cyber range that combines GAN-based attack generation 
with LLM-driven red team curricula for comprehensive cybersecurity training and research.
"""

__version__ = "2.0.0"
__author__ = "Daniel Schmidt"
__license__ = "MIT"

# Core imports (only import what exists)
from .core.attack_gan import AttackGAN
from .core.cyber_range import CyberRange  
from .core.network_sim import NetworkTopology
from .core.attack_engine import AttackSimulator

# Red team imports (only import what exists)
from .red_team.llm_adversary import RedTeamLLM

__all__ = [
    # Core
    "AttackGAN",
    "CyberRange", 
    "NetworkTopology",
    "AttackSimulator",
    
    # Red Team
    "RedTeamLLM",
]