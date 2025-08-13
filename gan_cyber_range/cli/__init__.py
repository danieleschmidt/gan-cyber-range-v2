"""
Command-line interface for GAN-Cyber-Range-v2.

This module provides comprehensive CLI tools for managing cyber ranges,
training GANs, and executing security scenarios.
"""

from .main import main
from .train import train_gan
from .manager import manage_range
from .scenarios import run_scenario

__all__ = ['main', 'train_gan', 'manage_range', 'run_scenario']