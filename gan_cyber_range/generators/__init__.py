"""
GAN-based attack generators for creating synthetic attack patterns.

This module provides specialized generators for different attack types including
malware, network attacks, web exploits, and social engineering campaigns.
"""

from .malware_gan import MalwareGAN
from .network_gan import NetworkAttackGAN
from .web_attack_gan import WebAttackGAN
from .social_gan import SocialEngineeringGAN

__all__ = [
    'MalwareGAN',
    'NetworkAttackGAN', 
    'WebAttackGAN',
    'SocialEngineeringGAN'
]