"""
GAN-Cyber-Range-v2: Advanced Cybersecurity Training Platform

A second-generation adversarial cyber range that combines GAN-based attack generation 
with LLM-driven red team curricula for comprehensive cybersecurity training and research.
"""

__version__ = "2.0.0"
__author__ = "Daniel Schmidt"
__license__ = "MIT"

# Core imports
from .core.attack_gan import AttackGAN
from .core.cyber_range import CyberRange  
from .core.network_sim import NetworkTopology
from .core.attack_engine import AttackSimulator

# Generator imports
from .generators.malware_gan import MalwareGAN
from .generators.network_gan import NetworkAttackGAN
from .generators.web_attack_gan import WebAttackGAN
from .generators.social_gan import SocialEngineeringGAN

# Red team imports
from .red_team.llm_adversary import RedTeamLLM
from .red_team.campaign_planner import ScenarioGenerator
from .red_team.technique_library import MitreAttackMapper
from .red_team.payload_generator import PayloadGenerator

# Blue team imports
from .blue_team.defense_suite import BlueTeamEvaluator
from .blue_team.incident_response import IncidentResponse
from .blue_team.threat_hunting import ThreatHunter
from .blue_team.forensics import DigitalForensics

# Analysis imports
from .analysis.attack_analyzer import AttackAnalyzer
from .analysis.defense_metrics import DefenseMetrics
from .analysis.visualization import CyberRangeViz
from .analysis.reporting import ReportGenerator

__all__ = [
    # Core
    "AttackGAN",
    "CyberRange", 
    "NetworkTopology",
    "AttackSimulator",
    
    # Generators
    "MalwareGAN",
    "NetworkAttackGAN", 
    "WebAttackGAN",
    "SocialEngineeringGAN",
    
    # Red Team
    "RedTeamLLM",
    "ScenarioGenerator",
    "MitreAttackMapper",
    "PayloadGenerator",
    
    # Blue Team
    "BlueTeamEvaluator",
    "IncidentResponse",
    "ThreatHunter", 
    "DigitalForensics",
    
    # Analysis
    "AttackAnalyzer",
    "DefenseMetrics",
    "CyberRangeViz",
    "ReportGenerator"
]