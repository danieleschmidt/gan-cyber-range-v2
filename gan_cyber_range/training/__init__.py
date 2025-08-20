"""
Enhanced Training Module for GAN-Cyber-Range-v2

This module provides advanced training capabilities for defensive cybersecurity skills,
specifically designed to help security professionals learn to detect and respond to 
AI-generated attack campaigns.
"""

from .defensive_training_enhancer import (
    DefensiveTrainingEnhancer,
    DefensiveSkill,
    TrainingScenario,
    TrainingSession,
    LearningPath,
    TrainingDifficulty,
    TrainingOutcome,
    create_training_enhancer
)

__all__ = [
    "DefensiveTrainingEnhancer",
    "DefensiveSkill",
    "TrainingScenario", 
    "TrainingSession",
    "LearningPath",
    "TrainingDifficulty",
    "TrainingOutcome",
    "create_training_enhancer"
]