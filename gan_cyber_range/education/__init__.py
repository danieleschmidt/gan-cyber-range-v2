"""
Enhanced Education Module for GAN-Cyber-Range-v2

This module provides advanced educational capabilities including adaptive curriculum
management, personalized learning paths, and comprehensive progress tracking for
cybersecurity professionals learning to defend against AI-powered threats.
"""

from .curriculum_manager import (
    CurriculumManager,
    LearnerProfile,
    LearningObjective,
    EducationalContent,
    CurriculumPath,
    LearningSession,
    LearningStyle,
    KnowledgeLevel,
    ContentType,
    create_curriculum_manager
)

__all__ = [
    "CurriculumManager",
    "LearnerProfile", 
    "LearningObjective",
    "EducationalContent",
    "CurriculumPath",
    "LearningSession",
    "LearningStyle",
    "KnowledgeLevel", 
    "ContentType",
    "create_curriculum_manager"
]