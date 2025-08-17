"""
Research module for academic validation and publication-ready experiments.

This module provides comprehensive research infrastructure for validating
novel cybersecurity AI algorithms and generating reproducible results.
"""

from .baseline_comparator import BaselineComparator, BaselineExperiment
from .experiment_framework import ExperimentFramework, Experiment, ExperimentConfig
from .statistical_validator import StatisticalValidator, StatisticalTest
from .reproducibility_engine import ReproducibilityEngine, ReproducibleResult
from .publication_generator import PublicationGenerator, ResearchPaper

__all__ = [
    "BaselineComparator",
    "BaselineExperiment", 
    "ExperimentFramework",
    "Experiment",
    "ExperimentConfig",
    "StatisticalValidator",
    "StatisticalTest",
    "ReproducibilityEngine", 
    "ReproducibleResult",
    "PublicationGenerator",
    "ResearchPaper"
]