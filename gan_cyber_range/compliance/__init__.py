"""
Global compliance module for multi-regional cyber range deployment.

This module provides comprehensive compliance frameworks for GDPR, CCPA, 
PDPA, and other international data protection regulations.
"""

from .gdpr_compliance import GDPRCompliance, GDPRArticle
from .ccpa_compliance import CCPACompliance, CCPARequirement
from .iso_compliance import ISOCompliance, ISOStandard
from .global_compliance_manager import GlobalComplianceManager, ComplianceRegion
from .data_governance import DataGovernance, DataClassification
from .privacy_controls import PrivacyControls, ConsentManager

__all__ = [
    "GDPRCompliance",
    "GDPRArticle",
    "CCPACompliance", 
    "CCPARequirement",
    "ISOCompliance",
    "ISOStandard",
    "GlobalComplianceManager",
    "ComplianceRegion",
    "DataGovernance",
    "DataClassification",
    "PrivacyControls",
    "ConsentManager"
]