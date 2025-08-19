"""
Threat hunting capabilities for proactive defense.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HuntStatus(Enum):
    """Status of threat hunting session"""
    ACTIVE = "active"
    COMPLETED = "completed"
    SUSPENDED = "suspended"


@dataclass
class IoC:
    """Indicator of Compromise"""
    indicator_type: str
    value: str
    confidence: float
    source: str
    first_seen: datetime
    last_seen: Optional[datetime] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ThreatHunter:
    """Proactive threat hunting system"""
    
    def __init__(self, name: str = "default_hunter"):
        self.name = name
        self.active_hunts = {}
        
    def start_hunt(self, hunt_id: str, hypothesis: str) -> "ThreatHuntingSession":
        """Start a new threat hunting session"""
        session = ThreatHuntingSession(hunt_id, hypothesis)
        self.active_hunts[hunt_id] = session
        logger.info(f"Started threat hunt: {hunt_id}")
        return session
        
    def get_hunt(self, hunt_id: str) -> Optional["ThreatHuntingSession"]:
        """Get threat hunting session"""
        return self.active_hunts.get(hunt_id)


class ThreatHuntingSession:
    """Individual threat hunting session"""
    
    def __init__(self, hunt_id: str, hypothesis: str):
        self.hunt_id = hunt_id
        self.hypothesis = hypothesis
        self.status = HuntStatus.ACTIVE
        self.start_time = datetime.now()
        self.iocs_found = []
        self.findings = []
        
    def add_ioc(self, ioc: IoC):
        """Add indicator of compromise"""
        self.iocs_found.append(ioc)
        
    def add_finding(self, finding: str):
        """Add finding to hunt"""
        self.findings.append({
            'timestamp': datetime.now(),
            'finding': finding
        })
        
    def complete_hunt(self):
        """Complete the hunting session"""
        self.status = HuntStatus.COMPLETED
        logger.info(f"Completed threat hunt: {self.hunt_id}")