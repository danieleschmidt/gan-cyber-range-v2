"""
Database Module for GAN-Cyber-Range-v2
Persistence layer for users, ranges, campaigns, and training data
"""

from .database import Database, get_database
from .models import *
from .repositories import *

__all__ = [
    'Database', 'get_database', 
    'UserRepository', 'CyberRangeRepository', 
    'CampaignRepository', 'TrainingRepository'
]