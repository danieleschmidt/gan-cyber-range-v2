"""
Database Models for GAN-Cyber-Range-v2
SQLAlchemy ORM models for all entities
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    String, Integer, Float, Boolean, DateTime, Text, JSON, 
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.types import TypeDecorator, CHAR

from .database import Base


class GUID(TypeDecorator):
    """Platform-independent GUID type"""
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value


# User Management Models
class User(Base):
    """User account model"""
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    organization: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Preferences and settings
    preferences: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    api_key: Mapped[Optional[str]] = mapped_column(String(255), unique=True)
    
    # Relationships
    cyber_ranges: Mapped[List["CyberRange"]] = relationship("CyberRange", back_populates="owner")
    campaigns: Mapped[List["AttackCampaign"]] = relationship("AttackCampaign", back_populates="creator")
    training_sessions: Mapped[List["TrainingSession"]] = relationship("TrainingSession", back_populates="participant")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_active', 'is_active'),
        Index('idx_user_role', 'role'),
        Index('idx_user_created', 'created_at'),
    )

    def __repr__(self):
        return f"<User(username='{self.username}', role='{self.role}')>"


class UserSession(Base):
    """User session tracking"""
    __tablename__ = "user_sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id"), nullable=False)
    session_token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))  # IPv6 compatible
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    
    # Relationship
    user: Mapped["User"] = relationship("User")
    
    __table_args__ = (
        Index('idx_session_active', 'is_active'),
        Index('idx_session_expires', 'expires_at'),
    )


# Cyber Range Models
class CyberRange(Base):
    """Cyber range deployment model"""
    __tablename__ = "cyber_ranges"
    
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    owner_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id"), nullable=False)
    
    # Configuration
    size: Mapped[str] = mapped_column(String(20), nullable=False)  # small, medium, large, enterprise
    template: Mapped[str] = mapped_column(String(50), nullable=False)
    topology_config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    resource_limits: Mapped[Dict[str, Any]] = mapped_column(JSON)
    
    # Status and lifecycle
    status: Mapped[str] = mapped_column(String(20), default="stopped")  # stopped, starting, running, stopping, error
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    stopped_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Runtime information
    dashboard_url: Mapped[Optional[str]] = mapped_column(String(500))
    vpn_config: Mapped[Optional[str]] = mapped_column(Text)
    container_ids: Mapped[Optional[List[str]]] = mapped_column(JSON)
    
    # Settings
    isolation_level: Mapped[str] = mapped_column(String(20), default="strict")
    monitoring_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_cleanup: Mapped[bool] = mapped_column(Boolean, default=True)
    cleanup_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Tags and metadata
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON)
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="cyber_ranges")
    attack_executions: Mapped[List["AttackExecution"]] = relationship("AttackExecution", back_populates="cyber_range")
    training_sessions: Mapped[List["TrainingSession"]] = relationship("TrainingSession", back_populates="cyber_range")
    metrics: Mapped[List["RangeMetrics"]] = relationship("RangeMetrics", back_populates="cyber_range")
    
    __table_args__ = (
        Index('idx_range_owner', 'owner_id'),
        Index('idx_range_status', 'status'),
        Index('idx_range_created', 'created_at'),
    )

    def __repr__(self):
        return f"<CyberRange(name='{self.name}', status='{self.status}')>"


class RangeMetrics(Base):
    """Cyber range performance metrics"""
    __tablename__ = "range_metrics"
    
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    cyber_range_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("cyber_ranges.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    # Resource metrics
    cpu_usage: Mapped[Optional[float]] = mapped_column(Float)
    memory_usage: Mapped[Optional[float]] = mapped_column(Float)
    disk_usage: Mapped[Optional[float]] = mapped_column(Float)
    network_rx: Mapped[Optional[float]] = mapped_column(Float)
    network_tx: Mapped[Optional[float]] = mapped_column(Float)
    
    # Attack metrics
    active_attacks: Mapped[int] = mapped_column(Integer, default=0)
    successful_attacks: Mapped[int] = mapped_column(Integer, default=0)
    blocked_attacks: Mapped[int] = mapped_column(Integer, default=0)
    
    # Defense metrics
    alerts_generated: Mapped[int] = mapped_column(Integer, default=0)
    incidents_created: Mapped[int] = mapped_column(Integer, default=0)
    response_actions: Mapped[int] = mapped_column(Integer, default=0)
    
    # Additional metrics
    custom_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Relationship
    cyber_range: Mapped["CyberRange"] = relationship("CyberRange", back_populates="metrics")
    
    __table_args__ = (
        Index('idx_metrics_timestamp', 'timestamp'),
        Index('idx_metrics_range_time', 'cyber_range_id', 'timestamp'),
    )


# Attack Generation Models
class AttackVector(Base):
    """Generated attack vector model"""
    __tablename__ = "attack_vectors"
    
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    attack_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Attack details
    techniques: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON)
    
    # Quality metrics
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    sophistication: Mapped[float] = mapped_column(Float, nullable=False)
    detectability: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Generation info
    generator_model: Mapped[str] = mapped_column(String(100))
    generation_params: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_by: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id"))
    
    # Usage tracking
    used_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    creator: Mapped["User"] = relationship("User")
    executions: Mapped[List["AttackExecution"]] = relationship("AttackExecution", back_populates="attack_vector")
    
    __table_args__ = (
        Index('idx_attack_type', 'attack_type'),
        Index('idx_attack_created', 'created_at'),
        Index('idx_attack_confidence', 'confidence'),
    )


class AttackCampaign(Base):
    """Red team attack campaign model"""
    __tablename__ = "attack_campaigns"
    
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Campaign configuration
    target_profile: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    red_team_config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    tactics: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    
    # Timeline
    total_duration: Mapped[int] = mapped_column(Integer, nullable=False)  # Days
    stages: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    
    # Risk assessment
    overall_risk: Mapped[str] = mapped_column(String(20), nullable=False)
    success_probability: Mapped[float] = mapped_column(Float, nullable=False)
    stealth_level: Mapped[str] = mapped_column(String(20), default="medium")
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_by: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id"), nullable=False)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="created")  # created, active, completed, cancelled
    
    # Relationships
    creator: Mapped["User"] = relationship("User", back_populates="campaigns")
    executions: Mapped[List["AttackExecution"]] = relationship("AttackExecution", back_populates="campaign")
    
    __table_args__ = (
        Index('idx_campaign_creator', 'created_by'),
        Index('idx_campaign_status', 'status'),
        Index('idx_campaign_risk', 'overall_risk'),
    )


class AttackExecution(Base):
    """Attack execution tracking"""
    __tablename__ = "attack_executions"
    
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    
    # Relationships
    cyber_range_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("cyber_ranges.id"), nullable=False)
    attack_vector_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("attack_vectors.id"))
    campaign_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("attack_campaigns.id"))
    
    # Execution details
    stage_number: Mapped[Optional[int]] = mapped_column(Integer)
    target_host: Mapped[Optional[str]] = mapped_column(String(255))
    technique: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Timeline
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Results
    status: Mapped[str] = mapped_column(String(20), default="running")  # running, successful, failed, blocked
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    detection_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    response_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Details
    execution_log: Mapped[Optional[str]] = mapped_column(Text)
    artifacts: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    network_captures: Mapped[Optional[List[str]]] = mapped_column(JSON)
    
    # Relationships
    cyber_range: Mapped["CyberRange"] = relationship("CyberRange", back_populates="attack_executions")
    attack_vector: Mapped[Optional["AttackVector"]] = relationship("AttackVector", back_populates="executions")
    campaign: Mapped[Optional["AttackCampaign"]] = relationship("AttackCampaign", back_populates="executions")
    
    __table_args__ = (
        Index('idx_execution_range', 'cyber_range_id'),
        Index('idx_execution_status', 'status'),
        Index('idx_execution_started', 'started_at'),
    )


# Training Models
class TrainingScenario(Base):
    """Training scenario template"""
    __tablename__ = "training_scenarios"
    
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Configuration
    level: Mapped[str] = mapped_column(String(20), nullable=False)  # beginner, intermediate, advanced, expert
    focus_areas: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    learning_objectives: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    
    # Content
    scenario_config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    attack_patterns: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    defense_objectives: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON)
    
    # Settings
    duration_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    max_team_size: Mapped[int] = mapped_column(Integer, default=10)
    hints_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    scoring_model: Mapped[str] = mapped_column(String(50), default="mitre_attack")
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_by: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id"), nullable=False)
    version: Mapped[str] = mapped_column(String(20), default="1.0")
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON)
    
    # Usage stats
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_rating: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationships
    creator: Mapped["User"] = relationship("User")
    training_sessions: Mapped[List["TrainingSession"]] = relationship("TrainingSession", back_populates="scenario")
    
    __table_args__ = (
        Index('idx_scenario_level', 'level'),
        Index('idx_scenario_creator', 'created_by'),
        Index('idx_scenario_usage', 'usage_count'),
    )


class TrainingSession(Base):
    """Training session instance"""
    __tablename__ = "training_sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    
    # Relationships
    scenario_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("training_scenarios.id"), nullable=False)
    cyber_range_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("cyber_ranges.id"), nullable=False)
    participant_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id"), nullable=False)
    
    # Session info
    team_name: Mapped[Optional[str]] = mapped_column(String(255))
    instructor_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("users.id"))
    
    # Timeline
    scheduled_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="scheduled")  # scheduled, active, completed, cancelled
    completion_percentage: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Performance metrics
    detection_rate: Mapped[float] = mapped_column(Float, default=0.0)
    mean_time_to_detect: Mapped[float] = mapped_column(Float, default=0.0)  # Minutes
    mean_time_to_respond: Mapped[float] = mapped_column(Float, default=0.0)  # Minutes
    incidents_handled: Mapped[int] = mapped_column(Integer, default=0)
    false_positives: Mapped[int] = mapped_column(Integer, default=0)
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Feedback
    participant_feedback: Mapped[Optional[str]] = mapped_column(Text)
    instructor_notes: Mapped[Optional[str]] = mapped_column(Text)
    rating: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5 scale
    improvement_areas: Mapped[Optional[List[str]]] = mapped_column(JSON)
    
    # Relationships
    scenario: Mapped["TrainingScenario"] = relationship("TrainingScenario", back_populates="training_sessions")
    cyber_range: Mapped["CyberRange"] = relationship("CyberRange", back_populates="training_sessions")
    participant: Mapped["User"] = relationship("User", back_populates="training_sessions", foreign_keys=[participant_id])
    instructor: Mapped[Optional["User"]] = relationship("User", foreign_keys=[instructor_id])
    
    __table_args__ = (
        Index('idx_session_participant', 'participant_id'),
        Index('idx_session_scenario', 'scenario_id'),
        Index('idx_session_status', 'status'),
        Index('idx_session_started', 'started_at'),
    )


# System Models
class AuditLog(Base):
    """System audit logging"""
    __tablename__ = "audit_logs"
    
    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    # Actor information
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("users.id"))
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    
    # Action details
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Context
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship("User")
    
    __table_args__ = (
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_timestamp', 'timestamp'),
    )


class SystemConfiguration(Base):
    """System configuration storage"""
    __tablename__ = "system_config"
    
    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(String(20), default="string")  # string, json, boolean, integer
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(GUID(), ForeignKey("users.id"))
    
    # Security
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    creator: Mapped[Optional["User"]] = relationship("User")
    
    __table_args__ = (
        Index('idx_config_updated', 'updated_at'),
    )