"""
API Models for GAN-Cyber-Range-v2
Pydantic models for request/response validation
"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class AttackType(str, Enum):
    """Supported attack types"""
    MALWARE = "malware"
    NETWORK = "network"
    WEB = "web"
    SOCIAL_ENGINEERING = "social_engineering"
    APT = "apt"
    RANSOMWARE = "ransomware"


class RangeSize(str, Enum):
    """Cyber range deployment sizes"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"


class RangeStatus(str, Enum):
    """Cyber range status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class TrainingLevel(str, Enum):
    """Training difficulty levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# Authentication Models
class UserCreate(BaseModel):
    """User creation request"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    organization: Optional[str] = None
    role: str = Field(default="user", regex=r'^(admin|researcher|educator|student|user)$')


class UserLogin(BaseModel):
    """User login request"""
    username: str
    password: str


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class User(BaseModel):
    """User information response"""
    id: uuid.UUID
    username: str
    email: str
    full_name: Optional[str] = None
    organization: Optional[str] = None
    role: str
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


# Attack Generation Models
class AttackGenerationRequest(BaseModel):
    """Request to generate synthetic attacks"""
    attack_types: List[AttackType] = Field(..., min_items=1)
    num_samples: int = Field(100, ge=1, le=10000)
    diversity_threshold: float = Field(0.8, ge=0.0, le=1.0)
    filter_detectable: bool = True
    privacy_budget: Optional[float] = Field(None, ge=0.1, le=100.0)
    
    @validator('attack_types')
    def validate_attack_types(cls, v):
        if len(set(v)) != len(v):
            raise ValueError('Duplicate attack types not allowed')
        return v


class AttackVector(BaseModel):
    """Generated attack vector"""
    id: uuid.UUID
    attack_type: AttackType
    techniques: List[str]
    payload: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sophistication: float = Field(..., ge=0.0, le=1.0)
    detectability: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = {}
    created_at: datetime


class AttackGenerationResponse(BaseModel):
    """Response from attack generation"""
    job_id: uuid.UUID
    status: str
    attacks: List[AttackVector]
    generation_stats: Dict[str, Any]
    diversity_score: float
    total_generated: int


# Cyber Range Models
class NetworkTopologyConfig(BaseModel):
    """Network topology configuration"""
    template: str = Field("enterprise", regex=r'^(enterprise|small_office|cloud|custom)$')
    subnets: List[str] = Field(default=["dmz", "internal", "management"])
    hosts_per_subnet: Dict[str, int] = Field(default_factory=dict)
    services: List[str] = Field(default=["web", "database", "email", "vpn"])
    vulnerabilities: str = Field("realistic", regex=r'^(none|minimal|realistic|high)$')


class CyberRangeConfig(BaseModel):
    """Cyber range deployment configuration"""
    name: str = Field(..., min_length=1, max_length=100)
    size: RangeSize = RangeSize.MEDIUM
    topology: NetworkTopologyConfig = Field(default_factory=NetworkTopologyConfig)
    resource_limits: Dict[str, Union[int, str]] = Field(default_factory=dict)
    isolation_level: str = Field("strict", regex=r'^(minimal|standard|strict)$')
    monitoring_enabled: bool = True
    auto_cleanup: bool = True
    duration_hours: Optional[int] = Field(None, ge=1, le=168)  # Max 7 days


class CyberRangeDeployRequest(BaseModel):
    """Request to deploy a cyber range"""
    config: CyberRangeConfig
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class CyberRangeInfo(BaseModel):
    """Cyber range information"""
    id: uuid.UUID
    name: str
    status: RangeStatus
    config: CyberRangeConfig
    created_at: datetime
    started_at: Optional[datetime] = None
    dashboard_url: Optional[str] = None
    vpn_config: Optional[str] = None
    resource_usage: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}


# Red Team Models
class RedTeamConfig(BaseModel):
    """Red team LLM configuration"""
    model: str = Field("llama2-70b-security")
    creativity: float = Field(0.8, ge=0.0, le=1.0)
    risk_tolerance: float = Field(0.6, ge=0.0, le=1.0)
    objective: str = Field(..., min_length=1)
    max_campaign_duration: int = Field(30, ge=1, le=90)  # Days


class TargetProfile(BaseModel):
    """Target organization profile"""
    industry: str
    size: str = Field(..., regex=r'^(small|medium|large|enterprise)$')
    security_maturity: str = Field(..., regex=r'^(basic|intermediate|advanced|expert)$')
    crown_jewels: List[str] = Field(..., min_items=1)
    known_vulnerabilities: List[str] = Field(default_factory=list)


class CampaignRequest(BaseModel):
    """Request to generate attack campaign"""
    red_team_config: RedTeamConfig
    target_profile: TargetProfile
    campaign_duration: int = Field(30, ge=1, le=90)  # Days
    tactics: List[str] = Field(..., min_items=1)
    stealth_level: str = Field("medium", regex=r'^(low|medium|high)$')


class AttackStage(BaseModel):
    """Single stage in attack campaign"""
    number: int
    name: str
    objective: str
    techniques: List[str]
    success_criteria: str
    estimated_duration: str
    risk_level: str
    detection_likelihood: float = Field(..., ge=0.0, le=1.0)


class AttackCampaign(BaseModel):
    """Generated attack campaign"""
    id: uuid.UUID
    name: str
    objective: str
    target_profile: TargetProfile
    stages: List[AttackStage]
    total_duration: int
    overall_risk: str
    success_probability: float = Field(..., ge=0.0, le=1.0)
    created_at: datetime


# Training Models
class TrainingScenarioConfig(BaseModel):
    """Training scenario configuration"""
    name: str = Field(..., min_length=1, max_length=200)
    level: TrainingLevel = TrainingLevel.INTERMEDIATE
    focus_areas: List[str] = Field(..., min_items=1)
    learning_objectives: List[str] = Field(..., min_items=1)
    duration_hours: int = Field(8, ge=1, le=40)
    team_size: int = Field(4, ge=1, le=20)
    hints_enabled: bool = True
    scoring_model: str = Field("mitre_attack")


class TrainingProgramRequest(BaseModel):
    """Request to create training program"""
    scenario_config: TrainingScenarioConfig
    cyber_range_id: uuid.UUID
    start_time: Optional[datetime] = None
    participants: List[uuid.UUID] = Field(..., min_items=1)


class TrainingMetrics(BaseModel):
    """Training performance metrics"""
    participant_id: uuid.UUID
    detection_rate: float = Field(..., ge=0.0, le=1.0)
    mean_time_to_detect: float  # Minutes
    mean_time_to_respond: float  # Minutes
    incidents_handled: int
    false_positives: int
    overall_score: float = Field(..., ge=0, le=100)
    improvement_areas: List[str]


class TrainingResults(BaseModel):
    """Training program results"""
    program_id: uuid.UUID
    scenario_name: str
    participants: List[TrainingMetrics]
    team_metrics: Dict[str, float]
    completion_status: str
    started_at: datetime
    completed_at: Optional[datetime] = None


# System Status Models
class ComponentHealth(BaseModel):
    """Health status of system component"""
    component: str
    status: str = Field(..., regex=r'^(healthy|degraded|unhealthy|unknown)$')
    message: Optional[str] = None
    last_check: datetime
    response_time_ms: Optional[float] = None


class SystemStatus(BaseModel):
    """Overall system status"""
    status: str = Field(..., regex=r'^(healthy|degraded|unhealthy)$')
    version: str
    uptime_seconds: int
    components: List[ComponentHealth]
    metrics: Dict[str, Any]
    timestamp: datetime


# Error Models
class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: Optional[uuid.UUID] = None


class ValidationError(BaseModel):
    """Validation error details"""
    field: str
    message: str
    invalid_value: Any


class ValidationErrorResponse(ErrorResponse):
    """Validation error response"""
    validation_errors: List[ValidationError]


# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1, le=100)
    sort_by: Optional[str] = None
    sort_order: str = Field("desc", regex=r'^(asc|desc)$')


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool


# WebSocket Models
class WSMessage(BaseModel):
    """WebSocket message"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class WSAttackEvent(WSMessage):
    """Attack event WebSocket message"""
    type: str = "attack_event"
    attack_id: uuid.UUID
    event_type: str
    severity: str
    details: Dict[str, Any]


class WSMetricsUpdate(WSMessage):
    """Metrics update WebSocket message"""
    type: str = "metrics_update"
    component: str
    metrics: Dict[str, float]
    timestamp: datetime