"""
Repository Pattern Implementation
Data access layer for database operations
"""

import uuid
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import (
    User, UserSession, CyberRange, RangeMetrics, 
    AttackVector, AttackCampaign, AttackExecution,
    TrainingScenario, TrainingSession, AuditLog, SystemConfiguration
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseRepository:
    """Base repository with common database operations"""
    
    def __init__(self, session: AsyncSession, model_class):
        self.session = session
        self.model_class = model_class
    
    async def create(self, **kwargs) -> Any:
        """Create a new record"""
        obj = self.model_class(**kwargs)
        self.session.add(obj)
        await self.session.flush()
        return obj
    
    async def get_by_id(self, obj_id: uuid.UUID) -> Optional[Any]:
        """Get record by ID"""
        result = await self.session.execute(
            select(self.model_class).where(self.model_class.id == obj_id)
        )
        return result.scalar_one_or_none()
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """Get all records with pagination"""
        result = await self.session.execute(
            select(self.model_class)
            .limit(limit)
            .offset(offset)
            .order_by(desc(self.model_class.created_at))
        )
        return result.scalars().all()
    
    async def update(self, obj_id: uuid.UUID, **kwargs) -> Optional[Any]:
        """Update record by ID"""
        await self.session.execute(
            update(self.model_class)
            .where(self.model_class.id == obj_id)
            .values(**kwargs)
        )
        return await self.get_by_id(obj_id)
    
    async def delete(self, obj_id: uuid.UUID) -> bool:
        """Delete record by ID"""
        result = await self.session.execute(
            delete(self.model_class).where(self.model_class.id == obj_id)
        )
        return result.rowcount > 0
    
    async def count(self, **filters) -> int:
        """Count records with optional filters"""
        query = select(func.count()).select_from(self.model_class)
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    conditions.append(getattr(self.model_class, key) == value)
            if conditions:
                query = query.where(and_(*conditions))
        
        result = await self.session.execute(query)
        return result.scalar()
    
    async def exists(self, **filters) -> bool:
        """Check if record exists with given filters"""
        return await self.count(**filters) > 0


class UserRepository(BaseRepository):
    """User management repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, User)
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        result = await self.session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def get_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key"""
        result = await self.session.execute(
            select(User).where(User.api_key == api_key)
        )
        return result.scalar_one_or_none()
    
    async def create_user(self, username: str, email: str, password_hash: str,
                         full_name: str = None, organization: str = None,
                         role: str = "user") -> User:
        """Create a new user"""
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            organization=organization,
            role=role
        )
        self.session.add(user)
        await self.session.flush()
        return user
    
    async def update_last_login(self, user_id: uuid.UUID) -> None:
        """Update user's last login timestamp"""
        await self.session.execute(
            update(User)
            .where(User.id == user_id)
            .values(last_login=datetime.utcnow())
        )
    
    async def get_active_users(self, limit: int = 100) -> List[User]:
        """Get active users"""
        result = await self.session.execute(
            select(User)
            .where(User.is_active == True)
            .order_by(desc(User.last_login))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def search_users(self, query: str, role: str = None) -> List[User]:
        """Search users by username, email, or name"""
        conditions = [
            User.is_active == True,
            or_(
                User.username.ilike(f"%{query}%"),
                User.email.ilike(f"%{query}%"),
                User.full_name.ilike(f"%{query}%")
            )
        ]
        
        if role:
            conditions.append(User.role == role)
        
        result = await self.session.execute(
            select(User)
            .where(and_(*conditions))
            .order_by(User.username)
        )
        return result.scalars().all()


class UserSessionRepository(BaseRepository):
    """User session management repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, UserSession)
    
    async def create_session(self, user_id: uuid.UUID, session_token: str,
                           expires_at: datetime, ip_address: str = None,
                           user_agent: str = None) -> UserSession:
        """Create a new user session"""
        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        self.session.add(session)
        await self.session.flush()
        return session
    
    async def get_by_token(self, token: str) -> Optional[UserSession]:
        """Get session by token"""
        result = await self.session.execute(
            select(UserSession)
            .where(
                and_(
                    UserSession.session_token == token,
                    UserSession.is_active == True,
                    UserSession.expires_at > datetime.utcnow()
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def invalidate_session(self, token: str) -> bool:
        """Invalidate session"""
        result = await self.session.execute(
            update(UserSession)
            .where(UserSession.session_token == token)
            .values(is_active=False)
        )
        return result.rowcount > 0
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        result = await self.session.execute(
            delete(UserSession)
            .where(UserSession.expires_at < datetime.utcnow())
        )
        return result.rowcount


class CyberRangeRepository(BaseRepository):
    """Cyber range management repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, CyberRange)
    
    async def create_range(self, name: str, owner_id: uuid.UUID, 
                          size: str, template: str, topology_config: dict,
                          **kwargs) -> CyberRange:
        """Create a new cyber range"""
        cyber_range = CyberRange(
            name=name,
            owner_id=owner_id,
            size=size,
            template=template,
            topology_config=topology_config,
            **kwargs
        )
        self.session.add(cyber_range)
        await self.session.flush()
        return cyber_range
    
    async def get_user_ranges(self, user_id: uuid.UUID, 
                             status: str = None) -> List[CyberRange]:
        """Get cyber ranges for a user"""
        conditions = [CyberRange.owner_id == user_id]
        
        if status:
            conditions.append(CyberRange.status == status)
        
        result = await self.session.execute(
            select(CyberRange)
            .where(and_(*conditions))
            .order_by(desc(CyberRange.created_at))
        )
        return result.scalars().all()
    
    async def get_ranges_for_cleanup(self) -> List[CyberRange]:
        """Get ranges that need cleanup"""
        result = await self.session.execute(
            select(CyberRange)
            .where(
                and_(
                    CyberRange.auto_cleanup == True,
                    CyberRange.cleanup_at < datetime.utcnow(),
                    CyberRange.status.in_(["running", "stopped"])
                )
            )
        )
        return result.scalars().all()
    
    async def update_status(self, range_id: uuid.UUID, status: str,
                           **additional_fields) -> None:
        """Update range status"""
        updates = {"status": status}
        
        if status == "running":
            updates["started_at"] = datetime.utcnow()
        elif status == "stopped":
            updates["stopped_at"] = datetime.utcnow()
        
        updates.update(additional_fields)
        
        await self.session.execute(
            update(CyberRange)
            .where(CyberRange.id == range_id)
            .values(**updates)
        )
    
    async def get_resource_usage_summary(self) -> Dict[str, Any]:
        """Get resource usage summary across all ranges"""
        result = await self.session.execute(
            select(
                func.count(CyberRange.id).label("total_ranges"),
                func.sum(func.case((CyberRange.status == "running", 1), else_=0)).label("running_ranges"),
                func.count(func.distinct(CyberRange.owner_id)).label("active_users")
            )
        )
        return result.first()._asdict()


class RangeMetricsRepository(BaseRepository):
    """Range metrics repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, RangeMetrics)
    
    async def record_metrics(self, cyber_range_id: uuid.UUID, 
                           metrics: Dict[str, Any]) -> RangeMetrics:
        """Record metrics for a cyber range"""
        range_metrics = RangeMetrics(
            cyber_range_id=cyber_range_id,
            **metrics
        )
        self.session.add(range_metrics)
        await self.session.flush()
        return range_metrics
    
    async def get_metrics_by_timerange(self, cyber_range_id: uuid.UUID,
                                     start_time: datetime, 
                                     end_time: datetime) -> List[RangeMetrics]:
        """Get metrics for a time range"""
        result = await self.session.execute(
            select(RangeMetrics)
            .where(
                and_(
                    RangeMetrics.cyber_range_id == cyber_range_id,
                    RangeMetrics.timestamp >= start_time,
                    RangeMetrics.timestamp <= end_time
                )
            )
            .order_by(RangeMetrics.timestamp)
        )
        return result.scalars().all()
    
    async def get_latest_metrics(self, cyber_range_id: uuid.UUID) -> Optional[RangeMetrics]:
        """Get latest metrics for a cyber range"""
        result = await self.session.execute(
            select(RangeMetrics)
            .where(RangeMetrics.cyber_range_id == cyber_range_id)
            .order_by(desc(RangeMetrics.timestamp))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def cleanup_old_metrics(self, days_to_keep: int = 30) -> int:
        """Clean up old metrics"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        result = await self.session.execute(
            delete(RangeMetrics)
            .where(RangeMetrics.timestamp < cutoff_date)
        )
        return result.rowcount


class AttackVectorRepository(BaseRepository):
    """Attack vector repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, AttackVector)
    
    async def create_attack_vector(self, attack_type: str, techniques: List[str],
                                 payload: str, confidence: float, 
                                 sophistication: float, detectability: float,
                                 created_by: uuid.UUID, **kwargs) -> AttackVector:
        """Create a new attack vector"""
        attack = AttackVector(
            attack_type=attack_type,
            techniques=techniques,
            payload=payload,
            confidence=confidence,
            sophistication=sophistication,
            detectability=detectability,
            created_by=created_by,
            **kwargs
        )
        self.session.add(attack)
        await self.session.flush()
        return attack
    
    async def get_by_type(self, attack_type: str, limit: int = 100) -> List[AttackVector]:
        """Get attack vectors by type"""
        result = await self.session.execute(
            select(AttackVector)
            .where(AttackVector.attack_type == attack_type)
            .order_by(desc(AttackVector.confidence))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_high_confidence_attacks(self, min_confidence: float = 0.8) -> List[AttackVector]:
        """Get high confidence attack vectors"""
        result = await self.session.execute(
            select(AttackVector)
            .where(AttackVector.confidence >= min_confidence)
            .order_by(desc(AttackVector.confidence))
        )
        return result.scalars().all()
    
    async def increment_usage(self, attack_id: uuid.UUID) -> None:
        """Increment usage counter for attack vector"""
        await self.session.execute(
            update(AttackVector)
            .where(AttackVector.id == attack_id)
            .values(
                used_count=AttackVector.used_count + 1,
                last_used=datetime.utcnow()
            )
        )
    
    async def search_attacks(self, query: str, attack_types: List[str] = None) -> List[AttackVector]:
        """Search attack vectors"""
        conditions = []
        
        if attack_types:
            conditions.append(AttackVector.attack_type.in_(attack_types))
        
        # Search in techniques (JSON array)
        conditions.append(
            or_(
                func.json_extract(AttackVector.techniques, '$').like(f'%{query}%'),
                AttackVector.payload.ilike(f'%{query}%')
            )
        )
        
        result = await self.session.execute(
            select(AttackVector)
            .where(and_(*conditions))
            .order_by(desc(AttackVector.confidence))
        )
        return result.scalars().all()


class AttackCampaignRepository(BaseRepository):
    """Attack campaign repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, AttackCampaign)
    
    async def create_campaign(self, name: str, objective: str, 
                            target_profile: dict, red_team_config: dict,
                            tactics: List[str], stages: List[dict],
                            created_by: uuid.UUID, **kwargs) -> AttackCampaign:
        """Create a new attack campaign"""
        campaign = AttackCampaign(
            name=name,
            objective=objective,
            target_profile=target_profile,
            red_team_config=red_team_config,
            tactics=tactics,
            stages=stages,
            created_by=created_by,
            **kwargs
        )
        self.session.add(campaign)
        await self.session.flush()
        return campaign
    
    async def get_user_campaigns(self, user_id: uuid.UUID) -> List[AttackCampaign]:
        """Get campaigns created by user"""
        result = await self.session.execute(
            select(AttackCampaign)
            .where(AttackCampaign.created_by == user_id)
            .order_by(desc(AttackCampaign.created_at))
        )
        return result.scalars().all()
    
    async def get_active_campaigns(self) -> List[AttackCampaign]:
        """Get active campaigns"""
        result = await self.session.execute(
            select(AttackCampaign)
            .where(AttackCampaign.status == "active")
            .order_by(AttackCampaign.created_at)
        )
        return result.scalars().all()


class AttackExecutionRepository(BaseRepository):
    """Attack execution repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, AttackExecution)
    
    async def create_execution(self, cyber_range_id: uuid.UUID, technique: str,
                             attack_vector_id: uuid.UUID = None,
                             campaign_id: uuid.UUID = None,
                             **kwargs) -> AttackExecution:
        """Create a new attack execution"""
        execution = AttackExecution(
            cyber_range_id=cyber_range_id,
            attack_vector_id=attack_vector_id,
            campaign_id=campaign_id,
            technique=technique,
            **kwargs
        )
        self.session.add(execution)
        await self.session.flush()
        return execution
    
    async def get_range_executions(self, cyber_range_id: uuid.UUID) -> List[AttackExecution]:
        """Get executions for a cyber range"""
        result = await self.session.execute(
            select(AttackExecution)
            .where(AttackExecution.cyber_range_id == cyber_range_id)
            .order_by(desc(AttackExecution.started_at))
        )
        return result.scalars().all()
    
    async def update_execution_status(self, execution_id: uuid.UUID, 
                                    status: str, success: bool = None,
                                    **kwargs) -> None:
        """Update execution status"""
        updates = {"status": status}
        
        if success is not None:
            updates["success"] = success
        
        if status in ["successful", "failed", "blocked"]:
            updates["completed_at"] = datetime.utcnow()
        
        updates.update(kwargs)
        
        await self.session.execute(
            update(AttackExecution)
            .where(AttackExecution.id == execution_id)
            .values(**updates)
        )
    
    async def get_success_rate(self, cyber_range_id: uuid.UUID = None,
                             time_period_hours: int = 24) -> float:
        """Calculate attack success rate"""
        conditions = [
            AttackExecution.started_at >= datetime.utcnow() - timedelta(hours=time_period_hours),
            AttackExecution.status.in_(["successful", "failed", "blocked"])
        ]
        
        if cyber_range_id:
            conditions.append(AttackExecution.cyber_range_id == cyber_range_id)
        
        result = await self.session.execute(
            select(
                func.count(AttackExecution.id).label("total"),
                func.sum(func.case((AttackExecution.success == True, 1), else_=0)).label("successful")
            )
            .where(and_(*conditions))
        )
        
        row = result.first()
        if row.total == 0:
            return 0.0
        return float(row.successful) / float(row.total)


class TrainingScenarioRepository(BaseRepository):
    """Training scenario repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, TrainingScenario)
    
    async def create_scenario(self, name: str, description: str, level: str,
                            focus_areas: List[str], learning_objectives: List[str],
                            scenario_config: dict, duration_hours: int,
                            created_by: uuid.UUID, **kwargs) -> TrainingScenario:
        """Create a new training scenario"""
        scenario = TrainingScenario(
            name=name,
            description=description,
            level=level,
            focus_areas=focus_areas,
            learning_objectives=learning_objectives,
            scenario_config=scenario_config,
            duration_hours=duration_hours,
            created_by=created_by,
            **kwargs
        )
        self.session.add(scenario)
        await self.session.flush()
        return scenario
    
    async def get_by_level(self, level: str) -> List[TrainingScenario]:
        """Get scenarios by difficulty level"""
        result = await self.session.execute(
            select(TrainingScenario)
            .where(TrainingScenario.level == level)
            .order_by(desc(TrainingScenario.usage_count))
        )
        return result.scalars().all()
    
    async def search_scenarios(self, query: str, level: str = None,
                             focus_areas: List[str] = None) -> List[TrainingScenario]:
        """Search training scenarios"""
        conditions = [
            or_(
                TrainingScenario.name.ilike(f"%{query}%"),
                TrainingScenario.description.ilike(f"%{query}%")
            )
        ]
        
        if level:
            conditions.append(TrainingScenario.level == level)
        
        # Search in focus areas (JSON array)
        if focus_areas:
            for area in focus_areas:
                conditions.append(
                    func.json_extract(TrainingScenario.focus_areas, '$').like(f'%{area}%')
                )
        
        result = await self.session.execute(
            select(TrainingScenario)
            .where(and_(*conditions))
            .order_by(desc(TrainingScenario.avg_rating), desc(TrainingScenario.usage_count))
        )
        return result.scalars().all()
    
    async def increment_usage(self, scenario_id: uuid.UUID) -> None:
        """Increment usage counter"""
        await self.session.execute(
            update(TrainingScenario)
            .where(TrainingScenario.id == scenario_id)
            .values(usage_count=TrainingScenario.usage_count + 1)
        )


class TrainingSessionRepository(BaseRepository):
    """Training session repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, TrainingSession)
    
    async def create_session(self, scenario_id: uuid.UUID, cyber_range_id: uuid.UUID,
                           participant_id: uuid.UUID, **kwargs) -> TrainingSession:
        """Create a new training session"""
        session = TrainingSession(
            scenario_id=scenario_id,
            cyber_range_id=cyber_range_id,
            participant_id=participant_id,
            **kwargs
        )
        self.session.add(session)
        await self.session.flush()
        return session
    
    async def get_user_sessions(self, user_id: uuid.UUID) -> List[TrainingSession]:
        """Get training sessions for a user"""
        result = await self.session.execute(
            select(TrainingSession)
            .options(selectinload(TrainingSession.scenario))
            .where(TrainingSession.participant_id == user_id)
            .order_by(desc(TrainingSession.started_at))
        )
        return result.scalars().all()
    
    async def update_performance(self, session_id: uuid.UUID, 
                               metrics: Dict[str, Any]) -> None:
        """Update session performance metrics"""
        await self.session.execute(
            update(TrainingSession)
            .where(TrainingSession.id == session_id)
            .values(**metrics)
        )
    
    async def get_performance_stats(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """Get performance statistics for a user"""
        result = await self.session.execute(
            select(
                func.count(TrainingSession.id).label("total_sessions"),
                func.avg(TrainingSession.overall_score).label("avg_score"),
                func.avg(TrainingSession.detection_rate).label("avg_detection_rate"),
                func.avg(TrainingSession.mean_time_to_detect).label("avg_mttd"),
                func.avg(TrainingSession.mean_time_to_respond).label("avg_mttr")
            )
            .where(
                and_(
                    TrainingSession.participant_id == user_id,
                    TrainingSession.status == "completed"
                )
            )
        )
        return result.first()._asdict()


class AuditLogRepository(BaseRepository):
    """Audit log repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, AuditLog)
    
    async def log_action(self, action: str, resource_type: str, 
                        user_id: uuid.UUID = None, resource_id: str = None,
                        details: dict = None, success: bool = True,
                        ip_address: str = None, user_agent: str = None,
                        error_message: str = None) -> AuditLog:
        """Log an action"""
        audit_log = AuditLog(
            action=action,
            resource_type=resource_type,
            user_id=user_id,
            resource_id=resource_id,
            details=details,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=error_message
        )
        self.session.add(audit_log)
        await self.session.flush()
        return audit_log
    
    async def get_user_activity(self, user_id: uuid.UUID, 
                              limit: int = 100) -> List[AuditLog]:
        """Get user activity"""
        result = await self.session.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(desc(AuditLog.timestamp))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_recent_activity(self, hours: int = 24, 
                                limit: int = 1000) -> List[AuditLog]:
        """Get recent system activity"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        result = await self.session.execute(
            select(AuditLog)
            .where(AuditLog.timestamp >= cutoff_time)
            .order_by(desc(AuditLog.timestamp))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def cleanup_old_logs(self, days_to_keep: int = 90) -> int:
        """Clean up old audit logs"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        result = await self.session.execute(
            delete(AuditLog)
            .where(AuditLog.timestamp < cutoff_date)
        )
        return result.rowcount


class SystemConfigRepository(BaseRepository):
    """System configuration repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, SystemConfiguration)
    
    async def get_config(self, key: str) -> Optional[SystemConfiguration]:
        """Get configuration by key"""
        result = await self.session.execute(
            select(SystemConfiguration).where(SystemConfiguration.key == key)
        )
        return result.scalar_one_or_none()
    
    async def set_config(self, key: str, value: str, value_type: str = "string",
                        description: str = None, is_sensitive: bool = False,
                        created_by: uuid.UUID = None) -> SystemConfiguration:
        """Set configuration value"""
        config = await self.get_config(key)
        
        if config:
            await self.session.execute(
                update(SystemConfiguration)
                .where(SystemConfiguration.key == key)
                .values(
                    value=value,
                    value_type=value_type,
                    description=description,
                    updated_at=datetime.utcnow()
                )
            )
            config = await self.get_config(key)
        else:
            config = SystemConfiguration(
                key=key,
                value=value,
                value_type=value_type,
                description=description,
                is_sensitive=is_sensitive,
                created_by=created_by
            )
            self.session.add(config)
            await self.session.flush()
        
        return config
    
    async def get_all_config(self, include_sensitive: bool = False) -> List[SystemConfiguration]:
        """Get all configuration"""
        query = select(SystemConfiguration)
        
        if not include_sensitive:
            query = query.where(SystemConfiguration.is_sensitive == False)
        
        result = await self.session.execute(query.order_by(SystemConfiguration.key))
        return result.scalars().all()
    
    async def delete_config(self, key: str) -> bool:
        """Delete configuration"""
        result = await self.session.execute(
            delete(SystemConfiguration).where(SystemConfiguration.key == key)
        )
        return result.rowcount > 0