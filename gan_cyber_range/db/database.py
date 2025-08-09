"""
Database Configuration and Connection Management
SQLAlchemy-based database layer with async support
"""

import os
import asyncio
from typing import Optional, Any, Dict
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy import create_engine, text
import redis.asyncio as redis

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite+aiosqlite:///./data/gan_cyber_range.db"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Connection pools
MAX_CONNECTIONS = int(os.getenv("DB_MAX_CONNECTIONS", "20"))
CONNECTION_TIMEOUT = int(os.getenv("DB_CONNECTION_TIMEOUT", "30"))


class Base(DeclarativeBase):
    """Base class for all database models"""
    pass


class Database:
    """Database connection and session management"""
    
    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self.engine = None
        self.async_session_factory = None
        self.redis_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections"""
        if self._initialized:
            return
        
        try:
            # Create async SQLAlchemy engine
            self.engine = create_async_engine(
                self.database_url,
                echo=os.getenv("DEBUG", "false").lower() == "true",
                pool_size=MAX_CONNECTIONS,
                max_overflow=10,
                pool_timeout=CONNECTION_TIMEOUT,
                pool_recycle=3600,  # 1 hour
                connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {}
            )
            
            # Create session factory
            self.async_session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Initialize Redis connection
            try:
                self.redis_client = redis.Redis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True
                )
                
                # Test Redis connection
                await self.redis_client.ping()
                logger.info("Redis connection established")
                
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Cache features disabled.")
                self.redis_client = None
            
            # Create tables
            await self.create_tables()
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def create_tables(self):
        """Create database tables"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created/updated")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    async def close(self):
        """Close database connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connection closed")
            
            if self.engine:
                await self.engine.dispose()
                logger.info("Database engine disposed")
            
            self._initialized = False
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session context manager"""
        if not self._initialized:
            await self.initialize()
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def execute_raw_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute raw SQL query"""
        async with self.get_session() as session:
            result = await session.execute(text(query), params or {})
            return result.fetchall()
    
    async def health_check(self) -> Dict[str, str]:
        """Check database health"""
        health = {
            "database": "unhealthy",
            "redis": "unhealthy"
        }
        
        try:
            # Test database connection
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            health["database"] = "healthy"
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
        
        try:
            # Test Redis connection
            if self.redis_client:
                await self.redis_client.ping()
                health["redis"] = "healthy"
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
        
        return health
    
    # Cache methods (Redis)
    async def cache_get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def cache_set(self, key: str, value: str, expire: int = 3600) -> bool:
        """Set value in cache with expiration"""
        if not self.redis_client:
            return False
        
        try:
            return await self.redis_client.setex(key, expire, value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def cache_delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False
        
        try:
            return await self.redis_client.delete(key) > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def cache_exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False
        
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    # Bulk operations
    async def bulk_insert(self, model_class, objects: list) -> bool:
        """Bulk insert objects"""
        try:
            async with self.get_session() as session:
                session.add_all([model_class(**obj) for obj in objects])
                await session.commit()
            return True
        except Exception as e:
            logger.error(f"Bulk insert error: {e}")
            return False
    
    async def bulk_update(self, model_class, updates: list) -> int:
        """Bulk update objects"""
        try:
            updated_count = 0
            async with self.get_session() as session:
                for update in updates:
                    obj_id = update.pop('id')
                    result = await session.execute(
                        model_class.__table__.update()
                        .where(model_class.id == obj_id)
                        .values(**update)
                    )
                    updated_count += result.rowcount
                await session.commit()
            return updated_count
        except Exception as e:
            logger.error(f"Bulk update error: {e}")
            return 0


# Global database instance
_database_instance: Optional[Database] = None


async def get_database() -> Database:
    """Get or create global database instance"""
    global _database_instance
    
    if _database_instance is None:
        _database_instance = Database()
        await _database_instance.initialize()
    
    return _database_instance


# Dependency for FastAPI
async def get_db_session():
    """Dependency to get database session in FastAPI"""
    db = await get_database()
    async with db.get_session() as session:
        yield session


# Database initialization for CLI/scripts
async def init_database():
    """Initialize database for CLI usage"""
    db = await get_database()
    logger.info("Database initialization complete")
    return db


# Migration support
class DatabaseMigration:
    """Database migration utilities"""
    
    def __init__(self, database: Database):
        self.database = database
    
    async def create_migration_table(self):
        """Create migration tracking table"""
        query = """
        CREATE TABLE IF NOT EXISTS migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version VARCHAR(50) UNIQUE NOT NULL,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        await self.database.execute_raw_query(query)
    
    async def apply_migration(self, version: str, description: str, sql_commands: list):
        """Apply database migration"""
        await self.create_migration_table()
        
        # Check if migration already applied
        existing = await self.database.execute_raw_query(
            "SELECT version FROM migrations WHERE version = :version",
            {"version": version}
        )
        
        if existing:
            logger.info(f"Migration {version} already applied")
            return
        
        try:
            async with self.database.get_session() as session:
                # Execute migration commands
                for command in sql_commands:
                    await session.execute(text(command))
                
                # Record migration
                await session.execute(
                    text("INSERT INTO migrations (version, description) VALUES (:version, :description)"),
                    {"version": version, "description": description}
                )
                
                await session.commit()
                logger.info(f"Applied migration {version}: {description}")
                
        except Exception as e:
            logger.error(f"Migration {version} failed: {e}")
            raise
    
    async def get_applied_migrations(self) -> list:
        """Get list of applied migrations"""
        await self.create_migration_table()
        result = await self.database.execute_raw_query(
            "SELECT version, description, applied_at FROM migrations ORDER BY applied_at"
        )
        return [dict(row) for row in result]


# Connection pooling and health monitoring
class DatabaseHealth:
    """Database health monitoring"""
    
    def __init__(self, database: Database):
        self.database = database
        self.health_stats = {
            "connections": 0,
            "queries": 0,
            "errors": 0,
            "avg_query_time": 0.0
        }
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information"""
        if not self.database.engine:
            return {"status": "not_initialized"}
        
        pool = self.database.engine.pool
        
        return {
            "status": "connected",
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
    
    async def monitor_performance(self):
        """Background task for performance monitoring"""
        while True:
            try:
                health = await self.database.health_check()
                connection_info = await self.get_connection_info()
                
                logger.debug(f"Database health: {health}")
                logger.debug(f"Connection info: {connection_info}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Database monitoring error: {e}")
                await asyncio.sleep(60)  # Longer sleep on error