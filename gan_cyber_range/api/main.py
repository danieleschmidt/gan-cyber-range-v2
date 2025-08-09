"""
Main FastAPI Application
REST API for GAN-Cyber-Range-v2 platform
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
import uvicorn

from .models import *
from .auth import (
    user_manager, create_access_token, get_current_user, require_admin, 
    require_researcher, check_rate_limit, AuthenticationError
)
from ..core.attack_gan import AttackGAN
from ..core.cyber_range import CyberRange
from ..core.network_sim import NetworkTopology
from ..red_team.llm_adversary import RedTeamLLM
from ..utils.logging_config import get_logger
from ..utils.monitoring import MetricsCollector
from ..utils.security import SecurityManager
from ..utils.error_handling import CyberRangeError


# Initialize logger
logger = get_logger(__name__)

# Global state management
app_state = {
    "attack_gan": None,
    "cyber_ranges": {},
    "active_campaigns": {},
    "training_programs": {},
    "websocket_connections": set(),
    "metrics_collector": None,
    "security_manager": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting GAN-Cyber-Range-v2 API server")
    
    # Initialize core components
    app_state["attack_gan"] = AttackGAN()
    app_state["metrics_collector"] = MetricsCollector()
    app_state["security_manager"] = SecurityManager()
    
    # Start background tasks
    asyncio.create_task(metrics_collection_task())
    asyncio.create_task(cleanup_task())
    
    logger.info("API server startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GAN-Cyber-Range-v2 API server")
    
    # Cleanup cyber ranges
    for range_id, cyber_range in app_state["cyber_ranges"].items():
        try:
            await cyber_range.stop()
            logger.info(f"Stopped cyber range: {range_id}")
        except Exception as e:
            logger.error(f"Error stopping cyber range {range_id}: {e}")
    
    logger.info("API server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="GAN-Cyber-Range-v2 API",
    description="REST API for AI-driven cybersecurity training and research platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.terragonlabs.com"]
)


# Error handlers
@app.exception_handler(CyberRangeError)
async def cyber_range_error_handler(request: Request, exc: CyberRangeError):
    """Handle cyber range specific errors"""
    logger.error(f"CyberRange error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "CyberRangeError",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4())
        }
    )


@app.exception_handler(AuthenticationError)
async def auth_error_handler(request: Request, exc: AuthenticationError):
    """Handle authentication errors"""
    return JSONResponse(
        status_code=401,
        content={
            "error": "AuthenticationError",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Background tasks
async def metrics_collection_task():
    """Background task for metrics collection"""
    while True:
        try:
            if app_state["metrics_collector"]:
                # Collect system metrics
                metrics = app_state["metrics_collector"].collect_system_metrics()
                
                # Broadcast to WebSocket clients
                if app_state["websocket_connections"]:
                    message = WSMetricsUpdate(
                        component="system",
                        metrics=metrics,
                        timestamp=datetime.utcnow()
                    )
                    await broadcast_websocket_message(message.dict())
            
            await asyncio.sleep(30)  # Collect every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in metrics collection task: {e}")
            await asyncio.sleep(60)  # Longer sleep on error


async def cleanup_task():
    """Background task for cleanup operations"""
    while True:
        try:
            # Clean up expired ranges
            current_time = datetime.utcnow()
            ranges_to_cleanup = []
            
            for range_id, cyber_range in app_state["cyber_ranges"].items():
                if (hasattr(cyber_range, 'auto_cleanup_time') and 
                    current_time > cyber_range.auto_cleanup_time):
                    ranges_to_cleanup.append(range_id)
            
            for range_id in ranges_to_cleanup:
                try:
                    await app_state["cyber_ranges"][range_id].stop()
                    del app_state["cyber_ranges"][range_id]
                    logger.info(f"Auto-cleaned up cyber range: {range_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up range {range_id}: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(600)  # Longer sleep on error


# WebSocket management
async def broadcast_websocket_message(message: Dict[str, Any]):
    """Broadcast message to all WebSocket connections"""
    disconnected = []
    for websocket in app_state["websocket_connections"]:
        try:
            await websocket.send_json(message)
        except Exception:
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for websocket in disconnected:
        app_state["websocket_connections"].discard(websocket)


# API Routes

# Health check
@app.get("/health", response_model=SystemStatus)
async def health_check():
    """System health check"""
    components = []
    
    # Check core components
    components.append(ComponentHealth(
        component="attack_gan",
        status="healthy" if app_state["attack_gan"] else "unhealthy",
        message="AttackGAN initialized" if app_state["attack_gan"] else "Not initialized",
        last_check=datetime.utcnow()
    ))
    
    components.append(ComponentHealth(
        component="metrics_collector", 
        status="healthy" if app_state["metrics_collector"] else "unhealthy",
        message="Metrics collector active" if app_state["metrics_collector"] else "Not active",
        last_check=datetime.utcnow()
    ))
    
    overall_status = "healthy" if all(c.status == "healthy" for c in components) else "degraded"
    
    return SystemStatus(
        status=overall_status,
        version="2.0.0",
        uptime_seconds=int((datetime.utcnow() - datetime.utcnow()).total_seconds()),
        components=components,
        metrics={
            "active_ranges": len(app_state["cyber_ranges"]),
            "active_campaigns": len(app_state["active_campaigns"]),
            "websocket_connections": len(app_state["websocket_connections"])
        },
        timestamp=datetime.utcnow()
    )


# Authentication endpoints
@app.post("/auth/register", response_model=User)
async def register_user(user_data: UserCreate):
    """Register a new user"""
    try:
        user = user_manager.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            organization=user_data.organization,
            role=user_data.role
        )
        
        return User(
            id=uuid.UUID(user["id"]),
            username=user["username"],
            email=user["email"],
            full_name=user["full_name"],
            organization=user["organization"],
            role=user["role"],
            is_active=user["is_active"],
            created_at=user["created_at"],
            last_login=user["last_login"]
        )
    
    except AuthenticationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login", response_model=Token)
async def login_user(login_data: UserLogin):
    """User login"""
    user = user_manager.authenticate_user(login_data.username, login_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=3600  # 1 hour
    )


@app.get("/auth/me", response_model=User)
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information"""
    return User(
        id=uuid.UUID(current_user["id"]),
        username=current_user["username"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        organization=current_user["organization"],
        role=current_user["role"],
        is_active=current_user["is_active"],
        created_at=current_user["created_at"],
        last_login=current_user["last_login"]
    )


# Attack generation endpoints
@app.post("/attacks/generate", response_model=AttackGenerationResponse)
async def generate_attacks(
    request: AttackGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_researcher)
):
    """Generate synthetic attacks using GAN"""
    try:
        job_id = uuid.uuid4()
        
        # Security validation
        if not app_state["security_manager"].validate_attack_generation_request(request.dict()):
            raise HTTPException(
                status_code=400,
                detail="Attack generation request failed security validation"
            )
        
        # Generate attacks
        attack_gan = app_state["attack_gan"]
        attacks = attack_gan.generate(
            attack_types=[t.value for t in request.attack_types],
            num_samples=request.num_samples,
            diversity_threshold=request.diversity_threshold,
            filter_detectable=request.filter_detectable
        )
        
        # Convert to response format
        attack_vectors = []
        for attack in attacks:
            vector = AttackVector(
                id=uuid.uuid4(),
                attack_type=AttackType(attack.get("type", "network")),
                techniques=attack.get("techniques", []),
                payload=attack.get("payload", ""),
                confidence=attack.get("confidence", 0.8),
                sophistication=attack.get("sophistication", 0.5),
                detectability=attack.get("detectability", 0.3),
                metadata=attack.get("metadata", {}),
                created_at=datetime.utcnow()
            )
            attack_vectors.append(vector)
        
        # Calculate diversity score
        diversity_score = attack_gan.diversity_score(attacks)
        
        response = AttackGenerationResponse(
            job_id=job_id,
            status="completed",
            attacks=attack_vectors,
            generation_stats={
                "total_requested": request.num_samples,
                "total_generated": len(attacks),
                "generation_time_seconds": 0,  # Would be calculated in real implementation
                "filter_rate": 1.0 - (len(attacks) / request.num_samples)
            },
            diversity_score=diversity_score,
            total_generated=len(attacks)
        )
        
        logger.info(f"Generated {len(attacks)} attacks for user {current_user['username']}")
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating attacks: {e}")
        raise HTTPException(status_code=500, detail="Attack generation failed")


# Cyber range endpoints
@app.post("/ranges", response_model=CyberRangeInfo)
async def create_cyber_range(
    request: CyberRangeDeployRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_researcher)
):
    """Create and deploy a new cyber range"""
    try:
        range_id = uuid.uuid4()
        
        # Security validation
        if not app_state["security_manager"].validate_cyber_range_config(request.config.dict()):
            raise HTTPException(
                status_code=400,
                detail="Cyber range configuration failed security validation"
            )
        
        # Create network topology
        topology = NetworkTopology()
        topology.generate_topology(
            template=request.config.topology.template,
            subnets=request.config.topology.subnets,
            hosts_per_subnet=request.config.topology.hosts_per_subnet,
            services=request.config.topology.services,
            vulnerabilities=request.config.topology.vulnerabilities
        )
        
        # Create cyber range
        cyber_range = CyberRange(
            range_id=str(range_id),
            topology=topology
        )
        
        # Set auto-cleanup time
        if request.config.auto_cleanup and request.config.duration_hours:
            cyber_range.auto_cleanup_time = (
                datetime.utcnow() + timedelta(hours=request.config.duration_hours)
            )
        
        # Deploy in background
        background_tasks.add_task(deploy_cyber_range_background, cyber_range, request.config)
        
        # Store in state
        app_state["cyber_ranges"][str(range_id)] = cyber_range
        
        range_info = CyberRangeInfo(
            id=range_id,
            name=request.config.name,
            status=RangeStatus.STARTING,
            config=request.config,
            created_at=datetime.utcnow(),
            resource_usage={},
            metrics={}
        )
        
        logger.info(f"Created cyber range {range_id} for user {current_user['username']}")
        
        return range_info
    
    except Exception as e:
        logger.error(f"Error creating cyber range: {e}")
        raise HTTPException(status_code=500, detail="Cyber range creation failed")


async def deploy_cyber_range_background(cyber_range: CyberRange, config: CyberRangeConfig):
    """Background task to deploy cyber range"""
    try:
        await cyber_range.deploy(
            resource_limits=config.resource_limits,
            isolation_level=config.isolation_level,
            monitoring=config.monitoring_enabled
        )
        logger.info(f"Cyber range {cyber_range.range_id} deployed successfully")
    except Exception as e:
        logger.error(f"Error deploying cyber range {cyber_range.range_id}: {e}")


@app.get("/ranges", response_model=List[CyberRangeInfo])
async def list_cyber_ranges(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List all cyber ranges"""
    ranges = []
    
    for range_id, cyber_range in app_state["cyber_ranges"].items():
        # Get current status
        status = RangeStatus.RUNNING if cyber_range.is_running() else RangeStatus.STOPPED
        
        range_info = CyberRangeInfo(
            id=uuid.UUID(range_id),
            name=getattr(cyber_range, 'name', f"Range-{range_id}"),
            status=status,
            config=CyberRangeConfig(
                name=getattr(cyber_range, 'name', f"Range-{range_id}"),
                size=RangeSize.MEDIUM  # Default value
            ),
            created_at=getattr(cyber_range, 'created_at', datetime.utcnow()),
            dashboard_url=cyber_range.get_dashboard_url() if hasattr(cyber_range, 'get_dashboard_url') else None,
            resource_usage=cyber_range.get_resource_usage() if hasattr(cyber_range, 'get_resource_usage') else {},
            metrics=cyber_range.get_metrics() if hasattr(cyber_range, 'get_metrics') else {}
        )
        ranges.append(range_info)
    
    return ranges


@app.get("/ranges/{range_id}", response_model=CyberRangeInfo)
async def get_cyber_range(
    range_id: uuid.UUID,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get specific cyber range details"""
    cyber_range = app_state["cyber_ranges"].get(str(range_id))
    
    if not cyber_range:
        raise HTTPException(status_code=404, detail="Cyber range not found")
    
    status = RangeStatus.RUNNING if cyber_range.is_running() else RangeStatus.STOPPED
    
    return CyberRangeInfo(
        id=range_id,
        name=getattr(cyber_range, 'name', f"Range-{range_id}"),
        status=status,
        config=CyberRangeConfig(
            name=getattr(cyber_range, 'name', f"Range-{range_id}"),
            size=RangeSize.MEDIUM
        ),
        created_at=getattr(cyber_range, 'created_at', datetime.utcnow()),
        dashboard_url=cyber_range.get_dashboard_url() if hasattr(cyber_range, 'get_dashboard_url') else None,
        resource_usage=cyber_range.get_resource_usage() if hasattr(cyber_range, 'get_resource_usage') else {},
        metrics=cyber_range.get_metrics() if hasattr(cyber_range, 'get_metrics') else {}
    )


@app.post("/ranges/{range_id}/start")
async def start_cyber_range(
    range_id: uuid.UUID,
    current_user: Dict[str, Any] = Depends(require_researcher)
):
    """Start a cyber range"""
    cyber_range = app_state["cyber_ranges"].get(str(range_id))
    
    if not cyber_range:
        raise HTTPException(status_code=404, detail="Cyber range not found")
    
    try:
        await cyber_range.start()
        logger.info(f"Started cyber range {range_id}")
        return {"status": "started", "message": "Cyber range is starting"}
    except Exception as e:
        logger.error(f"Error starting cyber range {range_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start cyber range")


@app.post("/ranges/{range_id}/stop")
async def stop_cyber_range(
    range_id: uuid.UUID,
    current_user: Dict[str, Any] = Depends(require_researcher)
):
    """Stop a cyber range"""
    cyber_range = app_state["cyber_ranges"].get(str(range_id))
    
    if not cyber_range:
        raise HTTPException(status_code=404, detail="Cyber range not found")
    
    try:
        await cyber_range.stop()
        logger.info(f"Stopped cyber range {range_id}")
        return {"status": "stopped", "message": "Cyber range is stopping"}
    except Exception as e:
        logger.error(f"Error stopping cyber range {range_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop cyber range")


@app.delete("/ranges/{range_id}")
async def delete_cyber_range(
    range_id: uuid.UUID,
    current_user: Dict[str, Any] = Depends(require_researcher)
):
    """Delete a cyber range"""
    cyber_range = app_state["cyber_ranges"].get(str(range_id))
    
    if not cyber_range:
        raise HTTPException(status_code=404, detail="Cyber range not found")
    
    try:
        await cyber_range.destroy()
        del app_state["cyber_ranges"][str(range_id)]
        logger.info(f"Deleted cyber range {range_id}")
        return {"status": "deleted", "message": "Cyber range deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting cyber range {range_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete cyber range")


# Red Team endpoints
@app.post("/redteam/campaigns", response_model=AttackCampaign)
async def create_attack_campaign(
    request: CampaignRequest,
    current_user: Dict[str, Any] = Depends(require_researcher)
):
    """Generate attack campaign using LLM red team"""
    try:
        # Security validation
        if not app_state["security_manager"].validate_campaign_request(request.dict()):
            raise HTTPException(
                status_code=400,
                detail="Campaign request failed security validation"
            )
        
        # Create red team LLM
        red_team = RedTeamLLM(
            model=request.red_team_config.model,
            creativity=request.red_team_config.creativity,
            risk_tolerance=request.red_team_config.risk_tolerance
        )
        
        # Generate campaign
        campaign_data = red_team.generate_campaign(
            objective=request.red_team_config.objective,
            target_profile=request.target_profile.dict(),
            duration_days=request.campaign_duration,
            tactics=request.tactics
        )
        
        # Convert to response format
        stages = []
        for i, stage_data in enumerate(campaign_data.get("stages", [])):
            stage = AttackStage(
                number=i + 1,
                name=stage_data.get("name", f"Stage {i + 1}"),
                objective=stage_data.get("objective", ""),
                techniques=stage_data.get("techniques", []),
                success_criteria=stage_data.get("success_criteria", ""),
                estimated_duration=stage_data.get("duration", "1 day"),
                risk_level=stage_data.get("risk_level", "medium"),
                detection_likelihood=stage_data.get("detection_likelihood", 0.5)
            )
            stages.append(stage)
        
        campaign_id = uuid.uuid4()
        campaign = AttackCampaign(
            id=campaign_id,
            name=campaign_data.get("name", "Generated Campaign"),
            objective=request.red_team_config.objective,
            target_profile=request.target_profile,
            stages=stages,
            total_duration=request.campaign_duration,
            overall_risk=campaign_data.get("overall_risk", "medium"),
            success_probability=campaign_data.get("success_probability", 0.7),
            created_at=datetime.utcnow()
        )
        
        # Store campaign
        app_state["active_campaigns"][str(campaign_id)] = campaign
        
        logger.info(f"Created attack campaign {campaign_id} for user {current_user['username']}")
        
        return campaign
    
    except Exception as e:
        logger.error(f"Error creating attack campaign: {e}")
        raise HTTPException(status_code=500, detail="Campaign generation failed")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    app_state["websocket_connections"].add(websocket)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Echo back for testing
            await websocket.send_json({
                "type": "echo",
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    except WebSocketDisconnect:
        app_state["websocket_connections"].discard(websocket)


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "gan_cyber_range.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )