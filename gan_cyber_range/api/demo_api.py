"""
FastAPI-based REST API for GAN-Cyber-Range-v2 Demo System

Provides secure, production-ready API endpoints for cyber range management,
attack generation, and monitoring with comprehensive error handling.
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import logging
import json
import uuid
from datetime import datetime, timedelta
import secrets
import hashlib
import hmac
from contextlib import asynccontextmanager
import asyncio

# Import our demo system
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from demo import DemoAPI, SimpleCyberRange, SimpleAttackVector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Security configuration
security = HTTPBearer()
API_SECRET_KEY = secrets.token_urlsafe(32)
logger.info(f"API initialized with secret key: {API_SECRET_KEY[:8]}...")

# Request/Response Models
class RangeCreateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    networks: Optional[List[str]] = Field(default=["dmz", "internal"])
    hosts_count: Optional[int] = Field(default=6, ge=1, le=50)
    
    @validator('name')
    def validate_name(cls, v):
        if v and not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Name must contain only alphanumeric characters, hyphens, and underscores')
        return v


class AttackRequest(BaseModel):
    count: int = Field(default=5, ge=1, le=100)
    attack_type: Optional[str] = Field(None, regex="^(malware|network|web|social_engineering)$")
    severity_filter: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @validator('attack_type')
    def validate_attack_type(cls, v):
        allowed_types = ["malware", "network", "web", "social_engineering"]
        if v and v not in allowed_types:
            raise ValueError(f'Attack type must be one of: {", ".join(allowed_types)}')
        return v


class RangeResponse(BaseModel):
    range_id: str
    name: str
    status: str
    dashboard_url: str
    created_at: str


class MetricsResponse(BaseModel):
    range_id: str
    status: str
    uptime_seconds: float
    total_attacks: int
    successful_attacks: int
    detected_attacks: int
    detection_rate: float
    success_rate: float
    active_hosts: int


class AttackResponse(BaseModel):
    generated_attacks: int
    attack_type: str
    execution_time: float
    summary: Dict[str, Any]


# Enhanced DemoAPI with security and monitoring
class SecureDemoAPI(DemoAPI):
    """Enhanced demo API with security and monitoring features"""
    
    def __init__(self):
        super().__init__()
        self.api_keys = set()
        self.rate_limits = {}
        self.security_events = []
        self.performance_metrics = {
            "requests": 0,
            "errors": 0,
            "avg_response_time": 0.0,
            "start_time": datetime.now()
        }
    
    def generate_api_key(self, user_id: str) -> str:
        """Generate secure API key for user"""
        raw_key = f"{user_id}:{secrets.token_urlsafe(32)}"
        api_key = hashlib.sha256(raw_key.encode()).hexdigest()
        self.api_keys.add(api_key)
        
        logger.info(f"Generated API key for user: {user_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        return api_key in self.api_keys
    
    def check_rate_limit(self, client_ip: str, limit: int = 100) -> bool:
        """Check rate limiting for client"""
        now = datetime.now()
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = {"count": 1, "reset_time": now + timedelta(minutes=15)}
            return True
        
        if now > self.rate_limits[client_ip]["reset_time"]:
            self.rate_limits[client_ip] = {"count": 1, "reset_time": now + timedelta(minutes=15)}
            return True
        
        if self.rate_limits[client_ip]["count"] >= limit:
            return False
        
        self.rate_limits[client_ip]["count"] += 1
        return True
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security events for monitoring"""
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.security_events.append(event)
        logger.warning(f"Security event: {event_type} - {details}")
    
    def update_metrics(self, success: bool, response_time: float) -> None:
        """Update performance metrics"""
        self.performance_metrics["requests"] += 1
        if not success:
            self.performance_metrics["errors"] += 1
        
        # Update average response time
        current_avg = self.performance_metrics["avg_response_time"]
        total_requests = self.performance_metrics["requests"]
        new_avg = (current_avg * (total_requests - 1) + response_time) / total_requests
        self.performance_metrics["avg_response_time"] = new_avg


# Global API instance
demo_api = SecureDemoAPI()

# Generate demo API key
demo_api_key = demo_api.generate_api_key("demo_user")
logger.info(f"Demo API key: {demo_api_key[:16]}...")


# FastAPI app initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting GAN-Cyber-Range-v2 API")
    
    # Startup
    yield
    
    # Shutdown
    logger.info("Shutting down API")
    # Clean up resources
    for range_id, cyber_range in demo_api.ranges.items():
        if cyber_range.status == "running":
            logger.info(f"Stopping range: {range_id}")


app = FastAPI(
    title="GAN-Cyber-Range-v2 API",
    description="Secure API for cybersecurity training platform with GAN-based attack generation",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.terragonlabs.com"]
)


# Authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify API token"""
    if not demo_api.validate_api_key(credentials.credentials):
        demo_api.log_security_event("invalid_token", {"token": credentials.credentials[:8]})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials


# Rate limiting dependency
async def rate_limit_check(request):
    """Check rate limiting"""
    client_ip = request.client.host if hasattr(request, 'client') else "unknown"
    if not demo_api.check_rate_limit(client_ip):
        demo_api.log_security_event("rate_limit_exceeded", {"client_ip": client_ip})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


# API Endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """API health check and information"""
    uptime = (datetime.now() - demo_api.performance_metrics["start_time"]).total_seconds()
    
    return {
        "service": "GAN-Cyber-Range-v2 API",
        "version": "2.0.0",
        "status": "operational",
        "uptime_seconds": uptime,
        "active_ranges": len(demo_api.ranges),
        "total_requests": demo_api.performance_metrics["requests"],
        "error_rate": demo_api.performance_metrics["errors"] / max(1, demo_api.performance_metrics["requests"]),
        "avg_response_time": demo_api.performance_metrics["avg_response_time"],
        "demo_api_key": demo_api_key[:16] + "...",
        "documentation": "/docs"
    }


@app.post("/ranges", response_model=RangeResponse)
async def create_range(
    request: RangeCreateRequest,
    token: str = Depends(verify_token)
):
    """Create a new cyber range"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Creating new range with request: {request.dict()}")
        
        # Validate request
        if len(demo_api.ranges) >= 10:  # Limit concurrent ranges
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum number of concurrent ranges reached (10)"
            )
        
        # Create range
        range_response = demo_api.create_range(request.name)
        
        response_time = (datetime.now() - start_time).total_seconds()
        demo_api.update_metrics(True, response_time)
        
        return RangeResponse(
            range_id=range_response["range_id"],
            name=range_response["name"],
            status=range_response["status"],
            dashboard_url=range_response["dashboard_url"],
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        response_time = (datetime.now() - start_time).total_seconds()
        demo_api.update_metrics(False, response_time)
        logger.error(f"Error creating range: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create range: {str(e)}"
        )


@app.get("/ranges", response_model=List[Dict[str, Any]])
async def list_ranges(token: str = Depends(verify_token)):
    """List all active cyber ranges"""
    try:
        ranges = []
        for range_id, cyber_range in demo_api.ranges.items():
            metrics = cyber_range.get_metrics()
            ranges.append({
                "range_id": range_id,
                "name": cyber_range.name,
                "status": cyber_range.status,
                "uptime_seconds": metrics["uptime_seconds"],
                "total_attacks": metrics["total_attacks"],
                "detection_rate": metrics["detection_rate"]
            })
        
        return ranges
        
    except Exception as e:
        logger.error(f"Error listing ranges: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list ranges: {str(e)}"
        )


@app.get("/ranges/{range_id}", response_model=Dict[str, Any])
async def get_range_info(
    range_id: str,
    token: str = Depends(verify_token)
):
    """Get detailed information about a specific cyber range"""
    try:
        if range_id not in demo_api.ranges:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Range not found"
            )
        
        info_response = demo_api.get_range_info(range_id)
        return info_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting range info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get range info: {str(e)}"
        )


@app.post("/ranges/{range_id}/attacks", response_model=AttackResponse)
async def generate_attacks(
    range_id: str,
    request: AttackRequest,
    token: str = Depends(verify_token)
):
    """Generate and execute attacks on a cyber range"""
    start_time = datetime.now()
    
    try:
        if range_id not in demo_api.ranges:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Range not found"
            )
        
        logger.info(f"Generating {request.count} attacks of type {request.attack_type} on range {range_id}")
        
        # Security check: Limit attack generation
        if request.count > 50:
            demo_api.log_security_event("excessive_attack_generation", {
                "range_id": range_id,
                "requested_count": request.count
            })
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Attack count exceeds maximum limit (50)"
            )
        
        # Generate attacks
        attack_response = demo_api.generate_attacks(
            range_id, 
            request.count, 
            request.attack_type
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        demo_api.update_metrics(True, execution_time)
        
        return AttackResponse(
            generated_attacks=attack_response["generated_attacks"],
            attack_type=attack_response["attack_type"],
            execution_time=execution_time,
            summary=attack_response["summary"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        demo_api.update_metrics(False, execution_time)
        logger.error(f"Error generating attacks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate attacks: {str(e)}"
        )


@app.delete("/ranges/{range_id}")
async def delete_range(
    range_id: str,
    token: str = Depends(verify_token)
):
    """Delete a cyber range and clean up resources"""
    try:
        if range_id not in demo_api.ranges:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Range not found"
            )
        
        # Cleanup range
        cyber_range = demo_api.ranges[range_id]
        logger.info(f"Deleting range: {range_id} ({cyber_range.name})")
        
        # Remove from registry
        del demo_api.ranges[range_id]
        
        return {
            "message": f"Range {range_id} deleted successfully",
            "range_id": range_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting range: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete range: {str(e)}"
        )


@app.get("/metrics", response_model=Dict[str, Any])
async def get_system_metrics(token: str = Depends(verify_token)):
    """Get system-wide metrics and performance data"""
    try:
        total_attacks = sum(len(cr.completed_attacks) for cr in demo_api.ranges.values())
        total_detections = sum(len(cr.detection_events) for cr in demo_api.ranges.values())
        
        return {
            "system_metrics": demo_api.performance_metrics,
            "active_ranges": len(demo_api.ranges),
            "total_attacks_generated": total_attacks,
            "total_detections": total_detections,
            "detection_rate_global": total_detections / max(1, total_attacks),
            "recent_security_events": demo_api.security_events[-10:],  # Last 10 events
            "rate_limit_status": {
                "active_clients": len(demo_api.rate_limits),
                "blocked_requests": sum(1 for rl in demo_api.rate_limits.values() if rl["count"] >= 100)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system metrics: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Simple health check endpoint (no authentication required)"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc)
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error"
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting GAN-Cyber-Range-v2 API server")
    logger.info(f"Demo API Key: {demo_api_key}")
    logger.info("Use this key for authentication in requests")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False
    )