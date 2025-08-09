"""
Authentication and Authorization Module
JWT-based authentication for API access
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uuid
import os
from functools import wraps

from ..utils.logging_config import get_logger
from ..utils.security import SecurityManager

logger = get_logger(__name__)
security_scheme = HTTPBearer()

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Mock user database (replace with real database in production)
USERS_DB: Dict[str, Dict[str, Any]] = {}


class AuthenticationError(Exception):
    """Authentication related errors"""
    pass


class AuthorizationError(Exception):
    """Authorization related errors"""
    pass


class UserManager:
    """User management functionality"""
    
    def __init__(self):
        self.security_manager = SecurityManager()
    
    def create_user(self, username: str, email: str, password: str, 
                   full_name: Optional[str] = None, organization: Optional[str] = None,
                   role: str = "user") -> Dict[str, Any]:
        """Create a new user"""
        if username in USERS_DB:
            raise AuthenticationError("Username already exists")
        
        # Validate inputs
        if not self._validate_username(username):
            raise AuthenticationError("Invalid username format")
        if not self._validate_email(email):
            raise AuthenticationError("Invalid email format")
        if not self._validate_password(password):
            raise AuthenticationError("Password does not meet requirements")
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        user = {
            "id": str(uuid.uuid4()),
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "full_name": full_name,
            "organization": organization,
            "role": role,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "last_login": None
        }
        
        USERS_DB[username] = user
        logger.info(f"User created: {username} ({role})")
        
        # Return user without password hash
        user_safe = user.copy()
        del user_safe["password_hash"]
        return user_safe
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials"""
        user = USERS_DB.get(username)
        if not user or not user.get("is_active", True):
            return None
        
        if bcrypt.checkpw(password.encode('utf-8'), user["password_hash"].encode('utf-8')):
            # Update last login
            user["last_login"] = datetime.utcnow()
            
            logger.info(f"User authenticated: {username}")
            
            # Return user without password hash
            user_safe = user.copy()
            del user_safe["password_hash"]
            return user_safe
        
        return None
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        user = USERS_DB.get(username)
        if user and user.get("is_active", True):
            user_safe = user.copy()
            del user_safe["password_hash"]
            return user_safe
        return None
    
    def _validate_username(self, username: str) -> bool:
        """Validate username format"""
        if not username or len(username) < 3 or len(username) > 50:
            return False
        return username.isalnum() or '_' in username or '-' in username
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[^@]+@[^@]+\.[^@]+$'
        return bool(re.match(pattern, email))
    
    def _validate_password(self, password: str) -> bool:
        """Validate password requirements"""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        return has_upper and has_lower and has_digit


user_manager = UserManager()


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access_token"
    })
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    logger.debug(f"Access token created for user: {data.get('sub')}")
    
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        if payload.get("type") != "access_token":
            raise AuthenticationError("Invalid token type")
        
        username = payload.get("sub")
        if username is None:
            raise AuthenticationError("Invalid token payload")
        
        return payload
    
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.JWTError:
        raise AuthenticationError("Invalid token")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> Dict[str, Any]:
    """Get current authenticated user"""
    try:
        payload = verify_token(credentials.credentials)
        username = payload.get("sub")
        
        user = user_manager.get_user(username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
    
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(required_roles: list):
    """Decorator to require specific roles"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current user from kwargs (should be injected by FastAPI)
            current_user = None
            for key, value in kwargs.items():
                if isinstance(value, dict) and 'role' in value:
                    current_user = value
                    break
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if current_user.get('role') not in required_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required role: {' or '.join(required_roles)}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require admin role"""
    if current_user.get('role') != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    return current_user


def require_researcher(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to require researcher role or higher"""
    allowed_roles = ['admin', 'researcher', 'educator']
    if current_user.get('role') not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Researcher role or higher required"
        )
    return current_user


class RateLimiter:
    """Simple rate limiting for API endpoints"""
    
    def __init__(self):
        self._requests = {}
    
    def is_allowed(self, user_id: str, endpoint: str, limit: int = 100, 
                   window_minutes: int = 60) -> bool:
        """Check if request is within rate limit"""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=window_minutes)
        
        key = f"{user_id}:{endpoint}"
        if key not in self._requests:
            self._requests[key] = []
        
        # Clean old requests
        self._requests[key] = [
            req_time for req_time in self._requests[key]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self._requests[key]) >= limit:
            return False
        
        # Add current request
        self._requests[key].append(now)
        return True


rate_limiter = RateLimiter()


def check_rate_limit(limit: int = 100, window_minutes: int = 60):
    """Dependency to check rate limits"""
    def dependency(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_id = current_user.get('id', 'unknown')
        endpoint = "api_call"  # Can be customized per endpoint
        
        if not rate_limiter.is_allowed(user_id, endpoint, limit, window_minutes):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        return current_user
    
    return dependency


# Initialize default admin user if not exists
def initialize_default_users():
    """Initialize default users for development"""
    if not USERS_DB:
        try:
            # Create default admin user
            user_manager.create_user(
                username="admin",
                email="admin@terragonlabs.com",
                password="AdminPass123!",
                full_name="System Administrator",
                organization="Terragon Labs",
                role="admin"
            )
            
            # Create default researcher user
            user_manager.create_user(
                username="researcher",
                email="researcher@terragonlabs.com", 
                password="ResearchPass123!",
                full_name="Research User",
                organization="Terragon Labs",
                role="researcher"
            )
            
            logger.info("Default users initialized")
            
        except AuthenticationError as e:
            logger.warning(f"Could not initialize default users: {e}")


# Initialize on import
initialize_default_users()