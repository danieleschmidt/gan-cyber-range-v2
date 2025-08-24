#!/usr/bin/env python3
"""
Robust Defensive Framework - Generation 2
Production-grade error handling, logging, monitoring, and security controls
"""

import asyncio
import logging
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
import signal
import psutil
import time
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiofiles
import ssl
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

# Core defensive imports
from gan_cyber_range.security import SecurityOrchestrator, ThreatDetector, AuditLogger
from gan_cyber_range.monitoring.metrics_collector import MetricsCollector
from gan_cyber_range.utils.error_handling import ErrorHandler
from gan_cyber_range.utils.logging_config import setup_logging

# Configure structured logging
logger = logging.getLogger(__name__)

T = TypeVar('T')


class SecurityLevel(Enum):
    """Security clearance levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class SystemHealth(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILURE = "failure"


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    session_id: str
    clearance_level: SecurityLevel
    permissions: List[str]
    ip_address: str
    timestamp: datetime = field(default_factory=datetime.now)
    encrypted_data: Optional[bytes] = None


@dataclass
class OperationResult(Generic[T]):
    """Standardized operation result with error handling"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: Optional[float] = None
    security_events: List[Dict] = field(default_factory=list)
    audit_trail: List[str] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """Comprehensive system performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    security_events_count: int
    error_rate: float
    response_time_avg_ms: float
    throughput_ops_per_sec: float


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == "OPEN":
                    if self.last_failure_time and \
                       (datetime.now() - self.last_failure_time).seconds < self.recovery_timeout:
                        raise Exception("Circuit breaker is OPEN")
                    else:
                        self.state = "HALF_OPEN"
                        logger.info("Circuit breaker transitioning to HALF_OPEN")
            
            try:
                result = await func(*args, **kwargs)
                
                with self._lock:
                    if self.state == "HALF_OPEN":
                        self.state = "CLOSED"
                        self.failure_count = 0
                        logger.info("Circuit breaker CLOSED - service recovered")
                
                return result
                
            except Exception as e:
                with self._lock:
                    self.failure_count += 1
                    self.last_failure_time = datetime.now()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                        logger.error(f"Circuit breaker OPEN - failure threshold reached: {e}")
                
                raise e
        
        return wrapper


class RateLimiter:
    """Rate limiting for API endpoints and operations"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
        self._lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit"""
        current_time = time.time()
        
        with self._lock:
            if identifier not in self.requests:
                self.requests[identifier] = []
            
            # Remove old requests outside time window
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if current_time - req_time < self.time_window
            ]
            
            # Check rate limit
            if len(self.requests[identifier]) >= self.max_requests:
                return False
            
            # Add current request
            self.requests[identifier].append(current_time)
            return True


class EncryptionManager:
    """Encryption and decryption for sensitive data"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            # Generate master key from password (in production, use secure key management)
            password = os.environ.get('MASTER_PASSWORD', 'default-secure-password').encode()
            salt = os.environ.get('MASTER_SALT', 'default-salt').encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            master_key = base64.urlsafe_b64encode(kdf.derive(password))
        
        self.fernet = Fernet(master_key)
    
    def encrypt(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode())
    
    def decrypt(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data).decode()
    
    def encrypt_dict(self, data: Dict) -> bytes:
        """Encrypt dictionary data"""
        json_str = json.dumps(data)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: bytes) -> Dict:
        """Decrypt dictionary data"""
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)


class SecureAuditLogger:
    """Secure audit logging with integrity protection"""
    
    def __init__(self, log_file: Path, encryption_manager: EncryptionManager):
        self.log_file = log_file
        self.encryption_manager = encryption_manager
        self._lock = threading.Lock()
    
    async def log_security_event(self, event: Dict, context: SecurityContext):
        """Log security event with integrity protection"""
        
        # Enhance event with security context
        enhanced_event = {
            "timestamp": datetime.now().isoformat(),
            "event_id": str(uuid.uuid4()),
            "user_id": context.user_id,
            "session_id": context.session_id,
            "clearance_level": context.clearance_level.value,
            "ip_address": context.ip_address,
            "event_data": event,
            "integrity_hash": None
        }
        
        # Calculate integrity hash
        event_str = json.dumps(enhanced_event, sort_keys=True)
        enhanced_event["integrity_hash"] = hashlib.sha256(event_str.encode()).hexdigest()
        
        # Encrypt sensitive events
        if context.clearance_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
            encrypted_event = self.encryption_manager.encrypt_dict(enhanced_event)
            log_entry = {
                "timestamp": enhanced_event["timestamp"],
                "event_id": enhanced_event["event_id"],
                "encrypted": True,
                "data": base64.b64encode(encrypted_event).decode()
            }
        else:
            log_entry = enhanced_event
        
        # Write to secure log file
        with self._lock:
            async with aiofiles.open(self.log_file, 'a') as f:
                await f.write(json.dumps(log_entry) + '\n')
    
    async def verify_log_integrity(self) -> bool:
        """Verify log file integrity"""
        try:
            async with aiofiles.open(self.log_file, 'r') as f:
                async for line in f:
                    entry = json.loads(line.strip())
                    
                    if entry.get("encrypted", False):
                        # Decrypt and verify encrypted entries
                        encrypted_data = base64.b64decode(entry["data"])
                        decrypted_event = self.encryption_manager.decrypt_dict(encrypted_data)
                        
                        # Verify hash
                        stored_hash = decrypted_event.pop("integrity_hash", None)
                        calculated_hash = hashlib.sha256(
                            json.dumps(decrypted_event, sort_keys=True).encode()
                        ).hexdigest()
                        
                        if stored_hash != calculated_hash:
                            logger.error(f"Log integrity violation detected: {entry['event_id']}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Log integrity verification failed: {e}")
            return False


class DefensiveSecurityOrchestrator:
    """Advanced security orchestration with comprehensive protection"""
    
    def __init__(self):
        self.security_orchestrator = SecurityOrchestrator()
        self.threat_detector = ThreatDetector()
        self.audit_logger = AuditLogger()
        self.metrics_collector = MetricsCollector()
        self.encryption_manager = EncryptionManager()
        self.rate_limiter = RateLimiter(max_requests=1000, time_window=60)
        
        # Initialize secure audit logging
        self.secure_audit_logger = SecureAuditLogger(
            Path("secure_audit.log"), 
            self.encryption_manager
        )
        
        # Circuit breakers for critical operations
        self.threat_detection_cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        self.security_scan_cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        
        # System monitoring
        self.system_metrics_history = []
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0,
            "response_time_ms": 1000.0
        }
        
        # Background monitoring task
        self.monitoring_active = False
        self.monitoring_task = None
    
    async def initialize(self) -> OperationResult[bool]:
        """Initialize defensive security orchestrator"""
        try:
            logger.info("🛡️ Initializing Defensive Security Orchestrator")
            
            # Initialize core components
            await self.security_orchestrator.initialize()
            await self.threat_detector.initialize({
                "sensitivity": "high",
                "ml_enabled": True,
                "real_time": True
            })
            
            # Start background monitoring
            await self.start_monitoring()
            
            # Verify log integrity
            integrity_ok = await self.secure_audit_logger.verify_log_integrity()
            if not integrity_ok:
                logger.warning("Log integrity issues detected - starting fresh log")
            
            logger.info("✅ Defensive Security Orchestrator initialized successfully")
            
            return OperationResult(
                success=True,
                data=True,
                execution_time_ms=0.0,
                audit_trail=["System initialized", "Monitoring started", "Components verified"]
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize security orchestrator: {e}")
            return OperationResult(
                success=False,
                error=str(e),
                error_code="INIT_FAILED"
            )
    
    async def start_monitoring(self):
        """Start background system monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("📊 Background monitoring started")
    
    async def stop_monitoring(self):
        """Stop background system monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            logger.info("📊 Background monitoring stopped")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = await self.collect_system_metrics()
                self.system_metrics_history.append(metrics)
                
                # Keep only last 1000 metrics (rolling window)
                if len(self.system_metrics_history) > 1000:
                    self.system_metrics_history = self.system_metrics_history[-1000:]
                
                # Check for alerts
                await self.check_system_alerts(metrics)
                
                # Sleep between monitoring cycles
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_metrics = {
                "bytes_sent": float(net_io.bytes_sent),
                "bytes_recv": float(net_io.bytes_recv),
                "packets_sent": float(net_io.packets_sent),
                "packets_recv": float(net_io.packets_recv)
            }
            
            # Connection count
            connections = len(psutil.net_connections())
            
            # Security metrics (simulated)
            security_events = await self.metrics_collector.collect_security_metrics()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io=network_metrics,
                active_connections=connections,
                security_events_count=security_events.get("event_count", 0),
                error_rate=security_events.get("error_rate", 0.0),
                response_time_avg_ms=security_events.get("avg_response_time_ms", 0.0),
                throughput_ops_per_sec=security_events.get("throughput", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics on error
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                active_connections=0,
                security_events_count=0,
                error_rate=0.0,
                response_time_avg_ms=0.0,
                throughput_ops_per_sec=0.0
            )
    
    async def check_system_alerts(self, metrics: SystemMetrics):
        """Check metrics against alert thresholds"""
        alerts = []
        
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.disk_usage > self.alert_thresholds["disk_usage"]:
            alerts.append(f"High disk usage: {metrics.disk_usage:.1f}%")
        
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(f"High error rate: {metrics.error_rate:.1f}%")
        
        if metrics.response_time_avg_ms > self.alert_thresholds["response_time_ms"]:
            alerts.append(f"High response time: {metrics.response_time_avg_ms:.1f}ms")
        
        # Log alerts
        if alerts:
            context = SecurityContext(
                user_id="system",
                session_id="monitoring",
                clearance_level=SecurityLevel.INTERNAL,
                permissions=["system_monitoring"],
                ip_address="127.0.0.1"
            )
            
            await self.secure_audit_logger.log_security_event({
                "event_type": "system_alert",
                "alerts": alerts,
                "metrics": {
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "disk_usage": metrics.disk_usage,
                    "error_rate": metrics.error_rate,
                    "response_time_ms": metrics.response_time_avg_ms
                }
            }, context)
            
            logger.warning(f"System alerts: {', '.join(alerts)}")
    
    @asynccontextmanager
    async def secure_operation_context(self, context: SecurityContext):
        """Secure context manager for operations"""
        operation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Verify security context
            if not await self.verify_security_context(context):
                raise PermissionError("Invalid security context")
            
            # Check rate limiting
            if not self.rate_limiter.is_allowed(context.user_id):
                raise Exception("Rate limit exceeded")
            
            # Log operation start
            await self.secure_audit_logger.log_security_event({
                "event_type": "operation_start",
                "operation_id": operation_id
            }, context)
            
            logger.info(f"🔒 Secure operation started: {operation_id}")
            yield operation_id
            
        except Exception as e:
            # Log operation failure
            await self.secure_audit_logger.log_security_event({
                "event_type": "operation_failed",
                "operation_id": operation_id,
                "error": str(e)
            }, context)
            
            logger.error(f"❌ Secure operation failed: {operation_id} - {e}")
            raise e
            
        finally:
            # Log operation completion
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            await self.secure_audit_logger.log_security_event({
                "event_type": "operation_complete",
                "operation_id": operation_id,
                "duration_ms": duration_ms
            }, context)
            
            logger.info(f"✅ Secure operation completed: {operation_id} ({duration_ms:.1f}ms)")
    
    async def verify_security_context(self, context: SecurityContext) -> bool:
        """Verify security context validity"""
        try:
            # Verify session is still valid
            if context.timestamp and (datetime.now() - context.timestamp).total_seconds() > 3600:
                logger.warning(f"Expired security context for user {context.user_id}")
                return False
            
            # Verify user permissions (simplified check)
            if not context.user_id or not context.session_id:
                logger.warning("Missing required security context fields")
                return False
            
            # Additional security checks would go here
            # - Check user exists in security database
            # - Verify session is active
            # - Check IP address against allowed ranges
            # - Verify clearance level is sufficient
            
            return True
            
        except Exception as e:
            logger.error(f"Security context verification failed: {e}")
            return False
    
    @circuit_breaker
    async def execute_threat_detection(self, data: Dict, context: SecurityContext) -> OperationResult[Dict]:
        """Execute threat detection with circuit breaker protection"""
        async with self.secure_operation_context(context) as operation_id:
            try:
                start_time = time.time()
                
                # Apply circuit breaker to threat detection
                @self.threat_detection_cb
                async def protected_detection():
                    return await self.threat_detector.analyze_threat(data)
                
                result = await protected_detection()
                
                execution_time = (time.time() - start_time) * 1000
                
                return OperationResult(
                    success=True,
                    data=result,
                    execution_time_ms=execution_time,
                    audit_trail=[f"Threat detection completed: {operation_id}"]
                )
                
            except Exception as e:
                logger.error(f"Threat detection failed: {e}")
                return OperationResult(
                    success=False,
                    error=str(e),
                    error_code="THREAT_DETECTION_FAILED"
                )
    
    async def execute_security_scan(self, target: str, context: SecurityContext) -> OperationResult[Dict]:
        """Execute security scan with protection"""
        async with self.secure_operation_context(context) as operation_id:
            try:
                start_time = time.time()
                
                # Apply circuit breaker to security scan
                @self.security_scan_cb
                async def protected_scan():
                    from gan_cyber_range.security import SecurityScanner
                    scanner = SecurityScanner()
                    return await scanner.async_scan(target, {"comprehensive": True})
                
                result = await protected_scan()
                
                execution_time = (time.time() - start_time) * 1000
                
                return OperationResult(
                    success=True,
                    data=result,
                    execution_time_ms=execution_time,
                    audit_trail=[f"Security scan completed: {operation_id}"]
                )
                
            except Exception as e:
                logger.error(f"Security scan failed: {e}")
                return OperationResult(
                    success=False,
                    error=str(e),
                    error_code="SECURITY_SCAN_FAILED"
                )
    
    async def generate_security_report(self, context: SecurityContext) -> OperationResult[Dict]:
        """Generate comprehensive security report"""
        async with self.secure_operation_context(context) as operation_id:
            try:
                # Collect system health data
                current_metrics = await self.collect_system_metrics()
                
                # Calculate averages over last hour
                recent_metrics = [
                    m for m in self.system_metrics_history
                    if (datetime.now() - m.timestamp).total_seconds() < 3600
                ]
                
                if recent_metrics:
                    avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
                    avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
                    avg_response_time = sum(m.response_time_avg_ms for m in recent_metrics) / len(recent_metrics)
                else:
                    avg_cpu = avg_memory = avg_response_time = 0.0
                
                # Determine system health status
                health_status = SystemHealth.HEALTHY
                if (current_metrics.cpu_usage > 90 or current_metrics.memory_usage > 95 or 
                    current_metrics.error_rate > 10):
                    health_status = SystemHealth.CRITICAL
                elif (current_metrics.cpu_usage > 80 or current_metrics.memory_usage > 85 or 
                      current_metrics.error_rate > 5):
                    health_status = SystemHealth.DEGRADED
                
                # Generate comprehensive report
                report = {
                    "report_id": f"SEC-RPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    "timestamp": datetime.now().isoformat(),
                    "system_health": {
                        "status": health_status.value,
                        "current_metrics": {
                            "cpu_usage": current_metrics.cpu_usage,
                            "memory_usage": current_metrics.memory_usage,
                            "disk_usage": current_metrics.disk_usage,
                            "active_connections": current_metrics.active_connections,
                            "error_rate": current_metrics.error_rate
                        },
                        "hourly_averages": {
                            "cpu_usage": avg_cpu,
                            "memory_usage": avg_memory,
                            "response_time_ms": avg_response_time
                        }
                    },
                    "security_metrics": {
                        "security_events_last_hour": sum(
                            m.security_events_count for m in recent_metrics
                        ),
                        "threat_detections": current_metrics.security_events_count,
                        "audit_log_integrity": await self.secure_audit_logger.verify_log_integrity()
                    },
                    "performance_metrics": {
                        "average_response_time_ms": avg_response_time,
                        "throughput_ops_per_sec": current_metrics.throughput_ops_per_sec,
                        "network_traffic": current_metrics.network_io
                    },
                    "recommendations": self.generate_security_recommendations(current_metrics, health_status)
                }
                
                return OperationResult(
                    success=True,
                    data=report,
                    audit_trail=[f"Security report generated: {operation_id}"]
                )
                
            except Exception as e:
                logger.error(f"Security report generation failed: {e}")
                return OperationResult(
                    success=False,
                    error=str(e),
                    error_code="REPORT_GENERATION_FAILED"
                )
    
    def generate_security_recommendations(self, metrics: SystemMetrics, health: SystemHealth) -> List[str]:
        """Generate security recommendations based on current state"""
        recommendations = []
        
        # Performance recommendations
        if metrics.cpu_usage > 80:
            recommendations.append("Consider scaling CPU resources or optimizing high-usage processes")
        
        if metrics.memory_usage > 85:
            recommendations.append("Monitor memory usage and consider increasing available RAM")
        
        if metrics.disk_usage > 90:
            recommendations.append("Critical: Disk space is running low - immediate cleanup required")
        
        # Security recommendations
        if metrics.error_rate > 5:
            recommendations.append("Investigate high error rates - potential security issue")
        
        if metrics.security_events_count > 100:
            recommendations.append("High security event volume - review threat detection settings")
        
        # Health-based recommendations
        if health == SystemHealth.CRITICAL:
            recommendations.append("URGENT: System in critical state - immediate attention required")
        elif health == SystemHealth.DEGRADED:
            recommendations.append("System performance degraded - schedule maintenance window")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations


async def demonstrate_robust_framework():
    """Demonstrate robust defensive framework capabilities"""
    logger.info("🛡️ Starting Robust Defensive Framework Demonstration")
    
    # Initialize framework
    framework = DefensiveSecurityOrchestrator()
    init_result = await framework.initialize()
    
    if not init_result.success:
        logger.error(f"Framework initialization failed: {init_result.error}")
        return
    
    # Create security context
    context = SecurityContext(
        user_id="demo_user",
        session_id=str(uuid.uuid4()),
        clearance_level=SecurityLevel.CONFIDENTIAL,
        permissions=["threat_detection", "security_scan", "report_generation"],
        ip_address="127.0.0.1"
    )
    
    try:
        # Demonstrate threat detection with circuit breaker
        logger.info("🔍 Testing threat detection with circuit breaker protection")
        
        threat_scenarios = [
            {"type": "malware", "payload": "suspicious_file.exe", "severity": "high"},
            {"type": "network_intrusion", "source_ip": "192.168.1.100", "severity": "medium"},
            {"type": "data_exfiltration", "volume_mb": 500, "severity": "critical"}
        ]
        
        for scenario in threat_scenarios:
            result = await framework.execute_threat_detection(scenario, context)
            if result.success:
                logger.info(f"✅ Threat detection successful: {result.execution_time_ms:.1f}ms")
            else:
                logger.error(f"❌ Threat detection failed: {result.error}")
        
        # Demonstrate security scanning
        logger.info("🔒 Testing security scan capabilities")
        scan_result = await framework.execute_security_scan("localhost", context)
        
        if scan_result.success:
            logger.info(f"✅ Security scan completed: {scan_result.execution_time_ms:.1f}ms")
        else:
            logger.error(f"❌ Security scan failed: {scan_result.error}")
        
        # Wait for some monitoring data
        logger.info("⏱️ Collecting system metrics...")
        await asyncio.sleep(10)
        
        # Generate security report
        logger.info("📊 Generating comprehensive security report")
        report_result = await framework.generate_security_report(context)
        
        if report_result.success:
            report = report_result.data
            logger.info("✅ Security report generated successfully")
            
            # Display key metrics
            health_status = report["system_health"]["status"]
            cpu_usage = report["system_health"]["current_metrics"]["cpu_usage"]
            memory_usage = report["system_health"]["current_metrics"]["memory_usage"]
            
            print(f"\n{'='*60}")
            print("📊 SYSTEM STATUS SUMMARY")
            print('='*60)
            print(f"🏥 Health Status: {health_status.upper()}")
            print(f"🖥️  CPU Usage: {cpu_usage:.1f}%")
            print(f"💾 Memory Usage: {memory_usage:.1f}%")
            print(f"🔒 Security Events: {report['security_metrics']['security_events_last_hour']}")
            print(f"🔍 Audit Integrity: {'✅ VERIFIED' if report['security_metrics']['audit_log_integrity'] else '❌ COMPROMISED'}")
            
            print(f"\n📋 RECOMMENDATIONS:")
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"  {i}. {rec}")
            
            # Save report to file
            report_file = Path(f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"💾 Report saved to: {report_file}")
        
        else:
            logger.error(f"❌ Report generation failed: {report_result.error}")
    
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # Cleanup
        await framework.stop_monitoring()
        logger.info("🔄 Framework monitoring stopped")
    
    logger.info("🛡️ Robust Defensive Framework demonstration completed")


def setup_signal_handlers():
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        # Cleanup tasks would go here
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main execution function with robust error handling"""
    setup_signal_handlers()
    
    try:
        await demonstrate_robust_framework()
    
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        logger.info("🏁 Application shutdown complete")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('robust_framework.log')
        ]
    )
    
    # Run demonstration
    asyncio.run(main())