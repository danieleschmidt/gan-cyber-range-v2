#!/usr/bin/env python3
"""
Comprehensive monitoring and health checking for defensive cybersecurity systems

This module provides robust monitoring, alerting, and health validation
for defensive security training environments.
"""

import time
import json
import logging
import threading
import psutil
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import hmac

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/defensive_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    FAILING = "failing"

class AlertSeverity(Enum):
    """Alert severity levels for monitoring"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """System performance and health metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    uptime_seconds: float
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for serialization"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'status': self._determine_health_status().value
        }
    
    def _determine_health_status(self) -> HealthStatus:
        """Determine overall system health status"""
        if self.cpu_percent > 90 or self.memory_percent > 90:
            return HealthStatus.CRITICAL
        elif self.cpu_percent > 80 or self.memory_percent > 80:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

@dataclass
class SecurityEvent:
    """Security monitoring event"""
    event_id: str
    timestamp: datetime
    event_type: str
    severity: AlertSeverity
    source: str
    description: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with timestamp formatting"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value
        }

class DefensiveHealthChecker:
    """Comprehensive health checking for defensive systems"""
    
    def __init__(self):
        self.checks_registry = {}
        self.last_check_results = {}
        self.health_thresholds = {
            'cpu_threshold': 85.0,
            'memory_threshold': 85.0,
            'disk_threshold': 90.0,
            'response_time_threshold': 5.0
        }
        
        # Register default health checks
        self._register_default_checks()
        
    def _register_default_checks(self):
        """Register standard defensive system health checks"""
        
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("network_connectivity", self._check_network_connectivity)
        self.register_check("process_health", self._check_process_health)
        self.register_check("log_integrity", self._check_log_integrity)
        
        logger.info("Default defensive health checks registered")
    
    def register_check(self, check_name: str, check_function: Callable) -> None:
        """Register a custom health check function"""
        self.checks_registry[check_name] = check_function
        logger.info(f"Registered health check: {check_name}")
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization"""
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        status = HealthStatus.HEALTHY
        if cpu_percent > self.health_thresholds['cpu_threshold']:
            status = HealthStatus.CRITICAL
        elif memory.percent > self.health_thresholds['memory_threshold']:
            status = HealthStatus.CRITICAL
        elif cpu_percent > 70 or memory.percent > 70:
            status = HealthStatus.DEGRADED
            
        return {
            "status": status.value,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "details": f"CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%"
        }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        status = HealthStatus.HEALTHY
        if disk_percent > self.health_thresholds['disk_threshold']:
            status = HealthStatus.CRITICAL
        elif disk_percent > 75:
            status = HealthStatus.DEGRADED
            
        return {
            "status": status.value,
            "disk_percent": disk_percent,
            "free_gb": disk.free / (1024**3),
            "details": f"Disk usage: {disk_percent:.1f}%"
        }
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity and response times"""
        
        try:
            start_time = time.time()
            sock = socket.create_connection(("8.8.8.8", 53), timeout=5)
            sock.close()
            response_time = time.time() - start_time
            
            status = HealthStatus.HEALTHY
            if response_time > self.health_thresholds['response_time_threshold']:
                status = HealthStatus.DEGRADED
                
            return {
                "status": status.value,
                "response_time": response_time,
                "details": f"Network response: {response_time:.2f}s"
            }
        
        except Exception as e:
            return {
                "status": HealthStatus.FAILING.value,
                "error": str(e),
                "details": "Network connectivity failed"
            }
    
    def _check_process_health(self) -> Dict[str, Any]:
        """Check critical process health"""
        
        try:
            process_count = len(psutil.pids())
            load_avg = psutil.getloadavg()
            
            status = HealthStatus.HEALTHY
            if load_avg[0] > psutil.cpu_count() * 2:
                status = HealthStatus.DEGRADED
            
            return {
                "status": status.value,
                "process_count": process_count,
                "load_average": load_avg,
                "details": f"Processes: {process_count}, Load: {load_avg[0]:.2f}"
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.FAILING.value,
                "error": str(e),
                "details": "Process health check failed"
            }
    
    def _check_log_integrity(self) -> Dict[str, Any]:
        """Check log file integrity and accessibility"""
        
        try:
            log_dir = Path("logs")
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if we can write to log directory
            test_file = log_dir / "health_check.tmp"
            with open(test_file, 'w') as f:
                f.write("health_check_test")
            
            test_file.unlink()  # Clean up
            
            return {
                "status": HealthStatus.HEALTHY.value,
                "details": "Log integrity verified"
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.FAILING.value,
                "error": str(e),
                "details": "Log integrity check failed"
            }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": HealthStatus.HEALTHY.value,
            "checks": {}
        }
        
        failing_checks = 0
        critical_checks = 0
        
        for check_name, check_function in self.checks_registry.items():
            try:
                check_result = check_function()
                results["checks"][check_name] = check_result
                
                # Track failing and critical checks
                if check_result["status"] == HealthStatus.FAILING.value:
                    failing_checks += 1
                elif check_result["status"] == HealthStatus.CRITICAL.value:
                    critical_checks += 1
                    
            except Exception as e:
                logger.error(f"Health check '{check_name}' failed: {e}")
                results["checks"][check_name] = {
                    "status": HealthStatus.FAILING.value,
                    "error": str(e),
                    "details": f"Check execution failed: {e}"
                }
                failing_checks += 1
        
        # Determine overall status
        if failing_checks > 0:
            results["overall_status"] = HealthStatus.FAILING.value
        elif critical_checks > 0:
            results["overall_status"] = HealthStatus.CRITICAL.value
        elif any(check.get("status") == HealthStatus.DEGRADED.value 
                for check in results["checks"].values()):
            results["overall_status"] = HealthStatus.DEGRADED.value
        
        self.last_check_results = results
        return results

class DefensiveSecurityMonitor:
    """Real-time security monitoring for defensive systems"""
    
    def __init__(self):
        self.is_monitoring = False
        self.monitor_thread = None
        self.alert_handlers = []
        self.security_events = []
        self.metrics_history = []
        
        # Security monitoring configuration
        self.monitoring_config = {
            'check_interval': 30,  # seconds
            'max_events_history': 1000,
            'max_metrics_history': 100,
            'enable_file_monitoring': True,
            'enable_network_monitoring': True
        }
        
        self.health_checker = DefensiveHealthChecker()
        
    def add_alert_handler(self, handler: Callable[[SecurityEvent], None]):
        """Add custom alert handler for security events"""
        self.alert_handlers.append(handler)
        logger.info("Added security alert handler")
    
    def _generate_security_event(self, event_type: str, severity: AlertSeverity,
                               description: str, metadata: Dict = None) -> SecurityEvent:
        """Generate a new security event"""
        
        event = SecurityEvent(
            event_id=f"SEC-{int(time.time())}-{hash(description) % 10000:04d}",
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            source="defensive_monitor",
            description=description,
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        return event
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # Process count and load average
            process_count = len(psutil.pids())
            load_avg = list(psutil.getloadavg())
            
            # System uptime
            uptime = time.time() - psutil.boot_time()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk_percent,
                network_io=network_io,
                process_count=process_count,
                load_average=load_avg,
                uptime_seconds=uptime
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.monitoring_config['max_metrics_history']:
                self.metrics_history.pop(0)
            
            # Check for alerts based on metrics
            self._check_metrics_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            raise
    
    def _check_metrics_alerts(self, metrics: SystemMetrics):
        """Check metrics for alert conditions"""
        
        # CPU usage alert
        if metrics.cpu_percent > 90:
            self._generate_security_event(
                "system_alert",
                AlertSeverity.CRITICAL,
                f"Critical CPU usage: {metrics.cpu_percent:.1f}%",
                {"cpu_percent": metrics.cpu_percent}
            )
        elif metrics.cpu_percent > 80:
            self._generate_security_event(
                "system_alert", 
                AlertSeverity.WARNING,
                f"High CPU usage: {metrics.cpu_percent:.1f}%",
                {"cpu_percent": metrics.cpu_percent}
            )
        
        # Memory usage alert
        if metrics.memory_percent > 90:
            self._generate_security_event(
                "system_alert",
                AlertSeverity.CRITICAL,
                f"Critical memory usage: {metrics.memory_percent:.1f}%",
                {"memory_percent": metrics.memory_percent}
            )
        
        # Disk usage alert
        if metrics.disk_percent > 95:
            self._generate_security_event(
                "system_alert",
                AlertSeverity.CRITICAL,
                f"Critical disk usage: {metrics.disk_percent:.1f}%",
                {"disk_percent": metrics.disk_percent}
            )
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        logger.info("Defensive security monitoring started")
        
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Run health checks periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10th iteration
                    health_results = self.health_checker.run_all_checks()
                    
                    # Generate alert if health is failing
                    if health_results["overall_status"] in [HealthStatus.FAILING.value, 
                                                          HealthStatus.CRITICAL.value]:
                        self._generate_security_event(
                            "health_check_alert",
                            AlertSeverity.ERROR,
                            f"System health check failed: {health_results['overall_status']}",
                            health_results
                        )
                
                logger.debug(f"Metrics collected - CPU: {metrics.cpu_percent:.1f}%, "
                           f"Memory: {metrics.memory_percent:.1f}%")
                
                # Sleep for configured interval
                time.sleep(self.monitoring_config['check_interval'])
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Short sleep on error
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Defensive security monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Defensive security monitoring stopped")
    
    def get_current_status(self) -> Dict:
        """Get current system status and metrics"""
        
        current_metrics = self._collect_system_metrics()
        health_results = self.health_checker.run_all_checks()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": current_metrics.to_dict(),
            "health_checks": health_results,
            "recent_events": [event.to_dict() for event in self.security_events[-10:]],
            "monitoring_active": self.is_monitoring
        }
    
    def export_monitoring_data(self, filepath: str):
        """Export monitoring data for analysis"""
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "monitoring_config": self.monitoring_config,
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "security_events": [e.to_dict() for e in self.security_events],
            "last_health_check": self.health_checker.last_check_results
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Monitoring data exported to: {filepath}")

def create_default_alert_handler():
    """Create default alert handler for security events"""
    
    def default_handler(event: SecurityEvent):
        """Default security event handler"""
        
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️", 
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        emoji = severity_emoji.get(event.severity, "🔍")
        
        logger.warning(f"{emoji} SECURITY ALERT: {event.description}")
        logger.warning(f"Event ID: {event.event_id}, Type: {event.event_type}, "
                      f"Severity: {event.severity.value}")
        
        # Save critical events to separate log
        if event.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            alerts_dir = Path("logs")
            alerts_dir.mkdir(exist_ok=True)
            
            alert_file = alerts_dir / "security_alerts.json"
            
            # Append to alerts file
            alerts = []
            if alert_file.exists():
                try:
                    with open(alert_file, 'r') as f:
                        alerts = json.load(f)
                except:
                    pass
            
            alerts.append(event.to_dict())
            
            with open(alert_file, 'w') as f:
                json.dump(alerts, f, indent=2)
    
    return default_handler

def main():
    """Main defensive monitoring demo"""
    
    print("🛡️  Defensive Security Monitoring System")
    print("=" * 50)
    
    # Create monitoring system
    monitor = DefensiveSecurityMonitor()
    
    # Add default alert handler
    default_handler = create_default_alert_handler()
    monitor.add_alert_handler(default_handler)
    
    # Get initial status
    initial_status = monitor.get_current_status()
    
    print("\n📊 INITIAL SYSTEM STATUS")
    print("-" * 30)
    print(f"CPU Usage: {initial_status['system_metrics']['cpu_percent']:.1f}%")
    print(f"Memory Usage: {initial_status['system_metrics']['memory_percent']:.1f}%")
    print(f"Disk Usage: {initial_status['system_metrics']['disk_percent']:.1f}%")
    print(f"Process Count: {initial_status['system_metrics']['process_count']}")
    print(f"System Status: {initial_status['system_metrics']['status']}")
    
    print(f"\n🏥 HEALTH CHECK RESULTS")
    print("-" * 30)
    health_checks = initial_status['health_checks']
    print(f"Overall Status: {health_checks['overall_status']}")
    
    for check_name, result in health_checks['checks'].items():
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'critical': '🚨',
            'failing': '❌'
        }
        emoji = status_emoji.get(result['status'], '🔍')
        print(f"  {emoji} {check_name}: {result.get('details', result['status'])}")
    
    # Demonstrate monitoring for a short period
    print(f"\n🔄 Starting 30-second monitoring demonstration...")
    monitor.start_monitoring()
    
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    
    monitor.stop_monitoring()
    
    # Export results
    export_file = f"logs/monitoring_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    monitor.export_monitoring_data(export_file)
    
    print(f"\n💾 Monitoring data exported to: {export_file}")
    print("✅ Defensive security monitoring demonstration completed")

if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    main()