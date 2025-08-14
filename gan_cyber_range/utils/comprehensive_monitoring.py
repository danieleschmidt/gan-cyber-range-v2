"""
Comprehensive monitoring and observability system for GAN Cyber Range.

This module provides advanced monitoring, metrics collection, alerting,
and observability features for all components of the cyber range.
"""

import logging
import asyncio
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import json
import uuid
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Alert definition and state"""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    timestamp: datetime
    active: bool = True
    acknowledged: bool = False
    count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    description: str
    check_function: Callable[[], bool]
    interval_seconds: int = 60
    timeout_seconds: int = 10
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    failure_count: int = 0
    max_failures: int = 3


class MetricsCollector:
    """Advanced metrics collection and storage"""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.retention_hours = retention_hours
        self.labels_index: Dict[str, set] = defaultdict(set)
        self._lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self.cleanup_thread.start()
    
    def record_metric(self, name: str, value: Union[int, float], 
                     labels: Dict[str, str] = None, metric_type: MetricType = MetricType.GAUGE) -> None:
        """Record a metric data point"""
        labels = labels or {}
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels,
            metric_type=metric_type
        )
        
        with self._lock:
            metric_key = self._create_metric_key(name, labels)
            self.metrics[metric_key].append(metric)
            
            # Update labels index
            for label_key, label_value in labels.items():
                self.labels_index[f"{name}:{label_key}"].add(label_value)
    
    def increment_counter(self, name: str, amount: Union[int, float] = 1, 
                         labels: Dict[str, str] = None) -> None:
        """Increment a counter metric"""
        labels = labels or {}
        metric_key = self._create_metric_key(name, labels)
        
        with self._lock:
            if metric_key in self.metrics and self.metrics[metric_key]:
                # Get the last value and increment
                last_metric = self.metrics[metric_key][-1]
                new_value = last_metric.value + amount
            else:
                new_value = amount
            
            self.record_metric(name, new_value, labels, MetricType.COUNTER)
    
    def record_timer(self, name: str, duration_seconds: float, 
                    labels: Dict[str, str] = None) -> None:
        """Record a timing metric"""
        self.record_metric(name, duration_seconds, labels, MetricType.TIMER)
    
    def get_metrics(self, name: str, labels: Dict[str, str] = None, 
                   since: Optional[datetime] = None) -> List[Metric]:
        """Retrieve metrics matching criteria"""
        metric_key = self._create_metric_key(name, labels or {})
        
        with self._lock:
            if metric_key not in self.metrics:
                return []
            
            metrics = list(self.metrics[metric_key])
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return metrics
    
    def get_latest_metric(self, name: str, labels: Dict[str, str] = None) -> Optional[Metric]:
        """Get the latest metric value"""
        metrics = self.get_metrics(name, labels)
        return metrics[-1] if metrics else None
    
    def get_metric_statistics(self, name: str, labels: Dict[str, str] = None, 
                             since: Optional[datetime] = None) -> Dict[str, float]:
        """Calculate statistics for metrics"""
        metrics = self.get_metrics(name, labels, since)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'latest': values[-1],
            'first': values[0]
        }
    
    def query_metrics(self, query: str) -> List[Metric]:
        """Query metrics using a simple query language"""
        # Simple implementation - can be extended
        # Format: "metric_name{label1=value1,label2=value2}"
        
        parts = query.split('{', 1)
        metric_name = parts[0].strip()
        
        labels = {}
        if len(parts) > 1:
            labels_str = parts[1].rstrip('}')
            for label_pair in labels_str.split(','):
                if '=' in label_pair:
                    key, value = label_pair.split('=', 1)
                    labels[key.strip()] = value.strip()
        
        return self.get_metrics(metric_name, labels)
    
    def _create_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for metric storage"""
        if not labels:
            return name
        
        label_str = ','.join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics beyond retention period"""
        while True:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
                
                with self._lock:
                    for metric_key in list(self.metrics.keys()):
                        metric_deque = self.metrics[metric_key]
                        
                        # Remove old metrics
                        while metric_deque and metric_deque[0].timestamp < cutoff_time:
                            metric_deque.popleft()
                        
                        # Remove empty deques
                        if not metric_deque:
                            del self.metrics[metric_key]
                
                time.sleep(3600)  # Clean up every hour
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
                time.sleep(300)  # Wait 5 minutes on error


class AlertManager:
    """Alert management and notification system"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        
        # Start alert evaluation thread
        self.evaluation_thread = threading.Thread(target=self._evaluate_alerts, daemon=True)
        self.evaluation_thread.start()
    
    def add_alert_rule(self, name: str, description: str, metric_query: str, 
                      condition: str, threshold: float, severity: AlertSeverity,
                      evaluation_interval: int = 60) -> None:
        """Add an alert rule"""
        rule = {
            'name': name,
            'description': description,
            'metric_query': metric_query,
            'condition': condition,  # 'gt', 'lt', 'eq', 'gte', 'lte'
            'threshold': threshold,
            'severity': severity,
            'evaluation_interval': evaluation_interval,
            'last_evaluation': None
        }
        
        with self._lock:
            self.alert_rules.append(rule)
        
        logger.info(f"Added alert rule: {name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler for alerts"""
        self.notification_handlers.append(handler)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].acknowledged = True
                self.alerts[alert_id].metadata['acknowledged_by'] = acknowledged_by
                self.alerts[alert_id].metadata['acknowledged_at'] = datetime.now().isoformat()
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert"""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].active = False
                self.alerts[alert_id].metadata['resolved_by'] = resolved_by
                self.alerts[alert_id].metadata['resolved_at'] = datetime.now().isoformat()
                logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
        
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        with self._lock:
            alerts = [alert for alert in self.alerts.values() if alert.active]
            
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        with self._lock:
            active_alerts = [a for a in self.alerts.values() if a.active]
            
            stats = {
                'total_alerts': len(self.alerts),
                'active_alerts': len(active_alerts),
                'acknowledged_alerts': len([a for a in active_alerts if a.acknowledged]),
                'by_severity': {},
                'alert_rate_last_hour': 0
            }
            
            # Count by severity
            for severity in AlertSeverity:
                stats['by_severity'][severity.value] = len([
                    a for a in active_alerts if a.severity == severity
                ])
            
            # Calculate alert rate
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_alerts = [a for a in self.alerts.values() if a.timestamp >= one_hour_ago]
            stats['alert_rate_last_hour'] = len(recent_alerts)
            
            return stats
    
    def _evaluate_alerts(self) -> None:
        """Evaluate alert rules periodically"""
        while True:
            try:
                current_time = datetime.now()
                
                for rule in self.alert_rules[:]:  # Copy to avoid modification during iteration
                    # Check if it's time to evaluate this rule
                    if (rule['last_evaluation'] is None or 
                        (current_time - rule['last_evaluation']).total_seconds() >= rule['evaluation_interval']):
                        
                        self._evaluate_single_rule(rule, current_time)
                        rule['last_evaluation'] = current_time
                
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
                time.sleep(30)
    
    def _evaluate_single_rule(self, rule: Dict[str, Any], current_time: datetime) -> None:
        """Evaluate a single alert rule"""
        try:
            # Get metrics for the rule
            metrics = self.metrics_collector.query_metrics(rule['metric_query'])
            
            if not metrics:
                return
            
            # Get the latest value
            latest_metric = max(metrics, key=lambda m: m.timestamp)
            value = latest_metric.value
            
            # Evaluate condition
            condition_met = self._evaluate_condition(value, rule['condition'], rule['threshold'])
            
            # Check if alert should be triggered
            alert_key = f"{rule['name']}_{rule['metric_query']}"
            
            if condition_met:
                # Trigger or update alert
                if alert_key in self.alerts and self.alerts[alert_key].active:
                    # Update existing alert
                    self.alerts[alert_key].count += 1
                    self.alerts[alert_key].timestamp = current_time
                else:
                    # Create new alert
                    alert = Alert(
                        alert_id=str(uuid.uuid4()),
                        name=rule['name'],
                        description=rule['description'],
                        severity=rule['severity'],
                        condition=f"{rule['condition']} {rule['threshold']}",
                        threshold=rule['threshold'],
                        timestamp=current_time,
                        metadata={
                            'metric_value': value,
                            'metric_query': rule['metric_query'],
                            'rule': rule
                        }
                    )
                    
                    with self._lock:
                        self.alerts[alert_key] = alert
                    
                    # Send notifications
                    self._send_notifications(alert)
            else:
                # Resolve alert if it exists and is active
                if alert_key in self.alerts and self.alerts[alert_key].active:
                    self.resolve_alert(self.alerts[alert_key].alert_id, "auto_resolved")
        
        except Exception as e:
            logger.error(f"Error evaluating rule {rule['name']}: {e}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        conditions = {
            'gt': lambda v, t: v > t,
            'gte': lambda v, t: v >= t,
            'lt': lambda v, t: v < t,
            'lte': lambda v, t: v <= t,
            'eq': lambda v, t: abs(v - t) < 0.001,  # Float equality with tolerance
            'neq': lambda v, t: abs(v - t) >= 0.001
        }
        
        condition_func = conditions.get(condition.lower())
        if not condition_func:
            logger.error(f"Unknown condition: {condition}")
            return False
        
        return condition_func(value, threshold)
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications"""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")


class HealthMonitor:
    """System health monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks: Dict[str, HealthCheck] = {}
        self.system_info = {}
        self._lock = threading.Lock()
        
        # Add default system health checks
        self._add_default_health_checks()
        
        # Start health check thread
        self.health_thread = threading.Thread(target=self._run_health_checks, daemon=True)
        self.health_thread.start()
    
    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a custom health check"""
        with self._lock:
            self.health_checks[health_check.name] = health_check
        
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, name: str) -> bool:
        """Remove a health check"""
        with self._lock:
            if name in self.health_checks:
                del self.health_checks[name]
                logger.info(f"Removed health check: {name}")
                return True
        
        return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        with self._lock:
            health_status = {
                'overall_healthy': True,
                'checks': {},
                'system_info': self.system_info,
                'timestamp': datetime.now().isoformat()
            }
            
            for name, check in self.health_checks.items():
                check_status = {
                    'healthy': check.last_result if check.last_result is not None else False,
                    'last_check': check.last_check.isoformat() if check.last_check else None,
                    'failure_count': check.failure_count,
                    'description': check.description
                }
                
                health_status['checks'][name] = check_status
                
                # Update overall health
                if not check_status['healthy']:
                    health_status['overall_healthy'] = False
            
            return health_status
    
    def _add_default_health_checks(self) -> None:
        """Add default system health checks"""
        # CPU usage check
        cpu_check = HealthCheck(
            name="cpu_usage",
            description="Monitor CPU usage",
            check_function=lambda: psutil.cpu_percent(interval=1) < 90,
            interval_seconds=30
        )
        self.add_health_check(cpu_check)
        
        # Memory usage check
        memory_check = HealthCheck(
            name="memory_usage",
            description="Monitor memory usage",
            check_function=lambda: psutil.virtual_memory().percent < 90,
            interval_seconds=30
        )
        self.add_health_check(memory_check)
        
        # Disk usage check
        disk_check = HealthCheck(
            name="disk_usage",
            description="Monitor disk usage",
            check_function=lambda: psutil.disk_usage('/').percent < 90,
            interval_seconds=60
        )
        self.add_health_check(disk_check)
    
    def _run_health_checks(self) -> None:
        """Run health checks periodically"""
        while True:
            try:
                current_time = datetime.now()
                
                # Update system info
                self._update_system_info()
                
                # Run health checks
                for name, check in list(self.health_checks.items()):
                    # Check if it's time to run this check
                    if (check.last_check is None or 
                        (current_time - check.last_check).total_seconds() >= check.interval_seconds):
                        
                        self._run_single_health_check(check, current_time)
                
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(30)
    
    def _run_single_health_check(self, check: HealthCheck, current_time: datetime) -> None:
        """Run a single health check"""
        try:
            # Run the check with timeout
            result = self._run_with_timeout(check.check_function, check.timeout_seconds)
            
            check.last_check = current_time
            check.last_result = result
            
            if result:
                check.failure_count = 0
                # Record success metric
                self.metrics_collector.record_metric(
                    f"health_check_{check.name}",
                    1,
                    {"status": "healthy"}
                )
            else:
                check.failure_count += 1
                # Record failure metric
                self.metrics_collector.record_metric(
                    f"health_check_{check.name}",
                    0,
                    {"status": "unhealthy"}
                )
            
            logger.debug(f"Health check {check.name}: {'PASS' if result else 'FAIL'}")
            
        except Exception as e:
            check.last_check = current_time
            check.last_result = False
            check.failure_count += 1
            logger.error(f"Health check {check.name} failed with exception: {e}")
    
    def _run_with_timeout(self, func: Callable[[], bool], timeout_seconds: int) -> bool:
        """Run function with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Health check timeout")
        
        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            result = func()
            
            # Clear timeout
            signal.alarm(0)
            
            return result
        except TimeoutError:
            logger.warning(f"Health check timed out after {timeout_seconds} seconds")
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
        finally:
            signal.alarm(0)
    
    def _update_system_info(self) -> None:
        """Update system information"""
        try:
            self.system_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'memory_percent': psutil.virtual_memory().percent,
                'disk_total': psutil.disk_usage('/').total,
                'disk_used': psutil.disk_usage('/').used,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None,
                'uptime': time.time() - psutil.boot_time()
            }
            
            # Record system metrics
            self.metrics_collector.record_metric("system_cpu_percent", self.system_info['cpu_percent'])
            self.metrics_collector.record_metric("system_memory_percent", self.system_info['memory_percent'])
            self.metrics_collector.record_metric("system_disk_percent", self.system_info['disk_percent'])
            
        except Exception as e:
            logger.error(f"Error updating system info: {e}")


class PerformanceProfiler:
    """Performance profiling and analysis"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def start_profile(self, profile_name: str, context: Dict[str, Any] = None) -> str:
        """Start a performance profile"""
        profile_id = str(uuid.uuid4())
        
        profile_data = {
            'profile_id': profile_id,
            'name': profile_name,
            'start_time': time.time(),
            'context': context or {},
            'metrics': defaultdict(list)
        }
        
        with self._lock:
            self.active_profiles[profile_id] = profile_data
        
        return profile_id
    
    def add_profile_point(self, profile_id: str, point_name: str, value: float) -> None:
        """Add a data point to an active profile"""
        with self._lock:
            if profile_id in self.active_profiles:
                self.active_profiles[profile_id]['metrics'][point_name].append({
                    'value': value,
                    'timestamp': time.time()
                })
    
    def end_profile(self, profile_id: str) -> Dict[str, Any]:
        """End a performance profile and return results"""
        with self._lock:
            if profile_id not in self.active_profiles:
                return {}
            
            profile_data = self.active_profiles.pop(profile_id)
        
        # Calculate profile statistics
        end_time = time.time()
        duration = end_time - profile_data['start_time']
        
        results = {
            'profile_id': profile_id,
            'name': profile_data['name'],
            'duration': duration,
            'context': profile_data['context'],
            'metrics': {},
            'summary': {}
        }
        
        # Process metrics
        for metric_name, data_points in profile_data['metrics'].items():
            if data_points:
                values = [dp['value'] for dp in data_points]
                results['metrics'][metric_name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'data_points': data_points
                }
        
        # Record profile completion
        self.metrics_collector.record_metric(
            f"profile_duration_{profile_data['name']}",
            duration,
            profile_data['context']
        )
        
        return results
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Profile a function execution"""
        profile_id = self.start_profile(f"function_{func.__name__}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = e
            success = False
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.add_profile_point(profile_id, "execution_time", execution_time)
            self.add_profile_point(profile_id, "success", 1 if success else 0)
        
        profile_results = self.end_profile(profile_id)
        
        if not success:
            raise result
        
        return result, profile_results


# Context manager for easy timing
class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, metrics_collector: MetricsCollector, metric_name: str, 
                 labels: Dict[str, str] = None):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_collector.record_timer(self.metric_name, duration, self.labels)


# Global monitoring components
metrics_collector = MetricsCollector(retention_hours=48)
alert_manager = AlertManager(metrics_collector)
health_monitor = HealthMonitor(metrics_collector)
performance_profiler = PerformanceProfiler(metrics_collector)


def setup_default_alerts() -> None:
    """Set up default alert rules"""
    # High CPU usage alert
    alert_manager.add_alert_rule(
        name="High CPU Usage",
        description="CPU usage is above 90%",
        metric_query="system_cpu_percent",
        condition="gt",
        threshold=90.0,
        severity=AlertSeverity.WARNING
    )
    
    # High memory usage alert
    alert_manager.add_alert_rule(
        name="High Memory Usage",
        description="Memory usage is above 90%",
        metric_query="system_memory_percent",
        condition="gt",
        threshold=90.0,
        severity=AlertSeverity.WARNING
    )
    
    # Critical memory usage alert
    alert_manager.add_alert_rule(
        name="Critical Memory Usage",
        description="Memory usage is above 95%",
        metric_query="system_memory_percent",
        condition="gt",
        threshold=95.0,
        severity=AlertSeverity.CRITICAL
    )


def setup_default_notifications() -> None:
    """Set up default notification handlers"""
    
    def log_notification(alert: Alert) -> None:
        """Log alert notifications"""
        logger.warning(
            f"ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.description} "
            f"(Threshold: {alert.threshold}, Count: {alert.count})"
        )
    
    alert_manager.add_notification_handler(log_notification)


# Initialize default monitoring
setup_default_alerts()
setup_default_notifications()


# Monitoring decorators
def monitor_performance(metric_name: str, labels: Dict[str, str] = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with Timer(metrics_collector, metric_name, labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def monitor_errors(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to monitor function errors"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            error_metric = metric_name or f"{func.__name__}_errors"
            success_metric = metric_name or f"{func.__name__}_success"
            
            try:
                result = func(*args, **kwargs)
                metrics_collector.increment_counter(success_metric, 1, labels)
                return result
            except Exception as e:
                error_labels = (labels or {}).copy()
                error_labels['error_type'] = type(e).__name__
                metrics_collector.increment_counter(error_metric, 1, error_labels)
                raise
        return wrapper
    return decorator