"""
Comprehensive monitoring and metrics collection for GAN-Cyber-Range-v2.

This module provides real-time monitoring, performance metrics, alerting,
and health checks for all cyber range components.
"""

import time
import threading
import logging
import psutil
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor


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
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'unit': self.unit
        }


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    severity: AlertSeverity
    message: str
    component: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    response_time: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates system metrics"""
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_definitions: Dict[str, MetricType] = {}
        self.running = False
        self.collection_thread: Optional[threading.Thread] = None
        self.custom_collectors: List[Callable[[], List[Metric]]] = []
        
    def start_collection(self) -> None:
        """Start metric collection"""
        if self.running:
            return
            
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self) -> None:
        """Stop metric collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        unit: str = ""
    ) -> None:
        """Record a single metric"""
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {},
            unit=unit
        )
        
        self.metrics[name].append(metric)
        self.metric_definitions[name] = metric_type
    
    def increment_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        self.record_metric(name, amount, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, unit: str = "") -> None:
        """Set a gauge metric"""
        self.record_metric(name, value, MetricType.GAUGE, labels, unit)
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timer metric"""
        self.record_metric(name, duration, MetricType.TIMER, labels, "seconds")
    
    def get_metrics(
        self,
        name_pattern: Optional[str] = None,
        time_range: Optional[timedelta] = None
    ) -> List[Metric]:
        """Get metrics matching pattern and time range"""
        
        all_metrics = []
        cutoff_time = datetime.now() - time_range if time_range else None
        
        for metric_name, metric_deque in self.metrics.items():
            if name_pattern and name_pattern not in metric_name:
                continue
                
            for metric in metric_deque:
                if cutoff_time and metric.timestamp < cutoff_time:
                    continue
                all_metrics.append(metric)
        
        return sorted(all_metrics, key=lambda m: m.timestamp)
    
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """Get the latest metric value"""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None
    
    def get_metric_summary(self, name: str, time_range: timedelta) -> Dict[str, float]:
        """Get statistical summary of a metric over time range"""
        
        metrics = self.get_metrics(name, time_range)
        values = [m.value for m in metrics if m.name == name]
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else 0.0
        }
    
    def add_custom_collector(self, collector: Callable[[], List[Metric]]) -> None:
        """Add a custom metric collector function"""
        self.custom_collectors.append(collector)
    
    def _collection_loop(self) -> None:
        """Main collection loop"""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Run custom collectors
                for collector in self.custom_collectors:
                    try:
                        custom_metrics = collector()
                        for metric in custom_metrics:
                            self.metrics[metric.name].append(metric)
                    except Exception as e:
                        logger.error(f"Custom collector failed: {e}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        
        now = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        
        self.metrics['system_cpu_percent'].append(
            Metric('system_cpu_percent', cpu_percent, MetricType.GAUGE, now, unit='percent')
        )
        self.metrics['system_cpu_count'].append(
            Metric('system_cpu_count', cpu_count, MetricType.GAUGE, now, unit='cores')
        )
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics['system_memory_percent'].append(
            Metric('system_memory_percent', memory.percent, MetricType.GAUGE, now, unit='percent')
        )
        self.metrics['system_memory_used'].append(
            Metric('system_memory_used', memory.used / (1024**3), MetricType.GAUGE, now, unit='GB')
        )
        self.metrics['system_memory_available'].append(
            Metric('system_memory_available', memory.available / (1024**3), MetricType.GAUGE, now, unit='GB')
        )
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics['system_disk_percent'].append(
            Metric('system_disk_percent', (disk.used / disk.total) * 100, MetricType.GAUGE, now, unit='percent')
        )
        self.metrics['system_disk_used'].append(
            Metric('system_disk_used', disk.used / (1024**3), MetricType.GAUGE, now, unit='GB')
        )
        
        # Network metrics
        network = psutil.net_io_counters()
        self.metrics['system_network_bytes_sent'].append(
            Metric('system_network_bytes_sent', network.bytes_sent, MetricType.COUNTER, now, unit='bytes')
        )
        self.metrics['system_network_bytes_recv'].append(
            Metric('system_network_bytes_recv', network.bytes_recv, MetricType.COUNTER, now, unit='bytes')
        )


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_handlers: List[Callable[[Alert], None]] = []
        self.alert_history: deque = deque(maxlen=1000)
        
    def add_alert_rule(
        self,
        metric_name: str,
        threshold: float,
        comparison: str = "greater_than",
        severity: AlertSeverity = AlertSeverity.WARNING,
        message_template: str = "Metric {metric_name} is {comparison} threshold {threshold}"
    ) -> None:
        """Add an alert rule"""
        
        rule = {
            'metric_name': metric_name,
            'threshold': threshold,
            'comparison': comparison,
            'severity': severity,
            'message_template': message_template
        }
        
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule for {metric_name}")
    
    def check_metrics(self, metrics_collector: MetricsCollector) -> None:
        """Check metrics against alert rules"""
        
        for rule in self.alert_rules:
            metric_name = rule['metric_name']
            latest_metric = metrics_collector.get_latest_metric(metric_name)
            
            if not latest_metric:
                continue
            
            should_alert = self._evaluate_alert_condition(latest_metric.value, rule)
            
            if should_alert:
                self._create_alert(latest_metric, rule)
    
    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler"""
        self.notification_handlers.append(handler)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                logger.info(f"Resolved alert: {alert_id}")
                return True
        return False
    
    def _evaluate_alert_condition(self, value: float, rule: Dict[str, Any]) -> bool:
        """Evaluate if metric value triggers alert"""
        
        threshold = rule['threshold']
        comparison = rule['comparison']
        
        if comparison == "greater_than":
            return value > threshold
        elif comparison == "less_than":
            return value < threshold
        elif comparison == "equals":
            return abs(value - threshold) < 0.001
        elif comparison == "not_equals":
            return abs(value - threshold) >= 0.001
        
        return False
    
    def _create_alert(self, metric: Metric, rule: Dict[str, Any]) -> None:
        """Create a new alert"""
        
        alert_id = f"alert_{int(time.time() * 1000)}"
        
        # Check if similar alert already exists
        existing_alerts = [
            a for a in self.get_active_alerts()
            if a.metric_name == metric.name and a.component == "system"
        ]
        
        if existing_alerts:
            # Don't create duplicate alerts
            return
        
        message = rule['message_template'].format(
            metric_name=metric.name,
            comparison=rule['comparison'],
            threshold=rule['threshold'],
            value=metric.value
        )
        
        alert = Alert(
            alert_id=alert_id,
            severity=rule['severity'],
            message=message,
            component="system",
            metric_name=metric.name,
            metric_value=metric.value,
            threshold=rule['threshold']
        )
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
        
        logger.warning(f"Created alert: {alert.message}")


class HealthMonitor:
    """Monitors component health"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.health_status: Dict[str, HealthCheck] = {}
        self.check_interval = 30.0
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def register_health_check(self, component: str, check_func: Callable[[], HealthCheck]) -> None:
        """Register a health check for a component"""
        self.health_checks[component] = check_func
        logger.info(f"Registered health check for component: {component}")
    
    def start_monitoring(self) -> None:
        """Start health monitoring"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def check_component_health(self, component: str) -> Optional[HealthCheck]:
        """Check health of a specific component"""
        if component not in self.health_checks:
            return None
        
        try:
            start_time = time.time()
            health_check = self.health_checks[component]()
            health_check.response_time = time.time() - start_time
            
            self.health_status[component] = health_check
            return health_check
            
        except Exception as e:
            error_check = HealthCheck(
                component=component,
                status="unhealthy",
                message=f"Health check failed: {e}",
                response_time=time.time() - start_time if 'start_time' in locals() else None
            )
            self.health_status[component] = error_check
            return error_check
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        
        if not self.health_status:
            return {'status': 'unknown', 'components': {}}
        
        component_statuses = list(self.health_status.values())
        
        # Determine overall status
        if any(h.status == "unhealthy" for h in component_statuses):
            overall_status = "unhealthy"
        elif any(h.status == "degraded" for h in component_statuses):
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'components': {h.component: h.status for h in component_statuses},
            'details': [asdict(h) for h in component_statuses]
        }
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                for component in self.health_checks:
                    self.check_component_health(component)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)


class PerformanceMonitor:
    """Monitors performance metrics and provides profiling"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_monitor = HealthMonitor()
        self.active_timers: Dict[str, float] = {}
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def start(self) -> None:
        """Start all monitoring components"""
        self.metrics_collector.start_collection()
        self.health_monitor.start_monitoring()
        
        # Start alert checking
        threading.Thread(target=self._alert_checking_loop, daemon=True).start()
        
        logger.info("Performance monitoring started")
    
    def stop(self) -> None:
        """Stop all monitoring components"""
        self.metrics_collector.stop_collection()
        self.health_monitor.stop_monitoring()
        logger.info("Performance monitoring stopped")
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        self.active_timers[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and record metric"""
        if operation not in self.active_timers:
            logger.warning(f"Timer not found for operation: {operation}")
            return 0.0
        
        start_time = self.active_timers.pop(operation)
        duration = time.time() - start_time
        
        self.metrics_collector.record_timer(f"operation_duration", duration, {"operation": operation})
        return duration
    
    def record_attack_metric(self, attack_id: str, technique_id: str, success: bool, duration: float) -> None:
        """Record attack execution metrics"""
        labels = {
            "attack_id": attack_id,
            "technique_id": technique_id,
            "success": str(success)
        }
        
        self.metrics_collector.increment_counter("attacks_total", 1.0, labels)
        self.metrics_collector.record_timer("attack_duration", duration, labels)
        
        if success:
            self.metrics_collector.increment_counter("attacks_successful", 1.0, labels)
    
    def record_detection_metric(self, detection_type: str, confidence: float, response_time: float) -> None:
        """Record detection metrics"""
        labels = {"detection_type": detection_type}
        
        self.metrics_collector.increment_counter("detections_total", 1.0, labels)
        self.metrics_collector.set_gauge("detection_confidence", confidence, labels)
        self.metrics_collector.record_timer("detection_response_time", response_time, labels)
    
    def record_range_metric(self, range_id: str, operation: str, success: bool, duration: float) -> None:
        """Record cyber range operation metrics"""
        labels = {
            "range_id": range_id,
            "operation": operation,
            "success": str(success)
        }
        
        self.metrics_collector.increment_counter("range_operations_total", 1.0, labels)
        self.metrics_collector.record_timer("range_operation_duration", duration, labels)
    
    def get_performance_report(self, time_range: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Generate performance report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'time_range': str(time_range),
            'system_health': self.health_monitor.get_overall_health(),
            'active_alerts': len(self.alert_manager.get_active_alerts()),
            'metrics_summary': {}
        }
        
        # Get key metric summaries
        key_metrics = [
            'system_cpu_percent',
            'system_memory_percent',
            'system_disk_percent',
            'attacks_total',
            'detections_total',
            'range_operations_total'
        ]
        
        for metric_name in key_metrics:
            summary = self.metrics_collector.get_metric_summary(metric_name, time_range)
            if summary:
                report['metrics_summary'][metric_name] = summary
        
        return report
    
    def _setup_default_alerts(self) -> None:
        """Setup default alert rules"""
        
        # System resource alerts
        self.alert_manager.add_alert_rule(
            "system_cpu_percent", 80.0, "greater_than", AlertSeverity.WARNING,
            "High CPU usage: {value}% (threshold: {threshold}%)"
        )
        
        self.alert_manager.add_alert_rule(
            "system_memory_percent", 85.0, "greater_than", AlertSeverity.WARNING,
            "High memory usage: {value}% (threshold: {threshold}%)"
        )
        
        self.alert_manager.add_alert_rule(
            "system_disk_percent", 90.0, "greater_than", AlertSeverity.CRITICAL,
            "High disk usage: {value}% (threshold: {threshold}%)"
        )
        
        # Add console notification handler
        def console_alert_handler(alert: Alert):
            print(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
        
        self.alert_manager.add_notification_handler(console_alert_handler)
    
    def _setup_default_health_checks(self) -> None:
        """Setup default health checks"""
        
        def system_health_check() -> HealthCheck:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent > 90 or memory_percent > 90:
                    return HealthCheck(
                        component="system",
                        status="unhealthy",
                        message=f"High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%"
                    )
                elif cpu_percent > 70 or memory_percent > 70:
                    return HealthCheck(
                        component="system",
                        status="degraded",
                        message=f"Moderate resource usage: CPU {cpu_percent}%, Memory {memory_percent}%"
                    )
                else:
                    return HealthCheck(
                        component="system",
                        status="healthy",
                        message="System resources normal"
                    )
            except Exception as e:
                return HealthCheck(
                    component="system",
                    status="unhealthy",
                    message=f"Health check failed: {e}"
                )
        
        def disk_health_check() -> HealthCheck:
            try:
                disk_usage = psutil.disk_usage('/')
                disk_percent = (disk_usage.used / disk_usage.total) * 100
                
                if disk_percent > 95:
                    return HealthCheck(
                        component="disk",
                        status="unhealthy",
                        message=f"Critical disk usage: {disk_percent:.1f}%"
                    )
                elif disk_percent > 85:
                    return HealthCheck(
                        component="disk",
                        status="degraded",
                        message=f"High disk usage: {disk_percent:.1f}%"
                    )
                else:
                    return HealthCheck(
                        component="disk",
                        status="healthy",
                        message=f"Disk usage normal: {disk_percent:.1f}%"
                    )
            except Exception as e:
                return HealthCheck(
                    component="disk",
                    status="unhealthy",
                    message=f"Disk check failed: {e}"
                )
        
        self.health_monitor.register_health_check("system", system_health_check)
        self.health_monitor.register_health_check("disk", disk_health_check)
    
    def _alert_checking_loop(self) -> None:
        """Alert checking loop"""
        while True:
            try:
                self.alert_manager.check_metrics(self.metrics_collector)
                time.sleep(30.0)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Alert checking error: {e}")
                time.sleep(30.0)


# Context manager for timing operations
class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, monitor: PerformanceMonitor, operation: str):
        self.monitor = monitor
        self.operation = operation
        
    def __enter__(self):
        self.monitor.start_timer(self.operation)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.monitor.end_timer(self.operation)
        if exc_type is not None:
            # Operation failed
            self.monitor.metrics_collector.increment_counter(
                "operation_failures",
                1.0,
                {"operation": self.operation}
            )
        return False


def monitor_performance(operation: str):
    """Decorator for monitoring function performance"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get or create performance monitor
            monitor = getattr(func, '_performance_monitor', None)
            if not monitor:
                monitor = PerformanceMonitor()
                func._performance_monitor = monitor
            
            with TimerContext(monitor, operation):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator