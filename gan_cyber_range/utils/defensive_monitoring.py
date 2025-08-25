#!/usr/bin/env python3
"""
Defensive monitoring and alerting system
Provides comprehensive monitoring for defensive cybersecurity operations
"""

import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import queue
import uuid

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event for defensive monitoring"""
    event_id: str
    timestamp: str
    event_type: str
    severity: str
    source: str
    description: str
    data: Dict[str, Any]
    status: str = "new"
    response_actions: List[str] = None
    
    def __post_init__(self):
        if self.response_actions is None:
            self.response_actions = []


@dataclass
class DefenseMetric:
    """Defense performance metric"""
    metric_name: str
    value: Union[int, float]
    unit: str
    timestamp: str
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class DefensiveMonitor:
    """Comprehensive defensive monitoring system"""
    
    def __init__(self, 
                 alert_threshold: Dict[str, Any] = None,
                 retention_days: int = 30):
        self.alert_threshold = alert_threshold or self._default_thresholds()
        self.retention_days = retention_days
        
        # Event storage
        self.events = []
        self.metrics = []
        self.alerts = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.event_queue = queue.Queue()
        
        # Event handlers
        self.event_handlers = {}
        self.metric_handlers = {}
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "total_metrics": 0,
            "total_alerts": 0,
            "start_time": None,
            "last_event": None
        }
        
        logger.info("Defensive monitor initialized")
    
    def _default_thresholds(self) -> Dict[str, Any]:
        """Default alert thresholds for defensive operations"""
        return {
            "attack_success_rate": {"critical": 0.8, "warning": 0.5},
            "detection_failure_rate": {"critical": 0.7, "warning": 0.4},
            "response_time": {"critical": 300, "warning": 180},  # seconds
            "resource_usage": {"critical": 90, "warning": 75},   # percentage
            "error_rate": {"critical": 10, "warning": 5},        # per minute
            "anomaly_score": {"critical": 0.9, "warning": 0.7}
        }
    
    def start_monitoring(self) -> None:
        """Start the monitoring system"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self.stats["start_time"] = datetime.now().isoformat()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Defensive monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Defensive monitoring stopped")
    
    def record_event(self, 
                    event_type: str,
                    severity: str,
                    source: str,
                    description: str,
                    data: Dict[str, Any] = None) -> str:
        """Record a security event"""
        
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            severity=severity,
            source=source,
            description=description,
            data=data or {}
        )
        
        self.events.append(event)
        self.stats["total_events"] += 1
        self.stats["last_event"] = event.timestamp
        
        # Queue for processing
        self.event_queue.put(event)
        
        logger.debug(f"Recorded event: {event_type} - {severity}")
        return event.event_id
    
    def record_metric(self,
                     metric_name: str,
                     value: Union[int, float],
                     unit: str = "count",
                     tags: Dict[str, str] = None) -> None:
        """Record a defense metric"""
        
        metric = DefenseMetric(
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.now().isoformat(),
            tags=tags or {}
        )
        
        self.metrics.append(metric)
        self.stats["total_metrics"] += 1
        
        # Check for alerts
        self._check_metric_alerts(metric)
        
        logger.debug(f"Recorded metric: {metric_name} = {value} {unit}")
    
    def record_attack_detection(self,
                              attack_id: str,
                              attack_type: str,
                              detected: bool,
                              confidence: float,
                              detection_time: float) -> None:
        """Record attack detection event"""
        
        severity = "high" if detected else "critical"
        description = f"Attack {attack_type} {'detected' if detected else 'missed'}"
        
        data = {
            "attack_id": attack_id,
            "attack_type": attack_type,
            "detected": detected,
            "confidence": confidence,
            "detection_time": detection_time
        }
        
        self.record_event("attack_detection", severity, "ids", description, data)
        
        # Record metrics
        self.record_metric("detection_success", 1 if detected else 0)
        self.record_metric("detection_confidence", confidence)
        self.record_metric("detection_time", detection_time, "seconds")
    
    def record_incident_response(self,
                               incident_id: str,
                               response_time: float,
                               containment_success: bool,
                               response_actions: List[str]) -> None:
        """Record incident response metrics"""
        
        severity = "medium" if containment_success else "high"
        description = f"Incident response {'successful' if containment_success else 'failed'}"
        
        data = {
            "incident_id": incident_id,
            "response_time": response_time,
            "containment_success": containment_success,
            "response_actions": response_actions
        }
        
        self.record_event("incident_response", severity, "soc", description, data)
        
        # Record metrics
        self.record_metric("response_time", response_time, "seconds")
        self.record_metric("containment_success", 1 if containment_success else 0)
        self.record_metric("response_actions_count", len(response_actions))
    
    def record_training_performance(self,
                                  trainee_id: str,
                                  scenario_id: str,
                                  score: float,
                                  completion_time: float,
                                  objectives_met: int,
                                  total_objectives: int) -> None:
        """Record training performance metrics"""
        
        completion_rate = objectives_met / total_objectives if total_objectives > 0 else 0
        severity = "low" if completion_rate >= 0.8 else "medium"
        
        data = {
            "trainee_id": trainee_id,
            "scenario_id": scenario_id,
            "score": score,
            "completion_time": completion_time,
            "objectives_met": objectives_met,
            "total_objectives": total_objectives,
            "completion_rate": completion_rate
        }
        
        self.record_event("training_performance", severity, "training", 
                         f"Training completed with {score:.1f}% score", data)
        
        # Record metrics
        self.record_metric("training_score", score)
        self.record_metric("training_completion_time", completion_time, "minutes")
        self.record_metric("training_completion_rate", completion_rate)
    
    def add_event_handler(self, event_type: str, handler: Callable[[SecurityEvent], None]) -> None:
        """Add event handler for specific event types"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Added event handler for {event_type}")
    
    def add_metric_handler(self, metric_name: str, handler: Callable[[DefenseMetric], None]) -> None:
        """Add metric handler for specific metrics"""
        if metric_name not in self.metric_handlers:
            self.metric_handlers[metric_name] = []
        self.metric_handlers[metric_name].append(handler)
        logger.info(f"Added metric handler for {metric_name}")
    
    def get_events(self, 
                  event_type: str = None,
                  severity: str = None,
                  since: datetime = None,
                  limit: int = None) -> List[SecurityEvent]:
        """Get filtered events"""
        
        filtered_events = self.events
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
        
        if since:
            since_str = since.isoformat()
            filtered_events = [e for e in filtered_events if e.timestamp >= since_str]
        
        if limit:
            filtered_events = filtered_events[-limit:]
        
        return filtered_events
    
    def get_metrics(self,
                   metric_name: str = None,
                   since: datetime = None,
                   limit: int = None) -> List[DefenseMetric]:
        """Get filtered metrics"""
        
        filtered_metrics = self.metrics
        
        if metric_name:
            filtered_metrics = [m for m in filtered_metrics if m.metric_name == metric_name]
        
        if since:
            since_str = since.isoformat()
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= since_str]
        
        if limit:
            filtered_metrics = filtered_metrics[-limit:]
        
        return filtered_metrics
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Recent events by severity
        recent_events = self.get_events(since=last_hour)
        events_by_severity = {}
        for event in recent_events:
            severity = event.severity
            events_by_severity[severity] = events_by_severity.get(severity, 0) + 1
        
        # Recent metrics summary
        recent_metrics = self.get_metrics(since=last_hour)
        metrics_summary = {}
        for metric in recent_metrics:
            name = metric.metric_name
            if name not in metrics_summary:
                metrics_summary[name] = {"count": 0, "avg_value": 0, "values": []}
            metrics_summary[name]["count"] += 1
            metrics_summary[name]["values"].append(metric.value)
        
        # Calculate averages
        for name, data in metrics_summary.items():
            if data["values"]:
                data["avg_value"] = sum(data["values"]) / len(data["values"])
        
        # Detection performance (last 24h)
        day_events = self.get_events(event_type="attack_detection", since=last_day)
        detections = [e for e in day_events if e.data.get("detected", False)]
        detection_rate = len(detections) / len(day_events) if day_events else 0
        
        # Response performance (last 24h)
        response_events = self.get_events(event_type="incident_response", since=last_day)
        if response_events:
            avg_response_time = sum(e.data.get("response_time", 0) for e in response_events) / len(response_events)
            successful_responses = sum(1 for e in response_events if e.data.get("containment_success", False))
            response_success_rate = successful_responses / len(response_events)
        else:
            avg_response_time = 0
            response_success_rate = 0
        
        return {
            "monitoring_status": "active" if self.is_monitoring else "inactive",
            "uptime": self._calculate_uptime(),
            "total_events": self.stats["total_events"],
            "total_metrics": self.stats["total_metrics"],
            "total_alerts": self.stats["total_alerts"],
            "events_by_severity": events_by_severity,
            "metrics_summary": metrics_summary,
            "detection_rate_24h": round(detection_rate, 3),
            "avg_response_time_24h": round(avg_response_time, 2),
            "response_success_rate_24h": round(response_success_rate, 3),
            "last_event": self.stats["last_event"]
        }
    
    def export_data(self, file_path: Union[str, Path], format: str = "json") -> None:
        """Export monitoring data"""
        
        file_path = Path(file_path)
        
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "monitoring_stats": self.stats,
                "total_events": len(self.events),
                "total_metrics": len(self.metrics)
            },
            "events": [asdict(event) for event in self.events],
            "metrics": [asdict(metric) for metric in self.metrics],
            "alerts": self.alerts
        }
        
        if format.lower() == "json":
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Monitoring data exported to {file_path}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        logger.info("Monitoring loop started")
        
        while self.is_monitoring:
            try:
                # Process events from queue
                while not self.event_queue.empty():
                    try:
                        event = self.event_queue.get_nowait()
                        self._process_event(event)
                    except queue.Empty:
                        break
                
                # Periodic maintenance
                self._cleanup_old_data()
                
                # Sleep briefly
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)
        
        logger.info("Monitoring loop stopped")
    
    def _process_event(self, event: SecurityEvent) -> None:
        """Process individual event"""
        try:
            # Run event handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"Event handler error: {str(e)}")
            
            # Check for alerts
            self._check_event_alerts(event)
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {str(e)}")
    
    def _check_event_alerts(self, event: SecurityEvent) -> None:
        """Check if event triggers alerts"""
        
        # High/critical events always create alerts
        if event.severity in ["high", "critical"]:
            alert = {
                "alert_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "type": "event_severity",
                "message": f"{event.severity.upper()} event: {event.description}",
                "source_event": event.event_id,
                "data": asdict(event)
            }
            
            self.alerts.append(alert)
            self.stats["total_alerts"] += 1
            
            logger.warning(f"Alert generated for {event.severity} event: {event.event_id}")
    
    def _check_metric_alerts(self, metric: DefenseMetric) -> None:
        """Check if metric triggers alerts"""
        
        metric_name = metric.metric_name
        if metric_name in self.alert_threshold:
            thresholds = self.alert_threshold[metric_name]
            
            alert_level = None
            if "critical" in thresholds and metric.value >= thresholds["critical"]:
                alert_level = "critical"
            elif "warning" in thresholds and metric.value >= thresholds["warning"]:
                alert_level = "warning"
            
            if alert_level:
                alert = {
                    "alert_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "type": "metric_threshold",
                    "level": alert_level,
                    "message": f"{metric_name} exceeded {alert_level} threshold: {metric.value}",
                    "metric_data": asdict(metric),
                    "threshold": thresholds[alert_level]
                }
                
                self.alerts.append(alert)
                self.stats["total_alerts"] += 1
                
                logger.warning(f"{alert_level.upper()} alert: {metric_name} = {metric.value}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old data based on retention policy"""
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        cutoff_str = cutoff_date.isoformat()
        
        # Clean events
        self.events = [e for e in self.events if e.timestamp >= cutoff_str]
        
        # Clean metrics
        self.metrics = [m for m in self.metrics if m.timestamp >= cutoff_str]
        
        # Clean alerts (keep longer)
        alert_cutoff = datetime.now() - timedelta(days=self.retention_days * 2)
        alert_cutoff_str = alert_cutoff.isoformat()
        self.alerts = [a for a in self.alerts if a["timestamp"] >= alert_cutoff_str]
    
    def _calculate_uptime(self) -> str:
        """Calculate monitoring uptime"""
        if not self.stats["start_time"]:
            return "00:00:00"
        
        start = datetime.fromisoformat(self.stats["start_time"])
        uptime_seconds = (datetime.now() - start).total_seconds()
        
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# Global monitor instance
_global_monitor = DefensiveMonitor()


def start_monitoring():
    """Start global monitoring"""
    _global_monitor.start_monitoring()


def stop_monitoring():
    """Stop global monitoring"""
    _global_monitor.stop_monitoring()


def record_event(event_type: str, severity: str, source: str, description: str, data: Dict = None) -> str:
    """Record event using global monitor"""
    return _global_monitor.record_event(event_type, severity, source, description, data)


def record_metric(metric_name: str, value: Union[int, float], unit: str = "count", tags: Dict = None):
    """Record metric using global monitor"""
    _global_monitor.record_metric(metric_name, value, unit, tags)


def get_dashboard_data() -> Dict[str, Any]:
    """Get dashboard data from global monitor"""
    return _global_monitor.get_dashboard_data()


if __name__ == "__main__":
    # Test defensive monitoring
    monitor = DefensiveMonitor()
    monitor.start_monitoring()
    
    # Simulate some events and metrics
    monitor.record_attack_detection("attack_1", "malware", True, 0.95, 2.5)
    monitor.record_incident_response("incident_1", 120.0, True, ["isolate", "scan", "clean"])
    monitor.record_training_performance("trainee_1", "scenario_1", 85.0, 45.0, 8, 10)
    
    time.sleep(2)
    
    dashboard = monitor.get_dashboard_data()
    print(f"Dashboard data: {json.dumps(dashboard, indent=2)}")
    
    monitor.stop_monitoring()
    print("Defensive monitoring test completed ✅")