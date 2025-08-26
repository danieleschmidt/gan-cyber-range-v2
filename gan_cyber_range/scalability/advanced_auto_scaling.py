"""
Advanced Auto-Scaling Framework

Intelligent auto-scaling system with predictive scaling, multi-metric monitoring,
and cost optimization for cyber range infrastructure.
"""

import asyncio
import time
import json
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import statistics
import math

from ..utils.robust_error_handler import robust, critical, ErrorSeverity
from ..utils.comprehensive_logging import comprehensive_logger
from ..optimization.intelligent_performance import performance_optimizer


class ScalingDirection(Enum):
    """Scaling direction"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    """What triggered the scaling decision"""
    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    PREDICTIVE = "predictive"
    COST_OPTIMIZATION = "cost_optimization"
    MANUAL = "manual"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    disk_io_utilization: float = 0.0
    network_io_utilization: float = 0.0
    queue_length: int = 0
    active_connections: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingRule:
    """Auto-scaling rule definition"""
    name: str
    metric: str
    threshold_up: float
    threshold_down: float
    scale_up_amount: int
    scale_down_amount: int
    cooldown_period: int  # seconds
    priority: int = 1
    enabled: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingEvent:
    """Scaling event record"""
    timestamp: datetime
    direction: ScalingDirection
    trigger: ScalingTrigger
    before_capacity: int
    after_capacity: int
    metrics: ScalingMetrics
    reason: str
    success: bool = True
    error_message: Optional[str] = None


class PredictiveScaler:
    """Predictive scaling using historical patterns"""
    
    def __init__(self, history_window: int = 168):  # 1 week in hours
        self.history_window = history_window
        self.metrics_history: deque = deque(maxlen=history_window)
        self.pattern_cache: Dict[str, Any] = {}
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics to history for pattern analysis"""
        self.metrics_history.append(metrics)
        
        # Update patterns every hour
        if len(self.metrics_history) % 60 == 0:  # Assuming metrics every minute
            self._update_patterns()
    
    def _update_patterns(self):
        """Update usage patterns from historical data"""
        if len(self.metrics_history) < 24:  # Need at least 24 hours
            return
        
        # Daily patterns
        hourly_cpu = [[] for _ in range(24)]
        hourly_memory = [[] for _ in range(24)]
        
        for i, metrics in enumerate(self.metrics_history):
            hour = metrics.timestamp.hour
            hourly_cpu[hour].append(metrics.cpu_utilization)
            hourly_memory[hour].append(metrics.memory_utilization)
        
        # Calculate averages for each hour
        self.pattern_cache['hourly_cpu'] = [
            statistics.mean(values) if values else 0
            for values in hourly_cpu
        ]
        self.pattern_cache['hourly_memory'] = [
            statistics.mean(values) if values else 0
            for values in hourly_memory
        ]
        
        # Weekly patterns (if we have enough data)
        if len(self.metrics_history) >= 168:
            weekly_cpu = [[] for _ in range(7)]
            weekly_memory = [[] for _ in range(7)]
            
            for metrics in self.metrics_history:
                day = metrics.timestamp.weekday()
                weekly_cpu[day].append(metrics.cpu_utilization)
                weekly_memory[day].append(metrics.memory_utilization)
            
            self.pattern_cache['weekly_cpu'] = [
                statistics.mean(values) if values else 0
                for values in weekly_cpu
            ]
            self.pattern_cache['weekly_memory'] = [
                statistics.mean(values) if values else 0
                for values in weekly_memory
            ]
    
    def predict_load(self, hours_ahead: int = 1) -> Dict[str, float]:
        """Predict load N hours ahead"""
        if not self.pattern_cache:
            return {"cpu": 50.0, "memory": 50.0}  # Default prediction
        
        future_time = datetime.now() + timedelta(hours=hours_ahead)
        future_hour = future_time.hour
        future_day = future_time.weekday()
        
        # Combine hourly and weekly patterns
        hourly_cpu = self.pattern_cache.get('hourly_cpu', [50] * 24)
        hourly_memory = self.pattern_cache.get('hourly_memory', [50] * 24)
        
        predicted_cpu = hourly_cpu[future_hour]
        predicted_memory = hourly_memory[future_hour]
        
        # Adjust for weekly patterns if available
        if 'weekly_cpu' in self.pattern_cache:
            weekly_cpu = self.pattern_cache['weekly_cpu']
            weekly_memory = self.pattern_cache['weekly_memory']
            
            # Weighted combination
            predicted_cpu = (predicted_cpu * 0.7) + (weekly_cpu[future_day] * 0.3)
            predicted_memory = (predicted_memory * 0.7) + (weekly_memory[future_day] * 0.3)
        
        # Add trend analysis
        if len(self.metrics_history) >= 6:  # At least 6 data points
            recent_metrics = list(self.metrics_history)[-6:]
            
            # Simple linear trend
            cpu_values = [m.cpu_utilization for m in recent_metrics]
            memory_values = [m.memory_utilization for m in recent_metrics]
            
            cpu_trend = self._calculate_trend(cpu_values)
            memory_trend = self._calculate_trend(memory_values)
            
            # Apply trend
            predicted_cpu += cpu_trend * hours_ahead
            predicted_memory += memory_trend * hours_ahead
        
        return {
            "cpu": max(0, min(100, predicted_cpu)),
            "memory": max(0, min(100, predicted_memory))
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend from values"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Linear regression slope
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


class CostOptimizer:
    """Cost optimization for scaling decisions"""
    
    def __init__(self):
        # Cost models (per hour)
        self.instance_costs = {
            "small": 0.10,
            "medium": 0.20,
            "large": 0.40,
            "xlarge": 0.80
        }
        
        # Performance characteristics
        self.instance_performance = {
            "small": {"cpu": 1, "memory": 2},
            "medium": {"cpu": 2, "memory": 4},
            "large": {"cpu": 4, "memory": 8},
            "xlarge": {"cpu": 8, "memory": 16}
        }
    
    def optimize_capacity(self, required_cpu: float, required_memory: float) -> Dict[str, Any]:
        """Find optimal instance configuration"""
        options = []
        
        for instance_type, perf in self.instance_performance.items():
            # Calculate how many instances needed
            cpu_instances = math.ceil(required_cpu / perf["cpu"])
            memory_instances = math.ceil(required_memory / perf["memory"])
            instances_needed = max(cpu_instances, memory_instances)
            
            total_cost = instances_needed * self.instance_costs[instance_type]
            
            options.append({
                "instance_type": instance_type,
                "count": instances_needed,
                "total_cost": total_cost,
                "cpu_capacity": instances_needed * perf["cpu"],
                "memory_capacity": instances_needed * perf["memory"],
                "efficiency": (required_cpu + required_memory) / (
                    instances_needed * (perf["cpu"] + perf["memory"])
                )
            })
        
        # Sort by cost efficiency
        options.sort(key=lambda x: (x["total_cost"], -x["efficiency"]))
        
        return options[0] if options else None


class AdvancedAutoScaler:
    """Main auto-scaling orchestrator"""
    
    def __init__(
        self,
        min_capacity: int = 1,
        max_capacity: int = 100,
        target_cpu_utilization: float = 70.0,
        target_memory_utilization: float = 80.0
    ):
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.target_cpu_utilization = target_cpu_utilization
        self.target_memory_utilization = target_memory_utilization
        
        # Current state
        self.current_capacity = min_capacity
        self.last_scale_time = datetime.now()
        self.is_enabled = True
        
        # Components
        self.predictive_scaler = PredictiveScaler()
        self.cost_optimizer = CostOptimizer()
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = []
        self._initialize_default_rules()
        
        # Event history
        self.scaling_events: deque = deque(maxlen=1000)
        self.metrics_buffer: deque = deque(maxlen=100)
        
        # Callbacks
        self.scale_callbacks: List[Callable] = []
        
        # Configuration
        self.config = {
            "cooldown_period": 300,  # 5 minutes
            "predictive_enabled": True,
            "cost_optimization_enabled": True,
            "aggressive_scaling": False,
            "scale_up_threshold": 0.8,  # Scale up when utilization > 80%
            "scale_down_threshold": 0.3,  # Scale down when utilization < 30%
            "scale_factor": 1.5,  # How much to scale by
            "evaluation_window": 300  # 5 minutes
        }
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        comprehensive_logger.info(
            "Advanced auto-scaler initialized",
            additional_data={
                "min_capacity": min_capacity,
                "max_capacity": max_capacity,
                "target_cpu": target_cpu_utilization,
                "target_memory": target_memory_utilization
            }
        )
    
    def _initialize_default_rules(self):
        """Initialize default scaling rules"""
        self.scaling_rules.extend([
            ScalingRule(
                name="High CPU",
                metric="cpu_utilization",
                threshold_up=self.target_cpu_utilization,
                threshold_down=self.target_cpu_utilization * 0.5,
                scale_up_amount=2,
                scale_down_amount=1,
                cooldown_period=300,
                priority=1
            ),
            ScalingRule(
                name="High Memory",
                metric="memory_utilization",
                threshold_up=self.target_memory_utilization,
                threshold_down=self.target_memory_utilization * 0.5,
                scale_up_amount=2,
                scale_down_amount=1,
                cooldown_period=300,
                priority=2
            ),
            ScalingRule(
                name="High Response Time",
                metric="average_response_time",
                threshold_up=2000,  # 2 seconds
                threshold_down=500,  # 0.5 seconds
                scale_up_amount=3,
                scale_down_amount=1,
                cooldown_period=180,
                priority=3
            ),
            ScalingRule(
                name="Queue Length",
                metric="queue_length",
                threshold_up=50,
                threshold_down=10,
                scale_up_amount=5,
                scale_down_amount=2,
                cooldown_period=120,
                priority=4
            )
        ])
    
    @robust(severity=ErrorSeverity.MEDIUM)
    def evaluate_scaling(self, metrics: ScalingMetrics) -> Optional[ScalingEvent]:
        """Evaluate if scaling is needed"""
        
        # Add metrics to history
        self.metrics_buffer.append(metrics)
        self.predictive_scaler.add_metrics(metrics)
        
        # Check if we're in cooldown period
        time_since_last_scale = (datetime.now() - self.last_scale_time).total_seconds()
        if time_since_last_scale < self.config["cooldown_period"]:
            return None
        
        # Get recent metrics for analysis
        if len(self.metrics_buffer) < 3:
            return None  # Need some history
        
        recent_metrics = list(self.metrics_buffer)[-3:]
        avg_metrics = self._calculate_average_metrics(recent_metrics)
        
        # Check scaling rules
        triggered_rules = []
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            metric_value = getattr(avg_metrics, rule.metric, 0)
            
            if metric_value > rule.threshold_up:
                triggered_rules.append((rule, ScalingDirection.UP, metric_value))
            elif metric_value < rule.threshold_down and self.current_capacity > self.min_capacity:
                triggered_rules.append((rule, ScalingDirection.DOWN, metric_value))
        
        if not triggered_rules:
            # Check predictive scaling
            if self.config["predictive_enabled"]:
                return self._evaluate_predictive_scaling(avg_metrics)
            return None
        
        # Sort by priority and select highest priority rule
        triggered_rules.sort(key=lambda x: x[0].priority)
        rule, direction, metric_value = triggered_rules[0]
        
        # Calculate scaling amount
        if direction == ScalingDirection.UP:
            scale_amount = rule.scale_up_amount
            new_capacity = min(
                self.max_capacity,
                self.current_capacity + scale_amount
            )
        else:
            scale_amount = rule.scale_down_amount
            new_capacity = max(
                self.min_capacity,
                self.current_capacity - scale_amount
            )
        
        if new_capacity == self.current_capacity:
            return None  # No change needed
        
        # Create scaling event
        scaling_event = ScalingEvent(
            timestamp=datetime.now(),
            direction=direction,
            trigger=ScalingTrigger.CPU_THRESHOLD if rule.metric == "cpu_utilization" else ScalingTrigger.MEMORY_THRESHOLD,
            before_capacity=self.current_capacity,
            after_capacity=new_capacity,
            metrics=avg_metrics,
            reason=f"Rule '{rule.name}' triggered: {rule.metric}={metric_value:.1f}"
        )
        
        return scaling_event
    
    def _evaluate_predictive_scaling(self, current_metrics: ScalingMetrics) -> Optional[ScalingEvent]:
        """Evaluate predictive scaling"""
        
        # Predict load 1 hour ahead
        prediction = self.predictive_scaler.predict_load(1)
        
        # Check if predicted load requires scaling
        predicted_cpu = prediction["cpu"]
        predicted_memory = prediction["memory"]
        
        if predicted_cpu > self.target_cpu_utilization * 1.2 or \
           predicted_memory > self.target_memory_utilization * 1.2:
            
            # Predict capacity needed
            cpu_ratio = predicted_cpu / 100
            memory_ratio = predicted_memory / 100
            capacity_multiplier = max(cpu_ratio, memory_ratio)
            
            suggested_capacity = int(self.current_capacity * capacity_multiplier * 1.1)  # 10% buffer
            new_capacity = min(self.max_capacity, max(self.min_capacity, suggested_capacity))
            
            if new_capacity > self.current_capacity:
                return ScalingEvent(
                    timestamp=datetime.now(),
                    direction=ScalingDirection.UP,
                    trigger=ScalingTrigger.PREDICTIVE,
                    before_capacity=self.current_capacity,
                    after_capacity=new_capacity,
                    metrics=current_metrics,
                    reason=f"Predictive scaling: CPU={predicted_cpu:.1f}%, Memory={predicted_memory:.1f}%"
                )
        
        return None
    
    @critical(max_retries=3)
    def execute_scaling(self, scaling_event: ScalingEvent) -> bool:
        """Execute scaling decision"""
        
        try:
            comprehensive_logger.info(
                f"Executing scaling: {scaling_event.direction.value} from {scaling_event.before_capacity} to {scaling_event.after_capacity}",
                additional_data={
                    "trigger": scaling_event.trigger.value,
                    "reason": scaling_event.reason
                }
            )
            
            # Cost optimization check
            if self.config["cost_optimization_enabled"] and scaling_event.direction == ScalingDirection.UP:
                optimization = self.cost_optimizer.optimize_capacity(
                    scaling_event.metrics.cpu_utilization,
                    scaling_event.metrics.memory_utilization
                )
                if optimization:
                    comprehensive_logger.info(
                        f"Cost optimization suggests {optimization['instance_type']} x{optimization['count']}",
                        additional_data=optimization
                    )
            
            # Execute scaling callbacks
            for callback in self.scale_callbacks:
                try:
                    callback(scaling_event)
                except Exception as e:
                    comprehensive_logger.error(f"Scaling callback failed: {e}")
            
            # Update capacity
            self.current_capacity = scaling_event.after_capacity
            self.last_scale_time = datetime.now()
            scaling_event.success = True
            
            # Record event
            self.scaling_events.append(scaling_event)
            
            comprehensive_logger.info(
                f"Scaling completed successfully: capacity now {self.current_capacity}",
                additional_data={
                    "scaling_event_id": id(scaling_event),
                    "new_capacity": self.current_capacity
                }
            )
            
            return True
            
        except Exception as e:
            scaling_event.success = False
            scaling_event.error_message = str(e)
            self.scaling_events.append(scaling_event)
            
            comprehensive_logger.error(
                f"Scaling failed: {e}",
                additional_data={
                    "scaling_event": scaling_event.__dict__
                }
            )
            
            return False
    
    def _calculate_average_metrics(self, metrics_list: List[ScalingMetrics]) -> ScalingMetrics:
        """Calculate average of metrics"""
        if not metrics_list:
            return ScalingMetrics()
        
        return ScalingMetrics(
            cpu_utilization=sum(m.cpu_utilization for m in metrics_list) / len(metrics_list),
            memory_utilization=sum(m.memory_utilization for m in metrics_list) / len(metrics_list),
            disk_io_utilization=sum(m.disk_io_utilization for m in metrics_list) / len(metrics_list),
            network_io_utilization=sum(m.network_io_utilization for m in metrics_list) / len(metrics_list),
            queue_length=int(sum(m.queue_length for m in metrics_list) / len(metrics_list)),
            active_connections=int(sum(m.active_connections for m in metrics_list) / len(metrics_list)),
            average_response_time=sum(m.average_response_time for m in metrics_list) / len(metrics_list),
            error_rate=sum(m.error_rate for m in metrics_list) / len(metrics_list),
            throughput=sum(m.throughput for m in metrics_list) / len(metrics_list),
            timestamp=datetime.now()
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                if self.is_enabled:
                    # Collect current metrics (placeholder - would integrate with actual monitoring)
                    current_metrics = self._collect_current_metrics()
                    
                    # Evaluate scaling
                    scaling_event = self.evaluate_scaling(current_metrics)
                    
                    if scaling_event:
                        self.execute_scaling(scaling_event)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                comprehensive_logger.error(f"Auto-scaler monitoring error: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        # This would integrate with actual monitoring systems
        # For now, return sample metrics
        import psutil
        import random
        
        return ScalingMetrics(
            cpu_utilization=psutil.cpu_percent(),
            memory_utilization=psutil.virtual_memory().percent,
            disk_io_utilization=random.uniform(10, 30),
            network_io_utilization=random.uniform(5, 25),
            queue_length=random.randint(0, 20),
            active_connections=random.randint(10, 100),
            average_response_time=random.uniform(100, 1000),
            error_rate=random.uniform(0, 5),
            throughput=random.uniform(50, 200)
        )
    
    def add_scaling_callback(self, callback: Callable[[ScalingEvent], None]):
        """Add callback for scaling events"""
        self.scale_callbacks.append(callback)
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add custom scaling rule"""
        self.scaling_rules.append(rule)
        comprehensive_logger.info(f"Added scaling rule: {rule.name}")
    
    def enable(self):
        """Enable auto-scaling"""
        self.is_enabled = True
        comprehensive_logger.info("Auto-scaling enabled")
    
    def disable(self):
        """Disable auto-scaling"""
        self.is_enabled = False
        comprehensive_logger.info("Auto-scaling disabled")
    
    def get_status(self) -> Dict[str, Any]:
        """Get auto-scaler status"""
        recent_events = [
            {
                "timestamp": event.timestamp.isoformat(),
                "direction": event.direction.value,
                "trigger": event.trigger.value,
                "before_capacity": event.before_capacity,
                "after_capacity": event.after_capacity,
                "success": event.success,
                "reason": event.reason
            }
            for event in list(self.scaling_events)[-10:]  # Last 10 events
        ]
        
        return {
            "enabled": self.is_enabled,
            "current_capacity": self.current_capacity,
            "min_capacity": self.min_capacity,
            "max_capacity": self.max_capacity,
            "last_scale_time": self.last_scale_time.isoformat(),
            "total_scaling_events": len(self.scaling_events),
            "recent_events": recent_events,
            "scaling_rules": len(self.scaling_rules),
            "configuration": self.config
        }
    
    def shutdown(self):
        """Shutdown auto-scaler"""
        comprehensive_logger.info("Shutting down auto-scaler")
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)


# Global auto-scaler instance
auto_scaler = AdvancedAutoScaler()