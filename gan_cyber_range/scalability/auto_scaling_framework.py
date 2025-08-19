"""
Auto-scaling framework for GAN-Cyber-Range-v2.
Implements intelligent scaling based on load, resource usage, and performance metrics.
"""

import logging
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling directions"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    """Types of scaling triggers"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    CUSTOM_METRIC = "custom_metric"


class ComponentType(Enum):
    """Types of scalable components"""
    ATTACK_GENERATOR = "attack_generator"
    RANGE_SIMULATOR = "range_simulator"
    BLUE_TEAM_ANALYZER = "blue_team_analyzer"
    DATA_PROCESSOR = "data_processor"
    API_WORKER = "api_worker"
    CACHE_LAYER = "cache_layer"


@dataclass
class ScalingMetric:
    """Metrics for scaling decisions"""
    metric_name: str
    current_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    cooldown_seconds: int = 300
    last_triggered: Optional[datetime] = None


@dataclass
class ScalingRule:
    """Rules for component scaling"""
    component_type: ComponentType
    min_instances: int
    max_instances: int
    metrics: List[ScalingMetric]
    scale_up_step: int = 1
    scale_down_step: int = 1
    evaluation_period_seconds: int = 60
    enabled: bool = True


@dataclass
class ComponentInstance:
    """Represents a scalable component instance"""
    instance_id: str
    component_type: ComponentType
    status: str
    created_at: datetime
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoadBalancer:
    """Intelligent load balancing for scaled components"""
    
    def __init__(self):
        self.instance_pools = defaultdict(list)
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(deque)
        self.health_status = defaultdict(bool)
        self.algorithms = {
            'round_robin': self._round_robin,
            'least_connections': self._least_connections,
            'weighted_response_time': self._weighted_response_time,
            'adaptive': self._adaptive_routing
        }
        self.current_algorithm = 'adaptive'
        self.robin_counters = defaultdict(int)
    
    def add_instance(self, component_type: ComponentType, instance: ComponentInstance):
        """Add instance to load balancer pool"""
        self.instance_pools[component_type].append(instance)
        self.health_status[instance.instance_id] = True
        logger.info(f"Added instance {instance.instance_id} to {component_type.value} pool")
    
    def remove_instance(self, component_type: ComponentType, instance_id: str):
        """Remove instance from load balancer pool"""
        instances = self.instance_pools[component_type]
        self.instance_pools[component_type] = [
            inst for inst in instances if inst.instance_id != instance_id
        ]
        if instance_id in self.health_status:
            del self.health_status[instance_id]
        logger.info(f"Removed instance {instance_id} from {component_type.value} pool")
    
    def get_instance(self, component_type: ComponentType) -> Optional[ComponentInstance]:
        """Get best instance for request using current algorithm"""
        instances = self.instance_pools[component_type]
        healthy_instances = [
            inst for inst in instances 
            if self.health_status.get(inst.instance_id, False)
        ]
        
        if not healthy_instances:
            return None
        
        algorithm = self.algorithms[self.current_algorithm]
        return algorithm(component_type, healthy_instances)
    
    def _round_robin(self, component_type: ComponentType, instances: List[ComponentInstance]) -> ComponentInstance:
        """Round-robin load balancing"""
        if not instances:
            return None
        
        counter = self.robin_counters[component_type]
        instance = instances[counter % len(instances)]
        self.robin_counters[component_type] = (counter + 1) % len(instances)
        return instance
    
    def _least_connections(self, component_type: ComponentType, instances: List[ComponentInstance]) -> ComponentInstance:
        """Least connections load balancing"""
        if not instances:
            return None
        
        return min(instances, key=lambda inst: self.request_counts[inst.instance_id])
    
    def _weighted_response_time(self, component_type: ComponentType, instances: List[ComponentInstance]) -> ComponentInstance:
        """Weighted response time load balancing"""
        if not instances:
            return None
        
        best_instance = None
        best_score = float('inf')
        
        for instance in instances:
            response_times = self.response_times[instance.instance_id]
            avg_response_time = sum(response_times) / max(1, len(response_times))
            connection_count = self.request_counts[instance.instance_id]
            
            # Lower score is better
            score = avg_response_time * (1 + connection_count * 0.1)
            
            if score < best_score:
                best_score = score
                best_instance = instance
        
        return best_instance
    
    def _adaptive_routing(self, component_type: ComponentType, instances: List[ComponentInstance]) -> ComponentInstance:
        """Adaptive routing based on multiple factors"""
        if not instances:
            return None
        
        # Use weighted response time for now, can be enhanced
        return self._weighted_response_time(component_type, instances)
    
    def record_request(self, instance_id: str, response_time_ms: float):
        """Record request metrics for load balancing decisions"""
        self.request_counts[instance_id] += 1
        
        # Keep only recent response times (last 100)
        response_times = self.response_times[instance_id]
        response_times.append(response_time_ms)
        if len(response_times) > 100:
            response_times.popleft()
    
    def update_health_status(self, instance_id: str, is_healthy: bool):
        """Update instance health status"""
        self.health_status[instance_id] = is_healthy
    
    def get_pool_status(self, component_type: ComponentType) -> Dict[str, Any]:
        """Get status of instance pool"""
        instances = self.instance_pools[component_type]
        healthy_count = sum(
            1 for inst in instances 
            if self.health_status.get(inst.instance_id, False)
        )
        
        return {
            'total_instances': len(instances),
            'healthy_instances': healthy_count,
            'unhealthy_instances': len(instances) - healthy_count,
            'algorithm': self.current_algorithm
        }


class MetricsCollector:
    """Collect and aggregate metrics for scaling decisions"""
    
    def __init__(self):
        self.metrics_history = defaultdict(deque)
        self.current_metrics = {}
        self.collection_interval = 30  # seconds
        self.max_history_size = 1000
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        try:
            # Try to use actual system monitoring
            import psutil
            return self._collect_with_psutil()
        except ImportError:
            # Fallback to simulated metrics
            return self._collect_simulated_metrics()
    
    def _collect_with_psutil(self) -> Dict[str, float]:
        """Collect metrics using psutil"""
        import psutil
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        return {
            'cpu_utilization': cpu_percent,
            'memory_utilization': memory_percent,
            'disk_utilization': disk_percent,
            'cpu_cores': cpu_count,
            'memory_total_gb': memory.total / (1024**3),
            'load_average_1m': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else cpu_percent / 100
        }
    
    def _collect_simulated_metrics(self) -> Dict[str, float]:
        """Collect simulated metrics for development"""
        import random
        import time
        
        # Generate realistic-looking metrics
        base_cpu = 30 + random.uniform(-10, 30)
        base_memory = 40 + random.uniform(-15, 40)
        
        return {
            'cpu_utilization': max(0, min(100, base_cpu)),
            'memory_utilization': max(0, min(100, base_memory)),
            'disk_utilization': 50 + random.uniform(-10, 20),
            'cpu_cores': 4,
            'memory_total_gb': 16,
            'load_average_1m': base_cpu / 100
        }
    
    def collect_application_metrics(self, component_type: ComponentType) -> Dict[str, float]:
        """Collect application-specific metrics"""
        # This would be implemented based on actual application instrumentation
        # For now, return simulated metrics
        import random
        
        base_metrics = {
            'request_rate': random.uniform(10, 100),
            'response_time_ms': random.uniform(50, 500),
            'error_rate': random.uniform(0, 5),
            'queue_length': random.randint(0, 50),
            'active_connections': random.randint(5, 100)
        }
        
        # Add component-specific metrics
        if component_type == ComponentType.ATTACK_GENERATOR:
            base_metrics.update({
                'attacks_generated_per_sec': random.uniform(1, 10),
                'generation_queue_length': random.randint(0, 20)
            })
        elif component_type == ComponentType.RANGE_SIMULATOR:
            base_metrics.update({
                'active_simulations': random.randint(1, 10),
                'simulation_cpu_usage': random.uniform(20, 80)
            })
        
        return base_metrics
    
    def record_metrics(self, component_type: ComponentType, metrics: Dict[str, float]):
        """Record metrics with timestamp"""
        timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            key = f"{component_type.value}_{metric_name}"
            history = self.metrics_history[key]
            history.append((timestamp, value))
            
            # Keep history size manageable
            if len(history) > self.max_history_size:
                history.popleft()
            
            # Update current metrics
            self.current_metrics[key] = value
    
    def get_metric_statistics(self, component_type: ComponentType, metric_name: str, 
                            time_window_minutes: int = 5) -> Dict[str, float]:
        """Get statistics for a metric over time window"""
        key = f"{component_type.value}_{metric_name}"
        history = self.metrics_history[key]
        
        if not history:
            return {'avg': 0, 'min': 0, 'max': 0, 'current': 0}
        
        # Filter to time window
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_values = [
            value for timestamp, value in history 
            if timestamp > cutoff_time
        ]
        
        if not recent_values:
            recent_values = [history[-1][1]]  # Use most recent value
        
        return {
            'avg': sum(recent_values) / len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'current': recent_values[-1] if recent_values else 0,
            'count': len(recent_values)
        }


class AutoScaler:
    """Main auto-scaling orchestrator"""
    
    def __init__(self):
        self.scaling_rules = {}
        self.component_instances = defaultdict(list)
        self.load_balancer = LoadBalancer()
        self.metrics_collector = MetricsCollector()
        self.scaling_active = False
        self.scaling_thread = None
        self.scaling_history = []
    
    def register_scaling_rule(self, rule: ScalingRule):
        """Register scaling rule for component type"""
        self.scaling_rules[rule.component_type] = rule
        logger.info(f"Registered scaling rule for {rule.component_type.value}")
    
    def add_component_instance(self, instance: ComponentInstance):
        """Add component instance to scaling management"""
        self.component_instances[instance.component_type].append(instance)
        self.load_balancer.add_instance(instance.component_type, instance)
        logger.info(f"Added instance {instance.instance_id} for scaling")
    
    def remove_component_instance(self, component_type: ComponentType, instance_id: str):
        """Remove component instance from scaling management"""
        instances = self.component_instances[component_type]
        self.component_instances[component_type] = [
            inst for inst in instances if inst.instance_id != instance_id
        ]
        self.load_balancer.remove_instance(component_type, instance_id)
        logger.info(f"Removed instance {instance_id} from scaling")
    
    def start_auto_scaling(self):
        """Start auto-scaling monitoring and decisions"""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop auto-scaling"""
        self.scaling_active = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10)
        logger.info("Auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main scaling decision loop"""
        while self.scaling_active:
            try:
                for component_type, rule in self.scaling_rules.items():
                    if rule.enabled:
                        self._evaluate_scaling_rule(component_type, rule)
                
                time.sleep(30)  # Evaluate every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _evaluate_scaling_rule(self, component_type: ComponentType, rule: ScalingRule):
        """Evaluate scaling rule and make scaling decisions"""
        current_instances = len(self.component_instances[component_type])
        
        # Collect current metrics
        system_metrics = self.metrics_collector.collect_system_metrics()
        app_metrics = self.metrics_collector.collect_application_metrics(component_type)
        
        # Record metrics
        all_metrics = {**system_metrics, **app_metrics}
        self.metrics_collector.record_metrics(component_type, all_metrics)
        
        # Evaluate each metric in the rule
        scale_up_votes = 0
        scale_down_votes = 0
        total_weight = 0
        
        for metric in rule.metrics:
            if metric.metric_name in all_metrics:
                current_value = all_metrics[metric.metric_name]
                weight = metric.weight
                total_weight += weight
                
                # Check if metric is in cooldown
                if (metric.last_triggered and 
                    (datetime.now() - metric.last_triggered).total_seconds() < metric.cooldown_seconds):
                    continue
                
                # Evaluate scaling thresholds
                if current_value > metric.threshold_up:
                    scale_up_votes += weight
                    metric.last_triggered = datetime.now()
                elif current_value < metric.threshold_down:
                    scale_down_votes += weight
                    metric.last_triggered = datetime.now()
        
        # Make scaling decision
        if total_weight > 0:
            scale_up_ratio = scale_up_votes / total_weight
            scale_down_ratio = scale_down_votes / total_weight
            
            if scale_up_ratio > 0.5 and current_instances < rule.max_instances:
                self._scale_up(component_type, rule)
            elif scale_down_ratio > 0.5 and current_instances > rule.min_instances:
                self._scale_down(component_type, rule)
    
    def _scale_up(self, component_type: ComponentType, rule: ScalingRule):
        """Scale up component instances"""
        current_count = len(self.component_instances[component_type])
        target_count = min(current_count + rule.scale_up_step, rule.max_instances)
        
        instances_to_add = target_count - current_count
        
        for i in range(instances_to_add):
            instance = self._create_instance(component_type)
            self.add_component_instance(instance)
        
        scaling_event = {
            'timestamp': datetime.now(),
            'component_type': component_type.value,
            'action': 'scale_up',
            'from_count': current_count,
            'to_count': target_count,
            'instances_added': instances_to_add
        }
        
        self.scaling_history.append(scaling_event)
        logger.info(f"Scaled up {component_type.value} from {current_count} to {target_count} instances")
    
    def _scale_down(self, component_type: ComponentType, rule: ScalingRule):
        """Scale down component instances"""
        current_count = len(self.component_instances[component_type])
        target_count = max(current_count - rule.scale_down_step, rule.min_instances)
        
        instances_to_remove = current_count - target_count
        
        # Remove least utilized instances
        instances = self.component_instances[component_type]
        for i in range(instances_to_remove):
            if instances:
                # Simple strategy: remove oldest instance
                # In production, this would consider utilization
                instance_to_remove = instances[0]
                self.remove_component_instance(component_type, instance_to_remove.instance_id)
        
        scaling_event = {
            'timestamp': datetime.now(),
            'component_type': component_type.value,
            'action': 'scale_down',
            'from_count': current_count,
            'to_count': target_count,
            'instances_removed': instances_to_remove
        }
        
        self.scaling_history.append(scaling_event)
        logger.info(f"Scaled down {component_type.value} from {current_count} to {target_count} instances")
    
    def _create_instance(self, component_type: ComponentType) -> ComponentInstance:
        """Create new component instance"""
        import uuid
        
        instance_id = f"{component_type.value}_{uuid.uuid4().hex[:8]}"
        
        instance = ComponentInstance(
            instance_id=instance_id,
            component_type=component_type,
            status="starting",
            created_at=datetime.now(),
            metadata={'auto_created': True}
        )
        
        # Simulate instance startup
        instance.status = "running"
        
        return instance
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        status = {
            'scaling_active': self.scaling_active,
            'components': {},
            'recent_scaling_events': self.scaling_history[-10:] if self.scaling_history else []
        }
        
        for component_type in ComponentType:
            instances = self.component_instances[component_type]
            rule = self.scaling_rules.get(component_type)
            
            status['components'][component_type.value] = {
                'current_instances': len(instances),
                'min_instances': rule.min_instances if rule else 0,
                'max_instances': rule.max_instances if rule else 0,
                'scaling_enabled': rule.enabled if rule else False,
                'load_balancer_status': self.load_balancer.get_pool_status(component_type)
            }
        
        return status
    
    def manual_scale(self, component_type: ComponentType, target_instances: int) -> bool:
        """Manually scale component to target instance count"""
        current_count = len(self.component_instances[component_type])
        
        if target_instances > current_count:
            # Scale up
            for i in range(target_instances - current_count):
                instance = self._create_instance(component_type)
                self.add_component_instance(instance)
        elif target_instances < current_count:
            # Scale down
            instances = self.component_instances[component_type]
            for i in range(current_count - target_instances):
                if instances:
                    instance_to_remove = instances[0]
                    self.remove_component_instance(component_type, instance_to_remove.instance_id)
        
        logger.info(f"Manually scaled {component_type.value} to {target_instances} instances")
        return True


# Global auto-scaler instance
auto_scaler = AutoScaler()