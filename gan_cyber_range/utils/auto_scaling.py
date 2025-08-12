"""
Auto-scaling and resource management for cyber range infrastructure.

This module provides intelligent auto-scaling capabilities to handle varying
workloads and optimize resource utilization in cyber range deployments.
"""

import logging
import asyncio
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"


class ResourceType(Enum):
    """Resource types for scaling"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    CONTAINERS = "containers"


@dataclass
class ScalingMetric:
    """Scaling metric configuration"""
    metric_name: str
    resource_type: ResourceType
    scale_up_threshold: float
    scale_down_threshold: float
    measurement_window: int = 300  # seconds
    cooldown_period: int = 300  # seconds
    weight: float = 1.0


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration"""
    policy_name: str
    target_resource: str
    metrics: List[ScalingMetric]
    min_instances: int = 1
    max_instances: int = 10
    scale_up_step: int = 1
    scale_down_step: int = 1
    enabled: bool = True


@dataclass
class ResourceUsage:
    """Current resource usage metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    network_io_mbps: float
    active_connections: int
    container_count: int


@dataclass
class ScalingEvent:
    """Scaling action event"""
    event_id: str
    timestamp: datetime
    policy_name: str
    resource_target: str
    scaling_direction: ScalingDirection
    trigger_metric: str
    trigger_value: float
    instances_before: int
    instances_after: int
    success: bool
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """System and application metrics collector"""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics_history = {}
        self.is_collecting = False
        self.collection_thread = None
        
    def start_collection(self) -> None:
        """Start metrics collection"""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self.collection_thread.start()
        
        logger.info("Metrics collection started")
        
    def stop_collection(self) -> None:
        """Stop metrics collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
            
        logger.info("Metrics collection stopped")
        
    def get_current_usage(self) -> ResourceUsage:
        """Get current system resource usage"""
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_percent = 0  # Placeholder - would need baseline
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_mbps = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)
        
        # Active connections
        connections = len(psutil.net_connections())
        
        # Container count (placeholder - would integrate with Docker API)
        container_count = self._get_container_count()
        
        return ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_io_percent=disk_io_percent,
            network_io_mbps=network_mbps,
            active_connections=connections,
            container_count=container_count
        )
        
    def get_metrics_history(self, metric_name: str, duration_minutes: int = 10) -> List[float]:
        """Get historical metrics for the specified duration"""
        
        if metric_name not in self.metrics_history:
            return []
            
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        # Filter metrics within time window
        recent_metrics = [
            (timestamp, value) for timestamp, value in self.metrics_history[metric_name]
            if timestamp >= cutoff_time
        ]
        
        return [value for _, value in recent_metrics]
        
    def _collect_metrics(self) -> None:
        """Background metrics collection loop"""
        
        while self.is_collecting:
            try:
                usage = self.get_current_usage()
                
                # Store metrics
                metrics = {
                    'cpu_percent': usage.cpu_percent,
                    'memory_percent': usage.memory_percent,
                    'disk_io_percent': usage.disk_io_percent,
                    'network_io_mbps': usage.network_io_mbps,
                    'active_connections': usage.active_connections,
                    'container_count': usage.container_count
                }
                
                for metric_name, value in metrics.items():
                    if metric_name not in self.metrics_history:
                        self.metrics_history[metric_name] = []
                        
                    self.metrics_history[metric_name].append((usage.timestamp, value))
                    
                    # Keep only last hour of data
                    cutoff_time = datetime.now() - timedelta(hours=1)
                    self.metrics_history[metric_name] = [
                        (ts, val) for ts, val in self.metrics_history[metric_name]
                        if ts >= cutoff_time
                    ]
                    
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
                
    def _get_container_count(self) -> int:
        """Get current container count"""
        try:
            import docker
            client = docker.from_env()
            containers = client.containers.list()
            return len(containers)
        except Exception:
            return 0


class AutoScaler:
    """Auto-scaling engine for cyber range resources"""
    
    def __init__(self):
        self.policies = {}
        self.metrics_collector = MetricsCollector()
        self.scaling_events = []
        self.last_scaling_actions = {}  # Track cooldown periods
        self.is_running = False
        self.scaling_thread = None
        
        # Load default policies
        self._load_default_policies()
        
    def add_scaling_policy(self, policy: ScalingPolicy) -> None:
        """Add auto-scaling policy"""
        self.policies[policy.policy_name] = policy
        logger.info(f"Added scaling policy: {policy.policy_name}")
        
    def remove_scaling_policy(self, policy_name: str) -> bool:
        """Remove scaling policy"""
        if policy_name in self.policies:
            del self.policies[policy_name]
            logger.info(f"Removed scaling policy: {policy_name}")
            return True
        return False
        
    def start_auto_scaling(self) -> None:
        """Start auto-scaling engine"""
        if self.is_running:
            return
            
        self.is_running = True
        self.metrics_collector.start_collection()
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logger.info("Auto-scaling engine started")
        
    def stop_auto_scaling(self) -> None:
        """Stop auto-scaling engine"""
        self.is_running = False
        self.metrics_collector.stop_collection()
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10)
            
        logger.info("Auto-scaling engine stopped")
        
    def evaluate_scaling_decision(
        self, 
        policy: ScalingPolicy,
        current_usage: ResourceUsage
    ) -> Optional[Dict[str, Any]]:
        """Evaluate if scaling action is needed"""
        
        scaling_decision = None
        
        for metric in policy.metrics:
            # Get historical values for metric
            metric_values = self._get_metric_value(metric, current_usage)
            history = self.metrics_collector.get_metrics_history(
                metric.metric_name, 
                metric.measurement_window // 60
            )
            
            if len(history) < 3:  # Need enough data points
                continue
                
            # Calculate average over measurement window
            avg_value = statistics.mean(history)
            
            # Check scaling thresholds
            if avg_value > metric.scale_up_threshold:
                scaling_decision = {
                    'action': ScalingDirection.SCALE_UP,
                    'trigger_metric': metric.metric_name,
                    'trigger_value': avg_value,
                    'threshold': metric.scale_up_threshold,
                    'confidence': min(1.0, avg_value / metric.scale_up_threshold)
                }
                break
                
            elif avg_value < metric.scale_down_threshold:
                scaling_decision = {
                    'action': ScalingDirection.SCALE_DOWN,
                    'trigger_metric': metric.metric_name,
                    'trigger_value': avg_value,
                    'threshold': metric.scale_down_threshold,
                    'confidence': min(1.0, metric.scale_down_threshold / max(avg_value, 0.1))
                }
                
        return scaling_decision
        
    def execute_scaling_action(
        self,
        policy: ScalingPolicy,
        scaling_decision: Dict[str, Any],
        current_instances: int
    ) -> ScalingEvent:
        """Execute scaling action"""
        
        import uuid
        event_id = str(uuid.uuid4())
        start_time = time.time()
        
        scaling_event = ScalingEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            policy_name=policy.policy_name,
            resource_target=policy.target_resource,
            scaling_direction=scaling_decision['action'],
            trigger_metric=scaling_decision['trigger_metric'],
            trigger_value=scaling_decision['trigger_value'],
            instances_before=current_instances,
            instances_after=current_instances,
            success=False,
            duration=0.0
        )
        
        try:
            # Calculate new instance count
            if scaling_decision['action'] == ScalingDirection.SCALE_UP:
                new_instances = min(
                    current_instances + policy.scale_up_step,
                    policy.max_instances
                )
            else:  # SCALE_DOWN
                new_instances = max(
                    current_instances - policy.scale_down_step,
                    policy.min_instances
                )
                
            if new_instances == current_instances:
                scaling_event.details['reason'] = 'No scaling needed - limits reached'
                scaling_event.success = True
                return scaling_event
                
            # Execute the scaling action
            success = self._perform_scaling(
                policy.target_resource,
                current_instances,
                new_instances,
                scaling_decision['action']
            )
            
            scaling_event.instances_after = new_instances if success else current_instances
            scaling_event.success = success
            scaling_event.duration = time.time() - start_time
            
            if success:
                logger.info(
                    f"Scaling successful: {policy.target_resource} "
                    f"{current_instances} -> {new_instances} instances"
                )
                
                # Update cooldown
                self.last_scaling_actions[policy.policy_name] = datetime.now()
            else:
                logger.error(f"Scaling failed for {policy.target_resource}")
                
        except Exception as e:
            scaling_event.success = False
            scaling_event.duration = time.time() - start_time
            scaling_event.details['error'] = str(e)
            logger.error(f"Scaling error: {e}")
            
        self.scaling_events.append(scaling_event)
        return scaling_event
        
    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Get scaling recommendations without executing"""
        
        recommendations = []
        current_usage = self.metrics_collector.get_current_usage()
        
        for policy in self.policies.values():
            if not policy.enabled:
                continue
                
            scaling_decision = self.evaluate_scaling_decision(policy, current_usage)
            
            if scaling_decision:
                current_instances = self._get_current_instance_count(policy.target_resource)
                
                recommendation = {
                    'policy_name': policy.policy_name,
                    'target_resource': policy.target_resource,
                    'current_instances': current_instances,
                    'recommended_action': scaling_decision['action'].value,
                    'trigger_metric': scaling_decision['trigger_metric'],
                    'trigger_value': scaling_decision['trigger_value'],
                    'confidence': scaling_decision['confidence'],
                    'urgency': 'high' if scaling_decision['confidence'] > 0.8 else 'medium'
                }
                
                recommendations.append(recommendation)
                
        return recommendations
        
    def get_scaling_history(self, hours: int = 24) -> List[ScalingEvent]:
        """Get scaling event history"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            event for event in self.scaling_events
            if event.timestamp >= cutoff_time
        ]
        
    def _scaling_loop(self) -> None:
        """Main auto-scaling loop"""
        
        while self.is_running:
            try:
                current_usage = self.metrics_collector.get_current_usage()
                
                for policy in self.policies.values():
                    if not policy.enabled:
                        continue
                        
                    # Check cooldown period
                    if policy.policy_name in self.last_scaling_actions:
                        last_action = self.last_scaling_actions[policy.policy_name]
                        cooldown_end = last_action + timedelta(seconds=300)  # 5 minute cooldown
                        if datetime.now() < cooldown_end:
                            continue
                            
                    # Evaluate scaling decision
                    scaling_decision = self.evaluate_scaling_decision(policy, current_usage)
                    
                    if scaling_decision:
                        current_instances = self._get_current_instance_count(policy.target_resource)
                        self.execute_scaling_action(policy, scaling_decision, current_instances)
                        
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(60)
                
    def _get_metric_value(self, metric: ScalingMetric, usage: ResourceUsage) -> float:
        """Extract metric value from usage data"""
        
        metric_map = {
            'cpu_percent': usage.cpu_percent,
            'memory_percent': usage.memory_percent,
            'disk_io_percent': usage.disk_io_percent,
            'network_io_mbps': usage.network_io_mbps,
            'active_connections': usage.active_connections,
            'container_count': usage.container_count
        }
        
        return metric_map.get(metric.metric_name, 0.0)
        
    def _get_current_instance_count(self, resource_target: str) -> int:
        """Get current instance count for resource"""
        
        # This would integrate with container orchestration platform
        # For now, return simulated count
        try:
            import docker
            client = docker.from_env()
            containers = client.containers.list(filters={'label': f'resource={resource_target}'})
            return len(containers)
        except Exception:
            return 1  # Default to 1 instance
            
    def _perform_scaling(
        self,
        resource_target: str,
        current_instances: int,
        new_instances: int,
        action: ScalingDirection
    ) -> bool:
        """Perform the actual scaling operation"""
        
        try:
            # This would integrate with container orchestration platform
            # For now, simulate scaling operation
            
            if action == ScalingDirection.SCALE_UP:
                # Would create new container instances
                logger.info(f"Scaling up {resource_target}: {current_instances} -> {new_instances}")
                
            elif action == ScalingDirection.SCALE_DOWN:
                # Would terminate container instances
                logger.info(f"Scaling down {resource_target}: {current_instances} -> {new_instances}")
                
            # Simulate operation delay
            time.sleep(2)
            
            return True  # Simulate success
            
        except Exception as e:
            logger.error(f"Scaling operation failed: {e}")
            return False
            
    def _load_default_policies(self) -> None:
        """Load default auto-scaling policies"""
        
        # CPU-based scaling policy
        cpu_policy = ScalingPolicy(
            policy_name="cpu_scaling",
            target_resource="cyber_range_workers",
            metrics=[
                ScalingMetric(
                    metric_name="cpu_percent",
                    resource_type=ResourceType.CPU,
                    scale_up_threshold=80.0,
                    scale_down_threshold=30.0,
                    measurement_window=300,
                    cooldown_period=300,
                    weight=1.0
                )
            ],
            min_instances=1,
            max_instances=5,
            scale_up_step=1,
            scale_down_step=1
        )
        
        # Memory-based scaling policy
        memory_policy = ScalingPolicy(
            policy_name="memory_scaling",
            target_resource="cyber_range_workers",
            metrics=[
                ScalingMetric(
                    metric_name="memory_percent",
                    resource_type=ResourceType.MEMORY,
                    scale_up_threshold=85.0,
                    scale_down_threshold=40.0,
                    measurement_window=300,
                    cooldown_period=300,
                    weight=1.0
                )
            ],
            min_instances=1,
            max_instances=3,
            scale_up_step=1,
            scale_down_step=1
        )
        
        # Connection-based scaling policy
        connection_policy = ScalingPolicy(
            policy_name="connection_scaling",
            target_resource="api_servers",
            metrics=[
                ScalingMetric(
                    metric_name="active_connections",
                    resource_type=ResourceType.NETWORK,
                    scale_up_threshold=100.0,
                    scale_down_threshold=20.0,
                    measurement_window=180,
                    cooldown_period=240,
                    weight=0.8
                )
            ],
            min_instances=1,
            max_instances=10,
            scale_up_step=2,
            scale_down_step=1
        )
        
        self.add_scaling_policy(cpu_policy)
        self.add_scaling_policy(memory_policy)
        self.add_scaling_policy(connection_policy)


class LoadBalancer:
    """Intelligent load balancer for cyber range services"""
    
    def __init__(self):
        self.backends = {}
        self.health_checks = {}
        self.routing_algorithms = {
            'round_robin': self._round_robin,
            'least_connections': self._least_connections,
            'weighted_round_robin': self._weighted_round_robin,
            'least_response_time': self._least_response_time
        }
        self.current_algorithm = 'round_robin'
        self.round_robin_counter = 0
        
    def add_backend(
        self,
        service_name: str,
        backend_url: str,
        weight: float = 1.0,
        max_connections: int = 100
    ) -> None:
        """Add backend server"""
        
        if service_name not in self.backends:
            self.backends[service_name] = []
            
        backend = {
            'url': backend_url,
            'weight': weight,
            'max_connections': max_connections,
            'current_connections': 0,
            'response_times': [],
            'healthy': True,
            'last_health_check': datetime.now()
        }
        
        self.backends[service_name].append(backend)
        logger.info(f"Added backend {backend_url} for service {service_name}")
        
    def remove_backend(self, service_name: str, backend_url: str) -> bool:
        """Remove backend server"""
        
        if service_name not in self.backends:
            return False
            
        self.backends[service_name] = [
            backend for backend in self.backends[service_name]
            if backend['url'] != backend_url
        ]
        
        logger.info(f"Removed backend {backend_url} from service {service_name}")
        return True
        
    def route_request(self, service_name: str) -> Optional[str]:
        """Route request to appropriate backend"""
        
        if service_name not in self.backends:
            return None
            
        healthy_backends = [
            backend for backend in self.backends[service_name]
            if backend['healthy']
        ]
        
        if not healthy_backends:
            logger.warning(f"No healthy backends available for service {service_name}")
            return None
            
        # Use configured routing algorithm
        algorithm = self.routing_algorithms.get(self.current_algorithm, self._round_robin)
        selected_backend = algorithm(healthy_backends)
        
        if selected_backend:
            selected_backend['current_connections'] += 1
            return selected_backend['url']
            
        return None
        
    def _round_robin(self, backends: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Round-robin load balancing"""
        if not backends:
            return None
            
        backend = backends[self.round_robin_counter % len(backends)]
        self.round_robin_counter += 1
        return backend
        
    def _least_connections(self, backends: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Least connections load balancing"""
        if not backends:
            return None
            
        return min(backends, key=lambda b: b['current_connections'])
        
    def _weighted_round_robin(self, backends: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Weighted round-robin load balancing"""
        if not backends:
            return None
            
        # Simple weighted selection based on weights
        total_weight = sum(backend['weight'] for backend in backends)
        if total_weight == 0:
            return self._round_robin(backends)
            
        # Use round-robin with weight consideration
        weighted_backends = []
        for backend in backends:
            weight_factor = int(backend['weight'] * 10)  # Scale for integer math
            weighted_backends.extend([backend] * weight_factor)
            
        if weighted_backends:
            return self._round_robin(weighted_backends)
            
        return backends[0]
        
    def _least_response_time(self, backends: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Least response time load balancing"""
        if not backends:
            return None
            
        # Select backend with lowest average response time
        def avg_response_time(backend):
            times = backend['response_times']
            return statistics.mean(times) if times else 0
            
        return min(backends, key=avg_response_time)


# Global auto-scaler instance
auto_scaler = AutoScaler()
load_balancer = LoadBalancer()


def start_auto_scaling() -> None:
    """Start auto-scaling for the cyber range"""
    auto_scaler.start_auto_scaling()


def stop_auto_scaling() -> None:
    """Stop auto-scaling"""
    auto_scaler.stop_auto_scaling()


def get_scaling_status() -> Dict[str, Any]:
    """Get current auto-scaling status"""
    
    recommendations = auto_scaler.get_scaling_recommendations()
    recent_events = auto_scaler.get_scaling_history(hours=1)
    
    return {
        'auto_scaling_enabled': auto_scaler.is_running,
        'active_policies': len([p for p in auto_scaler.policies.values() if p.enabled]),
        'pending_recommendations': len(recommendations),
        'recent_scaling_events': len(recent_events),
        'recommendations': recommendations,
        'recent_events': [
            {
                'timestamp': event.timestamp.isoformat(),
                'action': event.scaling_direction.value,
                'resource': event.resource_target,
                'success': event.success
            }
            for event in recent_events[-5:]  # Last 5 events
        ]
    }