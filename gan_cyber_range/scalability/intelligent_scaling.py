#!/usr/bin/env python3
"""
Intelligent auto-scaling system for defensive cybersecurity platform
Implements predictive scaling, load balancing, and resource optimization
"""

import time
import threading
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import uuid
import queue

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction enumeration"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Resource type enumeration"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    CUSTOM = "custom"


@dataclass
class ScalingMetric:
    """Scaling metric data structure"""
    metric_name: str
    current_value: float
    threshold_up: float
    threshold_down: float
    weight: float
    resource_type: ResourceType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ScalingDecision:
    """Scaling decision data structure"""
    decision_id: str
    direction: ScalingDirection
    magnitude: int
    confidence: float
    reasoning: str
    metrics_analyzed: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    executed: bool = False


@dataclass
class ResourceInstance:
    """Resource instance representation"""
    instance_id: str
    instance_type: str
    status: str
    cpu_capacity: float
    memory_capacity: float
    current_load: float
    created_at: str
    health_score: float = 1.0


class PredictiveScaler:
    """Predictive scaling engine with machine learning-like behavior"""
    
    def __init__(self, 
                 prediction_window: int = 300,  # 5 minutes
                 history_size: int = 1000):
        self.prediction_window = prediction_window
        self.history_size = history_size
        self.metric_history = []
        self.scaling_history = []
        self.prediction_accuracy = 0.8
        
        logger.info("Predictive scaler initialized")
    
    def add_metric_point(self, metrics: Dict[str, float]) -> None:
        """Add metric point to history"""
        data_point = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy()
        }
        
        self.metric_history.append(data_point)
        
        # Maintain history size
        if len(self.metric_history) > self.history_size:
            self.metric_history = self.metric_history[-self.history_size:]
    
    def predict_load(self, metric_name: str, minutes_ahead: int = 5) -> Tuple[float, float]:
        """Predict future load for given metric"""
        
        if len(self.metric_history) < 10:
            # Not enough data for prediction
            recent = self.metric_history[-1] if self.metric_history else {"metrics": {metric_name: 0.5}}
            current_value = recent["metrics"].get(metric_name, 0.5)
            return current_value, 0.1
        
        # Extract recent values
        recent_values = []
        for point in self.metric_history[-20:]:  # Last 20 points
            if metric_name in point["metrics"]:
                recent_values.append(point["metrics"][metric_name])
        
        if not recent_values:
            return 0.5, 0.1
        
        # Simple trend analysis
        if len(recent_values) >= 3:
            # Linear trend approximation
            trend = (recent_values[-1] - recent_values[-3]) / 2
            predicted_value = recent_values[-1] + (trend * minutes_ahead / 5)
        else:
            predicted_value = statistics.mean(recent_values)
        
        # Calculate confidence based on variance
        if len(recent_values) >= 2:
            variance = statistics.variance(recent_values)
            confidence = max(0.1, min(0.9, 1.0 - variance))
        else:
            confidence = 0.5
        
        # Bound prediction
        predicted_value = max(0.0, min(1.0, predicted_value))
        
        return predicted_value, confidence
    
    def analyze_scaling_patterns(self) -> Dict[str, Any]:
        """Analyze historical scaling patterns"""
        if not self.scaling_history:
            return {"patterns": [], "recommendations": []}
        
        # Analyze time-based patterns
        time_patterns = {}
        for decision in self.scaling_history[-50:]:  # Last 50 decisions
            timestamp = datetime.fromisoformat(decision.timestamp)
            hour = timestamp.hour
            
            if hour not in time_patterns:
                time_patterns[hour] = {"up": 0, "down": 0, "stable": 0}
            
            time_patterns[hour][decision.direction.value] += 1
        
        # Generate recommendations
        recommendations = []
        for hour, patterns in time_patterns.items():
            total = sum(patterns.values())
            if total >= 3:  # Enough data
                if patterns["up"] / total > 0.6:
                    recommendations.append(f"Consider proactive scaling up at hour {hour}")
                elif patterns["down"] / total > 0.6:
                    recommendations.append(f"Consider proactive scaling down at hour {hour}")
        
        return {
            "time_patterns": time_patterns,
            "recommendations": recommendations,
            "total_decisions": len(self.scaling_history)
        }


class LoadBalancer:
    """Intelligent load balancer for defensive workloads"""
    
    def __init__(self):
        self.instances = {}
        self.routing_algorithm = "weighted_round_robin"
        self.health_check_interval = 30  # seconds
        self.last_health_check = datetime.now()
        
        # Load balancing state
        self.current_index = 0
        self.request_count = 0
        
        logger.info("Load balancer initialized")
    
    def register_instance(self, instance: ResourceInstance) -> None:
        """Register a resource instance"""
        self.instances[instance.instance_id] = instance
        logger.info(f"Registered instance: {instance.instance_id} ({instance.instance_type})")
    
    def unregister_instance(self, instance_id: str) -> None:
        """Unregister a resource instance"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            logger.info(f"Unregistered instance: {instance_id}")
    
    def select_instance(self, workload_requirements: Dict[str, Any] = None) -> Optional[str]:
        """Select best instance for workload"""
        
        if not self.instances:
            return None
        
        healthy_instances = [
            inst for inst in self.instances.values()
            if inst.status == "running" and inst.health_score >= 0.7
        ]
        
        if not healthy_instances:
            # Fall back to any available instance
            healthy_instances = [inst for inst in self.instances.values() if inst.status == "running"]
        
        if not healthy_instances:
            return None
        
        # Select based on algorithm
        if self.routing_algorithm == "least_connections":
            return self._select_least_loaded(healthy_instances)
        elif self.routing_algorithm == "weighted_round_robin":
            return self._select_weighted_round_robin(healthy_instances)
        elif self.routing_algorithm == "resource_aware":
            return self._select_resource_aware(healthy_instances, workload_requirements)
        else:
            # Default round robin
            return self._select_round_robin(healthy_instances)
    
    def _select_round_robin(self, instances: List[ResourceInstance]) -> str:
        """Simple round robin selection"""
        if not instances:
            return None
        
        selected = instances[self.current_index % len(instances)]
        self.current_index += 1
        return selected.instance_id
    
    def _select_least_loaded(self, instances: List[ResourceInstance]) -> str:
        """Select instance with lowest current load"""
        return min(instances, key=lambda x: x.current_load).instance_id
    
    def _select_weighted_round_robin(self, instances: List[ResourceInstance]) -> str:
        """Weighted round robin based on capacity"""
        # Weight by inverse of current load and capacity
        weights = []
        for inst in instances:
            weight = (inst.cpu_capacity + inst.memory_capacity) / max(0.1, inst.current_load + 0.1)
            weights.append(weight * inst.health_score)
        
        # Select based on weights
        total_weight = sum(weights)
        if total_weight == 0:
            return self._select_round_robin(instances)
        
        # Weighted random selection
        import random
        r = random.uniform(0, total_weight)
        for i, weight in enumerate(weights):
            r -= weight
            if r <= 0:
                return instances[i].instance_id
        
        return instances[-1].instance_id
    
    def _select_resource_aware(self, 
                             instances: List[ResourceInstance],
                             requirements: Dict[str, Any]) -> str:
        """Select instance based on workload requirements"""
        
        if not requirements:
            return self._select_least_loaded(instances)
        
        cpu_req = requirements.get("cpu", 0.1)
        memory_req = requirements.get("memory", 0.1)
        
        # Score instances based on requirement fit
        scored_instances = []
        for inst in instances:
            available_cpu = inst.cpu_capacity - (inst.current_load * inst.cpu_capacity)
            available_memory = inst.memory_capacity - (inst.current_load * inst.memory_capacity)
            
            # Score based on resource availability
            cpu_score = min(1.0, available_cpu / cpu_req) if cpu_req > 0 else 1.0
            memory_score = min(1.0, available_memory / memory_req) if memory_req > 0 else 1.0
            
            # Combined score with health
            total_score = (cpu_score + memory_score) * 0.5 * inst.health_score
            
            scored_instances.append((inst, total_score))
        
        # Select best scoring instance
        if scored_instances:
            best_instance = max(scored_instances, key=lambda x: x[1])[0]
            return best_instance.instance_id
        
        return self._select_least_loaded(instances)
    
    def update_instance_load(self, instance_id: str, load: float) -> None:
        """Update instance load"""
        if instance_id in self.instances:
            self.instances[instance_id].current_load = max(0.0, min(1.0, load))
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all instances"""
        
        now = datetime.now()
        if now - self.last_health_check < timedelta(seconds=self.health_check_interval):
            return {"status": "skipped", "reason": "too_frequent"}
        
        self.last_health_check = now
        
        health_report = {
            "timestamp": now.isoformat(),
            "total_instances": len(self.instances),
            "healthy_instances": 0,
            "unhealthy_instances": 0,
            "instance_details": {}
        }
        
        for instance_id, instance in self.instances.items():
            # Simulate health check (in real system, this would be actual checks)
            health_score = max(0.0, instance.health_score - (instance.current_load * 0.1))
            
            # Update health score
            instance.health_score = health_score
            
            # Determine health status
            if health_score >= 0.7 and instance.status == "running":
                health_status = "healthy"
                health_report["healthy_instances"] += 1
            else:
                health_status = "unhealthy"
                health_report["unhealthy_instances"] += 1
            
            health_report["instance_details"][instance_id] = {
                "status": instance.status,
                "health_score": health_score,
                "health_status": health_status,
                "current_load": instance.current_load
            }
        
        logger.info(f"Health check completed: {health_report['healthy_instances']}/{health_report['total_instances']} healthy")
        return health_report


class IntelligentAutoScaler:
    """Comprehensive auto-scaling system with predictive capabilities"""
    
    def __init__(self, 
                 min_instances: int = 2,
                 max_instances: int = 20,
                 target_cpu_utilization: float = 0.7,
                 scale_cooldown: int = 300):  # 5 minutes
        
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu_utilization = target_cpu_utilization
        self.scale_cooldown = scale_cooldown
        
        # Components
        self.predictor = PredictiveScaler()
        self.load_balancer = LoadBalancer()
        
        # Scaling state
        self.last_scaling_action = datetime.now() - timedelta(seconds=scale_cooldown)
        self.scaling_decisions = []
        self.current_instances = min_instances
        
        # Metrics
        self.scaling_metrics = [
            ScalingMetric("cpu_utilization", 0.5, 0.8, 0.3, 1.0, ResourceType.CPU),
            ScalingMetric("memory_utilization", 0.5, 0.8, 0.3, 0.8, ResourceType.MEMORY),
            ScalingMetric("network_io", 0.3, 0.7, 0.2, 0.5, ResourceType.NETWORK),
            ScalingMetric("response_time", 1.0, 5.0, 1.0, 0.9, ResourceType.CUSTOM),
            ScalingMetric("error_rate", 0.01, 0.05, 0.001, 1.0, ResourceType.CUSTOM)
        ]
        
        # Monitoring
        self.monitoring = False
        self.monitor_thread = None
        
        logger.info(f"Intelligent auto-scaler initialized (min={min_instances}, max={max_instances})")
    
    def start_monitoring(self) -> None:
        """Start auto-scaling monitoring"""
        if self.monitoring:
            logger.warning("Auto-scaler monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Initialize instances
        self._initialize_instances()
        
        logger.info("Auto-scaler monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop auto-scaling monitoring"""
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        logger.info("Auto-scaler monitoring stopped")
    
    def _initialize_instances(self) -> None:
        """Initialize minimum required instances"""
        for i in range(self.min_instances):
            instance = ResourceInstance(
                instance_id=f"instance_{i:03d}",
                instance_type="defensive_worker",
                status="running",
                cpu_capacity=1.0,
                memory_capacity=1.0,
                current_load=0.0,
                created_at=datetime.now().isoformat()
            )
            
            self.load_balancer.register_instance(instance)
        
        logger.info(f"Initialized {self.min_instances} instances")
    
    def update_metrics(self, metric_updates: Dict[str, float]) -> None:
        """Update scaling metrics"""
        
        # Update existing metrics
        for metric in self.scaling_metrics:
            if metric.metric_name in metric_updates:
                metric.current_value = metric_updates[metric.metric_name]
                metric.timestamp = datetime.now().isoformat()
        
        # Add to predictor history
        self.predictor.add_metric_point(metric_updates)
        
        logger.debug(f"Updated metrics: {list(metric_updates.keys())}")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for auto-scaling"""
        logger.info("Auto-scaling monitoring loop started")
        
        while self.monitoring:
            try:
                # Collect current metrics (in real system, these would be actual system metrics)
                current_metrics = self._collect_current_metrics()
                self.update_metrics(current_metrics)
                
                # Make scaling decision
                decision = self._make_scaling_decision()
                
                if decision.direction != ScalingDirection.STABLE:
                    # Execute scaling action
                    self._execute_scaling_decision(decision)
                
                # Update instance loads
                self._update_instance_loads()
                
                # Perform health check
                self.load_balancer.health_check()
                
                # Sleep before next iteration
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Longer sleep on error
        
        logger.info("Auto-scaling monitoring loop stopped")
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics (simulated)"""
        
        # Simulate realistic metrics
        base_load = 0.4 + 0.3 * (time.time() % 3600) / 3600  # Hourly variation
        
        return {
            "cpu_utilization": min(0.95, base_load + 0.1 * (hash(str(time.time())) % 100) / 100),
            "memory_utilization": min(0.9, base_load + 0.05 * (hash(str(time.time() + 1)) % 100) / 100),
            "network_io": min(0.8, base_load * 0.7 + 0.2 * (hash(str(time.time() + 2)) % 100) / 100),
            "response_time": max(0.5, 1.0 + base_load * 3 + (hash(str(time.time() + 3)) % 100) / 100),
            "error_rate": max(0.001, min(0.1, 0.01 + base_load * 0.02))
        }
    
    def _make_scaling_decision(self) -> ScalingDecision:
        """Make intelligent scaling decision"""
        
        decision_id = str(uuid.uuid4())
        reasons = []
        scale_up_signals = 0
        scale_down_signals = 0
        confidence_sum = 0.0
        
        # Analyze current metrics
        for metric in self.scaling_metrics:
            if metric.current_value >= metric.threshold_up:
                scale_up_signals += metric.weight
                reasons.append(f"{metric.metric_name} high ({metric.current_value:.3f} >= {metric.threshold_up})")
            elif metric.current_value <= metric.threshold_down:
                scale_down_signals += metric.weight
                reasons.append(f"{metric.metric_name} low ({metric.current_value:.3f} <= {metric.threshold_down})")
            
            confidence_sum += metric.weight
        
        # Get predictive insights
        cpu_prediction, cpu_confidence = self.predictor.predict_load("cpu_utilization", 5)
        if cpu_prediction > 0.8 and cpu_confidence > 0.7:
            scale_up_signals += 0.5
            reasons.append(f"CPU predicted to reach {cpu_prediction:.2f} (confidence: {cpu_confidence:.2f})")
        
        # Check cooldown period
        time_since_last_scale = datetime.now() - self.last_scaling_action
        if time_since_last_scale.total_seconds() < self.scale_cooldown:
            return ScalingDecision(
                decision_id=decision_id,
                direction=ScalingDirection.STABLE,
                magnitude=0,
                confidence=1.0,
                reasoning="Cooldown period active",
                metrics_analyzed=[m.metric_name for m in self.scaling_metrics]
            )
        
        # Make decision
        if scale_up_signals > scale_down_signals and scale_up_signals >= 1.0:
            if self.current_instances < self.max_instances:
                magnitude = min(2, max(1, int(scale_up_signals)))
                return ScalingDecision(
                    decision_id=decision_id,
                    direction=ScalingDirection.UP,
                    magnitude=magnitude,
                    confidence=min(0.95, scale_up_signals / confidence_sum),
                    reasoning="; ".join(reasons),
                    metrics_analyzed=[m.metric_name for m in self.scaling_metrics]
                )
        
        elif scale_down_signals > scale_up_signals and scale_down_signals >= 1.0:
            if self.current_instances > self.min_instances:
                magnitude = min(1, max(1, int(scale_down_signals)))
                return ScalingDecision(
                    decision_id=decision_id,
                    direction=ScalingDirection.DOWN,
                    magnitude=magnitude,
                    confidence=min(0.95, scale_down_signals / confidence_sum),
                    reasoning="; ".join(reasons),
                    metrics_analyzed=[m.metric_name for m in self.scaling_metrics]
                )
        
        # Default to stable
        return ScalingDecision(
            decision_id=decision_id,
            direction=ScalingDirection.STABLE,
            magnitude=0,
            confidence=0.8,
            reasoning="No scaling needed" + (f"; {'; '.join(reasons[:2])}" if reasons else ""),
            metrics_analyzed=[m.metric_name for m in self.scaling_metrics]
        )
    
    def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute scaling decision"""
        
        if decision.direction == ScalingDirection.UP:
            self._scale_up(decision.magnitude)
        elif decision.direction == ScalingDirection.DOWN:
            self._scale_down(decision.magnitude)
        
        # Record decision
        decision.executed = True
        self.scaling_decisions.append(decision)
        self.predictor.scaling_history.append(decision)
        self.last_scaling_action = datetime.now()
        
        logger.info(f"Executed scaling decision: {decision.direction.value} by {decision.magnitude} "
                   f"(confidence: {decision.confidence:.2f})")
    
    def _scale_up(self, magnitude: int) -> None:
        """Scale up by adding instances"""
        
        new_instances = min(magnitude, self.max_instances - self.current_instances)
        
        for i in range(new_instances):
            instance_id = f"instance_{self.current_instances + i:03d}"
            instance = ResourceInstance(
                instance_id=instance_id,
                instance_type="defensive_worker",
                status="running",
                cpu_capacity=1.0,
                memory_capacity=1.0,
                current_load=0.0,
                created_at=datetime.now().isoformat()
            )
            
            self.load_balancer.register_instance(instance)
        
        self.current_instances += new_instances
        logger.info(f"Scaled up: added {new_instances} instances (total: {self.current_instances})")
    
    def _scale_down(self, magnitude: int) -> None:
        """Scale down by removing instances"""
        
        instances_to_remove = min(magnitude, self.current_instances - self.min_instances)
        
        # Find instances with lowest load to remove
        instances = list(self.load_balancer.instances.values())
        instances.sort(key=lambda x: x.current_load)
        
        for i in range(instances_to_remove):
            if i < len(instances):
                instance = instances[i]
                self.load_balancer.unregister_instance(instance.instance_id)
        
        self.current_instances -= instances_to_remove
        logger.info(f"Scaled down: removed {instances_to_remove} instances (total: {self.current_instances})")
    
    def _update_instance_loads(self) -> None:
        """Update instance load information"""
        
        # Simulate load updates (in real system, this would be actual load data)
        for instance_id, instance in self.load_balancer.instances.items():
            # Simulate load based on current metrics
            base_load = sum(m.current_value * m.weight for m in self.scaling_metrics[:2]) / 2
            variation = (hash(instance_id + str(int(time.time()))) % 20 - 10) / 100
            
            new_load = max(0.0, min(1.0, base_load + variation))
            self.load_balancer.update_instance_load(instance_id, new_load)
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Generate comprehensive scaling report"""
        
        # Recent decisions
        recent_decisions = self.scaling_decisions[-10:] if self.scaling_decisions else []
        
        # Predictive analysis
        pattern_analysis = self.predictor.analyze_scaling_patterns()
        
        # Instance status
        instance_summary = {
            "total": len(self.load_balancer.instances),
            "running": sum(1 for i in self.load_balancer.instances.values() if i.status == "running"),
            "average_load": statistics.mean([i.current_load for i in self.load_balancer.instances.values()]) if self.load_balancer.instances else 0,
            "total_capacity": {
                "cpu": sum(i.cpu_capacity for i in self.load_balancer.instances.values()),
                "memory": sum(i.memory_capacity for i in self.load_balancer.instances.values())
            }
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_configuration": {
                "min_instances": self.min_instances,
                "max_instances": self.max_instances,
                "current_instances": self.current_instances,
                "target_utilization": self.target_cpu_utilization
            },
            "current_metrics": {m.metric_name: m.current_value for m in self.scaling_metrics},
            "recent_decisions": [
                {
                    "direction": d.direction.value,
                    "magnitude": d.magnitude,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning[:100] + "..." if len(d.reasoning) > 100 else d.reasoning
                }
                for d in recent_decisions
            ],
            "instance_summary": instance_summary,
            "pattern_analysis": pattern_analysis,
            "recommendations": self._generate_scaling_recommendations()
        }
    
    def _generate_scaling_recommendations(self) -> List[str]:
        """Generate scaling optimization recommendations"""
        recommendations = []
        
        # Analyze recent performance
        if len(self.scaling_decisions) >= 5:
            recent = self.scaling_decisions[-5:]
            scale_ups = sum(1 for d in recent if d.direction == ScalingDirection.UP)
            scale_downs = sum(1 for d in recent if d.direction == ScalingDirection.DOWN)
            
            if scale_ups > scale_downs * 2:
                recommendations.append("Consider increasing minimum instance count due to frequent scale-ups")
            elif scale_downs > scale_ups * 2:
                recommendations.append("Consider decreasing maximum instance count due to frequent scale-downs")
        
        # Check metric thresholds
        for metric in self.scaling_metrics:
            if metric.current_value > metric.threshold_up * 0.9:
                recommendations.append(f"Consider lowering {metric.metric_name} scale-up threshold")
            elif metric.current_value < metric.threshold_down * 1.1:
                recommendations.append(f"Consider raising {metric.metric_name} scale-down threshold")
        
        return recommendations


if __name__ == "__main__":
    # Test intelligent auto-scaling
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    scaler = IntelligentAutoScaler(min_instances=2, max_instances=8)
    scaler.start_monitoring()
    
    print("Testing intelligent auto-scaling system...")
    
    # Simulate varying load conditions
    for minute in range(5):
        print(f"\nMinute {minute + 1}:")
        
        # Simulate high load
        high_load_metrics = {
            "cpu_utilization": 0.85,
            "memory_utilization": 0.75,
            "response_time": 4.5,
            "error_rate": 0.03
        }
        
        scaler.update_metrics(high_load_metrics)
        time.sleep(2)
        
        # Get current status
        report = scaler.get_scaling_report()
        print(f"  Instances: {report['instance_summary']['total']}")
        print(f"  Average Load: {report['instance_summary']['average_load']:.2f}")
        print(f"  Recent Decisions: {len(report['recent_decisions'])}")
        
        # Simulate load reduction
        normal_load_metrics = {
            "cpu_utilization": 0.45,
            "memory_utilization": 0.35,
            "response_time": 1.2,
            "error_rate": 0.005
        }
        
        scaler.update_metrics(normal_load_metrics)
        time.sleep(2)
    
    # Final report
    final_report = scaler.get_scaling_report()
    print(f"\nFinal Report:")
    print(f"  Total Instances: {final_report['instance_summary']['total']}")
    print(f"  Scaling Decisions: {len(final_report['recent_decisions'])}")
    print(f"  Recommendations: {len(final_report['recommendations'])}")
    
    scaler.stop_monitoring()
    print("\nIntelligent auto-scaling test completed ✅")