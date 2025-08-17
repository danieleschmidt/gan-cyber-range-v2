"""
Advanced auto-scaling system for dynamic cyber range resource management.

This module provides intelligent scaling capabilities that adapt to workload
patterns and optimize resource utilization for cost and performance.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"  # Horizontal scaling
    SCALE_IN = "scale_in"    # Horizontal scaling inward
    NO_CHANGE = "no_change"


class ResourceType(Enum):
    """Types of resources that can be scaled"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    CONTAINER = "container"
    VM = "vm"
    GPU = "gpu"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    cpu_utilization: float
    memory_utilization: float
    network_utilization: float
    request_rate: float
    response_time: float
    queue_depth: int
    error_rate: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingPolicy:
    """Configuration for scaling behavior"""
    name: str
    resource_type: ResourceType
    target_utilization: float = 70.0  # Target utilization percentage
    scale_up_threshold: float = 80.0   # Scale up when above this
    scale_down_threshold: float = 50.0 # Scale down when below this
    min_instances: int = 1
    max_instances: int = 100
    scale_up_cooldown: int = 300      # Cooldown in seconds
    scale_down_cooldown: int = 600    # Cooldown in seconds
    scale_up_increment: int = 2       # How many instances to add
    scale_down_increment: int = 1     # How many instances to remove
    prediction_window: int = 300      # Seconds to look ahead
    stability_window: int = 180       # Seconds to wait for stability
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True


@dataclass
class ScalingAction:
    """Represents a scaling action to be executed"""
    action_id: str
    resource_type: ResourceType
    direction: ScalingDirection
    current_count: int
    target_count: int
    reason: str
    confidence: float
    estimated_duration: int
    cost_impact: float
    timestamp: datetime = field(default_factory=datetime.now)
    executed: bool = False
    execution_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class PredictionModel:
    """Model for predicting future resource needs"""
    model_type: str
    accuracy: float
    last_trained: datetime
    training_data_points: int
    feature_importance: Dict[str, float]
    parameters: Dict[str, Any] = field(default_factory=dict)


class AutoScaler:
    """Intelligent auto-scaling system with predictive capabilities"""
    
    def __init__(
        self,
        metrics_collector=None,
        resource_manager=None,
        config_path: Optional[Path] = None
    ):
        self.metrics_collector = metrics_collector
        self.resource_manager = resource_manager
        self.config_path = config_path or Path("autoscaler_config.json")
        
        # Scaling state
        self.scaling_policies: Dict[str, ScalingPolicy] = {}
        self.scaling_history: deque = deque(maxlen=1000)
        self.metric_history: deque = deque(maxlen=10000)
        self.pending_actions: List[ScalingAction] = []
        self.last_scaling_time: Dict[str, datetime] = {}
        
        # Prediction models
        self.prediction_models: Dict[str, PredictionModel] = {}
        self.enable_prediction = True
        self.model_update_interval = 3600  # Update models every hour
        
        # Monitoring
        self.running = False
        self.monitoring_thread = None
        self.monitoring_interval = 30  # Check every 30 seconds
        
        # Event handlers
        self.scaling_handlers: List[Callable] = []
        self.alert_handlers: List[Callable] = []
        
        # Performance tracking
        self.scaling_effectiveness: Dict[str, List[float]] = {}
        self.cost_optimization: Dict[str, float] = {}
        
        # Load configuration
        self._load_configuration()
        
        logger.info("AutoScaler initialized")
    
    def start(self) -> None:
        """Start the auto-scaling monitoring"""
        
        if self.running:
            logger.warning("AutoScaler already running")
            return
        
        self.running = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("AutoScaler monitoring started")
    
    def stop(self) -> None:
        """Stop the auto-scaling monitoring"""
        
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)
        
        # Save configuration and history
        self._save_configuration()
        
        logger.info("AutoScaler stopped")
    
    def add_scaling_policy(self, policy: ScalingPolicy) -> None:
        """Add a scaling policy"""
        
        self.scaling_policies[policy.name] = policy
        logger.info(f"Added scaling policy: {policy.name}")
    
    def remove_scaling_policy(self, policy_name: str) -> None:
        """Remove a scaling policy"""
        
        if policy_name in self.scaling_policies:
            del self.scaling_policies[policy_name]
            logger.info(f"Removed scaling policy: {policy_name}")
    
    def update_scaling_policy(self, policy_name: str, **updates) -> None:
        """Update a scaling policy"""
        
        if policy_name in self.scaling_policies:
            policy = self.scaling_policies[policy_name]
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            
            logger.info(f"Updated scaling policy: {policy_name}")
    
    def get_current_metrics(self) -> ScalingMetrics:
        """Get current resource utilization metrics"""
        
        # Get metrics from collector if available
        if self.metrics_collector:
            current_metrics = self.metrics_collector.get_current_metrics()
            
            return ScalingMetrics(
                cpu_utilization=current_metrics.get('system_cpu_usage', 0.0),
                memory_utilization=current_metrics.get('system_memory_usage', 0.0),
                network_utilization=current_metrics.get('network_utilization', 0.0),
                request_rate=current_metrics.get('request_rate', 0.0),
                response_time=current_metrics.get('avg_response_time', 0.0),
                queue_depth=current_metrics.get('queue_depth', 0),
                error_rate=current_metrics.get('error_rate', 0.0),
                custom_metrics={
                    k: v for k, v in current_metrics.items()
                    if k.startswith('custom_')
                }
            )
        
        # Fallback to system metrics
        import psutil
        
        return ScalingMetrics(
            cpu_utilization=psutil.cpu_percent(interval=1),
            memory_utilization=psutil.virtual_memory().percent,
            network_utilization=0.0,  # Would need more complex calculation
            request_rate=0.0,
            response_time=0.0,
            queue_depth=0,
            error_rate=0.0
        )
    
    def predict_future_load(
        self,
        resource_type: ResourceType,
        prediction_horizon: int = 300
    ) -> Dict[str, float]:
        """Predict future resource load using ML models"""
        
        if not self.enable_prediction or len(self.metric_history) < 100:
            return {'predicted_utilization': 0.0, 'confidence': 0.0}
        
        # Get historical data
        historical_metrics = list(self.metric_history)[-100:]  # Last 100 data points
        
        # Extract feature for prediction
        if resource_type == ResourceType.CPU:
            utilization_values = [m.cpu_utilization for m in historical_metrics]
        elif resource_type == ResourceType.MEMORY:
            utilization_values = [m.memory_utilization for m in historical_metrics]
        else:
            utilization_values = [50.0] * len(historical_metrics)  # Default
        
        # Simple time series prediction (moving average with trend)
        if len(utilization_values) >= 10:
            recent_avg = np.mean(utilization_values[-10:])
            older_avg = np.mean(utilization_values[-20:-10]) if len(utilization_values) >= 20 else recent_avg
            trend = recent_avg - older_avg
            
            # Predict future utilization
            predicted_utilization = recent_avg + (trend * prediction_horizon / 300)
            predicted_utilization = max(0, min(100, predicted_utilization))
            
            # Calculate confidence based on trend stability
            recent_std = np.std(utilization_values[-10:])
            confidence = max(0, min(1, 1 - (recent_std / 50)))  # Lower std = higher confidence
            
            return {
                'predicted_utilization': predicted_utilization,
                'confidence': confidence,
                'trend': trend,
                'current_avg': recent_avg
            }
        
        return {'predicted_utilization': 0.0, 'confidence': 0.0}
    
    def make_scaling_decision(
        self,
        policy: ScalingPolicy,
        current_metrics: ScalingMetrics
    ) -> Optional[ScalingAction]:
        """Make intelligent scaling decision based on policy and metrics"""
        
        # Check if policy is enabled
        if not policy.enabled:
            return None
        
        # Check cooldown period
        last_scaling = self.last_scaling_time.get(policy.name)
        if last_scaling:
            time_since_last = (datetime.now() - last_scaling).total_seconds()
            if time_since_last < policy.scale_up_cooldown:
                return None
        
        # Get current resource count
        current_count = self._get_current_resource_count(policy.resource_type)
        
        # Determine current utilization based on resource type
        current_utilization = self._get_utilization_for_resource(
            current_metrics, policy.resource_type
        )
        
        # Get prediction for future load
        prediction = self.predict_future_load(policy.resource_type, policy.prediction_window)
        predicted_utilization = prediction.get('predicted_utilization', current_utilization)
        prediction_confidence = prediction.get('confidence', 0.5)
        
        # Determine scaling direction and magnitude
        scaling_direction = ScalingDirection.NO_CHANGE
        target_count = current_count
        reason = "No scaling needed"
        confidence = 0.5
        
        # Check for scale-up conditions
        if (current_utilization > policy.scale_up_threshold or 
            (prediction_confidence > 0.7 and predicted_utilization > policy.scale_up_threshold)):
            
            if current_count < policy.max_instances:
                scaling_direction = ScalingDirection.SCALE_OUT
                target_count = min(
                    policy.max_instances,
                    current_count + policy.scale_up_increment
                )
                reason = f"High utilization: {current_utilization:.1f}% (threshold: {policy.scale_up_threshold}%)"
                if predicted_utilization > current_utilization:
                    reason += f", predicted: {predicted_utilization:.1f}%"
                confidence = 0.8 if prediction_confidence > 0.7 else 0.6
        
        # Check for scale-down conditions
        elif (current_utilization < policy.scale_down_threshold and
              predicted_utilization < policy.scale_down_threshold and
              current_count > policy.min_instances):
            
            # More conservative scaling down
            if self._check_scale_down_safety(policy, current_metrics):
                scaling_direction = ScalingDirection.SCALE_IN
                target_count = max(
                    policy.min_instances,
                    current_count - policy.scale_down_increment
                )
                reason = f"Low utilization: {current_utilization:.1f}% (threshold: {policy.scale_down_threshold}%)"
                confidence = 0.7
        
        # Check custom rules
        custom_action = self._evaluate_custom_rules(policy, current_metrics)
        if custom_action:
            scaling_direction = custom_action['direction']
            target_count = custom_action['target_count']
            reason = custom_action['reason']
            confidence = custom_action['confidence']
        
        # Create scaling action if needed
        if scaling_direction != ScalingDirection.NO_CHANGE:
            action = ScalingAction(
                action_id=self._generate_action_id(),
                resource_type=policy.resource_type,
                direction=scaling_direction,
                current_count=current_count,
                target_count=target_count,
                reason=reason,
                confidence=confidence,
                estimated_duration=self._estimate_scaling_duration(
                    policy.resource_type, abs(target_count - current_count)
                ),
                cost_impact=self._estimate_cost_impact(
                    policy.resource_type, target_count - current_count
                )
            )
            
            return action
        
        return None
    
    def execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action"""
        
        logger.info(f"Executing scaling action: {action.action_id}")
        logger.info(f"  Resource: {action.resource_type.value}")
        logger.info(f"  Direction: {action.direction.value}")
        logger.info(f"  Current: {action.current_count} -> Target: {action.target_count}")
        logger.info(f"  Reason: {action.reason}")
        
        try:
            # Execute scaling through resource manager
            if self.resource_manager:
                success = self._execute_through_resource_manager(action)
            else:
                # Mock execution for demonstration
                success = self._mock_scaling_execution(action)
            
            # Update action status
            action.executed = True
            action.execution_time = datetime.now()
            action.success = success
            
            if success:
                # Record successful scaling
                self.scaling_history.append(action)
                self._update_scaling_effectiveness(action)
                
                # Trigger scaling handlers
                for handler in self.scaling_handlers:
                    try:
                        handler(action)
                    except Exception as e:
                        logger.error(f"Error in scaling handler: {e}")
                
                logger.info(f"Scaling action {action.action_id} completed successfully")
                return True
            else:
                logger.error(f"Scaling action {action.action_id} failed")
                return False
                
        except Exception as e:
            action.executed = True
            action.execution_time = datetime.now()
            action.success = False
            action.error_message = str(e)
            
            logger.error(f"Error executing scaling action {action.action_id}: {e}")
            return False
    
    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Get scaling recommendations based on current state"""
        
        recommendations = []
        current_metrics = self.get_current_metrics()
        
        for policy_name, policy in self.scaling_policies.items():
            action = self.make_scaling_decision(policy, current_metrics)
            
            if action:
                recommendation = {
                    'policy': policy_name,
                    'resource_type': action.resource_type.value,
                    'current_count': action.current_count,
                    'recommended_count': action.target_count,
                    'direction': action.direction.value,
                    'reason': action.reason,
                    'confidence': action.confidence,
                    'estimated_cost_impact': action.cost_impact,
                    'estimated_duration': action.estimated_duration
                }
                recommendations.append(recommendation)
        
        return recommendations
    
    def get_scaling_history(self, hours: int = 24) -> List[ScalingAction]:
        """Get scaling history for the specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            action for action in self.scaling_history
            if action.timestamp >= cutoff_time
        ]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for scaling decisions"""
        
        recent_actions = self.get_scaling_history(24)
        
        # Calculate success rate
        total_actions = len(recent_actions)
        successful_actions = len([a for a in recent_actions if a.success])
        success_rate = (successful_actions / total_actions * 100) if total_actions > 0 else 0
        
        # Calculate cost savings
        total_cost_impact = sum(a.cost_impact for a in recent_actions)
        
        # Performance improvements
        effectiveness_scores = []
        for resource_type, scores in self.scaling_effectiveness.items():
            if scores:
                effectiveness_scores.extend(scores)
        
        avg_effectiveness = np.mean(effectiveness_scores) if effectiveness_scores else 0
        
        return {
            'period_hours': 24,
            'total_scaling_actions': total_actions,
            'successful_actions': successful_actions,
            'success_rate': success_rate,
            'total_cost_impact': total_cost_impact,
            'average_effectiveness': avg_effectiveness,
            'active_policies': len([p for p in self.scaling_policies.values() if p.enabled]),
            'prediction_accuracy': self._calculate_prediction_accuracy(),
            'resource_utilization': self._calculate_average_utilization(),
            'recommendations': len(self.get_scaling_recommendations())
        }
    
    def optimize_policies(self) -> Dict[str, Any]:
        """Optimize scaling policies based on historical performance"""
        
        optimizations = []
        
        for policy_name, policy in self.scaling_policies.items():
            # Analyze historical effectiveness for this policy
            policy_actions = [
                a for a in self.scaling_history
                if a.resource_type == policy.resource_type
            ]
            
            if len(policy_actions) < 5:  # Need sufficient data
                continue
            
            # Calculate average utilization after scaling
            post_scaling_metrics = self._get_post_scaling_metrics(policy_actions)
            
            suggestions = []
            
            # Check if thresholds are too conservative or aggressive
            if post_scaling_metrics.get('avg_utilization_after_scale_up', 0) < 60:
                suggestions.append({
                    'parameter': 'scale_up_threshold',
                    'current': policy.scale_up_threshold,
                    'suggested': policy.scale_up_threshold + 5,
                    'reason': 'Scale-up threshold too low, causing over-provisioning'
                })
            
            if post_scaling_metrics.get('avg_utilization_after_scale_down', 0) > 80:
                suggestions.append({
                    'parameter': 'scale_down_threshold',
                    'current': policy.scale_down_threshold,
                    'suggested': policy.scale_down_threshold + 10,
                    'reason': 'Scale-down threshold too aggressive, causing resource stress'
                })
            
            # Check cooldown periods
            failed_actions = [a for a in policy_actions if not a.success]
            if len(failed_actions) > len(policy_actions) * 0.2:  # More than 20% failures
                suggestions.append({
                    'parameter': 'scale_up_cooldown',
                    'current': policy.scale_up_cooldown,
                    'suggested': policy.scale_up_cooldown + 60,
                    'reason': 'High failure rate suggests cooldown too short'
                })
            
            if suggestions:
                optimizations.append({
                    'policy': policy_name,
                    'suggestions': suggestions,
                    'effectiveness_score': self.scaling_effectiveness.get(
                        policy.resource_type.value, [0.5]
                    )[-1] if self.scaling_effectiveness.get(policy.resource_type.value) else 0.5
                })
        
        return {
            'optimizations': optimizations,
            'auto_apply': False,  # Require manual approval
            'generated_at': datetime.now().isoformat()
        }
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for auto-scaling"""
        
        logger.info("Started auto-scaling monitoring loop")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Get current metrics
                current_metrics = self.get_current_metrics()
                
                # Store metrics for history
                self.metric_history.append(current_metrics)
                
                # Check each scaling policy
                for policy_name, policy in self.scaling_policies.items():
                    if not policy.enabled:
                        continue
                    
                    # Make scaling decision
                    action = self.make_scaling_decision(policy, current_metrics)
                    
                    if action:
                        # Add to pending actions for review or immediate execution
                        if action.confidence > 0.8:
                            # High confidence - execute immediately
                            success = self.execute_scaling_action(action)
                            if success:
                                self.last_scaling_time[policy_name] = datetime.now()
                        else:
                            # Lower confidence - add to pending for review
                            self.pending_actions.append(action)
                
                # Update prediction models periodically
                if len(self.metric_history) % 120 == 0:  # Every hour (120 * 30 seconds)
                    self._update_prediction_models()
                
                # Wait for next monitoring cycle
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _get_current_resource_count(self, resource_type: ResourceType) -> int:
        """Get current count of resources"""
        
        if self.resource_manager:
            return self.resource_manager.get_resource_count(resource_type)
        
        # Mock implementation
        return {
            ResourceType.CONTAINER: 3,
            ResourceType.VM: 2,
            ResourceType.CPU: 4,
            ResourceType.MEMORY: 8,
            ResourceType.GPU: 1
        }.get(resource_type, 1)
    
    def _get_utilization_for_resource(
        self,
        metrics: ScalingMetrics,
        resource_type: ResourceType
    ) -> float:
        """Get utilization percentage for specific resource type"""
        
        utilization_map = {
            ResourceType.CPU: metrics.cpu_utilization,
            ResourceType.MEMORY: metrics.memory_utilization,
            ResourceType.NETWORK: metrics.network_utilization,
            ResourceType.CONTAINER: max(metrics.cpu_utilization, metrics.memory_utilization),
            ResourceType.VM: max(metrics.cpu_utilization, metrics.memory_utilization),
            ResourceType.GPU: metrics.custom_metrics.get('gpu_utilization', 0.0)
        }
        
        return utilization_map.get(resource_type, 0.0)
    
    def _check_scale_down_safety(
        self,
        policy: ScalingPolicy,
        current_metrics: ScalingMetrics
    ) -> bool:
        """Check if it's safe to scale down"""
        
        # Don't scale down if error rate is high
        if current_metrics.error_rate > 5.0:  # 5% error rate
            return False
        
        # Don't scale down if response time is too high
        if current_metrics.response_time > 1000:  # 1 second
            return False
        
        # Don't scale down if queue is building up
        if current_metrics.queue_depth > 10:
            return False
        
        # Check stability window
        if len(self.metric_history) >= policy.stability_window // self.monitoring_interval:
            recent_metrics = list(self.metric_history)[-(policy.stability_window // self.monitoring_interval):]
            recent_utilizations = [
                self._get_utilization_for_resource(m, policy.resource_type)
                for m in recent_metrics
            ]
            
            # Check if utilization has been consistently low
            if all(u < policy.scale_down_threshold for u in recent_utilizations):
                return True
        
        return False
    
    def _evaluate_custom_rules(
        self,
        policy: ScalingPolicy,
        current_metrics: ScalingMetrics
    ) -> Optional[Dict[str, Any]]:
        """Evaluate custom scaling rules"""
        
        for rule in policy.custom_rules:
            try:
                condition = rule.get('condition', '')
                
                # Simple rule evaluation (could be enhanced with a proper rule engine)
                if 'error_rate' in condition:
                    threshold = rule.get('error_rate_threshold', 10.0)
                    if current_metrics.error_rate > threshold:
                        return {
                            'direction': ScalingDirection.SCALE_OUT,
                            'target_count': self._get_current_resource_count(policy.resource_type) + 2,
                            'reason': f"High error rate: {current_metrics.error_rate:.1f}%",
                            'confidence': 0.9
                        }
                
                if 'response_time' in condition:
                    threshold = rule.get('response_time_threshold', 2000)
                    if current_metrics.response_time > threshold:
                        return {
                            'direction': ScalingDirection.SCALE_OUT,
                            'target_count': self._get_current_resource_count(policy.resource_type) + 1,
                            'reason': f"High response time: {current_metrics.response_time:.0f}ms",
                            'confidence': 0.8
                        }
                        
            except Exception as e:
                logger.error(f"Error evaluating custom rule: {e}")
        
        return None
    
    def _estimate_scaling_duration(self, resource_type: ResourceType, count_change: int) -> int:
        """Estimate how long scaling will take"""
        
        base_duration = {
            ResourceType.CONTAINER: 30,  # 30 seconds per container
            ResourceType.VM: 120,        # 2 minutes per VM
            ResourceType.CPU: 10,        # 10 seconds for CPU scaling
            ResourceType.MEMORY: 10,     # 10 seconds for memory scaling
            ResourceType.GPU: 60         # 1 minute for GPU scaling
        }
        
        return base_duration.get(resource_type, 60) * count_change
    
    def _estimate_cost_impact(self, resource_type: ResourceType, count_change: int) -> float:
        """Estimate cost impact of scaling action"""
        
        hourly_cost = {
            ResourceType.CONTAINER: 0.10,  # $0.10 per hour per container
            ResourceType.VM: 0.50,         # $0.50 per hour per VM
            ResourceType.CPU: 0.05,        # $0.05 per hour per CPU core
            ResourceType.MEMORY: 0.02,     # $0.02 per hour per GB
            ResourceType.GPU: 2.50         # $2.50 per hour per GPU
        }
        
        return hourly_cost.get(resource_type, 0.25) * count_change
    
    def _execute_through_resource_manager(self, action: ScalingAction) -> bool:
        """Execute scaling action through resource manager"""
        
        try:
            if action.direction in [ScalingDirection.SCALE_OUT, ScalingDirection.SCALE_UP]:
                return self.resource_manager.scale_out(
                    action.resource_type,
                    action.target_count - action.current_count
                )
            else:
                return self.resource_manager.scale_in(
                    action.resource_type,
                    action.current_count - action.target_count
                )
        except Exception as e:
            logger.error(f"Resource manager execution failed: {e}")
            return False
    
    def _mock_scaling_execution(self, action: ScalingAction) -> bool:
        """Mock scaling execution for demonstration"""
        
        # Simulate execution time
        time.sleep(1)
        
        # Simulate 95% success rate
        import random
        return random.random() < 0.95
    
    def _update_scaling_effectiveness(self, action: ScalingAction) -> None:
        """Update effectiveness tracking for scaling actions"""
        
        resource_key = action.resource_type.value
        
        # Calculate effectiveness score based on confidence and success
        effectiveness = action.confidence if action.success else 0.0
        
        if resource_key not in self.scaling_effectiveness:
            self.scaling_effectiveness[resource_key] = deque(maxlen=100)
        
        self.scaling_effectiveness[resource_key].append(effectiveness)
    
    def _update_prediction_models(self) -> None:
        """Update prediction models based on recent data"""
        
        if len(self.metric_history) < 100:
            return
        
        try:
            # Simple model update (in practice, would use more sophisticated ML)
            recent_metrics = list(self.metric_history)[-100:]
            
            for resource_type in ResourceType:
                utilizations = [
                    self._get_utilization_for_resource(m, resource_type)
                    for m in recent_metrics
                ]
                
                # Calculate simple statistics for the model
                accuracy = 1.0 - (np.std(utilizations) / 100.0)  # Lower variance = higher accuracy
                
                self.prediction_models[resource_type.value] = PredictionModel(
                    model_type="simple_trend",
                    accuracy=max(0.1, min(0.9, accuracy)),
                    last_trained=datetime.now(),
                    training_data_points=len(utilizations),
                    feature_importance={'trend': 0.6, 'average': 0.4}
                )
            
            logger.info("Updated prediction models")
            
        except Exception as e:
            logger.error(f"Error updating prediction models: {e}")
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate overall prediction accuracy"""
        
        if not self.prediction_models:
            return 0.0
        
        accuracies = [model.accuracy for model in self.prediction_models.values()]
        return np.mean(accuracies)
    
    def _calculate_average_utilization(self) -> Dict[str, float]:
        """Calculate average utilization across resource types"""
        
        if not self.metric_history:
            return {}
        
        recent_metrics = list(self.metric_history)[-10:]  # Last 10 measurements
        
        return {
            'cpu': np.mean([m.cpu_utilization for m in recent_metrics]),
            'memory': np.mean([m.memory_utilization for m in recent_metrics]),
            'network': np.mean([m.network_utilization for m in recent_metrics])
        }
    
    def _get_post_scaling_metrics(self, actions: List[ScalingAction]) -> Dict[str, float]:
        """Analyze metrics after scaling actions"""
        
        # This would require correlation with historical metrics
        # For now, return mock analysis
        return {
            'avg_utilization_after_scale_up': 65.0,
            'avg_utilization_after_scale_down': 75.0,
            'response_time_improvement': 0.2,
            'cost_efficiency': 0.8
        }
    
    def _generate_action_id(self) -> str:
        """Generate unique action ID"""
        
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _load_configuration(self) -> None:
        """Load auto-scaler configuration"""
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load scaling policies
                for policy_config in config.get('policies', []):
                    policy = ScalingPolicy(**policy_config)
                    self.scaling_policies[policy.name] = policy
                
                # Load other configuration
                self.enable_prediction = config.get('enable_prediction', True)
                self.monitoring_interval = config.get('monitoring_interval', 30)
                
                logger.info(f"Loaded auto-scaler configuration from {self.config_path}")
                
            except Exception as e:
                logger.error(f"Failed to load auto-scaler config: {e}")
                self._create_default_policies()
        else:
            self._create_default_policies()
    
    def _save_configuration(self) -> None:
        """Save auto-scaler configuration"""
        
        try:
            config = {
                'policies': [
                    {
                        'name': policy.name,
                        'resource_type': policy.resource_type.value,
                        'target_utilization': policy.target_utilization,
                        'scale_up_threshold': policy.scale_up_threshold,
                        'scale_down_threshold': policy.scale_down_threshold,
                        'min_instances': policy.min_instances,
                        'max_instances': policy.max_instances,
                        'scale_up_cooldown': policy.scale_up_cooldown,
                        'scale_down_cooldown': policy.scale_down_cooldown,
                        'enabled': policy.enabled
                    }
                    for policy in self.scaling_policies.values()
                ],
                'enable_prediction': self.enable_prediction,
                'monitoring_interval': self.monitoring_interval
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Saved auto-scaler configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save auto-scaler config: {e}")
    
    def _create_default_policies(self) -> None:
        """Create default scaling policies"""
        
        default_policies = [
            ScalingPolicy(
                name="container_cpu_scaling",
                resource_type=ResourceType.CONTAINER,
                target_utilization=70.0,
                scale_up_threshold=80.0,
                scale_down_threshold=50.0,
                min_instances=2,
                max_instances=20
            ),
            ScalingPolicy(
                name="vm_memory_scaling",
                resource_type=ResourceType.VM,
                target_utilization=75.0,
                scale_up_threshold=85.0,
                scale_down_threshold=40.0,
                min_instances=1,
                max_instances=10
            )
        ]
        
        for policy in default_policies:
            self.scaling_policies[policy.name] = policy
        
        logger.info("Created default scaling policies")