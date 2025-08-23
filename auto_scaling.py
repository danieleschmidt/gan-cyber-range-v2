#!/usr/bin/env python3
"""
Auto-scaling framework for defensive cybersecurity operations

This module provides intelligent auto-scaling capabilities for defensive systems
including load-based scaling, predictive scaling, and resource optimization.
"""

import time
import threading
import json
import logging
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import queue
import statistics

# Setup logging
logger = logging.getLogger(__name__)

class ScalingTrigger(Enum):
    """Types of scaling triggers"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    QUEUE_DEPTH = "queue_depth"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"

class ScalingDirection(Enum):
    """Scaling direction"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

@dataclass
class ScalingMetric:
    """Metric used for scaling decisions"""
    name: str
    current_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    
    def needs_scale_up(self) -> bool:
        return self.current_value > self.threshold_up
    
    def needs_scale_down(self) -> bool:
        return self.current_value < self.threshold_down
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ScalingEvent:
    """Record of a scaling event"""
    timestamp: datetime
    trigger: ScalingTrigger
    direction: ScalingDirection
    previous_capacity: int
    new_capacity: int
    metrics: Dict[str, float]
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'trigger': self.trigger.value,
            'direction': self.direction.value
        }

class PredictiveModel:
    """Simple predictive model for load forecasting"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.historical_data = []
        
    def add_data_point(self, timestamp: datetime, value: float):
        """Add a data point to the historical data"""
        
        self.historical_data.append((timestamp, value))
        
        # Keep only recent data
        if len(self.historical_data) > self.window_size:
            self.historical_data.pop(0)
    
    def predict_next_value(self) -> Optional[float]:
        """Predict the next value based on historical data"""
        
        if len(self.historical_data) < 3:
            return None
        
        # Simple trend-based prediction
        values = [point[1] for point in self.historical_data]
        
        # Calculate trend
        recent_values = values[-5:] if len(values) >= 5 else values
        if len(recent_values) >= 2:
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
            prediction = values[-1] + trend
            return max(0, prediction)  # Ensure non-negative
        
        return values[-1]  # Return last value if no trend can be calculated
    
    def get_trend(self) -> str:
        """Get the trend direction"""
        
        if len(self.historical_data) < 2:
            return "stable"
        
        recent_avg = statistics.mean([p[1] for p in self.historical_data[-3:]])
        older_avg = statistics.mean([p[1] for p in self.historical_data[:-3]]) if len(self.historical_data) > 3 else recent_avg
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

class DefensiveAutoScaler:
    """Auto-scaling system for defensive cybersecurity operations"""
    
    def __init__(self, min_capacity: int = 1, max_capacity: int = 20):
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.current_capacity = min_capacity
        
        # Scaling configuration
        self.scaling_config = {
            'scale_up_cooldown_seconds': 300,      # 5 minutes
            'scale_down_cooldown_seconds': 600,    # 10 minutes
            'evaluation_interval_seconds': 30,     # 30 seconds
            'prediction_enabled': True,
            'conservative_scaling': True
        }
        
        # Metrics and models
        self.metrics = {}
        self.predictive_models = {}
        self.scaling_events = []
        self.last_scale_time = None
        self.last_scale_direction = ScalingDirection.STABLE
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.scaling_callbacks = []
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        logger.info(f"Auto-scaler initialized - Min: {min_capacity}, Max: {max_capacity}")
    
    def _initialize_default_metrics(self):
        """Initialize default scaling metrics"""
        
        # CPU usage metric
        self.add_metric(ScalingMetric(
            name="cpu_usage",
            current_value=0.0,
            threshold_up=70.0,
            threshold_down=30.0,
            weight=1.0
        ))
        
        # Memory usage metric
        self.add_metric(ScalingMetric(
            name="memory_usage",
            current_value=0.0,
            threshold_up=80.0,
            threshold_down=40.0,
            weight=0.8
        ))
        
        # Request queue depth
        self.add_metric(ScalingMetric(
            name="queue_depth",
            current_value=0.0,
            threshold_up=10.0,
            threshold_down=2.0,
            weight=1.2
        ))
        
        # Response time
        self.add_metric(ScalingMetric(
            name="response_time",
            current_value=0.0,
            threshold_up=2000.0,  # 2 seconds in ms
            threshold_down=500.0,  # 500ms
            weight=1.0
        ))
    
    def add_metric(self, metric: ScalingMetric):
        """Add or update a scaling metric"""
        
        self.metrics[metric.name] = metric
        
        # Initialize predictive model for this metric
        if metric.name not in self.predictive_models:
            self.predictive_models[metric.name] = PredictiveModel()
        
        logger.info(f"Added scaling metric: {metric.name}")
    
    def update_metric(self, metric_name: str, value: float):
        """Update a metric value"""
        
        if metric_name in self.metrics:
            self.metrics[metric_name].current_value = value
            
            # Update predictive model
            if metric_name in self.predictive_models:
                self.predictive_models[metric_name].add_data_point(datetime.now(), value)
            
            logger.debug(f"Updated metric {metric_name}: {value}")
        else:
            logger.warning(f"Unknown metric: {metric_name}")
    
    def add_scaling_callback(self, callback: Callable[[int, int], None]):
        """Add callback for scaling events"""
        
        self.scaling_callbacks.append(callback)
        logger.info("Added scaling callback")
    
    def evaluate_scaling_decision(self) -> ScalingDirection:
        """Evaluate whether scaling is needed"""
        
        # Check cooldown periods
        if self.last_scale_time:
            time_since_scale = (datetime.now() - self.last_scale_time).total_seconds()
            
            if self.last_scale_direction == ScalingDirection.UP:
                if time_since_scale < self.scaling_config['scale_up_cooldown_seconds']:
                    return ScalingDirection.STABLE
            elif self.last_scale_direction == ScalingDirection.DOWN:
                if time_since_scale < self.scaling_config['scale_down_cooldown_seconds']:
                    return ScalingDirection.STABLE
        
        # Calculate weighted scaling signals
        scale_up_signals = []
        scale_down_signals = []
        
        for metric_name, metric in self.metrics.items():
            if metric.needs_scale_up():
                scale_up_signals.append(metric.weight)
            elif metric.needs_scale_down():
                scale_down_signals.append(metric.weight)
        
        # Include predictive signals if enabled
        if self.scaling_config['prediction_enabled']:
            predictive_signals = self._get_predictive_signals()
            scale_up_signals.extend(predictive_signals['scale_up'])
            scale_down_signals.extend(predictive_signals['scale_down'])
        
        # Calculate total signals
        total_scale_up = sum(scale_up_signals)
        total_scale_down = sum(scale_down_signals)
        
        # Conservative scaling logic
        if self.scaling_config['conservative_scaling']:
            scale_up_threshold = 1.5  # Require stronger signal
            scale_down_threshold = 1.0
        else:
            scale_up_threshold = 1.0
            scale_down_threshold = 1.0
        
        # Make scaling decision
        if total_scale_up >= scale_up_threshold and total_scale_up > total_scale_down:
            if self.current_capacity < self.max_capacity:
                return ScalingDirection.UP
        elif total_scale_down >= scale_down_threshold and total_scale_down > total_scale_up:
            if self.current_capacity > self.min_capacity:
                return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    def _get_predictive_signals(self) -> Dict[str, List[float]]:
        """Get scaling signals from predictive models"""
        
        signals = {'scale_up': [], 'scale_down': []}
        
        for metric_name, model in self.predictive_models.items():
            if metric_name not in self.metrics:
                continue
            
            predicted_value = model.predict_next_value()
            if predicted_value is None:
                continue
            
            metric = self.metrics[metric_name]
            trend = model.get_trend()
            
            # Generate predictive scaling signals
            if predicted_value > metric.threshold_up and trend == "increasing":
                signals['scale_up'].append(metric.weight * 0.5)  # Predictive signals are weighted lower
            elif predicted_value < metric.threshold_down and trend == "decreasing":
                signals['scale_down'].append(metric.weight * 0.5)
        
        return signals
    
    def scale(self, direction: ScalingDirection, trigger: ScalingTrigger, reason: str):
        """Execute scaling action"""
        
        if direction == ScalingDirection.STABLE:
            return
        
        previous_capacity = self.current_capacity
        
        if direction == ScalingDirection.UP:
            new_capacity = min(self.current_capacity + 1, self.max_capacity)
        else:  # ScalingDirection.DOWN
            new_capacity = max(self.current_capacity - 1, self.min_capacity)
        
        if new_capacity == previous_capacity:
            logger.info(f"Scaling limit reached: {direction.value}")
            return
        
        # Update capacity
        self.current_capacity = new_capacity
        self.last_scale_time = datetime.now()
        self.last_scale_direction = direction
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=datetime.now(),
            trigger=trigger,
            direction=direction,
            previous_capacity=previous_capacity,
            new_capacity=new_capacity,
            metrics={name: metric.current_value for name, metric in self.metrics.items()},
            reason=reason
        )
        
        self.scaling_events.append(event)
        
        # Execute scaling callbacks
        for callback in self.scaling_callbacks:
            try:
                callback(previous_capacity, new_capacity)
            except Exception as e:
                logger.error(f"Scaling callback failed: {e}")
        
        logger.info(f"Scaled {direction.value}: {previous_capacity} -> {new_capacity} ({reason})")
    
    def _monitoring_loop(self):
        """Main monitoring and scaling loop"""
        
        logger.info("Auto-scaling monitoring started")
        
        while self.is_monitoring:
            try:
                # Evaluate scaling decision
                scaling_decision = self.evaluate_scaling_decision()
                
                # Execute scaling if needed
                if scaling_decision != ScalingDirection.STABLE:
                    # Determine primary trigger
                    primary_trigger = self._identify_primary_trigger()
                    reason = self._generate_scaling_reason(scaling_decision)
                    
                    self.scale(scaling_decision, primary_trigger, reason)
                
                # Sleep until next evaluation
                time.sleep(self.scaling_config['evaluation_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Auto-scaling monitoring error: {e}")
                time.sleep(5)  # Short sleep on error
    
    def _identify_primary_trigger(self) -> ScalingTrigger:
        """Identify the primary trigger for scaling"""
        
        # Find metric with highest weight that's triggering scaling
        triggering_metrics = []
        
        for metric_name, metric in self.metrics.items():
            if metric.needs_scale_up() or metric.needs_scale_down():
                triggering_metrics.append((metric_name, metric.weight))
        
        if triggering_metrics:
            primary_metric = max(triggering_metrics, key=lambda x: x[1])[0]
            
            # Map metric names to triggers
            trigger_mapping = {
                'cpu_usage': ScalingTrigger.CPU_USAGE,
                'memory_usage': ScalingTrigger.MEMORY_USAGE,
                'queue_depth': ScalingTrigger.QUEUE_DEPTH,
                'response_time': ScalingTrigger.RESPONSE_TIME
            }
            
            return trigger_mapping.get(primary_metric, ScalingTrigger.CUSTOM_METRIC)
        
        return ScalingTrigger.CUSTOM_METRIC
    
    def _generate_scaling_reason(self, direction: ScalingDirection) -> str:
        """Generate human-readable reason for scaling"""
        
        triggering_conditions = []
        
        for metric_name, metric in self.metrics.items():
            if direction == ScalingDirection.UP and metric.needs_scale_up():
                triggering_conditions.append(f"{metric_name}={metric.current_value:.1f} > {metric.threshold_up}")
            elif direction == ScalingDirection.DOWN and metric.needs_scale_down():
                triggering_conditions.append(f"{metric_name}={metric.current_value:.1f} < {metric.threshold_down}")
        
        if triggering_conditions:
            return f"Triggered by: {', '.join(triggering_conditions)}"
        else:
            return f"Predictive scaling based on trend analysis"
    
    def start_monitoring(self):
        """Start auto-scaling monitoring"""
        
        if self.is_monitoring:
            logger.warning("Auto-scaling monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Auto-scaling monitoring stopped")
    
    def get_scaling_report(self) -> Dict:
        """Generate comprehensive scaling report"""
        
        # Calculate scaling statistics
        scale_up_events = [e for e in self.scaling_events if e.direction == ScalingDirection.UP]
        scale_down_events = [e for e in self.scaling_events if e.direction == ScalingDirection.DOWN]
        
        # Get recent events (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_events = [e for e in self.scaling_events if e.timestamp > recent_cutoff]
        
        # Current metric states
        current_metrics = {}
        for name, metric in self.metrics.items():
            current_metrics[name] = {
                'current_value': metric.current_value,
                'threshold_up': metric.threshold_up,
                'threshold_down': metric.threshold_down,
                'needs_scale_up': metric.needs_scale_up(),
                'needs_scale_down': metric.needs_scale_down()
            }
        
        # Predictive insights
        predictive_insights = {}
        for name, model in self.predictive_models.items():
            predicted_value = model.predict_next_value()
            trend = model.get_trend()
            
            predictive_insights[name] = {
                'predicted_next_value': predicted_value,
                'trend': trend,
                'confidence': 'medium'  # Would be more sophisticated in real implementation
            }
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'current_capacity': self.current_capacity,
            'capacity_range': {
                'min': self.min_capacity,
                'max': self.max_capacity
            },
            'utilization_percent': round((self.current_capacity / self.max_capacity) * 100, 2),
            'scaling_statistics': {
                'total_events': len(self.scaling_events),
                'scale_up_events': len(scale_up_events),
                'scale_down_events': len(scale_down_events),
                'recent_events': len(recent_events)
            },
            'current_metrics': current_metrics,
            'predictive_insights': predictive_insights,
            'last_scaling_event': self.scaling_events[-1].to_dict() if self.scaling_events else None,
            'recommendations': self._generate_scaling_recommendations()
        }
    
    def _generate_scaling_recommendations(self) -> List[str]:
        """Generate scaling recommendations"""
        
        recommendations = []
        
        # Check for frequent scaling
        if len(self.scaling_events) > 0:
            recent_events = [e for e in self.scaling_events 
                           if (datetime.now() - e.timestamp).total_seconds() < 3600]
            
            if len(recent_events) > 5:
                recommendations.append(
                    "Frequent scaling detected. Consider adjusting thresholds or cooldown periods."
                )
        
        # Check capacity utilization
        utilization = (self.current_capacity / self.max_capacity) * 100
        if utilization > 80:
            recommendations.append("High capacity utilization. Consider increasing max_capacity.")
        elif utilization < 20:
            recommendations.append("Low capacity utilization. Consider decreasing min_capacity.")
        
        # Check metric thresholds
        for name, metric in self.metrics.items():
            if metric.current_value > metric.threshold_up * 1.2:
                recommendations.append(
                    f"Metric '{name}' consistently exceeds threshold. "
                    f"Consider lowering threshold or improving performance."
                )
        
        return recommendations if recommendations else ["Scaling configuration appears optimal"]

def simulate_defensive_workload():
    """Simulate varying workload for demonstration"""
    
    # Simulate realistic workload patterns
    time_factor = time.time() % 3600  # Hour cycle
    
    # Base load with periodic spikes
    base_load = 30 + 20 * math.sin(time_factor / 600)  # Sine wave over 10 minutes
    
    # Add random spikes
    if random.random() < 0.1:  # 10% chance of spike
        spike = random.uniform(20, 40)
        base_load += spike
    
    # Add random noise
    noise = random.uniform(-5, 5)
    base_load += noise
    
    return max(0, base_load)

def main():
    """Demonstrate auto-scaling capabilities"""
    
    print("🛡️  Auto-Scaling Framework for Defensive Systems")
    print("=" * 55)
    
    # Initialize auto-scaler
    auto_scaler = DefensiveAutoScaler(min_capacity=2, max_capacity=10)
    
    # Add scaling callback
    def scaling_callback(old_capacity: int, new_capacity: int):
        direction = "UP" if new_capacity > old_capacity else "DOWN"
        print(f"  🔄 SCALING {direction}: {old_capacity} -> {new_capacity} instances")
    
    auto_scaler.add_scaling_callback(scaling_callback)
    
    print(f"\n📊 INITIAL CONFIGURATION")
    print("-" * 30)
    print(f"Current Capacity: {auto_scaler.current_capacity}")
    print(f"Capacity Range: {auto_scaler.min_capacity} - {auto_scaler.max_capacity}")
    
    # Start monitoring
    print(f"\n🔄 STARTING AUTO-SCALING DEMONSTRATION")
    print("-" * 40)
    auto_scaler.start_monitoring()
    
    # Simulate workload for 2 minutes
    simulation_duration = 120  # seconds
    start_time = time.time()
    
    print("Simulating defensive workload with auto-scaling...")
    
    try:
        while time.time() - start_time < simulation_duration:
            # Simulate workload metrics
            cpu_usage = simulate_defensive_workload()
            memory_usage = cpu_usage * 0.8 + random.uniform(-5, 5)
            queue_depth = max(0, (cpu_usage - 40) / 5)  # Queue builds up with high CPU
            response_time = max(100, cpu_usage * 20)  # Response time correlates with load
            
            # Update metrics
            auto_scaler.update_metric("cpu_usage", cpu_usage)
            auto_scaler.update_metric("memory_usage", memory_usage)
            auto_scaler.update_metric("queue_depth", queue_depth)
            auto_scaler.update_metric("response_time", response_time)
            
            # Print current state every 10 seconds
            if int(time.time() - start_time) % 10 == 0:
                print(f"  Time: {int(time.time() - start_time):3d}s | "
                      f"CPU: {cpu_usage:5.1f}% | "
                      f"Memory: {memory_usage:5.1f}% | "
                      f"Queue: {queue_depth:4.1f} | "
                      f"Capacity: {auto_scaler.current_capacity}")
            
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    # Stop monitoring
    auto_scaler.stop_monitoring()
    
    # Generate final report
    print(f"\n📈 SCALING REPORT")
    print("-" * 20)
    
    report = auto_scaler.get_scaling_report()
    
    print(f"Final Capacity: {report['current_capacity']}")
    print(f"Capacity Utilization: {report['utilization_percent']}%")
    print(f"Total Scaling Events: {report['scaling_statistics']['total_events']}")
    print(f"Scale Up Events: {report['scaling_statistics']['scale_up_events']}")
    print(f"Scale Down Events: {report['scaling_statistics']['scale_down_events']}")
    
    if report['last_scaling_event']:
        last_event = report['last_scaling_event']
        print(f"Last Scaling: {last_event['direction']} at {last_event['timestamp'][:19]}")
        print(f"Trigger: {last_event['trigger']} - {last_event['reason']}")
    
    print(f"\nCurrent Metrics:")
    for name, metric_data in report['current_metrics'].items():
        status = ""
        if metric_data['needs_scale_up']:
            status = " 🔴 (needs scale up)"
        elif metric_data['needs_scale_down']:
            status = " 🔵 (needs scale down)"
        else:
            status = " ✅ (stable)"
        
        print(f"  • {name}: {metric_data['current_value']:.1f}{status}")
    
    print(f"\nPredictive Insights:")
    for name, insight in report['predictive_insights'].items():
        if insight['predicted_next_value'] is not None:
            print(f"  • {name}: {insight['predicted_next_value']:.1f} (trend: {insight['trend']})")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # Export scaling data
    export_file = f"logs/scaling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("logs").mkdir(exist_ok=True)
    
    # Add event history to report
    report['scaling_events'] = [event.to_dict() for event in auto_scaler.scaling_events]
    
    with open(export_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Scaling report exported to: {export_file}")
    print("✅ Auto-scaling demonstration completed successfully!")

if __name__ == "__main__":
    main()