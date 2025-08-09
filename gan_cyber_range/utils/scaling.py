"""
Auto-scaling and Distributed Computing Module
Handles automatic scaling of cyber range components and distributed processing
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
import psutil
import kubernetes
from kubernetes import client, config as k8s_config

from .logging_config import get_logger
from .monitoring import MetricsCollector
from .performance import PerformanceOptimizer

logger = get_logger(__name__)


class ScalingEvent(Enum):
    """Scaling event types"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"  # Horizontal scaling
    SCALE_IN = "scale_in"    # Horizontal scaling down


@dataclass
class ScalingDecision:
    """Scaling decision data structure"""
    event_type: ScalingEvent
    component: str
    current_replicas: int
    target_replicas: int
    reason: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceRequirements:
    """Resource requirements specification"""
    cpu_cores: float
    memory_gb: float
    gpu_count: int = 0
    storage_gb: float = 0
    network_bandwidth_mbps: float = 0
    
    def __mul__(self, factor: float) -> 'ResourceRequirements':
        """Scale resource requirements by factor"""
        return ResourceRequirements(
            cpu_cores=self.cpu_cores * factor,
            memory_gb=self.memory_gb * factor,
            gpu_count=int(self.gpu_count * factor),
            storage_gb=self.storage_gb * factor,
            network_bandwidth_mbps=self.network_bandwidth_mbps * factor
        )
    
    def __add__(self, other: 'ResourceRequirements') -> 'ResourceRequirements':
        """Add resource requirements"""
        return ResourceRequirements(
            cpu_cores=self.cpu_cores + other.cpu_cores,
            memory_gb=self.memory_gb + other.memory_gb,
            gpu_count=self.gpu_count + other.gpu_count,
            storage_gb=self.storage_gb + other.storage_gb,
            network_bandwidth_mbps=self.network_bandwidth_mbps + other.network_bandwidth_mbps
        )


class WorkloadPredictor:
    """Predicts future workload based on historical data"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.request_history = deque(maxlen=history_size)
        self.response_time_history = deque(maxlen=history_size)
        self.lock = threading.Lock()
    
    def record_metrics(self, cpu_usage: float, memory_usage: float, 
                      request_rate: float, avg_response_time: float):
        """Record current metrics for prediction"""
        with self.lock:
            timestamp = time.time()
            self.cpu_history.append((timestamp, cpu_usage))
            self.memory_history.append((timestamp, memory_usage))
            self.request_history.append((timestamp, request_rate))
            self.response_time_history.append((timestamp, avg_response_time))
    
    def predict_cpu_usage(self, horizon_minutes: int = 5) -> float:
        """Predict CPU usage for the next N minutes"""
        if len(self.cpu_history) < 10:
            return 50.0  # Default prediction
        
        with self.lock:
            recent_data = list(self.cpu_history)[-60:]  # Last hour
            
        if not recent_data:
            return 50.0
        
        # Simple linear trend prediction
        values = [data[1] for data in recent_data]
        avg_value = sum(values) / len(values)
        
        # Calculate trend
        if len(values) > 1:
            trend = (values[-1] - values[0]) / len(values)
            predicted = avg_value + (trend * horizon_minutes)
        else:
            predicted = avg_value
        
        return max(0, min(100, predicted))
    
    def predict_memory_usage(self, horizon_minutes: int = 5) -> float:
        """Predict memory usage for the next N minutes"""
        if len(self.memory_history) < 10:
            return 50.0
        
        with self.lock:
            recent_data = list(self.memory_history)[-60:]
        
        if not recent_data:
            return 50.0
        
        values = [data[1] for data in recent_data]
        avg_value = sum(values) / len(values)
        
        if len(values) > 1:
            trend = (values[-1] - values[0]) / len(values)
            predicted = avg_value + (trend * horizon_minutes)
        else:
            predicted = avg_value
        
        return max(0, min(100, predicted))
    
    def predict_request_rate(self, horizon_minutes: int = 5) -> float:
        """Predict request rate for the next N minutes"""
        if len(self.request_history) < 10:
            return 10.0  # Default requests per minute
        
        with self.lock:
            recent_data = list(self.request_history)[-60:]
        
        if not recent_data:
            return 10.0
        
        values = [data[1] for data in recent_data]
        avg_value = sum(values) / len(values)
        
        # Consider time-based patterns (e.g., higher load during work hours)
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 17:  # Work hours
            predicted = avg_value * 1.2
        else:
            predicted = avg_value * 0.8
        
        return max(0, predicted)
    
    def get_prediction_confidence(self) -> float:
        """Get confidence level of predictions based on data quality"""
        total_samples = len(self.cpu_history) + len(self.memory_history) + len(self.request_history)
        max_samples = self.history_size * 3
        
        # Confidence based on amount of historical data
        confidence = min(total_samples / (max_samples * 0.1), 1.0)
        
        return confidence


class AutoScaler:
    """Automatic scaling controller"""
    
    def __init__(self, 
                 scale_up_threshold: float = 70.0,
                 scale_down_threshold: float = 30.0,
                 min_replicas: int = 1,
                 max_replicas: int = 10,
                 cooldown_minutes: int = 5):
        
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.cooldown_minutes = cooldown_minutes
        
        self.last_scaling_actions = {}
        self.current_replicas = {}
        self.scaling_history = deque(maxlen=100)
        self.predictor = WorkloadPredictor()
        self.metrics_collector = MetricsCollector()
        
        self.is_running = False
        self._scaling_task = None
        self.lock = threading.Lock()
    
    async def start(self):
        """Start the auto-scaler"""
        if self.is_running:
            return
        
        self.is_running = True
        self._scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("AutoScaler started")
    
    async def stop(self):
        """Stop the auto-scaler"""
        self.is_running = False
        
        if self._scaling_task:
            await self._scaling_task
        
        logger.info("AutoScaler stopped")
    
    async def _scaling_loop(self):
        """Main scaling loop"""
        while self.is_running:
            try:
                # Collect current metrics
                metrics = self._collect_scaling_metrics()
                
                # Record for prediction
                self.predictor.record_metrics(
                    cpu_usage=metrics.get('cpu_usage', 0),
                    memory_usage=metrics.get('memory_usage', 0),
                    request_rate=metrics.get('request_rate', 0),
                    avg_response_time=metrics.get('avg_response_time', 0)
                )
                
                # Make scaling decisions
                decisions = await self._make_scaling_decisions(metrics)
                
                # Execute scaling decisions
                for decision in decisions:
                    await self._execute_scaling_decision(decision)
                
                # Wait before next evaluation
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"AutoScaler error: {e}")
                await asyncio.sleep(120)  # Longer wait on error
    
    def _collect_scaling_metrics(self) -> Dict[str, Any]:
        """Collect metrics for scaling decisions"""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Get application metrics from metrics collector
            app_metrics = self.metrics_collector.get_metrics()
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory.percent,
                'request_rate': app_metrics.get('requests_per_minute', 0),
                'avg_response_time': app_metrics.get('avg_response_time', 0),
                'active_connections': app_metrics.get('active_connections', 0),
                'queue_size': app_metrics.get('queue_size', 0)
            }
            
        except Exception as e:
            logger.error(f"Error collecting scaling metrics: {e}")
            return {}
    
    async def _make_scaling_decisions(self, metrics: Dict[str, Any]) -> List[ScalingDecision]:
        """Make scaling decisions based on current and predicted metrics"""
        decisions = []
        
        # Get predictions
        predicted_cpu = self.predictor.predict_cpu_usage(5)
        predicted_memory = self.predictor.predict_memory_usage(5)
        confidence = self.predictor.get_prediction_confidence()
        
        current_cpu = metrics.get('cpu_usage', 0)
        current_memory = metrics.get('memory_usage', 0)
        queue_size = metrics.get('queue_size', 0)
        
        # Check for scale-up conditions
        scale_up_needed = (
            current_cpu > self.scale_up_threshold or
            current_memory > self.scale_up_threshold or
            predicted_cpu > self.scale_up_threshold * 1.2 or
            queue_size > 100  # High queue indicates need for more capacity
        )
        
        # Check for scale-down conditions
        scale_down_possible = (
            current_cpu < self.scale_down_threshold and
            current_memory < self.scale_down_threshold and
            predicted_cpu < self.scale_down_threshold * 1.5 and
            queue_size < 10
        )
        
        # Make decisions for each component
        components = ['api', 'worker', 'range-manager']
        
        for component in components:
            current_replicas = self.current_replicas.get(component, self.min_replicas)
            
            # Check cooldown period
            if not self._is_cooldown_expired(component):
                continue
            
            if scale_up_needed and current_replicas < self.max_replicas:
                target_replicas = min(current_replicas + 1, self.max_replicas)
                
                # Calculate confidence-based scaling
                if confidence > 0.7:  # High confidence
                    target_replicas = min(current_replicas + 2, self.max_replicas)
                
                decision = ScalingDecision(
                    event_type=ScalingEvent.SCALE_OUT,
                    component=component,
                    current_replicas=current_replicas,
                    target_replicas=target_replicas,
                    reason=f"High resource usage: CPU={current_cpu:.1f}%, Memory={current_memory:.1f}%",
                    confidence=confidence,
                    metadata={'predicted_cpu': predicted_cpu, 'predicted_memory': predicted_memory}
                )
                decisions.append(decision)
            
            elif scale_down_possible and current_replicas > self.min_replicas:
                target_replicas = max(current_replicas - 1, self.min_replicas)
                
                decision = ScalingDecision(
                    event_type=ScalingEvent.SCALE_IN,
                    component=component,
                    current_replicas=current_replicas,
                    target_replicas=target_replicas,
                    reason=f"Low resource usage: CPU={current_cpu:.1f}%, Memory={current_memory:.1f}%",
                    confidence=confidence,
                    metadata={'predicted_cpu': predicted_cpu, 'predicted_memory': predicted_memory}
                )
                decisions.append(decision)
        
        return decisions
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision"""
        try:
            logger.info(f"Executing scaling decision: {decision}")
            
            # Update tracking
            with self.lock:
                self.current_replicas[decision.component] = decision.target_replicas
                self.last_scaling_actions[decision.component] = time.time()
                self.scaling_history.append(decision)
            
            # Execute the actual scaling (this would integrate with Kubernetes, Docker, etc.)
            success = await self._perform_scaling(decision)
            
            if success:
                logger.info(f"Successfully scaled {decision.component} to {decision.target_replicas} replicas")
            else:
                logger.error(f"Failed to scale {decision.component}")
                
        except Exception as e:
            logger.error(f"Error executing scaling decision: {e}")
    
    async def _perform_scaling(self, decision: ScalingDecision) -> bool:
        """Perform the actual scaling operation"""
        try:
            # In a real implementation, this would integrate with:
            # - Kubernetes HPA/VPA
            # - Docker Swarm
            # - Cloud provider auto-scaling groups
            # - Custom container orchestration
            
            # For now, simulate scaling
            await asyncio.sleep(1)  # Simulate scaling time
            
            # Here you would implement actual scaling logic:
            # if running on Kubernetes:
            # await self._scale_kubernetes_deployment(decision)
            # elif running on Docker Swarm:
            # await self._scale_docker_service(decision)
            
            return True
            
        except Exception as e:
            logger.error(f"Scaling operation failed: {e}")
            return False
    
    def _is_cooldown_expired(self, component: str) -> bool:
        """Check if cooldown period has expired for a component"""
        last_action = self.last_scaling_actions.get(component, 0)
        cooldown_seconds = self.cooldown_minutes * 60
        
        return time.time() - last_action > cooldown_seconds
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        with self.lock:
            return {
                'is_running': self.is_running,
                'current_replicas': dict(self.current_replicas),
                'last_scaling_actions': dict(self.last_scaling_actions),
                'recent_decisions': list(self.scaling_history)[-10:],
                'prediction_confidence': self.predictor.get_prediction_confidence(),
                'thresholds': {
                    'scale_up': self.scale_up_threshold,
                    'scale_down': self.scale_down_threshold
                }
            }


class KubernetesScaler:
    """Kubernetes-specific scaling implementation"""
    
    def __init__(self, namespace: str = "gan-cyber-range"):
        self.namespace = namespace
        self.k8s_client = None
        self.apps_v1 = None
        
        try:
            # Try to load in-cluster config first
            k8s_config.load_incluster_config()
        except:
            try:
                # Fall back to local kubeconfig
                k8s_config.load_kube_config()
            except Exception as e:
                logger.warning(f"Could not load Kubernetes config: {e}")
                return
        
        self.k8s_client = client.ApiClient()
        self.apps_v1 = client.AppsV1Api()
        logger.info("KubernetesScaler initialized")
    
    async def scale_deployment(self, deployment_name: str, target_replicas: int) -> bool:
        """Scale a Kubernetes deployment"""
        if not self.apps_v1:
            logger.error("Kubernetes client not initialized")
            return False
        
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Update replica count
            deployment.spec.replicas = target_replicas
            
            # Apply the update
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Scaled deployment {deployment_name} to {target_replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment {deployment_name}: {e}")
            return False
    
    async def get_deployment_replicas(self, deployment_name: str) -> Tuple[int, int]:
        """Get current and ready replicas for a deployment"""
        if not self.apps_v1:
            return 0, 0
        
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            current_replicas = deployment.spec.replicas or 0
            ready_replicas = deployment.status.ready_replicas or 0
            
            return current_replicas, ready_replicas
            
        except Exception as e:
            logger.error(f"Failed to get deployment status for {deployment_name}: {e}")
            return 0, 0
    
    async def create_horizontal_pod_autoscaler(self, deployment_name: str,
                                             min_replicas: int = 1,
                                             max_replicas: int = 10,
                                             target_cpu_utilization: int = 70) -> bool:
        """Create a Horizontal Pod Autoscaler"""
        try:
            autoscaling_v2 = client.AutoscalingV2Api()
            
            hpa_spec = client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=deployment_name
                ),
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                metrics=[
                    client.V2MetricSpec(
                        type="Resource",
                        resource=client.V2ResourceMetricSource(
                            name="cpu",
                            target=client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=target_cpu_utilization
                            )
                        )
                    )
                ]
            )
            
            hpa = client.V2HorizontalPodAutoscaler(
                api_version="autoscaling/v2",
                kind="HorizontalPodAutoscaler",
                metadata=client.V1ObjectMeta(
                    name=f"{deployment_name}-hpa",
                    namespace=self.namespace
                ),
                spec=hpa_spec
            )
            
            autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace,
                body=hpa
            )
            
            logger.info(f"Created HPA for deployment {deployment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create HPA for {deployment_name}: {e}")
            return False


class DistributedProcessor:
    """Distributed processing coordinator for large-scale operations"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.process_pool = None
        self.thread_pool = None
        self.task_queue = asyncio.Queue()
        self.results = {}
        self.is_running = False
        self._processor_task = None
    
    async def start(self):
        """Start the distributed processor"""
        if self.is_running:
            return
        
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        self.is_running = True
        self._processor_task = asyncio.create_task(self._processor_loop())
        
        logger.info(f"DistributedProcessor started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the distributed processor"""
        self.is_running = False
        
        if self._processor_task:
            await self._processor_task
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        logger.info("DistributedProcessor stopped")
    
    async def submit_task(self, func: Callable, args: tuple = (), 
                         kwargs: dict = None, task_id: str = None,
                         use_process_pool: bool = True) -> str:
        """Submit a task for distributed processing"""
        if kwargs is None:
            kwargs = {}
        
        task_id = task_id or f"task_{int(time.time() * 1000)}"
        
        task_info = {
            'task_id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'use_process_pool': use_process_pool,
            'submitted_at': time.time()
        }
        
        await self.task_queue.put(task_info)
        return task_id
    
    async def _processor_loop(self):
        """Main processor loop"""
        while self.is_running:
            try:
                # Get next task
                task_info = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Execute task
                asyncio.create_task(self._execute_distributed_task(task_info))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Processor loop error: {e}")
    
    async def _execute_distributed_task(self, task_info: Dict[str, Any]):
        """Execute a distributed task"""
        task_id = task_info['task_id']
        
        try:
            loop = asyncio.get_event_loop()
            
            if task_info['use_process_pool']:
                # Execute in process pool
                result = await loop.run_in_executor(
                    self.process_pool,
                    task_info['func'],
                    *task_info['args'],
                    **task_info['kwargs']
                )
            else:
                # Execute in thread pool
                result = await loop.run_in_executor(
                    self.thread_pool,
                    task_info['func'],
                    *task_info['args'],
                    **task_info['kwargs']
                )
            
            self.results[task_id] = {
                'status': 'completed',
                'result': result,
                'completed_at': time.time()
            }
            
        except Exception as e:
            logger.error(f"Distributed task {task_id} failed: {e}")
            self.results[task_id] = {
                'status': 'failed',
                'error': str(e),
                'completed_at': time.time()
            }
    
    async def get_result(self, task_id: str, timeout: float = 30.0) -> Any:
        """Get result of a distributed task"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.results:
                result_info = self.results[task_id]
                
                if result_info['status'] == 'completed':
                    return result_info['result']
                elif result_info['status'] == 'failed':
                    raise Exception(result_info['error'])
            
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distributed processor statistics"""
        completed_tasks = sum(1 for r in self.results.values() if r['status'] == 'completed')
        failed_tasks = sum(1 for r in self.results.values() if r['status'] == 'failed')
        
        return {
            'is_running': self.is_running,
            'max_workers': self.max_workers,
            'queue_size': self.task_queue.qsize(),
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'total_tasks': len(self.results)
        }


class ScalingCoordinator:
    """Main scaling coordinator that combines all scaling components"""
    
    def __init__(self):
        self.auto_scaler = AutoScaler()
        self.k8s_scaler = KubernetesScaler() if self._is_kubernetes_available() else None
        self.distributed_processor = DistributedProcessor()
        self.performance_optimizer = PerformanceOptimizer()
        
        self.is_running = False
    
    def _is_kubernetes_available(self) -> bool:
        """Check if running in Kubernetes environment"""
        try:
            k8s_config.load_incluster_config()
            return True
        except:
            try:
                k8s_config.load_kube_config()
                return True
            except:
                return False
    
    async def start(self):
        """Start all scaling components"""
        if self.is_running:
            return
        
        await self.auto_scaler.start()
        await self.distributed_processor.start()
        await self.performance_optimizer.start_monitoring()
        
        self.is_running = True
        logger.info("ScalingCoordinator started")
    
    async def stop(self):
        """Stop all scaling components"""
        self.is_running = False
        
        await self.auto_scaler.stop()
        await self.distributed_processor.stop()
        await self.performance_optimizer.stop_monitoring()
        
        logger.info("ScalingCoordinator stopped")
    
    async def scale_component(self, component: str, target_replicas: int) -> bool:
        """Manually scale a component"""
        if self.k8s_scaler:
            return await self.k8s_scaler.scale_deployment(component, target_replicas)
        else:
            logger.warning("Manual scaling not available without Kubernetes")
            return False
    
    async def submit_distributed_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task for distributed processing"""
        return await self.distributed_processor.submit_task(func, args, kwargs)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status"""
        return {
            'is_running': self.is_running,
            'auto_scaler': self.auto_scaler.get_scaling_status(),
            'distributed_processor': self.distributed_processor.get_stats(),
            'performance_optimizer': self.performance_optimizer.get_comprehensive_stats(),
            'kubernetes_available': self.k8s_scaler is not None
        }


# Global scaling coordinator instance
global_scaling_coordinator = ScalingCoordinator()


# Utility functions for easy access
async def start_auto_scaling():
    """Start auto-scaling"""
    await global_scaling_coordinator.start()


async def stop_auto_scaling():
    """Stop auto-scaling"""
    await global_scaling_coordinator.stop()


async def scale_component(component: str, replicas: int) -> bool:
    """Scale a component to specified replicas"""
    return await global_scaling_coordinator.scale_component(component, replicas)


async def distribute_task(func: Callable, *args, **kwargs) -> str:
    """Distribute a task for parallel processing"""
    return await global_scaling_coordinator.submit_distributed_task(func, *args, **kwargs)