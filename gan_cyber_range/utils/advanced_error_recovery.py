"""
Advanced error recovery and resilience mechanisms.
Implements circuit breakers, retry logic, and graceful degradation.
"""

import logging
import time
import asyncio
import threading
from typing import Dict, Any, Optional, Callable, List, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import functools
import traceback

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"
    RECOVERING = "recovering"


class FailureMode(Enum):
    """Types of failure modes"""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    VALIDATION_ERROR = "validation_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_function: Callable
    interval_seconds: int = 30
    timeout_seconds: int = 5
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enabled: bool = True


@dataclass
class RecoveryAction:
    """Recovery action configuration"""
    name: str
    action_function: Callable
    trigger_conditions: List[str]
    max_attempts: int = 3
    backoff_factor: float = 2.0
    enabled: bool = True


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = ServiceState.HEALTHY
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            current_time = time.time()
            
            # Check if we should attempt recovery
            if (self.state == ServiceState.FAILING and 
                self.last_failure_time and
                current_time - self.last_failure_time > self.recovery_timeout):
                self.state = ServiceState.RECOVERING
                logger.info(f"Circuit breaker {self.name} attempting recovery")
            
            # Fail fast if circuit is open
            if self.state == ServiceState.FAILING:
                raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is open")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            with self.lock:
                if self.state == ServiceState.RECOVERING:
                    self.state = ServiceState.HEALTHY
                    logger.info(f"Circuit breaker {self.name} recovered")
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = current_time
                
                if self.failure_count >= self.failure_threshold:
                    self.state = ServiceState.FAILING
                    logger.error(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
                elif self.failure_count > self.failure_threshold // 2:
                    self.state = ServiceState.DEGRADED
            
            raise


class RetryManager:
    """Advanced retry mechanisms with exponential backoff"""
    
    def __init__(self):
        self.retry_configs = {}
    
    def retry_with_backoff(
        self,
        func: Callable,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retry_on: List[Type[Exception]] = None
    ):
        """Execute function with exponential backoff retry"""
        
        if retry_on is None:
            retry_on = [Exception]
        
        last_exception = None
        delay = base_delay
        
        for attempt in range(max_attempts):
            try:
                return func()
            except Exception as e:
                last_exception = e
                
                # Check if we should retry on this exception
                should_retry = any(isinstance(e, exc_type) for exc_type in retry_on)
                
                if not should_retry or attempt == max_attempts - 1:
                    raise
                
                # Add jitter to prevent thundering herd
                actual_delay = delay
                if jitter:
                    import random
                    actual_delay *= (0.5 + random.random() * 0.5)
                
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {actual_delay:.1f}s: {e}")
                time.sleep(actual_delay)
                
                # Increase delay for next attempt
                delay = min(delay * backoff_factor, max_delay)
        
        raise last_exception


class GracefulDegradation:
    """Graceful degradation when services are unavailable"""
    
    def __init__(self):
        self.fallback_strategies = {}
        self.feature_flags = {}
    
    def register_fallback(self, service_name: str, fallback_function: Callable):
        """Register fallback strategy for service"""
        self.fallback_strategies[service_name] = fallback_function
        logger.info(f"Registered fallback for service: {service_name}")
    
    def set_feature_flag(self, feature_name: str, enabled: bool):
        """Set feature flag"""
        self.feature_flags[feature_name] = enabled
        logger.info(f"Feature flag {feature_name} set to {enabled}")
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if feature is enabled"""
        return self.feature_flags.get(feature_name, True)
    
    def execute_with_fallback(self, service_name: str, primary_function: Callable, *args, **kwargs):
        """Execute with fallback if primary fails"""
        try:
            return primary_function(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed for {service_name}: {e}")
            
            if service_name in self.fallback_strategies:
                logger.info(f"Using fallback strategy for {service_name}")
                return self.fallback_strategies[service_name](*args, **kwargs)
            else:
                logger.error(f"No fallback available for {service_name}")
                raise


class HealthMonitor:
    """Continuous health monitoring"""
    
    def __init__(self):
        self.health_checks = {}
        self.health_status = {}
        self.monitoring_active = False
        self.monitor_thread = None
    
    def register_health_check(self, health_check: HealthCheck):
        """Register health check"""
        self.health_checks[health_check.name] = health_check
        self.health_status[health_check.name] = {
            'state': ServiceState.HEALTHY,
            'last_check': None,
            'consecutive_failures': 0,
            'consecutive_successes': 0
        }
        logger.info(f"Registered health check: {health_check.name}")
    
    def start_monitoring(self):
        """Start health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            for name, health_check in self.health_checks.items():
                if not health_check.enabled:
                    continue
                
                try:
                    self._execute_health_check(name, health_check)
                except Exception as e:
                    logger.error(f"Error executing health check {name}: {e}")
            
            time.sleep(1)  # Check every second for due health checks
    
    def _execute_health_check(self, name: str, health_check: HealthCheck):
        """Execute individual health check"""
        status = self.health_status[name]
        now = datetime.now()
        
        # Check if it's time for this health check
        if (status['last_check'] and 
            (now - status['last_check']).total_seconds() < health_check.interval_seconds):
            return
        
        try:
            # Execute health check with timeout
            result = self._execute_with_timeout(
                health_check.check_function,
                health_check.timeout_seconds
            )
            
            # Health check passed
            status['consecutive_failures'] = 0
            status['consecutive_successes'] += 1
            status['last_check'] = now
            
            # Update state based on recovery threshold
            if (status['state'] in [ServiceState.FAILING, ServiceState.DEGRADED] and
                status['consecutive_successes'] >= health_check.recovery_threshold):
                status['state'] = ServiceState.HEALTHY
                logger.info(f"Service {name} recovered")
            
        except Exception as e:
            # Health check failed
            status['consecutive_successes'] = 0
            status['consecutive_failures'] += 1
            status['last_check'] = now
            
            # Update state based on failure threshold
            if status['consecutive_failures'] >= health_check.failure_threshold:
                if status['state'] == ServiceState.HEALTHY:
                    status['state'] = ServiceState.DEGRADED
                elif status['state'] == ServiceState.DEGRADED:
                    status['state'] = ServiceState.FAILING
                logger.warning(f"Service {name} health degraded: {e}")
    
    def _execute_with_timeout(self, func: Callable, timeout_seconds: int):
        """Execute function with timeout"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Health check timed out after {timeout_seconds}s")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        overall_state = ServiceState.HEALTHY
        
        for status in self.health_status.values():
            if status['state'] == ServiceState.FAILING:
                overall_state = ServiceState.FAILING
                break
            elif status['state'] == ServiceState.DEGRADED and overall_state == ServiceState.HEALTHY:
                overall_state = ServiceState.DEGRADED
        
        return {
            'overall_state': overall_state,
            'services': dict(self.health_status),
            'monitoring_active': self.monitoring_active
        }


class ResilienceOrchestrator:
    """Main resilience orchestration"""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.retry_manager = RetryManager()
        self.degradation_manager = GracefulDegradation()
        self.health_monitor = HealthMonitor()
        self.recovery_actions = {}
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, **kwargs)
        return self.circuit_breakers[name]
    
    def register_recovery_action(self, action: RecoveryAction):
        """Register recovery action"""
        self.recovery_actions[action.name] = action
        logger.info(f"Registered recovery action: {action.name}")
    
    def execute_recovery_actions(self, failure_mode: FailureMode):
        """Execute applicable recovery actions"""
        for action in self.recovery_actions.values():
            if not action.enabled:
                continue
            
            if failure_mode.value in action.trigger_conditions:
                try:
                    logger.info(f"Executing recovery action: {action.name}")
                    action.action_function()
                except Exception as e:
                    logger.error(f"Recovery action {action.name} failed: {e}")
    
    def start_monitoring(self):
        """Start all monitoring"""
        self.health_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop all monitoring"""
        self.health_monitor.stop_monitoring()
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get overall resilience status"""
        return {
            'circuit_breakers': {
                name: {
                    'state': cb.state,
                    'failure_count': cb.failure_count
                } for name, cb in self.circuit_breakers.items()
            },
            'health_status': self.health_monitor.get_health_status(),
            'recovery_actions': len(self.recovery_actions),
            'fallback_strategies': len(self.degradation_manager.fallback_strategies)
        }


# Exceptions
class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open"""
    pass


class ResilienceException(Exception):
    """Base exception for resilience issues"""
    pass


# Decorators
def with_circuit_breaker(name: str, **kwargs):
    """Decorator to add circuit breaker protection"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs_inner):
            orchestrator = ResilienceOrchestrator()
            circuit_breaker = orchestrator.get_circuit_breaker(name, **kwargs)
            return circuit_breaker.call(func, *args, **kwargs_inner)
        return wrapper
    return decorator


def with_retry(max_attempts: int = 3, **retry_kwargs):
    """Decorator to add retry logic"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs_inner):
            orchestrator = ResilienceOrchestrator()
            return orchestrator.retry_manager.retry_with_backoff(
                lambda: func(*args, **kwargs_inner),
                max_attempts=max_attempts,
                **retry_kwargs
            )
        return wrapper
    return decorator


def with_fallback(service_name: str, fallback_func: Callable):
    """Decorator to add fallback execution"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            orchestrator = ResilienceOrchestrator()
            return orchestrator.degradation_manager.execute_with_fallback(
                service_name, func, *args, **kwargs
            )
        return wrapper
    return decorator


# Global resilience instance
resilience_orchestrator = ResilienceOrchestrator()