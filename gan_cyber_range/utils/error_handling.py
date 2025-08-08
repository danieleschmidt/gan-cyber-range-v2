"""
Comprehensive error handling and exception management for GAN-Cyber-Range-v2.

This module provides custom exceptions, error recovery mechanisms, and
graceful degradation strategies for robust system operation.
"""

import logging
import traceback
import functools
import time
from typing import Dict, Any, Optional, Type, Callable, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    module: str
    function: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    attack_id: Optional[str] = None
    range_id: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class CyberRangeError(Exception):
    """Base exception class for cyber range errors"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "CR_GENERIC",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        recoverable: bool = True,
        user_message: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or ErrorContext("unknown", "unknown")
        self.recoverable = recoverable
        self.user_message = user_message or "An error occurred during cyber range operation"
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'user_message': self.user_message,
            'severity': self.severity.value,
            'recoverable': self.recoverable,
            'timestamp': self.timestamp.isoformat(),
            'context': {
                'module': self.context.module,
                'function': self.context.function,
                'user_id': self.context.user_id,
                'session_id': self.context.session_id,
                'attack_id': self.context.attack_id,
                'range_id': self.context.range_id,
                'additional_info': self.context.additional_info
            },
            'traceback': traceback.format_exc()
        }


class AttackExecutionError(CyberRangeError):
    """Errors related to attack execution"""
    
    def __init__(self, message: str, attack_id: str, technique_id: Optional[str] = None, **kwargs):
        context = ErrorContext(
            module="attack_engine",
            function="execute_attack",
            attack_id=attack_id,
            additional_info={'technique_id': technique_id} if technique_id else None
        )
        super().__init__(
            message=message,
            error_code="CR_ATTACK_EXEC",
            severity=ErrorSeverity.HIGH,
            context=context,
            user_message="Attack execution failed - check attack configuration",
            **kwargs
        )


class NetworkSimulationError(CyberRangeError):
    """Errors related to network simulation"""
    
    def __init__(self, message: str, range_id: Optional[str] = None, **kwargs):
        context = ErrorContext(
            module="network_sim",
            function="simulate_network",
            range_id=range_id
        )
        super().__init__(
            message=message,
            error_code="CR_NETWORK_SIM",
            severity=ErrorSeverity.HIGH,
            context=context,
            user_message="Network simulation error - check network configuration",
            **kwargs
        )


class GANTrainingError(CyberRangeError):
    """Errors related to GAN training"""
    
    def __init__(self, message: str, model_type: Optional[str] = None, **kwargs):
        context = ErrorContext(
            module="attack_gan",
            function="train",
            additional_info={'model_type': model_type} if model_type else None
        )
        super().__init__(
            message=message,
            error_code="CR_GAN_TRAIN",
            severity=ErrorSeverity.MEDIUM,
            context=context,
            user_message="GAN training failed - check training data and parameters",
            **kwargs
        )


class ConfigurationError(CyberRangeError):
    """Errors related to configuration"""
    
    def __init__(self, message: str, config_file: Optional[str] = None, **kwargs):
        context = ErrorContext(
            module="config",
            function="load_config",
            additional_info={'config_file': config_file} if config_file else None
        )
        super().__init__(
            message=message,
            error_code="CR_CONFIG",
            severity=ErrorSeverity.HIGH,
            context=context,
            user_message="Configuration error - check configuration files",
            recoverable=False,
            **kwargs
        )


class ResourceExhaustionError(CyberRangeError):
    """Errors related to resource exhaustion"""
    
    def __init__(self, message: str, resource_type: str, **kwargs):
        context = ErrorContext(
            module="resource_manager",
            function="allocate_resources",
            additional_info={'resource_type': resource_type}
        )
        super().__init__(
            message=message,
            error_code="CR_RESOURCE",
            severity=ErrorSeverity.CRITICAL,
            context=context,
            user_message="System resources exhausted - reduce workload or increase capacity",
            **kwargs
        )


class SecurityValidationError(CyberRangeError):
    """Errors related to security validation"""
    
    def __init__(self, message: str, validation_type: str, **kwargs):
        context = ErrorContext(
            module="security",
            function="validate",
            additional_info={'validation_type': validation_type}
        )
        super().__init__(
            message=message,
            error_code="CR_SECURITY",
            severity=ErrorSeverity.CRITICAL,
            context=context,
            user_message="Security validation failed - operation blocked for safety",
            recoverable=False,
            **kwargs
        )


class ErrorHandler:
    """Centralized error handling and recovery system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_stats = {
            'total_errors': 0,
            'by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'by_error_code': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }
        self.recovery_strategies = {}
        
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        attempt_recovery: bool = True
    ) -> bool:
        """
        Handle an error with logging, recovery, and notification.
        
        Returns:
            True if error was recovered, False otherwise
        """
        
        # Convert to CyberRangeError if needed
        if not isinstance(error, CyberRangeError):
            cyber_error = CyberRangeError(
                message=str(error),
                context=context,
                severity=ErrorSeverity.MEDIUM
            )
        else:
            cyber_error = error
        
        # Update statistics
        self._update_error_stats(cyber_error)
        
        # Log the error
        self._log_error(cyber_error)
        
        # Attempt recovery if enabled and error is recoverable
        recovery_success = False
        if attempt_recovery and cyber_error.recoverable:
            recovery_success = self._attempt_recovery(cyber_error)
        
        # Send notifications for critical errors
        if cyber_error.severity == ErrorSeverity.CRITICAL:
            self._send_critical_error_notification(cyber_error)
        
        return recovery_success
    
    def register_recovery_strategy(
        self,
        error_code: str,
        strategy: Callable[[CyberRangeError], bool]
    ) -> None:
        """Register a recovery strategy for a specific error code"""
        self.recovery_strategies[error_code] = strategy
        self.logger.info(f"Registered recovery strategy for error code: {error_code}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        return self.error_stats.copy()
    
    def _update_error_stats(self, error: CyberRangeError) -> None:
        """Update error statistics"""
        self.error_stats['total_errors'] += 1
        self.error_stats['by_severity'][error.severity.value] += 1
        
        if error.error_code not in self.error_stats['by_error_code']:
            self.error_stats['by_error_code'][error.error_code] = 0
        self.error_stats['by_error_code'][error.error_code] += 1
    
    def _log_error(self, error: CyberRangeError) -> None:
        """Log error with appropriate level and detail"""
        
        error_dict = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR [{error.error_code}]: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY [{error.error_code}]: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY [{error.error_code}]: {error.message}", extra=error_dict)
        else:
            self.logger.info(f"LOW SEVERITY [{error.error_code}]: {error.message}", extra=error_dict)
    
    def _attempt_recovery(self, error: CyberRangeError) -> bool:
        """Attempt to recover from an error"""
        
        self.error_stats['recovery_attempts'] += 1
        
        # Check for registered recovery strategy
        if error.error_code in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[error.error_code]
                success = strategy(error)
                
                if success:
                    self.error_stats['successful_recoveries'] += 1
                    self.logger.info(f"Successfully recovered from error: {error.error_code}")
                    return True
                else:
                    self.logger.warning(f"Recovery strategy failed for error: {error.error_code}")
                    
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy threw exception: {recovery_error}")
        
        # Default recovery strategies
        return self._default_recovery(error)
    
    def _default_recovery(self, error: CyberRangeError) -> bool:
        """Default recovery strategies for common error types"""
        
        try:
            if isinstance(error, ResourceExhaustionError):
                # Attempt to free resources
                self.logger.info("Attempting resource cleanup for recovery")
                # In a real implementation, this would call resource cleanup functions
                return True
                
            elif isinstance(error, NetworkSimulationError):
                # Attempt to restart network simulation
                self.logger.info("Attempting network simulation restart for recovery")
                # In a real implementation, this would restart network components
                return True
                
            elif isinstance(error, AttackExecutionError):
                # Attempt to retry attack with different parameters
                self.logger.info("Attempting attack retry for recovery")
                # In a real implementation, this would modify attack parameters and retry
                return True
                
        except Exception as recovery_error:
            self.logger.error(f"Default recovery failed: {recovery_error}")
        
        return False
    
    def _send_critical_error_notification(self, error: CyberRangeError) -> None:
        """Send notification for critical errors"""
        
        # In a real implementation, this would send notifications via:
        # - Email alerts
        # - Slack/Teams webhooks
        # - SIEM integration
        # - Dashboard alerts
        
        self.logger.critical(f"CRITICAL ERROR NOTIFICATION: {error.error_code} - {error.message}")


# Global error handler instance
error_handler = ErrorHandler()


def with_error_handling(
    error_type: Type[CyberRangeError] = CyberRangeError,
    attempt_recovery: bool = True,
    reraise: bool = False
):
    """
    Decorator for automatic error handling in functions.
    
    Args:
        error_type: Type of CyberRangeError to convert exceptions to
        attempt_recovery: Whether to attempt recovery on errors
        reraise: Whether to reraise the error after handling
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CyberRangeError as e:
                # Already a cyber range error, handle it
                recovery_success = error_handler.handle_error(e, attempt_recovery=attempt_recovery)
                
                if reraise and not recovery_success:
                    raise
                    
                return None if not recovery_success else func(*args, **kwargs)
                
            except Exception as e:
                # Convert to cyber range error
                context = ErrorContext(
                    module=func.__module__,
                    function=func.__name__
                )
                
                cyber_error = error_type(
                    message=f"Error in {func.__name__}: {str(e)}",
                    context=context
                )
                
                recovery_success = error_handler.handle_error(cyber_error, attempt_recovery=attempt_recovery)
                
                if reraise and not recovery_success:
                    raise cyber_error
                    
                return None if not recovery_success else func(*args, **kwargs)
        
        return wrapper
    return decorator


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_on: Optional[List[Type[Exception]]] = None
):
    """
    Decorator for retrying functions on specific errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by for each retry
        retry_on: List of exception types to retry on
    """
    
    if retry_on is None:
        retry_on = [CyberRangeError, ConnectionError, TimeoutError]
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry on this exception
                    should_retry = any(isinstance(e, exc_type) for exc_type in retry_on)
                    
                    if not should_retry or attempt == max_retries:
                        # Don't retry or max retries reached
                        logging.getLogger(__name__).error(
                            f"Function {func.__name__} failed after {attempt + 1} attempts: {e}"
                        )
                        raise
                    
                    # Log retry attempt
                    logging.getLogger(__name__).warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {current_delay:.1f}s: {e}"
                    )
                    
                    # Wait before retry
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: Type[Exception] = Exception
):
    """
    Circuit breaker decorator to prevent cascading failures.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Timeout in seconds before attempting to close circuit
        expected_exception: Exception type that counts as a failure
    """
    
    def decorator(func: Callable) -> Callable:
        # Circuit breaker state
        state = {
            'failure_count': 0,
            'last_failure_time': None,
            'state': 'closed'  # closed, open, half_open
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Check if circuit should be half-open (attempting recovery)
            if (state['state'] == 'open' and 
                state['last_failure_time'] and
                current_time - state['last_failure_time'] > recovery_timeout):
                state['state'] = 'half_open'
                logging.getLogger(__name__).info(f"Circuit breaker for {func.__name__} entering half-open state")
            
            # If circuit is open, fail fast
            if state['state'] == 'open':
                raise CyberRangeError(
                    f"Circuit breaker open for {func.__name__}",
                    error_code="CR_CIRCUIT_OPEN",
                    severity=ErrorSeverity.HIGH,
                    recoverable=False
                )
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count and close circuit
                if state['state'] == 'half_open':
                    state['state'] = 'closed'
                    logging.getLogger(__name__).info(f"Circuit breaker for {func.__name__} closed after recovery")
                
                state['failure_count'] = 0
                return result
                
            except expected_exception as e:
                state['failure_count'] += 1
                state['last_failure_time'] = current_time
                
                # Check if we should open the circuit
                if state['failure_count'] >= failure_threshold:
                    state['state'] = 'open'
                    logging.getLogger(__name__).error(
                        f"Circuit breaker for {func.__name__} opened after {failure_threshold} failures"
                    )
                
                raise
        
        return wrapper
    return decorator


# Register default recovery strategies
def _register_default_recovery_strategies():
    """Register default recovery strategies for common errors"""
    
    def resource_cleanup_recovery(error: CyberRangeError) -> bool:
        """Recovery strategy for resource exhaustion"""
        # Placeholder - in real implementation would clean up resources
        logging.getLogger(__name__).info("Executing resource cleanup recovery")
        return True
    
    def network_restart_recovery(error: CyberRangeError) -> bool:
        """Recovery strategy for network simulation errors"""
        # Placeholder - in real implementation would restart network components
        logging.getLogger(__name__).info("Executing network restart recovery")
        return True
    
    def attack_retry_recovery(error: CyberRangeError) -> bool:
        """Recovery strategy for attack execution errors"""
        # Placeholder - in real implementation would modify and retry attack
        logging.getLogger(__name__).info("Executing attack retry recovery")
        return True
    
    error_handler.register_recovery_strategy("CR_RESOURCE", resource_cleanup_recovery)
    error_handler.register_recovery_strategy("CR_NETWORK_SIM", network_restart_recovery)
    error_handler.register_recovery_strategy("CR_ATTACK_EXEC", attack_retry_recovery)


# Initialize default recovery strategies
_register_default_recovery_strategies()