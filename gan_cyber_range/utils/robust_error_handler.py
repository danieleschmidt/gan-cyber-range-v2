"""
Robust Error Handling and Recovery System

Comprehensive error handling with automatic recovery,
graceful degradation, and detailed logging.
"""

import logging
import traceback
import time
import functools
from typing import Dict, Any, Optional, Callable, List, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    FAIL_SAFE = "fail_safe"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context information for errors"""
    timestamp: datetime
    function_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    stack_trace: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


class RobustErrorHandler:
    """Advanced error handler with recovery strategies"""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_handlers: Dict[Type[Exception], Callable] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.error_metrics = {
            'total_errors': 0,
            'critical_errors': 0,
            'recoverable_errors': 0,
            'recovery_success_rate': 0.0
        }
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self._setup_error_logging()
        
        # Register default handlers
        self._register_default_handlers()
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    ) -> Optional[Any]:
        """Handle an error with specified strategy"""
        
        error_context = ErrorContext(
            timestamp=datetime.now(),
            function_name=context.get('function') if context else 'unknown',
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            recovery_strategy=recovery_strategy,
            stack_trace=traceback.format_exc(),
            context_data=context or {}
        )
        
        # Log the error
        self._log_error(error_context)
        
        # Update metrics
        self._update_error_metrics(error_context)
        
        # Store error history
        self.error_history.append(error_context)
        
        # Apply recovery strategy
        return self._apply_recovery_strategy(error_context, error)
    
    def with_error_handling(
        self,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        max_retries: int = 3,
        fallback_value: Any = None
    ):
        """Decorator for robust error handling"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                context = {
                    'function': func.__name__,
                    'args': str(args)[:100],  # Truncate for logging
                    'kwargs': str(kwargs)[:100]
                }
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_retries:
                            # Final attempt failed
                            result = self.handle_error(
                                e, context, severity, recovery_strategy
                            )
                            return result if result is not None else fallback_value
                        else:
                            # Retry with backoff
                            wait_time = 2 ** attempt
                            self.logger.info(
                                f"Retrying {func.__name__} (attempt {attempt + 1}/{max_retries}) "
                                f"after {wait_time}s: {e}"
                            )
                            time.sleep(wait_time)
                            continue
            return wrapper
        return decorator
    
    def register_recovery_handler(self, exception_type: Type[Exception], handler: Callable):
        """Register a custom recovery handler for specific exceptions"""
        self.recovery_handlers[exception_type] = handler
    
    def register_fallback_handler(self, function_name: str, handler: Callable):
        """Register a fallback handler for a specific function"""
        self.fallback_handlers[function_name] = handler
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary and metrics"""
        recent_errors = [
            err for err in self.error_history
            if err.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "critical_errors": sum(
                1 for err in self.error_history 
                if err.severity == ErrorSeverity.CRITICAL
            ),
            "error_types": self._get_error_type_distribution(),
            "recovery_success_rate": self.error_metrics['recovery_success_rate'],
            "most_common_errors": self._get_most_common_errors()
        }
    
    def _apply_recovery_strategy(self, context: ErrorContext, error: Exception) -> Optional[Any]:
        """Apply the specified recovery strategy"""
        
        # Check for custom recovery handler
        for exc_type, handler in self.recovery_handlers.items():
            if isinstance(error, exc_type):
                try:
                    return handler(error, context)
                except Exception as recovery_error:
                    self.logger.error(f"Recovery handler failed: {recovery_error}")
        
        # Apply strategy
        if context.recovery_strategy == RecoveryStrategy.RETRY:
            return self._retry_strategy(context, error)
        elif context.recovery_strategy == RecoveryStrategy.FALLBACK:
            return self._fallback_strategy(context, error)
        elif context.recovery_strategy == RecoveryStrategy.DEGRADE:
            return self._degrade_strategy(context, error)
        elif context.recovery_strategy == RecoveryStrategy.FAIL_SAFE:
            return self._fail_safe_strategy(context, error)
        elif context.recovery_strategy == RecoveryStrategy.ABORT:
            raise error
        
        return None
    
    def _retry_strategy(self, context: ErrorContext, error: Exception) -> Optional[Any]:
        """Implement retry strategy with exponential backoff"""
        if context.recovery_attempts >= context.max_recovery_attempts:
            self.logger.error(f"Max retry attempts exceeded for {context.function_name}")
            return None
        
        wait_time = 2 ** context.recovery_attempts
        self.logger.info(f"Retrying {context.function_name} after {wait_time}s")
        time.sleep(wait_time)
        context.recovery_attempts += 1
        
        return None  # Caller should retry
    
    def _fallback_strategy(self, context: ErrorContext, error: Exception) -> Optional[Any]:
        """Implement fallback strategy"""
        function_name = context.function_name
        
        if function_name in self.fallback_handlers:
            try:
                self.logger.info(f"Using fallback handler for {function_name}")
                return self.fallback_handlers[function_name](error, context)
            except Exception as fallback_error:
                self.logger.error(f"Fallback handler failed: {fallback_error}")
        
        # Default fallbacks based on error type
        if isinstance(error, (ConnectionError, TimeoutError)):
            return {"status": "offline", "message": "Service temporarily unavailable"}
        elif isinstance(error, FileNotFoundError):
            return {"status": "not_found", "message": "Resource not found"}
        elif isinstance(error, PermissionError):
            return {"status": "permission_denied", "message": "Access denied"}
        
        return None
    
    def _degrade_strategy(self, context: ErrorContext, error: Exception) -> Optional[Any]:
        """Implement graceful degradation strategy"""
        self.logger.info(f"Degrading functionality for {context.function_name}")
        
        # Return minimal functionality response
        return {
            "status": "degraded",
            "message": "Operating with reduced functionality",
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }
    
    def _fail_safe_strategy(self, context: ErrorContext, error: Exception) -> Optional[Any]:
        """Implement fail-safe strategy"""
        self.logger.info(f"Activating fail-safe mode for {context.function_name}")
        
        # Return safe default values
        safe_defaults = {
            "list": [],
            "dict": {},
            "str": "",
            "int": 0,
            "float": 0.0,
            "bool": False
        }
        
        # Try to determine expected return type from context
        expected_type = context.context_data.get('expected_return_type', 'dict')
        return safe_defaults.get(expected_type, None)
    
    def _register_default_handlers(self):
        """Register default recovery handlers"""
        
        # Network-related errors
        def network_recovery_handler(error: Exception, context: ErrorContext):
            if isinstance(error, (ConnectionError, TimeoutError)):
                return {
                    "status": "network_error",
                    "retry_after": 30,
                    "message": "Network connectivity issues detected"
                }
            return None
        
        self.register_recovery_handler(ConnectionError, network_recovery_handler)
        self.register_recovery_handler(TimeoutError, network_recovery_handler)
        
        # Import errors (missing dependencies)
        def import_recovery_handler(error: ImportError, context: ErrorContext):
            missing_module = str(error).split("'")[1] if "'" in str(error) else "unknown"
            return {
                "status": "dependency_missing",
                "missing_module": missing_module,
                "message": f"Optional dependency {missing_module} not available",
                "install_command": f"pip install {missing_module}"
            }
        
        self.register_recovery_handler(ImportError, import_recovery_handler)
    
    def _setup_error_logging(self):
        """Setup comprehensive error logging"""
        # Create error log file handler
        error_handler = logging.FileHandler('logs/errors.log')
        error_handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(formatter)
        
        self.logger.addHandler(error_handler)
    
    def _log_error(self, context: ErrorContext):
        """Log error with appropriate level"""
        message = (
            f"Error in {context.function_name}: {context.error_message} "
            f"(Severity: {context.severity.value}, Strategy: {context.recovery_strategy.value})"
        )
        
        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(message)
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(message)
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def _update_error_metrics(self, context: ErrorContext):
        """Update error metrics"""
        self.error_metrics['total_errors'] += 1
        
        if context.severity == ErrorSeverity.CRITICAL:
            self.error_metrics['critical_errors'] += 1
        
        if context.recovery_strategy != RecoveryStrategy.ABORT:
            self.error_metrics['recoverable_errors'] += 1
        
        # Calculate recovery success rate
        total_recoverable = self.error_metrics['recoverable_errors']
        if total_recoverable > 0:
            successful_recoveries = sum(
                1 for err in self.error_history
                if err.recovery_strategy != RecoveryStrategy.ABORT
                and err.recovery_attempts < err.max_recovery_attempts
            )
            self.error_metrics['recovery_success_rate'] = (
                successful_recoveries / total_recoverable
            )
    
    def _get_error_type_distribution(self) -> Dict[str, int]:
        """Get distribution of error types"""
        distribution = {}
        for error in self.error_history:
            error_type = error.error_type
            distribution[error_type] = distribution.get(error_type, 0) + 1
        return distribution
    
    def _get_most_common_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most common errors"""
        error_counts = self._get_error_type_distribution()
        sorted_errors = sorted(
            error_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        return [
            {"error_type": error_type, "count": count}
            for error_type, count in sorted_errors
        ]


# Global error handler instance
error_handler = RobustErrorHandler()

# Convenience decorators
def robust(severity: ErrorSeverity = ErrorSeverity.MEDIUM, **kwargs):
    """Simple robust error handling decorator"""
    return error_handler.with_error_handling(severity=severity, **kwargs)


def critical(max_retries: int = 5, **kwargs):
    """Critical operation with maximum recovery attempts"""
    return error_handler.with_error_handling(
        severity=ErrorSeverity.CRITICAL,
        recovery_strategy=RecoveryStrategy.RETRY,
        max_retries=max_retries,
        **kwargs
    )


def safe(fallback_value=None, **kwargs):
    """Safe operation with fallback value"""
    return error_handler.with_error_handling(
        severity=ErrorSeverity.LOW,
        recovery_strategy=RecoveryStrategy.FAIL_SAFE,
        fallback_value=fallback_value,
        **kwargs
    )