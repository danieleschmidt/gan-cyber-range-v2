#!/usr/bin/env python3
"""
Robust error handling and recovery system for defensive cybersecurity operations

This module provides comprehensive error handling, retry logic, circuit breakers,
and automatic recovery mechanisms for defensive security systems.
"""

import time
import logging
import threading
import traceback
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import hashlib

# Setup structured logging for error handling
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    MEMORY = "memory"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"

class RecoveryAction(Enum):
    """Available recovery actions"""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESET = "reset"
    ESCALATE = "escalate"
    IGNORE = "ignore"

@dataclass
class ErrorContext:
    """Context information for error analysis"""
    error_id: str
    timestamp: datetime
    function_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    stack_trace: str
    metadata: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    test_request_volume: int = 3

class CircuitBreaker:
    """Circuit breaker pattern for resilient defensive operations"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        self.lock = threading.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute(func, *args, **kwargs)
        
        return wrapper
    
    def _execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful execution"""
        
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.test_request_volume:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' moved to CLOSED")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed execution"""
        
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if (self.state == CircuitBreakerState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' moved to OPEN")
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' moved back to OPEN")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class RetryConfig:
    """Configuration for retry logic"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

def retry_on_error(config: RetryConfig = None, 
                  exceptions: tuple = (Exception,),
                  error_handler: Optional[Callable] = None):
    """Decorator for retrying functions on specific exceptions"""
    
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if error_handler:
                        error_handler(e, attempt + 1, config.max_attempts)
                    
                    if attempt < config.max_attempts - 1:  # Don't sleep on last attempt
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, "
                                     f"retrying in {delay:.1f}s: {str(e)}")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {config.max_attempts} attempts failed for {func.__name__}")
            
            # All attempts failed, raise the last exception
            raise last_exception
        
        return wrapper
    
    return decorator

class DefensiveErrorHandler:
    """Comprehensive error handling system for defensive operations"""
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        self.error_patterns = {}
        self.max_history_size = 1000
        
        # Register default recovery strategies
        self._register_default_strategies()
        
        logger.info("Defensive error handler initialized")
    
    def _register_default_strategies(self):
        """Register default error recovery strategies"""
        
        self.register_recovery_strategy(
            ErrorCategory.NETWORK,
            self._network_recovery_strategy
        )
        
        self.register_recovery_strategy(
            ErrorCategory.FILESYSTEM,
            self._filesystem_recovery_strategy
        )
        
        self.register_recovery_strategy(
            ErrorCategory.MEMORY,
            self._memory_recovery_strategy
        )
        
        self.register_recovery_strategy(
            ErrorCategory.CONFIGURATION,
            self._configuration_recovery_strategy
        )
        
        logger.info("Default recovery strategies registered")
    
    def register_recovery_strategy(self, category: ErrorCategory, 
                                 strategy: Callable[[ErrorContext], bool]):
        """Register a recovery strategy for an error category"""
        
        self.recovery_strategies[category] = strategy
        logger.info(f"Registered recovery strategy for {category.value}")
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        
        return self.circuit_breakers[name]
    
    def _classify_error(self, error: Exception, function_name: str) -> tuple[ErrorSeverity, ErrorCategory]:
        """Classify error by severity and category"""
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Determine severity
        severity = ErrorSeverity.MEDIUM
        if any(keyword in error_message for keyword in ['critical', 'fatal', 'security']):
            severity = ErrorSeverity.CRITICAL
        elif any(keyword in error_message for keyword in ['warning', 'deprecated']):
            severity = ErrorSeverity.LOW
        elif any(keyword in error_message for keyword in ['error', 'failed', 'exception']):
            severity = ErrorSeverity.HIGH
        
        # Determine category
        category = ErrorCategory.UNKNOWN
        if any(keyword in error_message for keyword in ['network', 'connection', 'timeout', 'socket']):
            category = ErrorCategory.NETWORK
        elif any(keyword in error_message for keyword in ['file', 'directory', 'path', 'permission']):
            category = ErrorCategory.FILESYSTEM
        elif any(keyword in error_message for keyword in ['memory', 'allocation', 'out of']):
            category = ErrorCategory.MEMORY
        elif any(keyword in error_message for keyword in ['auth', 'credential', 'token', 'forbidden']):
            category = ErrorCategory.AUTHENTICATION
        elif any(keyword in error_message for keyword in ['config', 'setting', 'parameter']):
            category = ErrorCategory.CONFIGURATION
        elif any(keyword in error_message for keyword in ['validation', 'invalid', 'format']):
            category = ErrorCategory.VALIDATION
        elif any(keyword in error_message for keyword in ['service', 'api', 'endpoint']):
            category = ErrorCategory.EXTERNAL_SERVICE
        
        return severity, category
    
    def handle_error(self, error: Exception, function_name: str, 
                    metadata: Dict[str, Any] = None) -> ErrorContext:
        """Handle and classify an error"""
        
        # Generate error ID
        error_id = hashlib.md5(
            f"{function_name}_{type(error).__name__}_{str(error)[:100]}".encode()
        ).hexdigest()[:12]
        
        # Classify error
        severity, category = self._classify_error(error, function_name)
        
        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            function_name=function_name,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            stack_trace=traceback.format_exc(),
            metadata=metadata or {}
        )
        
        # Add to history
        self.error_history.append(error_context)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logger.info,
            ErrorSeverity.MEDIUM: logger.warning,
            ErrorSeverity.HIGH: logger.error,
            ErrorSeverity.CRITICAL: logger.critical
        }[severity]
        
        log_level(f"Error handled: {error_id} - {category.value} - {error}")
        
        # Attempt recovery
        if category in self.recovery_strategies:
            try:
                recovery_successful = self.recovery_strategies[category](error_context)
                error_context.recovery_attempted = True
                error_context.recovery_successful = recovery_successful
                
                if recovery_successful:
                    logger.info(f"Recovery successful for error: {error_id}")
                else:
                    logger.warning(f"Recovery failed for error: {error_id}")
                    
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
        
        return error_context
    
    def _network_recovery_strategy(self, context: ErrorContext) -> bool:
        """Recovery strategy for network errors"""
        
        logger.info(f"Attempting network recovery for error: {context.error_id}")
        
        # Simulate network recovery attempts
        recovery_attempts = [
            "Checking network connectivity",
            "Refreshing DNS resolution", 
            "Testing alternate endpoints",
            "Validating network configuration"
        ]
        
        for attempt in recovery_attempts:
            logger.info(f"Network recovery: {attempt}")
            time.sleep(0.1)  # Simulate recovery time
        
        # Simulate success/failure (in real implementation, perform actual recovery)
        import random
        success = random.random() > 0.3  # 70% success rate for demo
        
        if success:
            logger.info("Network recovery successful")
        else:
            logger.warning("Network recovery failed")
        
        return success
    
    def _filesystem_recovery_strategy(self, context: ErrorContext) -> bool:
        """Recovery strategy for filesystem errors"""
        
        logger.info(f"Attempting filesystem recovery for error: {context.error_id}")
        
        try:
            # Create necessary directories
            if "directory" in context.error_message.lower():
                # Extract potential directory from error message
                potential_dirs = ["logs", "data", "config", "temp"]
                for dir_name in potential_dirs:
                    if dir_name in context.error_message.lower():
                        Path(dir_name).mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created directory: {dir_name}")
            
            # Check disk space
            import shutil
            disk_usage = shutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 1.0:  # Less than 1GB free
                logger.warning("Low disk space detected during filesystem recovery")
                return False
            
            logger.info("Filesystem recovery successful")
            return True
            
        except Exception as e:
            logger.error(f"Filesystem recovery failed: {e}")
            return False
    
    def _memory_recovery_strategy(self, context: ErrorContext) -> bool:
        """Recovery strategy for memory errors"""
        
        logger.info(f"Attempting memory recovery for error: {context.error_id}")
        
        try:
            import gc
            import psutil
            
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                logger.warning("Critical memory usage detected")
                return False
            
            logger.info("Memory recovery successful")
            return True
            
        except Exception as e:
            logger.error(f"Memory recovery failed: {e}")
            return False
    
    def _configuration_recovery_strategy(self, context: ErrorContext) -> bool:
        """Recovery strategy for configuration errors"""
        
        logger.info(f"Attempting configuration recovery for error: {context.error_id}")
        
        try:
            # Create default configuration
            config_dir = Path("configs/defensive")
            config_dir.mkdir(parents=True, exist_ok=True)
            
            default_config = {
                "defensive_mode": True,
                "log_level": "INFO",
                "monitoring_enabled": True,
                "security_validation": True,
                "error_handling": {
                    "retry_attempts": 3,
                    "circuit_breaker_enabled": True
                }
            }
            
            config_file = config_dir / "default_config.json"
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            logger.info(f"Created default configuration: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration recovery failed: {e}")
            return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and patterns"""
        
        if not self.error_history:
            return {"total_errors": 0}
        
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history 
                        if (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
        
        # Count by category
        category_counts = {}
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for error in self.error_history:
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        # Calculate recovery rate
        recovery_attempts = sum(1 for e in self.error_history if e.recovery_attempted)
        successful_recoveries = sum(1 for e in self.error_history if e.recovery_successful)
        recovery_rate = (successful_recoveries / max(recovery_attempts, 1)) * 100
        
        return {
            "total_errors": total_errors,
            "recent_errors": len(recent_errors),
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "recovery_attempts": recovery_attempts,
            "successful_recoveries": successful_recoveries,
            "recovery_rate_percent": round(recovery_rate, 2),
            "circuit_breakers": {name: cb.state.value for name, cb in self.circuit_breakers.items()}
        }

# Utility decorators for defensive error handling
def defensive_operation(error_handler: DefensiveErrorHandler = None,
                       circuit_breaker_name: str = None,
                       retry_config: RetryConfig = None):
    """Comprehensive decorator for defensive operations"""
    
    if error_handler is None:
        error_handler = DefensiveErrorHandler()
    
    def decorator(func: Callable) -> Callable:
        
        # Apply circuit breaker if specified
        if circuit_breaker_name:
            circuit_breaker = error_handler.get_circuit_breaker(circuit_breaker_name)
            func = circuit_breaker(func)
        
        # Apply retry logic if specified
        if retry_config:
            func = retry_on_error(retry_config)(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle error through error handler
                error_context = error_handler.handle_error(e, func.__name__)
                
                # Re-raise if recovery was not successful
                if not error_context.recovery_successful:
                    raise
                
                # If recovery was successful, try the operation once more
                try:
                    return func(*args, **kwargs)
                except Exception:
                    # If it still fails after recovery, raise the original error
                    raise e
        
        return wrapper
    
    return decorator

def main():
    """Demonstrate robust error handling capabilities"""
    
    print("🛡️  Robust Error Handling & Recovery System")
    print("=" * 55)
    
    # Initialize error handler
    error_handler = DefensiveErrorHandler()
    
    # Demonstrate circuit breaker
    @defensive_operation(
        error_handler=error_handler,
        circuit_breaker_name="demo_service",
        retry_config=RetryConfig(max_attempts=2)
    )
    def unreliable_defensive_operation():
        """Demo function that fails sometimes"""
        import random
        if random.random() < 0.7:  # 70% failure rate for demo
            raise ConnectionError("Simulated network failure in defensive system")
        return "Defensive operation successful"
    
    # Test error handling and recovery
    print("\n🔄 Testing Error Handling & Recovery")
    print("-" * 40)
    
    successes = 0
    for i in range(10):
        try:
            result = unreliable_defensive_operation()
            print(f"  ✅ Attempt {i+1}: {result}")
            successes += 1
        except Exception as e:
            print(f"  ❌ Attempt {i+1}: Failed after all recovery attempts")
        
        time.sleep(0.5)  # Brief pause between attempts
    
    # Show statistics
    print(f"\n📊 ERROR HANDLING STATISTICS")
    print("-" * 35)
    
    stats = error_handler.get_error_statistics()
    print(f"Total Errors: {stats['total_errors']}")
    print(f"Recent Errors: {stats['recent_errors']}")
    print(f"Recovery Rate: {stats['recovery_rate_percent']}%")
    print(f"Successful Operations: {successes}/10")
    
    print(f"\nError Categories:")
    for category, count in stats['category_distribution'].items():
        print(f"  • {category}: {count}")
    
    print(f"\nCircuit Breaker States:")
    for name, state in stats['circuit_breakers'].items():
        state_emoji = {
            'closed': '✅',
            'open': '🚨', 
            'half_open': '⚠️'
        }
        emoji = state_emoji.get(state, '🔍')
        print(f"  {emoji} {name}: {state}")
    
    # Export error data
    export_file = f"logs/error_handling_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("logs").mkdir(exist_ok=True)
    
    with open(export_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "error_history": [
                {
                    "error_id": e.error_id,
                    "timestamp": e.timestamp.isoformat(),
                    "category": e.category.value,
                    "severity": e.severity.value,
                    "recovery_successful": e.recovery_successful
                }
                for e in error_handler.error_history
            ]
        }, f, indent=2)
    
    print(f"\n💾 Error handling data exported to: {export_file}")
    print("✅ Robust error handling demonstration completed")

if __name__ == "__main__":
    main()