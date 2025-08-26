"""
Comprehensive Logging and Monitoring System

Advanced logging system with structured logging, multiple outputs,
performance monitoring, and real-time analytics.
"""

import logging
import logging.handlers
import json
import time
import traceback
import sys
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
import os
from enum import Enum

from .robust_error_handler import robust, ErrorSeverity


class LogLevel(Enum):
    """Enhanced log levels"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60
    PERFORMANCE = 70


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    process_id: int
    thread_id: int
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    security_context: Optional[Dict[str, Any]] = None
    additional_data: Optional[Dict[str, Any]] = None


class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
        self.lock = Lock()
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(value)
            
            # Keep only last 1000 values to prevent memory issues
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
    
    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter"""
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + amount
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = self.metrics[name]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "last": values[-1]
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics and counters"""
        with self.lock:
            return {
                "metrics": {
                    name: self.get_statistics(name) 
                    for name in self.metrics.keys()
                },
                "counters": self.counters.copy()
            }


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs"""
    
    def __init__(self, include_performance: bool = True):
        super().__init__()
        self.include_performance = include_performance
        self.performance_monitor = PerformanceMonitor()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Get additional context from record
        session_id = getattr(record, 'session_id', None)
        user_id = getattr(record, 'user_id', None)
        request_id = getattr(record, 'request_id', None)
        performance_metrics = getattr(record, 'performance_metrics', None)
        security_context = getattr(record, 'security_context', None)
        additional_data = getattr(record, 'additional_data', None)
        
        # Create structured log entry
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            process_id=record.process,
            thread_id=record.thread,
            session_id=session_id,
            user_id=user_id,
            request_id=request_id,
            performance_metrics=performance_metrics,
            security_context=security_context,
            additional_data=additional_data
        )
        
        # Add exception info if present
        entry_dict = asdict(log_entry)
        if record.exc_info:
            entry_dict['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(entry_dict, default=str)


class ComprehensiveLogger:
    """Main logging system with multiple outputs and monitoring"""
    
    def __init__(self, name: str = "gan_cyber_range"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.performance_monitor = PerformanceMonitor()
        
        # Create logs directory
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self._setup_logging()
        
        # Performance tracking
        self.request_times: Dict[str, float] = {}
        
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set base level
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for general logs
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "application.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(error_handler)
        
        # Security logs
        security_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "security.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=10
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(security_handler)
        
        # Performance logs
        performance_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=5
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(StructuredFormatter(include_performance=True))
        self.logger.addHandler(performance_handler)
        
        self.logger.info("Comprehensive logging system initialized")
    
    @robust(severity=ErrorSeverity.LOW)
    def log_with_context(
        self,
        level: int,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        security_context: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Log message with full context"""
        
        # Create log record with extra context
        extra = {
            'session_id': session_id,
            'user_id': user_id,
            'request_id': request_id,
            'performance_metrics': performance_metrics,
            'security_context': security_context,
            'additional_data': additional_data
        }
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self.log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self.log_with_context(logging.CRITICAL, message, **kwargs)
    
    def security_event(self, message: str, **kwargs):
        """Log security event"""
        security_context = kwargs.get('security_context', {})
        security_context['event_type'] = 'security'
        kwargs['security_context'] = security_context
        self.log_with_context(logging.WARNING, f"SECURITY: {message}", **kwargs)
    
    def performance_event(self, operation: str, duration: float, **kwargs):
        """Log performance event"""
        performance_metrics = kwargs.get('performance_metrics', {})
        performance_metrics.update({
            'operation': operation,
            'duration_ms': duration * 1000,
            'timestamp': datetime.now().isoformat()
        })
        kwargs['performance_metrics'] = performance_metrics
        
        # Record in performance monitor
        self.performance_monitor.record_metric(f"{operation}_duration", duration)
        self.performance_monitor.increment_counter(f"{operation}_count")
        
        self.log_with_context(
            logging.INFO, 
            f"PERFORMANCE: {operation} completed in {duration:.3f}s",
            **kwargs
        )
    
    def start_request(self, request_id: str, operation: str):
        """Start tracking a request"""
        self.request_times[request_id] = time.time()
        self.info(
            f"Started {operation}",
            request_id=request_id,
            additional_data={'operation': operation}
        )
    
    def end_request(self, request_id: str, operation: str, success: bool = True):
        """End tracking a request"""
        if request_id in self.request_times:
            duration = time.time() - self.request_times[request_id]
            del self.request_times[request_id]
            
            self.performance_event(
                operation,
                duration,
                request_id=request_id,
                additional_data={
                    'success': success,
                    'operation': operation
                }
            )
        else:
            self.warning(f"No start time found for request {request_id}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance monitoring report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": self.performance_monitor.get_all_metrics(),
            "active_requests": len(self.request_times),
            "log_directory": str(self.log_dir),
            "log_files": [
                {
                    "name": f.name,
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                }
                for f in self.log_dir.glob("*.log")
            ]
        }
    
    def get_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get log summary for specified time period"""
        # This would normally parse log files, but for now return basic info
        return {
            "period_hours": hours,
            "log_directory": str(self.log_dir),
            "available_logs": list(self.log_dir.glob("*.log")),
            "performance_summary": self.performance_monitor.get_all_metrics()
        }


def timed_operation(operation_name: str, logger: Optional[ComprehensiveLogger] = None):
    """Decorator to automatically time operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            request_id = f"{operation_name}_{int(start_time * 1000)}"
            
            if logger:
                logger.start_request(request_id, operation_name)
            
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                if logger:
                    logger.error(
                        f"Operation {operation_name} failed: {e}",
                        request_id=request_id,
                        additional_data={'exception': str(e)}
                    )
                raise
            finally:
                if logger:
                    logger.end_request(request_id, operation_name, success)
            
            return result
        return wrapper
    return decorator


# Global logger instance
comprehensive_logger = ComprehensiveLogger()

# Convenience functions
def log_info(message: str, **kwargs):
    """Global info logging"""
    comprehensive_logger.info(message, **kwargs)

def log_error(message: str, **kwargs):
    """Global error logging"""
    comprehensive_logger.error(message, **kwargs)

def log_security(message: str, **kwargs):
    """Global security logging"""
    comprehensive_logger.security_event(message, **kwargs)

def log_performance(operation: str, duration: float, **kwargs):
    """Global performance logging"""
    comprehensive_logger.performance_event(operation, duration, **kwargs)