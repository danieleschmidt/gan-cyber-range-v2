"""
Comprehensive logging configuration for GAN-Cyber-Range-v2.

This module provides centralized logging configuration with structured logging,
multiple output formats, and security event logging capabilities.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
from dataclasses import dataclass, asdict


@dataclass
class LogEvent:
    """Structured log event for security and attack events"""
    timestamp: str
    level: str
    module: str
    event_type: str
    message: str
    details: Dict[str, Any]
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SecurityEventFormatter(logging.Formatter):
    """Custom formatter for security events with JSON output"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Create base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'module': record.module,
            'message': record.getMessage(),
            'file': record.filename,
            'line': record.lineno,
            'function': record.funcName
        }
        
        # Add security-specific fields if present
        if hasattr(record, 'event_type'):
            log_entry['event_type'] = record.event_type
            
        if hasattr(record, 'attack_id'):
            log_entry['attack_id'] = record.attack_id
            
        if hasattr(record, 'technique_id'):
            log_entry['technique_id'] = record.technique_id
            
        if hasattr(record, 'target_host'):
            log_entry['target_host'] = record.target_host
            
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)


class CyberRangeLogger:
    """Enhanced logger for cyber range operations"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.session_id = None
        self._local = threading.local()
    
    def set_session_id(self, session_id: str) -> None:
        """Set session ID for this thread"""
        self.session_id = session_id
        self._local.session_id = session_id
    
    def get_session_id(self) -> Optional[str]:
        """Get session ID for current thread"""
        return getattr(self._local, 'session_id', self.session_id)
    
    def log_attack_event(
        self,
        level: str,
        message: str,
        attack_id: str,
        technique_id: Optional[str] = None,
        target_host: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log attack-related events with structured data"""
        
        extra = {
            'event_type': 'attack',
            'attack_id': attack_id,
            'session_id': self.get_session_id(),
            **kwargs
        }
        
        if technique_id:
            extra['technique_id'] = technique_id
        if target_host:
            extra['target_host'] = target_host
        
        getattr(self.logger, level.lower())(message, extra=extra)
    
    def log_detection_event(
        self,
        level: str,
        message: str,
        detection_type: str,
        confidence: float,
        source_host: str,
        **kwargs
    ) -> None:
        """Log detection events from security tools"""
        
        extra = {
            'event_type': 'detection',
            'detection_type': detection_type,
            'confidence': confidence,
            'source_host': source_host,
            'session_id': self.get_session_id(),
            **kwargs
        }
        
        getattr(self.logger, level.lower())(message, extra=extra)
    
    def log_range_event(
        self,
        level: str,
        message: str,
        range_id: str,
        operation: str,
        **kwargs
    ) -> None:
        """Log cyber range operational events"""
        
        extra = {
            'event_type': 'range_operation',
            'range_id': range_id,
            'operation': operation,
            'session_id': self.get_session_id(),
            **kwargs
        }
        
        getattr(self.logger, level.lower())(message, extra=extra)
    
    def log_training_event(
        self,
        level: str,
        message: str,
        team_id: str,
        scenario: str,
        **kwargs
    ) -> None:
        """Log training and evaluation events"""
        
        extra = {
            'event_type': 'training',
            'team_id': team_id,
            'scenario': scenario,
            'session_id': self.get_session_id(),
            **kwargs
        }
        
        getattr(self.logger, level.lower())(message, extra=extra)
    
    # Delegate standard logging methods
    def debug(self, message: str, **kwargs) -> None:
        extra = {'session_id': self.get_session_id(), **kwargs}
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs) -> None:
        extra = {'session_id': self.get_session_id(), **kwargs}
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs) -> None:
        extra = {'session_id': self.get_session_id(), **kwargs}
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs) -> None:
        extra = {'session_id': self.get_session_id(), **kwargs}
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, **kwargs) -> None:
        extra = {'session_id': self.get_session_id(), **kwargs}
        self.logger.critical(message, extra=extra)


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_file_logging: bool = True,
    enable_json_logging: bool = True,
    enable_syslog: bool = False,
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5
) -> None:
    """
    Configure comprehensive logging for the cyber range system.
    
    Args:
        log_level: Minimum logging level
        log_dir: Directory for log files
        enable_file_logging: Enable file-based logging
        enable_json_logging: Enable JSON-formatted security logs
        enable_syslog: Enable syslog integration
        max_file_size: Maximum size per log file
        backup_count: Number of backup files to keep
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    if enable_file_logging:
        # General application log
        app_handler = logging.handlers.RotatingFileHandler(
            log_path / "cyber_range.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        app_handler.setLevel(logging.DEBUG)
        
        app_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        app_handler.setFormatter(app_format)
        root_logger.addHandler(app_handler)
        
        # Error log
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "errors.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(app_format)
        root_logger.addHandler(error_handler)
    
    if enable_json_logging:
        # Security events log with JSON formatting
        security_handler = logging.handlers.RotatingFileHandler(
            log_path / "security_events.json",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        security_handler.setLevel(logging.INFO)
        security_handler.setFormatter(SecurityEventFormatter())
        
        # Create security logger
        security_logger = logging.getLogger('security')
        security_logger.addHandler(security_handler)
        security_logger.propagate = False
        
        # Attack events log
        attack_handler = logging.handlers.RotatingFileHandler(
            log_path / "attack_events.json",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        attack_handler.setLevel(logging.INFO)
        attack_handler.setFormatter(SecurityEventFormatter())
        
        attack_logger = logging.getLogger('attacks')
        attack_logger.addHandler(attack_handler)
        attack_logger.propagate = False
        
        # Detection events log
        detection_handler = logging.handlers.RotatingFileHandler(
            log_path / "detection_events.json",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        detection_handler.setLevel(logging.INFO)
        detection_handler.setFormatter(SecurityEventFormatter())
        
        detection_logger = logging.getLogger('detections')
        detection_logger.addHandler(detection_handler)
        detection_logger.propagate = False
    
    if enable_syslog:
        # Syslog handler for centralized logging
        try:
            syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
            syslog_handler.setLevel(logging.INFO)
            
            syslog_format = logging.Formatter(
                'cyber_range[%(process)d]: %(name)s - %(levelname)s - %(message)s'
            )
            syslog_handler.setFormatter(syslog_format)
            root_logger.addHandler(syslog_handler)
            
        except Exception as e:
            # Fallback if syslog is not available
            root_logger.warning(f"Could not configure syslog: {e}")
    
    # Configure specific loggers
    _configure_module_loggers()
    
    logging.info("Logging system initialized successfully")


def _configure_module_loggers() -> None:
    """Configure module-specific loggers with appropriate levels"""
    
    # Set appropriate levels for third-party libraries
    logging.getLogger('docker').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('paramiko').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    # Configure cyber range module loggers
    logging.getLogger('gan_cyber_range.core').setLevel(logging.DEBUG)
    logging.getLogger('gan_cyber_range.red_team').setLevel(logging.DEBUG)
    logging.getLogger('gan_cyber_range.blue_team').setLevel(logging.DEBUG)
    logging.getLogger('gan_cyber_range.generators').setLevel(logging.DEBUG)


def get_logger(name: str) -> CyberRangeLogger:
    """
    Get an enhanced logger instance for cyber range operations.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Enhanced logger with cyber range specific methods
    """
    return CyberRangeLogger(name)


class AuditLogger:
    """Specialized logger for audit trails and compliance"""
    
    def __init__(self, audit_file: str = "logs/audit.log"):
        self.audit_file = Path(audit_file)
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure audit logger
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        # Audit log should be append-only with no rotation for compliance
        handler = logging.FileHandler(self.audit_file)
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False
    
    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log user actions for audit trail"""
        
        audit_entry = {
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'result': result,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        if details:
            audit_entry['details'] = details
        
        self.logger.info(json.dumps(audit_entry))
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        severity: str = "INFO",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log system events for audit trail"""
        
        audit_entry = {
            'event_type': event_type,
            'description': description,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        if details:
            audit_entry['details'] = details
        
        self.logger.info(json.dumps(audit_entry))


class PerformanceLogger:
    """Logger for performance metrics and timing"""
    
    def __init__(self):
        self.logger = get_logger('performance')
        self.timers = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        self.timers[operation] = datetime.now()
    
    def end_timer(self, operation: str, details: Optional[Dict[str, Any]] = None) -> float:
        """End timing and log duration"""
        if operation not in self.timers:
            self.logger.warning(f"Timer not found for operation: {operation}")
            return 0.0
        
        start_time = self.timers.pop(operation)
        duration = (datetime.now() - start_time).total_seconds()
        
        log_details = {'duration_seconds': duration}
        if details:
            log_details.update(details)
        
        self.logger.info(f"Operation '{operation}' completed", **log_details)
        return duration
    
    def log_metric(self, metric_name: str, value: float, unit: str = "", **kwargs) -> None:
        """Log a performance metric"""
        details = {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            **kwargs
        }
        
        self.logger.info(f"Performance metric: {metric_name} = {value} {unit}", **details)