"""Utility modules for GAN-Cyber-Range-v2"""

# Import defensive utilities (no external dependencies)
try:
    from .robust_validation import DefensiveValidator, RobustErrorHandler
    from .defensive_monitoring import DefensiveMonitor
    ROBUST_AVAILABLE = True
except ImportError:
    ROBUST_AVAILABLE = False

# Import original utilities with graceful fallback
try:
    from .logging_config import setup_logging, get_logger
except ImportError:
    def setup_logging():
        import logging
        logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        import logging
        return logging.getLogger(name)

try:
    from .validation import validate_config, ValidationError
except ImportError:
    ValidationError = ValueError
    def validate_config(config):
        return True

try:
    from .error_handling import CyberRangeError, AttackExecutionError, NetworkSimulationError
except ImportError:
    CyberRangeError = Exception
    AttackExecutionError = Exception
    NetworkSimulationError = Exception

try:
    from .monitoring import MetricsCollector, PerformanceMonitor
except ImportError:
    MetricsCollector = None
    PerformanceMonitor = None

try:
    from .security import SecurityValidator, EthicalFramework
except ImportError:
    SecurityValidator = None
    EthicalFramework = None

__all__ = [
    "setup_logging",
    "get_logger", 
    "validate_config",
    "ValidationError",
    "CyberRangeError",
    "AttackExecutionError", 
    "NetworkSimulationError",
    "MetricsCollector",
    "PerformanceMonitor",
    "SecurityValidator",
    "EthicalFramework",
    "DefensiveValidator",
    "RobustErrorHandler",
    "DefensiveMonitor",
    "ROBUST_AVAILABLE"
]