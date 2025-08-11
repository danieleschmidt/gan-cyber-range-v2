"""
Configuration Management System for GAN-Cyber-Range-v2

Advanced configuration management with:
- Environment-based configuration loading
- Dynamic configuration updates
- Configuration validation and schema enforcement
- Secure secrets management
- Hot-reloading capabilities
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import threading
from enum import Enum
import hashlib
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigScope(Enum):
    """Configuration scope levels"""
    GLOBAL = "global"
    RANGE = "range"
    USER = "user"
    SESSION = "session"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "gan_cyber_range"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    
    def get_connection_string(self) -> str:
        """Get database connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"


@dataclass
class CacheConfig:
    """Cache configuration"""
    type: str = "memory"  # memory, redis, memcached
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    database: int = 0
    max_size: int = 10000
    ttl_seconds: int = 3600
    cleanup_interval: int = 300


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = ""
    token_expiry_hours: int = 24
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    password_min_length: int = 8
    require_mfa: bool = False
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    rate_limit_requests: int = 100
    rate_limit_window_minutes: int = 15
    encryption_algorithm: str = "AES-256-GCM"


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_concurrent_ranges: int = 10
    max_attacks_per_batch: int = 100
    worker_threads: int = 4
    worker_processes: int = 2
    request_timeout_seconds: int = 30
    cache_enabled: bool = True
    compression_enabled: bool = True
    monitoring_interval_seconds: int = 5
    auto_scaling_enabled: bool = True
    memory_limit_mb: int = 2048


@dataclass
class AttackConfig:
    """Attack generation configuration"""
    default_count: int = 10
    max_count: int = 1000
    allowed_types: List[str] = field(default_factory=lambda: ["malware", "network", "web", "social_engineering"])
    enable_diversity_filter: bool = True
    min_diversity_threshold: float = 0.5
    max_severity_level: float = 0.8
    enable_stealth_filtering: bool = True
    containment_enabled: bool = True
    auto_mitigation: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/gan_cyber_range.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    enable_json: bool = False
    security_log_path: str = "logs/security.log"


@dataclass
class AppConfig:
    """Main application configuration"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    version: str = "2.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    reload: bool = True
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Feature flags
    enable_ml_features: bool = False
    enable_advanced_attacks: bool = True
    enable_real_time_monitoring: bool = True
    enable_api_versioning: bool = True


class ConfigValidator:
    """Configuration validation and schema enforcement"""
    
    def __init__(self):
        self.validation_rules = {}
        self._setup_validation_rules()
    
    def _setup_validation_rules(self) -> None:
        """Set up configuration validation rules"""
        self.validation_rules = {
            'database.port': lambda x: 1 <= x <= 65535,
            'database.pool_size': lambda x: 1 <= x <= 100,
            'cache.port': lambda x: 1 <= x <= 65535,
            'cache.max_size': lambda x: x > 0,
            'cache.ttl_seconds': lambda x: x > 0,
            'security.token_expiry_hours': lambda x: 1 <= x <= 168,  # 1 hour to 1 week
            'security.max_failed_attempts': lambda x: 1 <= x <= 10,
            'security.password_min_length': lambda x: 4 <= x <= 64,
            'security.rate_limit_requests': lambda x: x > 0,
            'performance.max_concurrent_ranges': lambda x: 1 <= x <= 100,
            'performance.max_attacks_per_batch': lambda x: 1 <= x <= 10000,
            'performance.worker_threads': lambda x: 1 <= x <= 64,
            'performance.memory_limit_mb': lambda x: x >= 512,
            'attack.default_count': lambda x: 1 <= x <= 1000,
            'attack.max_count': lambda x: x >= 1,
            'attack.min_diversity_threshold': lambda x: 0.0 <= x <= 1.0,
            'attack.max_severity_level': lambda x: 0.0 <= x <= 1.0,
            'logging.max_file_size_mb': lambda x: x > 0,
            'logging.backup_count': lambda x: x >= 0,
        }
    
    def validate_config(self, config: AppConfig) -> Dict[str, Any]:
        """Validate configuration against rules"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Convert config to flat dictionary for validation
        flat_config = self._flatten_config(asdict(config))
        
        for rule_path, validator in self.validation_rules.items():
            if rule_path in flat_config:
                value = flat_config[rule_path]
                try:
                    if not validator(value):
                        validation_result['is_valid'] = False
                        validation_result['errors'].append(f"Validation failed for {rule_path}: {value}")
                except Exception as e:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(f"Validation error for {rule_path}: {e}")
        
        # Environment-specific validations
        if config.environment == Environment.PRODUCTION:
            if config.debug:
                validation_result['warnings'].append("Debug mode should be disabled in production")
            if config.security.secret_key == "":
                validation_result['is_valid'] = False
                validation_result['errors'].append("Secret key must be set in production")
        
        return validation_result
    
    def _flatten_config(self, config_dict: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten nested configuration dictionary"""
        items = []
        for key, value in config_dict.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_config(value, new_key).items())
            else:
                items.append((new_key, value))
        return dict(items)


class SecureConfigManager:
    """Secure configuration management with encryption"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or self._generate_key()
        self.encrypted_fields = {
            'database.password',
            'cache.password', 
            'security.secret_key'
        }
    
    def _generate_key(self) -> str:
        """Generate encryption key"""
        return hashlib.sha256(os.urandom(32)).hexdigest()
    
    def encrypt_sensitive_data(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive configuration data"""
        flat_config = self._flatten_dict(config_dict)
        
        for field_path in self.encrypted_fields:
            if field_path in flat_config and flat_config[field_path]:
                # Simple encryption (in production, use proper encryption)
                encrypted_value = self._simple_encrypt(str(flat_config[field_path]))
                self._set_nested_value(config_dict, field_path, encrypted_value)
        
        return config_dict
    
    def decrypt_sensitive_data(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive configuration data"""
        flat_config = self._flatten_dict(config_dict)
        
        for field_path in self.encrypted_fields:
            if field_path in flat_config and flat_config[field_path]:
                try:
                    decrypted_value = self._simple_decrypt(flat_config[field_path])
                    self._set_nested_value(config_dict, field_path, decrypted_value)
                except Exception:
                    # Value might not be encrypted
                    pass
        
        return config_dict
    
    def _simple_encrypt(self, value: str) -> str:
        """Simple encryption (replace with proper encryption in production)"""
        return f"ENC:{hashlib.md5((value + self.encryption_key).encode()).hexdigest()}"
    
    def _simple_decrypt(self, encrypted_value: str) -> str:
        """Simple decryption (replace with proper decryption in production)"""
        if encrypted_value.startswith("ENC:"):
            # This is a placeholder - in production, implement proper decryption
            return "decrypted_value"
        return encrypted_value
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _set_nested_value(self, d: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation"""
        keys = key_path.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value


class ConfigManager:
    """Advanced configuration management system"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.validator = ConfigValidator()
        self.secure_manager = SecureConfigManager()
        
        self._config = None
        self._config_lock = threading.RLock()
        self._file_watchers = {}
        self._callbacks = []
        
        # Configuration file paths
        self.base_config_file = self.config_dir / "base.yaml"
        self.env_config_file = None
        self.secrets_file = self.config_dir / "secrets.yaml"
        
        logger.info(f"Initialized ConfigManager with config_dir: {config_dir}")
    
    def load_config(self, environment: Environment = Environment.DEVELOPMENT) -> AppConfig:
        """Load configuration for specified environment"""
        with self._config_lock:
            # Determine environment config file
            self.env_config_file = self.config_dir / f"{environment.value}.yaml"
            
            # Load base configuration
            base_config = self._load_config_file(self.base_config_file)
            
            # Load environment-specific configuration
            env_config = self._load_config_file(self.env_config_file)
            
            # Load secrets
            secrets_config = self._load_config_file(self.secrets_file)
            
            # Merge configurations (environment overrides base, secrets override all)
            merged_config = self._deep_merge(base_config, env_config)
            merged_config = self._deep_merge(merged_config, secrets_config)
            
            # Load from environment variables
            env_overrides = self._load_from_environment()
            merged_config = self._deep_merge(merged_config, env_overrides)
            
            # Decrypt sensitive data
            merged_config = self.secure_manager.decrypt_sensitive_data(merged_config)
            
            # Create AppConfig object
            try:
                self._config = self._dict_to_appconfig(merged_config)
                
                # Validate configuration
                validation_result = self.validator.validate_config(self._config)
                
                if not validation_result['is_valid']:
                    logger.error(f"Configuration validation failed: {validation_result['errors']}")
                    raise ValueError("Invalid configuration")
                
                if validation_result['warnings']:
                    for warning in validation_result['warnings']:
                        logger.warning(f"Configuration warning: {warning}")
                
                logger.info(f"Configuration loaded successfully for {environment.value}")
                return self._config
                
            except Exception as e:
                logger.error(f"Error creating configuration: {e}")
                raise
    
    def get_config(self) -> Optional[AppConfig]:
        """Get current configuration"""
        return self._config
    
    def update_config(self, updates: Dict[str, Any], persist: bool = True) -> None:
        """Update configuration dynamically"""
        with self._config_lock:
            if not self._config:
                raise RuntimeError("Configuration not loaded")
            
            # Apply updates
            config_dict = asdict(self._config)
            updated_dict = self._deep_merge(config_dict, updates)
            
            # Validate updated configuration
            temp_config = self._dict_to_appconfig(updated_dict)
            validation_result = self.validator.validate_config(temp_config)
            
            if not validation_result['is_valid']:
                raise ValueError(f"Configuration update validation failed: {validation_result['errors']}")
            
            # Apply updates
            self._config = temp_config
            
            # Persist if requested
            if persist and self.env_config_file:
                self._save_config_file(self.env_config_file, updated_dict)
            
            # Notify callbacks
            self._notify_config_changed()
            
            logger.info("Configuration updated successfully")
    
    def watch_config_changes(self, callback: callable) -> None:
        """Register callback for configuration changes"""
        self._callbacks.append(callback)
    
    def create_default_configs(self) -> None:
        """Create default configuration files"""
        
        # Base configuration
        base_config = {
            'environment': 'development',
            'debug': True,
            'version': '2.0.0',
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'gan_cyber_range',
                'username': 'postgres'
            },
            'security': {
                'token_expiry_hours': 24,
                'max_failed_attempts': 5,
                'require_mfa': False
            },
            'performance': {
                'max_concurrent_ranges': 10,
                'worker_threads': 4,
                'cache_enabled': True
            }
        }
        
        # Environment-specific configurations
        environments = {
            'development': {
                'debug': True,
                'logging': {'level': 'DEBUG'}
            },
            'testing': {
                'debug': False,
                'database': {'database': 'test_gan_cyber_range'}
            },
            'production': {
                'debug': False,
                'security': {'require_mfa': True},
                'logging': {'level': 'WARNING'}
            }
        }
        
        # Save configurations
        self._save_config_file(self.base_config_file, base_config)
        
        for env_name, env_config in environments.items():
            env_file = self.config_dir / f"{env_name}.yaml"
            self._save_config_file(env_file, env_config)
        
        # Create secrets template
        secrets_template = {
            'database': {'password': 'your_db_password'},
            'security': {'secret_key': 'your_secret_key'}
        }
        
        if not self.secrets_file.exists():
            self._save_config_file(self.secrets_file, secrets_template)
        
        logger.info("Default configuration files created")
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from file"""
        if not file_path.exists():
            logger.warning(f"Configuration file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    logger.error(f"Unsupported config file format: {file_path}")
                    return {}
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            return {}
    
    def _save_config_file(self, file_path: Path, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif file_path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config file {file_path}: {e}")
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Map of environment variables to config paths
        env_mappings = {
            'GCR_DATABASE_HOST': 'database.host',
            'GCR_DATABASE_PORT': 'database.port',
            'GCR_DATABASE_PASSWORD': 'database.password',
            'GCR_SECRET_KEY': 'security.secret_key',
            'GCR_DEBUG': 'debug',
            'GCR_WORKERS': 'workers',
            'GCR_MAX_RANGES': 'performance.max_concurrent_ranges'
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion
                if config_path.endswith('.port') or config_path == 'workers' or 'max_' in config_path:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif config_path == 'debug':
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                self._set_nested_dict_value(env_config, config_path, value)
        
        return env_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_appconfig(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object"""
        # Handle environment enum
        if 'environment' in config_dict:
            config_dict['environment'] = Environment(config_dict['environment'])
        
        # Create nested config objects
        if 'database' in config_dict:
            config_dict['database'] = DatabaseConfig(**config_dict['database'])
        
        if 'cache' in config_dict:
            config_dict['cache'] = CacheConfig(**config_dict['cache'])
        
        if 'security' in config_dict:
            config_dict['security'] = SecurityConfig(**config_dict['security'])
        
        if 'performance' in config_dict:
            config_dict['performance'] = PerformanceConfig(**config_dict['performance'])
        
        if 'attack' in config_dict:
            config_dict['attack'] = AttackConfig(**config_dict['attack'])
        
        if 'logging' in config_dict:
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
        
        return AppConfig(**config_dict)
    
    def _set_nested_dict_value(self, d: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation"""
        keys = key_path.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    def _notify_config_changed(self) -> None:
        """Notify registered callbacks of configuration changes"""
        for callback in self._callbacks:
            try:
                callback(self._config)
            except Exception as e:
                logger.error(f"Error in config change callback: {e}")


# Global configuration manager
config_manager = ConfigManager()

# Initialize default configurations if they don't exist
if not (config_manager.config_dir / "base.yaml").exists():
    config_manager.create_default_configs()

logger.info("Configuration Management System initialized")