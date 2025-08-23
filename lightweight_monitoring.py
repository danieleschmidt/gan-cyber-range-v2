#!/usr/bin/env python3
"""
Lightweight monitoring system for defensive cybersecurity operations

This provides essential monitoring capabilities without requiring
external dependencies like psutil.
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """System status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"

@dataclass
class LightweightMetrics:
    """Lightweight system metrics without external dependencies"""
    timestamp: datetime
    uptime_seconds: float
    process_id: int
    memory_info: Dict[str, Any]
    disk_info: Dict[str, Any]
    system_load: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

class LightweightMonitor:
    """Lightweight monitoring for defensive systems"""
    
    def __init__(self):
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self.max_history = 100
        self.alerts = []
        
        # Configuration
        self.config = {
            'check_interval': 30,  # seconds
            'log_directory': 'logs',
            'max_alerts': 50
        }
        
        # Ensure log directory exists
        Path(self.config['log_directory']).mkdir(exist_ok=True)
        
    def _collect_lightweight_metrics(self) -> LightweightMetrics:
        """Collect basic system metrics without external dependencies"""
        
        # Basic process information
        process_id = os.getpid()
        
        # Uptime (approximate)
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
        except:
            uptime_seconds = time.time() - getattr(self, '_start_time', time.time())
        
        # Memory information (basic)
        memory_info = {}
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        memory_info[key.strip()] = value.strip()
        except:
            memory_info = {"status": "unavailable"}
        
        # Disk information (basic)
        disk_info = {}
        try:
            stat = os.statvfs('/')
            total_bytes = stat.f_frsize * stat.f_blocks
            free_bytes = stat.f_frsize * stat.f_bavail
            used_bytes = total_bytes - free_bytes
            
            disk_info = {
                'total_gb': round(total_bytes / (1024**3), 2),
                'free_gb': round(free_bytes / (1024**3), 2),
                'used_gb': round(used_bytes / (1024**3), 2),
                'usage_percent': round((used_bytes / total_bytes) * 100, 2)
            }
        except:
            disk_info = {"status": "unavailable"}
        
        # System load (if available)
        system_load = None
        try:
            load_avg = os.getloadavg()[0]  # 1-minute load average
            system_load = load_avg
        except:
            pass
        
        return LightweightMetrics(
            timestamp=datetime.now(),
            uptime_seconds=uptime_seconds,
            process_id=process_id,
            memory_info=memory_info,
            disk_info=disk_info,
            system_load=system_load
        )
    
    def _analyze_metrics(self, metrics: LightweightMetrics) -> SystemStatus:
        """Analyze metrics to determine system status"""
        
        status = SystemStatus.HEALTHY
        
        # Check disk usage
        if isinstance(metrics.disk_info, dict) and 'usage_percent' in metrics.disk_info:
            disk_usage = metrics.disk_info['usage_percent']
            if disk_usage > 95:
                status = SystemStatus.CRITICAL
                self._add_alert(f"Critical disk usage: {disk_usage}%")
            elif disk_usage > 85:
                status = SystemStatus.DEGRADED
                self._add_alert(f"High disk usage: {disk_usage}%")
        
        # Check system load
        if metrics.system_load is not None:
            # Assume 4 CPU cores for threshold calculation
            cpu_cores = 4
            if metrics.system_load > cpu_cores * 2:
                if status == SystemStatus.HEALTHY:
                    status = SystemStatus.CRITICAL
                self._add_alert(f"High system load: {metrics.system_load:.2f}")
            elif metrics.system_load > cpu_cores * 1.5:
                if status == SystemStatus.HEALTHY:
                    status = SystemStatus.DEGRADED
        
        return status
    
    def _add_alert(self, message: str):
        """Add a monitoring alert"""
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        
        # Limit alert history
        if len(self.alerts) > self.config['max_alerts']:
            self.alerts.pop(0)
        
        logger.warning(f"🚨 ALERT: {message}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        logger.info("Lightweight monitoring started")
        self._start_time = time.time()
        
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_lightweight_metrics()
                
                # Analyze system status
                status = self._analyze_metrics(metrics)
                
                # Store metrics
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
                
                logger.debug(f"Metrics collected - PID: {metrics.process_id}, "
                           f"Uptime: {metrics.uptime_seconds:.0f}s")
                
                # Sleep for configured interval
                time.sleep(self.config['check_interval'])
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)  # Short sleep on error
    
    def start_monitoring(self):
        """Start lightweight monitoring"""
        
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Lightweight monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Lightweight monitoring stopped")
    
    def get_current_status(self) -> Dict:
        """Get current system status"""
        
        current_metrics = self._collect_lightweight_metrics()
        status = self._analyze_metrics(current_metrics)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": status.value,
            "metrics": current_metrics.to_dict(),
            "recent_alerts": self.alerts[-5:],  # Last 5 alerts
            "monitoring_active": self.is_monitoring,
            "metrics_collected": len(self.metrics_history)
        }
    
    def perform_health_checks(self) -> Dict:
        """Perform basic health checks"""
        
        checks = {}
        
        # File system check
        try:
            log_dir = Path(self.config['log_directory'])
            test_file = log_dir / 'health_check.tmp'
            
            with open(test_file, 'w') as f:
                f.write('health_check')
            
            test_file.unlink()
            checks['filesystem'] = {'status': 'healthy', 'details': 'Read/write test passed'}
            
        except Exception as e:
            checks['filesystem'] = {'status': 'failing', 'error': str(e)}
        
        # Process check
        try:
            pid = os.getpid()
            checks['process'] = {
                'status': 'healthy',
                'details': f'Process {pid} running normally'
            }
        except Exception as e:
            checks['process'] = {'status': 'failing', 'error': str(e)}
        
        # Memory check (basic)
        try:
            current_metrics = self._collect_lightweight_metrics()
            if 'MemTotal' in current_metrics.memory_info and 'MemFree' in current_metrics.memory_info:
                checks['memory'] = {
                    'status': 'healthy',
                    'details': 'Memory information accessible'
                }
            else:
                checks['memory'] = {
                    'status': 'degraded', 
                    'details': 'Limited memory information available'
                }
        except Exception as e:
            checks['memory'] = {'status': 'failing', 'error': str(e)}
        
        # Overall status
        failing_checks = sum(1 for check in checks.values() if check['status'] == 'failing')
        overall_status = 'failing' if failing_checks > 0 else 'healthy'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'individual_checks': checks,
            'summary': f"{len(checks) - failing_checks}/{len(checks)} checks passed"
        }
    
    def export_data(self, filepath: str):
        """Export monitoring data"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'metrics_history': [m.to_dict() for m in self.metrics_history],
            'alerts_history': self.alerts,
            'current_status': self.get_current_status(),
            'health_checks': self.perform_health_checks()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Monitoring data exported to: {filepath}")

class DefensiveValidationFramework:
    """Framework for validating defensive security configurations"""
    
    def __init__(self):
        self.validation_rules = {}
        self.validation_results = []
        
        # Register default validation rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default validation rules for defensive systems"""
        
        self.register_rule("logs_directory", self._validate_logs_directory)
        self.register_rule("configuration_files", self._validate_configuration)
        self.register_rule("defensive_mode", self._validate_defensive_mode)
        self.register_rule("security_permissions", self._validate_permissions)
        
    def register_rule(self, rule_name: str, validation_function):
        """Register a validation rule"""
        
        self.validation_rules[rule_name] = validation_function
        logger.info(f"Registered validation rule: {rule_name}")
    
    def _validate_logs_directory(self) -> Dict:
        """Validate logs directory exists and is writable"""
        
        try:
            logs_dir = Path("logs")
            
            if not logs_dir.exists():
                logs_dir.mkdir(parents=True)
            
            # Test write permissions
            test_file = logs_dir / "validation_test.tmp"
            with open(test_file, 'w') as f:
                f.write("validation_test")
            test_file.unlink()
            
            return {
                'status': 'passed',
                'details': 'Logs directory exists and is writable'
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'details': 'Failed to validate logs directory'
            }
    
    def _validate_configuration(self) -> Dict:
        """Validate configuration files"""
        
        try:
            config_paths = [
                "configs/defensive",
                "configs/defensive/config.json"
            ]
            
            issues = []
            for path in config_paths:
                path_obj = Path(path)
                if not path_obj.exists():
                    if path.endswith('.json'):
                        # Create default config file
                        path_obj.parent.mkdir(parents=True, exist_ok=True)
                        default_config = {
                            "defensive_mode": True,
                            "security_validation": True,
                            "monitoring_enabled": True
                        }
                        with open(path_obj, 'w') as f:
                            json.dump(default_config, f, indent=2)
                        issues.append(f"Created default config: {path}")
                    else:
                        # Create directory
                        path_obj.mkdir(parents=True, exist_ok=True)
                        issues.append(f"Created directory: {path}")
            
            return {
                'status': 'passed',
                'details': f"Configuration validated. Actions taken: {len(issues)}",
                'actions': issues
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'details': 'Failed to validate configuration'
            }
    
    def _validate_defensive_mode(self) -> Dict:
        """Validate defensive mode is enabled"""
        
        try:
            config_file = Path("configs/defensive/config.json")
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                if config.get('defensive_mode', False):
                    return {
                        'status': 'passed',
                        'details': 'Defensive mode is properly configured'
                    }
                else:
                    return {
                        'status': 'failed',
                        'details': 'Defensive mode is not enabled in configuration'
                    }
            else:
                return {
                    'status': 'failed',
                    'details': 'Configuration file does not exist'
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'details': 'Failed to validate defensive mode'
            }
    
    def _validate_permissions(self) -> Dict:
        """Validate file permissions for security"""
        
        try:
            sensitive_dirs = ["configs", "logs", "data"]
            issues = []
            
            for dir_name in sensitive_dirs:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    # Check if directory is readable/writable by owner
                    if not os.access(dir_path, os.R_OK | os.W_OK):
                        issues.append(f"Insufficient permissions on {dir_name}")
            
            if issues:
                return {
                    'status': 'failed',
                    'details': 'Permission issues found',
                    'issues': issues
                }
            else:
                return {
                    'status': 'passed',
                    'details': 'File permissions validated'
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'details': 'Failed to validate permissions'
            }
    
    def run_all_validations(self) -> Dict:
        """Run all validation rules"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'validations': {},
            'overall_status': 'passed',
            'summary': {}
        }
        
        passed = 0
        failed = 0
        
        for rule_name, validation_func in self.validation_rules.items():
            try:
                result = validation_func()
                results['validations'][rule_name] = result
                
                if result['status'] == 'passed':
                    passed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                results['validations'][rule_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'details': f'Validation rule execution failed'
                }
                failed += 1
        
        results['overall_status'] = 'failed' if failed > 0 else 'passed'
        results['summary'] = {
            'total_validations': passed + failed,
            'passed': passed,
            'failed': failed,
            'success_rate': round((passed / max(passed + failed, 1)) * 100, 1)
        }
        
        self.validation_results.append(results)
        return results

def main():
    """Main demonstration of lightweight monitoring and validation"""
    
    print("🛡️  Lightweight Defensive Monitoring & Validation")
    print("=" * 55)
    
    # Initialize monitoring
    monitor = LightweightMonitor()
    
    # Get initial status
    initial_status = monitor.get_current_status()
    
    print("\n📊 INITIAL SYSTEM STATUS")
    print("-" * 30)
    print(f"System Status: {initial_status['status']}")
    print(f"Process ID: {initial_status['metrics']['process_id']}")
    print(f"Uptime: {initial_status['metrics']['uptime_seconds']:.0f} seconds")
    
    if 'usage_percent' in initial_status['metrics']['disk_info']:
        disk_info = initial_status['metrics']['disk_info']
        print(f"Disk Usage: {disk_info['usage_percent']}% ({disk_info['free_gb']} GB free)")
    
    if initial_status['metrics']['system_load'] is not None:
        print(f"System Load: {initial_status['metrics']['system_load']:.2f}")
    
    # Run health checks
    print(f"\n🏥 HEALTH CHECKS")
    print("-" * 30)
    health_results = monitor.perform_health_checks()
    print(f"Overall Status: {health_results['overall_status']}")
    print(f"Summary: {health_results['summary']}")
    
    for check_name, result in health_results['individual_checks'].items():
        status_emoji = {'healthy': '✅', 'degraded': '⚠️', 'failing': '❌'}
        emoji = status_emoji.get(result['status'], '🔍')
        print(f"  {emoji} {check_name}: {result.get('details', result['status'])}")
    
    # Run defensive validations
    print(f"\n🔒 DEFENSIVE VALIDATIONS")
    print("-" * 30)
    validator = DefensiveValidationFramework()
    validation_results = validator.run_all_validations()
    
    print(f"Overall Validation: {validation_results['overall_status']}")
    print(f"Success Rate: {validation_results['summary']['success_rate']}%")
    
    for rule_name, result in validation_results['validations'].items():
        status_emoji = {'passed': '✅', 'failed': '❌'}
        emoji = status_emoji.get(result['status'], '🔍')
        print(f"  {emoji} {rule_name}: {result.get('details', result['status'])}")
    
    # Start monitoring briefly
    print(f"\n🔄 Starting 15-second monitoring demonstration...")
    monitor.start_monitoring()
    
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    
    monitor.stop_monitoring()
    
    # Final status
    final_status = monitor.get_current_status()
    print(f"\n📈 MONITORING RESULTS")
    print("-" * 25)
    print(f"Metrics Collected: {final_status['metrics_collected']}")
    print(f"Alerts Generated: {len(final_status['recent_alerts'])}")
    
    # Export data
    export_file = f"logs/lightweight_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    monitor.export_data(export_file)
    
    print(f"\n💾 Data exported to: {export_file}")
    print("✅ Lightweight defensive monitoring completed successfully")

if __name__ == "__main__":
    main()