"""
Advanced performance monitoring and profiling capabilities.
"""

import logging
import time
import threading
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
import functools
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileData:
    """Function profiling data"""
    function_name: str
    total_calls: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_called: Optional[datetime] = None
    call_history: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass  
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: datetime


class PerformanceMonitor:
    """Advanced performance monitoring system"""
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 history_size: int = 1000):
        self.collection_interval = collection_interval
        self.history_size = history_size
        
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self._system_metrics: deque[SystemMetrics] = deque(maxlen=history_size)
        self._custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread = None
        
        # Alert thresholds
        self._thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "response_time": 5.0  # seconds
        }
        
        self._alert_callbacks: List[Callable] = []
        
    def start(self):
        """Start performance monitoring"""
        if self._running:
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        logger.info("Performance monitoring started")
        
    def stop(self):
        """Stop performance monitoring"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
            
        logger.info("Performance monitoring stopped")
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a custom metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        with self._lock:
            self._custom_metrics[name].append(metric)
            
    def get_metric_summary(self, name: str, window_minutes: int = 10) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self._lock:
            metrics = self._custom_metrics.get(name, [])
            recent_values = [
                m.value for m in metrics 
                if m.timestamp >= cutoff_time
            ]
            
        if not recent_values:
            return {"count": 0}
            
        return {
            "count": len(recent_values),
            "avg": sum(recent_values) / len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "sum": sum(recent_values)
        }
        
    def get_system_metrics_summary(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Get system metrics summary"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self._lock:
            recent_metrics = [
                m for m in self._system_metrics
                if m.timestamp >= cutoff_time
            ]
            
        if not recent_metrics:
            return {"count": 0}
            
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            "count": len(recent_metrics),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "current": recent_metrics[-1].cpu_percent if recent_metrics else 0
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values), 
                "max": max(memory_values),
                "current": recent_metrics[-1].memory_percent if recent_metrics else 0
            },
            "timestamp": recent_metrics[-1].timestamp if recent_metrics else None
        }
        
    def set_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold for metric"""
        self._thresholds[metric_name] = threshold
        
    def add_alert_callback(self, callback: Callable[[str, float, float], None]):
        """Add alert callback function"""
        self._alert_callbacks.append(callback)
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                self._collect_system_metrics()
                self._check_alerts()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
                
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network
            network = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk_percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                timestamp=datetime.now()
            )
            
            with self._lock:
                self._system_metrics.append(metrics)
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
    def _check_alerts(self):
        """Check for threshold violations and trigger alerts"""
        if not self._system_metrics:
            return
            
        latest_metrics = self._system_metrics[-1]
        
        # Check system metric thresholds
        checks = [
            ("cpu_percent", latest_metrics.cpu_percent),
            ("memory_percent", latest_metrics.memory_percent), 
            ("disk_usage_percent", latest_metrics.disk_usage_percent)
        ]
        
        for metric_name, value in checks:
            threshold = self._thresholds.get(metric_name)
            if threshold and value > threshold:
                self._trigger_alert(metric_name, value, threshold)
                
    def _trigger_alert(self, metric_name: str, value: float, threshold: float):
        """Trigger alert callbacks"""
        for callback in self._alert_callbacks:
            try:
                callback(metric_name, value, threshold)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")


class PerformanceProfiler:
    """Function performance profiler"""
    
    def __init__(self):
        self._profiles: Dict[str, ProfileData] = {}
        self._lock = threading.RLock()
        
    def profile_function(self, func: Optional[Callable] = None, name: Optional[str] = None):
        """Decorator to profile function performance"""
        
        def decorator(f: Callable) -> Callable:
            function_name = name or f.__name__
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                
                try:
                    result = f(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    self._record_execution(function_name, execution_time)
                    
            @functools.wraps(f)  
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                
                try:
                    result = await f(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    self._record_execution(function_name, execution_time)
                    
            if asyncio.iscoroutinefunction(f):
                return async_wrapper
            else:
                return wrapper
                
        if func is None:
            return decorator
        else:
            return decorator(func)
            
    @contextmanager
    def profile_block(self, block_name: str):
        """Context manager to profile code blocks"""
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            self._record_execution(block_name, execution_time)
            
    def _record_execution(self, function_name: str, execution_time: float):
        """Record function execution data"""
        with self._lock:
            if function_name not in self._profiles:
                self._profiles[function_name] = ProfileData(function_name=function_name)
                
            profile = self._profiles[function_name]
            profile.total_calls += 1
            profile.total_time += execution_time
            profile.avg_time = profile.total_time / profile.total_calls
            profile.min_time = min(profile.min_time, execution_time)
            profile.max_time = max(profile.max_time, execution_time)
            profile.last_called = datetime.now()
            profile.call_history.append({
                "timestamp": datetime.now(),
                "duration": execution_time
            })
            
    def get_profile_data(self, function_name: str) -> Optional[ProfileData]:
        """Get profiling data for specific function"""
        with self._lock:
            return self._profiles.get(function_name)
            
    def get_all_profiles(self) -> Dict[str, ProfileData]:
        """Get all profiling data"""
        with self._lock:
            return self._profiles.copy()
            
    def get_top_functions(self, metric: str = "avg_time", limit: int = 10) -> List[ProfileData]:
        """Get top functions by specified metric"""
        with self._lock:
            profiles = list(self._profiles.values())
            
            if metric == "avg_time":
                profiles.sort(key=lambda p: p.avg_time, reverse=True)
            elif metric == "total_time":
                profiles.sort(key=lambda p: p.total_time, reverse=True)
            elif metric == "total_calls":
                profiles.sort(key=lambda p: p.total_calls, reverse=True)
            elif metric == "max_time":
                profiles.sort(key=lambda p: p.max_time, reverse=True)
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            return profiles[:limit]
            
    def reset_profiles(self):
        """Reset all profiling data"""
        with self._lock:
            self._profiles.clear()
            
    def generate_report(self) -> str:
        """Generate performance report"""
        with self._lock:
            if not self._profiles:
                return "No profiling data available"
                
            report_lines = [
                "Performance Profiling Report",
                "=" * 40,
                f"Total Functions Profiled: {len(self._profiles)}",
                ""
            ]
            
            # Top functions by average time
            top_avg = self.get_top_functions("avg_time", 5)
            if top_avg:
                report_lines.extend([
                    "Top Functions by Average Time:",
                    "-" * 30
                ])
                
                for profile in top_avg:
                    report_lines.append(
                        f"{profile.function_name}: {profile.avg_time:.4f}s "
                        f"(calls: {profile.total_calls}, total: {profile.total_time:.4f}s)"
                    )
                    
                report_lines.append("")
                
            # Top functions by total time
            top_total = self.get_top_functions("total_time", 5)
            if top_total:
                report_lines.extend([
                    "Top Functions by Total Time:",
                    "-" * 30
                ])
                
                for profile in top_total:
                    report_lines.append(
                        f"{profile.function_name}: {profile.total_time:.4f}s "
                        f"(calls: {profile.total_calls}, avg: {profile.avg_time:.4f}s)"
                    )
                    
                report_lines.append("")
                
            return "\n".join(report_lines)


class PerformanceAnalyzer:
    """Advanced performance analysis and optimization suggestions"""
    
    def __init__(self, monitor: PerformanceMonitor, profiler: PerformanceProfiler):
        self.monitor = monitor
        self.profiler = profiler
        
    def analyze_performance(self, window_minutes: int = 30) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        
        analysis = {
            "timestamp": datetime.now(),
            "window_minutes": window_minutes,
            "system_analysis": self._analyze_system_performance(window_minutes),
            "function_analysis": self._analyze_function_performance(),
            "bottlenecks": self._identify_bottlenecks(),
            "recommendations": self._generate_recommendations()
        }
        
        return analysis
        
    def _analyze_system_performance(self, window_minutes: int) -> Dict[str, Any]:
        """Analyze system performance metrics"""
        
        system_summary = self.monitor.get_system_metrics_summary(window_minutes)
        
        analysis = {
            "cpu_status": "normal",
            "memory_status": "normal", 
            "performance_score": 100.0,
            "issues": []
        }
        
        if system_summary.get("count", 0) == 0:
            analysis["issues"].append("No system metrics available")
            return analysis
            
        cpu_avg = system_summary["cpu"]["avg"]
        memory_avg = system_summary["memory"]["avg"]
        
        # CPU analysis
        if cpu_avg > 80:
            analysis["cpu_status"] = "critical"
            analysis["issues"].append(f"High CPU usage: {cpu_avg:.1f}%")
            analysis["performance_score"] -= 30
        elif cpu_avg > 60:
            analysis["cpu_status"] = "warning"
            analysis["issues"].append(f"Elevated CPU usage: {cpu_avg:.1f}%")
            analysis["performance_score"] -= 15
            
        # Memory analysis
        if memory_avg > 85:
            analysis["memory_status"] = "critical"
            analysis["issues"].append(f"High memory usage: {memory_avg:.1f}%")
            analysis["performance_score"] -= 25
        elif memory_avg > 70:
            analysis["memory_status"] = "warning"
            analysis["issues"].append(f"Elevated memory usage: {memory_avg:.1f}%")
            analysis["performance_score"] -= 10
            
        return analysis
        
    def _analyze_function_performance(self) -> Dict[str, Any]:
        """Analyze function performance profiles"""
        
        profiles = self.profiler.get_all_profiles()
        
        if not profiles:
            return {"status": "no_data", "functions_profiled": 0}
            
        slow_functions = []
        frequent_functions = []
        
        for name, profile in profiles.items():
            # Identify slow functions
            if profile.avg_time > 1.0:  # Slower than 1 second average
                slow_functions.append({
                    "name": name,
                    "avg_time": profile.avg_time,
                    "total_calls": profile.total_calls
                })
                
            # Identify frequently called functions
            if profile.total_calls > 100:
                frequent_functions.append({
                    "name": name,
                    "total_calls": profile.total_calls,
                    "total_time": profile.total_time
                })
                
        return {
            "status": "analyzed",
            "functions_profiled": len(profiles),
            "slow_functions": sorted(slow_functions, key=lambda x: x["avg_time"], reverse=True)[:10],
            "frequent_functions": sorted(frequent_functions, key=lambda x: x["total_calls"], reverse=True)[:10]
        }
        
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        
        bottlenecks = []
        
        # System bottlenecks
        system_summary = self.monitor.get_system_metrics_summary(10)
        if system_summary.get("cpu", {}).get("max", 0) > 90:
            bottlenecks.append({
                "type": "system",
                "component": "cpu",
                "severity": "high",
                "description": "CPU usage spikes detected"
            })
            
        # Function bottlenecks
        top_functions = self.profiler.get_top_functions("total_time", 5)
        for profile in top_functions:
            if profile.total_time > 10.0:  # More than 10 seconds total
                bottlenecks.append({
                    "type": "function",
                    "component": profile.function_name,
                    "severity": "medium" if profile.total_time < 30 else "high",
                    "description": f"Function consuming {profile.total_time:.2f}s total time"
                })
                
        return bottlenecks
        
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        
        recommendations = []
        
        # System recommendations
        system_summary = self.monitor.get_system_metrics_summary(15)
        if system_summary.get("memory", {}).get("avg", 0) > 70:
            recommendations.append("Consider increasing available memory or optimizing memory usage")
            
        if system_summary.get("cpu", {}).get("avg", 0) > 60:
            recommendations.append("CPU usage is elevated - consider optimizing algorithms or adding parallel processing")
            
        # Function recommendations  
        slow_functions = self.profiler.get_top_functions("avg_time", 3)
        for profile in slow_functions:
            if profile.avg_time > 0.5:
                recommendations.append(f"Optimize function '{profile.function_name}' - average execution time is {profile.avg_time:.3f}s")
                
        # General recommendations
        if len(self.profiler.get_all_profiles()) > 50:
            recommendations.append("Consider implementing function result caching for frequently called functions")
            
        return recommendations