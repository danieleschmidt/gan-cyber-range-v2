"""
Advanced metrics collection system for comprehensive cyber range monitoring.

This module provides real-time metrics collection, aggregation, and storage
with support for custom metrics, alerting, and performance optimization.
"""

import time
import logging
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    unit: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric values"""
    name: str
    metric_type: MetricType
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedMetric:
    """Aggregated metric statistics"""
    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    std: float
    percentiles: Dict[str, float]
    labels: Dict[str, str] = field(default_factory=dict)
    time_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Advanced metrics collection and aggregation system"""
    
    def __init__(
        self,
        collection_interval: float = 1.0,
        max_series_length: int = 1000,
        aggregation_window: int = 300,  # 5 minutes
        enable_system_metrics: bool = True,
        storage_path: Optional[Path] = None
    ):
        self.collection_interval = collection_interval
        self.max_series_length = max_series_length
        self.aggregation_window = aggregation_window
        self.enable_system_metrics = enable_system_metrics
        self.storage_path = storage_path or Path("metrics")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Metric storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.aggregated_metrics: Dict[str, AggregatedMetric] = {}
        self.metric_collectors: Dict[str, Callable] = {}
        
        # Collection state
        self.running = False
        self.collection_thread = None
        self.aggregation_thread = None
        self.metric_queue = queue.Queue()
        
        # Event handlers
        self.metric_handlers: List[Callable[[Metric], None]] = []
        self.threshold_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.thresholds: Dict[str, Dict[str, float]] = {}
        
        # System metrics
        self.system_collectors = {
            'cpu_usage': self._collect_cpu_usage,
            'memory_usage': self._collect_memory_usage,
            'disk_usage': self._collect_disk_usage,
            'network_io': self._collect_network_io,
            'process_metrics': self._collect_process_metrics
        }
        
        # Custom metric tracking
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        logger.info("MetricsCollector initialized")
    
    def start(self) -> None:
        """Start metrics collection"""
        
        if self.running:
            logger.warning("Metrics collection already running")
            return
        
        self.running = True
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        
        # Start aggregation thread
        self.aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True
        )
        self.aggregation_thread.start()
        
        # Start metric processing thread
        processing_thread = threading.Thread(
            target=self._process_metrics_loop,
            daemon=True
        )
        processing_thread.start()
        
        logger.info("Metrics collection started")
    
    def stop(self) -> None:
        """Stop metrics collection"""
        
        self.running = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5.0)
        
        # Save final metrics
        self._save_metrics()
        
        logger.info("Metrics collection stopped")
    
    def register_metric_collector(
        self,
        name: str,
        collector_func: Callable[[], Union[Metric, List[Metric]]]
    ) -> None:
        """Register a custom metric collector function"""
        
        self.metric_collectors[name] = collector_func
        logger.info(f"Registered metric collector: {name}")
    
    def add_metric_handler(self, handler: Callable[[Metric], None]) -> None:
        """Add a metric event handler"""
        
        self.metric_handlers.append(handler)
        logger.info("Added metric handler")
    
    def set_threshold(
        self,
        metric_name: str,
        threshold_type: str,
        value: float,
        handler: Optional[Callable] = None
    ) -> None:
        """Set threshold for metric alerting"""
        
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        
        self.thresholds[metric_name][threshold_type] = value
        
        if handler:
            self.threshold_handlers[f"{metric_name}_{threshold_type}"].append(handler)
        
        logger.info(f"Set {threshold_type} threshold for {metric_name}: {value}")
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """Record a single metric value"""
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
            unit=unit,
            description=description
        )
        
        self.metric_queue.put(metric)
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric"""
        
        key = self._get_metric_key(name, labels)
        self.counters[key] += value
        
        self.record_metric(
            name, self.counters[key], MetricType.COUNTER, labels
        )
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value"""
        
        key = self._get_metric_key(name, labels)
        self.gauges[key] = value
        
        self.record_metric(
            name, value, MetricType.GAUGE, labels
        )
    
    def record_timer(
        self,
        name: str,
        duration: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timing measurement"""
        
        key = self._get_metric_key(name, labels)
        self.timers[key].append(duration)
        
        # Keep only recent values
        if len(self.timers[key]) > 100:
            self.timers[key] = self.timers[key][-100:]
        
        self.record_metric(
            name, duration, MetricType.TIMER, labels, unit="seconds"
        )
    
    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram value"""
        
        key = self._get_metric_key(name, labels)
        self.histograms[key].append(value)
        
        # Keep only recent values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        
        self.record_metric(
            name, value, MetricType.HISTOGRAM, labels
        )
    
    def get_metric_series(self, name: str) -> Optional[MetricSeries]:
        """Get metric time series"""
        
        return self.metrics.get(name)
    
    def get_aggregated_metric(self, name: str) -> Optional[AggregatedMetric]:
        """Get aggregated metric statistics"""
        
        return self.aggregated_metrics.get(name)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        
        current_metrics = {}
        
        # Counters
        for key, value in self.counters.items():
            current_metrics[f"counter_{key}"] = value
        
        # Gauges
        for key, value in self.gauges.items():
            current_metrics[f"gauge_{key}"] = value
        
        # Timer summaries
        for key, values in self.timers.items():
            if values:
                current_metrics[f"timer_{key}_avg"] = np.mean(values)
                current_metrics[f"timer_{key}_p95"] = np.percentile(values, 95)
        
        # Histogram summaries
        for key, values in self.histograms.items():
            if values:
                current_metrics[f"histogram_{key}_avg"] = np.mean(values)
                current_metrics[f"histogram_{key}_p99"] = np.percentile(values, 99)
        
        return current_metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        
        summary = {
            'collection_status': 'running' if self.running else 'stopped',
            'total_series': len(self.metrics),
            'total_aggregated': len(self.aggregated_metrics),
            'collection_interval': self.collection_interval,
            'aggregation_window': self.aggregation_window,
            'metric_counts': {
                'counters': len(self.counters),
                'gauges': len(self.gauges),
                'timers': len(self.timers),
                'histograms': len(self.histograms)
            },
            'recent_metrics': self._get_recent_metric_stats()
        }
        
        return summary
    
    def export_metrics(
        self,
        format_type: str = "prometheus",
        time_range: Optional[timedelta] = None
    ) -> str:
        """Export metrics in specified format"""
        
        if format_type == "prometheus":
            return self._export_prometheus_format(time_range)
        elif format_type == "json":
            return self._export_json_format(time_range)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def timer_context(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        
        class TimerContext:
            def __init__(self, collector, metric_name, metric_labels):
                self.collector = collector
                self.metric_name = metric_name
                self.metric_labels = metric_labels
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.collector.record_timer(
                    self.metric_name, duration, self.metric_labels
                )
        
        return TimerContext(self, name, labels)
    
    def _collection_loop(self) -> None:
        """Main collection loop"""
        
        logger.info("Started metrics collection loop")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Collect system metrics
                if self.enable_system_metrics:
                    self._collect_system_metrics()
                
                # Collect custom metrics
                self._collect_custom_metrics()
                
                # Wait for next collection interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.collection_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.collection_interval)
    
    def _aggregation_loop(self) -> None:
        """Aggregation loop for metric statistics"""
        
        logger.info("Started metrics aggregation loop")
        
        while self.running:
            try:
                self._aggregate_metrics()
                time.sleep(self.aggregation_window)
                
            except Exception as e:
                logger.error(f"Error in metrics aggregation loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _process_metrics_loop(self) -> None:
        """Process queued metrics"""
        
        logger.info("Started metrics processing loop")
        
        while self.running:
            try:
                # Process metrics from queue
                processed_count = 0
                while not self.metric_queue.empty() and processed_count < 100:
                    try:
                        metric = self.metric_queue.get_nowait()
                        self._process_metric(metric)
                        processed_count += 1
                    except queue.Empty:
                        break
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in metrics processing loop: {e}")
                time.sleep(1)
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics"""
        
        for name, collector in self.system_collectors.items():
            try:
                metrics = collector()
                if isinstance(metrics, list):
                    for metric in metrics:
                        self.metric_queue.put(metric)
                else:
                    self.metric_queue.put(metrics)
            except Exception as e:
                logger.error(f"Error collecting {name}: {e}")
    
    def _collect_custom_metrics(self) -> None:
        """Collect custom registered metrics"""
        
        for name, collector in self.metric_collectors.items():
            try:
                result = collector()
                if isinstance(result, list):
                    for metric in result:
                        self.metric_queue.put(metric)
                else:
                    self.metric_queue.put(result)
            except Exception as e:
                logger.error(f"Error collecting custom metric {name}: {e}")
    
    def _process_metric(self, metric: Metric) -> None:
        """Process a single metric"""
        
        # Store in time series
        series_key = self._get_metric_key(metric.name, metric.labels)
        
        if series_key not in self.metrics:
            self.metrics[series_key] = MetricSeries(
                name=metric.name,
                metric_type=metric.metric_type,
                labels=metric.labels,
                unit=metric.unit,
                description=metric.description
            )
        
        series = self.metrics[series_key]
        series.values.append((metric.timestamp, metric.value))
        series.last_updated = metric.timestamp
        
        # Check thresholds
        self._check_thresholds(metric)
        
        # Trigger handlers
        for handler in self.metric_handlers:
            try:
                handler(metric)
            except Exception as e:
                logger.error(f"Error in metric handler: {e}")
    
    def _aggregate_metrics(self) -> None:
        """Aggregate metrics over time windows"""
        
        now = datetime.now()
        window_start = now - timedelta(seconds=self.aggregation_window)
        
        for series_key, series in self.metrics.items():
            # Filter values in time window
            window_values = [
                value for timestamp, value in series.values
                if timestamp >= window_start
            ]
            
            if window_values:
                # Calculate aggregated statistics
                aggregated = AggregatedMetric(
                    name=series.name,
                    count=len(window_values),
                    sum=sum(window_values),
                    min=min(window_values),
                    max=max(window_values),
                    mean=np.mean(window_values),
                    std=np.std(window_values),
                    percentiles={
                        'p50': np.percentile(window_values, 50),
                        'p90': np.percentile(window_values, 90),
                        'p95': np.percentile(window_values, 95),
                        'p99': np.percentile(window_values, 99)
                    },
                    labels=series.labels,
                    time_window=timedelta(seconds=self.aggregation_window),
                    timestamp=now
                )
                
                self.aggregated_metrics[series_key] = aggregated
    
    def _check_thresholds(self, metric: Metric) -> None:
        """Check metric against configured thresholds"""
        
        if metric.name in self.thresholds:
            thresholds = self.thresholds[metric.name]
            
            for threshold_type, threshold_value in thresholds.items():
                triggered = False
                
                if threshold_type == "max" and metric.value > threshold_value:
                    triggered = True
                elif threshold_type == "min" and metric.value < threshold_value:
                    triggered = True
                elif threshold_type == "critical" and metric.value > threshold_value:
                    triggered = True
                
                if triggered:
                    handler_key = f"{metric.name}_{threshold_type}"
                    for handler in self.threshold_handlers[handler_key]:
                        try:
                            handler(metric, threshold_type, threshold_value)
                        except Exception as e:
                            logger.error(f"Error in threshold handler: {e}")
    
    def _collect_cpu_usage(self) -> Metric:
        """Collect CPU usage metrics"""
        
        cpu_percent = psutil.cpu_percent(interval=None)
        return Metric(
            name="system_cpu_usage",
            value=cpu_percent,
            metric_type=MetricType.GAUGE,
            unit="percent",
            description="System CPU usage percentage"
        )
    
    def _collect_memory_usage(self) -> List[Metric]:
        """Collect memory usage metrics"""
        
        memory = psutil.virtual_memory()
        
        return [
            Metric(
                name="system_memory_usage",
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                unit="percent",
                description="System memory usage percentage"
            ),
            Metric(
                name="system_memory_available",
                value=memory.available,
                metric_type=MetricType.GAUGE,
                unit="bytes",
                description="Available system memory"
            ),
            Metric(
                name="system_memory_total",
                value=memory.total,
                metric_type=MetricType.GAUGE,
                unit="bytes",
                description="Total system memory"
            )
        ]
    
    def _collect_disk_usage(self) -> List[Metric]:
        """Collect disk usage metrics"""
        
        disk_usage = psutil.disk_usage('/')
        
        return [
            Metric(
                name="system_disk_usage",
                value=(disk_usage.used / disk_usage.total) * 100,
                metric_type=MetricType.GAUGE,
                unit="percent",
                description="System disk usage percentage"
            ),
            Metric(
                name="system_disk_free",
                value=disk_usage.free,
                metric_type=MetricType.GAUGE,
                unit="bytes",
                description="Free disk space"
            )
        ]
    
    def _collect_network_io(self) -> List[Metric]:
        """Collect network I/O metrics"""
        
        net_io = psutil.net_io_counters()
        
        return [
            Metric(
                name="system_network_bytes_sent",
                value=net_io.bytes_sent,
                metric_type=MetricType.COUNTER,
                unit="bytes",
                description="Total bytes sent over network"
            ),
            Metric(
                name="system_network_bytes_recv",
                value=net_io.bytes_recv,
                metric_type=MetricType.COUNTER,
                unit="bytes",
                description="Total bytes received over network"
            )
        ]
    
    def _collect_process_metrics(self) -> List[Metric]:
        """Collect current process metrics"""
        
        process = psutil.Process()
        
        return [
            Metric(
                name="process_cpu_usage",
                value=process.cpu_percent(),
                metric_type=MetricType.GAUGE,
                unit="percent",
                description="Process CPU usage percentage"
            ),
            Metric(
                name="process_memory_usage",
                value=process.memory_percent(),
                metric_type=MetricType.GAUGE,
                unit="percent",
                description="Process memory usage percentage"
            ),
            Metric(
                name="process_thread_count",
                value=process.num_threads(),
                metric_type=MetricType.GAUGE,
                description="Number of process threads"
            )
        ]
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate unique key for metric with labels"""
        
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
    
    def _get_recent_metric_stats(self) -> Dict[str, Any]:
        """Get statistics about recent metrics"""
        
        now = datetime.now()
        last_minute = now - timedelta(minutes=1)
        
        recent_count = 0
        for series in self.metrics.values():
            recent_count += len([
                v for timestamp, v in series.values
                if timestamp >= last_minute
            ])
        
        return {
            'recent_count_1min': recent_count,
            'total_series': len(self.metrics),
            'queue_size': self.metric_queue.qsize()
        }
    
    def _export_prometheus_format(self, time_range: Optional[timedelta]) -> str:
        """Export metrics in Prometheus format"""
        
        lines = []
        
        for series_key, series in self.metrics.items():
            # Get recent values if time_range specified
            values = series.values
            if time_range:
                cutoff = datetime.now() - time_range
                values = deque([
                    (ts, val) for ts, val in values if ts >= cutoff
                ], maxlen=series.values.maxlen)
            
            if values:
                # Use most recent value
                _, latest_value = values[-1]
                
                # Format labels
                label_str = ""
                if series.labels:
                    labels = ",".join(f'{k}="{v}"' for k, v in series.labels.items())
                    label_str = f"{{{labels}}}"
                
                lines.append(f"{series.name}{label_str} {latest_value}")
        
        return "\n".join(lines)
    
    def _export_json_format(self, time_range: Optional[timedelta]) -> str:
        """Export metrics in JSON format"""
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': []
        }
        
        for series_key, series in self.metrics.items():
            # Get values in time range
            values = series.values
            if time_range:
                cutoff = datetime.now() - time_range
                values = [(ts, val) for ts, val in values if ts >= cutoff]
            
            metric_data = {
                'name': series.name,
                'type': series.metric_type.value,
                'labels': series.labels,
                'unit': series.unit,
                'description': series.description,
                'values': [
                    {'timestamp': ts.isoformat(), 'value': val}
                    for ts, val in values
                ]
            }
            export_data['metrics'].append(metric_data)
        
        return json.dumps(export_data, indent=2)
    
    def _save_metrics(self) -> None:
        """Save metrics to storage"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = self.storage_path / f"metrics_{timestamp}.json"
            
            export_data = self._export_json_format(None)
            
            with open(metrics_file, 'w') as f:
                f.write(export_data)
            
            logger.info(f"Metrics saved to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


# Decorator for automatic timing
def timed_metric(metrics_collector: MetricsCollector, metric_name: str):
    """Decorator to automatically time function execution"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            with metrics_collector.timer_context(metric_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Decorator for automatic counting
def counted_metric(metrics_collector: MetricsCollector, metric_name: str):
    """Decorator to automatically count function calls"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            metrics_collector.increment_counter(metric_name)
            return result
        return wrapper
    return decorator