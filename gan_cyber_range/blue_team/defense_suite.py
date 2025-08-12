"""
Comprehensive blue team defense suite with advanced detection and response capabilities.

This module provides integrated defensive tools including SIEM, IDS/IPS, EDR,
and threat intelligence platforms for comprehensive security monitoring.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
import queue
import statistics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionStatus(Enum):
    """Detection status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    INVESTIGATING = "investigating"


@dataclass
class SecurityAlert:
    """Represents a security alert from defensive tools"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    source: str  # SIEM, IDS, EDR, etc.
    rule_name: str
    description: str
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    username: Optional[str] = None
    process_name: Optional[str] = None
    file_hash: Optional[str] = None
    mitre_techniques: List[str] = field(default_factory=list)
    raw_log: Optional[str] = None
    status: DetectionStatus = DetectionStatus.ACTIVE
    confidence: float = 0.8


@dataclass
class DefenseMetrics:
    """Metrics for blue team performance"""
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    mean_time_to_detect: float = 0.0  # seconds
    mean_time_to_respond: float = 0.0  # seconds
    mean_time_to_resolve: float = 0.0  # seconds
    alerts_processed: int = 0
    incidents_created: int = 0
    incidents_resolved: int = 0
    coverage_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'detection_rate': self.detection_rate,
            'false_positive_rate': self.false_positive_rate,
            'mttd': self.mean_time_to_detect,
            'mttr': self.mean_time_to_respond,
            'mttr_resolve': self.mean_time_to_resolve,
            'alerts_processed': self.alerts_processed,
            'incidents_created': self.incidents_created,
            'incidents_resolved': self.incidents_resolved,
            'coverage_score': self.coverage_score
        }


class SIEMEngine:
    """Security Information and Event Management engine"""
    
    def __init__(self, name: str = "SIEM"):
        self.name = name
        self.is_running = False
        self.rules = {}
        self.alert_queue = queue.Queue()
        self.correlation_rules = []
        self.baseline_behaviors = {}
        
        # Load default rules
        self._load_default_rules()
        
    def start(self) -> None:
        """Start SIEM processing"""
        self.is_running = True
        logger.info(f"{self.name} started")
        
    def stop(self) -> None:
        """Stop SIEM processing"""
        self.is_running = False
        logger.info(f"{self.name} stopped")
        
    def process_log(self, log_entry: Dict[str, Any]) -> List[SecurityAlert]:
        """Process log entry and generate alerts"""
        if not self.is_running:
            return []
            
        alerts = []
        
        # Check against detection rules
        for rule_id, rule in self.rules.items():
            if self._matches_rule(log_entry, rule):
                alert = self._create_alert_from_rule(log_entry, rule)
                alerts.append(alert)
                self.alert_queue.put(alert)
                
        return alerts
        
    def add_rule(self, rule_id: str, rule: Dict[str, Any]) -> None:
        """Add detection rule"""
        self.rules[rule_id] = rule
        logger.info(f"Added SIEM rule: {rule_id}")
        
    def get_alerts(self) -> List[SecurityAlert]:
        """Get pending alerts"""
        alerts = []
        while not self.alert_queue.empty():
            try:
                alerts.append(self.alert_queue.get_nowait())
            except queue.Empty:
                break
        return alerts
        
    def _load_default_rules(self) -> None:
        """Load default detection rules"""
        
        default_rules = {
            'failed_login_attempts': {
                'name': 'Multiple Failed Login Attempts',
                'conditions': {
                    'event_type': 'authentication',
                    'result': 'failure',
                    'threshold': 5,
                    'timeframe': 300  # 5 minutes
                },
                'severity': AlertSeverity.MEDIUM,
                'mitre_techniques': ['T1110']
            },
            
            'privilege_escalation': {
                'name': 'Privilege Escalation Detected',
                'conditions': {
                    'event_type': 'process_creation',
                    'elevated_privileges': True,
                    'suspicious_process': True
                },
                'severity': AlertSeverity.HIGH,
                'mitre_techniques': ['T1548']
            },
            
            'lateral_movement': {
                'name': 'Lateral Movement Activity',
                'conditions': {
                    'event_type': 'network_connection',
                    'internal_to_internal': True,
                    'unusual_protocol': True
                },
                'severity': AlertSeverity.HIGH,
                'mitre_techniques': ['T1021']
            },
            
            'data_exfiltration': {
                'name': 'Suspicious Data Transfer',
                'conditions': {
                    'event_type': 'network_traffic',
                    'bytes_out': {'gt': 100000000},  # > 100MB
                    'external_destination': True
                },
                'severity': AlertSeverity.CRITICAL,
                'mitre_techniques': ['T1041']
            }
        }
        
        for rule_id, rule in default_rules.items():
            self.add_rule(rule_id, rule)
            
    def _matches_rule(self, log_entry: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Check if log entry matches detection rule"""
        
        conditions = rule.get('conditions', {})
        
        # Simple condition matching
        for key, expected_value in conditions.items():
            if key == 'threshold' or key == 'timeframe':
                continue
                
            log_value = log_entry.get(key)
            
            if isinstance(expected_value, dict):
                # Handle comparison operators
                if 'gt' in expected_value:
                    if not (log_value and log_value > expected_value['gt']):
                        return False
                elif 'lt' in expected_value:
                    if not (log_value and log_value < expected_value['lt']):
                        return False
                elif 'eq' in expected_value:
                    if log_value != expected_value['eq']:
                        return False
            else:
                # Direct comparison
                if log_value != expected_value:
                    return False
                    
        return True
        
    def _create_alert_from_rule(self, log_entry: Dict[str, Any], rule: Dict[str, Any]) -> SecurityAlert:
        """Create security alert from rule match"""
        
        alert_id = f"siem_{int(time.time())}_{len(self.rules)}"
        
        return SecurityAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=rule.get('severity', AlertSeverity.MEDIUM),
            source=self.name,
            rule_name=rule.get('name', 'Unknown Rule'),
            description=f"Rule triggered: {rule.get('name', 'Unknown')}",
            source_ip=log_entry.get('source_ip'),
            destination_ip=log_entry.get('destination_ip'),
            username=log_entry.get('username'),
            process_name=log_entry.get('process_name'),
            file_hash=log_entry.get('file_hash'),
            mitre_techniques=rule.get('mitre_techniques', []),
            raw_log=json.dumps(log_entry),
            confidence=rule.get('confidence', 0.8)
        )


class IDSEngine:
    """Intrusion Detection System engine"""
    
    def __init__(self, name: str = "IDS"):
        self.name = name
        self.is_running = False
        self.signatures = {}
        self.anomaly_baselines = {}
        
        # Load signatures
        self._load_default_signatures()
        
    def start(self) -> None:
        """Start IDS processing"""
        self.is_running = True
        logger.info(f"{self.name} started")
        
    def stop(self) -> None:
        """Stop IDS processing"""
        self.is_running = False
        logger.info(f"{self.name} stopped")
        
    def analyze_traffic(self, packet_data: Dict[str, Any]) -> List[SecurityAlert]:
        """Analyze network traffic for threats"""
        if not self.is_running:
            return []
            
        alerts = []
        
        # Signature-based detection
        for sig_id, signature in self.signatures.items():
            if self._matches_signature(packet_data, signature):
                alert = self._create_ids_alert(packet_data, signature, 'signature')
                alerts.append(alert)
                
        # Anomaly-based detection
        anomaly_alert = self._check_anomalies(packet_data)
        if anomaly_alert:
            alerts.append(anomaly_alert)
            
        return alerts
        
    def _load_default_signatures(self) -> None:
        """Load default network signatures"""
        
        signatures = {
            'port_scan': {
                'name': 'Port Scan Detection',
                'pattern': {
                    'tcp_flags': 'SYN',
                    'destination_ports': {'count_gt': 10},
                    'timeframe': 60
                },
                'severity': AlertSeverity.MEDIUM,
                'mitre_techniques': ['T1046']
            },
            
            'sql_injection': {
                'name': 'SQL Injection Attempt',
                'pattern': {
                    'http_payload': ['UNION SELECT', 'DROP TABLE', "' OR 1=1"],
                    'protocol': 'HTTP'
                },
                'severity': AlertSeverity.HIGH,
                'mitre_techniques': ['T1190']
            },
            
            'malware_c2': {
                'name': 'Malware C2 Communication',
                'pattern': {
                    'destination_port': [8080, 9999, 1337],
                    'encrypted_payload': True,
                    'periodic_beaconing': True
                },
                'severity': AlertSeverity.CRITICAL,
                'mitre_techniques': ['T1071']
            }
        }
        
        self.signatures.update(signatures)
        
    def _matches_signature(self, packet_data: Dict[str, Any], signature: Dict[str, Any]) -> bool:
        """Check if packet matches signature"""
        
        pattern = signature.get('pattern', {})
        
        for key, expected in pattern.items():
            packet_value = packet_data.get(key)
            
            if isinstance(expected, list):
                if not any(exp in str(packet_value) for exp in expected):
                    return False
            elif isinstance(expected, dict):
                if 'count_gt' in expected:
                    # Handle count-based patterns
                    continue
            else:
                if packet_value != expected:
                    return False
                    
        return True
        
    def _check_anomalies(self, packet_data: Dict[str, Any]) -> Optional[SecurityAlert]:
        """Check for anomalous behavior"""
        
        # Simple anomaly detection based on traffic volume
        src_ip = packet_data.get('source_ip')
        bytes_transferred = packet_data.get('bytes', 0)
        
        if src_ip and bytes_transferred > 10000000:  # > 10MB
            return SecurityAlert(
                alert_id=f"ids_anomaly_{int(time.time())}",
                timestamp=datetime.now(),
                severity=AlertSeverity.MEDIUM,
                source=self.name,
                rule_name='Anomalous Traffic Volume',
                description=f'Unusual traffic volume from {src_ip}: {bytes_transferred} bytes',
                source_ip=src_ip,
                confidence=0.6
            )
            
        return None
        
    def _create_ids_alert(
        self, 
        packet_data: Dict[str, Any], 
        signature: Dict[str, Any], 
        detection_type: str
    ) -> SecurityAlert:
        """Create IDS alert"""
        
        alert_id = f"ids_{detection_type}_{int(time.time())}"
        
        return SecurityAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=signature.get('severity', AlertSeverity.MEDIUM),
            source=self.name,
            rule_name=signature.get('name', 'Unknown Signature'),
            description=f"IDS {detection_type} detection: {signature.get('name')}",
            source_ip=packet_data.get('source_ip'),
            destination_ip=packet_data.get('destination_ip'),
            mitre_techniques=signature.get('mitre_techniques', []),
            raw_log=json.dumps(packet_data),
            confidence=signature.get('confidence', 0.8)
        )


class EDREngine:
    """Endpoint Detection and Response engine"""
    
    def __init__(self, name: str = "EDR"):
        self.name = name
        self.is_running = False
        self.behavioral_rules = {}
        self.process_monitoring = True
        self.file_monitoring = True
        
        # Load behavioral rules
        self._load_behavioral_rules()
        
    def start(self) -> None:
        """Start EDR monitoring"""
        self.is_running = True
        logger.info(f"{self.name} started")
        
    def stop(self) -> None:
        """Stop EDR monitoring"""
        self.is_running = False
        logger.info(f"{self.name} stopped")
        
    def analyze_endpoint_event(self, event_data: Dict[str, Any]) -> List[SecurityAlert]:
        """Analyze endpoint events for threats"""
        if not self.is_running:
            return []
            
        alerts = []
        
        # Behavioral analysis
        for rule_id, rule in self.behavioral_rules.items():
            if self._matches_behavioral_rule(event_data, rule):
                alert = self._create_edr_alert(event_data, rule)
                alerts.append(alert)
                
        return alerts
        
    def _load_behavioral_rules(self) -> None:
        """Load behavioral detection rules"""
        
        rules = {
            'process_injection': {
                'name': 'Process Injection Detected',
                'conditions': {
                    'event_type': 'process_access',
                    'access_type': 'write',
                    'target_process_different': True
                },
                'severity': AlertSeverity.HIGH,
                'mitre_techniques': ['T1055']
            },
            
            'credential_dumping': {
                'name': 'Credential Dumping Activity',
                'conditions': {
                    'event_type': 'memory_access',
                    'target_process': 'lsass.exe',
                    'access_type': 'read'
                },
                'severity': AlertSeverity.CRITICAL,
                'mitre_techniques': ['T1003']
            },
            
            'persistence_mechanism': {
                'name': 'Persistence Mechanism Created',
                'conditions': {
                    'event_type': 'registry_modification',
                    'registry_key': ['Run', 'RunOnce', 'Services'],
                    'executable_path': True
                },
                'severity': AlertSeverity.HIGH,
                'mitre_techniques': ['T1547']
            },
            
            'suspicious_powershell': {
                'name': 'Suspicious PowerShell Activity',
                'conditions': {
                    'event_type': 'process_creation',
                    'process_name': 'powershell.exe',
                    'command_line_contains': ['Invoke-Expression', 'DownloadString', 'EncodedCommand']
                },
                'severity': AlertSeverity.HIGH,
                'mitre_techniques': ['T1059.001']
            }
        }
        
        self.behavioral_rules.update(rules)
        
    def _matches_behavioral_rule(self, event_data: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Check if event matches behavioral rule"""
        
        conditions = rule.get('conditions', {})
        
        for key, expected in conditions.items():
            event_value = event_data.get(key)
            
            if isinstance(expected, list):
                if not any(exp in str(event_value) for exp in expected):
                    return False
            elif isinstance(expected, bool):
                if bool(event_value) != expected:
                    return False
            else:
                if event_value != expected:
                    return False
                    
        return True
        
    def _create_edr_alert(self, event_data: Dict[str, Any], rule: Dict[str, Any]) -> SecurityAlert:
        """Create EDR alert"""
        
        alert_id = f"edr_{int(time.time())}_{rule.get('name', 'unknown').replace(' ', '_')}"
        
        return SecurityAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=rule.get('severity', AlertSeverity.MEDIUM),
            source=self.name,
            rule_name=rule.get('name', 'Unknown Rule'),
            description=f"EDR behavioral detection: {rule.get('name')}",
            username=event_data.get('username'),
            process_name=event_data.get('process_name'),
            file_hash=event_data.get('file_hash'),
            mitre_techniques=rule.get('mitre_techniques', []),
            raw_log=json.dumps(event_data),
            confidence=rule.get('confidence', 0.85)
        )


class DefenseSuite:
    """Integrated defense suite managing multiple security tools"""
    
    def __init__(self, cyber_range):
        self.cyber_range = cyber_range
        
        # Initialize defensive engines
        self.siem = SIEMEngine("SIEM")
        self.ids = IDSEngine("Suricata-IDS")
        self.edr = EDREngine("CrowdStrike-EDR")
        
        # Alert management
        self.all_alerts = []
        self.alert_processors = []
        self.alert_correlation_rules = []
        
        # Metrics tracking
        self.metrics = DefenseMetrics()
        self.detection_times = []
        self.response_times = []
        
        # Threading for background processing
        self.processing_thread = None
        self.is_running = False
        
        logger.info("Defense suite initialized")
        
    def deploy_defenses(self, config: Dict[str, Any]) -> None:
        """Deploy defensive tools based on configuration"""
        
        logger.info("Deploying defensive tools")
        
        # Configure and start SIEM
        if config.get('siem', True):
            self.siem.start()
            
        # Configure and start IDS
        if config.get('ids', True):
            self.ids.start()
            
        # Configure and start EDR
        if config.get('edr', True):
            self.edr.start()
            
        # Start background processing
        self.start_processing()
        
        logger.info("Defense deployment completed")
        
    def start_processing(self) -> None:
        """Start background alert processing"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.processing_thread.start()
        
    def stop_processing(self) -> None:
        """Stop background processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
            
        # Stop all engines
        self.siem.stop()
        self.ids.stop()
        self.edr.stop()
        
    def process_event(self, event_data: Dict[str, Any]) -> List[SecurityAlert]:
        """Process security event through all defensive tools"""
        
        all_alerts = []
        
        # Route to appropriate engines based on event type
        event_type = event_data.get('event_type', 'unknown')
        
        if event_type in ['authentication', 'process_creation', 'file_access']:
            # SIEM processing
            siem_alerts = self.siem.process_log(event_data)
            all_alerts.extend(siem_alerts)
            
        if event_type in ['network_traffic', 'network_connection']:
            # IDS processing
            ids_alerts = self.ids.analyze_traffic(event_data)
            all_alerts.extend(ids_alerts)
            
        if event_type in ['process_creation', 'memory_access', 'registry_modification']:
            # EDR processing
            edr_alerts = self.edr.analyze_endpoint_event(event_data)
            all_alerts.extend(edr_alerts)
            
        # Store alerts
        self.all_alerts.extend(all_alerts)
        
        # Update metrics
        self._update_metrics(all_alerts)
        
        return all_alerts
        
    def get_alerts(
        self, 
        severity: Optional[AlertSeverity] = None,
        time_range: Optional[timedelta] = None
    ) -> List[SecurityAlert]:
        """Get alerts with optional filtering"""
        
        filtered_alerts = self.all_alerts.copy()
        
        # Filter by severity
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
            
        # Filter by time range
        if time_range:
            cutoff_time = datetime.now() - time_range
            filtered_alerts = [a for a in filtered_alerts if a.timestamp >= cutoff_time]
            
        return filtered_alerts
        
    def get_metrics(self) -> DefenseMetrics:
        """Get current defense metrics"""
        
        # Calculate real-time metrics
        if self.detection_times:
            self.metrics.mean_time_to_detect = statistics.mean(self.detection_times)
            
        if self.response_times:
            self.metrics.mean_time_to_respond = statistics.mean(self.response_times)
            
        # Calculate detection rate
        total_attacks = getattr(self.cyber_range, 'total_attacks_executed', 0)
        if total_attacks > 0:
            detected_attacks = len([a for a in self.all_alerts if a.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]])
            self.metrics.detection_rate = detected_attacks / total_attacks
            
        return self.metrics
        
    def add_custom_rule(self, engine: str, rule_id: str, rule: Dict[str, Any]) -> None:
        """Add custom detection rule to specific engine"""
        
        if engine.lower() == 'siem':
            self.siem.add_rule(rule_id, rule)
        elif engine.lower() == 'ids':
            self.ids.signatures[rule_id] = rule
        elif engine.lower() == 'edr':
            self.edr.behavioral_rules[rule_id] = rule
        else:
            raise ValueError(f"Unknown engine: {engine}")
            
        logger.info(f"Added custom rule {rule_id} to {engine}")
        
    def correlate_alerts(self, alerts: List[SecurityAlert]) -> List[SecurityAlert]:
        """Correlate related alerts to reduce noise"""
        
        # Simple correlation based on IP addresses and time windows
        correlated = []
        processed_alert_ids = set()
        
        for alert in alerts:
            if alert.alert_id in processed_alert_ids:
                continue
                
            # Find related alerts
            related_alerts = [alert]
            
            for other_alert in alerts:
                if (other_alert.alert_id != alert.alert_id and 
                    other_alert.alert_id not in processed_alert_ids and
                    self._are_alerts_related(alert, other_alert)):
                    related_alerts.append(other_alert)
                    processed_alert_ids.add(other_alert.alert_id)
                    
            # Create correlated alert if multiple related alerts found
            if len(related_alerts) > 1:
                correlated_alert = self._create_correlated_alert(related_alerts)
                correlated.append(correlated_alert)
            else:
                correlated.append(alert)
                
            processed_alert_ids.add(alert.alert_id)
            
        return correlated
        
    def _process_alerts(self) -> None:
        """Background alert processing thread"""
        
        while self.is_running:
            try:
                # Get new alerts from engines
                new_alerts = []
                new_alerts.extend(self.siem.get_alerts())
                
                if new_alerts:
                    # Correlate alerts
                    correlated_alerts = self.correlate_alerts(new_alerts)
                    
                    # Process through alert processors
                    for processor in self.alert_processors:
                        try:
                            processor(correlated_alerts)
                        except Exception as e:
                            logger.error(f"Alert processor error: {e}")
                            
                time.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                time.sleep(5)
                
    def _update_metrics(self, alerts: List[SecurityAlert]) -> None:
        """Update defense metrics"""
        
        self.metrics.alerts_processed += len(alerts)
        
        # Record detection times for high/critical alerts
        for alert in alerts:
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                # Simulate detection time (in real implementation would calculate actual time)
                detection_time = 30 + (alert.confidence * 60)  # 30-90 seconds
                self.detection_times.append(detection_time)
                
    def _are_alerts_related(self, alert1: SecurityAlert, alert2: SecurityAlert) -> bool:
        """Check if two alerts are related"""
        
        # Time window check (within 5 minutes)
        time_diff = abs((alert1.timestamp - alert2.timestamp).total_seconds())
        if time_diff > 300:
            return False
            
        # IP address correlation
        if (alert1.source_ip and alert2.source_ip and 
            alert1.source_ip == alert2.source_ip):
            return True
            
        # Username correlation
        if (alert1.username and alert2.username and 
            alert1.username == alert2.username):
            return True
            
        # MITRE technique correlation
        if (alert1.mitre_techniques and alert2.mitre_techniques and
            set(alert1.mitre_techniques) & set(alert2.mitre_techniques)):
            return True
            
        return False
        
    def _create_correlated_alert(self, related_alerts: List[SecurityAlert]) -> SecurityAlert:
        """Create correlated alert from multiple related alerts"""
        
        # Use highest severity
        max_severity = max(alert.severity for alert in related_alerts)
        
        # Combine MITRE techniques
        all_techniques = set()
        for alert in related_alerts:
            all_techniques.update(alert.mitre_techniques)
            
        # Create correlated alert
        correlated_alert = SecurityAlert(
            alert_id=f"correlated_{int(time.time())}",
            timestamp=min(alert.timestamp for alert in related_alerts),
            severity=max_severity,
            source="Defense Suite Correlation",
            rule_name="Correlated Attack Activity",
            description=f"Correlated {len(related_alerts)} related alerts",
            source_ip=related_alerts[0].source_ip,
            destination_ip=related_alerts[0].destination_ip,
            username=related_alerts[0].username,
            mitre_techniques=list(all_techniques),
            confidence=min(alert.confidence for alert in related_alerts)
        )
        
        return correlated_alert


class BlueTeamEvaluator:
    """Evaluates blue team performance and capabilities"""
    
    def __init__(self, cyber_range):
        self.cyber_range = cyber_range
        self.defense_suite = None
        self.evaluation_sessions = []
        
    def deploy_defenses(self, defense_config: Dict[str, Any]) -> None:
        """Deploy defensive tools for evaluation"""
        
        self.defense_suite = DefenseSuite(self.cyber_range)
        self.defense_suite.deploy_defenses(defense_config)
        
        logger.info("Defenses deployed for evaluation")
        
    def evaluate(
        self,
        duration: str = "1h",
        attack_intensity: str = "medium",
        scoring_model: str = "mitre_attack"
    ) -> Dict[str, Any]:
        """Evaluate blue team performance"""
        
        logger.info(f"Starting blue team evaluation - Duration: {duration}, Intensity: {attack_intensity}")
        
        # Parse duration
        duration_seconds = self._parse_duration(duration)
        
        # Start evaluation session
        start_time = datetime.now()
        
        # Generate and execute attacks
        attack_results = self._execute_evaluation_attacks(duration_seconds, attack_intensity)
        
        # Collect defense metrics
        defense_metrics = self.defense_suite.get_metrics() if self.defense_suite else DefenseMetrics()
        
        # Calculate scores
        scores = self._calculate_scores(attack_results, defense_metrics, scoring_model)
        
        # Create evaluation result
        evaluation = {
            'evaluation_id': f"eval_{int(time.time())}",
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration': duration,
            'attack_intensity': attack_intensity,
            'scoring_model': scoring_model,
            'attack_results': attack_results,
            'defense_metrics': defense_metrics.to_dict(),
            'scores': scores,
            'detection_rate': scores.get('detection_rate', 0.0),
            'mttd': defense_metrics.mean_time_to_detect,
            'mttr': defense_metrics.mean_time_to_respond,
            'score': scores.get('overall_score', 0)
        }
        
        self.evaluation_sessions.append(evaluation)
        
        logger.info(f"Blue team evaluation completed - Overall score: {scores.get('overall_score', 0)}/100")
        
        return evaluation
        
    def _parse_duration(self, duration: str) -> int:
        """Parse duration string to seconds"""
        
        duration = duration.lower()
        
        if duration.endswith('s'):
            return int(duration[:-1])
        elif duration.endswith('m'):
            return int(duration[:-1]) * 60
        elif duration.endswith('h'):
            return int(duration[:-1]) * 3600
        elif duration.endswith('d'):
            return int(duration[:-1]) * 86400
        else:
            return int(duration)  # Assume seconds
            
    def _execute_evaluation_attacks(self, duration: int, intensity: str) -> Dict[str, Any]:
        """Execute attacks for evaluation"""
        
        # Attack intensity configuration
        intensity_config = {
            'low': {'attacks_per_minute': 0.5, 'complexity': 'basic'},
            'medium': {'attacks_per_minute': 2.0, 'complexity': 'intermediate'}, 
            'high': {'attacks_per_minute': 5.0, 'complexity': 'advanced'}
        }
        
        config = intensity_config.get(intensity, intensity_config['medium'])
        
        total_attacks = int(duration / 60 * config['attacks_per_minute'])
        successful_attacks = 0
        detected_attacks = 0
        
        # Simulate attack execution and detection
        for i in range(total_attacks):
            # Simulate attack success (varies by complexity)
            success_rate = {'basic': 0.7, 'intermediate': 0.8, 'advanced': 0.9}
            attack_successful = time.time() % 1 < success_rate.get(config['complexity'], 0.8)
            
            if attack_successful:
                successful_attacks += 1
                
                # Simulate detection (defense effectiveness)
                detection_rate = 0.7  # 70% detection rate
                if time.time() % 1 < detection_rate:
                    detected_attacks += 1
                    
        return {
            'total_attacks': total_attacks,
            'successful_attacks': successful_attacks,
            'detected_attacks': detected_attacks,
            'attack_success_rate': successful_attacks / max(1, total_attacks),
            'detection_rate': detected_attacks / max(1, successful_attacks)
        }
        
    def _calculate_scores(
        self, 
        attack_results: Dict[str, Any], 
        defense_metrics: DefenseMetrics,
        scoring_model: str
    ) -> Dict[str, float]:
        """Calculate evaluation scores"""
        
        scores = {}
        
        # Detection score (40% of total)
        detection_rate = attack_results.get('detection_rate', 0.0)
        scores['detection_score'] = detection_rate * 40
        
        # Response time score (30% of total)
        mttd = defense_metrics.mean_time_to_detect
        mttr = defense_metrics.mean_time_to_respond
        
        # Score based on response times (lower is better)
        response_score = 30
        if mttd > 300:  # > 5 minutes
            response_score -= 10
        if mttr > 900:  # > 15 minutes
            response_score -= 10
            
        scores['response_score'] = max(0, response_score)
        
        # False positive score (20% of total)
        fp_rate = defense_metrics.false_positive_rate
        fp_score = max(0, 20 - (fp_rate * 100))  # Penalty for high FP rate
        scores['false_positive_score'] = fp_score
        
        # Coverage score (10% of total)
        coverage_score = defense_metrics.coverage_score * 10
        scores['coverage_score'] = coverage_score
        
        # Overall score
        scores['overall_score'] = sum(scores.values())
        
        return scores