"""
Blue team evaluation and defense effectiveness measurement.

This module provides comprehensive evaluation of blue team defensive capabilities,
including detection performance, incident response effectiveness, and tool utilization.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class DefenseTool(Enum):
    """Types of defensive security tools"""
    SIEM = "siem"
    IDS = "ids"
    IPS = "ips"
    EDR = "edr"
    FIREWALL = "firewall"
    ANTIVIRUS = "antivirus"
    DECEPTION = "deception"
    THREAT_HUNTING = "threat_hunting"
    FORENSICS = "forensics"
    VULNERABILITY_SCANNER = "vulnerability_scanner"


@dataclass
class DetectionEvent:
    """Represents a security detection event"""
    event_id: str
    timestamp: datetime
    source: str
    attack_id: Optional[str]
    technique_id: Optional[str]
    severity: str
    confidence: float
    true_positive: bool
    response_time: Optional[float] = None  # minutes
    escalated: bool = False
    resolved: bool = False


@dataclass
class DefenseMetrics:
    """Comprehensive defense performance metrics"""
    detection_rate: float
    false_positive_rate: float
    false_negative_rate: float
    mean_time_to_detect: float  # minutes
    mean_time_to_respond: float  # minutes
    mean_time_to_contain: float  # minutes
    mean_time_to_eradicate: float  # minutes
    mean_time_to_recover: float  # minutes
    coverage_percentage: float
    tool_effectiveness: Dict[str, float]
    incident_resolution_rate: float
    escalation_accuracy: float
    total_events_processed: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


@dataclass
class IncidentResponse:
    """Incident response tracking"""
    incident_id: str
    start_time: datetime
    detection_time: Optional[datetime]
    response_time: Optional[datetime]
    containment_time: Optional[datetime]
    eradication_time: Optional[datetime]
    recovery_time: Optional[datetime]
    severity: str
    attack_type: str
    affected_systems: List[str]
    responder_actions: List[Dict[str, Any]]
    effectiveness_score: float
    lessons_learned: List[str]


class BlueTeamEvaluator:
    """Main blue team evaluation system"""
    
    def __init__(self, cyber_range):
        self.cyber_range = cyber_range
        self.deployed_defenses = {}
        self.detection_events = []
        self.incident_responses = []
        self.performance_history = []
        
        # Event handlers
        self.event_handlers = {
            'detection': [],
            'incident_response': [],
            'containment': [],
            'eradication': []
        }
        
        # Tool configurations
        self.tool_configs = {
            'siem': {'detection_rate': 0.8, 'false_positive_rate': 0.15},
            'ids': {'detection_rate': 0.7, 'false_positive_rate': 0.20},
            'edr': {'detection_rate': 0.85, 'false_positive_rate': 0.10},
            'deception': {'detection_rate': 0.95, 'false_positive_rate': 0.02}
        }
    
    def deploy_defenses(self, defense_config: Dict[str, str]) -> None:
        """Deploy defensive tools and systems"""
        
        logger.info(f"Deploying {len(defense_config)} defensive tools")
        
        for tool_type, tool_name in defense_config.items():
            self._deploy_single_defense(tool_type, tool_name)
            self.deployed_defenses[tool_type] = {
                'name': tool_name,
                'status': 'active',
                'deployment_time': datetime.now(),
                'events_detected': 0,
                'false_positives': 0
            }
        
        # Set up event monitoring
        self._setup_defense_monitoring()
        
        logger.info("Defense deployment completed")
    
    def on_event(self, event_type: str) -> Callable:
        """Decorator for registering event handlers"""
        def decorator(func):
            if event_type in self.event_handlers:
                self.event_handlers[event_type].append(func)
            return func
        return decorator
    
    def process_attack_event(self, attack_event: Dict[str, Any]) -> List[DetectionEvent]:
        """Process an attack event and generate detection results"""
        
        detections = []
        
        # Simulate detection by each deployed tool
        for tool_type, tool_info in self.deployed_defenses.items():
            if tool_info['status'] == 'active':
                detection = self._simulate_tool_detection(tool_type, attack_event)
                if detection:
                    detections.append(detection)
                    tool_info['events_detected'] += 1
                    
                    # Track false positives
                    if not detection.true_positive:
                        tool_info['false_positives'] += 1
        
        # Store detection events
        self.detection_events.extend(detections)
        
        # Trigger event handlers
        for detection in detections:
            self._trigger_event_handlers('detection', detection)
        
        return detections
    
    def evaluate(
        self,
        duration: str = "24h",
        attack_intensity: str = "medium",
        scoring_model: str = "mitre_attack"
    ) -> DefenseMetrics:
        """Comprehensive blue team evaluation"""
        
        logger.info(f"Starting {duration} blue team evaluation")
        
        # Parse duration
        duration_hours = self._parse_duration(duration)
        
        # Run evaluation period
        evaluation_results = self._run_evaluation_period(duration_hours, attack_intensity)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_defense_metrics(evaluation_results)
        
        # Store in history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'duration': duration,
            'metrics': metrics,
            'scoring_model': scoring_model
        })
        
        logger.info(f"Evaluation completed - Overall score: {self._calculate_overall_score(metrics)}")
        return metrics
    
    def generate_performance_report(
        self,
        metrics: DefenseMetrics,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'overall_performance_score': self._calculate_overall_score(metrics),
            'detection_performance': {
                'detection_rate': metrics.detection_rate,
                'false_positive_rate': metrics.false_positive_rate,
                'false_negative_rate': metrics.false_negative_rate,
                'precision': metrics.true_positives / max(1, metrics.true_positives + metrics.false_positives),
                'recall': metrics.detection_rate,
                'f1_score': self._calculate_f1_score(metrics)
            },
            'response_performance': {
                'mean_time_to_detect': metrics.mean_time_to_detect,
                'mean_time_to_respond': metrics.mean_time_to_respond,
                'mean_time_to_contain': metrics.mean_time_to_contain,
                'mean_time_to_recover': metrics.mean_time_to_recover,
                'incident_resolution_rate': metrics.incident_resolution_rate
            },
            'tool_effectiveness': metrics.tool_effectiveness,
            'coverage_analysis': {
                'coverage_percentage': metrics.coverage_percentage,
                'gaps_identified': self._identify_coverage_gaps(),
                'redundancy_analysis': self._analyze_tool_redundancy()
            },
            'trend_analysis': self._analyze_performance_trends(),
            'maturity_assessment': self._assess_security_maturity(metrics)
        }
        
        if include_recommendations:
            report['recommendations'] = self._generate_improvement_recommendations(metrics)
        
        return report
    
    def track_incident_response(
        self,
        incident_id: str,
        attack_type: str,
        severity: str,
        affected_systems: List[str]
    ) -> IncidentResponse:
        """Track incident response process"""
        
        incident = IncidentResponse(
            incident_id=incident_id,
            start_time=datetime.now(),
            detection_time=None,
            response_time=None,
            containment_time=None,
            eradication_time=None,
            recovery_time=None,
            severity=severity,
            attack_type=attack_type,
            affected_systems=affected_systems,
            responder_actions=[],
            effectiveness_score=0.0,
            lessons_learned=[]
        )
        
        self.incident_responses.append(incident)
        logger.info(f"Started tracking incident {incident_id}")
        
        return incident
    
    def _deploy_single_defense(self, tool_type: str, tool_name: str) -> None:
        """Deploy a single defensive tool"""
        
        # In real implementation, this would actually deploy the tool
        logger.info(f"Deploying {tool_type}: {tool_name}")
        
        # Simulate deployment based on tool type
        deployment_configs = {
            'siem': self._deploy_siem,
            'ids': self._deploy_ids,
            'edr': self._deploy_edr,
            'deception': self._deploy_deception_tools
        }
        
        deploy_func = deployment_configs.get(tool_type, self._deploy_generic_tool)
        deploy_func(tool_name)
    
    def _deploy_siem(self, tool_name: str) -> None:
        """Deploy SIEM solution"""
        # Simulate SIEM deployment
        logger.debug(f"Configuring SIEM: {tool_name}")
    
    def _deploy_ids(self, tool_name: str) -> None:
        """Deploy IDS solution"""
        # Simulate IDS deployment
        logger.debug(f"Configuring IDS: {tool_name}")
    
    def _deploy_edr(self, tool_name: str) -> None:
        """Deploy EDR solution"""
        # Simulate EDR deployment
        logger.debug(f"Configuring EDR: {tool_name}")
    
    def _deploy_deception_tools(self, tool_name: str) -> None:
        """Deploy deception/honeypot tools"""
        # Simulate deception tool deployment
        logger.debug(f"Configuring deception tools: {tool_name}")
    
    def _deploy_generic_tool(self, tool_name: str) -> None:
        """Deploy generic security tool"""
        logger.debug(f"Configuring generic tool: {tool_name}")
    
    def _setup_defense_monitoring(self) -> None:
        """Set up monitoring for deployed defenses"""
        
        # Register with cyber range event system
        @self.cyber_range.on_event('detection')
        def handle_detection_event(event_data):
            self.process_attack_event(event_data)
        
        @self.cyber_range.on_event('incident')
        def handle_incident_event(event_data):
            self._handle_incident_event(event_data)
    
    def _simulate_tool_detection(self, tool_type: str, attack_event: Dict[str, Any]) -> Optional[DetectionEvent]:
        """Simulate detection by a specific tool"""
        
        tool_config = self.tool_configs.get(tool_type, {'detection_rate': 0.5, 'false_positive_rate': 0.2})
        
        # Determine if tool detects the attack
        detects_attack = np.random.random() < tool_config['detection_rate']
        
        if detects_attack or np.random.random() < tool_config['false_positive_rate']:
            # Generate detection event
            detection = DetectionEvent(
                event_id=f"det_{np.random.randint(100000, 999999)}",
                timestamp=datetime.now(),
                source=tool_type,
                attack_id=attack_event.get('attack_id'),
                technique_id=attack_event.get('technique_id'),
                severity=attack_event.get('severity', 'medium'),
                confidence=np.random.uniform(0.6, 0.95) if detects_attack else np.random.uniform(0.3, 0.7),
                true_positive=detects_attack,
                response_time=None
            )
            
            return detection
        
        return None
    
    def _parse_duration(self, duration: str) -> float:
        """Parse duration string to hours"""
        if duration.endswith('h'):
            return float(duration[:-1])
        elif duration.endswith('d'):
            return float(duration[:-1]) * 24
        elif duration.endswith('m'):
            return float(duration[:-1]) / 60
        else:
            return 24.0  # Default to 24 hours
    
    def _run_evaluation_period(self, duration_hours: float, intensity: str) -> Dict[str, Any]:
        """Run evaluation period with simulated attacks"""
        
        # Attack intensity configuration
        intensity_config = {
            'low': {'attacks_per_hour': 2, 'complexity_modifier': 0.7},
            'medium': {'attacks_per_hour': 5, 'complexity_modifier': 1.0},
            'high': {'attacks_per_hour': 10, 'complexity_modifier': 1.3}
        }
        
        config = intensity_config.get(intensity, intensity_config['medium'])
        total_attacks = int(duration_hours * config['attacks_per_hour'])
        
        results = {
            'total_attacks': total_attacks,
            'detected_attacks': 0,
            'false_positives': 0,
            'response_times': [],
            'detection_times': [],
            'tool_performance': {}
        }
        
        # Simulate attacks and detection
        for i in range(total_attacks):
            attack_event = self._generate_simulated_attack(config['complexity_modifier'])
            detections = self.process_attack_event(attack_event)
            
            if detections:
                # At least one tool detected
                results['detected_attacks'] += 1
                
                # Track detection timing
                detection_time = np.random.uniform(1, 15)  # 1-15 minutes
                results['detection_times'].append(detection_time)
                
                # Simulate response time
                response_time = detection_time + np.random.uniform(5, 30)
                results['response_times'].append(response_time)
                
                # Track false positives
                false_positive_count = sum(1 for d in detections if not d.true_positive)
                results['false_positives'] += false_positive_count
        
        return results
    
    def _generate_simulated_attack(self, complexity_modifier: float) -> Dict[str, Any]:
        """Generate a simulated attack for evaluation"""
        
        attack_types = ['malware', 'phishing', 'lateral_movement', 'data_exfiltration', 'privilege_escalation']
        techniques = ['T1059', 'T1078', 'T1190', 'T1021', 'T1110', 'T1046', 'T1003', 'T1041']
        severities = ['low', 'medium', 'high', 'critical']
        
        return {
            'attack_id': f"sim_attack_{np.random.randint(100000, 999999)}",
            'attack_type': np.random.choice(attack_types),
            'technique_id': np.random.choice(techniques),
            'severity': np.random.choice(severities),
            'complexity': np.random.uniform(0.3, 1.0) * complexity_modifier,
            'stealth_level': np.random.uniform(0.2, 0.9),
            'timestamp': datetime.now()
        }
    
    def _calculate_defense_metrics(self, evaluation_results: Dict[str, Any]) -> DefenseMetrics:
        """Calculate comprehensive defense metrics"""
        
        total_attacks = evaluation_results['total_attacks']
        detected_attacks = evaluation_results['detected_attacks']
        false_positives = evaluation_results['false_positives']
        
        # Basic metrics
        detection_rate = detected_attacks / max(1, total_attacks)
        false_negative_rate = (total_attacks - detected_attacks) / max(1, total_attacks)
        
        # Timing metrics
        mean_time_to_detect = np.mean(evaluation_results['detection_times']) if evaluation_results['detection_times'] else 0
        mean_time_to_respond = np.mean(evaluation_results['response_times']) if evaluation_results['response_times'] else 0
        
        # Tool effectiveness
        tool_effectiveness = {}
        for tool_type, tool_info in self.deployed_defenses.items():
            events_detected = tool_info['events_detected']
            false_pos = tool_info['false_positives']
            
            if events_detected > 0:
                precision = (events_detected - false_pos) / events_detected
                tool_effectiveness[tool_type] = precision
            else:
                tool_effectiveness[tool_type] = 0.0
        
        # Coverage percentage (simplified)
        coverage_percentage = min(1.0, len(self.deployed_defenses) / 5.0)  # Assume 5 tools = full coverage
        
        return DefenseMetrics(
            detection_rate=detection_rate,
            false_positive_rate=false_positives / max(1, detected_attacks + false_positives),
            false_negative_rate=false_negative_rate,
            mean_time_to_detect=mean_time_to_detect,
            mean_time_to_respond=mean_time_to_respond,
            mean_time_to_contain=mean_time_to_respond + 15,  # Estimated
            mean_time_to_eradicate=mean_time_to_respond + 45,  # Estimated
            mean_time_to_recover=mean_time_to_respond + 120,  # Estimated
            coverage_percentage=coverage_percentage,
            tool_effectiveness=tool_effectiveness,
            incident_resolution_rate=0.85,  # Estimated
            escalation_accuracy=0.75,  # Estimated
            total_events_processed=total_attacks + false_positives,
            true_positives=detected_attacks,
            false_positives=false_positives,
            false_negatives=total_attacks - detected_attacks,
            true_negatives=0  # Not applicable in this context
        )
    
    def _calculate_overall_score(self, metrics: DefenseMetrics) -> float:
        """Calculate overall defense effectiveness score"""
        
        # Weighted scoring
        weights = {
            'detection_rate': 0.25,
            'false_positive_penalty': 0.15,
            'response_time': 0.20,
            'coverage': 0.15,
            'tool_effectiveness': 0.15,
            'incident_resolution': 0.10
        }
        
        # Normalize response time (lower is better)
        response_time_score = max(0.0, 1.0 - (metrics.mean_time_to_respond / 60.0))
        
        # Average tool effectiveness
        avg_tool_effectiveness = np.mean(list(metrics.tool_effectiveness.values())) if metrics.tool_effectiveness else 0.0
        
        # False positive penalty
        false_positive_penalty = 1.0 - metrics.false_positive_rate
        
        overall_score = (
            metrics.detection_rate * weights['detection_rate'] +
            false_positive_penalty * weights['false_positive_penalty'] +
            response_time_score * weights['response_time'] +
            metrics.coverage_percentage * weights['coverage'] +
            avg_tool_effectiveness * weights['tool_effectiveness'] +
            metrics.incident_resolution_rate * weights['incident_resolution']
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _calculate_f1_score(self, metrics: DefenseMetrics) -> float:
        """Calculate F1 score for detection performance"""
        
        precision = metrics.true_positives / max(1, metrics.true_positives + metrics.false_positives)
        recall = metrics.detection_rate
        
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0
    
    def _identify_coverage_gaps(self) -> List[str]:
        """Identify security coverage gaps"""
        
        deployed_types = set(self.deployed_defenses.keys())
        recommended_types = {'siem', 'ids', 'edr', 'deception', 'firewall'}
        
        gaps = recommended_types - deployed_types
        return list(gaps)
    
    def _analyze_tool_redundancy(self) -> Dict[str, Any]:
        """Analyze tool redundancy and optimization opportunities"""
        
        return {
            'redundant_capabilities': [],
            'optimization_suggestions': [],
            'cost_effectiveness': {}
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        if len(self.performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend in overall scores
        recent_scores = [h['metrics'] for h in self.performance_history[-5:]]
        overall_scores = [self._calculate_overall_score(metrics) for metrics in recent_scores]
        
        if len(overall_scores) >= 2:
            trend = 'improving' if overall_scores[-1] > overall_scores[0] else 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'score_history': overall_scores,
            'improvement_rate': (overall_scores[-1] - overall_scores[0]) / len(overall_scores) if len(overall_scores) > 1 else 0
        }
    
    def _assess_security_maturity(self, metrics: DefenseMetrics) -> Dict[str, Any]:
        """Assess overall security maturity level"""
        
        overall_score = self._calculate_overall_score(metrics)
        
        if overall_score >= 0.9:
            maturity_level = 'optimized'
        elif overall_score >= 0.7:
            maturity_level = 'managed'
        elif overall_score >= 0.5:
            maturity_level = 'defined'
        elif overall_score >= 0.3:
            maturity_level = 'repeatable'
        else:
            maturity_level = 'initial'
        
        return {
            'level': maturity_level,
            'score': overall_score,
            'strengths': self._identify_strengths(metrics),
            'weaknesses': self._identify_weaknesses(metrics)
        }
    
    def _identify_strengths(self, metrics: DefenseMetrics) -> List[str]:
        """Identify security strengths"""
        
        strengths = []
        
        if metrics.detection_rate > 0.8:
            strengths.append('High detection rate')
        if metrics.false_positive_rate < 0.1:
            strengths.append('Low false positive rate')
        if metrics.mean_time_to_respond < 10:
            strengths.append('Fast response time')
        if metrics.coverage_percentage > 0.8:
            strengths.append('Comprehensive coverage')
        
        return strengths
    
    def _identify_weaknesses(self, metrics: DefenseMetrics) -> List[str]:
        """Identify security weaknesses"""
        
        weaknesses = []
        
        if metrics.detection_rate < 0.6:
            weaknesses.append('Low detection rate')
        if metrics.false_positive_rate > 0.3:
            weaknesses.append('High false positive rate')
        if metrics.mean_time_to_respond > 30:
            weaknesses.append('Slow response time')
        if metrics.coverage_percentage < 0.5:
            weaknesses.append('Insufficient coverage')
        
        return weaknesses
    
    def _generate_improvement_recommendations(self, metrics: DefenseMetrics) -> List[Dict[str, Any]]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        # Detection rate improvements
        if metrics.detection_rate < 0.7:
            recommendations.append({
                'area': 'Detection Capability',
                'priority': 'high',
                'recommendation': 'Deploy additional detection tools and tune existing ones',
                'expected_improvement': '15-25% detection rate increase'
            })
        
        # False positive reduction
        if metrics.false_positive_rate > 0.2:
            recommendations.append({
                'area': 'Alert Quality',
                'priority': 'medium',
                'recommendation': 'Tune detection rules and implement better correlation',
                'expected_improvement': '30-50% false positive reduction'
            })
        
        # Response time improvements
        if metrics.mean_time_to_respond > 20:
            recommendations.append({
                'area': 'Response Speed',
                'priority': 'high',
                'recommendation': 'Implement automated response and improve procedures',
                'expected_improvement': '40-60% response time reduction'
            })
        
        # Coverage gaps
        gaps = self._identify_coverage_gaps()
        if gaps:
            recommendations.append({
                'area': 'Security Coverage',
                'priority': 'medium',
                'recommendation': f'Deploy missing security tools: {", ".join(gaps)}',
                'expected_improvement': 'Improved attack coverage and visibility'
            })
        
        return recommendations
    
    def _trigger_event_handlers(self, event_type: str, event_data: Any) -> None:
        """Trigger registered event handlers"""
        
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
    
    def _handle_incident_event(self, event_data: Dict[str, Any]) -> None:
        """Handle incident-related events"""
        
        incident_id = event_data.get('incident_id')
        if incident_id:
            # Find existing incident or create new one
            incident = next((ir for ir in self.incident_responses if ir.incident_id == incident_id), None)
            
            if incident:
                # Update incident timeline
                event_type = event_data.get('type')
                if event_type == 'response_started':
                    incident.response_time = datetime.now()
                elif event_type == 'contained':
                    incident.containment_time = datetime.now()
                elif event_type == 'eradicated':
                    incident.eradication_time = datetime.now()
                elif event_type == 'recovered':
                    incident.recovery_time = datetime.now()
                
                # Log responder action
                incident.responder_actions.append({
                    'timestamp': datetime.now(),
                    'action': event_data.get('action', 'unknown'),
                    'responder': event_data.get('responder', 'unknown')
                })