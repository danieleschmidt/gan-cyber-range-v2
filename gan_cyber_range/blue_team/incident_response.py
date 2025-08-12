"""
Automated incident response and management system.

This module provides comprehensive incident response capabilities including
automated containment, investigation workflows, and response coordination.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import time

from .defense_suite import SecurityAlert, AlertSeverity

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status"""
    NEW = "new"
    ASSIGNED = "assigned"
    INVESTIGATING = "investigating"
    CONTAINING = "containing"
    ERADICATING = "eradicating"
    RECOVERING = "recovering"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ResponseAction(Enum):
    """Automated response actions"""
    ISOLATE_HOST = "isolate_host"
    BLOCK_IP = "block_ip"
    DISABLE_USER = "disable_user"
    QUARANTINE_FILE = "quarantine_file"
    RESET_PASSWORD = "reset_password"
    COLLECT_EVIDENCE = "collect_evidence"
    CREATE_SNAPSHOT = "create_snapshot"
    NOTIFY_TEAM = "notify_team"


@dataclass
class IncidentArtifact:
    """Artifact collected during incident investigation"""
    artifact_id: str
    artifact_type: str  # file, memory_dump, network_capture, log
    description: str
    file_path: Optional[str] = None
    hash_value: Optional[str] = None
    collection_time: datetime = field(default_factory=datetime.now)
    chain_of_custody: List[str] = field(default_factory=list)


@dataclass
class ResponseActionResult:
    """Result of an automated response action"""
    action_id: str
    action_type: ResponseAction
    success: bool
    timestamp: datetime
    details: str
    errors: List[str] = field(default_factory=list)


@dataclass
class Incident:
    """Represents a security incident"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    created_time: datetime
    updated_time: datetime
    assigned_to: Optional[str] = None
    source_alerts: List[SecurityAlert] = field(default_factory=list)
    artifacts: List[IncidentArtifact] = field(default_factory=list)
    response_actions: List[ResponseActionResult] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    containment_status: str = "not_started"
    eradication_status: str = "not_started"
    recovery_status: str = "not_started"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary"""
        return {
            'incident_id': self.incident_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'status': self.status.value,
            'created_time': self.created_time.isoformat(),
            'updated_time': self.updated_time.isoformat(),
            'assigned_to': self.assigned_to,
            'affected_systems': self.affected_systems,
            'mitre_techniques': self.mitre_techniques,
            'containment_status': self.containment_status,
            'eradication_status': self.eradication_status,
            'recovery_status': self.recovery_status,
            'artifact_count': len(self.artifacts),
            'action_count': len(self.response_actions)
        }


class PlaybookEngine:
    """Automated incident response playbook engine"""
    
    def __init__(self):
        self.playbooks = {}
        self.active_playbooks = {}
        self._load_default_playbooks()
        
    def add_playbook(self, name: str, playbook: Dict[str, Any]) -> None:
        """Add incident response playbook"""
        self.playbooks[name] = playbook
        logger.info(f"Added playbook: {name}")
        
    def execute_playbook(
        self, 
        playbook_name: str, 
        incident: Incident,
        incident_manager: 'IncidentManager'
    ) -> str:
        """Execute incident response playbook"""
        
        if playbook_name not in self.playbooks:
            raise ValueError(f"Playbook not found: {playbook_name}")
            
        playbook = self.playbooks[playbook_name]
        execution_id = str(uuid.uuid4())
        
        # Start playbook execution
        self.active_playbooks[execution_id] = {
            'playbook_name': playbook_name,
            'incident_id': incident.incident_id,
            'start_time': datetime.now(),
            'status': 'running',
            'current_step': 0,
            'steps_completed': 0,
            'steps_total': len(playbook.get('steps', []))
        }
        
        # Execute playbook steps in background
        thread = threading.Thread(
            target=self._execute_playbook_steps,
            args=(execution_id, playbook, incident, incident_manager),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started playbook execution: {playbook_name} for incident {incident.incident_id}")
        return execution_id
        
    def _execute_playbook_steps(
        self,
        execution_id: str,
        playbook: Dict[str, Any],
        incident: Incident,
        incident_manager: 'IncidentManager'
    ) -> None:
        """Execute playbook steps"""
        
        try:
            steps = playbook.get('steps', [])
            
            for i, step in enumerate(steps):
                # Update execution status
                self.active_playbooks[execution_id]['current_step'] = i
                
                # Execute step
                success = self._execute_step(step, incident, incident_manager)
                
                if success:
                    self.active_playbooks[execution_id]['steps_completed'] += 1
                    
                    # Add to incident timeline
                    incident.timeline.append({
                        'timestamp': datetime.now().isoformat(),
                        'action': 'playbook_step_completed',
                        'details': f"Completed step: {step.get('name', 'Unknown')}"
                    })
                else:
                    # Handle step failure
                    if step.get('required', True):
                        logger.error(f"Required playbook step failed: {step.get('name')}")
                        break
                    else:
                        logger.warning(f"Optional playbook step failed: {step.get('name')}")
                        
                # Delay between steps if specified
                delay = step.get('delay', 0)
                if delay > 0:
                    time.sleep(delay)
                    
            # Mark playbook as completed
            self.active_playbooks[execution_id]['status'] = 'completed'
            self.active_playbooks[execution_id]['end_time'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Playbook execution error: {e}")
            self.active_playbooks[execution_id]['status'] = 'failed'
            self.active_playbooks[execution_id]['error'] = str(e)
            
    def _execute_step(
        self,
        step: Dict[str, Any],
        incident: Incident,
        incident_manager: 'IncidentManager'
    ) -> bool:
        """Execute individual playbook step"""
        
        step_type = step.get('type')
        
        try:
            if step_type == 'isolate_host':
                return self._isolate_host_step(step, incident, incident_manager)
            elif step_type == 'collect_evidence':
                return self._collect_evidence_step(step, incident, incident_manager)
            elif step_type == 'block_network':
                return self._block_network_step(step, incident, incident_manager)
            elif step_type == 'notify':
                return self._notify_step(step, incident, incident_manager)
            elif step_type == 'update_status':
                return self._update_status_step(step, incident, incident_manager)
            else:
                logger.warning(f"Unknown step type: {step_type}")
                return False
                
        except Exception as e:
            logger.error(f"Step execution error: {e}")
            return False
            
    def _isolate_host_step(
        self,
        step: Dict[str, Any],
        incident: Incident,
        incident_manager: 'IncidentManager'
    ) -> bool:
        """Execute host isolation step"""
        
        # Get target hosts from incident or step config
        target_hosts = step.get('hosts', incident.affected_systems)
        
        for host in target_hosts:
            result = incident_manager.execute_response_action(
                ResponseAction.ISOLATE_HOST,
                {'host': host},
                incident.incident_id
            )
            
            if not result.success:
                return False
                
        return True
        
    def _collect_evidence_step(
        self,
        step: Dict[str, Any],
        incident: Incident,
        incident_manager: 'IncidentManager'
    ) -> bool:
        """Execute evidence collection step"""
        
        evidence_types = step.get('evidence_types', ['memory', 'disk', 'network'])
        
        for evidence_type in evidence_types:
            result = incident_manager.execute_response_action(
                ResponseAction.COLLECT_EVIDENCE,
                {'type': evidence_type, 'systems': incident.affected_systems},
                incident.incident_id
            )
            
            if not result.success:
                logger.warning(f"Failed to collect {evidence_type} evidence")
                
        return True
        
    def _block_network_step(
        self,
        step: Dict[str, Any],
        incident: Incident,
        incident_manager: 'IncidentManager'
    ) -> bool:
        """Execute network blocking step"""
        
        # Extract IPs from incident alerts
        ips_to_block = set()
        for alert in incident.source_alerts:
            if alert.source_ip:
                ips_to_block.add(alert.source_ip)
                
        # Add IPs from step config
        ips_to_block.update(step.get('ips', []))
        
        for ip in ips_to_block:
            result = incident_manager.execute_response_action(
                ResponseAction.BLOCK_IP,
                {'ip': ip},
                incident.incident_id
            )
            
            if not result.success:
                return False
                
        return True
        
    def _notify_step(
        self,
        step: Dict[str, Any],
        incident: Incident,
        incident_manager: 'IncidentManager'
    ) -> bool:
        """Execute notification step"""
        
        recipients = step.get('recipients', ['security_team'])
        message = step.get('message', f"Incident {incident.incident_id} requires attention")
        
        result = incident_manager.execute_response_action(
            ResponseAction.NOTIFY_TEAM,
            {'recipients': recipients, 'message': message},
            incident.incident_id
        )
        
        return result.success
        
    def _update_status_step(
        self,
        step: Dict[str, Any],
        incident: Incident,
        incident_manager: 'IncidentManager'
    ) -> bool:
        """Execute status update step"""
        
        new_status = step.get('status')
        if new_status:
            try:
                incident.status = IncidentStatus(new_status)
                incident.updated_time = datetime.now()
                return True
            except ValueError:
                logger.error(f"Invalid status: {new_status}")
                return False
                
        return True
        
    def _load_default_playbooks(self) -> None:
        """Load default incident response playbooks"""
        
        # Malware incident playbook
        malware_playbook = {
            'name': 'Malware Incident Response',
            'description': 'Standard response for malware infections',
            'triggers': ['malware_detected', 'suspicious_process'],
            'steps': [
                {
                    'name': 'Isolate Infected Hosts',
                    'type': 'isolate_host',
                    'description': 'Isolate all affected systems',
                    'required': True
                },
                {
                    'name': 'Collect Evidence',
                    'type': 'collect_evidence',
                    'evidence_types': ['memory', 'disk'],
                    'description': 'Collect forensic evidence',
                    'required': True
                },
                {
                    'name': 'Block Malicious IPs',
                    'type': 'block_network',
                    'description': 'Block command and control IPs',
                    'required': True
                },
                {
                    'name': 'Notify Security Team',
                    'type': 'notify',
                    'recipients': ['security_team', 'incident_commander'],
                    'message': 'Malware incident detected and initial containment initiated',
                    'required': True
                },
                {
                    'name': 'Update Status to Investigating',
                    'type': 'update_status',
                    'status': 'investigating',
                    'required': True
                }
            ]
        }
        
        # Data exfiltration playbook
        exfiltration_playbook = {
            'name': 'Data Exfiltration Response',
            'description': 'Response for suspected data theft',
            'triggers': ['data_exfiltration', 'unusual_data_transfer'],
            'steps': [
                {
                    'name': 'Block Suspicious Network Traffic',
                    'type': 'block_network',
                    'description': 'Block suspicious destinations',
                    'required': True
                },
                {
                    'name': 'Collect Network Evidence',
                    'type': 'collect_evidence',
                    'evidence_types': ['network', 'logs'],
                    'description': 'Capture network traffic and logs',
                    'required': True
                },
                {
                    'name': 'Isolate Source Systems',
                    'type': 'isolate_host',
                    'description': 'Isolate systems involved in exfiltration',
                    'required': True
                },
                {
                    'name': 'Notify Legal and Compliance',
                    'type': 'notify',
                    'recipients': ['legal_team', 'compliance_team', 'security_team'],
                    'message': 'Potential data exfiltration incident detected',
                    'required': True
                }
            ]
        }
        
        # Credential compromise playbook
        credential_playbook = {
            'name': 'Credential Compromise Response',
            'description': 'Response for compromised credentials',
            'triggers': ['credential_dumping', 'suspicious_login'],
            'steps': [
                {
                    'name': 'Disable Compromised Accounts',
                    'type': 'disable_user',
                    'description': 'Disable affected user accounts',
                    'required': True
                },
                {
                    'name': 'Force Password Reset',
                    'type': 'reset_password',
                    'description': 'Reset passwords for affected accounts',
                    'required': True
                },
                {
                    'name': 'Collect Authentication Logs',
                    'type': 'collect_evidence',
                    'evidence_types': ['logs'],
                    'description': 'Collect authentication and access logs',
                    'required': True
                },
                {
                    'name': 'Review Access Logs',
                    'type': 'update_status',
                    'status': 'investigating',
                    'description': 'Begin investigation of access patterns',
                    'required': True
                }
            ]
        }
        
        # Add playbooks
        self.add_playbook('malware_incident', malware_playbook)
        self.add_playbook('data_exfiltration', exfiltration_playbook)
        self.add_playbook('credential_compromise', credential_playbook)


class IncidentManager:
    """Main incident management system"""
    
    def __init__(self, cyber_range):
        self.cyber_range = cyber_range
        self.incidents = {}
        self.playbook_engine = PlaybookEngine()
        self.response_actions = {}
        self.auto_response_enabled = True
        
        # Alert to incident mapping
        self.alert_threshold = 3  # Create incident after 3 related alerts
        self.pending_alerts = {}
        
        logger.info("Incident manager initialized")
        
    def process_alerts(self, alerts: List[SecurityAlert]) -> List[Incident]:
        """Process security alerts and create incidents"""
        
        created_incidents = []
        
        for alert in alerts:
            # Check if alert should create incident
            incident = self._should_create_incident(alert)
            
            if incident:
                created_incidents.append(incident)
                
                # Trigger automated response if enabled
                if self.auto_response_enabled:
                    self._trigger_auto_response(incident)
                    
        return created_incidents
        
    def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        source_alerts: List[SecurityAlert] = None
    ) -> Incident:
        """Manually create incident"""
        
        incident_id = f"INC-{int(time.time())}"
        
        # Extract affected systems and MITRE techniques
        affected_systems = set()
        mitre_techniques = set()
        
        if source_alerts:
            for alert in source_alerts:
                if alert.source_ip:
                    affected_systems.add(alert.source_ip)
                if alert.destination_ip:
                    affected_systems.add(alert.destination_ip)
                mitre_techniques.update(alert.mitre_techniques)
                
        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.NEW,
            created_time=datetime.now(),
            updated_time=datetime.now(),
            source_alerts=source_alerts or [],
            affected_systems=list(affected_systems),
            mitre_techniques=list(mitre_techniques)
        )
        
        # Add to incident timeline
        incident.timeline.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'incident_created',
            'details': f'Incident created: {title}'
        })
        
        self.incidents[incident_id] = incident
        
        logger.info(f"Created incident: {incident_id} - {title}")
        return incident
        
    def update_incident(
        self,
        incident_id: str,
        status: Optional[IncidentStatus] = None,
        assigned_to: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Optional[Incident]:
        """Update incident details"""
        
        incident = self.incidents.get(incident_id)
        if not incident:
            return None
            
        # Update fields
        if status:
            incident.status = status
        if assigned_to:
            incident.assigned_to = assigned_to
            
        incident.updated_time = datetime.now()
        
        # Add to timeline
        updates = []
        if status:
            updates.append(f"Status: {status.value}")
        if assigned_to:
            updates.append(f"Assigned to: {assigned_to}")
        if notes:
            updates.append(f"Notes: {notes}")
            
        if updates:
            incident.timeline.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'incident_updated',
                'details': ', '.join(updates)
            })
            
        logger.info(f"Updated incident {incident_id}: {', '.join(updates)}")
        return incident
        
    def execute_response_action(
        self,
        action_type: ResponseAction,
        parameters: Dict[str, Any],
        incident_id: str
    ) -> ResponseActionResult:
        """Execute automated response action"""
        
        action_id = str(uuid.uuid4())
        
        try:
            # Execute action based on type
            if action_type == ResponseAction.ISOLATE_HOST:
                success, details = self._isolate_host(parameters)
            elif action_type == ResponseAction.BLOCK_IP:
                success, details = self._block_ip(parameters)
            elif action_type == ResponseAction.DISABLE_USER:
                success, details = self._disable_user(parameters)
            elif action_type == ResponseAction.COLLECT_EVIDENCE:
                success, details = self._collect_evidence(parameters)
            elif action_type == ResponseAction.NOTIFY_TEAM:
                success, details = self._notify_team(parameters)
            else:
                success, details = False, f"Unknown action type: {action_type}"
                
            result = ResponseActionResult(
                action_id=action_id,
                action_type=action_type,
                success=success,
                timestamp=datetime.now(),
                details=details
            )
            
            # Add to incident
            incident = self.incidents.get(incident_id)
            if incident:
                incident.response_actions.append(result)
                incident.timeline.append({
                    'timestamp': datetime.now().isoformat(),
                    'action': 'response_action_executed',
                    'details': f"{action_type.value}: {details}"
                })
                
            logger.info(f"Executed response action {action_type.value} for incident {incident_id}")
            return result
            
        except Exception as e:
            result = ResponseActionResult(
                action_id=action_id,
                action_type=action_type,
                success=False,
                timestamp=datetime.now(),
                details=f"Action failed: {str(e)}",
                errors=[str(e)]
            )
            
            logger.error(f"Response action failed: {action_type.value} - {e}")
            return result
            
    def get_incidents(
        self,
        status: Optional[IncidentStatus] = None,
        severity: Optional[IncidentSeverity] = None,
        assigned_to: Optional[str] = None
    ) -> List[Incident]:
        """Get incidents with optional filtering"""
        
        incidents = list(self.incidents.values())
        
        if status:
            incidents = [i for i in incidents if i.status == status]
        if severity:
            incidents = [i for i in incidents if i.severity == severity]
        if assigned_to:
            incidents = [i for i in incidents if i.assigned_to == assigned_to]
            
        # Sort by creation time (newest first)
        incidents.sort(key=lambda x: x.created_time, reverse=True)
        
        return incidents
        
    def get_incident_metrics(self) -> Dict[str, Any]:
        """Get incident management metrics"""
        
        incidents = list(self.incidents.values())
        
        if not incidents:
            return {
                'total_incidents': 0,
                'open_incidents': 0,
                'mean_resolution_time': 0,
                'incidents_by_severity': {},
                'incidents_by_status': {}
            }
            
        # Calculate metrics
        total_incidents = len(incidents)
        open_incidents = len([i for i in incidents if i.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]])
        
        # Resolution time for closed incidents
        closed_incidents = [i for i in incidents if i.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]]
        resolution_times = []
        
        for incident in closed_incidents:
            resolution_time = (incident.updated_time - incident.created_time).total_seconds()
            resolution_times.append(resolution_time)
            
        mean_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
        
        # Distribution by severity
        severity_counts = {}
        for incident in incidents:
            severity = incident.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
        # Distribution by status
        status_counts = {}
        for incident in incidents:
            status = incident.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
        return {
            'total_incidents': total_incidents,
            'open_incidents': open_incidents,
            'closed_incidents': len(closed_incidents),
            'mean_resolution_time_hours': mean_resolution_time / 3600,
            'incidents_by_severity': severity_counts,
            'incidents_by_status': status_counts
        }
        
    def _should_create_incident(self, alert: SecurityAlert) -> Optional[Incident]:
        """Determine if alert should create incident"""
        
        # High/critical alerts always create incidents
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            return self.create_incident(
                title=f"{alert.rule_name} - {alert.source_ip or 'Unknown'}",
                description=alert.description,
                severity=IncidentSeverity.HIGH if alert.severity == AlertSeverity.HIGH else IncidentSeverity.CRITICAL,
                source_alerts=[alert]
            )
            
        # For medium alerts, check threshold
        alert_key = f"{alert.source_ip}_{alert.rule_name}"
        
        if alert_key not in self.pending_alerts:
            self.pending_alerts[alert_key] = []
            
        self.pending_alerts[alert_key].append(alert)
        
        if len(self.pending_alerts[alert_key]) >= self.alert_threshold:
            # Create incident from accumulated alerts
            incident = self.create_incident(
                title=f"Multiple alerts from {alert.source_ip or 'Unknown'}",
                description=f"Created from {len(self.pending_alerts[alert_key])} related alerts",
                severity=IncidentSeverity.MEDIUM,
                source_alerts=self.pending_alerts[alert_key]
            )
            
            # Clear pending alerts
            del self.pending_alerts[alert_key]
            
            return incident
            
        return None
        
    def _trigger_auto_response(self, incident: Incident) -> None:
        """Trigger automated response for incident"""
        
        # Determine appropriate playbook based on MITRE techniques
        playbook_name = self._select_playbook(incident)
        
        if playbook_name:
            self.playbook_engine.execute_playbook(playbook_name, incident, self)
            
            incident.timeline.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'auto_response_triggered',
                'details': f'Executed playbook: {playbook_name}'
            })
            
    def _select_playbook(self, incident: Incident) -> Optional[str]:
        """Select appropriate playbook for incident"""
        
        techniques = set(incident.mitre_techniques)
        
        # Malware-related techniques
        malware_techniques = {'T1055', 'T1003', 'T1059', 'T1071'}
        if techniques & malware_techniques:
            return 'malware_incident'
            
        # Data exfiltration techniques
        exfil_techniques = {'T1041', 'T1048', 'T1567'}
        if techniques & exfil_techniques:
            return 'data_exfiltration'
            
        # Credential-related techniques
        cred_techniques = {'T1110', 'T1078', 'T1558'}
        if techniques & cred_techniques:
            return 'credential_compromise'
            
        return None
        
    def _isolate_host(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """Isolate host from network"""
        host = parameters.get('host')
        
        if not host:
            return False, "No host specified"
            
        # Simulate host isolation
        logger.info(f"Isolating host: {host}")
        
        # In real implementation, would interface with network controls
        return True, f"Host {host} isolated successfully"
        
    def _block_ip(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """Block IP address"""
        ip = parameters.get('ip')
        
        if not ip:
            return False, "No IP specified"
            
        # Simulate IP blocking
        logger.info(f"Blocking IP: {ip}")
        
        # In real implementation, would interface with firewall/IPS
        return True, f"IP {ip} blocked successfully"
        
    def _disable_user(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """Disable user account"""
        username = parameters.get('username')
        
        if not username:
            return False, "No username specified"
            
        # Simulate user disabling
        logger.info(f"Disabling user: {username}")
        
        # In real implementation, would interface with identity systems
        return True, f"User {username} disabled successfully"
        
    def _collect_evidence(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """Collect digital evidence"""
        evidence_type = parameters.get('type', 'memory')
        systems = parameters.get('systems', [])
        
        # Simulate evidence collection
        logger.info(f"Collecting {evidence_type} evidence from {len(systems)} systems")
        
        # In real implementation, would trigger forensic tools
        return True, f"Collected {evidence_type} evidence from {len(systems)} systems"
        
    def _notify_team(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """Notify incident response team"""
        recipients = parameters.get('recipients', [])
        message = parameters.get('message', 'Incident notification')
        
        # Simulate notification
        logger.info(f"Notifying {len(recipients)} recipients: {message}")
        
        # In real implementation, would send emails/alerts
        return True, f"Notified {len(recipients)} recipients"


class IncidentResponse:
    """High-level incident response coordination"""
    
    def __init__(self, cyber_range):
        self.cyber_range = cyber_range
        self.incident_manager = IncidentManager(cyber_range)
        
    def handle_security_event(self, event_data: Dict[str, Any]) -> None:
        """Handle security event and coordinate response"""
        
        # Convert event to alert format
        alert = self._create_alert_from_event(event_data)
        
        # Process through incident manager
        incidents = self.incident_manager.process_alerts([alert])
        
        if incidents:
            logger.info(f"Created {len(incidents)} incidents from security event")
            
    def _create_alert_from_event(self, event_data: Dict[str, Any]) -> SecurityAlert:
        """Create SecurityAlert from event data"""
        
        return SecurityAlert(
            alert_id=f"evt_{int(time.time())}",
            timestamp=datetime.now(),
            severity=AlertSeverity.MEDIUM,
            source="Event Processor",
            rule_name=event_data.get('event_type', 'Unknown Event'),
            description=event_data.get('description', 'Security event detected'),
            source_ip=event_data.get('source_ip'),
            destination_ip=event_data.get('destination_ip'),
            username=event_data.get('username'),
            process_name=event_data.get('process_name')
        )