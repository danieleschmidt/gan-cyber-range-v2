"""
Attack execution engine for cyber range simulations.

This module provides the core attack execution infrastructure, including
attack orchestration, payload deployment, and attack simulation management.
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
import time

from .network_sim import NetworkTopology, Host

logger = logging.getLogger(__name__)


class AttackStatus(Enum):
    """Attack execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class AttackPhase(Enum):
    """MITRE ATT&CK kill chain phases"""
    RECONNAISSANCE = "reconnaissance"
    WEAPONIZATION = "weaponization"
    DELIVERY = "delivery"
    EXPLOITATION = "exploitation"
    INSTALLATION = "installation"
    COMMAND_CONTROL = "command_control"
    ACTIONS = "actions"


@dataclass
class AttackStep:
    """Individual step in an attack sequence"""
    step_id: str
    name: str
    phase: AttackPhase
    technique_id: str  # MITRE ATT&CK technique ID
    target_host: str
    payload: Dict[str, Any]
    duration: int = 30  # seconds
    success_probability: float = 0.8
    prerequisites: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class AttackResult:
    """Result of an attack execution"""
    attack_id: str
    step_id: str
    status: AttackStatus
    success: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    detection_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AttackCampaign:
    """Multi-step attack campaign"""
    campaign_id: str
    name: str
    description: str
    steps: List[AttackStep]
    target_profile: Dict[str, Any]
    objectives: List[str]
    duration: timedelta
    stealth_level: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)


class AttackEngine:
    """Core attack execution engine"""
    
    def __init__(self, cyber_range):
        self.cyber_range = cyber_range
        self.active_attacks = {}
        self.attack_history = []
        self.attack_templates = {}
        self.execution_threads = {}
        
        # Load default attack templates
        self._load_default_templates()
        
        logger.info("Attack engine initialized")
    
    def execute_attack(self, attack_config: Dict[str, Any]) -> str:
        """Execute a single attack"""
        attack_id = str(uuid.uuid4())
        
        logger.info(f"Executing attack {attack_id}")
        
        # Create attack step from config
        attack_step = self._create_attack_step(attack_config)
        
        # Start execution in background thread
        thread = threading.Thread(
            target=self._execute_attack_step,
            args=(attack_id, attack_step),
            daemon=True
        )
        
        self.execution_threads[attack_id] = thread
        thread.start()
        
        return attack_id
    
    def execute_campaign(self, campaign: AttackCampaign) -> str:
        """Execute a multi-step attack campaign"""
        logger.info(f"Executing campaign: {campaign.name}")
        
        # Start campaign execution in background
        thread = threading.Thread(
            target=self._execute_campaign,
            args=(campaign,),
            daemon=True
        )
        
        self.execution_threads[campaign.campaign_id] = thread
        thread.start()
        
        return campaign.campaign_id
    
    def stop_attack(self, attack_id: str) -> bool:
        """Stop a running attack"""
        if attack_id in self.active_attacks:
            self.active_attacks[attack_id].status = AttackStatus.STOPPED
            logger.info(f"Stopped attack {attack_id}")
            return True
        return False
    
    def stop_all_attacks(self) -> int:
        """Stop all running attacks"""
        stopped_count = 0
        
        for attack_id in list(self.active_attacks.keys()):
            if self.stop_attack(attack_id):
                stopped_count += 1
        
        logger.info(f"Stopped {stopped_count} attacks")
        return stopped_count
    
    def get_attack_status(self, attack_id: str) -> Optional[AttackResult]:
        """Get status of an attack"""
        return self.active_attacks.get(attack_id)
    
    def get_active_attacks(self) -> List[AttackResult]:
        """Get all active attacks"""
        return list(self.active_attacks.values())
    
    def get_attack_history(self) -> List[AttackResult]:
        """Get attack execution history"""
        return self.attack_history.copy()
    
    def register_attack_template(self, name: str, template: Dict[str, Any]) -> None:
        """Register a new attack template"""
        self.attack_templates[name] = template
        logger.info(f"Registered attack template: {name}")
    
    def _create_attack_step(self, config: Dict[str, Any]) -> AttackStep:
        """Create an attack step from configuration"""
        
        # Use template if specified
        if 'template' in config:
            template = self.attack_templates.get(config['template'], {})
            # Merge template with config
            merged_config = {**template, **config}
        else:
            merged_config = config
        
        return AttackStep(
            step_id=str(uuid.uuid4()),
            name=merged_config.get('name', 'Unknown Attack'),
            phase=AttackPhase(merged_config.get('phase', 'exploitation')),
            technique_id=merged_config.get('technique_id', 'T1059'),
            target_host=merged_config.get('target_host', 'localhost'),
            payload=merged_config.get('payload', {}),
            duration=merged_config.get('duration', 30),
            success_probability=merged_config.get('success_probability', 0.8)
        )
    
    def _execute_attack_step(self, attack_id: str, step: AttackStep) -> None:
        """Execute a single attack step"""
        
        # Create attack result
        result = AttackResult(
            attack_id=attack_id,
            step_id=step.step_id,
            status=AttackStatus.RUNNING,
            success=False,
            start_time=datetime.now()
        )
        
        self.active_attacks[attack_id] = result
        
        try:
            logger.info(f"Executing attack step: {step.name} on {step.target_host}")
            
            # Validate target exists
            target_host = self._find_target_host(step.target_host)
            if not target_host:
                raise ValueError(f"Target host not found: {step.target_host}")
            
            # Execute attack technique
            success = self._execute_technique(step, target_host)
            
            # Simulate execution time
            time.sleep(min(step.duration, 5))  # Cap at 5 seconds for simulation
            
            # Update result
            result.success = success
            result.status = AttackStatus.COMPLETED if success else AttackStatus.FAILED
            result.end_time = datetime.now()
            
            # Generate artifacts
            result.artifacts = self._generate_attack_artifacts(step, target_host, success)
            
            # Trigger detection events if attack was successful
            if success:
                detection_events = self._generate_detection_events(step, target_host)
                result.detection_events = detection_events
                
                # Notify cyber range of detection events
                for event in detection_events:
                    self.cyber_range.trigger_event('detection', event)
            
            logger.info(f"Attack step completed: {step.name} - Success: {success}")
            
        except Exception as e:
            result.status = AttackStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            logger.error(f"Attack step failed: {step.name} - Error: {e}")
        
        finally:
            # Move to history and remove from active
            self.attack_history.append(result)
            if attack_id in self.active_attacks:
                del self.active_attacks[attack_id]
            if attack_id in self.execution_threads:
                del self.execution_threads[attack_id]
    
    def _execute_campaign(self, campaign: AttackCampaign) -> None:
        """Execute a multi-step attack campaign"""
        
        logger.info(f"Starting campaign execution: {campaign.name}")
        
        successful_steps = set()
        
        for step in campaign.steps:
            # Check if prerequisites are met
            if step.prerequisites and not all(prereq in successful_steps for prereq in step.prerequisites):
                logger.warning(f"Skipping step {step.name} - prerequisites not met")
                continue
            
            # Execute step
            step_result = self._execute_attack_step_sync(step)
            
            if step_result.success:
                successful_steps.add(step.step_id)
                logger.info(f"Campaign step successful: {step.name}")
            else:
                logger.warning(f"Campaign step failed: {step.name}")
                
                # Stop campaign if critical step fails
                if step.technique_id in ['T1078', 'T1190']:  # Valid Accounts, Exploit Public-Facing Application
                    logger.info(f"Critical step failed, stopping campaign: {campaign.name}")
                    break
            
            # Add delay between steps for realism
            time.sleep(random.uniform(5, 30))
        
        logger.info(f"Campaign execution completed: {campaign.name}")
    
    def _execute_attack_step_sync(self, step: AttackStep) -> AttackResult:
        """Execute attack step synchronously"""
        
        result = AttackResult(
            attack_id=str(uuid.uuid4()),
            step_id=step.step_id,
            status=AttackStatus.RUNNING,
            success=False,
            start_time=datetime.now()
        )
        
        try:
            target_host = self._find_target_host(step.target_host)
            if target_host:
                success = self._execute_technique(step, target_host)
                result.success = success
                result.status = AttackStatus.COMPLETED if success else AttackStatus.FAILED
                
                if success:
                    result.artifacts = self._generate_attack_artifacts(step, target_host, success)
                    result.detection_events = self._generate_detection_events(step, target_host)
            else:
                result.status = AttackStatus.FAILED
                result.error_message = f"Target host not found: {step.target_host}"
                
        except Exception as e:
            result.status = AttackStatus.FAILED
            result.error_message = str(e)
        
        result.end_time = datetime.now()
        return result
    
    def _find_target_host(self, target_identifier: str) -> Optional[Host]:
        """Find target host by name or IP"""
        if not self.cyber_range.topology:
            return None
            
        # Search by name
        for host in self.cyber_range.topology.hosts:
            if host.name == target_identifier or host.ip_address == target_identifier:
                return host
        
        return None
    
    def _execute_technique(self, step: AttackStep, target_host: Host) -> bool:
        """Execute a specific attack technique"""
        
        technique_implementations = {
            'T1059': self._execute_command_line,
            'T1078': self._execute_valid_accounts,
            'T1190': self._execute_exploit_public_app,
            'T1021': self._execute_remote_services,
            'T1110': self._execute_brute_force,
            'T1046': self._execute_network_scan,
            'T1083': self._execute_file_discovery,
            'T1082': self._execute_system_info_discovery
        }
        
        # Get implementation function
        impl_func = technique_implementations.get(step.technique_id, self._execute_generic)
        
        # Calculate success based on probability and target defenses
        base_probability = step.success_probability
        
        # Adjust probability based on target security level
        security_modifier = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.7
        }.get(target_host.security_level, 1.0)
        
        adjusted_probability = min(1.0, base_probability * security_modifier)
        
        # Execute technique
        success = impl_func(step, target_host)
        
        # Apply probability
        import random
        return success and (random.random() < adjusted_probability)
    
    def _execute_command_line(self, step: AttackStep, target_host: Host) -> bool:
        """Execute command line interface technique (T1059)"""
        logger.debug(f"Executing command line attack on {target_host.name}")
        
        payload = step.payload
        command = payload.get('command', 'whoami')
        
        # Simulate command execution
        return True
    
    def _execute_valid_accounts(self, step: AttackStep, target_host: Host) -> bool:
        """Execute valid accounts technique (T1078)"""
        logger.debug(f"Executing valid accounts attack on {target_host.name}")
        
        payload = step.payload
        username = payload.get('username', 'admin')
        password = payload.get('password', 'password123')
        
        # Simulate credential validation
        return True
    
    def _execute_exploit_public_app(self, step: AttackStep, target_host: Host) -> bool:
        """Execute exploit public-facing application (T1190)"""
        logger.debug(f"Executing public application exploit on {target_host.name}")
        
        payload = step.payload
        exploit_type = payload.get('exploit_type', 'web_app')
        
        # Check if target has vulnerable services
        vulnerable_services = ['web', 'email', 'ftp']
        has_vulnerable_service = any(service in target_host.services for service in vulnerable_services)
        
        return has_vulnerable_service
    
    def _execute_remote_services(self, step: AttackStep, target_host: Host) -> bool:
        """Execute remote services technique (T1021)"""
        logger.debug(f"Executing remote services attack on {target_host.name}")
        
        # Check if target has remote services
        remote_services = ['ssh', 'rdp', 'telnet', 'vnc']
        has_remote_service = any(service in target_host.services for service in remote_services)
        
        return has_remote_service
    
    def _execute_brute_force(self, step: AttackStep, target_host: Host) -> bool:
        """Execute brute force technique (T1110)"""
        logger.debug(f"Executing brute force attack on {target_host.name}")
        
        # Simulate brute force attempt
        return target_host.security_level != 'high'
    
    def _execute_network_scan(self, step: AttackStep, target_host: Host) -> bool:
        """Execute network service scanning (T1046)"""
        logger.debug(f"Executing network scan on {target_host.name}")
        
        # Network scanning typically succeeds unless blocked by firewall
        return True
    
    def _execute_file_discovery(self, step: AttackStep, target_host: Host) -> bool:
        """Execute file and directory discovery (T1083)"""
        logger.debug(f"Executing file discovery on {target_host.name}")
        
        # File discovery typically succeeds if we have access
        return True
    
    def _execute_system_info_discovery(self, step: AttackStep, target_host: Host) -> bool:
        """Execute system information discovery (T1082)"""
        logger.debug(f"Executing system info discovery on {target_host.name}")
        
        # System info discovery typically succeeds
        return True
    
    def _execute_generic(self, step: AttackStep, target_host: Host) -> bool:
        """Execute generic attack technique"""
        logger.debug(f"Executing generic technique {step.technique_id} on {target_host.name}")
        
        # Generic implementation with moderate success rate
        return True
    
    def _generate_attack_artifacts(self, step: AttackStep, target_host: Host, success: bool) -> Dict[str, Any]:
        """Generate attack artifacts for forensics"""
        
        artifacts = {
            'technique_id': step.technique_id,
            'target_host': target_host.name,
            'target_ip': target_host.ip_address,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'phase': step.phase.value
        }
        
        # Add technique-specific artifacts
        if step.technique_id == 'T1059':  # Command Line
            artifacts['command_executed'] = step.payload.get('command', 'unknown')
            artifacts['process_id'] = random.randint(1000, 9999)
            
        elif step.technique_id == 'T1078':  # Valid Accounts
            artifacts['username'] = step.payload.get('username', 'unknown')
            artifacts['logon_type'] = 'interactive'
            
        elif step.technique_id == 'T1190':  # Exploit Public App
            artifacts['exploit_type'] = step.payload.get('exploit_type', 'web_app')
            artifacts['user_agent'] = 'Mozilla/5.0 (Attack Tool)'
            
        return artifacts
    
    def _generate_detection_events(self, step: AttackStep, target_host: Host) -> List[Dict[str, Any]]:
        """Generate detection events for security tools"""
        
        events = []
        
        # Base detection event
        base_event = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'source_host': target_host.name,
            'source_ip': target_host.ip_address,
            'technique_id': step.technique_id,
            'phase': step.phase.value,
            'severity': self._get_technique_severity(step.technique_id)
        }
        
        # Technique-specific detection signatures
        if step.technique_id == 'T1059':  # Command Line
            events.append({
                **base_event,
                'detection_type': 'process_creation',
                'process_name': 'cmd.exe',
                'command_line': step.payload.get('command', ''),
                'confidence': 0.8
            })
            
        elif step.technique_id == 'T1078':  # Valid Accounts
            events.append({
                **base_event,
                'detection_type': 'authentication',
                'username': step.payload.get('username', ''),
                'logon_result': 'success',
                'confidence': 0.6
            })
            
        elif step.technique_id == 'T1190':  # Exploit Public App
            events.append({
                **base_event,
                'detection_type': 'web_attack',
                'request_uri': '/exploit/path',
                'http_method': 'POST',
                'confidence': 0.9
            })
            
        elif step.technique_id == 'T1046':  # Network Scan
            events.append({
                **base_event,
                'detection_type': 'network_scan',
                'ports_scanned': [80, 443, 22, 3389],
                'scan_type': 'tcp_syn',
                'confidence': 0.7
            })
        
        return events
    
    def _get_technique_severity(self, technique_id: str) -> str:
        """Get severity level for MITRE ATT&CK technique"""
        
        high_severity = ['T1078', 'T1190', 'T1055', 'T1003']  # Valid Accounts, Exploit Public App, etc.
        medium_severity = ['T1059', 'T1021', 'T1110']  # Command Line, Remote Services, Brute Force
        
        if technique_id in high_severity:
            return 'high'
        elif technique_id in medium_severity:
            return 'medium'
        else:
            return 'low'
    
    def _load_default_templates(self) -> None:
        """Load default attack templates"""
        
        templates = {
            'web_exploit': {
                'name': 'Web Application Exploit',
                'phase': 'exploitation',
                'technique_id': 'T1190',
                'payload': {
                    'exploit_type': 'sql_injection',
                    'target_url': '/login.php',
                    'payload_data': "' OR 1=1 --"
                },
                'success_probability': 0.7
            },
            
            'credential_dump': {
                'name': 'Credential Dumping',
                'phase': 'actions',
                'technique_id': 'T1003',
                'payload': {
                    'dump_type': 'lsass',
                    'tool': 'mimikatz'
                },
                'success_probability': 0.8
            },
            
            'lateral_movement': {
                'name': 'Lateral Movement via RDP',
                'phase': 'actions',
                'technique_id': 'T1021',
                'payload': {
                    'service': 'rdp',
                    'credentials': 'admin:password123'
                },
                'success_probability': 0.6
            },
            
            'reconnaissance': {
                'name': 'Network Discovery',
                'phase': 'reconnaissance',
                'technique_id': 'T1046',
                'payload': {
                    'scan_type': 'port_scan',
                    'ports': [22, 80, 443, 3389]
                },
                'success_probability': 0.9
            }
        }
        
        for name, template in templates.items():
            self.register_attack_template(name, template)


class AttackSimulator:
    """High-level attack simulation interface"""
    
    def __init__(self, cyber_range):
        self.cyber_range = cyber_range
        self.attack_engine = cyber_range.attack_engine
        
    def execute_campaign(
        self,
        campaign: AttackCampaign,
        blue_team_enabled: bool = True,
        record_pcaps: bool = False,
        generate_logs: bool = True
    ) -> str:
        """Execute attack campaign with monitoring options"""
        
        logger.info(f"Starting attack simulation: {campaign.name}")
        
        # Configure monitoring
        if record_pcaps:
            self._start_packet_capture()
            
        if generate_logs:
            self._configure_logging()
        
        # Execute the campaign
        campaign_id = self.attack_engine.execute_campaign(campaign)
        
        return campaign_id
    
    def _start_packet_capture(self) -> None:
        """Start packet capture for attack analysis"""
        logger.info("Starting packet capture")
        # Implementation would start tcpdump or similar
        
    def _configure_logging(self) -> None:
        """Configure attack logging"""
        logger.info("Configuring attack logging")
        # Implementation would configure centralized logging