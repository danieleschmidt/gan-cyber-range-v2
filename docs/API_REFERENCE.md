# GAN-Cyber-Range-v2 API Reference

## Table of Contents

1. [Core API](#core-api)
2. [Attack Generation API](#attack-generation-api)
3. [Blue Team API](#blue-team-api)
4. [Network Simulation API](#network-simulation-api)
5. [Security Framework API](#security-framework-api)
6. [Orchestration API](#orchestration-api)
7. [REST API Endpoints](#rest-api-endpoints)
8. [WebSocket API](#websocket-api)
9. [Configuration API](#configuration-api)
10. [Examples](#examples)

## Core API

### CyberRange Class

The main orchestration class for the platform.

```python
from gan_cyber_range import CyberRange

class CyberRange:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cyber range platform.
        
        Args:
            config: Configuration dictionary for the platform
        """
    
    def start(self, mode: str = "development") -> None:
        """
        Start the cyber range platform.
        
        Args:
            mode: Operation mode ("development", "production", "testing")
        """
    
    def stop(self) -> None:
        """Stop the cyber range platform."""
    
    def create_network_topology(self, name: str) -> NetworkTopology:
        """
        Create a new network topology for scenarios.
        
        Args:
            name: Unique name for the topology
            
        Returns:
            NetworkTopology instance
        """
    
    def deploy_blue_team_defenses(self, config: Dict[str, Any]) -> DefenseSuite:
        """
        Deploy blue team defensive tools.
        
        Args:
            config: Defense configuration dictionary
            
        Returns:
            DefenseSuite instance
        """
    
    def run_training_scenario(
        self,
        name: str,
        attacks: List[AttackVector],
        duration: str = "1h",
        evaluation_enabled: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a training scenario.
        
        Args:
            name: Scenario name
            attacks: List of attack vectors to execute
            duration: Scenario duration (e.g., "30m", "2h")
            evaluation_enabled: Enable blue team evaluation
            
        Returns:
            Scenario execution results
        """
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get platform performance metrics.
        
        Returns:
            Dictionary containing platform metrics
        """
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current platform status.
        
        Returns:
            Platform status information
        """
```

## Attack Generation API

### AttackGAN Base Class

```python
from gan_cyber_range.generators.base_gan import AttackGAN

class AttackGAN:
    def __init__(self, attack_type: str):
        """
        Initialize attack GAN.
        
        Args:
            attack_type: Type of attacks to generate
        """
    
    def train(
        self, 
        training_data: List[Dict[str, Any]], 
        epochs: int = 1000,
        batch_size: int = 32,
        learning_rate: float = 0.0002
    ) -> Dict[str, float]:
        """
        Train the GAN model.
        
        Args:
            training_data: Training dataset
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            
        Returns:
            Training metrics (loss, accuracy, etc.)
        """
    
    def generate_attacks(
        self, 
        count: int = 100,
        diversity_factor: float = 0.8,
        target_systems: Optional[List[str]] = None
    ) -> List[AttackVector]:
        """
        Generate attack vectors using the trained model.
        
        Args:
            count: Number of attacks to generate
            diversity_factor: Diversity in generated attacks (0.0-1.0)
            target_systems: Target systems for attacks
            
        Returns:
            List of generated attack vectors
        """
    
    def save_model(self, path: str) -> None:
        """Save trained model to file."""
    
    def load_model(self, path: str) -> None:
        """Load trained model from file."""
```

### Specialized GAN Classes

#### MalwareGAN

```python
from gan_cyber_range.generators.malware_gan import MalwareGAN

malware_gan = MalwareGAN()

# Generate malware samples
malware_samples = malware_gan.generate_malware_samples(
    count=50,
    malware_families=["trojan", "ransomware", "backdoor"],
    target_platforms=["windows", "linux"],
    evasion_techniques=["polymorphism", "encryption"]
)
```

#### NetworkGAN

```python
from gan_cyber_range.generators.network_gan import NetworkGAN

network_gan = NetworkGAN()

# Generate network attack patterns
network_attacks = network_gan.generate_attack_patterns(
    count=30,
    attack_types=["port_scan", "ddos", "lateral_movement"],
    target_networks=["192.168.1.0/24"],
    stealth_level=0.7
)
```

#### WebGAN

```python
from gan_cyber_range.generators.web_gan import WebGAN

web_gan = WebGAN()

# Generate web attack payloads
web_attacks = web_gan.generate_web_attacks(
    count=25,
    attack_vectors=["sql_injection", "xss", "csrf"],
    target_technologies=["php", "nodejs", "python"],
    complexity_level="intermediate"
)
```

#### SocialGAN

```python
from gan_cyber_range.generators.social_gan import SocialGAN

social_gan = SocialGAN()

# Generate social engineering campaigns
social_campaigns = social_gan.generate_campaigns(
    count=10,
    campaign_types=["phishing", "pretexting", "baiting"],
    target_demographics=["employees", "executives"],
    sophistication="high"
)
```

## Blue Team API

### DefenseSuite Class

```python
from gan_cyber_range.blue_team.defense_suite import DefenseSuite

class DefenseSuite:
    def __init__(self, cyber_range):
        """Initialize defense suite."""
    
    def deploy_defenses(self, config: Dict[str, Any]) -> None:
        """
        Deploy defensive tools.
        
        Args:
            config: Defense deployment configuration
        """
    
    def process_event(self, event_data: Dict[str, Any]) -> List[SecurityAlert]:
        """
        Process security event through all defensive tools.
        
        Args:
            event_data: Security event data
            
        Returns:
            List of generated security alerts
        """
    
    def get_alerts(
        self, 
        severity: Optional[AlertSeverity] = None,
        time_range: Optional[timedelta] = None
    ) -> List[SecurityAlert]:
        """
        Retrieve security alerts with optional filtering.
        
        Args:
            severity: Filter by alert severity
            time_range: Time range for alerts
            
        Returns:
            Filtered list of security alerts
        """
    
    def get_metrics(self) -> DefenseMetrics:
        """
        Get defense performance metrics.
        
        Returns:
            Defense performance metrics
        """
    
    def add_custom_rule(
        self, 
        engine: str, 
        rule_id: str, 
        rule: Dict[str, Any]
    ) -> None:
        """
        Add custom detection rule.
        
        Args:
            engine: Target engine ("siem", "ids", "edr")
            rule_id: Unique rule identifier
            rule: Rule configuration
        """
```

### SecurityAlert Class

```python
from gan_cyber_range.blue_team.defense_suite import SecurityAlert, AlertSeverity

@dataclass
class SecurityAlert:
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    source: str
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
```

### Incident Response API

```python
from gan_cyber_range.blue_team.incident_response import IncidentManager

class IncidentManager:
    def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        alerts: List[SecurityAlert]
    ) -> Incident:
        """Create new incident from alerts."""
    
    def process_alerts(self, alerts: List[SecurityAlert]) -> List[Incident]:
        """Process alerts and create incidents."""
    
    def execute_response_action(
        self, 
        action_type: ResponseAction
    ) -> ResponseActionResult:
        """Execute incident response action."""
    
    def get_active_incidents(self) -> List[Incident]:
        """Get all active incidents."""
```

## Network Simulation API

### NetworkTopology Class

```python
from gan_cyber_range.core.network_sim import NetworkTopology, Host, Subnet

class NetworkTopology:
    def __init__(self, name: str):
        """Initialize network topology."""
    
    def add_subnet(
        self,
        name: str,
        cidr: str,
        security_zone: str = "internal",
        vlan_id: Optional[int] = None
    ) -> Subnet:
        """
        Add subnet to topology.
        
        Args:
            name: Subnet name
            cidr: CIDR notation (e.g., "192.168.1.0/24")
            security_zone: Security zone classification
            vlan_id: VLAN identifier
            
        Returns:
            Created subnet instance
        """
    
    def add_host(
        self,
        name: str,
        subnet_name: str,
        host_type: HostType,
        os_type: OSType,
        services: List[str],
        vulnerabilities: Optional[List[str]] = None
    ) -> Host:
        """
        Add host to topology.
        
        Args:
            name: Host name
            subnet_name: Target subnet name
            host_type: Type of host (server, workstation, etc.)
            os_type: Operating system type
            services: Running services
            vulnerabilities: Known vulnerabilities
            
        Returns:
            Created host instance
        """
    
    def create_firewall_rule(
        self,
        name: str,
        source: str,
        destination: str,
        action: str,
        ports: Optional[List[int]] = None
    ) -> FirewallRule:
        """Create firewall rule between network segments."""
    
    def deploy_to_containers(self) -> Dict[str, str]:
        """Deploy topology to Docker containers."""
```

## Security Framework API

### EthicalFramework Class

```python
from gan_cyber_range.utils.enhanced_security import EthicalFramework

class EthicalFramework:
    def is_compliant(self, request: Dict[str, Any]) -> bool:
        """
        Check if request complies with ethical guidelines.
        
        Args:
            request: Request to validate
            
        Returns:
            True if compliant, False otherwise
        """
    
    def validate_target(self, target: str) -> bool:
        """Validate attack target is appropriate."""
    
    def check_authorization(self, user_id: str, action: str) -> bool:
        """Check user authorization for action."""
```

### Input Validation API

```python
from gan_cyber_range.utils.enhanced_security import validate_input, secure_hash

def validate_input(
    input_value: str, 
    pattern: str, 
    max_length: int = 500
) -> bool:
    """
    Validate user input against security patterns.
    
    Args:
        input_value: Input to validate
        pattern: Regex pattern for validation
        max_length: Maximum allowed length
        
    Returns:
        True if valid, False if dangerous
    """

def secure_hash(data: str, salt: Optional[str] = None) -> str:
    """
    Generate secure hash with salt.
    
    Args:
        data: Data to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Salted hash string
    """
```

## Orchestration API

### WorkflowEngine Class

```python
from gan_cyber_range.orchestration.workflow_engine import WorkflowEngine, Workflow

class WorkflowEngine:
    def create_workflow(
        self,
        workflow_id: str,
        name: str,
        steps: List[WorkflowStep]
    ) -> Workflow:
        """Create new workflow."""
    
    def execute_workflow(
        self, 
        workflow_id: str,
        context: Dict[str, Any]
    ) -> WorkflowExecution:
        """Execute workflow with context."""
    
    def get_execution_status(self, execution_id: str) -> ExecutionStatus:
        """Get workflow execution status."""
```

### ScenarioOrchestrator Class

```python
from gan_cyber_range.orchestration.scenario_orchestrator import ScenarioOrchestrator

class ScenarioOrchestrator:
    def create_scenario(
        self,
        name: str,
        attack_sequence: List[AttackVector],
        blue_team_config: Dict[str, Any],
        network_topology: NetworkTopology
    ) -> TrainingScenario:
        """Create training scenario."""
    
    def execute_scenario(
        self, 
        scenario_id: str,
        participants: List[str]
    ) -> ScenarioExecution:
        """Execute training scenario."""
    
    def get_scenario_results(self, execution_id: str) -> ScenarioResults:
        """Get scenario execution results."""
```

## REST API Endpoints

### Authentication

```bash
# Login
POST /api/v1/auth/login
Content-Type: application/json

{
    "username": "admin",
    "password": "secure_password"
}

# Response
{
    "access_token": "jwt_token_here",
    "token_type": "bearer",
    "expires_in": 3600
}
```

### Platform Management

```bash
# Get platform status
GET /api/v1/status
Authorization: Bearer {token}

# Start platform
POST /api/v1/start
Authorization: Bearer {token}
Content-Type: application/json

{
    "mode": "production",
    "config": {}
}

# Stop platform
POST /api/v1/stop
Authorization: Bearer {token}
```

### Attack Generation

```bash
# Generate attacks
POST /api/v1/attacks/generate
Authorization: Bearer {token}
Content-Type: application/json

{
    "attack_type": "web_attacks",
    "count": 10,
    "target_systems": ["web_server_1"],
    "complexity": "intermediate"
}

# Get attack templates
GET /api/v1/attacks/templates
Authorization: Bearer {token}

# Get generated attacks
GET /api/v1/attacks?type=malware&limit=50
Authorization: Bearer {token}
```

### Network Topology

```bash
# Create topology
POST /api/v1/topology
Authorization: Bearer {token}
Content-Type: application/json

{
    "name": "corporate_network",
    "subnets": [
        {
            "name": "dmz",
            "cidr": "10.0.1.0/24",
            "security_zone": "dmz"
        }
    ],
    "hosts": [
        {
            "name": "web_server",
            "subnet": "dmz",
            "type": "server",
            "os": "linux",
            "services": ["http", "https"]
        }
    ]
}

# Get topology
GET /api/v1/topology/{topology_id}
Authorization: Bearer {token}

# Deploy topology
POST /api/v1/topology/{topology_id}/deploy
Authorization: Bearer {token}
```

### Blue Team Operations

```bash
# Deploy defenses
POST /api/v1/blue_team/deploy
Authorization: Bearer {token}
Content-Type: application/json

{
    "siem": true,
    "ids": true,
    "edr": true,
    "custom_rules": [
        {
            "engine": "siem",
            "rule_id": "custom_rule_1",
            "rule": {...}
        }
    ]
}

# Get alerts
GET /api/v1/blue_team/alerts?severity=high&limit=100
Authorization: Bearer {token}

# Get metrics
GET /api/v1/blue_team/metrics
Authorization: Bearer {token}
```

### Training Scenarios

```bash
# Create scenario
POST /api/v1/scenarios
Authorization: Bearer {token}
Content-Type: application/json

{
    "name": "Web Security Training",
    "type": "web_attacks",
    "duration": "1h",
    "difficulty": "intermediate",
    "participants": 5
}

# Execute scenario
POST /api/v1/scenarios/{scenario_id}/execute
Authorization: Bearer {token}

# Get scenario results
GET /api/v1/scenarios/{scenario_id}/results
Authorization: Bearer {token}
```

## WebSocket API

### Real-time Updates

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8080/ws');

// Authentication
ws.send(JSON.stringify({
    type: 'auth',
    token: 'jwt_token_here'
}));

// Subscribe to events
ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['alerts', 'attacks', 'metrics']
}));

// Handle messages
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'alert':
            handleSecurityAlert(data.payload);
            break;
        case 'attack_executed':
            handleAttackExecution(data.payload);
            break;
        case 'metrics_update':
            updateMetricsDashboard(data.payload);
            break;
    }
};
```

### Event Types

- `alert`: New security alert generated
- `attack_executed`: Attack vector executed
- `attack_detected`: Attack detected by defenses
- `incident_created`: New incident created
- `metrics_update`: Performance metrics updated
- `scenario_started`: Training scenario started
- `scenario_completed`: Training scenario completed

## Configuration API

### Platform Configuration

```python
from gan_cyber_range.config import CyberRangeConfig

config = CyberRangeConfig({
    "platform": {
        "name": "Corporate Training Range",
        "mode": "production",
        "max_concurrent_scenarios": 10
    },
    "security": {
        "ethical_framework": True,
        "containment_enabled": True,
        "max_attack_intensity": "high"
    },
    "performance": {
        "auto_scaling": True,
        "max_workers": 20,
        "cache_size": "1GB"
    },
    "monitoring": {
        "prometheus_enabled": True,
        "log_level": "INFO",
        "metrics_retention": "30d"
    }
})
```

### Attack Generator Configuration

```yaml
# attack_config.yaml
malware_gan:
  model_path: "models/malware_gan.pkl"
  training_epochs: 1000
  batch_size: 32
  learning_rate: 0.0002
  
network_gan:
  model_path: "models/network_gan.pkl"
  signature_database: "signatures/network.db"
  stealth_techniques: ["fragmentation", "timing_evasion"]

web_gan:
  model_path: "models/web_gan.pkl"
  payload_database: "payloads/web_attacks.db"
  target_technologies: ["php", "python", "nodejs", "java"]
```

### Defense Configuration

```yaml
# defense_config.yaml
siem:
  enabled: true
  log_sources: ["syslog", "json", "csv"]
  correlation_rules: "rules/siem_rules.yaml"
  alert_threshold: "medium"

ids:
  enabled: true
  signature_files: ["emerging-threats.rules", "custom.rules"]
  anomaly_detection: true
  interface: "eth0"

edr:
  enabled: true
  process_monitoring: true
  file_monitoring: true
  network_monitoring: true
  behavioral_analysis: true
```

## Examples

### Complete Training Scenario

```python
#!/usr/bin/env python3
"""
Complete example of creating and running a training scenario.
"""

from gan_cyber_range import CyberRange
from gan_cyber_range.factories import AttackFactory, NetworkFactory
from gan_cyber_range.blue_team.defense_suite import DefenseSuite

def main():
    # Initialize cyber range
    cyber_range = CyberRange({
        "mode": "training",
        "security": {"ethical_framework": True}
    })
    
    # Create network topology
    topology = cyber_range.create_network_topology("corporate_network")
    
    # Add network segments
    dmz = topology.add_subnet("dmz", "10.0.1.0/24", "dmz")
    internal = topology.add_subnet("internal", "192.168.1.0/24", "internal")
    
    # Add hosts
    web_server = topology.add_host(
        "web_server", "dmz", "server", "linux",
        services=["http", "https", "ssh"],
        vulnerabilities=["CVE-2021-44228"]  # Log4j
    )
    
    db_server = topology.add_host(
        "db_server", "internal", "server", "linux",
        services=["mysql", "ssh"],
        vulnerabilities=["CVE-2020-14343"]  # MySQL
    )
    
    workstation = topology.add_host(
        "workstation_1", "internal", "workstation", "windows",
        services=["rdp", "smb"]
    )
    
    # Deploy topology
    container_mapping = topology.deploy_to_containers()
    
    # Generate attacks
    attack_factory = AttackFactory()
    
    # Web attacks targeting the web server
    web_attacks = attack_factory.generate_web_attacks(
        count=15,
        target_hosts=["web_server"],
        attack_types=["sql_injection", "xss", "csrf", "lfi"],
        complexity="intermediate"
    )
    
    # Network attacks for lateral movement
    network_attacks = attack_factory.generate_network_attacks(
        count=10,
        attack_types=["port_scan", "lateral_movement", "credential_dumping"],
        source_hosts=["web_server"],
        target_hosts=["db_server", "workstation_1"]
    )
    
    # Social engineering attacks
    social_attacks = attack_factory.generate_social_attacks(
        count=5,
        campaign_types=["phishing", "pretexting"],
        target_users=["employee_1", "admin_user"]
    )
    
    # Combine all attacks
    all_attacks = web_attacks + network_attacks + social_attacks
    
    # Deploy blue team defenses
    defense_config = {
        "siem": True,
        "ids": True,
        "edr": True,
        "custom_rules": [
            {
                "engine": "siem",
                "rule_id": "web_attack_detection",
                "rule": {
                    "name": "Web Attack Detection",
                    "conditions": {
                        "event_type": "http_request",
                        "suspicious_payload": True
                    },
                    "severity": "high"
                }
            }
        ]
    }
    
    defense_suite = cyber_range.deploy_blue_team_defenses(defense_config)
    
    # Execute training scenario
    scenario_results = cyber_range.run_training_scenario(
        name="Advanced Persistent Threat Simulation",
        attacks=all_attacks,
        duration="2h",
        evaluation_enabled=True,
        participants=["blue_team_1", "blue_team_2"],
        scenario_config={
            "attack_delay": "30s",  # Delay between attacks
            "escalation_enabled": True,
            "lateral_movement": True,
            "data_exfiltration": True
        }
    )
    
    # Print results
    print("=== Training Scenario Results ===")
    print(f"Scenario ID: {scenario_results['scenario_id']}")
    print(f"Duration: {scenario_results['actual_duration']}")
    print(f"Total Attacks: {scenario_results['total_attacks']}")
    print(f"Successful Attacks: {scenario_results['successful_attacks']}")
    print(f"Detected Attacks: {scenario_results['detected_attacks']}")
    print(f"Detection Rate: {scenario_results['detection_rate']:.2%}")
    print(f"Mean Time to Detect: {scenario_results['mttd']:.1f}s")
    print(f"Mean Time to Respond: {scenario_results['mttr']:.1f}s")
    print(f"Overall Score: {scenario_results['score']}/100")
    
    # Get detailed metrics
    defense_metrics = defense_suite.get_metrics()
    alerts = defense_suite.get_alerts()
    
    print(f"\n=== Defense Metrics ===")
    print(f"Alerts Generated: {len(alerts)}")
    print(f"Critical Alerts: {len([a for a in alerts if a.severity.value == 'critical'])}")
    print(f"False Positive Rate: {defense_metrics.false_positive_rate:.2%}")
    print(f"Coverage Score: {defense_metrics.coverage_score:.2f}")
    
    # Generate evaluation report
    report = cyber_range.generate_evaluation_report(scenario_results)
    with open("scenario_report.html", "w") as f:
        f.write(report)
    
    print("\nDetailed report saved to: scenario_report.html")

if __name__ == "__main__":
    main()
```

### Custom Attack Generator

```python
#!/usr/bin/env python3
"""
Example of creating a custom attack generator.
"""

from gan_cyber_range.generators.base_gan import BaseGAN
from gan_cyber_range.core.attack_engine import AttackVector

class IoTAttackGAN(BaseGAN):
    """Custom GAN for IoT device attacks."""
    
    def __init__(self):
        super().__init__("iot_attacks")
        self.iot_protocols = ["mqtt", "coap", "zigbee", "bluetooth"]
        self.device_types = ["camera", "sensor", "gateway", "thermostat"]
    
    def generate_attacks(self, count: int = 100) -> List[AttackVector]:
        """Generate IoT-specific attack vectors."""
        
        attacks = []
        
        for i in range(count):
            # Select random IoT protocol and device type
            protocol = self._select_weighted(self.iot_protocols)
            device_type = self._select_weighted(self.device_types)
            
            # Generate attack based on protocol
            if protocol == "mqtt":
                attack = self._generate_mqtt_attack(device_type)
            elif protocol == "coap":
                attack = self._generate_coap_attack(device_type)
            elif protocol == "zigbee":
                attack = self._generate_zigbee_attack(device_type)
            else:
                attack = self._generate_bluetooth_attack(device_type)
            
            attacks.append(attack)
        
        return attacks
    
    def _generate_mqtt_attack(self, device_type: str) -> AttackVector:
        """Generate MQTT-specific attack."""
        
        payloads = {
            "topic_hijacking": f"home/{device_type}/control",
            "message_injection": '{"command": "shutdown", "auth": "bypass"}',
            "dos_payload": "A" * 10000
        }
        
        return AttackVector(
            attack_id=f"iot_mqtt_{int(time.time())}",
            attack_type="iot_protocol_attack",
            payload=payloads,
            techniques=["T1499"],  # Endpoint Denial of Service
            severity=0.7,
            stealth_level=0.6,
            target_systems=[f"{device_type}_device"],
            protocol="mqtt",
            attack_vector="network"
        )
    
    def _generate_coap_attack(self, device_type: str) -> AttackVector:
        """Generate CoAP-specific attack."""
        
        return AttackVector(
            attack_id=f"iot_coap_{int(time.time())}",
            attack_type="iot_protocol_attack",
            payload={
                "method": "PUT",
                "uri": f"/{device_type}/config",
                "payload": "malicious_config_data"
            },
            techniques=["T1190"],  # Exploit Public-Facing Application
            severity=0.8,
            stealth_level=0.5,
            target_systems=[f"{device_type}_device"],
            protocol="coap",
            attack_vector="network"
        )

# Register custom generator
from gan_cyber_range.factories import AttackFactory

attack_factory = AttackFactory()
attack_factory.register_generator("iot_attacks", IoTAttackGAN)

# Use custom generator
iot_attacks = attack_factory.generate_attacks("iot_attacks", count=20)
print(f"Generated {len(iot_attacks)} IoT attacks")
```

### Blue Team Custom Rule

```python
#!/usr/bin/env python3
"""
Example of adding custom blue team detection rules.
"""

from gan_cyber_range.blue_team.defense_suite import DefenseSuite, AlertSeverity

def create_custom_rules():
    """Create custom detection rules for specialized threats."""
    
    # Advanced persistent threat detection
    apt_rule = {
        "name": "APT Lateral Movement Detection",
        "conditions": {
            "event_type": "network_connection",
            "internal_to_internal": True,
            "unusual_time": True,  # Outside business hours
            "encrypted_traffic": True,
            "large_data_transfer": True
        },
        "severity": AlertSeverity.CRITICAL,
        "mitre_techniques": ["T1021", "T1041"],
        "confidence": 0.9,
        "correlation_rules": [
            "previous_credential_access",
            "privilege_escalation_detected"
        ]
    }
    
    # IoT device compromise detection
    iot_rule = {
        "name": "IoT Device Compromise",
        "conditions": {
            "event_type": "iot_communication",
            "device_type": ["camera", "sensor", "gateway"],
            "unexpected_commands": True,
            "configuration_changes": True
        },
        "severity": AlertSeverity.HIGH,
        "mitre_techniques": ["T1499", "T1190"],
        "confidence": 0.8
    }
    
    # Supply chain attack detection
    supply_chain_rule = {
        "name": "Supply Chain Attack Indicators",
        "conditions": {
            "event_type": "software_installation",
            "source": "third_party",
            "unsigned_binary": True,
            "network_connections": "suspicious_domains"
        },
        "severity": AlertSeverity.CRITICAL,
        "mitre_techniques": ["T1195"],
        "confidence": 0.85
    }
    
    return [apt_rule, iot_rule, supply_chain_rule]

def deploy_custom_rules(defense_suite: DefenseSuite):
    """Deploy custom rules to defense suite."""
    
    custom_rules = create_custom_rules()
    
    for i, rule in enumerate(custom_rules):
        rule_id = f"custom_rule_{i+1}"
        
        # Add to SIEM engine
        defense_suite.add_custom_rule("siem", rule_id, rule)
        
        print(f"Deployed custom rule: {rule['name']}")

# Example usage
cyber_range = CyberRange()
defense_suite = cyber_range.deploy_blue_team_defenses({"siem": True})
deploy_custom_rules(defense_suite)
```

This comprehensive API reference provides detailed documentation for all major components and interfaces in the GAN-Cyber-Range-v2 platform. The examples demonstrate practical usage patterns and show how to extend the platform with custom functionality.