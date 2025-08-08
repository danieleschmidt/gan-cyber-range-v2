# GAN-Cyber-Range-v2 API Documentation

## Overview

The GAN-Cyber-Range-v2 provides a comprehensive API for managing AI-driven cybersecurity training environments. This document covers all available endpoints, classes, and methods.

## Core Components

### AttackGAN

The core GAN-based attack generation system.

#### Class: `AttackGAN`

**Constructor:**
```python
AttackGAN(
    architecture: str = "wasserstein",
    attack_types: List[str] = None,
    noise_dim: int = 100,
    training_mode: str = "standard",
    device: str = "auto"
)
```

**Parameters:**
- `architecture`: GAN architecture type ("standard", "wasserstein", "conditional", "cyclic")
- `attack_types`: List of attack types to generate ("malware", "network", "web", "social_engineering")
- `noise_dim`: Dimensionality of input noise vector (50-1000)
- `training_mode`: Training mode ("standard", "differential_privacy", "federated")
- `device`: Compute device ("auto", "cpu", "cuda", "mps")

**Methods:**

##### `train(real_attacks, epochs=1000, batch_size=64, privacy_budget=None)`

Train the GAN on real attack data.

**Parameters:**
- `real_attacks`: Attack data (file path, directory, or list of strings)
- `epochs`: Number of training epochs (1-10000)
- `batch_size`: Training batch size (power of 2, 1-1024)
- `privacy_budget`: Differential privacy budget (optional)

**Returns:**
- `Dict[str, List[float]]`: Training history with loss curves

**Example:**
```python
from gan_cyber_range import AttackGAN

# Initialize GAN
attack_gan = AttackGAN(
    architecture="wasserstein",
    attack_types=["malware", "network", "web"],
    noise_dim=100
)

# Train on attack data
history = attack_gan.train(
    real_attacks="data/mitre_attack_samples/",
    epochs=1000,
    batch_size=64,
    privacy_budget=10.0
)

print(f"Training completed. Final G loss: {history['g_loss'][-1]:.4f}")
```

##### `generate(num_samples=1000, diversity_threshold=0.8, filter_detectable=True)`

Generate synthetic attack vectors.

**Parameters:**
- `num_samples`: Number of attacks to generate (1-100000)
- `diversity_threshold`: Minimum diversity score (0.0-1.0)
- `filter_detectable`: Remove easily detectable attacks

**Returns:**
- `List[AttackVector]`: Generated attack vectors

**Example:**
```python
# Generate diverse attacks
attacks = attack_gan.generate(
    num_samples=1000,
    diversity_threshold=0.8,
    filter_detectable=True
)

print(f"Generated {len(attacks)} unique attacks")
for attack in attacks[:5]:
    print(f"Type: {attack.attack_type}, Severity: {attack.severity}")
```

##### `diversity_score(attacks)`

Calculate diversity score for generated attacks.

**Parameters:**
- `attacks`: List of AttackVector objects

**Returns:**
- `float`: Diversity score (0.0-1.0)

##### `save_model(path)` / `load_model(path)`

Save or load trained model.

**Parameters:**
- `path`: File system path for model storage

---

### CyberRange

Main cyber range orchestration and management.

#### Class: `CyberRange`

**Constructor:**
```python
CyberRange(
    topology: NetworkTopology,
    hypervisor: str = "docker",
    container_runtime: str = "docker",
    network_emulation: str = "bridge"
)
```

**Methods:**

##### `deploy(resource_limits=None, isolation_level="container", monitoring=True)`

Deploy the cyber range infrastructure.

**Parameters:**
- `resource_limits`: Dict with CPU, memory, storage limits
- `isolation_level`: "container", "vm", or "strict"
- `monitoring`: Enable monitoring and logging

**Returns:**
- `str`: Range ID

**Example:**
```python
from gan_cyber_range import CyberRange, NetworkTopology

# Create topology
topology = NetworkTopology.generate(
    template="enterprise",
    subnets=["dmz", "internal", "management"],
    hosts_per_subnet={"dmz": 5, "internal": 20, "management": 3}
)

# Deploy range
cyber_range = CyberRange(topology)
range_id = cyber_range.deploy(
    resource_limits={"cpu_cores": 8, "memory_gb": 16},
    isolation_level="container",
    monitoring=True
)

print(f"Range deployed: {range_id}")
```

##### `start()` / `stop()` / `destroy()`

Control range lifecycle.

##### `execute_attack(attack_config)`

Execute an attack in the range.

**Parameters:**
- `attack_config`: Attack configuration dictionary

**Returns:**
- `str`: Attack execution ID

---

### NetworkTopology

Network topology generation and management.

#### Class: `NetworkTopology`

**Class Methods:**

##### `generate(template, subnets=None, hosts_per_subnet=None, services=None, vulnerabilities="realistic")`

Generate network topology from template.

**Parameters:**
- `template`: Topology template ("enterprise", "small_office", "data_center")
- `subnets`: List of subnet names
- `hosts_per_subnet`: Dict mapping subnet names to host counts
- `services`: List of available services
- `vulnerabilities`: Vulnerability level ("none", "realistic", "high")

**Returns:**
- `NetworkTopology`: Generated topology

**Example:**
```python
from gan_cyber_range import NetworkTopology

topology = NetworkTopology.generate(
    template="enterprise",
    subnets=["dmz", "internal", "management", "development"],
    hosts_per_subnet={"dmz": 5, "internal": 50, "management": 10, "development": 20},
    services=["web", "database", "email", "file_share", "vpn"],
    vulnerabilities="realistic"
)

print(f"Generated topology with {topology.total_hosts} hosts")
```

**Instance Methods:**

##### `add_subnet(name, cidr, security_zone="internal", vlan_id=None)`

Add subnet to topology.

##### `add_host(name, subnet_name, host_type, os_type, services=None, crown_jewel=False)`

Add host to topology.

##### `get_hosts_by_subnet(subnet_name)` / `get_hosts_by_type(host_type)`

Query hosts by subnet or type.

---

### RedTeamLLM

LLM-based adversarial red team.

#### Class: `RedTeamLLM`

**Constructor:**
```python
RedTeamLLM(
    model: str = "gpt-4",
    creativity: float = 0.8,
    risk_tolerance: float = 0.6,
    objective: str = "data_exfiltration"
)
```

**Methods:**

##### `generate_attack_plan(target_profile, constraints=None)`

Generate comprehensive attack plan.

**Parameters:**
- `target_profile`: Dict describing target organization
- `constraints`: Optional attack constraints

**Returns:**
- `Dict[str, Any]`: Complete attack plan

**Example:**
```python
from gan_cyber_range import RedTeamLLM

red_team = RedTeamLLM(
    model="llama2-70b-security",
    creativity=0.8,
    risk_tolerance=0.6,
    objective="data_exfiltration"
)

target = {
    "industry": "healthcare",
    "size": "medium",
    "security_maturity": "intermediate",
    "crown_jewels": ["patient_records", "research_data"]
}

plan = red_team.generate_attack_plan(target)
print(f"Generated {len(plan['phases'])} attack phases")
```

##### `adapt_tactics(current_plan, detection_events, blue_team_response=None)`

Adapt attack tactics based on defensive responses.

##### `generate_social_engineering_campaign(target_employees, campaign_type="spear_phishing")`

Generate social engineering campaigns.

---

### BlueTeamEvaluator

Blue team training and evaluation.

#### Class: `BlueTeamEvaluator`

**Constructor:**
```python
BlueTeamEvaluator(cyber_range: CyberRange)
```

**Methods:**

##### `deploy_defenses(defense_config)`

Deploy defensive tools and configurations.

**Parameters:**
- `defense_config`: Dict mapping defense types to tools

**Example:**
```python
from gan_cyber_range import BlueTeamEvaluator

evaluator = BlueTeamEvaluator(cyber_range)

evaluator.deploy_defenses({
    "siem": "splunk",
    "ids": "suricata",
    "edr": "crowdstrike",
    "deception": "honeypots"
})
```

##### `evaluate(duration="24h", attack_intensity="medium", scoring_model="mitre_attack")`

Evaluate blue team performance.

**Returns:**
- `Dict[str, Any]`: Evaluation results with metrics

---

## Utility Classes

### Monitoring and Metrics

#### Class: `MetricsCollector`

Real-time metrics collection and aggregation.

**Methods:**

##### `record_metric(name, value, metric_type, labels=None, unit="")`

Record a single metric.

##### `increment_counter(name, amount=1.0, labels=None)`

Increment counter metric.

##### `set_gauge(name, value, labels=None, unit="")`

Set gauge metric value.

##### `get_metrics(name_pattern=None, time_range=None)`

Retrieve metrics matching criteria.

**Example:**
```python
from gan_cyber_range.utils import MetricsCollector, MetricType

collector = MetricsCollector()
collector.start_collection()

# Record custom metrics
collector.record_metric("attack_success_rate", 0.85, MetricType.GAUGE, {"campaign": "red_team_1"})
collector.increment_counter("attacks_executed", 1.0, {"type": "web"})

# Get recent metrics
metrics = collector.get_metrics("attack_*", timedelta(hours=1))
```

### Caching

#### Class: `CacheManager`

Multi-tier caching system.

**Methods:**

##### `get(key, namespace="default")` / `set(key, value, namespace="default", ttl=None)`

Basic cache operations with namespacing.

##### `configure_policy(namespace, default_ttl=None, max_value_size=None, auto_warm=False)`

Configure caching policies for namespaces.

**Example:**
```python
from gan_cyber_range.utils import CacheManager, cached

cache = CacheManager()

# Configure policies
cache.configure_policy("models", default_ttl=3600, max_value_size=100*1024*1024)

# Use caching decorator
@cached(namespace="results", ttl=300)
def expensive_computation(params):
    # Expensive operation
    return complex_result
```

### Security

#### Class: `SecurityManager`

Comprehensive security management.

**Methods:**

##### `validate_operation(operation, user_id)`

Validate operation against security policies.

##### `deploy_security_measures(target, security_level="standard")`

Deploy containment and monitoring.

##### `emergency_response(threat_id, threat_level)`

Execute emergency response procedures.

**Example:**
```python
from gan_cyber_range.utils import SecurityManager, ThreatLevel

security = SecurityManager()

# Validate operation
operation = {
    "type": "attack",
    "technique_id": "T1190",
    "target_host": "web_server",
    "purpose": "training"
}

security.validate_operation(operation, user_id="researcher_123")

# Deploy security measures
containment_id = security.deploy_security_measures("range_1", "high_security")
```

## Error Handling

All API methods raise specific exceptions for different error conditions:

- `CyberRangeError`: Base exception for all cyber range errors
- `AttackExecutionError`: Attack execution failures
- `NetworkSimulationError`: Network simulation issues
- `ValidationError`: Input validation failures
- `SecurityValidationError`: Security policy violations

**Example Error Handling:**
```python
from gan_cyber_range.utils.error_handling import CyberRangeError, with_error_handling

@with_error_handling(reraise=True)
def safe_operation():
    try:
        result = cyber_range.execute_attack(config)
        return result
    except CyberRangeError as e:
        logger.error(f"Operation failed: {e.message}")
        raise
```

## Configuration

### Environment Variables

- `CYBER_RANGE_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `CYBER_RANGE_DATA_DIR`: Data directory path
- `CYBER_RANGE_CACHE_BACKEND`: Cache backend ("memory", "redis")
- `CYBER_RANGE_REDIS_URL`: Redis connection URL
- `CYBER_RANGE_GPU_ENABLED`: Enable GPU acceleration (true/false)

### Configuration Files

Configuration can be provided via JSON or YAML files:

```yaml
# config.yaml
gan:
  architecture: "wasserstein"
  noise_dim: 100
  training_mode: "differential_privacy"

range:
  hypervisor: "docker"
  isolation_level: "container"
  monitoring: true

security:
  require_approval: true
  audit_all_actions: true
  network_isolation: true
```

## Rate Limits and Quotas

- Attack generation: 1000 attacks per hour per user
- Range deployment: 5 concurrent ranges per user
- Training jobs: 1 concurrent training job per user
- API calls: 1000 requests per minute per API key

## Webhooks and Events

The system supports webhook notifications for key events:

```python
# Register webhook
cyber_range.on_event('attack_complete')
def handle_attack_complete(event_data):
    # Process attack completion
    send_notification(event_data)

# Available events
events = [
    'range_deployed', 'range_started', 'range_stopped',
    'attack_started', 'attack_complete', 'attack_failed',
    'detection_triggered', 'incident_created',
    'training_started', 'training_complete'
]
```

## Examples and Tutorials

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Getting started with GAN training
- `enterprise_range.py`: Deploying enterprise-scale ranges
- `red_team_campaign.py`: LLM-driven attack campaigns
- `blue_team_training.py`: Defensive training programs
- `research_workflow.py`: Academic research workflows

## SDK and Language Bindings

Official SDKs available for:
- Python 3.9+ (native)
- JavaScript/Node.js
- Go
- Java
- C#/.NET

## Support and Community

- GitHub Issues: Bug reports and feature requests
- Discord: Real-time community support
- Documentation: Comprehensive guides and tutorials
- Academic Papers: Research publications and citations