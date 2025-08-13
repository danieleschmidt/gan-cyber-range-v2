# GAN-Cyber-Range-v2 Security Guide

## Table of Contents

1. [Ethical Framework](#ethical-framework)
2. [Security Architecture](#security-architecture)
3. [Containment Mechanisms](#containment-mechanisms)
4. [Input Validation and Sanitization](#input-validation-and-sanitization)
5. [Authentication and Authorization](#authentication-and-authorization)
6. [Network Security](#network-security)
7. [Data Protection](#data-protection)
8. [Audit and Compliance](#audit-and-compliance)
9. [Incident Response](#incident-response)
10. [Security Best Practices](#security-best-practices)

## Ethical Framework

### Core Principles

The GAN-Cyber-Range-v2 platform is built with a comprehensive ethical framework that ensures responsible use of artificial intelligence in cybersecurity training.

#### Allowed Use Cases

✅ **Permitted Activities:**
- Security research and education
- Defensive training and simulation
- Vulnerability assessment (authorized environments only)
- Red team exercises (with proper authorization)
- Academic cybersecurity research
- Professional security certification training
- Incident response training
- Security awareness programs

#### Prohibited Activities

❌ **Strictly Forbidden:**
- Attacks against production systems without authorization
- Malicious software development for harmful purposes
- Unauthorized penetration testing
- Attacks against third-party systems
- Criminal activities or illegal hacking
- Harassment or harmful targeting of individuals
- Circumvention of legal or regulatory requirements

### Ethical Compliance Implementation

```python
from gan_cyber_range.utils.enhanced_security import EthicalFramework

class EthicalFramework:
    def __init__(self):
        self.allowed_uses = [
            "education",
            "training", 
            "research",
            "defensive_simulation",
            "authorized_testing"
        ]
        
        self.prohibited_targets = [
            "production_systems",
            "unauthorized_networks",
            "public_infrastructure",
            "third_party_systems"
        ]
    
    def validate_request(self, request: Dict[str, Any]) -> ValidationResult:
        """Validate request against ethical guidelines."""
        
        # Check purpose compliance
        purpose = request.get('purpose', '').lower()
        if not any(allowed in purpose for allowed in self.allowed_uses):
            return ValidationResult(
                valid=False,
                reason="Purpose not aligned with allowed use cases"
            )
        
        # Check target compliance
        targets = request.get('targets', [])
        for target in targets:
            if self._is_prohibited_target(target):
                return ValidationResult(
                    valid=False,
                    reason=f"Target {target} is prohibited"
                )
        
        # Check authorization
        if not request.get('authorized', False):
            return ValidationResult(
                valid=False,
                reason="Proper authorization required"
            )
        
        return ValidationResult(valid=True, reason="Request compliant")
```

### Compliance Monitoring

The platform continuously monitors for ethical compliance through:

1. **Real-time Request Validation**: Every request is validated against ethical guidelines
2. **Target System Verification**: Ensures attacks only target authorized systems
3. **Activity Logging**: Comprehensive audit trail of all activities
4. **Automated Alerts**: Immediate notification of potential violations
5. **Regular Compliance Reviews**: Periodic assessment of platform usage

## Security Architecture

### Defense in Depth

The platform implements a multi-layered security approach:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  • Input Validation  • Session Management  • CSRF Protection│
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                        │
│  • Authentication  • Authorization  • Business Logic       │
├─────────────────────────────────────────────────────────────┤
│                    Service Layer                           │
│  • API Security  • Rate Limiting  • Encryption            │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                     │
│  • Network Isolation  • Container Security  • Monitoring   │
├─────────────────────────────────────────────────────────────┤
│                    Data Layer                              │
│  • Encryption at Rest  • Access Controls  • Backup Security│
└─────────────────────────────────────────────────────────────┘
```

### Security Components

#### 1. Secure Input Validator

```python
class SecureInputValidator:
    def __init__(self):
        self.dangerous_patterns = {
            'sql_injection': [
                r"(\bunion\b.*\bselect\b)",
                r"(\bselect\b.*\bfrom\b.*\bwhere\b)",
                r"(\bdrop\b\s+\btable\b)",
            ],
            'xss': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
            ],
            'command_injection': [
                r"[;&|`$(){}]",
                r"\b(cat|ls|pwd|whoami|id|uname)\b",
                r"\b(rm|mv|cp|chmod|chown)\b",
            ]
        }
    
    def validate_input(self, input_value: str, input_type: str = 'general') -> Dict[str, Any]:
        """Comprehensive input validation with threat detection."""
        
        validation_result = {
            'is_valid': True,
            'sanitized_value': input_value,
            'threats_detected': [],
            'severity': ThreatLevel.LOW
        }
        
        # Length validation
        max_length = self.max_lengths.get(input_type, 500)
        if len(input_value) > max_length:
            validation_result['is_valid'] = False
            validation_result['threats_detected'].append(f'Input exceeds maximum length')
        
        # Pattern-based threat detection
        for threat_type, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_value, re.IGNORECASE):
                    validation_result['is_valid'] = False
                    validation_result['threats_detected'].append(f'{threat_type}: {pattern}')
                    validation_result['severity'] = ThreatLevel.HIGH
        
        # Sanitize input
        validation_result['sanitized_value'] = self._sanitize_input(input_value)
        
        return validation_result
```

#### 2. Threat Detection Engine

```python
class ThreatDetectionEngine:
    def __init__(self):
        self.threat_signatures = self._load_threat_signatures()
        self.anomaly_thresholds = {
            'request_rate': 100,  # requests per minute
            'failed_auth_rate': 10,
            'unusual_patterns': 5
        }
    
    def analyze_payload(self, payload: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze payload for threats using multiple detection methods."""
        
        analysis_result = {
            'threat_detected': False,
            'threat_type': None,
            'confidence': 0.0,
            'signatures_matched': [],
            'risk_score': 0.0
        }
        
        # Signature-based detection
        risk_score = 0.0
        signatures_matched = []
        
        for threat_type, signatures in self.threat_signatures.items():
            for signature in signatures:
                if re.search(signature, payload, re.IGNORECASE):
                    signatures_matched.append(f'{threat_type}:{signature}')
                    risk_score += 0.3
        
        # Entropy analysis (detect encoded/encrypted content)
        entropy = self._calculate_entropy(payload)
        if entropy > 7.5:  # High entropy indicates potential encoding
            risk_score += 0.2
            signatures_matched.append('high_entropy')
        
        # Behavioral analysis
        if context:
            behavioral_risk = self._analyze_behavior(payload, context)
            risk_score += behavioral_risk
        
        # Determine overall threat level
        if risk_score > 0.7:
            analysis_result['threat_detected'] = True
            analysis_result['threat_type'] = 'high_risk'
            analysis_result['confidence'] = min(risk_score, 1.0)
        
        analysis_result['risk_score'] = risk_score
        analysis_result['signatures_matched'] = signatures_matched
        
        return analysis_result
```

## Containment Mechanisms

### Network Isolation

All attack simulations are contained within isolated network segments:

```python
class ContainmentEngine:
    def __init__(self):
        self.containment_policies = {
            'network_isolation': True,
            'resource_limits': True,
            'monitoring_enhanced': True,
            'logging_verbose': True
        }
    
    def enforce_containment(self, range_id: str, threat_level: ThreatLevel) -> Dict[str, Any]:
        """Enforce containment based on threat level."""
        
        containment_measures = {
            'containment_id': str(uuid.uuid4()),
            'range_id': range_id,
            'threat_level': threat_level,
            'timestamp': datetime.now(),
            'measures_applied': [],
            'status': 'active'
        }
        
        # Apply containment based on threat level
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            # Network isolation
            self._isolate_network(range_id)
            containment_measures['measures_applied'].append('network_isolation')
            
            # Resource throttling
            self._throttle_resources(range_id)
            containment_measures['measures_applied'].append('resource_throttling')
        
        if threat_level == ThreatLevel.CRITICAL:
            # Complete range suspension
            self._suspend_range(range_id)
            containment_measures['measures_applied'].append('range_suspension')
        
        return containment_measures
```

### Container Security

```yaml
# Container security configuration
security_context:
  runAsNonRoot: true
  runAsUser: 1001
  runAsGroup: 1001
  fsGroup: 1001
  seccompProfile:
    type: RuntimeDefault
  capabilities:
    drop:
      - ALL
    add:
      - NET_BIND_SERVICE

# Resource limits
resources:
  limits:
    cpu: "2"
    memory: "4Gi"
    ephemeral-storage: "10Gi"
  requests:
    cpu: "1"
    memory: "2Gi"
    ephemeral-storage: "5Gi"

# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cyber-range-isolation
spec:
  podSelector:
    matchLabels:
      app: cyber-range
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: cyber-range
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: cyber-range
```

### Resource Monitoring

```python
class ResourceMonitor:
    def __init__(self):
        self.resource_limits = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_io_mbps': 100,
            'network_connections': 1000
        }
    
    def monitor_resources(self, container_id: str) -> Dict[str, Any]:
        """Monitor container resource usage."""
        
        usage = self._get_container_stats(container_id)
        alerts = []
        
        for resource, limit in self.resource_limits.items():
            current_usage = usage.get(resource, 0)
            if current_usage > limit:
                alerts.append({
                    'resource': resource,
                    'current': current_usage,
                    'limit': limit,
                    'severity': 'warning' if current_usage < limit * 1.1 else 'critical'
                })
        
        return {
            'usage': usage,
            'alerts': alerts,
            'status': 'healthy' if not alerts else 'warning'
        }
```

## Input Validation and Sanitization

### Comprehensive Input Validation

```python
def validate_attack_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate attack configuration for safety."""
    
    validation_result = {
        'is_safe': True,
        'issues': [],
        'sanitized_config': config.copy()
    }
    
    # Check for production system targets
    production_indicators = [
        'prod', 'production', 'live', 'www', 'api',
        '.com', '.net', '.org', '.edu', '.gov'
    ]
    
    for key, value in config.items():
        if isinstance(value, str):
            for indicator in production_indicators:
                if indicator in value.lower():
                    validation_result['is_safe'] = False
                    validation_result['issues'].append(
                        f'Potential production target: {key}={value}'
                    )
    
    # Validate target IP addresses
    if 'target_ip' in config:
        ip = config['target_ip']
        if not _is_safe_target_ip(ip):
            validation_result['is_safe'] = False
            validation_result['issues'].append(f'Unsafe target IP: {ip}')
    
    return validation_result

def _is_safe_target_ip(ip: str) -> bool:
    """Check if target IP is safe for testing."""
    
    # Allow only RFC 1918 private networks and loopback
    safe_patterns = [
        r'^127\.',           # Loopback
        r'^10\.',            # 10.0.0.0/8
        r'^172\.(1[6-9]|2[0-9]|3[01])\.',  # 172.16.0.0/12
        r'^192\.168\.',      # 192.168.0.0/16
    ]
    
    return any(re.match(pattern, ip) for pattern in safe_patterns)
```

### Sanitization Functions

```python
def sanitize_input(input_value: str) -> str:
    """Sanitize input by removing dangerous characters."""
    
    # HTML entity encoding for basic XSS prevention
    sanitized = input_value.replace('<', '&lt;').replace('>', '&gt;')
    sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
    
    # Remove null bytes and control characters
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
    
    # Remove potentially dangerous file paths
    sanitized = re.sub(r'\.\./', '', sanitized)
    sanitized = re.sub(r'\.\.\\', '', sanitized)
    
    return sanitized

def validate_file_upload(file_data: bytes, filename: str) -> Dict[str, Any]:
    """Validate uploaded files for security."""
    
    validation_result = {
        'is_safe': True,
        'issues': [],
        'file_type': None
    }
    
    # Check file size
    if len(file_data) > 10 * 1024 * 1024:  # 10MB limit
        validation_result['is_safe'] = False
        validation_result['issues'].append('File size exceeds limit')
    
    # Check file extension
    dangerous_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com']
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext in dangerous_extensions:
        validation_result['is_safe'] = False
        validation_result['issues'].append(f'Dangerous file extension: {file_ext}')
    
    # Check file signature (magic bytes)
    file_signature = file_data[:16]
    if _is_executable_signature(file_signature):
        validation_result['is_safe'] = False
        validation_result['issues'].append('Executable file detected')
    
    return validation_result
```

## Authentication and Authorization

### Multi-Factor Authentication

```python
class AuthenticationManager:
    def __init__(self):
        self.totp_secrets = {}
        self.failed_attempts = {}
        self.lockout_duration = 900  # 15 minutes
    
    def authenticate_user(
        self, 
        username: str, 
        password: str, 
        totp_code: Optional[str] = None
    ) -> AuthResult:
        """Authenticate user with MFA support."""
        
        # Check if account is locked
        if self._is_account_locked(username):
            return AuthResult(
                success=False,
                reason="Account temporarily locked",
                requires_mfa=False
            )
        
        # Validate password
        user = self._get_user(username)
        if not user or not self._verify_password(password, user.password_hash):
            self._record_failed_attempt(username)
            return AuthResult(
                success=False,
                reason="Invalid credentials",
                requires_mfa=False
            )
        
        # Check if MFA is required
        if user.mfa_enabled:
            if not totp_code:
                return AuthResult(
                    success=False,
                    reason="MFA code required",
                    requires_mfa=True
                )
            
            if not self._verify_totp(username, totp_code):
                self._record_failed_attempt(username)
                return AuthResult(
                    success=False,
                    reason="Invalid MFA code",
                    requires_mfa=True
                )
        
        # Reset failed attempts on successful login
        self._reset_failed_attempts(username)
        
        # Generate session token
        session_token = self._generate_session_token(user)
        
        return AuthResult(
            success=True,
            session_token=session_token,
            user_id=user.id,
            requires_mfa=False
        )
```

### Role-Based Access Control (RBAC)

```python
class AuthorizationManager:
    def __init__(self):
        self.roles = {
            'admin': {
                'permissions': [
                    'create_scenario',
                    'delete_scenario',
                    'manage_users',
                    'view_audit_logs',
                    'configure_platform'
                ]
            },
            'instructor': {
                'permissions': [
                    'create_scenario',
                    'edit_scenario',
                    'run_scenario',
                    'view_results'
                ]
            },
            'student': {
                'permissions': [
                    'participate_scenario',
                    'view_own_results'
                ]
            },
            'analyst': {
                'permissions': [
                    'view_results',
                    'export_reports',
                    'view_metrics'
                ]
            }
        }
    
    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission."""
        
        user = self._get_user(user_id)
        if not user:
            return False
        
        user_permissions = set()
        for role in user.roles:
            role_permissions = self.roles.get(role, {}).get('permissions', [])
            user_permissions.update(role_permissions)
        
        return permission in user_permissions
    
    def require_permission(self, permission: str):
        """Decorator to require specific permission."""
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Extract user from context
                user_id = self._get_current_user_id()
                
                if not self.check_permission(user_id, permission):
                    raise AuthorizationError(f"Permission required: {permission}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
```

### Session Management

```python
class SessionManager:
    def __init__(self):
        self.session_timeout = 3600  # 1 hour
        self.active_sessions = {}
    
    def create_session(self, user_id: str) -> str:
        """Create new user session."""
        
        session_id = secrets.token_urlsafe(32)
        session_data = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'ip_address': self._get_client_ip(),
            'user_agent': self._get_user_agent()
        }
        
        self.active_sessions[session_id] = session_data
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and refresh session."""
        
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        # Check session timeout
        if datetime.now() - session['last_activity'] > timedelta(seconds=self.session_timeout):
            self.destroy_session(session_id)
            return None
        
        # Update last activity
        session['last_activity'] = datetime.now()
        
        return session
    
    def destroy_session(self, session_id: str) -> None:
        """Destroy user session."""
        
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
```

## Network Security

### Network Segmentation

```yaml
# Network security configuration
networks:
  # Management network (admin access only)
  management:
    driver: bridge
    ipam:
      config:
        - subnet: 172.16.1.0/24
    options:
      com.docker.network.bridge.enable_icc: "false"
  
  # Training network (isolated scenarios)
  training:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 192.168.100.0/24
  
  # Monitoring network (metrics and logs)
  monitoring:
    driver: bridge
    ipam:
      config:
        - subnet: 172.16.2.0/24
```

### Firewall Rules

```bash
#!/bin/bash
# firewall_config.sh

# Drop all traffic by default
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (restrict to specific IPs)
iptables -A INPUT -p tcp --dport 22 -s 10.0.0.0/8 -j ACCEPT

# Allow HTTPS
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow monitoring ports (restrict to monitoring network)
iptables -A INPUT -p tcp --dport 9090 -s 172.16.2.0/24 -j ACCEPT  # Prometheus
iptables -A INPUT -p tcp --dport 3000 -s 172.16.2.0/24 -j ACCEPT  # Grafana

# Block direct access to application ports
iptables -A INPUT -p tcp --dport 8080 -j DROP

# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "DROPPED: "
iptables -A INPUT -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### TLS Configuration

```nginx
# nginx SSL configuration
server {
    listen 443 ssl http2;
    server_name cyber-range.local;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/cyber-range.crt;
    ssl_certificate_key /etc/nginx/ssl/cyber-range.key;
    
    # SSL Security
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;
    
    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';";
    
    # Rate limiting
    limit_req zone=api burst=20 nodelay;
    limit_req zone=login burst=5 nodelay;
    
    location / {
        proxy_pass http://cyber-range-core:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Security headers for proxied content
        proxy_hide_header X-Powered-By;
        proxy_hide_header Server;
    }
}
```

## Data Protection

### Encryption at Rest

```python
class SecureDataManager:
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            master_key = Fernet.generate_key()
        
        self.cipher_suite = Fernet(master_key)
        self.master_key = master_key
    
    def encrypt_sensitive_data(self, data: Union[str, Dict[str, Any]]) -> str:
        """Encrypt sensitive data before storage."""
        
        if isinstance(data, dict):
            data = json.dumps(data)
        
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Union[str, Dict[str, Any]]:
        """Decrypt sensitive data after retrieval."""
        
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
        decrypted_str = decrypted_bytes.decode()
        
        # Try to parse as JSON
        try:
            return json.loads(decrypted_str)
        except json.JSONDecodeError:
            return decrypted_str
```

### Database Security

```sql
-- Database security configuration

-- Create encrypted tablespace
CREATE TABLESPACE encrypted_data 
LOCATION '/var/lib/postgresql/encrypted' 
WITH (encryption_key_id = 1);

-- Create secure tables
CREATE TABLE user_credentials (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt VARCHAR(255) NOT NULL,
    mfa_secret TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
) TABLESPACE encrypted_data;

-- Row level security
ALTER TABLE user_credentials ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_credentials_policy ON user_credentials
    FOR ALL
    TO cyber_range_app
    USING (username = current_setting('app.current_user'));

-- Audit logging
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    action VARCHAR(255) NOT NULL,
    resource VARCHAR(255),
    timestamp TIMESTAMP DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    details JSONB
);

-- Create audit trigger
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (user_id, action, resource, details)
    VALUES (
        current_setting('app.current_user_id')::INTEGER,
        TG_OP,
        TG_TABLE_NAME,
        row_to_json(NEW)
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;
```

### Backup Security

```bash
#!/bin/bash
# secure_backup.sh

BACKUP_KEY="/etc/cyber-range/backup.key"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Encrypt database backup
pg_dump cyber_range | \
gpg --symmetric --cipher-algo AES256 --compress-algo 2 \
    --passphrase-file "$BACKUP_KEY" \
    --output "cyber_range_backup_${TIMESTAMP}.sql.gpg"

# Encrypt file system backup
tar czf - data/ models/ | \
gpg --symmetric --cipher-algo AES256 --compress-algo 2 \
    --passphrase-file "$BACKUP_KEY" \
    --output "cyber_range_data_${TIMESTAMP}.tar.gz.gpg"

# Upload to secure storage with server-side encryption
aws s3 cp "cyber_range_backup_${TIMESTAMP}.sql.gpg" \
    s3://cyber-range-backups/database/ \
    --server-side-encryption AES256

aws s3 cp "cyber_range_data_${TIMESTAMP}.tar.gz.gpg" \
    s3://cyber-range-backups/data/ \
    --server-side-encryption AES256

# Secure cleanup
shred -vfz -n 3 "cyber_range_backup_${TIMESTAMP}.sql.gpg"
shred -vfz -n 3 "cyber_range_data_${TIMESTAMP}.tar.gz.gpg"
```

## Audit and Compliance

### Comprehensive Audit Logging

```python
class SecurityAuditLogger:
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.security_events = []
        
        # Set up dedicated security logger
        self.security_logger = logging.getLogger('security_audit')
        self.security_logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.security_logger.addHandler(handler)
    
    def log_security_event(self, event: SecurityEvent) -> None:
        """Log security event with full context."""
        
        self.security_events.append(event)
        
        log_message = json.dumps({
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'threat_level': event.threat_level.value,
            'timestamp': event.timestamp.isoformat(),
            'source_ip': event.source_ip,
            'user_id': event.user_id,
            'details': event.details,
            'mitigated': event.mitigated,
            'mitigation_actions': event.mitigation_actions
        })
        
        # Log based on severity
        if event.threat_level == ThreatLevel.CRITICAL:
            self.security_logger.critical(log_message)
        elif event.threat_level == ThreatLevel.HIGH:
            self.security_logger.error(log_message)
        elif event.threat_level == ThreatLevel.MEDIUM:
            self.security_logger.warning(log_message)
        else:
            self.security_logger.info(log_message)
```

### Compliance Reporting

```python
class ComplianceReporter:
    def __init__(self):
        self.compliance_standards = {
            'NIST': {
                'frameworks': ['CSF', 'SP 800-53'],
                'controls': [
                    'AC-2', 'AC-3', 'AC-6',  # Access Control
                    'AU-2', 'AU-3', 'AU-12', # Audit and Accountability
                    'SC-8', 'SC-13', 'SC-28' # System and Communications Protection
                ]
            },
            'ISO27001': {
                'frameworks': ['ISO/IEC 27001:2013'],
                'controls': [
                    'A.9.1', 'A.9.2', 'A.9.4',  # Access Control
                    'A.12.4', 'A.12.6',         # Operations Security
                    'A.10.1', 'A.10.2'          # Cryptography
                ]
            }
        }
    
    def generate_compliance_report(
        self, 
        standard: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for specified standard."""
        
        if standard not in self.compliance_standards:
            raise ValueError(f"Unknown compliance standard: {standard}")
        
        report = {
            'standard': standard,
            'report_period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'compliance_status': {},
            'findings': [],
            'recommendations': []
        }
        
        # Assess each control
        for control in self.compliance_standards[standard]['controls']:
            assessment = self._assess_control(control, start_date, end_date)
            report['compliance_status'][control] = assessment
            
            if not assessment['compliant']:
                report['findings'].append({
                    'control': control,
                    'issue': assessment['issue'],
                    'severity': assessment['severity']
                })
        
        # Calculate overall compliance score
        compliant_controls = sum(1 for c in report['compliance_status'].values() if c['compliant'])
        total_controls = len(report['compliance_status'])
        report['compliance_score'] = compliant_controls / total_controls
        
        return report
```

## Incident Response

### Automated Response System

```python
class IncidentResponseManager:
    def __init__(self):
        self.response_playbooks = {
            'unauthorized_access': [
                'lock_account',
                'notify_security_team',
                'preserve_evidence',
                'investigate_source'
            ],
            'malware_detected': [
                'isolate_system',
                'scan_network',
                'update_signatures',
                'notify_stakeholders'
            ],
            'data_exfiltration': [
                'block_traffic',
                'preserve_logs',
                'notify_legal',
                'initiate_investigation'
            ]
        }
    
    def handle_security_incident(self, incident: SecurityIncident) -> IncidentResponse:
        """Handle security incident with automated response."""
        
        response = IncidentResponse(
            incident_id=incident.incident_id,
            timestamp=datetime.now(),
            actions_taken=[],
            status='in_progress'
        )
        
        # Determine appropriate playbook
        playbook = self.response_playbooks.get(incident.incident_type, [])
        
        for action in playbook:
            try:
                result = self._execute_response_action(action, incident)
                response.actions_taken.append({
                    'action': action,
                    'result': result,
                    'timestamp': datetime.now()
                })
            except Exception as e:
                response.actions_taken.append({
                    'action': action,
                    'result': f'failed: {str(e)}',
                    'timestamp': datetime.now()
                })
        
        response.status = 'completed'
        return response
```

### Evidence Preservation

```python
class EvidenceManager:
    def __init__(self):
        self.evidence_storage = "/secure/evidence"
        self.chain_of_custody = []
    
    def preserve_evidence(
        self, 
        incident_id: str,
        evidence_type: str,
        data: Union[str, bytes, Dict[str, Any]]
    ) -> str:
        """Preserve digital evidence with chain of custody."""
        
        evidence_id = f"{incident_id}_{evidence_type}_{int(time.time())}"
        
        # Create evidence package
        evidence_package = {
            'evidence_id': evidence_id,
            'incident_id': incident_id,
            'evidence_type': evidence_type,
            'timestamp': datetime.now().isoformat(),
            'collector': self._get_current_user(),
            'hash': self._calculate_hash(data),
            'data': data
        }
        
        # Encrypt and store evidence
        encrypted_evidence = self._encrypt_evidence(evidence_package)
        evidence_path = os.path.join(self.evidence_storage, f"{evidence_id}.enc")
        
        with open(evidence_path, 'wb') as f:
            f.write(encrypted_evidence)
        
        # Record chain of custody
        custody_record = {
            'evidence_id': evidence_id,
            'action': 'preserved',
            'timestamp': datetime.now(),
            'user': self._get_current_user(),
            'location': evidence_path,
            'hash': self._calculate_hash(encrypted_evidence)
        }
        
        self.chain_of_custody.append(custody_record)
        
        return evidence_id
```

## Security Best Practices

### Secure Development Guidelines

1. **Input Validation**
   - Validate all inputs at the boundary
   - Use allowlists instead of blocklists
   - Sanitize data before processing
   - Implement length and format checks

2. **Authentication & Authorization**
   - Implement multi-factor authentication
   - Use strong password policies
   - Implement proper session management
   - Follow principle of least privilege

3. **Data Protection**
   - Encrypt sensitive data at rest and in transit
   - Use strong encryption algorithms (AES-256)
   - Implement proper key management
   - Regular backup and recovery testing

4. **Network Security**
   - Implement network segmentation
   - Use firewalls and intrusion detection
   - Monitor network traffic
   - Regular security assessments

5. **Monitoring & Logging**
   - Log all security-relevant events
   - Implement real-time monitoring
   - Set up automated alerting
   - Regular log analysis

### Security Checklist

#### Pre-Deployment
- [ ] Security code review completed
- [ ] Vulnerability scanning performed
- [ ] Penetration testing conducted
- [ ] Security configurations validated
- [ ] Backup and recovery procedures tested

#### Runtime Security
- [ ] Real-time monitoring active
- [ ] Intrusion detection systems operational
- [ ] Log aggregation functioning
- [ ] Automated alerting configured
- [ ] Incident response procedures defined

#### Ongoing Maintenance
- [ ] Regular security updates applied
- [ ] Vulnerability assessments scheduled
- [ ] Security training provided
- [ ] Compliance audits conducted
- [ ] Disaster recovery procedures tested

### Security Configuration Templates

```yaml
# security_config.yaml
security:
  authentication:
    mfa_required: true
    session_timeout: 3600
    password_policy:
      min_length: 12
      require_upper: true
      require_lower: true
      require_digits: true
      require_special: true
      
  authorization:
    rbac_enabled: true
    default_role: "student"
    admin_approval_required: true
    
  encryption:
    algorithm: "AES-256-GCM"
    key_rotation_days: 90
    tls_version: "1.3"
    
  monitoring:
    log_level: "INFO"
    audit_enabled: true
    real_time_alerts: true
    retention_days: 365
    
  compliance:
    standards: ["NIST", "ISO27001"]
    reporting_enabled: true
    automated_assessment: true
```

This comprehensive security guide provides detailed information about all security aspects of the GAN-Cyber-Range-v2 platform, ensuring that users can deploy and operate the system safely and in compliance with industry standards.