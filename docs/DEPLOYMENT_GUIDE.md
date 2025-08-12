# GAN-Cyber-Range-v2 Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Configuration](#configuration)
4. [Development Deployment](#development-deployment)
5. [Production Deployment](#production-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Security Considerations](#security-considerations)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 50GB free space
- **OS**: Linux (Ubuntu 20.04+), macOS 11+, Windows 10+
- **Network**: Internet connectivity for image downloads

#### Recommended Requirements
- **CPU**: 8+ cores (16+ for production)
- **RAM**: 16GB+ (32GB+ for production)
- **Storage**: 100GB+ SSD
- **GPU**: NVIDIA RTX 3070+ (for accelerated GAN training)
- **Network**: Dedicated network interface for cyber range traffic

### Software Dependencies

#### Core Dependencies
```bash
# Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Docker Compose v2
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Python 3.9+
sudo apt-get install python3.9 python3.9-venv python3.9-dev

# Git
sudo apt-get install git
```

#### Optional Dependencies
```bash
# NVIDIA Docker (for GPU support)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Kubernetes (for production deployments)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

## Environment Setup

### Clone Repository

```bash
git clone https://github.com/terragonlabs/gan-cyber-range-v2.git
cd gan-cyber-range-v2
```

### Python Environment

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

## Configuration

### Environment Configuration (.env)

```bash
# Platform Configuration
CYBER_RANGE_ENV=development
CYBER_RANGE_SECRET_KEY=your_super_secret_key_here_change_in_production
CYBER_RANGE_DEBUG=true

# Database Configuration
DATABASE_URL=postgresql://cyber_range:secure_password@postgres:5432/cyber_range
POSTGRES_DB=cyber_range
POSTGRES_USER=cyber_range
POSTGRES_PASSWORD=secure_password_change_in_production

# Redis Configuration
REDIS_URL=redis://:redis_password@redis:6379/0
REDIS_PASSWORD=redis_password_change_in_production

# RabbitMQ Configuration
RABBITMQ_URL=amqp://admin:rabbitmq_password@rabbitmq:5672/cyber_range
RABBITMQ_USER=admin
RABBITMQ_PASSWORD=rabbitmq_password_change_in_production

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_PASSWORD=grafana_password_change_in_production

# Security Configuration
ENABLE_ETHICAL_FRAMEWORK=true
ENABLE_CONTAINMENT=true
MAX_ATTACK_INTENSITY=high

# Performance Configuration
ENABLE_AUTO_SCALING=true
MAX_WORKERS=10
CACHE_SIZE=1GB

# Backup Configuration
BACKUP_S3_BUCKET=cyber-range-backups
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

### Network Configuration

```yaml
# config/network.yaml
networks:
  cyber_range_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1
    options:
      com.docker.network.bridge.name: cyber-range-br0
      com.docker.network.driver.mtu: 1500

  monitoring_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.21.0.0/16
          gateway: 172.21.0.1

  isolated_network:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 192.168.100.0/24
```

### Security Configuration

```yaml
# config/security.yaml
ethical_framework:
  enabled: true
  allowed_targets:
    - "192.168.0.0/16"
    - "10.0.0.0/8"
    - "172.16.0.0/12"
  prohibited_targets:
    - "0.0.0.0/0"
    - "public_domains"
  
containment:
  network_isolation: strict
  resource_limits:
    cpu: "80%"
    memory: "80%"
    disk_io: "70%"
  
monitoring:
  audit_logging: true
  real_time_alerts: true
  compliance_checking: true
```

## Development Deployment

### Quick Start (Development)

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Check service status
docker-compose -f docker-compose.dev.yml ps

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Access the platform
curl http://localhost:8080/health
```

### Development Compose File

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  cyber-range-core:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8080:8080"  # HTTP API
      - "8443:8443"  # HTTPS API
    environment:
      - CYBER_RANGE_ENV=development
      - CYBER_RANGE_DEBUG=true
    volumes:
      - .:/app
      - cyber_range_data:/app/data
    networks:
      - cyber-range-network
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=cyber_range_dev
      - POSTGRES_USER=cyber_range
      - POSTGRES_PASSWORD=dev_password
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - cyber-range-network

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass dev_password
    ports:
      - "6379:6379"
    networks:
      - cyber-range-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.dev.yml:/etc/prometheus/prometheus.yml
    networks:
      - cyber-range-network

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=dev_password
    ports:
      - "3000:3000"
    volumes:
      - grafana_dev_data:/var/lib/grafana
    networks:
      - cyber-range-network

volumes:
  postgres_dev_data:
  grafana_dev_data:
  cyber_range_data:

networks:
  cyber-range-network:
    driver: bridge
```

### Development Dockerfile

```dockerfile
# Dockerfile.dev
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install development dependencies
COPY requirements.dev.txt .
RUN pip install -r requirements.dev.txt

# Copy source code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Install package in development mode
RUN pip install -e .

# Expose ports
EXPOSE 8080 8443

# Development command with auto-reload
CMD ["python", "-m", "gan_cyber_range.cli", "start", "--mode", "development", "--reload"]
```

## Production Deployment

### Production Environment Setup

```bash
# Create production environment file
cp .env.example .env.production

# Generate secure secrets
export CYBER_RANGE_SECRET_KEY=$(openssl rand -hex 32)
export POSTGRES_PASSWORD=$(openssl rand -hex 24)
export REDIS_PASSWORD=$(openssl rand -hex 24)
export RABBITMQ_PASSWORD=$(openssl rand -hex 24)
export GRAFANA_PASSWORD=$(openssl rand -hex 16)

# Update production environment file
sed -i "s/your_super_secret_key_here_change_in_production/$CYBER_RANGE_SECRET_KEY/g" .env.production
sed -i "s/secure_password_change_in_production/$POSTGRES_PASSWORD/g" .env.production
sed -i "s/redis_password_change_in_production/$REDIS_PASSWORD/g" .env.production
sed -i "s/rabbitmq_password_change_in_production/$RABBITMQ_PASSWORD/g" .env.production
sed -i "s/grafana_password_change_in_production/$GRAFANA_PASSWORD/g" .env.production
```

### SSL Certificate Setup

```bash
# Create SSL certificate directory
mkdir -p deployment/nginx/ssl

# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout deployment/nginx/ssl/cyber-range.key \
    -out deployment/nginx/ssl/cyber-range.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=cyber-range.local"

# For production, use Let's Encrypt or proper CA certificates
# certbot --nginx -d your-domain.com
```

### Production Deployment

```bash
# Set production environment
export COMPOSE_FILE=deployment/docker-compose.prod.yml
export COMPOSE_PROJECT_NAME=cyber-range-prod

# Create required directories
mkdir -p {logs,data,models,backups}

# Deploy production stack
docker-compose up -d

# Check deployment status
docker-compose ps

# View logs
docker-compose logs -f cyber-range-core
```

### Production Health Checks

```bash
#!/bin/bash
# health_check.sh

echo "=== GAN-Cyber-Range-v2 Health Check ==="

# Check core service
echo "Checking core service..."
curl -f http://localhost:8080/health || echo "❌ Core service unhealthy"
curl -f https://localhost:8443/health -k || echo "❌ HTTPS endpoint unhealthy"

# Check database
echo "Checking database..."
docker-compose exec postgres pg_isready -U cyber_range || echo "❌ Database unhealthy"

# Check Redis
echo "Checking Redis..."
docker-compose exec redis redis-cli ping || echo "❌ Redis unhealthy"

# Check RabbitMQ
echo "Checking RabbitMQ..."
docker-compose exec rabbitmq rabbitmq-diagnostics ping || echo "❌ RabbitMQ unhealthy"

# Check monitoring
echo "Checking monitoring..."
curl -f http://localhost:9090/-/healthy || echo "❌ Prometheus unhealthy"
curl -f http://localhost:3000/api/health || echo "❌ Grafana unhealthy"

echo "✅ Health check completed"
```

## Kubernetes Deployment

### Kubernetes Manifests

#### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cyber-range
  labels:
    app: gan-cyber-range-v2
```

#### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cyber-range-config
  namespace: cyber-range
data:
  CYBER_RANGE_ENV: "production"
  ENABLE_ETHICAL_FRAMEWORK: "true"
  ENABLE_AUTO_SCALING: "true"
  MAX_WORKERS: "20"
```

#### Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: cyber-range-secrets
  namespace: cyber-range
type: Opaque
data:
  # Base64 encoded values
  CYBER_RANGE_SECRET_KEY: <base64-encoded-secret>
  POSTGRES_PASSWORD: <base64-encoded-password>
  REDIS_PASSWORD: <base64-encoded-password>
  RABBITMQ_PASSWORD: <base64-encoded-password>
```

#### Database Deployment

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: cyber-range
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: "cyber_range"
        - name: POSTGRES_USER
          value: "cyber_range"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: cyber-range-secrets
              key: POSTGRES_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: cyber-range
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: cyber-range
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

#### Core Application Deployment

```yaml
# k8s/cyber-range-core.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cyber-range-core
  namespace: cyber-range
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cyber-range-core
  template:
    metadata:
      labels:
        app: cyber-range-core
    spec:
      containers:
      - name: cyber-range-core
        image: gan-cyber-range:latest
        env:
        - name: CYBER_RANGE_ENV
          valueFrom:
            configMapKeyRef:
              name: cyber-range-config
              key: CYBER_RANGE_ENV
        - name: CYBER_RANGE_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: cyber-range-secrets
              key: CYBER_RANGE_SECRET_KEY
        - name: DATABASE_URL
          value: "postgresql://cyber_range:$(POSTGRES_PASSWORD)@postgres:5432/cyber_range"
        ports:
        - containerPort: 8080
        - containerPort: 8443
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"

---
apiVersion: v1
kind: Service
metadata:
  name: cyber-range-core
  namespace: cyber-range
spec:
  selector:
    app: cyber-range-core
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: https
    port: 8443
    targetPort: 8443
  type: LoadBalancer
```

#### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cyber-range-core-hpa
  namespace: cyber-range
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cyber-range-core
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n cyber-range
kubectl get services -n cyber-range

# Check logs
kubectl logs -f deployment/cyber-range-core -n cyber-range

# Port forward for local access
kubectl port-forward service/cyber-range-core 8080:8080 -n cyber-range
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'cyber-range-core'
    static_configs:
      - targets: ['cyber-range-core:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
- name: cyber-range-alerts
  rules:
  - alert: HighCPUUsage
    expr: rate(cpu_usage_total[5m]) > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is above 80% for more than 5 minutes"

  - alert: HighMemoryUsage
    expr: memory_usage_percent > 90
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is above 90%"

  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service is down"
      description: "Service {{ $labels.instance }} is down"
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "GAN-Cyber-Range-v2 Overview",
    "panels": [
      {
        "title": "Active Scenarios",
        "type": "stat",
        "targets": [
          {
            "expr": "cyber_range_active_scenarios_total"
          }
        ]
      },
      {
        "title": "Attack Generation Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cyber_range_attacks_generated_total[5m])"
          }
        ]
      },
      {
        "title": "Detection Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "cyber_range_detection_rate"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

```yaml
# logging/logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "cyber-range" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} - %{LOGLEVEL:level} - %{GREEDYDATA:message}" }
    }
    
    if [level] == "ERROR" or [level] == "CRITICAL" {
      mutate {
        add_tag => [ "alert" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "cyber-range-%{+YYYY.MM.dd}"
  }
}
```

## Security Considerations

### Network Security

```bash
# Configure firewall rules
sudo ufw enable
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow 22/tcp   # SSH (restrict to specific IPs)
sudo ufw deny 8080/tcp  # Block direct access to application

# Configure fail2ban
sudo apt-get install fail2ban
sudo systemctl enable fail2ban
```

### Container Security

```yaml
# security-policies.yaml
apiVersion: v1
kind: SecurityContext
metadata:
  name: cyber-range-security
spec:
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
```

### Secrets Management

```bash
# Use external secret management
kubectl create secret generic cyber-range-secrets \
  --from-literal=db_password="$(vault kv get -field=password secret/cyber-range/db)" \
  --from-literal=secret_key="$(vault kv get -field=secret_key secret/cyber-range/app)"
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="cyber_range_backup_${TIMESTAMP}.sql.gz"

# Create backup
docker-compose exec postgres pg_dump -U cyber_range cyber_range | gzip > "${BACKUP_DIR}/${BACKUP_FILE}"

# Upload to S3
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}" s3://cyber-range-backups/database/

# Cleanup old backups (keep last 7 days)
find "${BACKUP_DIR}" -name "*.sql.gz" -mtime +7 -delete

echo "Backup completed: ${BACKUP_FILE}"
```

### Application Data Backup

```bash
#!/bin/bash
# backup_data.sh

BACKUP_DIR="/backups/data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup application data
tar -czf "${BACKUP_DIR}/cyber_range_data_${TIMESTAMP}.tar.gz" \
  data/ models/ logs/

# Upload to S3
aws s3 cp "${BACKUP_DIR}/cyber_range_data_${TIMESTAMP}.tar.gz" s3://cyber-range-backups/data/

echo "Data backup completed"
```

### Disaster Recovery

```bash
#!/bin/bash
# restore.sh

BACKUP_DATE=$1

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    echo "Example: $0 20231201_143000"
    exit 1
fi

# Stop services
docker-compose down

# Download backups from S3
aws s3 cp s3://cyber-range-backups/database/cyber_range_backup_${BACKUP_DATE}.sql.gz /tmp/
aws s3 cp s3://cyber-range-backups/data/cyber_range_data_${BACKUP_DATE}.tar.gz /tmp/

# Restore database
gunzip /tmp/cyber_range_backup_${BACKUP_DATE}.sql.gz
docker-compose up -d postgres
sleep 30
docker-compose exec postgres psql -U cyber_range cyber_range < /tmp/cyber_range_backup_${BACKUP_DATE}.sql

# Restore application data
tar -xzf /tmp/cyber_range_data_${BACKUP_DATE}.tar.gz

# Start services
docker-compose up -d

echo "Restoration completed"
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check logs
docker-compose logs cyber-range-core

# Check resource usage
docker stats

# Check port conflicts
netstat -tulpn | grep :8080
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
docker-compose exec cyber-range-core psql -h postgres -U cyber_range -d cyber_range

# Check database logs
docker-compose logs postgres

# Reset database
docker-compose down
docker volume rm cyber-range-prod_postgres_data
docker-compose up -d postgres
```

#### 3. Memory Issues

```bash
# Increase Docker memory limits
# Edit Docker Desktop settings or daemon.json

# Monitor memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check for memory leaks
docker-compose exec cyber-range-core python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

#### 4. SSL Certificate Issues

```bash
# Check certificate validity
openssl x509 -in deployment/nginx/ssl/cyber-range.crt -text -noout

# Regenerate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout deployment/nginx/ssl/cyber-range.key \
    -out deployment/nginx/ssl/cyber-range.crt
```

### Performance Optimization

#### 1. Database Tuning

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
SELECT pg_reload_conf();
```

#### 2. Application Tuning

```python
# config/performance.py
PERFORMANCE_CONFIG = {
    "worker_processes": 4,
    "max_requests": 1000,
    "max_requests_jitter": 100,
    "preload_app": True,
    "timeout": 30,
    "keepalive": 2
}
```

#### 3. Monitoring Performance

```bash
# Monitor application performance
curl http://localhost:8080/metrics | grep cyber_range

# Monitor system performance
top -p $(docker-compose exec cyber-range-core pgrep python)

# Database performance
docker-compose exec postgres psql -U cyber_range cyber_range -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"
```

This comprehensive deployment guide covers all aspects of deploying GAN-Cyber-Range-v2 from development to production environments, including Kubernetes deployments, monitoring, security considerations, and troubleshooting procedures.