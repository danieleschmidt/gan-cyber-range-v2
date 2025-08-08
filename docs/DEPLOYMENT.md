# Deployment Guide for GAN-Cyber-Range-v2

## Overview

This guide covers deployment options for GAN-Cyber-Range-v2, from development setups to production-scale deployments.

## System Requirements

### Minimum Requirements

- **CPU**: 4 cores, 2.5 GHz
- **RAM**: 8 GB
- **Storage**: 50 GB available space
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10/11
- **Python**: 3.9+
- **Docker**: 20.10+

### Recommended Requirements

- **CPU**: 16+ cores, 3.0 GHz (Intel Xeon or AMD EPYC)
- **RAM**: 32+ GB
- **Storage**: 500+ GB SSD
- **GPU**: NVIDIA RTX 3080+ with 10GB+ VRAM (for GAN training)
- **Network**: Gigabit Ethernet

### Production Requirements

- **CPU**: 32+ cores across multiple nodes
- **RAM**: 128+ GB total
- **Storage**: 2+ TB NVMe SSD with backup
- **GPU**: Multiple NVIDIA A100 or V100 GPUs
- **Network**: 10 Gigabit with redundancy

## Installation Methods

### 1. Quick Start (Development)

```bash
# Clone repository
git clone https://github.com/terragonlabs/gan-cyber-range-v2.git
cd gan-cyber-range-v2

# Create virtual environment
python -m venv cyber-range-env
source cyber-range-env/bin/activate  # Linux/macOS
# cyber-range-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Verify installation
python -c "import gan_cyber_range; print('Installation successful')"
```

### 2. Docker Deployment

#### Single Container

```bash
# Build image
docker build -t gan-cyber-range-v2 .

# Run container
docker run -d \
  --name cyber-range \
  -p 8080:8080 \
  -p 3000:3000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v /var/run/docker.sock:/var/run/docker.sock \
  gan-cyber-range-v2
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  cyber-range:
    build: .
    ports:
      - "8080:8080"
      - "3000:3000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - CYBER_RANGE_LOG_LEVEL=INFO
      - CYBER_RANGE_GPU_ENABLED=true
    depends_on:
      - redis
      - postgres
    networks:
      - cyber-range-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - cyber-range-network

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=cyber_range
      - POSTGRES_USER=cyber_range
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - cyber-range-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - cyber-range-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - cyber-range-network

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  cyber-range-network:
    driver: bridge
```

Deploy with:
```bash
docker-compose up -d
```

### 3. Kubernetes Deployment

#### Namespace and Configuration

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cyber-range
  labels:
    name: cyber-range

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cyber-range-config
  namespace: cyber-range
data:
  CYBER_RANGE_LOG_LEVEL: "INFO"
  CYBER_RANGE_CACHE_BACKEND: "redis"
  CYBER_RANGE_REDIS_URL: "redis://redis-service:6379"
  CYBER_RANGE_GPU_ENABLED: "true"
```

#### Main Application

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cyber-range-app
  namespace: cyber-range
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cyber-range
  template:
    metadata:
      labels:
        app: cyber-range
    spec:
      containers:
      - name: cyber-range
        image: gan-cyber-range-v2:latest
        ports:
        - containerPort: 8080
        - containerPort: 3000
        envFrom:
        - configMapRef:
            name: cyber-range-config
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: cyber-range-data-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: cyber-range-logs-pvc

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: cyber-range-service
  namespace: cyber-range
spec:
  selector:
    app: cyber-range
  ports:
  - name: api
    port: 8080
    targetPort: 8080
  - name: dashboard
    port: 3000
    targetPort: 3000
  type: LoadBalancer
```

#### GPU Node Pool (for GAN training)

```yaml
# k8s/gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cyber-range-gpu
  namespace: cyber-range
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cyber-range-gpu
  template:
    metadata:
      labels:
        app: cyber-range-gpu
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-v100
      containers:
      - name: cyber-range-gpu
        image: gan-cyber-range-v2:gpu
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
        envFrom:
        - configMapRef:
            name: cyber-range-config
```

Deploy to Kubernetes:
```bash
kubectl apply -f k8s/
```

### 4. Cloud Deployments

#### AWS EKS

```bash
# Create EKS cluster
eksctl create cluster \
  --name cyber-range-cluster \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10

# Create GPU node group
eksctl create nodegroup \
  --cluster cyber-range-cluster \
  --region us-west-2 \
  --name gpu-workers \
  --node-type p3.2xlarge \
  --nodes 1 \
  --nodes-min 0 \
  --nodes-max 5 \
  --node-ami-family AmazonLinux2

# Deploy application
kubectl apply -f k8s/
```

#### Google GKE

```bash
# Create GKE cluster
gcloud container clusters create cyber-range-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Create GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster cyber-range-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-v100,count=1 \
  --num-nodes 1 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 3

# Deploy application
kubectl apply -f k8s/
```

#### Azure AKS

```bash
# Create resource group
az group create --name CyberRangeRG --location eastus

# Create AKS cluster
az aks create \
  --resource-group CyberRangeRG \
  --name cyber-range-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Add GPU node pool
az aks nodepool add \
  --resource-group CyberRangeRG \
  --cluster-name cyber-range-cluster \
  --name gpupool \
  --node-count 1 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 0 \
  --max-count 3

# Deploy application
kubectl apply -f k8s/
```

## Configuration

### Environment Variables

```bash
# Core Configuration
export CYBER_RANGE_LOG_LEVEL=INFO
export CYBER_RANGE_DATA_DIR=/opt/cyber-range/data
export CYBER_RANGE_LOG_DIR=/opt/cyber-range/logs

# Database Configuration
export CYBER_RANGE_DB_URL=postgresql://user:pass@localhost/cyber_range
export CYBER_RANGE_REDIS_URL=redis://localhost:6379

# Security Configuration
export CYBER_RANGE_SECRET_KEY=your-secret-key-here
export CYBER_RANGE_ENCRYPTION_KEY=base64-encoded-key
export CYBER_RANGE_JWT_SECRET=jwt-secret-key

# Performance Configuration
export CYBER_RANGE_GPU_ENABLED=true
export CYBER_RANGE_MAX_WORKERS=8
export CYBER_RANGE_CACHE_BACKEND=redis

# Network Configuration
export CYBER_RANGE_BIND_HOST=0.0.0.0
export CYBER_RANGE_API_PORT=8080
export CYBER_RANGE_DASHBOARD_PORT=3000
```

### Configuration File

```yaml
# config/production.yaml
server:
  host: 0.0.0.0
  api_port: 8080
  dashboard_port: 3000
  workers: 8
  timeout: 300

database:
  url: postgresql://user:pass@postgres:5432/cyber_range
  pool_size: 20
  max_overflow: 30

cache:
  backend: redis
  url: redis://redis:6379
  default_ttl: 3600
  max_memory: 1000000000  # 1GB

security:
  require_auth: true
  session_timeout: 3600
  rate_limit: 1000
  enable_audit: true

logging:
  level: INFO
  format: json
  file_rotation: daily
  retention_days: 30

monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 30

gan:
  default_architecture: wasserstein
  max_training_time: 7200  # 2 hours
  checkpoint_interval: 1000
  auto_save: true

range:
  max_concurrent_ranges: 10
  default_isolation: container
  auto_cleanup: true
  snapshot_retention: 7  # days
```

## Security Hardening

### SSL/TLS Configuration

```bash
# Generate certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configure nginx proxy
# /etc/nginx/sites-available/cyber-range
server {
    listen 443 ssl http2;
    server_name cyber-range.example.com;
    
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /dashboard {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 443/tcp
sudo ufw allow 80/tcp
sudo ufw enable

# iptables
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -j DROP
```

### Container Security

```dockerfile
# Dockerfile.security
FROM ubuntu:20.04

# Create non-root user
RUN useradd -r -s /bin/false cyberrange

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Copy application
COPY --chown=cyberrange:cyberrange . /app
WORKDIR /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Set security headers
USER cyberrange
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["python3", "-m", "gan_cyber_range.server"]
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'cyber-range'
    static_configs:
      - targets: ['cyber-range:9090']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'docker'
    static_configs:
      - targets: ['docker-exporter:9323']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "Cyber Range Metrics",
    "panels": [
      {
        "title": "Active Ranges",
        "type": "stat",
        "targets": [
          {
            "expr": "cyber_range_active_ranges",
            "legendFormat": "Active Ranges"
          }
        ]
      },
      {
        "title": "Attack Generation Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cyber_range_attacks_generated_total[5m])",
            "legendFormat": "Attacks/sec"
          }
        ]
      },
      {
        "title": "Training Jobs",
        "type": "table",
        "targets": [
          {
            "expr": "cyber_range_training_jobs",
            "legendFormat": "Training Jobs"
          }
        ]
      }
    ]
  }
}
```

### Log Aggregation

```yaml
# logging/fluentd.conf
<source>
  @type tail
  path /app/logs/*.log
  pos_file /var/log/td-agent/cyber-range.log.pos
  tag cyber-range.*
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S
</source>

<match cyber-range.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name cyber-range
  type_name _doc
</match>
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# PostgreSQL backup
pg_dump -h postgres -U cyber_range cyber_range > \
  $BACKUP_DIR/db_backup_$DATE.sql

# Redis backup
redis-cli --rdb $BACKUP_DIR/redis_backup_$DATE.rdb

# Compress backups
gzip $BACKUP_DIR/db_backup_$DATE.sql
gzip $BACKUP_DIR/redis_backup_$DATE.rdb

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
```

### Data Volume Backup

```bash
#!/bin/bash
# volume-backup.sh

# Create data snapshot
docker run --rm \
  -v cyber-range_data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/data_$(date +%Y%m%d).tar.gz -C /data .

# Upload to cloud storage
aws s3 cp backups/data_$(date +%Y%m%d).tar.gz \
  s3://cyber-range-backups/data/
```

## Scaling and Performance

### Horizontal Scaling

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cyber-range-hpa
  namespace: cyber-range
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cyber-range-app
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

### Load Balancing

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cyber-range-ingress
  namespace: cyber-range
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
spec:
  tls:
  - hosts:
    - cyber-range.example.com
    secretName: cyber-range-tls
  rules:
  - host: cyber-range.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: cyber-range-service
            port:
              number: 8080
```

## Troubleshooting

### Common Issues

1. **Out of Memory during GAN training**
   ```bash
   # Increase memory limits
   docker run --memory=16g gan-cyber-range-v2
   ```

2. **Docker socket permission denied**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   newgrp docker
   ```

3. **GPU not detected**
   ```bash
   # Install NVIDIA container toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

### Debug Mode

```bash
# Enable debug logging
export CYBER_RANGE_LOG_LEVEL=DEBUG

# Run with debug flags
python -m gan_cyber_range.server --debug --verbose

# Check container logs
docker logs -f cyber-range

# Monitor resource usage
docker stats cyber-range
```

### Health Checks

```bash
# API health check
curl http://localhost:8080/health

# Database connectivity
curl http://localhost:8080/health/db

# Redis connectivity  
curl http://localhost:8080/health/cache

# System metrics
curl http://localhost:9090/metrics
```

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Update security patches, check logs
2. **Monthly**: Backup verification, performance review
3. **Quarterly**: Security audit, dependency updates
4. **Annually**: Hardware refresh planning, disaster recovery testing

### Automated Maintenance

```bash
#!/bin/bash
# maintenance.sh

# Update system packages
apt-get update && apt-get upgrade -y

# Clean Docker images
docker image prune -a -f

# Rotate logs
logrotate /etc/logrotate.d/cyber-range

# Check disk space
df -h | grep -E "9[0-9]%" && echo "Warning: Disk space low"

# Restart services if needed
systemctl status cyber-range || systemctl restart cyber-range
```

This deployment guide provides comprehensive coverage for deploying GAN-Cyber-Range-v2 in various environments, from development to production-scale deployments.