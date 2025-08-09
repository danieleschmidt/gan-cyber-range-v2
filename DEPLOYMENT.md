# GAN-Cyber-Range-v2 Deployment Guide

This document provides comprehensive deployment instructions for the GAN-Cyber-Range-v2 platform across different environments.

## 🚀 Quick Start

### Local Development
```bash
# Clone and setup
git clone https://github.com/terragonlabs/gan-cyber-range-v2.git
cd gan-cyber-range-v2
make install

# Start development server
make dev
```

### Docker Compose (Recommended for Testing)
```bash
# Deploy locally with all services
make deploy-local

# Access services
# API: http://localhost:8000
# Grafana: http://localhost:3000 (admin/GrafanaAdmin123!)
```

### Kubernetes (Production)
```bash
# Deploy to existing Kubernetes cluster
make deploy-k8s

# Or use cloud deployment
make deploy-aws    # AWS EKS
make deploy-gcp    # Google GKE  
make deploy-azure  # Azure AKS
```

## 📋 Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 50 GB | 100+ GB |
| GPU | Optional | NVIDIA GPU with 8GB+ VRAM |

### Software Dependencies

- **Docker** 20.10+ and Docker Compose 2.0+
- **Python** 3.9+ (for development)
- **Kubernetes** 1.25+ (for production)
- **kubectl** and **helm** (for Kubernetes deployments)

### Cloud Provider CLIs (for cloud deployment)
- **AWS**: AWS CLI + eksctl
- **Google Cloud**: gcloud SDK
- **Azure**: Azure CLI

## 🏗️ Deployment Options

### 1. Local Development

Perfect for development and testing.

```bash
# Setup development environment
./scripts/setup-dev.sh

# Start development server
source venv/bin/activate
./scripts/dev/start-api.sh
```

**Services:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### 2. Docker Compose

Best for local testing with full stack.

```bash
# Deploy all services
./scripts/deploy.sh local --build

# View logs
docker-compose logs -f

# Stop services  
docker-compose down
```

**Services:**
- API: http://localhost:8000
- Nginx: http://localhost:80
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9091

### 3. Kubernetes

Production-ready deployment.

```bash
# Deploy to existing cluster
./scripts/deploy.sh kubernetes

# Or create new cluster
./scripts/deploy-cloud.sh --provider aws --region us-east-1
```

**Features:**
- Auto-scaling (HPA/VPA)
- Load balancing
- Rolling updates
- Health checks
- Persistent storage
- Monitoring & observability

### 4. Cloud Providers

#### AWS EKS
```bash
# Deploy to AWS
./scripts/deploy-cloud.sh --provider aws --region us-east-1 --nodes 3

# Access cluster
aws eks update-kubeconfig --region us-east-1 --name gan-cyber-range
kubectl get pods -n gan-cyber-range
```

#### Google Cloud GKE
```bash
# Deploy to GCP
./scripts/deploy-cloud.sh --provider gcp --region us-central1-a --nodes 3

# Access cluster
gcloud container clusters get-credentials gan-cyber-range --zone us-central1-a
kubectl get pods -n gan-cyber-range
```

#### Azure AKS
```bash
# Deploy to Azure
./scripts/deploy-cloud.sh --provider azure --region eastus --nodes 3

# Access cluster
az aks get-credentials --name gan-cyber-range --resource-group gan-cyber-range-rg
kubectl get pods -n gan-cyber-range
```

## ⚙️ Configuration

### Environment Variables

Create `.env.{environment}` files for different environments:

```bash
# .env.production
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0
JWT_SECRET=your-secure-secret-key
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_TOKEN=your-hf-token
```

### Kubernetes Configuration

Key configurations in `k8s/`:

```yaml
# k8s/configmap.yaml - Application settings
# k8s/secrets.yaml - Sensitive data
# k8s/api-deployment.yaml - Main application
# k8s/postgresql.yaml - Database
# k8s/redis.yaml - Cache
# k8s/monitoring.yaml - Observability
```

### Security Configuration

```yaml
# Network policies
# Resource quotas  
# Pod security standards
# RBAC permissions
# TLS certificates
# Secrets management
```

## 🔍 Monitoring & Observability

### Built-in Monitoring

- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and visualization
- **Jaeger**: Distributed tracing (optional)
- **ELK Stack**: Centralized logging (optional)

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Kubernetes health
kubectl get pods -n gan-cyber-range
kubectl describe deployment gan-cyber-range-api -n gan-cyber-range
```

### Metrics Endpoints

- Application metrics: `/metrics`
- System metrics: Prometheus scraping
- Custom metrics: Via MetricsCollector

## 🚨 Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check database status
kubectl get pods -n gan-cyber-range | grep postgres
kubectl logs postgresql-0 -n gan-cyber-range

# Reset database
make db-reset
```

#### 2. Memory/Resource Issues
```bash
# Check resource usage
kubectl top pods -n gan-cyber-range
kubectl describe node

# Adjust resource limits in k8s/api-deployment.yaml
```

#### 3. Image Pull Issues
```bash
# Build images locally
make docker-build

# Check image availability
docker images | grep terragon

# Push to registry
make docker-push
```

#### 4. Certificate/TLS Issues
```bash
# Check certificates
kubectl get certificates -n gan-cyber-range
kubectl describe certificate cyber-range-tls -n gan-cyber-range

# Regenerate certificates
kubectl delete certificate cyber-range-tls -n gan-cyber-range
kubectl apply -k k8s/
```

### Debugging Commands

```bash
# View logs
kubectl logs -l app.kubernetes.io/name=gan-cyber-range-api -n gan-cyber-range --tail=100

# Debug pod
kubectl exec -it deployment/gan-cyber-range-api -n gan-cyber-range -- bash

# Port forward for local access
kubectl port-forward service/nginx-service -n gan-cyber-range 8080:80

# Check events
kubectl get events -n gan-cyber-range --sort-by='.lastTimestamp'
```

## 📊 Performance Tuning

### Database Optimization
```bash
# PostgreSQL tuning in k8s/postgresql.yaml
shared_buffers = 1GB
effective_cache_size = 3GB
work_mem = 16MB
```

### Application Scaling
```bash
# Horizontal Pod Autoscaler
kubectl get hpa -n gan-cyber-range

# Vertical Pod Autoscaler (if enabled)
kubectl get vpa -n gan-cyber-range

# Manual scaling
kubectl scale deployment gan-cyber-range-api --replicas=5 -n gan-cyber-range
```

### Resource Requests/Limits
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
  limits:
    memory: "4Gi" 
    cpu: "2"
```

## 🔐 Security Best Practices

### Production Security Checklist

- [ ] Change default passwords
- [ ] Use strong JWT secrets
- [ ] Enable TLS/SSL
- [ ] Configure network policies
- [ ] Set up RBAC
- [ ] Enable pod security standards
- [ ] Scan images for vulnerabilities
- [ ] Encrypt data at rest
- [ ] Enable audit logging
- [ ] Regular security updates

### Network Security
```yaml
# Network policy example
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cyber-range-network-policy
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: gan-cyber-range-api
  policyTypes:
  - Ingress
  - Egress
```

## 🔄 Updates & Maintenance

### Rolling Updates
```bash
# Update deployment
kubectl set image deployment/gan-cyber-range-api api=terragon/gan-cyber-range-v2:v2.1.0 -n gan-cyber-range

# Check rollout status  
kubectl rollout status deployment/gan-cyber-range-api -n gan-cyber-range

# Rollback if needed
kubectl rollout undo deployment/gan-cyber-range-api -n gan-cyber-range
```

### Backup & Recovery
```bash
# Database backup
kubectl exec postgresql-0 -n gan-cyber-range -- pg_dump -U postgres gan_cyber_range > backup.sql

# Restore database
kubectl exec -i postgresql-0 -n gan-cyber-range -- psql -U postgres gan_cyber_range < backup.sql
```

### Maintenance Windows
```bash
# Drain nodes for maintenance
kubectl drain node-name --ignore-daemonsets

# Uncordon after maintenance
kubectl uncordon node-name
```

## 📚 Additional Resources

- [API Documentation](http://localhost:8000/docs)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Security Guide](docs/SECURITY.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🆘 Support

### Getting Help

1. **Documentation**: Check docs/ directory
2. **GitHub Issues**: Report bugs and request features
3. **Community Discord**: Real-time support
4. **Email**: support@terragon.com

### Emergency Contacts

- **Security Issues**: security@terragon.com
- **Critical Bugs**: critical@terragon.com
- **Infrastructure**: infra@terragon.com

---

**Last Updated**: 2024-12-19  
**Version**: 2.0.0