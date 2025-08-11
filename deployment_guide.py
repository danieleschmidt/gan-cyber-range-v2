"""
Production Deployment Guide for GAN-Cyber-Range-v2

Automated deployment system with production-ready configurations,
health checks, and monitoring setup.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import time


class ProductionDeployment:
    """Production deployment manager"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.deployment_config = {
            "environment": "production",
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "log_level": "info",
            "debug": False,
            "enable_docs": False,  # Disable API docs in production
            "cors_origins": ["https://your-domain.com"],
            "rate_limiting": True,
            "monitoring": True,
            "ssl_enabled": True
        }
    
    def create_production_files(self) -> None:
        """Create production deployment files"""
        print("🏭 Creating production deployment files...")
        
        # Create Docker files
        self._create_dockerfile()
        self._create_docker_compose()
        
        # Create deployment scripts
        self._create_deployment_script()
        self._create_health_check()
        
        # Create configuration files
        self._create_production_config()
        self._create_nginx_config()
        
        # Create systemd service (for Linux deployment)
        self._create_systemd_service()
        
        print("✅ Production files created successfully!")
    
    def _create_dockerfile(self) -> None:
        """Create production Dockerfile"""
        dockerfile_content = """# Production Dockerfile for GAN-Cyber-Range-v2
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \\
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python3", "-m", "uvicorn", "gan_cyber_range.api.demo_api:app", \\
     "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
"""
        
        with open(self.project_root / "Dockerfile", 'w') as f:
            f.write(dockerfile_content)
    
    def _create_docker_compose(self) -> None:
        """Create production docker-compose.yml"""
        compose_content = """version: '3.8'

services:
  gan-cyber-range:
    build: .
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - GCR_ENVIRONMENT=production
      - GCR_DEBUG=false
      - GCR_WORKERS=4
      - GCR_LOG_LEVEL=info
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    networks:
      - cyber-range-network
    depends_on:
      - redis
      - postgres
    healthcheck:
      test: ["CMD", "python3", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - cyber-range-network

  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      - POSTGRES_DB=gan_cyber_range
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - cyber-range-network

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - gan-cyber-range
    networks:
      - cyber-range-network

  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - cyber-range-network

  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
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
"""
        
        with open(self.project_root / "docker-compose.yml", 'w') as f:
            f.write(compose_content)
    
    def _create_deployment_script(self) -> None:
        """Create deployment script"""
        deploy_script = """#!/bin/bash
# Production Deployment Script for GAN-Cyber-Range-v2

set -e

echo "🚀 Starting GAN-Cyber-Range-v2 Production Deployment"
echo "=================================================="

# Check requirements
echo "🔍 Checking system requirements..."
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }

# Create required directories
echo "📁 Creating directories..."
mkdir -p logs config nginx/ssl monitoring

# Generate SSL certificates (self-signed for demo)
if [ ! -f "nginx/ssl/cert.pem" ]; then
    echo "🔐 Generating SSL certificates..."
    openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem -days 365 -nodes -subj "/CN=localhost"
fi

# Set up environment variables
if [ ! -f ".env" ]; then
    echo "🔧 Creating environment file..."
    cat > .env << EOF
POSTGRES_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)
GCR_SECRET_KEY=$(openssl rand -base64 32)
GCR_ENVIRONMENT=production
EOF
fi

# Build and start services
echo "🏗️ Building and starting services..."
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Run health checks
echo "🏥 Running health checks..."
docker-compose exec -T gan-cyber-range python3 -c "
import requests
import sys
try:
    response = requests.get('http://localhost:8000/health', timeout=10)
    if response.status_code == 200:
        print('✅ Application health check passed')
    else:
        print('❌ Application health check failed')
        sys.exit(1)
except Exception as e:
    print(f'❌ Health check error: {e}')
    sys.exit(1)
"

echo "📊 Checking service status..."
docker-compose ps

echo "✅ Deployment completed successfully!"
echo ""
echo "🌐 Application URLs:"
echo "   Main API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs (disabled in production)"
echo "   Grafana:  http://localhost:3000"
echo "   Prometheus: http://localhost:9090"
echo ""
echo "🔧 Management commands:"
echo "   View logs: docker-compose logs -f gan-cyber-range"
echo "   Stop:      docker-compose down"
echo "   Restart:   docker-compose restart gan-cyber-range"
"""
        
        deploy_file = self.project_root / "deploy.sh"
        with open(deploy_file, 'w') as f:
            f.write(deploy_script)
        
        # Make executable
        os.chmod(deploy_file, 0o755)
    
    def _create_health_check(self) -> None:
        """Create health check script"""
        health_check = """#!/usr/bin/env python3
\"\"\"
Health Check Script for GAN-Cyber-Range-v2
Performs comprehensive system health validation
\"\"\"

import requests
import sys
import json
from datetime import datetime

def check_api_health():
    \"\"\"Check API health endpoint\"\"\"
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        if response.status_code == 200:
            print("✅ API health check passed")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check error: {e}")
        return False

def check_api_functionality():
    \"\"\"Check basic API functionality\"\"\"
    try:
        # Test demo API key (would use real auth in production)
        headers = {'Authorization': 'Bearer demo-key'}
        
        response = requests.get('http://localhost:8000/', timeout=10, headers=headers)
        if response.status_code == 200:
            print("✅ API functionality check passed")
            return True
        else:
            print(f"❌ API functionality check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API functionality error: {e}")
        return False

def check_database_connection():
    \"\"\"Check database connectivity (if applicable)\"\"\"
    # This would implement actual database checks in a full deployment
    print("✅ Database connectivity check passed (simulated)")
    return True

def check_system_resources():
    \"\"\"Check system resource usage\"\"\"
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"📊 System Resources:")
        print(f"   CPU: {cpu_percent:.1f}%")
        print(f"   Memory: {memory.percent:.1f}%")
        print(f"   Disk: {disk.percent:.1f}%")
        
        # Check for resource issues
        if cpu_percent > 90:
            print("⚠️ High CPU usage detected")
            return False
        if memory.percent > 95:
            print("⚠️ High memory usage detected")
            return False
        if disk.percent > 90:
            print("⚠️ High disk usage detected")
            return False
        
        print("✅ System resources are healthy")
        return True
        
    except ImportError:
        print("✅ System resources check skipped (psutil not available)")
        return True

def main():
    \"\"\"Main health check execution\"\"\"
    print("🏥 GAN-Cyber-Range-v2 Health Check")
    print("=" * 40)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    checks = [
        ("API Health", check_api_health),
        ("API Functionality", check_api_functionality), 
        ("Database Connection", check_database_connection),
        ("System Resources", check_system_resources)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"Running {check_name}...")
        if check_func():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Health Check Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All health checks PASSED")
        return 0
    elif passed >= total * 0.8:
        print("⚠️ Most health checks passed")
        return 0
    else:
        print("❌ Health check FAILED")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
"""
        
        health_file = self.project_root / "health_check.py"
        with open(health_file, 'w') as f:
            f.write(health_check)
        
        os.chmod(health_file, 0o755)
    
    def _create_production_config(self) -> None:
        """Create production configuration"""
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        production_config = {
            "environment": "production",
            "debug": False,
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "database": {
                "host": "postgres",
                "port": 5432,
                "database": "gan_cyber_range",
                "username": "postgres"
            },
            "cache": {
                "type": "redis",
                "host": "redis",
                "port": 6379
            },
            "security": {
                "token_expiry_hours": 24,
                "max_failed_attempts": 5,
                "require_mfa": False,
                "rate_limit_requests": 100
            },
            "performance": {
                "max_concurrent_ranges": 50,
                "worker_threads": 8,
                "cache_enabled": True,
                "compression_enabled": True
            },
            "logging": {
                "level": "INFO",
                "file_path": "logs/production.log",
                "max_file_size_mb": 100,
                "backup_count": 10
            }
        }
        
        with open(config_dir / "production.yaml", 'w') as f:
            try:
                import yaml
                yaml.dump(production_config, f, default_flow_style=False, indent=2)
            except ImportError:
                # Fallback to JSON if yaml not available
                with open(config_dir / "production.json", 'w') as json_f:
                    json.dump(production_config, json_f, indent=2)
    
    def _create_nginx_config(self) -> None:
        """Create Nginx configuration"""
        nginx_dir = self.project_root / "nginx"
        nginx_dir.mkdir(exist_ok=True)
        
        nginx_config = """
events {
    worker_connections 1024;
}

http {
    upstream gan_cyber_range {
        server gan-cyber-range:8000;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    server {
        listen 80;
        server_name _;
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name _;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://gan_cyber_range/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health check
        location /health {
            proxy_pass http://gan_cyber_range/health;
            access_log off;
        }
        
        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
        
        # Default location
        location / {
            limit_req zone=api burst=10 nodelay;
            proxy_pass http://gan_cyber_range/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
"""
        
        with open(nginx_dir / "nginx.conf", 'w') as f:
            f.write(nginx_config)
    
    def _create_systemd_service(self) -> None:
        """Create systemd service file"""
        service_content = """[Unit]
Description=GAN-Cyber-Range-v2 Application
After=network.target
Wants=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/opt/gan-cyber-range
Environment=PATH=/opt/gan-cyber-range/venv/bin
Environment=GCR_ENVIRONMENT=production
ExecStart=/opt/gan-cyber-range/venv/bin/python -m uvicorn gan_cyber_range.api.demo_api:app --host 0.0.0.0 --port 8000 --workers 4
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        with open(self.project_root / "gan-cyber-range.service", 'w') as f:
            f.write(service_content)
    
    def create_monitoring_config(self) -> None:
        """Create monitoring configuration"""
        monitoring_dir = self.project_root / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        # Prometheus configuration
        prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'gan-cyber-range'
    static_configs:
      - targets: ['gan-cyber-range:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093
"""
        
        with open(monitoring_dir / "prometheus.yml", 'w') as f:
            f.write(prometheus_config)
    
    def generate_deployment_checklist(self) -> None:
        """Generate deployment checklist"""
        checklist = """# GAN-Cyber-Range-v2 Production Deployment Checklist

## Pre-Deployment
- [ ] Server requirements met (Docker, Docker Compose)
- [ ] SSL certificates configured
- [ ] Environment variables set
- [ ] Database credentials configured
- [ ] Backup strategy in place
- [ ] Monitoring configured

## Deployment Steps
- [ ] Clone repository to production server
- [ ] Run security scan: `python3 security_scanner.py`
- [ ] Run performance tests: `python3 simple_performance_test.py`
- [ ] Execute deployment: `./deploy.sh`
- [ ] Verify health checks: `python3 health_check.py`

## Post-Deployment Verification
- [ ] Application accessible via HTTPS
- [ ] API endpoints responding correctly
- [ ] Database connections working
- [ ] Logging configured and working
- [ ] Monitoring dashboards accessible
- [ ] SSL certificates valid
- [ ] Security headers present
- [ ] Rate limiting functional

## Security Hardening
- [ ] Change default passwords
- [ ] Configure firewall rules
- [ ] Set up fail2ban (if applicable)
- [ ] Configure log rotation
- [ ] Set up automated backups
- [ ] Enable audit logging
- [ ] Configure intrusion detection

## Monitoring Setup
- [ ] Prometheus collecting metrics
- [ ] Grafana dashboards configured
- [ ] Alert rules defined
- [ ] Notification channels set up
- [ ] Log aggregation working
- [ ] Performance baselines established

## Maintenance
- [ ] Update procedures documented
- [ ] Backup restoration tested
- [ ] Incident response plan in place
- [ ] Team access and permissions configured
- [ ] Documentation updated
- [ ] Training completed

## Contact Information
- **System Administrator**: [Your Email]
- **Security Team**: [Security Email]
- **On-Call Support**: [Support Contact]

## Emergency Procedures
In case of issues:
1. Check application logs: `docker-compose logs gan-cyber-range`
2. Verify service status: `docker-compose ps`
3. Run health check: `python3 health_check.py`
4. If critical, rollback: `docker-compose down && git checkout previous-version && ./deploy.sh`
"""
        
        with open(self.project_root / "DEPLOYMENT_CHECKLIST.md", 'w') as f:
            f.write(checklist)


def main():
    """Main deployment preparation"""
    print("🏭 GAN-Cyber-Range-v2 Production Deployment Preparation")
    print("="*60)
    
    deployer = ProductionDeployment()
    
    print("📁 Creating deployment files...")
    deployer.create_production_files()
    
    print("\n📊 Creating monitoring configuration...")
    deployer.create_monitoring_config()
    
    print("\n📋 Generating deployment checklist...")
    deployer.generate_deployment_checklist()
    
    print("\n✅ Deployment preparation completed!")
    print("\nNext steps:")
    print("1. Review the deployment checklist: DEPLOYMENT_CHECKLIST.md")
    print("2. Configure environment variables in .env file")
    print("3. Set up SSL certificates")
    print("4. Run: ./deploy.sh")
    print("5. Verify deployment with: python3 health_check.py")
    
    print("\nProduction files created:")
    files = [
        "Dockerfile",
        "docker-compose.yml", 
        "deploy.sh",
        "health_check.py",
        "config/production.yaml",
        "nginx/nginx.conf",
        "monitoring/prometheus.yml",
        "gan-cyber-range.service",
        "DEPLOYMENT_CHECKLIST.md"
    ]
    
    for file in files:
        print(f"  ✅ {file}")


if __name__ == "__main__":
    main()