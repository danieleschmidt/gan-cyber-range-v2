#!/bin/bash
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
