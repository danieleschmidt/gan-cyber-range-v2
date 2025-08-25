#!/bin/bash
set -e

echo "🚀 GAN Cyber Range - Production Deployment"
echo "=========================================="

# Create directories
mkdir -p logs data

# Build and start
echo "🔨 Building images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo "✅ Deployment completed!"
echo "🌐 Access at: http://localhost"
echo "📊 Status: docker-compose ps"
echo "📝 Logs: docker-compose logs -f"
