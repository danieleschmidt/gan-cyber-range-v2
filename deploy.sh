#!/bin/bash
set -e

echo "Starting deployment for production environment..."

# Pre-deployment checks
echo "Running pre-deployment validation..."
python3 comprehensive_quality_gates.py
if [ $? -ne 0 ]; then
    echo "Quality gates failed. Aborting deployment."
    exit 1
fi

# Build and deploy
echo "Building container..."
docker-compose -f docker-compose.prod.yml build

echo "Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for health checks
echo "Waiting for services to be healthy..."
sleep 30

# Verify deployment
echo "Running post-deployment health checks..."
python3 health_check.py
if [ $? -eq 0 ]; then
    echo "Deployment successful!"
else
    echo "Health checks failed. Consider rollback."
    exit 1
fi

echo "Deployment completed successfully"
