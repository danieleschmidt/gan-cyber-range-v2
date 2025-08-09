# GAN-Cyber-Range-v2 Makefile
# Provides convenient commands for development and deployment

.PHONY: help install dev test lint format build deploy clean docker-build docker-push

# Default target
help:
	@echo "GAN-Cyber-Range-v2 Development Commands"
	@echo "======================================="
	@echo ""
	@echo "Development:"
	@echo "  install     Install development dependencies"
	@echo "  dev         Start development environment"
	@echo "  test        Run tests"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code"
	@echo "  clean       Clean temporary files"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker images"
	@echo "  docker-push     Push Docker images to registry"
	@echo "  docker-run      Run application in Docker"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-local    Deploy locally with Docker Compose"
	@echo "  deploy-k8s      Deploy to Kubernetes"
	@echo "  deploy-cloud    Deploy to cloud provider"
	@echo ""
	@echo "Utilities:"
	@echo "  docs           Generate documentation"
	@echo "  security       Run security checks"
	@echo "  perf-test      Run performance tests"
	@echo ""

# Variables
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := terragon/gan-cyber-range-v2
DOCKER_TAG := latest
REGISTRY := registry.terragon.com

# Development environment
install:
	@echo "Setting up development environment..."
	./scripts/setup-dev.sh

dev:
	@echo "Starting development server..."
	./scripts/dev/start-api.sh

# Testing
test:
	@echo "Running tests..."
	source venv/bin/activate && \
	export $$(cat .env.development | grep -v '^#' | xargs) && \
	pytest tests/ -v --cov=gan_cyber_range --cov-report=html --cov-report=term

test-security:
	@echo "Running security tests..."
	source venv/bin/activate && \
	export $$(cat .env.development | grep -v '^#' | xargs) && \
	pytest tests/test_security.py -v

test-integration:
	@echo "Running integration tests..."
	source venv/bin/activate && \
	export $$(cat .env.development | grep -v '^#' | xargs) && \
	pytest tests/test_integration.py -v

test-api:
	@echo "Running API tests..."
	source venv/bin/activate && \
	export $$(cat .env.development | grep -v '^#' | xargs) && \
	pytest tests/test_api.py -v

# Code quality
lint:
	@echo "Running linting checks..."
	source venv/bin/activate && \
	flake8 gan_cyber_range/ tests/ --max-line-length=88 --extend-ignore=E203 && \
	mypy gan_cyber_range/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	source venv/bin/activate && \
	black gan_cyber_range/ tests/ scripts/ && \
	isort gan_cyber_range/ tests/ scripts/

security:
	@echo "Running security checks..."
	source venv/bin/activate && \
	bandit -r gan_cyber_range/ -f json -o security-report.json || true && \
	safety check --json --output safety-report.json || true && \
	echo "Security reports generated: security-report.json, safety-report.json"

# Docker operations
docker-build:
	@echo "Building Docker images..."
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker build -f docker/Dockerfile --target development -t $(DOCKER_IMAGE):dev .

docker-push:
	@echo "Pushing Docker images to registry..."
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run:
	@echo "Running application in Docker..."
	docker-compose up -d

docker-stop:
	@echo "Stopping Docker containers..."
	docker-compose down

docker-logs:
	@echo "Viewing Docker logs..."
	docker-compose logs -f

# Deployment
deploy-local:
	@echo "Deploying locally..."
	./scripts/deploy.sh local --build

deploy-local-update:
	@echo "Updating local deployment..."
	./scripts/deploy.sh local --update

deploy-k8s:
	@echo "Deploying to Kubernetes..."
	./scripts/deploy.sh kubernetes

deploy-k8s-update:
	@echo "Updating Kubernetes deployment..."
	./scripts/deploy.sh kubernetes --update

deploy-cloud:
	@echo "Deploying to cloud..."
	./scripts/deploy-cloud.sh

deploy-aws:
	@echo "Deploying to AWS..."
	./scripts/deploy-cloud.sh --provider aws

deploy-gcp:
	@echo "Deploying to GCP..."
	./scripts/deploy-cloud.sh --provider gcp

deploy-azure:
	@echo "Deploying to Azure..."
	./scripts/deploy-cloud.sh --provider azure

# Documentation
docs:
	@echo "Generating documentation..."
	./scripts/dev/generate-docs.sh
	@echo "Documentation generated in docs/"

api-docs:
	@echo "Starting API documentation server..."
	source venv/bin/activate && \
	export $$(cat .env.development | grep -v '^#' | xargs) && \
	uvicorn gan_cyber_range.api.main:app --host 0.0.0.0 --port 8000 &
	@echo "API docs available at: http://localhost:8000/docs"

# Performance testing
perf-test:
	@echo "Running performance tests..."
	source venv/bin/activate && \
	export $$(cat .env.development | grep -v '^#' | xargs) && \
	python -c "
from gan_cyber_range.utils.performance import global_optimizer
import asyncio

async def run_perf_test():
    await global_optimizer.start_monitoring()
    print('Performance monitoring started')
    
    # Run for 30 seconds
    await asyncio.sleep(30)
    
    stats = global_optimizer.get_comprehensive_stats()
    print('Performance Statistics:')
    for key, value in stats.items():
        print(f'  {key}: {value}')
    
    await global_optimizer.stop_monitoring()

asyncio.run(run_perf_test())
"

load-test:
	@echo "Running load tests..."
	@if command -v hey >/dev/null 2>&1; then \
		hey -n 1000 -c 10 http://localhost:8000/health; \
	elif command -v ab >/dev/null 2>&1; then \
		ab -n 1000 -c 10 http://localhost:8000/health; \
	else \
		echo "Install 'hey' or 'apache2-utils' for load testing"; \
	fi

# Database operations
db-init:
	@echo "Initializing database..."
	source venv/bin/activate && \
	export $$(cat .env.development | grep -v '^#' | xargs) && \
	python -c "
import asyncio
from gan_cyber_range.db.database import init_database

async def main():
    await init_database()
    print('Database initialized successfully')

asyncio.run(main())
"

db-migrate:
	@echo "Running database migrations..."
	source venv/bin/activate && \
	export $$(cat .env.development | grep -v '^#' | xargs) && \
	alembic upgrade head

db-reset:
	@echo "Resetting database..."
	docker-compose down
	docker volume rm $$(docker volume ls -q | grep postgres) || true
	docker-compose up -d postgresql redis
	sleep 10
	$(MAKE) db-init

# Monitoring and logs
logs:
	@echo "Viewing application logs..."
	docker-compose logs -f api

logs-all:
	@echo "Viewing all service logs..."
	docker-compose logs -f

monitor:
	@echo "Starting monitoring dashboard..."
	@echo "Grafana: http://localhost:3000 (admin/GrafanaAdmin123!)"
	@echo "Prometheus: http://localhost:9091"
	docker-compose up -d prometheus grafana

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	rm -rf .pytest_cache
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -f security-report.json safety-report.json

clean-docker:
	@echo "Cleaning Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

clean-all: clean clean-docker
	@echo "Full cleanup completed"

# Release management
version:
	@echo "Current version:"
	@grep "version" setup.py | grep -oE '[0-9]+\.[0-9]+\.[0-9]+'

bump-patch:
	@echo "Bumping patch version..."
	@python scripts/bump_version.py patch

bump-minor:
	@echo "Bumping minor version..."
	@python scripts/bump_version.py minor

bump-major:
	@echo "Bumping major version..."
	@python scripts/bump_version.py major

release: test lint security
	@echo "Creating release..."
	@echo "All checks passed. Ready for release."

# CI/CD helpers
ci-test:
	@echo "Running CI tests..."
	$(MAKE) test lint security

ci-build:
	@echo "Running CI build..."
	$(MAKE) docker-build

ci-deploy:
	@echo "Running CI deployment..."
	$(MAKE) deploy-k8s

# Development utilities
shell:
	@echo "Starting Python shell with project context..."
	source venv/bin/activate && \
	export $$(cat .env.development | grep -v '^#' | xargs) && \
	python -c "
from gan_cyber_range.core.attack_gan import AttackGAN
from gan_cyber_range.core.cyber_range import CyberRange
from gan_cyber_range.red_team.llm_adversary import RedTeamLLM
from gan_cyber_range.db.database import get_database
print('GAN-Cyber-Range-v2 Development Shell')
print('Available imports: AttackGAN, CyberRange, RedTeamLLM, get_database')
print('Starting IPython...')
import IPython; IPython.start_ipython(argv=[])
"

notebook:
	@echo "Starting Jupyter notebook..."
	source venv/bin/activate && \
	export $$(cat .env.development | grep -v '^#' | xargs) && \
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Status checks
status:
	@echo "System Status:"
	@echo "=============="
	@if [ -f "venv/bin/activate" ]; then echo "✓ Virtual environment exists"; else echo "✗ Virtual environment missing"; fi
	@if [ -f ".env.development" ]; then echo "✓ Development config exists"; else echo "✗ Development config missing"; fi
	@if docker info >/dev/null 2>&1; then echo "✓ Docker is running"; else echo "✗ Docker is not running"; fi
	@if docker-compose ps >/dev/null 2>&1; then echo "✓ Docker Compose available"; else echo "✗ Docker Compose not available"; fi
	@if kubectl cluster-info >/dev/null 2>&1; then echo "✓ Kubernetes cluster accessible"; else echo "✗ Kubernetes cluster not accessible"; fi

health:
	@echo "Health Check:"
	@echo "============"
	@curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "API not available"

# Help for specific sections
help-dev:
	@echo "Development Commands:"
	@echo "  make install     - Set up development environment"
	@echo "  make dev         - Start development server"
	@echo "  make test        - Run all tests"
	@echo "  make lint        - Check code quality"
	@echo "  make format      - Format code"
	@echo "  make shell       - Start Python shell"
	@echo "  make notebook    - Start Jupyter notebook"

help-docker:
	@echo "Docker Commands:"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-run   - Run with Docker Compose"
	@echo "  make docker-stop  - Stop Docker services"
	@echo "  make docker-logs  - View Docker logs"

help-deploy:
	@echo "Deployment Commands:"
	@echo "  make deploy-local  - Deploy locally"
	@echo "  make deploy-k8s    - Deploy to Kubernetes"
	@echo "  make deploy-cloud  - Deploy to cloud"
	@echo "  make deploy-aws    - Deploy to AWS"
	@echo "  make deploy-gcp    - Deploy to GCP"
	@echo "  make deploy-azure  - Deploy to Azure"