#!/bin/bash

# GAN-Cyber-Range-v2 Deployment Script
# Automated deployment for different environments
# Usage: ./scripts/deploy.sh [environment] [options]

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="gan-cyber-range-v2"
DOCKER_IMAGE="terragon/gan-cyber-range-v2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Show usage information
show_usage() {
    cat << EOF
GAN-Cyber-Range-v2 Deployment Script

Usage: $0 [environment] [options]

Environments:
  local         Deploy locally using Docker Compose
  kubernetes    Deploy to Kubernetes cluster
  cloud         Deploy to cloud provider (AWS/GCP/Azure)
  development   Deploy development environment
  staging       Deploy staging environment
  production    Deploy production environment

Options:
  --build       Build Docker images before deployment
  --pull        Pull latest images before deployment
  --update      Update existing deployment
  --destroy     Destroy existing deployment
  --dry-run     Show what would be deployed without executing
  --verbose     Enable verbose output
  --help        Show this help message

Examples:
  $0 local --build
  $0 kubernetes --update --verbose
  $0 production --pull --dry-run

Environment Variables:
  ENVIRONMENT         Target environment (overrides argument)
  DOCKER_REGISTRY     Docker registry URL
  KUBE_CONTEXT        Kubernetes context to use
  AWS_PROFILE         AWS profile for cloud deployment
  DEBUG               Enable debug logging (true/false)

EOF
}

# Parse command line arguments
parse_arguments() {
    ENVIRONMENT="${1:-local}"
    shift || true

    # Default values
    BUILD_IMAGES=false
    PULL_IMAGES=false
    UPDATE_DEPLOYMENT=false
    DESTROY_DEPLOYMENT=false
    DRY_RUN=false
    VERBOSE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                BUILD_IMAGES=true
                shift
                ;;
            --pull)
                PULL_IMAGES=true
                shift
                ;;
            --update)
                UPDATE_DEPLOYMENT=true
                shift
                ;;
            --destroy)
                DESTROY_DEPLOYMENT=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                DEBUG=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Override environment from environment variable if set
    ENVIRONMENT="${ENVIRONMENT:-${1:-local}}"

    log_debug "Parsed arguments: ENVIRONMENT=$ENVIRONMENT, BUILD=$BUILD_IMAGES, PULL=$PULL_IMAGES, UPDATE=$UPDATE_DEPLOYMENT, DESTROY=$DESTROY_DEPLOYMENT"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]] && [[ "$ENVIRONMENT" != "local" ]]; then
        log_warn "Running as root is not recommended for production deployments"
    fi

    # Check required commands
    local missing_commands=()

    if [[ "$ENVIRONMENT" == "local" ]] || [[ "$BUILD_IMAGES" == "true" ]]; then
        if ! command -v docker &> /dev/null; then
            missing_commands+=("docker")
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            missing_commands+=("docker-compose")
        fi
    fi

    if [[ "$ENVIRONMENT" == "kubernetes" ]] || [[ "$ENVIRONMENT" == "production" ]]; then
        if ! command -v kubectl &> /dev/null; then
            missing_commands+=("kubectl")
        fi
        
        if ! command -v helm &> /dev/null; then
            log_warn "Helm not found - some features may not be available"
        fi
    fi

    if [[ "$ENVIRONMENT" == "cloud" ]]; then
        if ! command -v aws &> /dev/null && ! command -v gcloud &> /dev/null && ! command -v az &> /dev/null; then
            log_warn "No cloud CLI tools found - manual configuration may be required"
        fi
    fi

    if [[ ${#missing_commands[@]} -gt 0 ]]; then
        log_error "Missing required commands: ${missing_commands[*]}"
        exit 1
    fi

    # Check Docker daemon
    if [[ "$ENVIRONMENT" == "local" ]] || [[ "$BUILD_IMAGES" == "true" ]]; then
        if ! docker info &> /dev/null; then
            log_error "Docker daemon is not running"
            exit 1
        fi
    fi

    # Check Kubernetes context
    if [[ "$ENVIRONMENT" == "kubernetes" ]] || [[ "$ENVIRONMENT" == "production" ]]; then
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Cannot connect to Kubernetes cluster"
            log_info "Available contexts:"
            kubectl config get-contexts
            exit 1
        fi
        
        local current_context
        current_context=$(kubectl config current-context)
        log_info "Using Kubernetes context: $current_context"
    fi

    log_info "Prerequisites check passed"
}

# Set up environment-specific configuration
setup_environment() {
    log_info "Setting up environment: $ENVIRONMENT"

    case $ENVIRONMENT in
        local|development)
            export COMPOSE_PROJECT_NAME="gan-cyber-range-dev"
            export API_PORT="8000"
            export WEB_PORT="3000"
            export DB_PORT="5432"
            export REDIS_PORT="6379"
            ;;
        staging)
            export COMPOSE_PROJECT_NAME="gan-cyber-range-staging"
            export API_PORT="8001"
            export WEB_PORT="3001"
            export DB_PORT="5433"
            export REDIS_PORT="6380"
            ;;
        production|kubernetes)
            export COMPOSE_PROJECT_NAME="gan-cyber-range-prod"
            export NAMESPACE="gan-cyber-range"
            ;;
        cloud)
            export COMPOSE_PROJECT_NAME="gan-cyber-range-cloud"
            # Cloud-specific configuration will be handled by cloud provider scripts
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac

    # Load environment-specific variables
    local env_file="${PROJECT_ROOT}/.env.${ENVIRONMENT}"
    if [[ -f "$env_file" ]]; then
        log_info "Loading environment file: $env_file"
        set -a  # Automatically export all variables
        # shellcheck source=/dev/null
        source "$env_file"
        set +a
    else
        log_warn "Environment file not found: $env_file"
    fi
}

# Build Docker images
build_images() {
    if [[ "$BUILD_IMAGES" != "true" ]]; then
        return 0
    fi

    log_info "Building Docker images..."

    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build: $DOCKER_IMAGE:latest"
        return 0
    fi

    # Build main application image
    log_info "Building main application image..."
    docker build \
        -f docker/Dockerfile \
        -t "$DOCKER_IMAGE:latest" \
        -t "$DOCKER_IMAGE:$(git rev-parse --short HEAD 2>/dev/null || echo 'dev')" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
        .

    # Build additional images if needed
    if [[ -d "$PROJECT_ROOT/docker/nginx" ]]; then
        log_info "Building Nginx image..."
        docker build \
            -f docker/nginx/Dockerfile \
            -t "$DOCKER_IMAGE-nginx:latest" \
            docker/nginx/
    fi

    log_info "Docker images built successfully"
}

# Pull Docker images
pull_images() {
    if [[ "$PULL_IMAGES" != "true" ]]; then
        return 0
    fi

    log_info "Pulling Docker images..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would pull latest images"
        return 0
    fi

    # Pull images specified in docker-compose.yml
    cd "$PROJECT_ROOT"
    docker-compose pull --ignore-pull-failures || log_warn "Some images failed to pull"

    log_info "Docker images pulled successfully"
}

# Deploy to local environment
deploy_local() {
    log_info "Deploying to local environment..."

    cd "$PROJECT_ROOT"

    if [[ "$DESTROY_DEPLOYMENT" == "true" ]]; then
        log_info "Destroying existing local deployment..."
        if [[ "$DRY_RUN" != "true" ]]; then
            docker-compose down -v --remove-orphans
        fi
        return 0
    fi

    # Create necessary directories
    mkdir -p data/{postgresql,redis,api,models,logs}

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: docker-compose up -d"
        docker-compose config --quiet && log_info "Docker Compose configuration is valid"
        return 0
    fi

    # Deploy services
    if [[ "$UPDATE_DEPLOYMENT" == "true" ]]; then
        log_info "Updating existing deployment..."
        docker-compose up -d --force-recreate
    else
        log_info "Starting services..."
        docker-compose up -d
    fi

    # Wait for services to be healthy
    log_info "Waiting for services to be ready..."
    sleep 10

    # Check service health
    if docker-compose ps | grep -q "unhealthy"; then
        log_warn "Some services appear to be unhealthy"
        docker-compose ps
    fi

    # Show access information
    log_info "Deployment completed!"
    log_info "Services available at:"
    log_info "  API: http://localhost:${API_PORT:-8000}"
    log_info "  Web UI: http://localhost:${WEB_PORT:-3000}"
    log_info "  Grafana: http://localhost:3000 (admin/GrafanaAdmin123!)"
    
    log_info "To view logs: docker-compose logs -f"
    log_info "To stop: docker-compose down"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."

    cd "$PROJECT_ROOT"

    if [[ "$DESTROY_DEPLOYMENT" == "true" ]]; then
        log_info "Destroying existing Kubernetes deployment..."
        if [[ "$DRY_RUN" != "true" ]]; then
            kubectl delete -k k8s/ --ignore-not-found=true
        fi
        return 0
    fi

    # Apply Kubernetes manifests
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would apply Kubernetes manifests:"
        kubectl apply -k k8s/ --dry-run=client --output=yaml | grep -E "^(kind|metadata)" | head -20
        return 0
    fi

    # Create namespace if it doesn't exist
    kubectl create namespace "${NAMESPACE:-gan-cyber-range}" --dry-run=client -o yaml | kubectl apply -f -

    # Apply configurations
    log_info "Applying Kubernetes manifests..."
    kubectl apply -k k8s/

    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/gan-cyber-range-api -n "${NAMESPACE:-gan-cyber-range}" || log_warn "Deployment may not be fully ready"

    # Show deployment status
    kubectl get pods,services -n "${NAMESPACE:-gan-cyber-range}"

    # Get access information
    local api_url
    api_url=$(kubectl get service nginx-service -n "${NAMESPACE:-gan-cyber-range}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    log_info "Deployment completed!"
    log_info "API URL: http://${api_url} (may take a few minutes to be available)"
    log_info "To view pods: kubectl get pods -n ${NAMESPACE:-gan-cyber-range}"
    log_info "To view logs: kubectl logs -l app.kubernetes.io/name=gan-cyber-range-api -n ${NAMESPACE:-gan-cyber-range}"
}

# Deploy to cloud environment
deploy_cloud() {
    log_info "Deploying to cloud environment..."

    # This would be expanded based on specific cloud provider
    case ${CLOUD_PROVIDER:-aws} in
        aws)
            log_info "Deploying to AWS..."
            # Would call AWS-specific deployment script
            "${SCRIPT_DIR}/deploy-aws.sh" "$@"
            ;;
        gcp)
            log_info "Deploying to Google Cloud..."
            # Would call GCP-specific deployment script
            "${SCRIPT_DIR}/deploy-gcp.sh" "$@"
            ;;
        azure)
            log_info "Deploying to Azure..."
            # Would call Azure-specific deployment script
            "${SCRIPT_DIR}/deploy-azure.sh" "$@"
            ;;
        *)
            log_error "Unknown cloud provider: ${CLOUD_PROVIDER:-aws}"
            exit 1
            ;;
    esac
}

# Run post-deployment tasks
post_deployment_tasks() {
    log_info "Running post-deployment tasks..."

    case $ENVIRONMENT in
        local)
            # Run local tests
            if command -v curl &> /dev/null; then
                log_info "Testing API health endpoint..."
                sleep 5  # Give services time to start
                if curl -f -s "http://localhost:${API_PORT:-8000}/health" > /dev/null; then
                    log_info "API health check passed"
                else
                    log_warn "API health check failed"
                fi
            fi
            ;;
        kubernetes|production)
            # Run Kubernetes readiness checks
            log_info "Checking Kubernetes deployment status..."
            kubectl get deployments -n "${NAMESPACE:-gan-cyber-range}"
            ;;
        cloud)
            # Run cloud-specific validation
            log_info "Running cloud deployment validation..."
            ;;
    esac

    log_info "Post-deployment tasks completed"
}

# Cleanup on exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
    fi
    exit $exit_code
}

# Main deployment function
main() {
    # Set up signal handling
    trap cleanup EXIT
    trap 'log_error "Script interrupted"; exit 130' INT TERM

    # Parse arguments
    parse_arguments "$@"

    # Show banner
    cat << "EOF"
   ____    _    _   _        ____      _               ____                        
  / ___|  / \  | \ | |      / ___|   _| |__   ___ _ __|  _ \ __ _ _ __   __ _  ___ 
 | |  _  / _ \ |  \| |_____| |  | | | | '_ \ / _ \ '__|  |_) / _` | '_ \ / _` |/ _ \
 | |_| |/ ___ \| |\  |_____| |__| |_| | |_) |  __/ |  |  _ < (_| | | | | (_| |  __/
  \____/_/   \_\_| \_|      \____\__, |_.__/ \___|_|  |_| \_\__,_|_| |_|\__, |\___|
                                 |___/                                   |___/     
EOF
    
    log_info "Starting deployment for environment: $ENVIRONMENT"
    log_info "Timestamp: $(date)"
    log_info "Project: $PROJECT_NAME"

    # Run deployment steps
    check_prerequisites
    setup_environment
    build_images
    pull_images

    # Deploy based on environment
    case $ENVIRONMENT in
        local|development|staging)
            deploy_local
            ;;
        kubernetes|production)
            deploy_kubernetes
            ;;
        cloud)
            deploy_cloud
            ;;
        *)
            log_error "Unknown deployment environment: $ENVIRONMENT"
            exit 1
            ;;
    esac

    post_deployment_tasks

    log_info "Deployment completed successfully!"
    log_info "Environment: $ENVIRONMENT"
    log_info "Timestamp: $(date)"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi