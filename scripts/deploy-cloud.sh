#!/bin/bash

# GAN-Cyber-Range-v2 Cloud Deployment Script
# Supports AWS, Google Cloud Platform, and Azure

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="gan-cyber-range-v2"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default values
CLOUD_PROVIDER="${CLOUD_PROVIDER:-aws}"
REGION="${REGION:-us-east-1}"
CLUSTER_NAME="${CLUSTER_NAME:-gan-cyber-range}"
NODE_COUNT="${NODE_COUNT:-3}"
INSTANCE_TYPE="${INSTANCE_TYPE:-}"
DRY_RUN=false
DESTROY=false

# Show usage
show_usage() {
    cat << EOF
GAN-Cyber-Range-v2 Cloud Deployment Script

Usage: $0 [options]

Options:
  --provider PROVIDER   Cloud provider (aws|gcp|azure) [default: aws]
  --region REGION       Cloud region [default: us-east-1]
  --cluster CLUSTER     Cluster name [default: gan-cyber-range]
  --nodes COUNT         Number of worker nodes [default: 3]
  --instance-type TYPE  Instance type for nodes
  --dry-run            Show what would be deployed
  --destroy            Destroy existing deployment
  --help               Show this help

Environment Variables:
  CLOUD_PROVIDER       Default cloud provider
  AWS_PROFILE          AWS profile to use
  GOOGLE_CLOUD_PROJECT GCP project ID
  AZURE_SUBSCRIPTION   Azure subscription ID

Examples:
  $0 --provider aws --region us-west-2
  $0 --provider gcp --region us-central1-a --nodes 5
  $0 --provider azure --region eastus --destroy

EOF
}

# Parse arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --provider)
                CLOUD_PROVIDER="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            --cluster)
                CLUSTER_NAME="$2"
                shift 2
                ;;
            --nodes)
                NODE_COUNT="$2"
                shift 2
                ;;
            --instance-type)
                INSTANCE_TYPE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --destroy)
                DESTROY=true
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
}

# Check prerequisites for cloud provider
check_prerequisites() {
    log_info "Checking prerequisites for $CLOUD_PROVIDER..."
    
    case $CLOUD_PROVIDER in
        aws)
            if ! command -v aws &> /dev/null; then
                log_error "AWS CLI not found. Please install awscli"
                exit 1
            fi
            
            if ! aws sts get-caller-identity &> /dev/null; then
                log_error "AWS credentials not configured"
                exit 1
            fi
            
            if ! command -v eksctl &> /dev/null; then
                log_warn "eksctl not found. Installing..."
                install_eksctl
            fi
            
            # Set default instance type
            INSTANCE_TYPE="${INSTANCE_TYPE:-t3.large}"
            ;;
            
        gcp)
            if ! command -v gcloud &> /dev/null; then
                log_error "Google Cloud SDK not found"
                exit 1
            fi
            
            if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
                log_error "Google Cloud authentication not configured"
                exit 1
            fi
            
            # Set default instance type
            INSTANCE_TYPE="${INSTANCE_TYPE:-e2-standard-4}"
            ;;
            
        azure)
            if ! command -v az &> /dev/null; then
                log_error "Azure CLI not found"
                exit 1
            fi
            
            if ! az account show &> /dev/null; then
                log_error "Azure authentication not configured"
                exit 1
            fi
            
            # Set default instance type
            INSTANCE_TYPE="${INSTANCE_TYPE:-Standard_D4s_v3}"
            ;;
            
        *)
            log_error "Unsupported cloud provider: $CLOUD_PROVIDER"
            exit 1
            ;;
    esac
    
    # Check for kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found"
        exit 1
    fi
    
    # Check for helm (optional)
    if ! command -v helm &> /dev/null; then
        log_warn "Helm not found - some features may not be available"
    fi
}

# Install eksctl for AWS
install_eksctl() {
    log_info "Installing eksctl..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
        sudo mv /tmp/eksctl /usr/local/bin
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew tap weaveworks/tap
            brew install weaveworks/tap/eksctl
        else
            log_error "Please install eksctl manually on macOS"
            exit 1
        fi
    fi
}

# Deploy to AWS EKS
deploy_aws() {
    log_info "Deploying to AWS EKS..."
    
    if [[ "$DESTROY" == "true" ]]; then
        log_info "Destroying AWS EKS cluster..."
        if [[ "$DRY_RUN" != "true" ]]; then
            eksctl delete cluster --name="$CLUSTER_NAME" --region="$REGION" --wait
        fi
        return 0
    fi
    
    # Create EKS cluster
    log_info "Creating EKS cluster: $CLUSTER_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create EKS cluster with:"
        log_info "  Name: $CLUSTER_NAME"
        log_info "  Region: $REGION" 
        log_info "  Node count: $NODE_COUNT"
        log_info "  Instance type: $INSTANCE_TYPE"
        return 0
    fi
    
    # Check if cluster already exists
    if eksctl get cluster --name="$CLUSTER_NAME" --region="$REGION" &> /dev/null; then
        log_info "EKS cluster already exists"
    else
        # Create cluster configuration
        cat > /tmp/eks-cluster.yaml << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $REGION
  version: "1.28"

nodeGroups:
  - name: worker-nodes
    instanceType: $INSTANCE_TYPE
    desiredCapacity: $NODE_COUNT
    minSize: 1
    maxSize: 10
    volumeSize: 100
    ssh:
      allow: false
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        ebs: true
        efs: true
        albIngress: true

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver

iam:
  withOIDC: true

cloudWatch:
  clusterLogging:
    enable: ["audit", "authenticator", "controllerManager"]

EOF
        
        # Create cluster
        eksctl create cluster -f /tmp/eks-cluster.yaml
    fi
    
    # Update kubeconfig
    aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER_NAME"
    
    # Install AWS Load Balancer Controller
    install_aws_load_balancer_controller
    
    # Deploy application
    deploy_to_kubernetes
    
    log_info "AWS deployment completed successfully!"
}

# Install AWS Load Balancer Controller
install_aws_load_balancer_controller() {
    log_info "Installing AWS Load Balancer Controller..."
    
    # Create IAM OIDC identity provider
    eksctl utils associate-iam-oidc-provider --region="$REGION" --cluster="$CLUSTER_NAME" --approve
    
    # Create service account
    eksctl create iamserviceaccount \
        --cluster="$CLUSTER_NAME" \
        --namespace=kube-system \
        --name=aws-load-balancer-controller \
        --role-name="AmazonEKSLoadBalancerControllerRole" \
        --attach-policy-arn=arn:aws:iam::aws:policy/ElasticLoadBalancingFullAccess \
        --approve
    
    # Install controller
    helm repo add eks https://aws.github.io/eks-charts
    helm repo update
    helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName="$CLUSTER_NAME" \
        --set serviceAccount.create=false \
        --set serviceAccount.name=aws-load-balancer-controller
}

# Deploy to Google Cloud GKE
deploy_gcp() {
    log_info "Deploying to Google Cloud GKE..."
    
    if [[ "$DESTROY" == "true" ]]; then
        log_info "Destroying GKE cluster..."
        if [[ "$DRY_RUN" != "true" ]]; then
            gcloud container clusters delete "$CLUSTER_NAME" --zone="$REGION" --quiet
        fi
        return 0
    fi
    
    # Set project
    if [[ -n "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
        gcloud config set project "$GOOGLE_CLOUD_PROJECT"
    fi
    
    log_info "Creating GKE cluster: $CLUSTER_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create GKE cluster with:"
        log_info "  Name: $CLUSTER_NAME"
        log_info "  Zone: $REGION"
        log_info "  Node count: $NODE_COUNT"
        log_info "  Machine type: $INSTANCE_TYPE"
        return 0
    fi
    
    # Check if cluster exists
    if gcloud container clusters describe "$CLUSTER_NAME" --zone="$REGION" &> /dev/null; then
        log_info "GKE cluster already exists"
    else
        # Create cluster
        gcloud container clusters create "$CLUSTER_NAME" \
            --zone="$REGION" \
            --num-nodes="$NODE_COUNT" \
            --machine-type="$INSTANCE_TYPE" \
            --disk-size=100GB \
            --enable-autoscaling \
            --min-nodes=1 \
            --max-nodes=10 \
            --enable-autorepair \
            --enable-autoupgrade \
            --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
            --workload-pool="$(gcloud config get-value project).svc.id.goog"
    fi
    
    # Get credentials
    gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$REGION"
    
    # Deploy application
    deploy_to_kubernetes
    
    log_info "GCP deployment completed successfully!"
}

# Deploy to Azure AKS
deploy_azure() {
    log_info "Deploying to Azure AKS..."
    
    local resource_group="${CLUSTER_NAME}-rg"
    
    if [[ "$DESTROY" == "true" ]]; then
        log_info "Destroying AKS cluster..."
        if [[ "$DRY_RUN" != "true" ]]; then
            az group delete --name "$resource_group" --yes --no-wait
        fi
        return 0
    fi
    
    log_info "Creating AKS cluster: $CLUSTER_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create AKS cluster with:"
        log_info "  Name: $CLUSTER_NAME"
        log_info "  Resource Group: $resource_group"
        log_info "  Location: $REGION"
        log_info "  Node count: $NODE_COUNT"
        log_info "  VM size: $INSTANCE_TYPE"
        return 0
    fi
    
    # Create resource group
    az group create --name "$resource_group" --location "$REGION"
    
    # Check if cluster exists
    if az aks show --name "$CLUSTER_NAME" --resource-group "$resource_group" &> /dev/null; then
        log_info "AKS cluster already exists"
    else
        # Create cluster
        az aks create \
            --resource-group "$resource_group" \
            --name "$CLUSTER_NAME" \
            --node-count "$NODE_COUNT" \
            --node-vm-size "$INSTANCE_TYPE" \
            --node-osdisk-size 100 \
            --enable-cluster-autoscaler \
            --min-count 1 \
            --max-count 10 \
            --enable-addons monitoring,http_application_routing \
            --generate-ssh-keys \
            --enable-managed-identity
    fi
    
    # Get credentials
    az aks get-credentials --resource-group "$resource_group" --name "$CLUSTER_NAME"
    
    # Deploy application
    deploy_to_kubernetes
    
    log_info "Azure deployment completed successfully!"
}

# Deploy application to Kubernetes cluster
deploy_to_kubernetes() {
    log_info "Deploying application to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Create namespace
    kubectl create namespace gan-cyber-range --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    kubectl apply -k k8s/
    
    # Wait for deployment
    kubectl wait --for=condition=available --timeout=300s deployment/gan-cyber-range-api -n gan-cyber-range
    
    # Get service information
    local service_info
    case $CLOUD_PROVIDER in
        aws)
            service_info=$(kubectl get service nginx-service -n gan-cyber-range -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
            ;;
        gcp|azure)
            service_info=$(kubectl get service nginx-service -n gan-cyber-range -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
            ;;
    esac
    
    log_info "Application deployed successfully!"
    log_info "Service endpoint: http://$service_info"
    log_info "API Documentation: http://$service_info/docs"
}

# Show deployment information
show_deployment_info() {
    log_info "Deployment Information:"
    log_info "  Cloud Provider: $CLOUD_PROVIDER"
    log_info "  Region: $REGION"
    log_info "  Cluster: $CLUSTER_NAME"
    log_info "  Nodes: $NODE_COUNT"
    log_info "  Instance Type: $INSTANCE_TYPE"
    
    case $CLOUD_PROVIDER in
        aws)
            log_info "AWS-specific commands:"
            log_info "  View cluster: eksctl get cluster --name=$CLUSTER_NAME --region=$REGION"
            log_info "  Update kubeconfig: aws eks update-kubeconfig --region=$REGION --name=$CLUSTER_NAME"
            ;;
        gcp)
            log_info "GCP-specific commands:"
            log_info "  View cluster: gcloud container clusters describe $CLUSTER_NAME --zone=$REGION"
            log_info "  Get credentials: gcloud container clusters get-credentials $CLUSTER_NAME --zone=$REGION"
            ;;
        azure)
            log_info "Azure-specific commands:"
            log_info "  View cluster: az aks show --name=$CLUSTER_NAME --resource-group=${CLUSTER_NAME}-rg"
            log_info "  Get credentials: az aks get-credentials --name=$CLUSTER_NAME --resource-group=${CLUSTER_NAME}-rg"
            ;;
    esac
    
    log_info "Kubernetes commands:"
    log_info "  View pods: kubectl get pods -n gan-cyber-range"
    log_info "  View services: kubectl get services -n gan-cyber-range"
    log_info "  View logs: kubectl logs -l app.kubernetes.io/name=gan-cyber-range-api -n gan-cyber-range"
}

# Main function
main() {
    parse_arguments "$@"
    
    log_info "Starting cloud deployment..."
    log_info "Provider: $CLOUD_PROVIDER"
    log_info "Region: $REGION"
    
    check_prerequisites
    
    case $CLOUD_PROVIDER in
        aws)
            deploy_aws
            ;;
        gcp)
            deploy_gcp
            ;;
        azure)
            deploy_azure
            ;;
        *)
            log_error "Unsupported cloud provider: $CLOUD_PROVIDER"
            exit 1
            ;;
    esac
    
    show_deployment_info
    
    log_info "Cloud deployment completed!"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi