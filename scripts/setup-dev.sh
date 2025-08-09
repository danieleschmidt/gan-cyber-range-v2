#!/bin/bash

# GAN-Cyber-Range-v2 Development Environment Setup
# Sets up complete development environment with all dependencies

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Show banner
cat << "EOF"
  ____             ____      _               ____                        
 |  _ \  _____   __/ ___|   _| |__   ___ _ __|  _ \ __ _ _ __   __ _  ___ 
 | | | |/ _ \ \ / / |  | | | | '_ \ / _ \ '__|  |_) / _` | '_ \ / _` |/ _ \
 | |_| |  __/\ V /| |__| |_| | |_) |  __/ |  |  _ < (_| | | | | (_| |  __/
 |____/ \___| \_/  \____\__, |_.__/ \___|_|  |_| \_\__,_|_| |_|\__, |\___|
                        |___/                                   |___/     
                     Development Environment Setup
EOF

log_info "Setting up development environment..."

# Check if running on supported OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    log_info "Detected Linux OS"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    log_info "Detected macOS"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    log_info "Detected Windows"
else
    log_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    case $OS in
        linux)
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y \
                    python3 python3-pip python3-venv python3-dev \
                    git curl wget \
                    build-essential \
                    libpq-dev \
                    redis-tools \
                    postgresql-client \
                    docker.io docker-compose \
                    kubectl
            elif command -v yum &> /dev/null; then
                sudo yum update -y
                sudo yum install -y \
                    python3 python3-pip \
                    git curl wget \
                    gcc gcc-c++ make \
                    postgresql-devel \
                    redis \
                    postgresql \
                    docker docker-compose
            else
                log_warn "Unsupported Linux distribution. Please install dependencies manually."
            fi
            ;;
        macos)
            if ! command -v brew &> /dev/null; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            brew install python@3.11 git curl wget
            brew install postgresql redis
            brew install --cask docker
            brew install kubectl
            ;;
        windows)
            log_warn "Windows detected. Please install the following manually:"
            log_info "  - Python 3.9+ from python.org"
            log_info "  - Git from git-scm.com"
            log_info "  - Docker Desktop from docker.com"
            log_info "  - PostgreSQL from postgresql.org"
            log_info "  - Redis (or use Docker)"
            ;;
    esac
}

# Set up Python virtual environment
setup_python_env() {
    log_info "Setting up Python virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        log_info "Created virtual environment"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install project dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Install development dependencies
    pip install \
        pytest pytest-cov pytest-asyncio \
        black flake8 mypy \
        pre-commit \
        jupyter ipython \
        httpx  # For async testing
    
    # Install project in development mode
    pip install -e .
    
    log_info "Python environment setup complete"
}

# Set up pre-commit hooks
setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Create pre-commit configuration if it doesn't exist
    if [[ ! -f ".pre-commit-config.yaml" ]]; then
        cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]
EOF
    fi
    
    # Install pre-commit hooks
    pre-commit install
    
    log_info "Pre-commit hooks installed"
}

# Set up development database
setup_dev_database() {
    log_info "Setting up development database..."
    
    # Check if PostgreSQL is running
    if command -v psql &> /dev/null; then
        if pg_isready -q; then
            log_info "PostgreSQL is running"
            
            # Create development database
            createdb gan_cyber_range_dev 2>/dev/null || log_warn "Database may already exist"
            
            # Create test database
            createdb gan_cyber_range_test 2>/dev/null || log_warn "Test database may already exist"
            
        else
            log_warn "PostgreSQL is not running. Please start PostgreSQL service."
        fi
    else
        log_warn "PostgreSQL not found. Using Docker for database..."
        
        # Start PostgreSQL in Docker
        docker run -d \
            --name gan-cyber-range-postgres \
            -e POSTGRES_DB=gan_cyber_range_dev \
            -e POSTGRES_USER=cyber_range_user \
            -e POSTGRES_PASSWORD=dev_password \
            -p 5432:5432 \
            postgres:15-alpine
    fi
}

# Set up development Redis
setup_dev_redis() {
    log_info "Setting up development Redis..."
    
    if command -v redis-server &> /dev/null; then
        if pgrep redis-server > /dev/null; then
            log_info "Redis is running"
        else
            log_warn "Redis is not running. Please start Redis service."
        fi
    else
        log_warn "Redis not found. Using Docker for Redis..."
        
        # Start Redis in Docker
        docker run -d \
            --name gan-cyber-range-redis \
            -p 6379:6379 \
            redis:7-alpine
    fi
}

# Create development configuration
setup_dev_config() {
    log_info "Setting up development configuration..."
    
    cd "$PROJECT_ROOT"
    
    # Create .env.development file
    if [[ ! -f ".env.development" ]]; then
        cat > .env.development << 'EOF'
# Development Environment Configuration

# Database
DATABASE_URL=postgresql://cyber_range_user:dev_password@localhost:5432/gan_cyber_range_dev
DATABASE_TEST_URL=postgresql://cyber_range_user:dev_password@localhost:5432/gan_cyber_range_test

# Redis
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
DEBUG=true
LOG_LEVEL=DEBUG

# JWT Configuration
JWT_SECRET=development-secret-key-not-for-production
JWT_EXPIRATION_HOURS=24

# Security (development only - change in production)
ADMIN_USERNAME=admin
ADMIN_PASSWORD=AdminDevPass123!

# External APIs (set your own keys)
OPENAI_API_KEY=your-openai-api-key
HUGGINGFACE_TOKEN=your-huggingface-token

# Development Features
ENABLE_DEBUG_ROUTES=true
MOCK_EXTERNAL_SERVICES=true
EOF
        log_info "Created .env.development file"
    fi
    
    # Create VS Code configuration
    mkdir -p .vscode
    
    if [[ ! -f ".vscode/settings.json" ]]; then
        cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
EOF
    fi
    
    if [[ ! -f ".vscode/launch.json" ]]; then
        cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/venv/bin/uvicorn",
            "args": [
                "gan_cyber_range.api.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ],
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env.development",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "envFile": "${workspaceFolder}/.env.development"
        }
    ]
}
EOF
    fi
}

# Create helpful development scripts
create_dev_scripts() {
    log_info "Creating development scripts..."
    
    mkdir -p "$PROJECT_ROOT/scripts/dev"
    
    # Start development server
    cat > "$PROJECT_ROOT/scripts/dev/start-api.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/../.."
source venv/bin/activate
export $(cat .env.development | grep -v '^#' | xargs)
uvicorn gan_cyber_range.api.main:app --host 0.0.0.0 --port 8000 --reload
EOF
    
    # Run tests
    cat > "$PROJECT_ROOT/scripts/dev/run-tests.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/../.."
source venv/bin/activate
export $(cat .env.development | grep -v '^#' | xargs)
pytest tests/ -v --cov=gan_cyber_range --cov-report=html
EOF
    
    # Format code
    cat > "$PROJECT_ROOT/scripts/dev/format-code.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/../.."
source venv/bin/activate
black gan_cyber_range/ tests/ scripts/
flake8 gan_cyber_range/ tests/
mypy gan_cyber_range/ --ignore-missing-imports
EOF
    
    # Generate documentation
    cat > "$PROJECT_ROOT/scripts/dev/generate-docs.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/../.."
source venv/bin/activate
export $(cat .env.development | grep -v '^#' | xargs)
python -c "
from gan_cyber_range.api.main import app
import json
from fastapi.openapi.utils import get_openapi

openapi_schema = get_openapi(
    title='GAN-Cyber-Range-v2 API',
    version='2.0.0',
    description='AI-driven cybersecurity training platform',
    routes=app.routes,
)

with open('docs/api_schema.json', 'w') as f:
    json.dump(openapi_schema, f, indent=2)

print('API documentation generated: docs/api_schema.json')
"
EOF
    
    # Make scripts executable
    chmod +x "$PROJECT_ROOT/scripts/dev"/*.sh
    
    log_info "Development scripts created in scripts/dev/"
}

# Show completion message
show_completion() {
    log_info "Development environment setup complete!"
    echo
    log_info "Next steps:"
    log_info "  1. Activate virtual environment: source venv/bin/activate"
    log_info "  2. Start development server: ./scripts/dev/start-api.sh"
    log_info "  3. Run tests: ./scripts/dev/run-tests.sh"
    log_info "  4. Format code: ./scripts/dev/format-code.sh"
    echo
    log_info "Development URLs:"
    log_info "  API: http://localhost:8000"
    log_info "  API Docs: http://localhost:8000/docs"
    log_info "  Health: http://localhost:8000/health"
    echo
    log_info "Useful commands:"
    log_info "  docker-compose up -d    # Start all services"
    log_info "  docker-compose logs -f  # View logs"
    log_info "  docker-compose down     # Stop services"
}

# Main setup function
main() {
    log_info "Starting development environment setup..."
    
    install_system_deps
    setup_python_env
    setup_pre_commit
    setup_dev_database
    setup_dev_redis
    setup_dev_config
    create_dev_scripts
    
    show_completion
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi