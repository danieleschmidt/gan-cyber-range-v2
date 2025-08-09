# GAN-Cyber-Range-v2 Dockerfile
# Multi-stage build for production deployment

# Stage 1: Base Python image with system dependencies
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    gcc \
    g++ \
    make \
    cmake \
    git \
    curl \
    wget \
    # Network tools for cyber range functionality
    net-tools \
    iproute2 \
    iptables \
    tcpdump \
    nmap \
    # Database clients
    postgresql-client \
    # Security tools
    openssl \
    ca-certificates \
    # Python development headers
    python3-dev \
    libpq-dev \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Stage 2: Dependencies installation
FROM base as dependencies

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install gunicorn uvicorn[standard]

# Install additional ML/AI dependencies
RUN pip install \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    transformers>=4.35.0 \
    datasets>=2.14.0 \
    accelerate>=0.24.0 \
    wandb>=0.16.0

# Stage 3: Application build
FROM dependencies as builder

# Copy application source code
COPY gan_cyber_range/ ./gan_cyber_range/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY conftest.py .
COPY pytest.ini .

# Install the application in development mode
RUN pip install -e .

# Run tests
RUN python -m pytest tests/ -v --tb=short

# Stage 4: Production image
FROM base as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application
COPY --from=builder /app/gan_cyber_range ./gan_cyber_range/
COPY --from=builder /app/setup.py ./
COPY --from=builder /app/README.md ./

# Create necessary directories
RUN mkdir -p \
    /app/data \
    /app/models \
    /app/logs \
    /app/backups \
    /app/tmp \
    && chown -R appuser:appuser /app

# Install the application
RUN pip install -e . --no-deps

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown appuser:appuser /entrypoint.sh

# Copy health check script
COPY docker/healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /healthcheck.sh

# Set default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["api"]

# Labels
LABEL maintainer="Terragon Labs <info@terragon.com>" \
      version="2.0.0" \
      description="GAN-Cyber-Range-v2 AI-driven cybersecurity training platform" \
      org.opencontainers.image.title="GAN-Cyber-Range-v2" \
      org.opencontainers.image.description="Second-Generation Adversarial Cyber Range with GAN-based Attack Generation" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.url="https://github.com/terragonlabs/gan-cyber-range-v2" \
      org.opencontainers.image.source="https://github.com/terragonlabs/gan-cyber-range-v2" \
      org.opencontainers.image.documentation="https://gan-cyber-range-v2.readthedocs.io"

# Stage 5: Development image
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    black>=23.9.0 \
    flake8>=6.1.0 \
    mypy>=1.6.0 \
    pre-commit>=3.4.0 \
    jupyter>=1.0.0 \
    ipython>=8.0.0

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    htop \
    tree \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER appuser

# Override default command for development
CMD ["api", "--reload"]