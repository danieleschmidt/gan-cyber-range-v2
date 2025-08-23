# Defensive Cybersecurity System Container
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r defenseuser && useradd -r -g defenseuser defenseuser

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs configs data reports && \
    chown -R defenseuser:defenseuser /app

# Set secure permissions
RUN chmod -R 755 /app && \
    chmod -R 700 /app/configs && \
    chmod -R 700 /app/logs

# Switch to non-root user
USER defenseuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 health_check.py || exit 1

# Default command
CMD ["python3", "defensive_demo.py"]

# Labels for metadata
LABEL maintainer="Defensive Security Team"
LABEL version="2.0.0"
LABEL description="Defensive Cybersecurity Training Platform"
LABEL security.scan="enabled"
