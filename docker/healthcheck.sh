#!/bin/bash
set -e

# GAN-Cyber-Range-v2 Health Check Script
# Comprehensive health monitoring for containerized deployment

# Configuration
HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-http://localhost:8000/health}"
TIMEOUT="${HEALTH_TIMEOUT:-10}"
MAX_RETRIES="${HEALTH_MAX_RETRIES:-3}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') [HEALTH] $1" >&2
}

log_error() {
    log "${RED}ERROR: $1${NC}"
}

log_warn() {
    log "${YELLOW}WARNING: $1${NC}"
}

log_info() {
    log "${GREEN}INFO: $1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local url="$1"
    local timeout="$2"
    local retry_count=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if command_exists curl; then
            if curl -f -s -m "$timeout" "$url" >/dev/null 2>&1; then
                return 0
            fi
        elif command_exists wget; then
            if wget -q -T "$timeout" -O /dev/null "$url" >/dev/null 2>&1; then
                return 0
            fi
        else
            log_error "Neither curl nor wget available for health check"
            return 1
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $MAX_RETRIES ]; then
            log_warn "Health check attempt $retry_count failed, retrying..."
            sleep 1
        fi
    done
    
    return 1
}

# Function to check detailed health status
check_detailed_health() {
    local url="$1"
    local timeout="$2"
    
    if command_exists curl; then
        local response
        response=$(curl -f -s -m "$timeout" "$url" 2>/dev/null)
        
        if [ $? -eq 0 ] && [ -n "$response" ]; then
            # Try to parse JSON response
            if command_exists python3; then
                python3 -c "
import json, sys
try:
    health = json.loads('$response')
    status = health.get('status', 'unknown')
    components = health.get('components', [])
    
    print(f'Overall Status: {status}')
    
    if components:
        print('Component Status:')
        for comp in components:
            comp_name = comp.get('component', 'unknown')
            comp_status = comp.get('status', 'unknown')
            print(f'  - {comp_name}: {comp_status}')
    
    # Exit with error if not healthy
    if status != 'healthy':
        sys.exit(1)
        
except Exception as e:
    print(f'Failed to parse health response: {e}')
    sys.exit(1)
                " 2>/dev/null
                return $?
            else
                # Just check if we got a response
                echo "Health endpoint responded (detailed parsing unavailable)"
                return 0
            fi
        fi
    fi
    
    return 1
}

# Function to check process health
check_process_health() {
    # Check if main Python process is running
    if pgrep -f "gan_cyber_range" >/dev/null 2>&1; then
        log_info "Main application process is running"
        return 0
    else
        log_error "Main application process not found"
        return 1
    fi
}

# Function to check memory usage
check_memory_usage() {
    if [ -f /proc/meminfo ]; then
        local total_mem
        local free_mem
        local used_percent
        
        total_mem=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        free_mem=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        
        if [ -n "$total_mem" ] && [ -n "$free_mem" ]; then
            used_percent=$(( (total_mem - free_mem) * 100 / total_mem ))
            
            if [ $used_percent -gt 90 ]; then
                log_warn "High memory usage: ${used_percent}%"
                return 1
            else
                log_info "Memory usage: ${used_percent}%"
                return 0
            fi
        fi
    fi
    
    return 0  # If we can't check, assume OK
}

# Function to check disk usage
check_disk_usage() {
    local usage
    usage=$(df /app 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ -n "$usage" ] && [ "$usage" -gt 85 ]; then
        log_warn "High disk usage: ${usage}%"
        return 1
    elif [ -n "$usage" ]; then
        log_info "Disk usage: ${usage}%"
    fi
    
    return 0
}

# Function to check database connectivity
check_database_connection() {
    if [ -n "$DATABASE_URL" ]; then
        # Try to connect to database using Python
        if command_exists python3; then
            python3 -c "
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, '/app')

async def check_db():
    try:
        from gan_cyber_range.db.database import get_database
        db = await get_database()
        health = await db.health_check()
        
        if health.get('database') == 'healthy':
            print('Database connection: healthy')
            return True
        else:
            print(f'Database connection: {health.get(\"database\", \"unknown\")}')
            return False
    except Exception as e:
        print(f'Database connection error: {e}')
        return False

try:
    result = asyncio.run(check_db())
    sys.exit(0 if result else 1)
except Exception as e:
    print(f'Database check failed: {e}')
    sys.exit(1)
            " 2>/dev/null
            return $?
        fi
    fi
    
    return 0  # If no DATABASE_URL or can't check, assume OK
}

# Function to check Redis connectivity  
check_redis_connection() {
    if [ -n "$REDIS_URL" ]; then
        # Try to connect to Redis using Python
        if command_exists python3; then
            python3 -c "
import asyncio
import sys
import os

# Add the app directory to Python path  
sys.path.insert(0, '/app')

async def check_redis():
    try:
        from gan_cyber_range.db.database import get_database
        db = await get_database()
        health = await db.health_check()
        
        if health.get('redis') == 'healthy':
            print('Redis connection: healthy')
            return True
        else:
            print(f'Redis connection: {health.get(\"redis\", \"unknown\")}')
            return False
    except Exception as e:
        print(f'Redis connection error: {e}')
        return False

try:
    result = asyncio.run(check_redis())
    sys.exit(0 if result else 1)
except Exception as e:
    print(f'Redis check failed: {e}')
    sys.exit(1)
            " 2>/dev/null
            return $?
        fi
    fi
    
    return 0  # If no REDIS_URL or can't check, assume OK
}

# Main health check function
main_health_check() {
    local exit_code=0
    local checks_failed=0
    local total_checks=0
    
    log_info "Starting comprehensive health check..."
    
    # HTTP endpoint check
    total_checks=$((total_checks + 1))
    if check_http_endpoint "$HEALTH_CHECK_URL" "$TIMEOUT"; then
        log_info "HTTP health endpoint: OK"
        
        # If basic check passes, try detailed check
        if check_detailed_health "$HEALTH_CHECK_URL" "$TIMEOUT"; then
            log_info "Detailed health check: OK"
        else
            log_warn "Detailed health check failed, but basic endpoint responsive"
        fi
    else
        log_error "HTTP health endpoint: FAILED"
        checks_failed=$((checks_failed + 1))
        exit_code=1
    fi
    
    # Process check
    total_checks=$((total_checks + 1))
    if check_process_health; then
        log_info "Process health: OK"
    else
        log_error "Process health: FAILED"
        checks_failed=$((checks_failed + 1))
        exit_code=1
    fi
    
    # Resource checks (warnings only)
    if ! check_memory_usage; then
        log_warn "Memory usage check: WARNING"
    fi
    
    if ! check_disk_usage; then
        log_warn "Disk usage check: WARNING"
    fi
    
    # Database connectivity (if configured)
    if [ -n "$DATABASE_URL" ]; then
        total_checks=$((total_checks + 1))
        if check_database_connection; then
            log_info "Database connectivity: OK"
        else
            log_error "Database connectivity: FAILED"
            checks_failed=$((checks_failed + 1))
            exit_code=1
        fi
    fi
    
    # Redis connectivity (if configured)
    if [ -n "$REDIS_URL" ]; then
        total_checks=$((total_checks + 1))
        if check_redis_connection; then
            log_info "Redis connectivity: OK"
        else
            log_error "Redis connectivity: FAILED"
            checks_failed=$((checks_failed + 1))
            exit_code=1
        fi
    fi
    
    # Summary
    local checks_passed=$((total_checks - checks_failed))
    log_info "Health check summary: $checks_passed/$total_checks checks passed"
    
    if [ $exit_code -eq 0 ]; then
        log_info "Overall health status: HEALTHY"
    else
        log_error "Overall health status: UNHEALTHY ($checks_failed failed checks)"
    fi
    
    return $exit_code
}

# Quick health check (for frequent monitoring)
quick_health_check() {
    if check_http_endpoint "$HEALTH_CHECK_URL" "$TIMEOUT"; then
        echo "OK"
        return 0
    else
        echo "FAILED"
        return 1
    fi
}

# Handle command line arguments
case "${1:-full}" in
    "quick"|"q")
        quick_health_check
        ;;
    "full"|"f"|"")
        main_health_check
        ;;
    "endpoint"|"http")
        if check_http_endpoint "$HEALTH_CHECK_URL" "$TIMEOUT"; then
            echo "HTTP endpoint: OK"
            exit 0
        else
            echo "HTTP endpoint: FAILED"
            exit 1
        fi
        ;;
    "process"|"proc")
        check_process_health
        ;;
    "memory"|"mem")
        check_memory_usage
        ;;
    "disk")
        check_disk_usage
        ;;
    "database"|"db")
        check_database_connection
        ;;
    "redis")
        check_redis_connection
        ;;
    "help"|"--help"|"-h")
        echo "GAN-Cyber-Range-v2 Health Check Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  full, f       - Run comprehensive health check (default)"
        echo "  quick, q      - Run quick HTTP endpoint check"
        echo "  endpoint      - Check HTTP endpoint only"
        echo "  process       - Check process health only"
        echo "  memory        - Check memory usage only"
        echo "  disk          - Check disk usage only"
        echo "  database      - Check database connectivity only"
        echo "  redis         - Check Redis connectivity only"
        echo "  help          - Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  HEALTH_CHECK_URL      - Health check endpoint (default: http://localhost:8000/health)"
        echo "  HEALTH_TIMEOUT        - HTTP timeout in seconds (default: 10)"
        echo "  HEALTH_MAX_RETRIES    - Maximum retry attempts (default: 3)"
        exit 0
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac