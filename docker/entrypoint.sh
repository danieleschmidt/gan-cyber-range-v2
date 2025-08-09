#!/bin/bash
set -e

# GAN-Cyber-Range-v2 Entrypoint Script
# Handles application startup with different modes

# Default values
export APP_MODE="${APP_MODE:-api}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
export API_WORKERS="${API_WORKERS:-4}"

# Function to wait for database
wait_for_database() {
    echo "Waiting for database connection..."
    until python -c "
import asyncio
import sys
from gan_cyber_range.db.database import get_database

async def check_db():
    try:
        db = await get_database()
        health = await db.health_check()
        if health['database'] == 'healthy':
            print('Database is ready!')
            return True
        else:
            return False
    except Exception as e:
        print(f'Database not ready: {e}')
        return False

result = asyncio.run(check_db())
sys.exit(0 if result else 1)
    "; do
        echo "Database is unavailable - sleeping"
        sleep 2
    done
}

# Function to run database migrations
run_migrations() {
    echo "Running database migrations..."
    python -c "
import asyncio
from gan_cyber_range.db.database import get_database, DatabaseMigration

async def migrate():
    db = await get_database()
    migration = DatabaseMigration(db)
    
    # Apply any pending migrations
    await migration.apply_migration(
        '001_initial',
        'Initial database schema',
        [
            # Migrations would be defined here
            'SELECT 1;'  # Placeholder
        ]
    )
    print('Migrations completed successfully')

asyncio.run(migrate())
    "
}

# Function to initialize default data
initialize_data() {
    echo "Initializing default data..."
    python -c "
import asyncio
from gan_cyber_range.api.auth import initialize_default_users

asyncio.run(initialize_default_users())
print('Default data initialized')
    "
}

# Function to start API server
start_api() {
    echo "Starting GAN-Cyber-Range-v2 API server..."
    
    if [ "${APP_MODE}" = "development" ] || [ "$1" = "--reload" ]; then
        echo "Starting in development mode with auto-reload..."
        exec uvicorn gan_cyber_range.api.main:app \
            --host "${API_HOST}" \
            --port "${API_PORT}" \
            --reload \
            --log-level "${LOG_LEVEL}" \
            --access-log \
            --reload-dir /app/gan_cyber_range
    else
        echo "Starting in production mode..."
        exec uvicorn gan_cyber_range.api.main:app \
            --host "${API_HOST}" \
            --port "${API_PORT}" \
            --workers "${API_WORKERS}" \
            --log-level "${LOG_LEVEL}" \
            --access-log \
            --loop uvloop \
            --http httptools
    fi
}

# Function to start worker processes
start_worker() {
    echo "Starting GAN-Cyber-Range-v2 background worker..."
    exec python -c "
import asyncio
from gan_cyber_range.utils.monitoring import MetricsCollector
from gan_cyber_range.core.attack_gan import AttackGAN

async def worker_main():
    print('Worker started')
    # Initialize components
    metrics_collector = MetricsCollector()
    attack_gan = AttackGAN()
    
    # Run worker tasks
    while True:
        try:
            # Collect metrics
            metrics = metrics_collector.collect_system_metrics()
            print(f'Collected metrics: {metrics}')
            
            await asyncio.sleep(30)  # Sleep for 30 seconds
        except Exception as e:
            print(f'Worker error: {e}')
            await asyncio.sleep(60)

asyncio.run(worker_main())
    "
}

# Function to run CLI commands
run_cli() {
    echo "Running CLI command: $*"
    exec python -m gan_cyber_range.cli.main "$@"
}

# Function to run tests
run_tests() {
    echo "Running tests..."
    export PYTHONPATH="/app:$PYTHONPATH"
    cd /app
    exec python -m pytest tests/ -v --tb=short --cov=gan_cyber_range --cov-report=term-missing
}

# Function to run interactive shell
run_shell() {
    echo "Starting interactive Python shell..."
    exec python -c "
import asyncio
from gan_cyber_range.core.attack_gan import AttackGAN
from gan_cyber_range.core.cyber_range import CyberRange
from gan_cyber_range.red_team.llm_adversary import RedTeamLLM
from gan_cyber_range.db.database import get_database

print('GAN-Cyber-Range-v2 Interactive Shell')
print('Available objects:')
print('  - AttackGAN')
print('  - CyberRange') 
print('  - RedTeamLLM')
print('  - get_database()')
print()

import IPython
IPython.start_ipython(argv=[])
    "
}

# Function to export data
export_data() {
    echo "Exporting data..."
    python -c "
import asyncio
import json
from gan_cyber_range.db.database import get_database
from gan_cyber_range.db.repositories import *

async def export_data():
    db = await get_database()
    async with db.get_session() as session:
        # Export users
        user_repo = UserRepository(session)
        users = await user_repo.get_all()
        
        # Export attack vectors
        attack_repo = AttackVectorRepository(session)
        attacks = await attack_repo.get_all()
        
        export_data = {
            'users': len(users),
            'attack_vectors': len(attacks),
            'export_timestamp': '$(date -Iseconds)'
        }
        
        print(json.dumps(export_data, indent=2))

asyncio.run(export_data())
    "
}

# Main execution logic
main() {
    echo "=== GAN-Cyber-Range-v2 Container Starting ==="
    echo "Mode: ${APP_MODE}"
    echo "Arguments: $*"
    echo "=========================================="
    
    case "${APP_MODE}" in
        "api"|"server")
            wait_for_database
            run_migrations
            initialize_data
            start_api "$@"
            ;;
        "worker")
            wait_for_database
            start_worker
            ;;
        "cli")
            run_cli "$@"
            ;;
        "test"|"tests")
            run_tests
            ;;
        "shell"|"python")
            wait_for_database
            run_shell
            ;;
        "migrate"|"migration")
            wait_for_database
            run_migrations
            ;;
        "export")
            wait_for_database
            export_data
            ;;
        "bash"|"sh")
            echo "Starting interactive bash shell..."
            exec bash
            ;;
        *)
            # Default: try to run as command
            echo "Running custom command: ${APP_MODE} $*"
            exec "${APP_MODE}" "$@"
            ;;
    esac
}

# Handle different command line arguments
if [ $# -gt 0 ]; then
    case "$1" in
        "api"|"server")
            export APP_MODE="api"
            shift
            main "$@"
            ;;
        "worker")
            export APP_MODE="worker"
            shift
            main "$@"
            ;;
        "cli")
            export APP_MODE="cli"
            shift
            main "$@"
            ;;
        "test"|"tests")
            export APP_MODE="test"
            shift
            main "$@"
            ;;
        "shell"|"python")
            export APP_MODE="shell"
            shift
            main "$@"
            ;;
        "migrate"|"migration")
            export APP_MODE="migrate"
            shift
            main "$@"
            ;;
        "bash"|"sh")
            exec bash
            ;;
        *)
            # Pass through to main
            main "$@"
            ;;
    esac
else
    # No arguments, use default APP_MODE
    main
fi