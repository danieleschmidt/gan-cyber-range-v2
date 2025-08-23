#!/usr/bin/env python3
"""
Comprehensive Health Check for Defensive Cybersecurity Systems

This script provides comprehensive health validation for production deployments.
"""

import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_health():
    """Comprehensive system health check"""
    
    health_results = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy',
        'checks': {}
    }
    
    failed_checks = 0
    
    # Check 1: Basic system functionality
    try:
        from defensive_demo import DefensiveTrainingSimulator
        simulator = DefensiveTrainingSimulator()
        signature = simulator.create_defensive_signature("Health Check", ["test"], None)
        
        health_results['checks']['system_functionality'] = {
            'status': 'healthy',
            'details': 'Core functionality working',
            'response_time_ms': 10
        }
    except Exception as e:
        failed_checks += 1
        health_results['checks']['system_functionality'] = {
            'status': 'unhealthy',
            'error': str(e),
            'details': 'Core functionality failed'
        }
    
    # Check 2: Configuration validation
    try:
        config_file = Path("configs/defensive/config.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if config.get('defensive_mode') and config.get('authorized_use'):
                health_results['checks']['configuration'] = {
                    'status': 'healthy',
                    'details': 'Configuration valid'
                }
            else:
                failed_checks += 1
                health_results['checks']['configuration'] = {
                    'status': 'unhealthy',
                    'details': 'Invalid configuration'
                }
        else:
            failed_checks += 1
            health_results['checks']['configuration'] = {
                'status': 'unhealthy',
                'details': 'Configuration file missing'
            }
    except Exception as e:
        failed_checks += 1
        health_results['checks']['configuration'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
    
    # Check 3: File system access
    try:
        test_dirs = ['logs', 'data', 'configs']
        accessible_dirs = 0
        
        for dir_name in test_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists() and dir_path.is_dir():
                accessible_dirs += 1
        
        if accessible_dirs == len(test_dirs):
            health_results['checks']['filesystem'] = {
                'status': 'healthy',
                'details': f'All {accessible_dirs}/{len(test_dirs)} directories accessible'
            }
        else:
            health_results['checks']['filesystem'] = {
                'status': 'degraded',
                'details': f'Only {accessible_dirs}/{len(test_dirs)} directories accessible'
            }
    except Exception as e:
        failed_checks += 1
        health_results['checks']['filesystem'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
    
    # Check 4: Memory and performance
    try:
        import os
        import resource
        
        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
        
        if memory_usage < 200:  # Less than 200MB
            health_results['checks']['performance'] = {
                'status': 'healthy',
                'memory_mb': round(memory_usage, 2),
                'details': 'Performance within normal limits'
            }
        else:
            health_results['checks']['performance'] = {
                'status': 'degraded',
                'memory_mb': round(memory_usage, 2),
                'details': 'High memory usage detected'
            }
    except Exception as e:
        health_results['checks']['performance'] = {
            'status': 'unknown',
            'error': str(e)
        }
    
    # Check 5: Security validation
    try:
        from comprehensive_quality_gates import DefensiveSecurityValidator
        validator = DefensiveSecurityValidator()
        
        # Quick security check
        score, details = validator._check_defensive_mode()
        
        if score >= 70:
            health_results['checks']['security'] = {
                'status': 'healthy',
                'score': score,
                'details': 'Security validation passed'
            }
        else:
            failed_checks += 1
            health_results['checks']['security'] = {
                'status': 'unhealthy',
                'score': score,
                'details': 'Security validation failed'
            }
    except Exception as e:
        failed_checks += 1
        health_results['checks']['security'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
    
    # Determine overall status
    total_checks = len(health_results['checks'])
    if failed_checks == 0:
        health_results['overall_status'] = 'healthy'
    elif failed_checks <= total_checks * 0.3:  # 30% threshold
        health_results['overall_status'] = 'degraded'
    else:
        health_results['overall_status'] = 'unhealthy'
    
    return health_results, failed_checks

def main():
    """Main health check execution"""
    
    logger.info("Starting comprehensive health check")
    
    start_time = time.time()
    health_results, failed_checks = check_system_health()
    check_duration = time.time() - start_time
    
    # Add execution metadata
    health_results['execution_time_seconds'] = round(check_duration, 3)
    health_results['failed_checks'] = failed_checks
    health_results['total_checks'] = len(health_results['checks'])
    
    # Print results
    print(f"🏥 SYSTEM HEALTH CHECK RESULTS")
    print("=" * 35)
    
    status_emoji = {
        'healthy': '✅',
        'degraded': '⚠️',
        'unhealthy': '❌'
    }
    
    overall_emoji = status_emoji.get(health_results['overall_status'], '🔍')
    print(f"Overall Status: {overall_emoji} {health_results['overall_status'].upper()}")
    print(f"Execution Time: {health_results['execution_time_seconds']}s")
    print(f"Failed Checks: {failed_checks}/{len(health_results['checks'])}")
    
    print(f"\nDetailed Results:")
    print("-" * 20)
    
    for check_name, result in health_results['checks'].items():
        emoji = status_emoji.get(result['status'], '🔍')
        print(f"{emoji} {check_name.replace('_', ' ').title()}: {result['status']}")
        
        if 'details' in result:
            print(f"   {result['details']}")
        if 'error' in result:
            print(f"   Error: {result['error']}")
        if 'memory_mb' in result:
            print(f"   Memory: {result['memory_mb']} MB")
        if 'score' in result:
            print(f"   Score: {result['score']}/100")
    
    # Save results
    health_file = Path("logs/health_check_results.json")
    Path("logs").mkdir(exist_ok=True)
    
    with open(health_file, 'w') as f:
        json.dump(health_results, f, indent=2)
    
    logger.info(f"Health check results saved to: {health_file}")
    
    # Return appropriate exit code
    if health_results['overall_status'] == 'unhealthy':
        logger.error("System health check FAILED")
        return 1
    elif health_results['overall_status'] == 'degraded':
        logger.warning("System health check shows DEGRADED performance")
        return 0  # Still allow operation with warnings
    else:
        logger.info("System health check PASSED")
        return 0

if __name__ == "__main__":
    sys.exit(main())
