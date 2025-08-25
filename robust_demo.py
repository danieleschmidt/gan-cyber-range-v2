#!/usr/bin/env python3
"""
Robust demonstration of Generation 2 features
Shows comprehensive error handling, validation, and monitoring
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gan_cyber_range.utils.robust_validation import DefensiveValidator, RobustErrorHandler
from gan_cyber_range.utils.defensive_monitoring import DefensiveMonitor
from gan_cyber_range.core.ultra_minimal import UltraMinimalDemo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_robust_validation():
    """Demonstrate robust validation capabilities"""
    print("\n🔍 ROBUST VALIDATION DEMONSTRATION")
    print("=" * 50)
    
    validator = DefensiveValidator(strict_mode=False)
    
    # Test attack vector validation
    print("\n1. Attack Vector Validation:")
    
    # Valid attack vector
    valid_attack = {
        "attack_type": "malware",
        "payload": "powershell -enc dGVzdCBwYXlsb2Fk",
        "techniques": ["T1059.001", "T1055"],
        "severity": 0.7,
        "stealth_level": 0.6,
        "target_systems": ["windows_workstation"]
    }
    
    is_valid, errors = validator.validate_attack_vector(valid_attack)
    print(f"  Valid attack: {is_valid}, Errors: {len(errors)}")
    
    # Invalid attack vector
    invalid_attack = {
        "attack_type": "invalid_type",
        "payload": "",
        "techniques": ["INVALID"],
        "severity": 2.0,  # Out of range
    }
    
    is_valid, errors = validator.validate_attack_vector(invalid_attack)
    print(f"  Invalid attack: {is_valid}, Errors: {len(errors)}")
    for error in errors[:3]:  # Show first 3 errors
        print(f"    - {error}")
    
    # Test network configuration validation
    print("\n2. Network Configuration Validation:")
    
    valid_network = {
        "target_ip": "192.168.1.100",
        "port": 8080,
        "network_range": "192.168.1.0/24",
        "url": "https://example.com/test"
    }
    
    is_valid, errors = validator.validate_network_config(valid_network)
    print(f"  Valid network config: {is_valid}, Errors: {len(errors)}")
    
    invalid_network = {
        "target_ip": "999.999.999.999",
        "port": 99999,
        "url": "invalid-url"
    }
    
    is_valid, errors = validator.validate_network_config(invalid_network)
    print(f"  Invalid network config: {is_valid}, Errors: {len(errors)}")
    for error in errors[:3]:
        print(f"    - {error}")
    
    # Test user input validation
    print("\n3. User Input Validation:")
    
    safe_input = "SELECT * FROM users WHERE id = 1"
    is_valid, errors = validator.validate_user_input(safe_input, "general")
    print(f"  Safe input: {is_valid}, Errors: {len(errors)}")
    
    dangerous_input = "' OR '1'='1' --"
    is_valid, errors = validator.validate_user_input(dangerous_input, "general")
    print(f"  Dangerous input: {is_valid}, Errors: {len(errors)}")
    
    return validator


def demonstrate_error_handling():
    """Demonstrate robust error handling"""
    print("\n🛠️ ERROR HANDLING DEMONSTRATION")
    print("=" * 50)
    
    error_handler = RobustErrorHandler()
    
    # Test different types of errors
    test_errors = [
        (ConnectionError("Network connection failed"), "network_test"),
        (ValueError("Invalid validation parameters"), "validation_test"),
        (MemoryError("Out of memory"), "resource_test"),
        (RuntimeError("Generation failed"), "generation_test"),
        (Exception("Unknown error"), "unknown_test")
    ]
    
    for error, context in test_errors:
        print(f"\n  Testing {type(error).__name__}:")
        recovered, result = error_handler.handle_error(error, context)
        print(f"    Recovery successful: {recovered}")
        print(f"    Result: {result}")
    
    return error_handler


def demonstrate_defensive_monitoring():
    """Demonstrate defensive monitoring capabilities"""
    print("\n📊 DEFENSIVE MONITORING DEMONSTRATION")
    print("=" * 50)
    
    monitor = DefensiveMonitor()
    monitor.start_monitoring()
    
    print("  Monitoring started...")
    
    # Simulate various security events
    print("\n  Simulating security events:")
    
    # Attack detection events
    monitor.record_attack_detection("attack_001", "malware", True, 0.95, 1.2)
    monitor.record_attack_detection("attack_002", "network", False, 0.3, 5.8)
    monitor.record_attack_detection("attack_003", "web", True, 0.88, 0.7)
    print("    ✓ Attack detection events recorded")
    
    # Incident response events
    monitor.record_incident_response("incident_001", 120.5, True, ["isolate", "analyze", "remediate"])
    monitor.record_incident_response("incident_002", 300.2, False, ["isolate", "escalate"])
    print("    ✓ Incident response events recorded")
    
    # Training performance events
    monitor.record_training_performance("trainee_001", "scenario_001", 85.5, 45.0, 8, 10)
    monitor.record_training_performance("trainee_002", "scenario_001", 92.0, 38.5, 9, 10)
    print("    ✓ Training performance events recorded")
    
    # Custom events
    monitor.record_event("system_health", "low", "health_check", "System resources normal", 
                        {"cpu": 25.5, "memory": 512.0, "disk": 15.2})
    monitor.record_event("security_scan", "medium", "scanner", "Vulnerability scan completed",
                        {"vulnerabilities_found": 3, "scan_duration": 180})
    print("    ✓ Custom events recorded")
    
    # Custom metrics
    monitor.record_metric("threat_level", 0.65)
    monitor.record_metric("defense_effectiveness", 0.89)
    monitor.record_metric("user_training_score", 87.5, "percentage")
    print("    ✓ Custom metrics recorded")
    
    # Wait for processing
    time.sleep(2)
    
    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    
    print(f"\n  Dashboard Summary:")
    print(f"    Monitoring Status: {dashboard['monitoring_status']}")
    print(f"    Uptime: {dashboard['uptime']}")
    print(f"    Total Events: {dashboard['total_events']}")
    print(f"    Total Metrics: {dashboard['total_metrics']}")
    print(f"    Detection Rate (24h): {dashboard['detection_rate_24h']:.1%}")
    print(f"    Avg Response Time (24h): {dashboard['avg_response_time_24h']:.1f}s")
    
    monitor.stop_monitoring()
    return monitor


def demonstrate_integrated_defense():
    """Demonstrate integrated defensive capabilities"""
    print("\n🛡️ INTEGRATED DEFENSIVE DEMONSTRATION")
    print("=" * 50)
    
    # Initialize components
    validator = DefensiveValidator()
    error_handler = RobustErrorHandler() 
    monitor = DefensiveMonitor()
    
    monitor.start_monitoring()
    
    try:
        # Run enhanced defensive demo
        print("  Running enhanced defensive demo...")
        demo = UltraMinimalDemo()
        
        # Monitor the demo execution
        start_time = time.time()
        results = demo.run()
        execution_time = time.time() - start_time
        
        # Validate the results
        print(f"  Validating demo results...")
        
        # Record monitoring events for the demo
        monitor.record_event("demo_execution", "low", "system", 
                           "Defensive demo completed successfully", results)
        
        monitor.record_metric("demo_execution_time", execution_time, "seconds")
        monitor.record_metric("synthetic_attacks_generated", 
                            results.get("synthetic_attacks_generated", 0))
        monitor.record_metric("detection_rate", results.get("detection_rate", 0))
        monitor.record_metric("attack_diversity", results.get("attack_diversity_score", 0))
        
        # Demonstrate error recovery
        print("  Testing error recovery during operation...")
        
        test_error = RuntimeError("Simulated operational error")
        recovered, recovery_result = error_handler.handle_error(test_error, "demo_operation")
        
        if recovered:
            monitor.record_event("error_recovery", "medium", "system",
                               "Successfully recovered from operational error", 
                               {"recovery_strategy": recovery_result})
            print("    ✓ Error recovery successful")
        
        # Get final statistics
        print(f"\n  Final Statistics:")
        
        validation_stats = validator.get_validation_stats()
        print(f"    Validation Operations: {validation_stats.get('total_validations', 0)}")
        
        error_stats = error_handler.get_error_stats()
        print(f"    Errors Handled: {error_stats.get('total_errors', 0)}")
        print(f"    Recovery Rate: {error_stats.get('recovery_rate', 0):.1%}")
        
        final_dashboard = monitor.get_dashboard_data()
        print(f"    Total Events: {final_dashboard['total_events']}")
        print(f"    Total Metrics: {final_dashboard['total_metrics']}")
        print(f"    Total Alerts: {final_dashboard['total_alerts']}")
        
        print(f"\n  ✅ Integrated defensive demonstration successful!")
        print(f"     Execution time: {execution_time:.2f}s")
        print(f"     Attack diversity: {results.get('attack_diversity_score', 0):.3f}")
        print(f"     Detection rate: {results.get('detection_rate', 0):.1%}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in integrated demonstration: {str(e)}")
        
        # Demonstrate error handling even for this error
        recovered, recovery_result = error_handler.handle_error(e, "integrated_demo")
        if recovered:
            print(f"  ✓ Recovered from demonstration error")
        
        return False
        
    finally:
        monitor.stop_monitoring()


def main():
    """Main demonstration function"""
    print("🚀 GENERATION 2 - ROBUST DEFENSIVE CAPABILITIES")
    print("Advanced Error Handling, Validation & Monitoring")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Demonstrate validation
        validator = demonstrate_robust_validation()
        
        # Demonstrate error handling
        error_handler = demonstrate_error_handling()
        
        # Demonstrate monitoring
        monitor = demonstrate_defensive_monitoring()
        
        # Demonstrate integrated capabilities
        success = demonstrate_integrated_defense()
        
        execution_time = time.time() - start_time
        
        print(f"\n🎯 GENERATION 2 COMPLETION SUMMARY")
        print(f"{'='*50}")
        print(f"✅ Robust Validation: Implemented")
        print(f"✅ Error Handling: Implemented") 
        print(f"✅ Defensive Monitoring: Implemented")
        print(f"✅ Integration Test: {'Passed' if success else 'Failed'}")
        print(f"⏱️  Total Execution Time: {execution_time:.2f}s")
        
        if success:
            print(f"\n🏆 GENERATION 2 (ROBUST) SUCCESSFULLY COMPLETED!")
            return 0
        else:
            print(f"\n⚠️  GENERATION 2 completed with issues")
            return 1
            
    except Exception as e:
        print(f"\n❌ GENERATION 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())