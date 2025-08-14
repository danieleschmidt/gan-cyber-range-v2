#!/usr/bin/env python3
"""
Basic integration test for GAN Cyber Range components.
Tests critical functionality without external dependencies.
"""

import os
import sys
import traceback
from datetime import datetime

def test_imports():
    """Test that all critical modules can be imported"""
    print("Testing imports...")
    
    try:
        # Core imports
        from gan_cyber_range.core.network_sim import NetworkTopology, HostType, OSType
        from gan_cyber_range.core.attack_engine import AttackEngine, AttackStep, AttackPhase
        print("✓ Core modules imported successfully")
        
        # Evaluation imports
        from gan_cyber_range.evaluation.attack_evaluator import AttackQualityEvaluator
        from gan_cyber_range.evaluation.training_evaluator import TrainingEffectiveness
        from gan_cyber_range.evaluation.blue_team_evaluator import BlueTeamEvaluator
        print("✓ Evaluation modules imported successfully")
        
        # Optimization imports  
        from gan_cyber_range.optimization.advanced_performance import AdvancedCacheManager, ResourcePool
        print("✓ Optimization modules imported successfully")
        
        # Utils imports
        from gan_cyber_range.utils.enhanced_security import SecureInputValidator, ThreatLevel
        from gan_cyber_range.utils.comprehensive_monitoring import MetricsCollector, AlertManager
        print("✓ Utility modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_network_topology():
    """Test network topology generation"""
    print("\\nTesting network topology...")
    
    try:
        from gan_cyber_range.core.network_sim import NetworkTopology, HostType, OSType
        
        # Generate a basic topology
        topology = NetworkTopology.generate(
            template="enterprise",
            subnets=["dmz", "internal"],
            hosts_per_subnet={"dmz": 3, "internal": 5},
            services=["web", "database"],
            vulnerabilities="realistic"
        )
        
        assert len(topology.subnets) == 2, f"Expected 2 subnets, got {len(topology.subnets)}"
        assert len(topology.hosts) == 8, f"Expected 8 hosts, got {len(topology.hosts)}"
        
        # Test host filtering
        dmz_hosts = topology.get_hosts_by_subnet("dmz")
        assert len(dmz_hosts) == 3, f"Expected 3 DMZ hosts, got {len(dmz_hosts)}"
        
        print("✓ Network topology generation works")
        return True
        
    except Exception as e:
        print(f"✗ Network topology test failed: {e}")
        traceback.print_exc()
        return False


def test_attack_evaluation():
    """Test attack quality evaluation"""
    print("\\nTesting attack evaluation...")
    
    try:
        from gan_cyber_range.evaluation.attack_evaluator import AttackQualityEvaluator
        from gan_cyber_range.core.attack_gan import AttackVector
        
        # Create mock attack vectors
        attacks = [
            AttackVector(
                attack_type="web",
                payload="SELECT * FROM users WHERE id=1",
                techniques=["T1190"],
                severity=0.7,
                stealth_level=0.5,
                target_systems=["web_server"]
            ),
            AttackVector(
                attack_type="malware",
                payload="powershell -enc [base64_encoded_payload]",
                techniques=["T1059", "T1055"],
                severity=0.9,
                stealth_level=0.8,
                target_systems=["workstation"]
            )
        ]
        
        # Evaluate attacks
        evaluator = AttackQualityEvaluator()
        report = evaluator.evaluate(attacks)
        
        assert report.num_attacks_evaluated == 2, f"Expected 2 attacks evaluated, got {report.num_attacks_evaluated}"
        assert 0 <= report.overall_score <= 1, f"Overall score should be 0-1, got {report.overall_score}"
        
        print(f"✓ Attack evaluation works (Score: {report.overall_score:.3f})")
        return True
        
    except Exception as e:
        print(f"✗ Attack evaluation test failed: {e}")
        traceback.print_exc()
        return False


def test_security_validation():
    """Test security input validation"""
    print("\\nTesting security validation...")
    
    try:
        from gan_cyber_range.utils.enhanced_security import SecureInputValidator, ThreatLevel
        
        validator = SecureInputValidator()
        
        # Test safe input
        safe_result = validator.validate_input("normal_text", "general", "test_client")
        assert safe_result['is_valid'], "Safe input should be valid"
        
        # Test dangerous input
        dangerous_result = validator.validate_input("<script>alert('xss')</script>", "general", "test_client")
        assert not dangerous_result['is_valid'], "Dangerous input should be invalid"
        assert len(dangerous_result['threats_detected']) > 0, "Should detect threats"
        
        print("✓ Security validation works")
        return True
        
    except Exception as e:
        print(f"✗ Security validation test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_optimization():
    """Test performance optimization components"""
    print("\\nTesting performance optimization...")
    
    try:
        from gan_cyber_range.optimization.advanced_performance import AdvancedCacheManager
        
        # Test cache functionality
        cache = AdvancedCacheManager(max_memory_mb=1, max_entries=100)
        
        # Test basic operations
        cache.set("test_key", "test_value", 60)
        value = cache.get("test_key")
        assert value == "test_value", f"Expected 'test_value', got {value}"
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats['total_entries'] > 0, "Should have entries"
        assert stats['hits'] > 0, "Should have cache hits"
        
        print("✓ Performance optimization works")
        return True
        
    except Exception as e:
        print(f"✗ Performance optimization test failed: {e}")
        traceback.print_exc()
        return False


def test_monitoring():
    """Test monitoring and metrics"""
    print("\\nTesting monitoring...")
    
    try:
        from gan_cyber_range.utils.comprehensive_monitoring import MetricsCollector, MetricType
        
        collector = MetricsCollector(retention_hours=1)
        
        # Record some metrics
        collector.record_metric("test_metric", 42.0, {"component": "test"})
        collector.increment_counter("test_counter", 1, {"component": "test"})
        
        # Retrieve metrics
        metrics = collector.get_metrics("test_metric", {"component": "test"})
        assert len(metrics) > 0, "Should have recorded metrics"
        
        stats = collector.get_metric_statistics("test_metric", {"component": "test"})
        assert stats['latest'] == 42.0, f"Expected 42.0, got {stats['latest']}"
        
        print("✓ Monitoring works")
        return True
        
    except Exception as e:
        print(f"✗ Monitoring test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("GAN CYBER RANGE - INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    tests = [
        test_imports,
        test_network_topology,
        test_attack_evaluation,
        test_security_validation,
        test_performance_optimization,
        test_monitoring
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    print(f"Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        print("\\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        return True
    else:
        print(f"\\n⚠️  {failed} test(s) failed. Review issues before deployment.")
        return False


if __name__ == "__main__":
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    success = run_all_tests()
    sys.exit(0 if success else 1)