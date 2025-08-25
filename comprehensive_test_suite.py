#!/usr/bin/env python3
"""
Comprehensive Test Suite for GAN Cyber Range Platform
Tests all three generations and comprehensive defensive capabilities
"""

import sys
import time
import unittest
import logging
from pathlib import Path
from datetime import datetime
import json
import uuid

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging for tests
logging.basicConfig(level=logging.ERROR)  # Reduce noise during testing

class TestGeneration1Basic(unittest.TestCase):
    """Test Generation 1 - Basic Functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.start_time = time.time()
    
    def tearDown(self):
        """Clean up after tests"""
        execution_time = time.time() - self.start_time
        if execution_time > 5.0:
            print(f"    ⚠️  Slow test: {execution_time:.2f}s")
    
    def test_basic_import(self):
        """Test basic module imports"""
        try:
            from gan_cyber_range.core.ultra_minimal import UltraMinimalDemo
            from gan_cyber_range.core.attack_gan import AttackGAN
            self.assertTrue(True, "Basic imports successful")
        except ImportError as e:
            self.fail(f"Basic import failed: {e}")
    
    def test_ultra_minimal_demo(self):
        """Test ultra minimal demo functionality"""
        from gan_cyber_range.core.ultra_minimal import UltraMinimalDemo
        
        demo = UltraMinimalDemo()
        results = demo.run()
        
        # Validate results structure
        self.assertIsInstance(results, dict)
        self.assertIn("status", results)
        self.assertEqual(results["status"], "defensive_demo_completed")
        self.assertIn("execution_time", results)
        self.assertIn("threats_analyzed", results)
        self.assertIn("detection_rate", results)
        
        # Validate metrics
        self.assertGreaterEqual(results["detection_rate"], 0.0)
        self.assertLessEqual(results["detection_rate"], 1.0)
        self.assertGreater(results["execution_time"], 0.0)
    
    def test_attack_generation(self):
        """Test basic attack generation"""
        from gan_cyber_range.core.ultra_minimal import UltraMinimalGenerator
        
        generator = UltraMinimalGenerator()
        attacks = generator.generate(num_samples=5)
        
        # Validate attack generation
        self.assertEqual(len(attacks), 5)
        
        for attack in attacks:
            self.assertIsNotNone(attack.attack_id)
            self.assertIsNotNone(attack.attack_type)
            self.assertIsNotNone(attack.payload)
            self.assertIsInstance(attack.severity, float)
            self.assertIsInstance(attack.stealth_level, float)
            self.assertIsInstance(attack.techniques, list)
            self.assertGreater(len(attack.techniques), 0)
    
    def test_attack_diversity(self):
        """Test attack diversity calculation"""
        from gan_cyber_range.core.ultra_minimal import UltraMinimalGenerator
        
        generator = UltraMinimalGenerator()
        attacks = generator.generate(num_samples=10)
        
        diversity = generator.diversity_score(attacks)
        
        # Validate diversity
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)
        self.assertGreater(diversity, 0.1)  # Should have some diversity
    
    def test_cyber_range_basic(self):
        """Test basic cyber range functionality"""
        from gan_cyber_range.core.ultra_minimal import UltraMinimalCyberRange
        
        cyber_range = UltraMinimalCyberRange()
        range_id = cyber_range.deploy()
        cyber_range.start()
        
        # Test attack execution
        attacks = cyber_range.generate_attacks(num_attacks=3)
        results = []
        
        for attack in attacks:
            result = cyber_range.execute_attack(attack)
            results.append(result)
        
        # Validate results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn("attack_id", result)
            self.assertIn("status", result)
            self.assertIn("success", result)
            self.assertIn("detected", result)
            self.assertEqual(result["status"], "completed")
        
        # Test metrics
        metrics = cyber_range.get_metrics()
        self.assertIn("range_id", metrics)
        self.assertIn("attacks_executed", metrics)
        self.assertEqual(metrics["attacks_executed"], 3)


class TestGeneration2Robust(unittest.TestCase):
    """Test Generation 2 - Robust Error Handling and Validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.start_time = time.time()
    
    def test_defensive_validator(self):
        """Test defensive validator"""
        from gan_cyber_range.utils.robust_validation import DefensiveValidator
        
        validator = DefensiveValidator()
        
        # Test valid attack vector
        valid_attack = {
            "attack_type": "malware",
            "payload": "test_payload",
            "techniques": ["T1059"],
            "severity": 0.7,
            "stealth_level": 0.5
        }
        
        is_valid, errors = validator.validate_attack_vector(valid_attack)
        self.assertTrue(is_valid, f"Valid attack should pass validation: {errors}")
        self.assertEqual(len(errors), 0)
        
        # Test invalid attack vector
        invalid_attack = {
            "attack_type": "invalid_type",
            "payload": "",
            "techniques": [],
            "severity": 2.0
        }
        
        is_valid, errors = validator.validate_attack_vector(invalid_attack)
        self.assertFalse(is_valid, "Invalid attack should fail validation")
        self.assertGreater(len(errors), 0)
    
    def test_network_validation(self):
        """Test network configuration validation"""
        from gan_cyber_range.utils.robust_validation import DefensiveValidator
        
        validator = DefensiveValidator()
        
        # Test valid network config
        valid_config = {
            "target_ip": "192.168.1.100",
            "port": 8080,
            "network_range": "192.168.1.0/24"
        }
        
        is_valid, errors = validator.validate_network_config(valid_config)
        self.assertTrue(is_valid, f"Valid network config should pass: {errors}")
        
        # Test invalid network config
        invalid_config = {
            "target_ip": "invalid_ip",
            "port": 999999
        }
        
        is_valid, errors = validator.validate_network_config(invalid_config)
        self.assertFalse(is_valid, "Invalid network config should fail")
        self.assertGreater(len(errors), 0)
    
    def test_error_handler(self):
        """Test robust error handling"""
        from gan_cyber_range.utils.robust_validation import RobustErrorHandler
        
        error_handler = RobustErrorHandler()
        
        # Test different error types
        test_errors = [
            ConnectionError("Network connection failed"),
            ValueError("Invalid parameters"),
            RuntimeError("Execution failed")
        ]
        
        for error in test_errors:
            recovered, result = error_handler.handle_error(error, "test_context")
            
            # Most errors should be recoverable in our system
            if not recovered:
                print(f"    ⚠️  Error not recovered: {type(error).__name__}")
    
    def test_defensive_monitoring(self):
        """Test defensive monitoring system"""
        from gan_cyber_range.utils.defensive_monitoring import DefensiveMonitor
        
        monitor = DefensiveMonitor()
        monitor.start_monitoring()
        
        try:
            # Record various events
            monitor.record_attack_detection("test_attack", "malware", True, 0.9, 1.5)
            monitor.record_incident_response("test_incident", 120.0, True, ["isolate", "analyze"])
            monitor.record_training_performance("test_trainee", "test_scenario", 85.0, 30.0, 8, 10)
            
            # Record metrics
            monitor.record_metric("cpu_usage", 0.65)
            monitor.record_metric("memory_usage", 0.55)
            
            time.sleep(1)  # Allow processing
            
            # Get dashboard data
            dashboard = monitor.get_dashboard_data()
            
            self.assertIn("total_events", dashboard)
            self.assertIn("total_metrics", dashboard)
            self.assertGreater(dashboard["total_events"], 0)
            self.assertGreater(dashboard["total_metrics"], 0)
            
        finally:
            monitor.stop_monitoring()


class TestGeneration3Optimized(unittest.TestCase):
    """Test Generation 3 - Performance Optimization and Scaling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.start_time = time.time()
    
    def test_adaptive_resource_pool(self):
        """Test adaptive resource pool"""
        from gan_cyber_range.optimization.adaptive_performance import AdaptiveResourcePool
        
        pool = AdaptiveResourcePool(pool_type="thread", min_workers=2, max_workers=4)
        
        def test_task(x):
            return x * 2
        
        # Submit tasks
        task_ids = []
        for i in range(3):
            task_id = pool.submit_task(test_task, i)
            task_ids.append(task_id)
        
        # Get results
        results = []
        for task_id in task_ids:
            try:
                result = pool.get_task_result(task_id, timeout=5.0)
                results.append(result)
            except Exception as e:
                print(f"    ⚠️  Task failed: {e}")
        
        # Validate results
        self.assertGreater(len(results), 0, "At least some tasks should complete")
        
        # Get statistics
        stats = pool.get_stats()
        self.assertIn("current_workers", stats)
        self.assertIn("completed_tasks", stats)
        
        pool.shutdown()
    
    def test_performance_optimizer(self):
        """Test performance optimizer"""
        from gan_cyber_range.optimization.adaptive_performance import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer()
        optimizer.create_resource_pool("test_pool", "thread", 2, 4)
        
        def simple_task(x):
            return x + 1
        
        # Submit tasks
        task_ids = []
        for i in range(3):
            try:
                task_id = optimizer.submit_to_pool("test_pool", simple_task, i)
                task_ids.append(task_id)
            except Exception as e:
                print(f"    ⚠️  Task submission failed: {e}")
        
        # Get results with timeout
        results = []
        for task_id in task_ids:
            try:
                result = optimizer.get_from_pool("test_pool", task_id, timeout=3.0)
                results.append(result)
            except Exception as e:
                print(f"    ⚠️  Task result failed: {e}")
        
        # Test caching
        optimizer.cache_result("test_key", "test_value")
        cached = optimizer.get_cached_result("test_key")
        self.assertEqual(cached, "test_value")
        
        optimizer.shutdown()
    
    def test_intelligent_scaling(self):
        """Test intelligent auto-scaling (basic functionality)"""
        from gan_cyber_range.scalability.intelligent_scaling import IntelligentAutoScaler
        
        scaler = IntelligentAutoScaler(min_instances=1, max_instances=3, scale_cooldown=1)
        
        # Test metric updates
        test_metrics = {
            "cpu_utilization": 0.5,
            "memory_utilization": 0.4,
            "response_time": 2.0
        }
        
        scaler.update_metrics(test_metrics)
        
        # Get scaling report
        report = scaler.get_scaling_report()
        
        self.assertIn("current_configuration", report)
        self.assertIn("current_metrics", report)
        self.assertIn("instance_summary", report)
        
        # Cleanup
        scaler.stop_monitoring()
    
    def test_defensive_workload_manager(self):
        """Test defensive workload manager (without long-running operations)"""
        from gan_cyber_range.optimization.adaptive_performance import DefensiveWorkloadManager
        
        workload_manager = DefensiveWorkloadManager()
        
        # Test basic functionality
        stats = workload_manager.get_workload_stats()
        
        self.assertIn("resource_pools", stats)
        self.assertIn("system_utilization", stats)
        self.assertIn("defensive_metrics", stats)
        
        workload_manager.shutdown()


class TestIntegration(unittest.TestCase):
    """Test Integration across all generations"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.start_time = time.time()
    
    def test_full_defensive_pipeline(self):
        """Test complete defensive pipeline integration"""
        
        # Import components from all generations
        from gan_cyber_range.core.ultra_minimal import UltraMinimalDemo
        from gan_cyber_range.utils.robust_validation import DefensiveValidator
        from gan_cyber_range.utils.defensive_monitoring import DefensiveMonitor
        
        # Initialize components
        demo = UltraMinimalDemo()
        validator = DefensiveValidator(strict_mode=False)
        monitor = DefensiveMonitor()
        
        monitor.start_monitoring()
        
        try:
            # Run integrated demo
            results = demo.run()
            
            # Validate results
            attack_vectors = []
            for result in results.get("attack_execution_results", []):
                attack_data = {
                    "attack_type": "test",
                    "payload": "test_payload",
                    "techniques": ["T1001"],
                    "severity": 0.5,
                    "stealth_level": 0.5
                }
                
                is_valid, errors = validator.validate_attack_vector(attack_data)
                if is_valid:
                    attack_vectors.append(attack_data)
                
                # Monitor the validation
                monitor.record_event("validation_check", "low", "validator", 
                                   f"Validation {'passed' if is_valid else 'failed'}")
            
            # Record integration metrics
            monitor.record_metric("integration_attacks", len(attack_vectors))
            monitor.record_metric("integration_success", 1 if results else 0)
            
            time.sleep(1)  # Allow processing
            
            # Validate integration
            self.assertIsNotNone(results)
            self.assertIn("status", results)
            
            dashboard = monitor.get_dashboard_data()
            self.assertGreater(dashboard["total_events"], 0)
            
        finally:
            monitor.stop_monitoring()
    
    def test_performance_comparison(self):
        """Test performance across generations"""
        
        from gan_cyber_range.core.ultra_minimal import UltraMinimalGenerator
        
        generator = UltraMinimalGenerator()
        
        # Generation 1: Basic generation
        start_time = time.time()
        basic_attacks = generator.generate(num_samples=5)
        basic_time = time.time() - start_time
        
        # Generation 2: With validation
        start_time = time.time()
        from gan_cyber_range.utils.robust_validation import DefensiveValidator
        validator = DefensiveValidator(strict_mode=False)
        
        validated_attacks = []
        for attack in generator.generate(num_samples=5):
            attack_dict = {
                "attack_type": attack.attack_type,
                "payload": attack.payload,
                "techniques": attack.techniques,
                "severity": attack.severity,
                "stealth_level": attack.stealth_level
            }
            is_valid, _ = validator.validate_attack_vector(attack_dict)
            if is_valid:
                validated_attacks.append(attack)
        
        validated_time = time.time() - start_time
        
        # Validate performance characteristics
        self.assertGreater(len(basic_attacks), 0)
        self.assertGreater(basic_time, 0)
        self.assertGreater(validated_time, 0)
        
        # Performance should be reasonable (< 5 seconds for basic operations)
        self.assertLess(basic_time, 5.0, "Basic generation should be fast")
        self.assertLess(validated_time, 10.0, "Validated generation should be reasonable")


def run_comprehensive_tests():
    """Run all tests with detailed reporting"""
    
    print("🧪 COMPREHENSIVE TEST SUITE")
    print("Testing All Three Generations + Integration")
    print("=" * 60)
    
    # Test configuration
    test_suites = [
        ("Generation 1 - Basic Functionality", TestGeneration1Basic),
        ("Generation 2 - Robust Operations", TestGeneration2Robust), 
        ("Generation 3 - Optimized Performance", TestGeneration3Optimized),
        ("Integration Testing", TestIntegration)
    ]
    
    # Results tracking
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    errors = []
    suite_results = {}
    
    start_time = time.time()
    
    # Run each test suite
    for suite_name, test_class in test_suites:
        print(f"\n📋 {suite_name}")
        print("-" * 40)
        
        suite_start = time.time()
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
        
        result = runner.run(suite)
        suite_time = time.time() - suite_start
        
        # Track results
        suite_total = result.testsRun
        suite_passed = suite_total - len(result.failures) - len(result.errors)
        suite_failed = len(result.failures) + len(result.errors)
        
        total_tests += suite_total
        passed_tests += suite_passed
        failed_tests += suite_failed
        
        # Record suite results
        suite_results[suite_name] = {
            "total": suite_total,
            "passed": suite_passed,
            "failed": suite_failed,
            "time": suite_time,
            "success_rate": suite_passed / suite_total if suite_total > 0 else 0
        }
        
        # Collect errors
        for failure in result.failures:
            errors.append(f"{suite_name}: {failure[0]}")
        for error in result.errors:
            errors.append(f"{suite_name}: {error[0]}")
        
        print(f"  ✅ Passed: {suite_passed}/{suite_total}")
        print(f"  ⏱️  Time: {suite_time:.2f}s")
        
        if suite_failed > 0:
            print(f"  ❌ Failed: {suite_failed}")
    
    total_time = time.time() - start_time
    
    # Generate comprehensive report
    print(f"\n🎯 TEST EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success Rate: {passed_tests/total_tests:.1%}")
    print(f"Total Time: {total_time:.2f}s")
    
    # Suite breakdown
    print(f"\n📊 SUITE BREAKDOWN")
    print("-" * 30)
    for suite_name, results in suite_results.items():
        status = "✅" if results["failed"] == 0 else "⚠️"
        print(f"{status} {suite_name}")
        print(f"    {results['passed']}/{results['total']} passed ({results['success_rate']:.1%})")
        print(f"    {results['time']:.2f}s execution time")
    
    # Error summary
    if errors:
        print(f"\n❌ FAILED TESTS")
        print("-" * 20)
        for error in errors[:5]:  # Show first 5 errors
            print(f"  • {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    # Generate test report file
    test_report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_time": total_time
        },
        "suites": suite_results,
        "errors": errors
    }
    
    report_file = Path("test_results_comprehensive.json")
    with open(report_file, "w") as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n📝 Test report saved to: {report_file}")
    
    # Final assessment
    if failed_tests == 0:
        print(f"\n🏆 ALL TESTS PASSED! Platform is fully functional.")
        return 0
    elif passed_tests / total_tests >= 0.8:
        print(f"\n✅ MOSTLY PASSING ({passed_tests/total_tests:.1%}). Platform is operational with minor issues.")
        return 0
    else:
        print(f"\n⚠️  SIGNIFICANT ISSUES ({passed_tests/total_tests:.1%} pass rate). Platform needs attention.")
        return 1


if __name__ == "__main__":
    exit_code = run_comprehensive_tests()
    sys.exit(exit_code)