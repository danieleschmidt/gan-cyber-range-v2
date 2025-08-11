"""
Simple test runner for GAN-Cyber-Range-v2 without external dependencies.
This validates core functionality and ensures code quality.
"""

import sys
import time
import traceback
import threading
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from gan_cyber_range.demo import (
    LightweightAttackGenerator, SimpleCyberRange, DemoAPI,
    SimpleAttackVector, demo_basic_usage
)


class SimpleTestRunner:
    """Simple test runner with basic assertions"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def assert_true(self, condition, message="Assertion failed"):
        """Simple assert true"""
        if not condition:
            raise AssertionError(message)
    
    def assert_equal(self, a, b, message=None):
        """Simple assert equal"""
        if a != b:
            if message is None:
                message = f"Expected {a} == {b}"
            raise AssertionError(message)
    
    def assert_in(self, item, container, message=None):
        """Simple assert in"""
        if item not in container:
            if message is None:
                message = f"Expected {item} in {container}"
            raise AssertionError(message)
    
    def assert_greater_equal(self, a, b, message=None):
        """Simple assert greater or equal"""
        if a < b:
            if message is None:
                message = f"Expected {a} >= {b}"
            raise AssertionError(message)
    
    def run_test(self, test_func, test_name):
        """Run a single test"""
        self.tests_run += 1
        try:
            test_func()
            print(f"✅ {test_name}")
            self.tests_passed += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            self.tests_failed += 1
            self.failures.append((test_name, str(e)))
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print(f"Test Results: {self.tests_run} run, {self.tests_passed} passed, {self.tests_failed} failed")
        print(f"Success Rate: {self.tests_passed/max(1, self.tests_run)*100:.1f}%")
        
        if self.failures:
            print("\nFailures:")
            for test_name, error in self.failures:
                print(f"  - {test_name}: {error}")
        
        print("="*60)


def test_attack_generator(runner):
    """Test the lightweight attack generator"""
    generator = LightweightAttackGenerator()
    
    # Test initialization
    runner.assert_true(generator is not None, "Generator should initialize")
    runner.assert_equal(len(generator.attack_templates), 4, "Should have 4 attack types")
    
    # Test single attack generation
    attack = generator.generate_attack()
    runner.assert_true(isinstance(attack, SimpleAttackVector), "Should generate SimpleAttackVector")
    runner.assert_in(attack.attack_type, ['malware', 'network', 'web', 'social_engineering'])
    runner.assert_true(0.0 <= attack.severity <= 1.0, "Severity should be 0-1")
    runner.assert_true(len(attack.target_systems) >= 1, "Should have target systems")
    
    # Test batch generation
    attacks = generator.generate_batch(5)
    runner.assert_equal(len(attacks), 5, "Should generate 5 attacks")
    
    # Test specific attack type
    web_attack = generator.generate_attack('web')
    runner.assert_equal(web_attack.attack_type, 'web', "Should generate web attack")


def test_cyber_range(runner):
    """Test the simple cyber range"""
    cyber_range = SimpleCyberRange("test-range")
    
    # Test initialization
    runner.assert_equal(cyber_range.name, "test-range", "Name should be set")
    runner.assert_equal(cyber_range.status, "initializing", "Should start in initializing state")
    runner.assert_equal(len(cyber_range.hosts), 6, "Should have 6 demo hosts")
    
    # Test deployment
    range_id = cyber_range.deploy()
    runner.assert_true(range_id is not None, "Should return range ID")
    runner.assert_equal(cyber_range.status, "deployed", "Should be deployed")
    runner.assert_true(cyber_range.start_time is not None, "Should set start time")
    
    # Test attack execution
    attack = cyber_range.attack_generator.generate_attack()
    result = cyber_range.execute_attack(attack)
    
    runner.assert_in('attack_id', result, "Result should have attack_id")
    runner.assert_in('success', result, "Result should have success")
    runner.assert_in('detected', result, "Result should have detected")
    runner.assert_true(isinstance(result['success'], bool), "Success should be boolean")
    
    # Test metrics
    metrics = cyber_range.get_metrics()
    expected_keys = ['range_id', 'status', 'total_attacks', 'detection_rate']
    for key in expected_keys:
        runner.assert_in(key, metrics, f"Metrics should have {key}")


def test_demo_api(runner):
    """Test the demo API functionality"""
    api = DemoAPI()
    
    # Test range creation
    response = api.create_range("test-api-range")
    runner.assert_in('range_id', response, "Response should have range_id")
    runner.assert_in('name', response, "Response should have name")
    runner.assert_equal(response['name'], "test-api-range", "Name should match")
    
    range_id = response['range_id']
    runner.assert_in(range_id, api.ranges, "Range should be stored in API")
    
    # Test range info
    info = api.get_range_info(range_id)
    runner.assert_in('range_info', info, "Should have range_info")
    runner.assert_in('metrics', info, "Should have metrics")
    
    # Test attack generation
    attack_response = api.generate_attacks(range_id, count=3, attack_type="malware")
    runner.assert_equal(attack_response['generated_attacks'], 3, "Should generate 3 attacks")
    runner.assert_equal(attack_response['attack_type'], "malware", "Attack type should match")
    
    # Test non-existent range
    error_response = api.get_range_info("nonexistent")
    runner.assert_in('error', error_response, "Should return error for nonexistent range")


def test_security_features(runner):
    """Test basic security features (without importing the full security module)"""
    
    # Test basic input validation patterns
    dangerous_inputs = [
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "rm -rf /",
        "../../../etc/passwd"
    ]
    
    safe_inputs = [
        "test_range_name",
        "user123",
        "192.168.1.100",
        "normal text input"
    ]
    
    # Simple validation logic
    def is_dangerous(input_str):
        dangerous_patterns = [
            'drop table', 'rm -rf', '<script>', '../../../',
            '; ', '|', '&', '`', '$('
        ]
        return any(pattern in input_str.lower() for pattern in dangerous_patterns)
    
    # Test dangerous inputs
    for dangerous_input in dangerous_inputs:
        runner.assert_true(is_dangerous(dangerous_input), 
                         f"Should detect danger in: {dangerous_input}")
    
    # Test safe inputs
    for safe_input in safe_inputs:
        runner.assert_true(not is_dangerous(safe_input), 
                         f"Should be safe: {safe_input}")


def test_performance(runner):
    """Test basic performance characteristics"""
    generator = LightweightAttackGenerator()
    
    # Test attack generation performance
    start_time = time.time()
    attacks = generator.generate_batch(100)
    generation_time = time.time() - start_time
    
    runner.assert_equal(len(attacks), 100, "Should generate 100 attacks")
    runner.assert_true(generation_time < 5.0, "Should generate 100 attacks in under 5 seconds")
    
    # Test attack variety
    attack_types = set(attack.attack_type for attack in attacks)
    runner.assert_greater_equal(len(attack_types), 2, "Should have variety in attack types")


def test_integration(runner):
    """Test integration scenarios"""
    api = DemoAPI()
    
    # Test multiple ranges
    ranges = []
    for i in range(3):
        response = api.create_range(f"integration-test-{i}")
        ranges.append(response['range_id'])
    
    runner.assert_equal(len(api.ranges), 3, "Should create 3 ranges")
    
    # Test operations on each range
    for range_id in ranges:
        attack_response = api.generate_attacks(range_id, count=2)
        runner.assert_equal(attack_response['generated_attacks'], 2, "Should generate 2 attacks per range")
        
        info = api.get_range_info(range_id)
        runner.assert_equal(info['metrics']['total_attacks'], 2, "Should have 2 attacks in metrics")


def test_concurrent_operations(runner):
    """Test concurrent operations"""
    api = DemoAPI()
    results = []
    
    def create_range_worker(name):
        try:
            response = api.create_range(f"concurrent-{name}")
            range_id = response['range_id']
            attack_response = api.generate_attacks(range_id, count=2)
            results.append(True)
        except Exception:
            results.append(False)
    
    # Create threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=create_range_worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join(timeout=5.0)
    
    runner.assert_equal(len(results), 3, "Should complete 3 concurrent operations")
    runner.assert_true(all(results), "All concurrent operations should succeed")


def test_demo_workflow(runner):
    """Test the complete demo workflow"""
    try:
        # Capture the demo output
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            api, range_id = demo_basic_usage()
        
        output = output_buffer.getvalue()
        
        # Verify demo ran successfully
        runner.assert_true("Demo completed successfully" in output, "Demo should complete successfully")
        runner.assert_true(range_id in api.ranges, "Demo should create a range")
        
        # Verify range has attacks
        info = api.get_range_info(range_id)
        runner.assert_true(info['metrics']['total_attacks'] > 0, "Demo should generate attacks")
        
    except Exception as e:
        # If demo fails, at least verify the basic components work
        runner.assert_true(False, f"Demo workflow failed: {e}")


def main():
    """Main test runner"""
    print("🛡️ GAN-Cyber-Range-v2 Test Suite")
    print("="*60)
    
    runner = SimpleTestRunner()
    
    # Run all tests
    test_cases = [
        (test_attack_generator, "Attack Generator Tests"),
        (test_cyber_range, "Cyber Range Tests"),
        (test_demo_api, "Demo API Tests"),
        (test_security_features, "Security Feature Tests"),
        (test_performance, "Performance Tests"),
        (test_integration, "Integration Tests"),
        (test_concurrent_operations, "Concurrent Operations Tests"),
        (test_demo_workflow, "Demo Workflow Test")
    ]
    
    for test_func, test_name in test_cases:
        print(f"\n📋 Running {test_name}...")
        runner.run_test(lambda: test_func(runner), test_name)
    
    runner.print_summary()
    
    # Return appropriate exit code
    return 0 if runner.tests_failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)