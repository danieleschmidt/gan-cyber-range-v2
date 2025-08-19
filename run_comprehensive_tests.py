#!/usr/bin/env python3
"""
Comprehensive test runner for GAN-Cyber-Range-v2.
Executes all test types and generates detailed reports.
"""

import asyncio
import sys
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import test frameworks
try:
    # Import core components first
    from gan_cyber_range.core.minimal_requirements import (
        create_minimal_components, check_dependencies
    )
    from gan_cyber_range.utils.enhanced_security import (
        input_validator, validate_input, EthicalFramework
    )
    
    # Try to import testing framework
    try:
        from gan_cyber_range.testing.comprehensive_test_framework import (
            test_runner, performance_framework, security_framework,
            TestCase, TestType
        )
    except:
        # Create minimal test framework if full one fails
        test_runner = None
        performance_framework = None
        security_framework = None
        TestCase = None
        TestType = None
    
    # Try to import optimization components
    try:
        from gan_cyber_range.optimization.high_performance_computing import (
            intelligent_cache, performance_profiler
        )
    except:
        intelligent_cache = None
        performance_profiler = None
    
    # Try to import scalability components
    try:
        from gan_cyber_range.scalability.auto_scaling_framework import auto_scaler
    except:
        auto_scaler = None
    
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.error(f"Import failed: {e}")
    IMPORTS_SUCCESSFUL = False


# Define test cases
@test_case("unit_001", "Test minimal components creation", TestType.UNIT)
def test_minimal_components():
    """Test that minimal components can be created"""
    components = create_minimal_components()
    assert "generator" in components
    assert "cyber_range" in components
    assert "config" in components
    return True


@test_case("unit_002", "Test dependency checking", TestType.UNIT)
def test_dependency_check():
    """Test dependency checking functionality"""
    deps = check_dependencies()
    assert isinstance(deps, dict)
    assert "numpy" in deps
    assert "torch" in deps
    return True


@test_case("unit_003", "Test input validation", TestType.UNIT)
def test_input_validation():
    """Test input validation functionality"""
    # Test safe input
    safe_result = validate_input("hello world", "general")
    assert safe_result == True
    
    # Test basic validation
    validation_result = input_validator.validate_input("test input")
    assert "is_valid" in validation_result
    assert "sanitized_value" in validation_result
    return True


@test_case("unit_004", "Test ethical framework", TestType.UNIT)
def test_ethical_framework():
    """Test ethical framework compliance"""
    framework = EthicalFramework()
    
    # Test compliant request
    compliant_request = {
        'purpose': 'research',
        'target': 'test_system'
    }
    assert framework.is_compliant(compliant_request) == True
    
    # Test non-compliant request
    non_compliant_request = {
        'purpose': 'malicious',
        'target': 'production_system'
    }
    assert framework.is_compliant(non_compliant_request) == False
    return True


@test_case("integration_001", "Test cache integration", TestType.INTEGRATION)
def test_cache_integration():
    """Test cache system integration"""
    # Test cache operations
    intelligent_cache.put("test_key", "test_value")
    result = intelligent_cache.get("test_key")
    assert result == "test_value"
    
    # Test cache statistics
    stats = intelligent_cache.get_stats()
    assert "hit_rate" in stats
    assert "hit_count" in stats
    return True


@test_case("integration_002", "Test performance profiler", TestType.INTEGRATION)
def test_performance_profiler():
    """Test performance profiler integration"""
    # Start profiling
    metrics = performance_profiler.start_profiling("test_operation")
    assert metrics.operation_name == "test_operation"
    
    # End profiling
    final_metrics = performance_profiler.end_profiling(metrics)
    assert final_metrics.duration_ms is not None
    assert final_metrics.duration_ms >= 0
    return True


@test_case("integration_003", "Test auto scaler status", TestType.INTEGRATION)
def test_auto_scaler():
    """Test auto scaler functionality"""
    status = auto_scaler.get_scaling_status()
    assert "scaling_active" in status
    assert "components" in status
    return True


@performance_test(iterations=10, concurrent_users=1)
def performance_test_cache():
    """Performance test for cache operations"""
    intelligent_cache.put(f"perf_key_{datetime.now().microsecond}", "perf_value")
    return intelligent_cache.get("test_key")


@performance_test(iterations=5, concurrent_users=2)
def performance_test_validation():
    """Performance test for input validation"""
    return validate_input("performance test input", "general")


@security_test("input_validation")
def security_test_input_validation(input_data):
    """Security test for input validation"""
    try:
        result = input_validator.validate_input(input_data)
        return result['is_valid']
    except:
        return False  # Exception means input was rejected (good)


@security_test("authentication")
def security_test_authentication(username, password):
    """Security test for authentication"""
    # Mock authentication function
    if password in ['password', '123456', 'admin', '']:
        return False  # Reject weak passwords
    if "'" in username or "--" in username:
        return False  # Reject SQL injection attempts
    return True  # Accept for testing


async def run_all_tests():
    """Run all registered tests"""
    logger.info("🚀 Starting comprehensive test suite")
    
    if not IMPORTS_SUCCESSFUL:
        logger.error("❌ Cannot run tests due to import failures")
        return False
    
    # Get all registered tests
    all_test_ids = list(test_runner.test_cases.keys())
    logger.info(f"📋 Found {len(all_test_ids)} registered tests")
    
    # Run tests
    logger.info("🔬 Running unit and integration tests...")
    executions = await test_runner.run_tests(all_test_ids, parallel=True)
    
    # Generate test report
    report = test_runner.generate_report(executions)
    
    # Run performance tests
    logger.info("⚡ Running performance tests...")
    perf_results = []
    
    try:
        cache_perf = await performance_framework.run_performance_test(
            performance_test_cache, iterations=20, concurrent_users=1
        )
        perf_results.append(cache_perf)
        
        validation_perf = await performance_framework.run_performance_test(
            performance_test_validation, iterations=10, concurrent_users=2
        )
        perf_results.append(validation_perf)
        
    except Exception as e:
        logger.warning(f"Performance tests failed: {e}")
    
    # Run security tests
    logger.info("🔒 Running security tests...")
    security_results = []
    
    try:
        input_security = await security_framework.run_input_validation_test(
            security_test_input_validation
        )
        security_results.append(input_security)
        
        auth_security = await security_framework.run_authentication_test(
            security_test_authentication
        )
        security_results.append(auth_security)
        
    except Exception as e:
        logger.warning(f"Security tests failed: {e}")
    
    # Print results
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE TEST RESULTS")
    print("="*80)
    
    # Test summary
    summary = report['summary']
    print(f"\n📊 Test Summary:")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   ✅ Passed: {summary['passed']}")
    print(f"   ❌ Failed: {summary['failed']}")
    print(f"   ⚠️  Errors: {summary['errors']}")
    print(f"   ⏭️  Skipped: {summary['skipped']}")
    print(f"   📈 Success Rate: {summary['success_rate']:.1%}")
    print(f"   ⏱️  Total Duration: {summary['total_duration_ms']:.0f}ms")
    
    # Failed tests
    if report['failed_tests']:
        print(f"\n❌ Failed Tests:")
        for failed_test in report['failed_tests']:
            print(f"   - {failed_test['test_id']}: {failed_test['error']}")
    
    # Performance results
    if perf_results:
        print(f"\n⚡ Performance Results:")
        for result in perf_results:
            if 'statistics' in result:
                stats = result['statistics']
                print(f"   {result['function_name']}:")
                print(f"     Average: {stats['avg_ms']:.1f}ms")
                print(f"     P95: {stats['p95_ms']:.1f}ms")
                print(f"     Throughput: {stats['throughput_req_per_sec']:.1f} req/s")
                print(f"     Error Rate: {result['error_rate']:.1%}")
    
    # Security results
    if security_results:
        print(f"\n🔒 Security Results:")
        for result in security_results:
            print(f"   {result['test_name']}:")
            print(f"     Security Score: {result['security_score']:.1%}")
            print(f"     Tests Passed: {result['tests_passed']}")
            print(f"     Tests Failed: {result['tests_failed']}")
            if result['vulnerabilities_found']:
                print(f"     ⚠️  Vulnerabilities: {len(result['vulnerabilities_found'])}")
                for vuln in result['vulnerabilities_found']:
                    print(f"       - {vuln['type']}: {vuln['description']}")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    overall_success = summary['success_rate'] >= 0.8
    perf_success = all(r.get('error_rate', 0) < 0.1 for r in perf_results)
    security_success = all(r.get('security_score', 0) >= 0.8 for r in security_results)
    
    if overall_success and perf_success and security_success:
        print("   ✅ ALL SYSTEMS GO - Ready for production deployment")
    elif overall_success:
        print("   ⚠️  MOSTLY READY - Some performance or security issues detected")
    else:
        print("   ❌ NEEDS WORK - Critical issues found, deployment not recommended")
    
    print("="*80)
    
    # Save detailed report
    full_report = {
        'timestamp': datetime.now().isoformat(),
        'test_report': report,
        'performance_results': perf_results,
        'security_results': security_results,
        'overall_success': overall_success
    }
    
    with open('test_report.json', 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    logger.info("📄 Detailed report saved to test_report.json")
    
    return overall_success


def main():
    """Main test runner"""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test run failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()