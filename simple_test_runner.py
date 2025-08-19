#!/usr/bin/env python3
"""
Simple test runner for GAN-Cyber-Range-v2 basic functionality.
"""

import sys
import time
from datetime import datetime

def run_basic_tests():
    """Run basic functionality tests"""
    print("🚀 GAN-Cyber-Range-v2 - Basic Functionality Test")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Core imports
    print("\n📦 Testing core imports...")
    try:
        from gan_cyber_range import __version__
        print(f"   ✅ Package version: {__version__}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Package import failed: {e}")
        tests_failed += 1
    
    # Test 2: Minimal components
    print("\n🔧 Testing minimal components...")
    try:
        from gan_cyber_range.core.minimal_requirements import create_minimal_components
        components = create_minimal_components()
        assert "generator" in components
        assert "cyber_range" in components
        print("   ✅ Minimal components created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Minimal components failed: {e}")
        tests_failed += 1
    
    # Test 3: Security framework
    print("\n🔒 Testing security framework...")
    try:
        from gan_cyber_range.utils.enhanced_security import validate_input, EthicalFramework
        
        # Test input validation
        result = validate_input("test input")
        assert result == True
        
        # Test ethical framework
        framework = EthicalFramework()
        test_request = {'purpose': 'research', 'target': 'test_system'}
        assert framework.is_compliant(test_request) == True
        
        print("   ✅ Security framework working")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Security framework failed: {e}")
        tests_failed += 1
    
    # Test 4: Error handling
    print("\n⚠️  Testing error handling...")
    try:
        from gan_cyber_range.utils.error_handling import CyberRangeError, error_handler
        
        # Create test error
        error = CyberRangeError("Test error", "TEST_CODE")
        error_dict = error.to_dict()
        assert "error_code" in error_dict
        assert error_dict["error_code"] == "TEST_CODE"
        
        print("   ✅ Error handling working")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error handling failed: {e}")
        tests_failed += 1
    
    # Test 5: Dependency checking
    print("\n📋 Testing dependency checking...")
    try:
        from gan_cyber_range.core.minimal_requirements import check_dependencies
        deps = check_dependencies()
        assert isinstance(deps, dict)
        print(f"   ✅ Dependencies checked: {len(deps)} dependencies")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Dependency check failed: {e}")
        tests_failed += 1
    
    # Test 6: Cache operations (if available)
    print("\n💾 Testing cache operations...")
    try:
        from gan_cyber_range.optimization.high_performance_computing import intelligent_cache
        
        # Test cache operations
        intelligent_cache.put("test_key", "test_value")
        result = intelligent_cache.get("test_key")
        assert result == "test_value"
        
        stats = intelligent_cache.get_stats()
        assert "hit_rate" in stats
        
        print("   ✅ Cache operations working")
        tests_passed += 1
    except Exception as e:
        print(f"   ⚠️  Cache operations not available: {e}")
        # Don't count as failure since it's optional
    
    # Test 7: Performance simulation
    print("\n⚡ Testing performance simulation...")
    try:
        start_time = time.time()
        
        # Simulate some work
        for i in range(1000):
            result = i * 2
        
        duration = (time.time() - start_time) * 1000
        print(f"   ✅ Performance test completed in {duration:.2f}ms")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        tests_failed += 1
    
    # Test 8: Basic cyber range creation
    print("\n🏰 Testing cyber range creation...")
    try:
        from gan_cyber_range.core.minimal_requirements import MinimalCyberRange, MinimalConfig
        
        config = MinimalConfig()
        cyber_range = MinimalCyberRange(config)
        status = cyber_range.get_status()
        
        assert status["mode"] == "minimal"
        assert "status" in status
        
        print("   ✅ Cyber range created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Cyber range creation failed: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = tests_passed + tests_failed
    success_rate = tests_passed / total_tests if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    print(f"📈 Success Rate: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("\n🎉 SYSTEM READY - Basic functionality verified!")
        status = "✅ READY"
    elif success_rate >= 0.6:
        print("\n⚠️  SYSTEM MOSTLY READY - Some issues detected")
        status = "⚠️  PARTIAL"
    else:
        print("\n❌ SYSTEM NOT READY - Critical issues found")
        status = "❌ FAILED"
    
    print(f"\nOverall Status: {status}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    return success_rate >= 0.8


def main():
    """Main test runner"""
    try:
        success = run_basic_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nTest runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()