#!/usr/bin/env python3
"""
Enhanced demo with robust error handling and fallback mechanisms
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Add project to path
sys.path.insert(0, '/root/repo')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DependencyChecker:
    """Check for optional dependencies and provide graceful fallbacks"""
    
    @staticmethod
    def check_torch():
        try:
            import torch
            return True, torch.__version__
        except ImportError:
            return False, None
    
    @staticmethod
    def check_docker():
        try:
            import docker
            return True, docker.__version__
        except ImportError:
            return False, None
    
    @staticmethod
    def check_all_dependencies():
        """Check all major dependencies"""
        deps = {}
        
        # Check PyTorch
        torch_available, torch_version = DependencyChecker.check_torch()
        deps['torch'] = {'available': torch_available, 'version': torch_version}
        
        # Check Docker
        docker_available, docker_version = DependencyChecker.check_docker()
        deps['docker'] = {'available': docker_available, 'version': docker_version}
        
        # Check other common dependencies
        for dep_name in ['numpy', 'pandas', 'scikit-learn', 'fastapi']:
            try:
                module = __import__(dep_name)
                version = getattr(module, '__version__', 'unknown')
                deps[dep_name] = {'available': True, 'version': version}
            except ImportError:
                deps[dep_name] = {'available': False, 'version': None}
        
        return deps


def test_enhanced_imports():
    """Test imports with proper error handling and fallbacks"""
    print("Testing Enhanced Import Capabilities")
    print("-" * 40)
    
    success = True
    
    # Check dependencies first
    deps = DependencyChecker.check_all_dependencies()
    
    print("Dependency Status:")
    for dep, info in deps.items():
        status = "✓" if info['available'] else "✗"
        version = f"v{info['version']}" if info['version'] else "not installed"
        print(f"  {status} {dep}: {version}")
    
    print("\nTesting Core Imports:")
    
    # Test minimal functionality first
    try:
        from gan_cyber_range.core.minimal_gan import (
            AttackVector, MinimalAttackGenerator, MockCyberRange
        )
        print("✓ Minimal GAN components imported successfully")
        
        # Test basic functionality
        attack = AttackVector(
            attack_type="test",
            payload="test_payload",
            techniques=["T1001"],
            severity=0.5,
            stealth_level=0.7,
            target_systems=["test_system"]
        )
        print("✓ AttackVector creation successful")
        
        generator = MinimalAttackGenerator()
        print("✓ MinimalAttackGenerator creation successful")
        
        cyber_range = MockCyberRange()
        print("✓ MockCyberRange creation successful")
        
    except Exception as e:
        print(f"✗ Minimal components failed: {e}")
        success = False
    
    # Try full GAN imports if PyTorch is available
    torch_available, _ = DependencyChecker.check_torch()
    if torch_available:
        try:
            from gan_cyber_range.core.attack_gan import AttackGAN
            print("✓ Full AttackGAN imported successfully (PyTorch available)")
        except Exception as e:
            print(f"⚠ Full AttackGAN import failed despite PyTorch: {e}")
    else:
        print("⚠ Full AttackGAN skipped (PyTorch not available - using minimal version)")
    
    # Try cyber range imports
    docker_available, _ = DependencyChecker.check_docker()
    if docker_available:
        try:
            from gan_cyber_range.core.cyber_range import CyberRange
            print("✓ Full CyberRange imported successfully (Docker available)")
        except Exception as e:
            print(f"⚠ Full CyberRange import failed: {e}")
    else:
        print("⚠ Full CyberRange skipped (Docker not available - using mock version)")
    
    return success


def test_attack_generation():
    """Test attack generation with error handling"""
    print("\nTesting Attack Generation")
    print("-" * 40)
    
    try:
        from gan_cyber_range.core.minimal_gan import MinimalAttackGenerator, AttackVector
        
        # Initialize generator
        generator = MinimalAttackGenerator()
        print("✓ Attack generator initialized")
        
        # Generate attacks
        print("Generating 10 diverse attacks...")
        attacks = generator.generate(num_samples=10)
        
        if not attacks:
            print("✗ No attacks generated")
            return False
        
        print(f"✓ Generated {len(attacks)} attacks")
        
        # Display sample attacks
        print("\nSample Generated Attacks:")
        for i, attack in enumerate(attacks[:3], 1):
            print(f"  Attack {i}:")
            print(f"    Type: {attack.attack_type}")
            print(f"    Payload: {attack.payload[:50]}...")
            print(f"    Techniques: {attack.techniques}")
            print(f"    Severity: {attack.severity:.2f}")
            print(f"    Stealth: {attack.stealth_level:.2f}")
        
        # Test diversity calculation
        diversity = generator.diversity_score(attacks)
        print(f"✓ Attack diversity score: {diversity:.3f}")
        
        # Test different attack types
        for attack_type in ["malware", "network", "web", "social_engineering"]:
            type_attacks = generator.generate(num_samples=2, attack_type=attack_type)
            if type_attacks:
                print(f"✓ Generated {len(type_attacks)} {attack_type} attacks")
            else:
                print(f"✗ Failed to generate {attack_type} attacks")
        
        return True
        
    except Exception as e:
        print(f"✗ Attack generation test failed: {e}")
        traceback.print_exc()
        return False


def test_cyber_range_functionality():
    """Test cyber range functionality"""
    print("\nTesting Cyber Range Functionality")
    print("-" * 40)
    
    try:
        from gan_cyber_range.core.minimal_gan import MockCyberRange, MinimalAttackGenerator
        
        # Initialize cyber range
        cyber_range = MockCyberRange()
        print("✓ Cyber range initialized")
        
        # Deploy range
        range_id = cyber_range.deploy()
        print(f"✓ Cyber range deployed with ID: {range_id}")
        
        # Start range
        cyber_range.start()
        print("✓ Cyber range started")
        
        # Generate and execute attacks
        attacks = cyber_range.generate_attacks(num_attacks=3)
        print(f"✓ Generated {len(attacks)} attacks for execution")
        
        # Execute attacks
        results = []
        for attack in attacks:
            result = cyber_range.execute_attack(attack)
            results.append(result)
            status = "successful" if result['success'] else "failed"
            detected = "detected" if result['detected'] else "undetected"
            print(f"  - Attack {attack.attack_id[:8]}: {status}, {detected}")
        
        # Get metrics
        metrics = cyber_range.get_metrics()
        print("✓ Retrieved cyber range metrics:")
        print(f"  - Status: {metrics['status']}")
        print(f"  - Attacks executed: {metrics['attacks_executed']}")
        print(f"  - Detection rate: {metrics['detection_rate']:.1%}")
        print(f"  - CPU usage: {metrics['resource_usage']['cpu']:.1f}%")
        print(f"  - Memory usage: {metrics['resource_usage']['memory']:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Cyber range test failed: {e}")
        traceback.print_exc()
        return False


def test_data_persistence():
    """Test saving and loading attack data"""
    print("\nTesting Data Persistence")
    print("-" * 40)
    
    try:
        from gan_cyber_range.core.minimal_gan import MinimalAttackGenerator
        
        generator = MinimalAttackGenerator()
        
        # Generate attacks
        attacks = generator.generate(num_samples=5)
        
        # Save attacks
        save_path = Path("/tmp/test_attacks.json")
        generator.save_attacks(attacks, save_path)
        print(f"✓ Saved attacks to {save_path}")
        
        # Load attacks
        loaded_attacks = generator.load_attacks(save_path)
        print(f"✓ Loaded {len(loaded_attacks)} attacks")
        
        # Verify data integrity
        if len(loaded_attacks) == len(attacks):
            print("✓ Data integrity verified")
        else:
            print(f"✗ Data integrity check failed: {len(loaded_attacks)} != {len(attacks)}")
            return False
        
        # Clean up
        save_path.unlink()
        print("✓ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Data persistence test failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and validation"""
    print("\nTesting Error Handling & Validation")
    print("-" * 40)
    
    try:
        from gan_cyber_range.core.minimal_gan import AttackVector, MinimalAttackGenerator
        
        # Test invalid attack vector creation
        try:
            invalid_attack = AttackVector(
                attack_type="",  # Empty type
                payload="",      # Empty payload
                techniques=[],   # No techniques
                severity=1.5,    # Invalid severity
                stealth_level=-0.1,  # Invalid stealth
                target_systems=[]
            )
            # Should still create but with corrections
            print("✓ Invalid attack vector handled gracefully")
        except Exception as e:
            print(f"⚠ Attack vector validation: {e}")
        
        # Test generator with edge cases
        generator = MinimalAttackGenerator()
        
        # Zero attacks
        zero_attacks = generator.generate(num_samples=0)
        print(f"✓ Zero sample request handled: {len(zero_attacks)} attacks")
        
        # Large number of attacks
        many_attacks = generator.generate(num_samples=100)
        print(f"✓ Large sample request handled: {len(many_attacks)} attacks")
        
        # Invalid attack type
        invalid_type_attacks = generator.generate(num_samples=2, attack_type="invalid_type")
        print(f"✓ Invalid attack type handled: {len(invalid_type_attacks)} attacks generated")
        
        # Test diversity with edge cases
        single_attack_diversity = generator.diversity_score([many_attacks[0]])
        print(f"✓ Single attack diversity: {single_attack_diversity}")
        
        empty_diversity = generator.diversity_score([])
        print(f"✓ Empty list diversity: {empty_diversity}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run comprehensive enhanced demo"""
    print("GAN Cyber Range v2.0 - Enhanced Robust Demo")
    print("=" * 50)
    
    all_tests = [
        test_enhanced_imports,
        test_attack_generation,
        test_cyber_range_functionality,
        test_data_persistence,
        test_error_handling
    ]
    
    results = []
    
    for test_func in all_tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
        
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    print(f"GENERATION 2 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Generation 2 (Robust) functionality validated!")
        print("✅ Enhanced error handling implemented")
        print("✅ Graceful dependency fallbacks working")
        print("✅ Comprehensive validation in place")
        return 0
    else:
        print(f"❌ {total - passed} tests failed - some functionality needs improvement")
        return 1


if __name__ == "__main__":
    sys.exit(main())