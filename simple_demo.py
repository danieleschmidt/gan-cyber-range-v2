#!/usr/bin/env python3
"""
Simple demo to validate core functionality is working
"""

import sys
import os
sys.path.insert(0, '/root/repo')

def test_basic_imports():
    """Test that core modules can be imported"""
    try:
        from gan_cyber_range.core.attack_gan import AttackGAN, AttackVector
        from gan_cyber_range.core.cyber_range import CyberRange, RangeStatus
        print("✓ Core modules import successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic object instantiation"""
    try:
        from gan_cyber_range.core.attack_gan import AttackGAN, AttackVector
        
        # Create a simple AttackVector
        attack = AttackVector(
            attack_type="test",
            payload="test payload",
            techniques=["T1001"],
            severity=0.5,
            stealth_level=0.7,
            target_systems=["test_system"]
        )
        
        print("✓ AttackVector creation successful")
        print(f"  - Attack type: {attack.attack_type}")
        print(f"  - Payload: {attack.payload}")
        print(f"  - Techniques: {attack.techniques}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def main():
    """Run simple validation tests"""
    print("Running GAN Cyber Range v2.0 Basic Validation")
    print("=" * 50)
    
    success = True
    
    # Test imports
    success &= test_basic_imports()
    
    # Test basic functionality
    success &= test_basic_functionality()
    
    print("=" * 50)
    if success:
        print("✓ All basic tests passed - Generation 1 functionality validated!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())