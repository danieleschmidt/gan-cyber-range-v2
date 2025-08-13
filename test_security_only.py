#!/usr/bin/env python3
"""
Security-only tests for GAN-Cyber-Range-v2

This script tests only the security components without heavy dependencies.
"""

import sys
import unittest
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


class TestSecurityFramework(unittest.TestCase):
    """Test security framework components directly"""
    
    def test_enhanced_security_import(self):
        """Test direct import of security utilities"""
        
        try:
            from gan_cyber_range.utils.enhanced_security import (
                SecurityLevel,
                ThreatLevel,
                SecurityEventType,
                EthicalFramework,
                SecureInputValidator,
                ThreatDetectionEngine,
                ContainmentEngine,
                SecurityAuditLogger,
                SecureDataManager,
                validate_input,
                secure_hash
            )
            
            # Test enums
            self.assertEqual(SecurityLevel.PUBLIC.value, "public")
            self.assertEqual(ThreatLevel.HIGH.value, "high")
            self.assertEqual(SecurityEventType.SUSPICIOUS_INPUT.value, "suspicious_input")
            
            # Test basic instantiation
            framework = EthicalFramework()
            self.assertIsInstance(framework.allowed_uses, list)
            self.assertIn("education", framework.allowed_uses)
            
            validator = SecureInputValidator()
            self.assertIsNotNone(validator)
            
            detector = ThreatDetectionEngine()
            self.assertIsNotNone(detector)
            
            print("✅ Security framework components imported successfully")
            
        except ImportError as e:
            self.fail(f"Security framework import failed: {e}")
    
    def test_input_validation(self):
        """Test input validation functions"""
        
        from gan_cyber_range.utils.enhanced_security import validate_input
        
        # Test safe input
        safe_input = "legitimate_user_input"
        self.assertTrue(validate_input(safe_input, r"^[a-zA-Z_]+$"))
        
        # Test dangerous inputs
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "$(rm -rf /)",
            "javascript:alert(1)"
        ]
        
        for dangerous in dangerous_inputs:
            self.assertFalse(
                validate_input(dangerous, r"^[a-zA-Z0-9_\s]+$"),
                f"Dangerous input should be rejected: {dangerous}"
            )
            
        print("✅ Input validation working correctly")
    
    def test_secure_hashing(self):
        """Test secure hashing functions"""
        
        from gan_cyber_range.utils.enhanced_security import secure_hash
        
        # Test basic hashing
        hash1 = secure_hash("test_data")
        hash2 = secure_hash("test_data")
        
        self.assertIsInstance(hash1, str)
        self.assertIsInstance(hash2, str)
        self.assertGreater(len(hash1), 32)  # Should include salt
        self.assertNotEqual(hash1, hash2)  # Different salts
        
        print("✅ Secure hashing working correctly")
    
    def test_ethical_framework(self):
        """Test ethical framework compliance"""
        
        from gan_cyber_range.utils.enhanced_security import EthicalFramework
        
        framework = EthicalFramework()
        
        # Test compliant request
        compliant_request = {
            'purpose': 'security training',
            'targets': ['192.168.1.100', '10.0.0.50'],
            'consent_obtained': True,
            'approved': True
        }
        
        self.assertTrue(framework.is_compliant(compliant_request))
        
        # Test non-compliant request
        non_compliant_request = {
            'purpose': 'real attack',
            'targets': ['production_systems'],
            'consent_obtained': False,
            'approved': False
        }
        
        self.assertFalse(framework.is_compliant(non_compliant_request))
        
        print("✅ Ethical framework working correctly")
    
    def test_threat_detection(self):
        """Test threat detection engine"""
        
        from gan_cyber_range.utils.enhanced_security import ThreatDetectionEngine
        
        detector = ThreatDetectionEngine()
        
        # Test malicious payload detection
        malicious_payload = "meterpreter reverse_tcp payload"
        result = detector.analyze_payload(malicious_payload)
        
        self.assertIsInstance(result, dict)
        self.assertIn('threat_detected', result)
        self.assertIn('risk_score', result)
        
        # Test benign payload
        benign_payload = "legitimate user input"
        result2 = detector.analyze_payload(benign_payload)
        
        self.assertIsInstance(result2, dict)
        
        print("✅ Threat detection working correctly")
    
    def test_input_validator(self):
        """Test secure input validator"""
        
        from gan_cyber_range.utils.enhanced_security import SecureInputValidator
        
        validator = SecureInputValidator()
        
        # Test safe input
        result = validator.validate_input("safe_input", "username")
        self.assertTrue(result['is_valid'])
        
        # Test malicious input
        malicious_input = "<script>alert('xss')</script>"
        result = validator.validate_input(malicious_input, "general")
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['threats_detected']), 0)
        
        print("✅ Input validator working correctly")
    
    def test_data_encryption(self):
        """Test secure data manager"""
        
        from gan_cyber_range.utils.enhanced_security import SecureDataManager
        
        manager = SecureDataManager()
        
        # Test string encryption
        test_data = "sensitive information"
        encrypted = manager.encrypt_sensitive_data(test_data)
        decrypted = manager.decrypt_sensitive_data(encrypted)
        
        self.assertEqual(test_data, decrypted)
        self.assertNotEqual(test_data, encrypted)
        
        # Test dict encryption
        test_dict = {"username": "admin", "password": "secret"}
        encrypted_dict = manager.encrypt_sensitive_data(test_dict)
        decrypted_dict = manager.decrypt_sensitive_data(encrypted_dict)
        
        self.assertEqual(test_dict, decrypted_dict)
        
        print("✅ Data encryption working correctly")


def run_security_tests():
    """Run security-focused test suite"""
    
    print("🔒 Running GAN-Cyber-Range-v2 Security Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityFramework))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🔍 SECURITY TEST SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    if not result.failures and not result.errors:
        print("\n✅ ALL SECURITY TESTS PASSED!")
        print("🛡️  Security framework is functioning correctly")
        return True
    else:
        print(f"\n⚠️  {len(result.failures + result.errors)} SECURITY TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)