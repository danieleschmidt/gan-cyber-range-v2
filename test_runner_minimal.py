#!/usr/bin/env python3
"""
Minimal test runner that works without external dependencies.

Tests core framework functionality without numpy, scipy, etc.
"""

import sys
import traceback
from pathlib import Path
import tempfile
import json

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_security_orchestrator_minimal():
    """Test basic security orchestrator without external deps"""
    print("Testing security orchestrator (minimal)...")
    
    try:
        # Import only the core security module
        from gan_cyber_range.security.security_orchestrator import (
            SecurityOrchestrator, SecurityPolicy, SecurityLevel, ThreatLevel,
            SecurityContext, SecurityEvent
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "security_config.json"
            orchestrator = SecurityOrchestrator(config_path=config_path)
            
            # Test basic initialization
            assert orchestrator.config_path == config_path
            assert len(orchestrator.active_policies) >= 0
            assert len(orchestrator.security_contexts) == 0
            
            # Test policy creation and registration
            policy = SecurityPolicy(
                name="test_policy",
                description="Test policy",
                security_level=SecurityLevel.MEDIUM,
                session_timeout=3600
            )
            
            orchestrator.register_security_policy(policy)
            assert policy.name in orchestrator.active_policies
            
            # Test context creation
            context = orchestrator.create_security_context(
                user_id="test_user",
                ip_address="127.0.0.1",
                user_agent="Test Agent",
                requested_permissions={"read"}
            )
            
            assert context is not None
            assert context.user_id == "test_user"
            assert "read" in context.permissions
            
            # Test encryption/decryption
            test_data = "sensitive information"
            encrypted = orchestrator.encrypt_sensitive_data(test_data)
            decrypted = orchestrator.decrypt_sensitive_data(encrypted)
            assert decrypted == test_data
            
            # Test operation validation
            result = orchestrator.validate_operation(
                session_id=context.session_id,
                operation="read",
                target="test_resource"
            )
            assert result == True
            
            # Test blocked operation
            policy.blocked_operations.add("dangerous_operation")
            result = orchestrator.validate_operation(
                session_id=context.session_id,
                operation="dangerous_operation",
                target="test_resource"
            )
            assert result == False
            
            # Test security report generation
            report = orchestrator.generate_security_report()
            assert "summary" in report
            assert "event_breakdown" in report
            
        print("✓ Security orchestrator minimal functionality works")
        return True
    except Exception as e:
        print(f"✗ Security orchestrator test failed: {e}")
        traceback.print_exc()
        return False

def test_core_structure():
    """Test that core project structure is valid"""
    print("Testing core project structure...")
    
    try:
        # Check that key directories exist
        project_root = Path(__file__).parent
        
        required_dirs = [
            "gan_cyber_range",
            "gan_cyber_range/core",
            "gan_cyber_range/research", 
            "gan_cyber_range/security",
            "gan_cyber_range/monitoring",
            "gan_cyber_range/scalability",
            "gan_cyber_range/internationalization",
            "tests"
        ]
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Missing directory: {dir_path}"
            assert (full_path / "__init__.py").exists(), f"Missing __init__.py in: {dir_path}"
        
        # Check key files exist
        key_files = [
            "README.md",
            "requirements.txt",
            "setup.py",
            "gan_cyber_range/__init__.py"
        ]
        
        for file_path in key_files:
            full_path = project_root / file_path
            assert full_path.exists(), f"Missing file: {file_path}"
        
        print("✓ Core project structure is valid")
        return True
    except Exception as e:
        print(f"✗ Project structure test failed: {e}")
        return False

def test_configuration_loading():
    """Test configuration file handling"""
    print("Testing configuration loading...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"
            
            # Test config creation
            test_config = {
                "name": "test_config",
                "version": "1.0.0",
                "settings": {
                    "debug": True,
                    "timeout": 30
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(test_config, f, indent=2)
            
            # Test config loading
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
            
            assert loaded_config["name"] == "test_config"
            assert loaded_config["settings"]["debug"] == True
            assert loaded_config["settings"]["timeout"] == 30
            
        print("✓ Configuration loading works")
        return True
    except Exception as e:
        print(f"✗ Configuration loading test failed: {e}")
        return False

def test_enum_definitions():
    """Test that enum definitions work correctly"""
    print("Testing enum definitions...")
    
    try:
        from gan_cyber_range.security.security_orchestrator import SecurityLevel, ThreatLevel
        
        # Test SecurityLevel enum
        assert SecurityLevel.LOW.value == "low"
        assert SecurityLevel.MEDIUM.value == "medium"
        assert SecurityLevel.HIGH.value == "high"
        assert SecurityLevel.CRITICAL.value == "critical"
        
        # Test ThreatLevel enum
        assert ThreatLevel.INFO.value == "info"
        assert ThreatLevel.LOW.value == "low"
        assert ThreatLevel.MEDIUM.value == "medium"
        assert ThreatLevel.HIGH.value == "high"
        assert ThreatLevel.CRITICAL.value == "critical"
        
        # Test enum comparison
        assert SecurityLevel.HIGH != SecurityLevel.LOW
        assert ThreatLevel.CRITICAL != ThreatLevel.HIGH  # Enum comparison
        
        print("✓ Enum definitions work correctly")
        return True
    except Exception as e:
        print(f"✗ Enum definitions test failed: {e}")
        return False

def test_dataclass_functionality():
    """Test dataclass functionality"""
    print("Testing dataclass functionality...")
    
    try:
        from gan_cyber_range.security.security_orchestrator import SecurityPolicy, SecurityLevel
        from datetime import datetime
        
        # Test dataclass creation
        policy = SecurityPolicy(
            name="test_policy",
            description="Test description",
            security_level=SecurityLevel.HIGH,
            session_timeout=3600
        )
        
        assert policy.name == "test_policy"
        assert policy.security_level == SecurityLevel.HIGH
        assert policy.session_timeout == 3600
        assert policy.encryption_required == True  # Default value
        
        # Test field modification
        policy.session_timeout = 7200
        assert policy.session_timeout == 7200
        
        # Test default factory fields
        assert isinstance(policy.allowed_operations, set)
        assert isinstance(policy.compliance_frameworks, list)
        
        print("✓ Dataclass functionality works")
        return True
    except Exception as e:
        print(f"✗ Dataclass functionality test failed: {e}")
        return False

def test_basic_imports_selective():
    """Test selective imports that don't require external dependencies"""
    print("Testing selective imports...")
    
    try:
        # Test security module imports
        from gan_cyber_range.security.security_orchestrator import SecurityOrchestrator
        assert SecurityOrchestrator is not None
        
        # Test that we can create instances
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            orchestrator = SecurityOrchestrator(config_path=config_path)
            assert orchestrator is not None
        
        print("✓ Selective imports work")
        return True
    except Exception as e:
        print(f"✗ Selective imports test failed: {e}")
        return False

def main():
    """Run minimal test suite"""
    print("=" * 60)
    print("Running Minimal Functionality Tests")
    print("(Without external dependencies)")
    print("=" * 60)
    
    tests = [
        test_core_structure,
        test_basic_imports_selective,
        test_enum_definitions,
        test_dataclass_functionality,
        test_configuration_loading,
        test_security_orchestrator_minimal
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
            print(f"✗ Test {test.__name__} raised exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All minimal tests passed!")
        print("Note: Full functionality requires installing dependencies from requirements.txt")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())