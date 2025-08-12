#!/usr/bin/env python3
"""
Basic functionality tests for GAN-Cyber-Range-v2

This script tests core functionality without requiring heavy dependencies
to validate the implementation works correctly.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Mock heavy dependencies before imports
sys.modules['torch'] = Mock()
sys.modules['torch.nn'] = Mock()
sys.modules['torch.optim'] = Mock()
sys.modules['torch.utils'] = Mock()
sys.modules['torch.utils.data'] = Mock()
sys.modules['transformers'] = Mock()
sys.modules['docker'] = Mock()
sys.modules['paramiko'] = Mock()
sys.modules['scapy'] = Mock()
sys.modules['psutil'] = Mock()
sys.modules['cryptography'] = Mock()
sys.modules['cryptography.fernet'] = Mock()
sys.modules['cryptography.hazmat'] = Mock()
sys.modules['cryptography.hazmat.primitives'] = Mock()
sys.modules['cryptography.hazmat.primitives.kdf'] = Mock()
sys.modules['cryptography.hazmat.primitives.kdf.pbkdf2'] = Mock()


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality without heavy dependencies"""
    
    def test_package_structure(self):
        """Test that package structure is correct"""
        
        # Check main package directories exist
        package_dirs = [
            'gan_cyber_range',
            'gan_cyber_range/core',
            'gan_cyber_range/generators', 
            'gan_cyber_range/red_team',
            'gan_cyber_range/blue_team',
            'gan_cyber_range/orchestration',
            'gan_cyber_range/optimization',
            'gan_cyber_range/factories',
            'gan_cyber_range/utils',
            'gan_cyber_range/cli'
        ]
        
        for dir_path in package_dirs:
            self.assertTrue(
                Path(dir_path).exists(),
                f"Package directory {dir_path} should exist"
            )
            self.assertTrue(
                Path(dir_path, '__init__.py').exists(),
                f"Package {dir_path} should have __init__.py"
            )
    
    def test_config_files_exist(self):
        """Test that configuration files exist"""
        
        config_files = [
            'requirements.txt',
            'setup.py',
            'README.md',
            'LICENSE',
            'pytest.ini'
        ]
        
        for config_file in config_files:
            self.assertTrue(
                Path(config_file).exists(),
                f"Configuration file {config_file} should exist"
            )
    
    def test_security_framework_import(self):
        """Test security framework can be imported"""
        
        try:
            from gan_cyber_range.utils.enhanced_security import (
                EthicalFramework, 
                Containment,
                SecurityLevel
            )
            
            # Test basic functionality
            framework = EthicalFramework()
            self.assertIsInstance(framework.allowed_uses, list)
            self.assertIn("education", framework.allowed_uses)
            
            containment = Containment()
            self.assertEqual(containment.network_isolation, "strict")
            
            # Test enum
            self.assertEqual(SecurityLevel.PUBLIC.value, "public")
            
        except ImportError as e:
            self.fail(f"Security framework import failed: {e}")
    
    def test_network_topology_creation(self):
        """Test network topology creation without Docker"""
        
        try:
            from gan_cyber_range.core.network_sim import (
                NetworkTopology, 
                Host, 
                Subnet,
                HostType,
                OSType
            )
            
            # Create basic topology
            topology = NetworkTopology("test-topology")
            
            # Add subnet
            subnet = topology.add_subnet(
                name="test-subnet",
                cidr="192.168.1.0/24",
                security_zone="internal"
            )
            
            self.assertEqual(subnet.name, "test-subnet")
            self.assertEqual(subnet.cidr, "192.168.1.0/24")
            self.assertEqual(len(topology.subnets), 1)
            
            # Add host
            host = topology.add_host(
                name="test-host",
                subnet_name="test-subnet",
                host_type=HostType.WORKSTATION,
                os_type=OSType.LINUX,
                services=["ssh", "web"]
            )
            
            self.assertEqual(host.name, "test-host")
            self.assertEqual(host.subnet, "test-subnet")
            self.assertEqual(len(host.services), 2)
            self.assertEqual(len(topology.hosts), 1)
            
        except Exception as e:
            self.fail(f"Network topology creation failed: {e}")
    
    def test_attack_patterns_basic(self):
        """Test basic attack pattern structures"""
        
        try:
            from gan_cyber_range.core.attack_gan import AttackVector
            from gan_cyber_range.core.attack_engine import AttackStep, AttackPhase
            
            # Create attack vector
            attack_vector = AttackVector(
                attack_type="test",
                payload="test payload",
                techniques=["T1059"],
                severity=0.8,
                stealth_level=0.6,
                target_systems=["test-system"]
            )
            
            self.assertEqual(attack_vector.attack_type, "test")
            self.assertEqual(len(attack_vector.techniques), 1)
            
            # Create attack step
            attack_step = AttackStep(
                step_id="step-1",
                name="Test Step",
                phase=AttackPhase.EXPLOITATION,
                technique_id="T1059",
                target_host="test-host",
                payload={"command": "whoami"}
            )
            
            self.assertEqual(attack_step.name, "Test Step")
            self.assertEqual(attack_step.phase, AttackPhase.EXPLOITATION)
            
        except Exception as e:
            self.fail(f"Attack pattern creation failed: {e}")
    
    def test_blue_team_structures(self):
        """Test blue team defensive structures"""
        
        try:
            from gan_cyber_range.blue_team.defense_suite import (
                SecurityAlert,
                AlertSeverity,
                DefenseMetrics
            )
            from gan_cyber_range.blue_team.incident_response import (
                Incident,
                IncidentSeverity,
                IncidentStatus
            )
            
            # Create security alert
            alert = SecurityAlert(
                alert_id="alert-1",
                timestamp=__import__('datetime').datetime.now(),
                severity=AlertSeverity.HIGH,
                source="test-source",
                rule_name="Test Rule",
                description="Test alert"
            )
            
            self.assertEqual(alert.severity, AlertSeverity.HIGH)
            self.assertEqual(alert.source, "test-source")
            
            # Test metrics
            metrics = DefenseMetrics()
            self.assertEqual(metrics.detection_rate, 0.0)
            
            # Create incident
            incident = Incident(
                incident_id="inc-1",
                title="Test Incident",
                description="Test incident description",
                severity=IncidentSeverity.HIGH,
                status=IncidentStatus.NEW,
                created_time=__import__('datetime').datetime.now(),
                updated_time=__import__('datetime').datetime.now()
            )
            
            self.assertEqual(incident.title, "Test Incident")
            self.assertEqual(incident.status, IncidentStatus.NEW)
            
        except Exception as e:
            self.fail(f"Blue team structures failed: {e}")
    
    def test_configuration_validation(self):
        """Test configuration validation without external dependencies"""
        
        try:
            from gan_cyber_range.utils.validation import validate_config
            from gan_cyber_range.utils.enhanced_security import (
                validate_input,
                secure_hash
            )
            
            # Test input validation
            valid_result = validate_input("test_input", r"^[a-zA-Z_]+$")
            self.assertTrue(valid_result)
            
            invalid_result = validate_input("<script>alert('xss')</script>", r"^[a-zA-Z_]+$")
            self.assertFalse(invalid_result)
            
            # Test secure hashing
            hash_result = secure_hash("test_data")
            self.assertIsInstance(hash_result, str)
            self.assertGreater(len(hash_result), 32)  # Should include salt
            
        except Exception as e:
            self.fail(f"Configuration validation failed: {e}")
    
    def test_factory_patterns(self):
        """Test factory patterns work correctly"""
        
        try:
            from gan_cyber_range.factories.attack_factory import AttackFactory, AttackConfig
            from gan_cyber_range.factories.network_factory import NetworkFactory
            
            # Test attack factory
            attack_config = AttackConfig(
                attack_type="web_exploit",
                target_systems=["web-server"],
                intensity="medium",
                duration=300
            )
            
            self.assertEqual(attack_config.attack_type, "web_exploit")
            self.assertEqual(attack_config.intensity, "medium")
            
            # Test attack factory creation
            factory = AttackFactory()
            self.assertIsNotNone(factory)
            
            # Test network factory
            network_factory = NetworkFactory()
            self.assertIsNotNone(network_factory)
            
        except Exception as e:
            self.fail(f"Factory pattern test failed: {e}")
    
    def test_orchestration_components(self):
        """Test orchestration components"""
        
        try:
            from gan_cyber_range.orchestration.workflow_engine import (
                WorkflowEngine,
                Workflow,
                WorkflowStep
            )
            from gan_cyber_range.orchestration.scenario_orchestrator import (
                ScenarioOrchestrator,
                TrainingScenario
            )
            
            # Test workflow step
            step = WorkflowStep(
                step_id="step-1",
                name="Test Step",
                step_type="action",
                configuration={"param": "value"}
            )
            
            self.assertEqual(step.name, "Test Step")
            self.assertEqual(step.step_type, "action")
            
            # Test workflow
            workflow = Workflow(
                workflow_id="wf-1",
                name="Test Workflow",
                steps=[step]
            )
            
            self.assertEqual(workflow.name, "Test Workflow")
            self.assertEqual(len(workflow.steps), 1)
            
            # Test workflow engine
            engine = WorkflowEngine()
            self.assertIsNotNone(engine)
            
        except Exception as e:
            self.fail(f"Orchestration components test failed: {e}")
    
    def test_performance_optimization(self):
        """Test performance optimization components"""
        
        try:
            from gan_cyber_range.optimization.cache_optimizer import (
                CacheOptimizer,
                CacheStrategy
            )
            from gan_cyber_range.optimization.performance_monitor import (
                PerformanceMonitor,
                PerformanceProfiler
            )
            
            # Test cache strategy
            strategy = CacheStrategy(
                strategy_name="lru",
                max_size=1000,
                ttl_seconds=3600
            )
            
            self.assertEqual(strategy.strategy_name, "lru")
            self.assertEqual(strategy.max_size, 1000)
            
            # Test performance monitor
            monitor = PerformanceMonitor()
            self.assertIsNotNone(monitor)
            
        except Exception as e:
            self.fail(f"Performance optimization test failed: {e}")


class TestSecurityCompliance(unittest.TestCase):
    """Test security compliance and ethical framework"""
    
    def test_ethical_framework_compliance(self):
        """Test ethical framework compliance checks"""
        
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
        
        # Test non-compliant request (production target)
        non_compliant_request = {
            'purpose': 'real attack',
            'targets': ['production_server'],
            'consent_obtained': False,
            'approved': False
        }
        
        self.assertFalse(framework.is_compliant(non_compliant_request))
    
    def test_containment_mechanisms(self):
        """Test security containment mechanisms"""
        
        from gan_cyber_range.utils.enhanced_security import Containment
        
        containment = Containment()
        
        # Test decorator functionality
        @containment.contained
        def test_function():
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")
    
    def test_input_sanitization(self):
        """Test input sanitization and validation"""
        
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


def run_comprehensive_tests():
    """Run comprehensive test suite"""
    
    print("🧪 Running GAN-Cyber-Range-v2 Comprehensive Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBasicFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityCompliance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🔍 TEST SUMMARY")
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
        print("\n✅ ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️  {len(result.failures + result.errors)} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)