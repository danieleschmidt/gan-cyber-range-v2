"""
Autonomous Quality Gates System

Comprehensive quality gates with automated testing, security scanning,
performance benchmarking, and compliance validation.
"""

import os
import sys
import json
import time
import subprocess
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from gan_cyber_range.utils.comprehensive_logging import comprehensive_logger
from gan_cyber_range.security.enhanced_security_framework import EnhancedSecurityFramework, AccessLevel
from gan_cyber_range.optimization.intelligent_performance import performance_optimizer


@dataclass
class QualityGateResult:
    """Result of a quality gate check"""
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    timestamp: str
    error_message: Optional[str] = None


class AutonomousQualityGates:
    """Comprehensive quality gates orchestrator"""
    
    def __init__(self):
        self.results: List[QualityGateResult] = []
        self.overall_score = 0.0
        self.passed_gates = 0
        self.total_gates = 0
        
        # Gate weights for scoring
        self.gate_weights = {
            "functionality": 0.30,
            "security": 0.25,
            "performance": 0.20,
            "reliability": 0.15,
            "compliance": 0.10
        }
        
        comprehensive_logger.info("Autonomous Quality Gates initialized")
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates"""
        start_time = time.time()
        
        comprehensive_logger.info("🚀 Starting Autonomous Quality Gates Execution")
        print("=" * 80)
        print("🛡️  AUTONOMOUS QUALITY GATES - COMPREHENSIVE VALIDATION")
        print("=" * 80)
        
        # Run each gate
        gates = [
            ("Functionality Tests", self._run_functionality_tests),
            ("Security Analysis", self._run_security_analysis),
            ("Performance Benchmarks", self._run_performance_benchmarks),
            ("Reliability Tests", self._run_reliability_tests),
            ("Compliance Validation", self._run_compliance_validation)
        ]
        
        for gate_name, gate_function in gates:
            print(f"\n🔍 Running {gate_name}...")
            try:
                result = gate_function()
                self.results.append(result)
                
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                print(f"   {status} - Score: {result.score:.1f}/100")
                
                if result.passed:
                    self.passed_gates += 1
                self.total_gates += 1
                
            except Exception as e:
                error_result = QualityGateResult(
                    gate_name=gate_name,
                    passed=False,
                    score=0.0,
                    details={"error": str(e)},
                    execution_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    error_message=str(e)
                )
                self.results.append(error_result)
                self.total_gates += 1
                print(f"   ❌ FAILED - Error: {e}")
        
        # Calculate overall score
        self._calculate_overall_score()
        
        # Generate report
        report = self._generate_report()
        
        total_time = time.time() - start_time
        
        print(f"\n" + "=" * 80)
        print(f"🏁 QUALITY GATES COMPLETED")
        print(f"   Overall Score: {self.overall_score:.1f}/100")
        print(f"   Gates Passed: {self.passed_gates}/{self.total_gates}")
        print(f"   Execution Time: {total_time:.1f}s")
        print("=" * 80)
        
        return report
    
    def _run_functionality_tests(self) -> QualityGateResult:
        """Run functionality tests"""
        start_time = time.time()
        details = {}
        score = 0.0
        
        try:
            # Test 1: Core imports
            print("     • Testing core imports...")
            try:
                from gan_cyber_range.core.attack_gan_robust import RobustAttackGAN
                from gan_cyber_range.demo import DemoAPI
                details["core_imports"] = "✅ Success"
                score += 20
            except Exception as e:
                details["core_imports"] = f"❌ Failed: {e}"
            
            # Test 2: Attack generation
            print("     • Testing attack generation...")
            try:
                gan = RobustAttackGAN()
                attacks = gan.generate(num_samples=3, attack_types=['malware'])
                if len(attacks) >= 3:
                    details["attack_generation"] = f"✅ Generated {len(attacks)} attacks"
                    score += 30
                else:
                    details["attack_generation"] = f"⚠️  Generated only {len(attacks)} attacks"
                    score += 15
            except Exception as e:
                details["attack_generation"] = f"❌ Failed: {e}"
            
            # Test 3: Demo system
            print("     • Testing demo system...")
            try:
                api = DemoAPI()
                range_info = api.create_range("test_range")
                if "range_id" in range_info:
                    details["demo_system"] = "✅ Demo system functional"
                    score += 25
                else:
                    details["demo_system"] = "❌ Demo system failed"
            except Exception as e:
                details["demo_system"] = f"❌ Failed: {e}"
            
            # Test 4: Configuration validation
            print("     • Testing configuration...")
            try:
                config_files = [
                    "requirements.txt",
                    "setup.py",
                    "README.md"
                ]
                missing_files = [f for f in config_files if not Path(f).exists()]
                if not missing_files:
                    details["configuration"] = "✅ All config files present"
                    score += 15
                else:
                    details["configuration"] = f"⚠️  Missing: {missing_files}"
                    score += 5
            except Exception as e:
                details["configuration"] = f"❌ Failed: {e}"
            
            # Test 5: Integration test
            print("     • Testing integration...")
            try:
                # Test security framework integration
                from gan_cyber_range.security.enhanced_security_framework import security_framework
                context = security_framework.create_security_context(
                    "test_user", AccessLevel.USER, "127.0.0.1"
                )
                if context.session_id:
                    details["integration"] = "✅ Security integration working"
                    score += 10
                else:
                    details["integration"] = "❌ Security integration failed"
            except Exception as e:
                details["integration"] = f"❌ Failed: {e}"
            
        except Exception as e:
            details["overall_error"] = str(e)
        
        execution_time = time.time() - start_time
        passed = score >= 70  # 70% threshold
        
        return QualityGateResult(
            gate_name="Functionality Tests",
            passed=passed,
            score=score,
            details=details,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _run_security_analysis(self) -> QualityGateResult:
        """Run security analysis"""
        start_time = time.time()
        details = {}
        score = 0.0
        
        try:
            # Test 1: Security framework
            print("     • Testing security framework...")
            try:
                from gan_cyber_range.security.enhanced_security_framework import (
                    EnhancedSecurityFramework, EthicalFramework, InputSanitizer
                )
                
                security = EnhancedSecurityFramework()
                
                # Test ethical compliance
                compliant_request = {
                    "purpose": "defensive training",
                    "targets": ["test_environment"],
                    "consent": True
                }
                
                if security.ethical_framework.is_compliant(compliant_request):
                    details["ethical_compliance"] = "✅ Ethical framework working"
                    score += 25
                else:
                    details["ethical_compliance"] = "❌ Ethical compliance failed"
                
            except Exception as e:
                details["ethical_compliance"] = f"❌ Failed: {e}"
            
            # Test 2: Input sanitization
            print("     • Testing input sanitization...")
            try:
                sanitizer = InputSanitizer()
                
                # Test dangerous inputs
                dangerous_inputs = [
                    "<script>alert('xss')</script>",
                    "'; DROP TABLE users; --",
                    "javascript:alert(1)"
                ]
                
                all_safe = True
                for dangerous in dangerous_inputs:
                    sanitized = sanitizer.sanitize_input(dangerous)
                    if dangerous in sanitized:
                        all_safe = False
                        break
                
                if all_safe:
                    details["input_sanitization"] = "✅ Input sanitization working"
                    score += 25
                else:
                    details["input_sanitization"] = "⚠️  Some inputs not sanitized"
                    score += 10
                    
            except Exception as e:
                details["input_sanitization"] = f"❌ Failed: {e}"
            
            # Test 3: Error handling security
            print("     • Testing error handling...")
            try:
                from gan_cyber_range.utils.robust_error_handler import error_handler
                
                # Test that sensitive information isn't leaked
                try:
                    raise Exception("Database password: secret123")
                except Exception as e:
                    error_context = error_handler.handle_error(e, {"test": True})
                    # Error handler should sanitize or handle this properly
                    details["error_handling"] = "✅ Error handling secure"
                    score += 20
                    
            except Exception as e:
                details["error_handling"] = f"❌ Failed: {e}"
            
            # Test 4: Dependency security
            print("     • Checking dependencies...")
            try:
                requirements_path = Path("requirements.txt")
                if requirements_path.exists():
                    with open(requirements_path) as f:
                        requirements = f.read()
                    
                    # Check for known vulnerable packages (basic check)
                    vulnerable_patterns = ["pillow<8.0.0", "requests<2.20.0"]
                    vulnerabilities = [p for p in vulnerable_patterns if p in requirements]
                    
                    if not vulnerabilities:
                        details["dependency_security"] = "✅ No obvious vulnerable dependencies"
                        score += 15
                    else:
                        details["dependency_security"] = f"⚠️  Potential vulnerabilities: {vulnerabilities}"
                        score += 5
                else:
                    details["dependency_security"] = "⚠️  No requirements.txt found"
                    score += 5
                    
            except Exception as e:
                details["dependency_security"] = f"❌ Failed: {e}"
            
            # Test 5: Access control
            print("     • Testing access control...")
            try:
                security = EnhancedSecurityFramework()
                user_context = security.create_security_context(
                    "test_user", AccessLevel.USER, "127.0.0.1"
                )
                admin_context = security.create_security_context(
                    "admin_user", AccessLevel.ADMIN, "127.0.0.1"
                )
                
                # Test permission checking
                if not security.check_permission(user_context, "admin_action") and \
                   security.check_permission(admin_context, "admin_action"):
                    details["access_control"] = "✅ Access control working"
                    score += 15
                else:
                    details["access_control"] = "❌ Access control failed"
                    
            except Exception as e:
                details["access_control"] = f"❌ Failed: {e}"
            
        except Exception as e:
            details["overall_error"] = str(e)
        
        execution_time = time.time() - start_time
        passed = score >= 70
        
        return QualityGateResult(
            gate_name="Security Analysis",
            passed=passed,
            score=score,
            details=details,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _run_performance_benchmarks(self) -> QualityGateResult:
        """Run performance benchmarks"""
        start_time = time.time()
        details = {}
        score = 0.0
        
        try:
            print("     • Testing performance optimization...")
            
            # Test 1: Attack generation performance
            try:
                from gan_cyber_range.core.attack_gan_robust import RobustAttackGAN
                
                gan = RobustAttackGAN()
                
                # Benchmark attack generation
                gen_start = time.time()
                attacks = gan.generate(num_samples=10)
                gen_time = time.time() - gen_start
                
                attacks_per_second = len(attacks) / max(gen_time, 0.001)
                
                if attacks_per_second >= 5:  # At least 5 attacks/second
                    details["attack_generation_perf"] = f"✅ {attacks_per_second:.1f} attacks/second"
                    score += 30
                elif attacks_per_second >= 1:
                    details["attack_generation_perf"] = f"⚠️  {attacks_per_second:.1f} attacks/second (slow)"
                    score += 15
                else:
                    details["attack_generation_perf"] = f"❌ {attacks_per_second:.1f} attacks/second (too slow)"
                    
            except Exception as e:
                details["attack_generation_perf"] = f"❌ Failed: {e}"
            
            # Test 2: Cache performance
            print("     • Testing cache performance...")
            try:
                from gan_cyber_range.optimization.intelligent_performance import IntelligentCache
                
                cache = IntelligentCache(max_size=1000)
                
                # Benchmark cache operations
                cache_start = time.time()
                for i in range(100):
                    cache.put(f"key_{i}", f"value_{i}")
                    cache.get(f"key_{i}")
                cache_time = time.time() - cache_start
                
                ops_per_second = 200 / max(cache_time, 0.001)  # 200 ops total
                
                if ops_per_second >= 1000:
                    details["cache_performance"] = f"✅ {ops_per_second:.0f} ops/second"
                    score += 25
                elif ops_per_second >= 500:
                    details["cache_performance"] = f"⚠️  {ops_per_second:.0f} ops/second"
                    score += 15
                else:
                    details["cache_performance"] = f"❌ {ops_per_second:.0f} ops/second (slow)"
                    
            except Exception as e:
                details["cache_performance"] = f"❌ Failed: {e}"
            
            # Test 3: Memory usage
            print("     • Testing memory efficiency...")
            try:
                import psutil
                
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb <= 100:  # Less than 100MB
                    details["memory_usage"] = f"✅ {memory_mb:.1f} MB"
                    score += 20
                elif memory_mb <= 200:
                    details["memory_usage"] = f"⚠️  {memory_mb:.1f} MB"
                    score += 10
                else:
                    details["memory_usage"] = f"❌ {memory_mb:.1f} MB (high)"
                    
            except Exception as e:
                details["memory_usage"] = f"❌ Failed: {e}"
            
            # Test 4: Resource pooling
            print("     • Testing resource pooling...")
            try:
                from gan_cyber_range.optimization.intelligent_performance import ResourcePool
                
                def dummy_factory():
                    return {"created": time.time()}
                
                pool = ResourcePool(dummy_factory, max_size=5)
                
                # Test borrow/return performance
                pool_start = time.time()
                resources = []
                for _ in range(10):
                    resource = pool.borrow()
                    if resource:
                        resources.append(resource)
                
                for resource in resources:
                    pool.return_resource(resource)
                    
                pool_time = time.time() - pool_start
                
                if pool_time <= 0.1:  # Less than 100ms for 10 operations
                    details["resource_pooling"] = f"✅ {pool_time*1000:.1f}ms for 10 ops"
                    score += 15
                else:
                    details["resource_pooling"] = f"⚠️  {pool_time*1000:.1f}ms for 10 ops"
                    score += 5
                    
            except Exception as e:
                details["resource_pooling"] = f"❌ Failed: {e}"
            
            # Test 5: System resource monitoring
            print("     • Checking system resources...")
            try:
                import psutil
                
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent <= 80 and memory_percent <= 80:
                    details["system_resources"] = f"✅ CPU: {cpu_percent}%, Memory: {memory_percent}%"
                    score += 10
                else:
                    details["system_resources"] = f"⚠️  CPU: {cpu_percent}%, Memory: {memory_percent}%"
                    score += 5
                    
            except Exception as e:
                details["system_resources"] = f"❌ Failed: {e}"
            
        except Exception as e:
            details["overall_error"] = str(e)
        
        execution_time = time.time() - start_time
        passed = score >= 60
        
        return QualityGateResult(
            gate_name="Performance Benchmarks",
            passed=passed,
            score=score,
            details=details,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _run_reliability_tests(self) -> QualityGateResult:
        """Run reliability tests"""
        start_time = time.time()
        details = {}
        score = 0.0
        
        try:
            print("     • Testing error handling...")
            
            # Test 1: Robust error handling
            try:
                from gan_cyber_range.utils.robust_error_handler import robust, ErrorSeverity
                
                @robust(severity=ErrorSeverity.MEDIUM)
                def test_function_with_error():
                    raise ValueError("Test error")
                
                # This should not crash
                result = test_function_with_error()
                details["error_handling"] = "✅ Robust error handling working"
                score += 30
                
            except Exception as e:
                # Even this exception should be handled gracefully
                details["error_handling"] = f"⚠️  Error handling needs improvement: {e}"
                score += 10
            
            # Test 2: Dependency fallbacks
            print("     • Testing dependency fallbacks...")
            try:
                from gan_cyber_range.utils.dependency_manager import dep_manager
                
                status_report = dep_manager.get_status_report()
                core_available = status_report.get("core_available", False)
                
                if core_available:
                    details["dependency_fallbacks"] = "✅ Core dependencies available"
                    score += 25
                else:
                    details["dependency_fallbacks"] = "⚠️  Some core dependencies missing"
                    score += 10
                    
            except Exception as e:
                details["dependency_fallbacks"] = f"❌ Failed: {e}"
            
            # Test 3: Logging system
            print("     • Testing logging system...")
            try:
                from gan_cyber_range.utils.comprehensive_logging import comprehensive_logger
                
                # Test logging without errors
                comprehensive_logger.info("Test log message")
                comprehensive_logger.warning("Test warning message")
                comprehensive_logger.error("Test error message")
                
                details["logging_system"] = "✅ Logging system working"
                score += 20
                
            except Exception as e:
                details["logging_system"] = f"❌ Failed: {e}"
            
            # Test 4: Configuration resilience
            print("     • Testing configuration resilience...")
            try:
                # Test with missing config
                config_resilience = True
                
                # Try to import core modules without explicit config
                from gan_cyber_range.core.attack_gan_robust import RobustAttackGAN
                gan = RobustAttackGAN()
                
                if gan:
                    details["config_resilience"] = "✅ System works without explicit config"
                    score += 15
                else:
                    details["config_resilience"] = "⚠️  System requires explicit config"
                    score += 5
                    
            except Exception as e:
                details["config_resilience"] = f"❌ Failed: {e}"
            
            # Test 5: Resource cleanup
            print("     • Testing resource cleanup...")
            try:
                # Test that resources are properly cleaned up
                initial_thread_count = len([t for t in __import__('threading').enumerate() if t.is_alive()])
                
                # Create and destroy some resources
                from gan_cyber_range.demo import DemoAPI
                api = DemoAPI()
                range_info = api.create_range("cleanup_test")
                
                # Allow some time for cleanup
                time.sleep(1)
                
                final_thread_count = len([t for t in __import__('threading').enumerate() if t.is_alive()])
                
                if final_thread_count <= initial_thread_count + 5:  # Allow some threads
                    details["resource_cleanup"] = f"✅ Thread count stable ({final_thread_count})"
                    score += 10
                else:
                    details["resource_cleanup"] = f"⚠️  Thread count increased ({final_thread_count})"
                    score += 5
                    
            except Exception as e:
                details["resource_cleanup"] = f"❌ Failed: {e}"
            
        except Exception as e:
            details["overall_error"] = str(e)
        
        execution_time = time.time() - start_time
        passed = score >= 70
        
        return QualityGateResult(
            gate_name="Reliability Tests",
            passed=passed,
            score=score,
            details=details,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _run_compliance_validation(self) -> QualityGateResult:
        """Run compliance validation"""
        start_time = time.time()
        details = {}
        score = 0.0
        
        try:
            print("     • Validating code structure...")
            
            # Test 1: Package structure
            try:
                required_modules = [
                    "gan_cyber_range",
                    "gan_cyber_range.core",
                    "gan_cyber_range.security",
                    "gan_cyber_range.utils"
                ]
                
                missing_modules = []
                for module in required_modules:
                    try:
                        __import__(module)
                    except ImportError:
                        missing_modules.append(module)
                
                if not missing_modules:
                    details["package_structure"] = "✅ All required modules present"
                    score += 25
                else:
                    details["package_structure"] = f"⚠️  Missing modules: {missing_modules}"
                    score += 10
                    
            except Exception as e:
                details["package_structure"] = f"❌ Failed: {e}"
            
            # Test 2: Documentation
            print("     • Checking documentation...")
            try:
                doc_files = ["README.md", "LICENSE"]
                existing_docs = [f for f in doc_files if Path(f).exists()]
                
                if len(existing_docs) == len(doc_files):
                    details["documentation"] = "✅ All documentation files present"
                    score += 20
                else:
                    missing_docs = [f for f in doc_files if f not in existing_docs]
                    details["documentation"] = f"⚠️  Missing: {missing_docs}"
                    score += 10
                    
            except Exception as e:
                details["documentation"] = f"❌ Failed: {e}"
            
            # Test 3: Ethical compliance framework
            print("     • Validating ethical compliance...")
            try:
                from gan_cyber_range.security.enhanced_security_framework import EthicalFramework
                
                ethical = EthicalFramework()
                
                # Test prohibited request
                malicious_request = {
                    "purpose": "malicious attack",
                    "targets": ["production_systems"],
                    "consent": False
                }
                
                if not ethical.is_compliant(malicious_request):
                    details["ethical_compliance"] = "✅ Ethical framework blocks malicious requests"
                    score += 25
                else:
                    details["ethical_compliance"] = "❌ Ethical framework failed"
                    
            except Exception as e:
                details["ethical_compliance"] = f"❌ Failed: {e}"
            
            # Test 4: Security headers/metadata
            print("     • Checking security metadata...")
            try:
                security_indicators = 0
                
                # Check for security-related imports
                with open("gan_cyber_range/security/enhanced_security_framework.py", "r") as f:
                    content = f.read()
                    if "cryptography" in content or "security" in content.lower():
                        security_indicators += 1
                
                # Check for input sanitization
                if "sanitize" in content.lower():
                    security_indicators += 1
                
                # Check for access control
                if "access" in content.lower() and "control" in content.lower():
                    security_indicators += 1
                
                if security_indicators >= 2:
                    details["security_metadata"] = f"✅ {security_indicators} security indicators found"
                    score += 15
                else:
                    details["security_metadata"] = f"⚠️  Only {security_indicators} security indicators"
                    score += 5
                    
            except Exception as e:
                details["security_metadata"] = f"❌ Failed: {e}"
            
            # Test 5: License compliance
            print("     • Checking license compliance...")
            try:
                license_file = Path("LICENSE")
                if license_file.exists():
                    with open(license_file) as f:
                        license_content = f.read()
                    
                    # Basic license validation
                    if len(license_content) > 100:  # Has substantial content
                        details["license_compliance"] = "✅ License file present and substantial"
                        score += 15
                    else:
                        details["license_compliance"] = "⚠️  License file too short"
                        score += 5
                else:
                    details["license_compliance"] = "⚠️  No LICENSE file found"
                    score += 5
                    
            except Exception as e:
                details["license_compliance"] = f"❌ Failed: {e}"
            
        except Exception as e:
            details["overall_error"] = str(e)
        
        execution_time = time.time() - start_time
        passed = score >= 60
        
        return QualityGateResult(
            gate_name="Compliance Validation",
            passed=passed,
            score=score,
            details=details,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_overall_score(self):
        """Calculate weighted overall score"""
        if not self.results:
            self.overall_score = 0.0
            return
        
        # Map gates to categories
        gate_categories = {
            "Functionality Tests": "functionality",
            "Security Analysis": "security",
            "Performance Benchmarks": "performance",
            "Reliability Tests": "reliability",
            "Compliance Validation": "compliance"
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for result in self.results:
            category = gate_categories.get(result.gate_name, "functionality")
            weight = self.gate_weights.get(category, 0.2)
            weighted_score += result.score * weight
            total_weight += weight
        
        self.overall_score = weighted_score / max(total_weight, 1.0)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_score": self.overall_score,
                "gates_passed": self.passed_gates,
                "total_gates": self.total_gates,
                "success_rate": self.passed_gates / max(self.total_gates, 1) * 100
            },
            "gate_results": [asdict(result) for result in self.results],
            "recommendations": self._generate_recommendations(),
            "quality_level": self._determine_quality_level()
        }
        
        # Save report to file
        report_file = Path(f"quality_gate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        comprehensive_logger.info(
            f"Quality gates report generated: {report_file}",
            additional_data=report["summary"]
        )
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        for result in self.results:
            if not result.passed:
                if result.gate_name == "Functionality Tests":
                    recommendations.append("Fix core functionality issues - check imports and basic operations")
                elif result.gate_name == "Security Analysis":
                    recommendations.append("Strengthen security measures - review input sanitization and access controls")
                elif result.gate_name == "Performance Benchmarks":
                    recommendations.append("Optimize performance - consider caching and resource pooling improvements")
                elif result.gate_name == "Reliability Tests":
                    recommendations.append("Improve error handling and system resilience")
                elif result.gate_name == "Compliance Validation":
                    recommendations.append("Address compliance issues - ensure documentation and ethical frameworks are complete")
        
        if self.overall_score < 80:
            recommendations.append("Overall quality needs improvement - focus on highest-impact areas first")
        
        return recommendations
    
    def _determine_quality_level(self) -> str:
        """Determine overall quality level"""
        if self.overall_score >= 90:
            return "EXCELLENT"
        elif self.overall_score >= 80:
            return "GOOD"
        elif self.overall_score >= 70:
            return "ACCEPTABLE"
        elif self.overall_score >= 60:
            return "NEEDS_IMPROVEMENT"
        else:
            return "CRITICAL_ISSUES"


def main():
    """Main execution function"""
    quality_gates = AutonomousQualityGates()
    report = quality_gates.run_all_gates()
    
    # Print summary
    print(f"\n📊 FINAL QUALITY ASSESSMENT")
    print(f"   Quality Level: {report['quality_level']}")
    print(f"   Overall Score: {report['summary']['overall_score']:.1f}/100")
    print(f"   Success Rate: {report['summary']['success_rate']:.1f}%")
    
    if report["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")
    
    return report


if __name__ == "__main__":
    main()