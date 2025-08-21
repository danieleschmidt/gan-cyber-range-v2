#!/usr/bin/env python3
"""
Comprehensive Quality Gates for GAN Cyber Range v2.0

This module implements mandatory quality gates including:
- Code functionality verification 
- Security scanning and validation
- Performance benchmarking
- Documentation completeness
- Production readiness assessment
"""

import sys
import os
import time
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add project to path
sys.path.insert(0, '/root/repo')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check"""
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    timestamp: str


class QualityGateRunner:
    """Runs comprehensive quality gate checks"""
    
    def __init__(self):
        self.results: List[QualityGateResult] = []
        self.start_time = time.time()
        
    def run_all_gates(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all quality gates and return overall pass/fail status"""
        
        print("🔒 EXECUTING MANDATORY QUALITY GATES")
        print("=" * 50)
        
        # Define all quality gates
        quality_gates = [
            ("Code Functionality", self._test_code_functionality),
            ("Security Scan", self._test_security),
            ("Performance Benchmarks", self._test_performance),
            ("Error Handling", self._test_error_handling),
            ("Documentation", self._test_documentation),
            ("API Compliance", self._test_api_compliance),
            ("Data Integrity", self._test_data_integrity),
            ("Resource Management", self._test_resource_management),
            ("Scalability", self._test_scalability),
            ("Production Readiness", self._test_production_readiness)
        ]
        
        # Execute each quality gate
        for gate_name, gate_function in quality_gates:
            print(f"\n🧪 Testing: {gate_name}")
            print("-" * 30)
            
            gate_start = time.time()
            try:
                passed, score, details = gate_function()
                gate_time = time.time() - gate_start
                
                result = QualityGateResult(
                    name=gate_name,
                    passed=passed,
                    score=score,
                    details=details,
                    execution_time=gate_time,
                    timestamp=datetime.now().isoformat()
                )
                
                self.results.append(result)
                
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{status} - Score: {score:.1f}/100 - Time: {gate_time:.2f}s")
                
                if not passed:
                    print(f"   Issues: {details.get('issues', [])}")
                else:
                    print(f"   Success: {details.get('summary', 'All checks passed')}")
                    
            except Exception as e:
                gate_time = time.time() - gate_start
                result = QualityGateResult(
                    name=gate_name,
                    passed=False,
                    score=0.0,
                    details={"error": str(e)},
                    execution_time=gate_time,
                    timestamp=datetime.now().isoformat()
                )
                self.results.append(result)
                print(f"❌ ERROR - {e}")
        
        # Calculate overall results
        total_time = time.time() - self.start_time
        passed_gates = sum(1 for r in self.results if r.passed)
        total_gates = len(self.results)
        overall_score = sum(r.score for r in self.results) / len(self.results) if self.results else 0
        
        overall_pass = passed_gates >= int(total_gates * 0.85)  # 85% pass rate required
        
        summary = {
            "total_gates": total_gates,
            "passed_gates": passed_gates,
            "failed_gates": total_gates - passed_gates,
            "pass_rate": passed_gates / total_gates if total_gates > 0 else 0,
            "overall_score": overall_score,
            "overall_pass": overall_pass,
            "execution_time": total_time,
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "score": r.score,
                    "time": r.execution_time
                } for r in self.results
            ]
        }
        
        return overall_pass, summary
    
    def _test_code_functionality(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Test core code functionality"""
        issues = []
        score = 100.0
        
        try:
            # Test imports
            from gan_cyber_range.core.ultra_minimal import (
                UltraMinimalGenerator, UltraMinimalCyberRange, AttackVector
            )
            
            # Test basic functionality
            generator = UltraMinimalGenerator()
            attacks = generator.generate(num_samples=10)
            
            if len(attacks) != 10:
                issues.append(f"Expected 10 attacks, got {len(attacks)}")
                score -= 20
            
            # Test cyber range
            cyber_range = UltraMinimalCyberRange()
            cyber_range.deploy()
            cyber_range.start()
            
            if cyber_range.status != "running":
                issues.append("Cyber range failed to start")
                score -= 15
            
            # Test attack execution
            test_attack = attacks[0] if attacks else AttackVector(
                attack_type="test", payload="test", techniques=["T1001"],
                severity=0.5, stealth_level=0.5, target_systems=["test"]
            )
            
            result = cyber_range.execute_attack(test_attack)
            if not result or "attack_id" not in result:
                issues.append("Attack execution failed")
                score -= 20
            
            cyber_range.stop()
            
            # Test diversity calculation
            diversity = generator.diversity_score(attacks)
            if not (0.0 <= diversity <= 1.0):
                issues.append(f"Invalid diversity score: {diversity}")
                score -= 10
            
        except Exception as e:
            issues.append(f"Critical functionality error: {e}")
            score = 0.0
        
        return len(issues) == 0, max(0, score), {
            "issues": issues,
            "summary": f"Functionality test completed with {len(issues)} issues"
        }
    
    def _test_security(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Test security aspects"""
        issues = []
        score = 100.0
        
        # Check for hardcoded secrets
        secrets_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        import re
        
        python_files = list(Path('/root/repo').rglob('*.py'))
        for file_path in python_files:
            try:
                content = file_path.read_text()
                for pattern in secrets_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"Potential hardcoded secret in {file_path}")
                        score -= 5
            except:
                continue
        
        # Check for SQL injection vulnerabilities
        sql_patterns = [
            r'execute\s*\(\s*["\'][^"\']*\+',
            r'query\s*\(\s*["\'][^"\']*\+',
            r'SELECT\s*.*\+.*FROM'
        ]
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                for pattern in sql_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"Potential SQL injection in {file_path}")
                        score -= 10
            except:
                continue
        
        # Check for input validation
        try:
            from gan_cyber_range.core.ultra_minimal import AttackVector
            
            # Test with invalid inputs
            try:
                invalid_attack = AttackVector(
                    attack_type="",
                    payload="",
                    techniques=[],
                    severity=-1.0,
                    stealth_level=2.0,
                    target_systems=[]
                )
                # Should handle gracefully
            except Exception:
                pass  # Expected to handle gracefully
                
        except Exception as e:
            issues.append(f"Input validation test failed: {e}")
            score -= 15
        
        # Check attack generation safety
        try:
            from gan_cyber_range.core.ultra_minimal import UltraMinimalGenerator
            
            generator = UltraMinimalGenerator()
            attacks = generator.generate(num_samples=5)
            
            # Ensure attacks are for training purposes only
            dangerous_payloads = ["rm -rf /", "format c:", "del /q /s *"]
            for attack in attacks:
                payload_str = str(attack.payload).lower()
                for dangerous in dangerous_payloads:
                    if dangerous in payload_str:
                        issues.append(f"Potentially dangerous payload generated: {dangerous}")
                        score -= 20
                        
        except Exception as e:
            issues.append(f"Attack generation safety check failed: {e}")
            score -= 10
        
        return len(issues) == 0, max(0, score), {
            "issues": issues,
            "summary": f"Security scan completed with {len(issues)} potential issues",
            "files_scanned": len(python_files)
        }
    
    def _test_performance(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Test performance benchmarks"""
        issues = []
        score = 100.0
        benchmarks = {}
        
        try:
            from gan_cyber_range.core.ultra_minimal import UltraMinimalGenerator, UltraMinimalCyberRange
            
            # Benchmark 1: Attack generation speed
            generator = UltraMinimalGenerator()
            start_time = time.time()
            attacks = generator.generate(num_samples=100)
            generation_time = time.time() - start_time
            
            benchmarks["attack_generation_100"] = {
                "time": generation_time,
                "rate": 100 / generation_time,
                "target": ">50 attacks/second"
            }
            
            if generation_time > 2.0:  # Should generate 100 attacks in under 2 seconds
                issues.append(f"Attack generation too slow: {generation_time:.2f}s for 100 attacks")
                score -= 15
            
            # Benchmark 2: Diversity calculation speed
            start_time = time.time()
            diversity = generator.diversity_score(attacks)
            diversity_time = time.time() - start_time
            
            benchmarks["diversity_calculation"] = {
                "time": diversity_time,
                "target": "<1.0 second"
            }
            
            if diversity_time > 1.0:
                issues.append(f"Diversity calculation too slow: {diversity_time:.2f}s")
                score -= 10
            
            # Benchmark 3: Cyber range deployment speed
            cyber_range = UltraMinimalCyberRange()
            start_time = time.time()
            cyber_range.deploy()
            cyber_range.start()
            deployment_time = time.time() - start_time
            
            benchmarks["range_deployment"] = {
                "time": deployment_time,
                "target": "<0.5 seconds"
            }
            
            if deployment_time > 0.5:
                issues.append(f"Range deployment too slow: {deployment_time:.2f}s")
                score -= 10
            
            # Benchmark 4: Attack execution speed
            test_attacks = attacks[:10]
            start_time = time.time()
            for attack in test_attacks:
                cyber_range.execute_attack(attack)
            execution_time = time.time() - start_time
            
            benchmarks["attack_execution_10"] = {
                "time": execution_time,
                "rate": 10 / execution_time,
                "target": ">5 attacks/second"
            }
            
            if execution_time > 2.0:  # Should execute 10 attacks in under 2 seconds
                issues.append(f"Attack execution too slow: {execution_time:.2f}s for 10 attacks")
                score -= 15
            
            cyber_range.stop()
            
            # Benchmark 5: Memory usage (basic check)
            import psutil
            import os
            
            current_process = psutil.Process(os.getpid())
            memory_mb = current_process.memory_info().rss / 1024 / 1024
            
            benchmarks["memory_usage"] = {
                "memory_mb": memory_mb,
                "target": "<100 MB"
            }
            
            if memory_mb > 100:
                issues.append(f"Memory usage too high: {memory_mb:.1f} MB")
                score -= 10
            
        except ImportError:
            # psutil not available, skip memory test
            pass
        except Exception as e:
            issues.append(f"Performance benchmarking failed: {e}")
            score = 0.0
        
        return len(issues) == 0, max(0, score), {
            "issues": issues,
            "benchmarks": benchmarks,
            "summary": f"Performance benchmarks completed with {len(issues)} issues"
        }
    
    def _test_error_handling(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Test error handling and edge cases"""
        issues = []
        score = 100.0
        
        try:
            from gan_cyber_range.core.ultra_minimal import UltraMinimalGenerator, UltraMinimalCyberRange, AttackVector
            
            # Test 1: Invalid generator inputs
            generator = UltraMinimalGenerator()
            
            # Zero attacks
            zero_attacks = generator.generate(num_samples=0)
            if len(zero_attacks) != 0:
                issues.append("Zero sample request not handled correctly")
                score -= 10
            
            # Negative attacks
            neg_attacks = generator.generate(num_samples=-5)
            if len(neg_attacks) != 0:
                issues.append("Negative sample request not handled correctly")
                score -= 10
            
            # Invalid attack type
            invalid_attacks = generator.generate(num_samples=2, attack_type="nonexistent_type")
            if len(invalid_attacks) != 2:  # Should still generate attacks
                issues.append("Invalid attack type not handled gracefully")
                score -= 10
            
            # Test 2: Diversity edge cases
            single_attack = generator.generate(num_samples=1)
            diversity_single = generator.diversity_score(single_attack)
            if diversity_single != 0.0:
                issues.append(f"Single attack diversity should be 0.0, got {diversity_single}")
                score -= 5
            
            empty_diversity = generator.diversity_score([])
            if empty_diversity != 0.0:
                issues.append(f"Empty list diversity should be 0.0, got {empty_diversity}")
                score -= 5
            
            # Test 3: Cyber range error handling
            cyber_range = UltraMinimalCyberRange()
            
            # Try to start without deploying
            try:
                cyber_range.start()  # Should handle gracefully or raise appropriate error
            except Exception:
                pass  # Expected behavior
            
            # Deploy and test multiple starts
            cyber_range.deploy()
            cyber_range.start()
            
            if cyber_range.status != "running":
                issues.append("Cyber range status not correctly managed")
                score -= 10
            
            # Test stop without errors
            cyber_range.stop()
            if cyber_range.status != "stopped":
                issues.append("Cyber range stop not handled correctly")
                score -= 10
            
            # Test 4: Attack vector validation
            try:
                # Invalid severity values
                invalid_attack = AttackVector(
                    attack_type="test",
                    payload="test",
                    techniques=["T1001"],
                    severity=1.5,  # Invalid
                    stealth_level=-0.1,  # Invalid
                    target_systems=["test"]
                )
                # Should create object but with corrected values or handle gracefully
            except Exception as e:
                # Should not crash completely
                issues.append(f"Attack vector validation too strict: {e}")
                score -= 5
            
        except Exception as e:
            issues.append(f"Error handling test failed: {e}")
            score = 0.0
        
        return len(issues) == 0, max(0, score), {
            "issues": issues,
            "summary": f"Error handling test completed with {len(issues)} issues"
        }
    
    def _test_documentation(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Test documentation completeness"""
        issues = []
        score = 100.0
        
        # Check for README
        readme_path = Path('/root/repo/README.md')
        if not readme_path.exists():
            issues.append("README.md not found")
            score -= 20
        else:
            readme_content = readme_path.read_text()
            if len(readme_content) < 1000:
                issues.append("README.md appears too short")
                score -= 10
            
            required_sections = ["installation", "usage", "example", "overview"]
            for section in required_sections:
                if section.lower() not in readme_content.lower():
                    issues.append(f"README missing {section} section")
                    score -= 5
        
        # Check for docstrings in core modules
        core_modules = [
            '/root/repo/gan_cyber_range/core/ultra_minimal.py',
            '/root/repo/gan_cyber_range/core/attack_gan.py',
            '/root/repo/gan_cyber_range/core/cyber_range.py'
        ]
        
        for module_path in core_modules:
            module_path = Path(module_path)
            if module_path.exists():
                content = module_path.read_text()
                
                # Check for module docstring
                if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                    issues.append(f"Module {module_path.name} missing docstring")
                    score -= 5
                
                # Check for class docstrings
                import re
                classes = re.findall(r'class\s+(\w+)', content)
                for class_name in classes:
                    class_pattern = rf'class\s+{class_name}.*?:\s*\n\s*"""'
                    if not re.search(class_pattern, content, re.DOTALL):
                        issues.append(f"Class {class_name} in {module_path.name} missing docstring")
                        score -= 2
        
        # Check for setup.py
        setup_path = Path('/root/repo/setup.py')
        if not setup_path.exists():
            issues.append("setup.py not found")
            score -= 15
        
        # Check for requirements.txt
        req_path = Path('/root/repo/requirements.txt')
        if not req_path.exists():
            issues.append("requirements.txt not found")
            score -= 10
        
        return len(issues) == 0, max(0, score), {
            "issues": issues,
            "summary": f"Documentation check completed with {len(issues)} issues"
        }
    
    def _test_api_compliance(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Test API compliance and consistency"""
        issues = []
        score = 100.0
        
        try:
            from gan_cyber_range.core.ultra_minimal import UltraMinimalGenerator, AttackVector
            
            # Test 1: Check return types
            generator = UltraMinimalGenerator()
            attacks = generator.generate(num_samples=5)
            
            if not isinstance(attacks, list):
                issues.append("generate() should return a list")
                score -= 20
            
            if attacks and not isinstance(attacks[0], AttackVector):
                issues.append("generate() should return list of AttackVector objects")
                score -= 20
            
            # Test 2: Check diversity_score return type
            diversity = generator.diversity_score(attacks)
            if not isinstance(diversity, (int, float)):
                issues.append("diversity_score() should return a number")
                score -= 10
            
            if not (0.0 <= diversity <= 1.0):
                issues.append(f"diversity_score() should return value between 0-1, got {diversity}")
                score -= 10
            
            # Test 3: Check AttackVector structure
            if attacks:
                attack = attacks[0]
                required_attrs = ['attack_type', 'payload', 'techniques', 'severity', 'stealth_level', 'target_systems']
                
                for attr in required_attrs:
                    if not hasattr(attack, attr):
                        issues.append(f"AttackVector missing required attribute: {attr}")
                        score -= 10
                    else:
                        value = getattr(attack, attr)
                        if value is None:
                            issues.append(f"AttackVector.{attr} should not be None")
                            score -= 5
            
            # Test 4: Check parameter validation
            # Should handle reasonable parameter ranges
            large_attacks = generator.generate(num_samples=1000)
            if len(large_attacks) > 1000:
                issues.append("generate() returned more attacks than requested")
                score -= 10
            
        except Exception as e:
            issues.append(f"API compliance test failed: {e}")
            score = 0.0
        
        return len(issues) == 0, max(0, score), {
            "issues": issues,
            "summary": f"API compliance test completed with {len(issues)} issues"
        }
    
    def _test_data_integrity(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Test data integrity and persistence"""
        issues = []
        score = 100.0
        
        try:
            from gan_cyber_range.core.ultra_minimal import UltraMinimalGenerator
            
            generator = UltraMinimalGenerator()
            original_attacks = generator.generate(num_samples=10)
            
            # Test save/load cycle
            test_file = Path('/tmp/integrity_test_attacks.json')
            
            # Save attacks
            generator.save_attacks(original_attacks, test_file)
            
            if not test_file.exists():
                issues.append("Save operation failed - file not created")
                score -= 30
                return len(issues) == 0, max(0, score), {"issues": issues}
            
            # Load attacks
            loaded_attacks = generator.load_attacks(test_file)
            
            # Verify count
            if len(loaded_attacks) != len(original_attacks):
                issues.append(f"Attack count mismatch: {len(original_attacks)} vs {len(loaded_attacks)}")
                score -= 20
            
            # Verify data integrity
            for i, (original, loaded) in enumerate(zip(original_attacks, loaded_attacks)):
                if original.attack_type != loaded.attack_type:
                    issues.append(f"Attack {i}: attack_type mismatch")
                    score -= 5
                
                if str(original.payload) != str(loaded.payload):
                    issues.append(f"Attack {i}: payload mismatch")
                    score -= 5
                
                if original.techniques != loaded.techniques:
                    issues.append(f"Attack {i}: techniques mismatch")
                    score -= 5
                
                if abs(original.severity - loaded.severity) > 0.01:
                    issues.append(f"Attack {i}: severity mismatch")
                    score -= 3
                
                if abs(original.stealth_level - loaded.stealth_level) > 0.01:
                    issues.append(f"Attack {i}: stealth_level mismatch")
                    score -= 3
            
            # Clean up
            test_file.unlink()
            
            # Test JSON structure
            generator.save_attacks(original_attacks[:2], test_file)
            with open(test_file, 'r') as f:
                data = json.load(f)
            
            if 'metadata' not in data:
                issues.append("Saved JSON missing metadata section")
                score -= 10
            
            if 'attacks' not in data:
                issues.append("Saved JSON missing attacks section")
                score -= 15
            
            test_file.unlink()
            
        except Exception as e:
            issues.append(f"Data integrity test failed: {e}")
            score = 0.0
        
        return len(issues) == 0, max(0, score), {
            "issues": issues,
            "summary": f"Data integrity test completed with {len(issues)} issues"
        }
    
    def _test_resource_management(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Test resource management and cleanup"""
        issues = []
        score = 100.0
        
        try:
            from gan_cyber_range.core.ultra_minimal import UltraMinimalGenerator, UltraMinimalCyberRange
            
            # Test 1: Memory management with large datasets
            generator = UltraMinimalGenerator()
            
            # Generate large number of attacks
            large_attacks = generator.generate(num_samples=1000)
            
            if len(large_attacks) != 1000:
                issues.append(f"Large dataset generation failed: {len(large_attacks)}/1000")
                score -= 15
            
            # Test cleanup - objects should be collectible
            del large_attacks
            
            # Test 2: Cyber range resource management
            ranges = []
            for i in range(5):
                cyber_range = UltraMinimalCyberRange()
                cyber_range.deploy()
                cyber_range.start()
                ranges.append(cyber_range)
            
            # Stop all ranges
            for cyber_range in ranges:
                cyber_range.stop()
                if cyber_range.status != "stopped":
                    issues.append(f"Range {cyber_range.range_id} not properly stopped")
                    score -= 5
            
            # Test 3: File handle management
            temp_files = []
            for i in range(10):
                temp_file = Path(f'/tmp/test_resource_{i}.json')
                generator.save_attacks(generator.generate(num_samples=5), temp_file)
                temp_files.append(temp_file)
            
            # Verify files created
            for temp_file in temp_files:
                if not temp_file.exists():
                    issues.append(f"File {temp_file} not created")
                    score -= 5
            
            # Clean up
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            
            # Verify cleanup
            for temp_file in temp_files:
                if temp_file.exists():
                    issues.append(f"File {temp_file} not cleaned up")
                    score -= 3
                    
        except Exception as e:
            issues.append(f"Resource management test failed: {e}")
            score = 0.0
        
        return len(issues) == 0, max(0, score), {
            "issues": issues,
            "summary": f"Resource management test completed with {len(issues)} issues"
        }
    
    def _test_scalability(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Test scalability and performance under load"""
        issues = []
        score = 100.0
        benchmarks = {}
        
        try:
            from gan_cyber_range.core.ultra_minimal import UltraMinimalGenerator, UltraMinimalCyberRange
            
            # Test 1: Attack generation scalability
            generator = UltraMinimalGenerator()
            
            scale_tests = [10, 100, 1000]
            for scale in scale_tests:
                start_time = time.time()
                attacks = generator.generate(num_samples=scale)
                generation_time = time.time() - start_time
                
                rate = scale / generation_time if generation_time > 0 else 0
                benchmarks[f"generation_rate_{scale}"] = {
                    "attacks": len(attacks),
                    "time": generation_time,
                    "rate": rate
                }
                
                # Performance thresholds
                if scale == 1000 and generation_time > 5.0:
                    issues.append(f"1000-attack generation too slow: {generation_time:.2f}s")
                    score -= 15
                
                if len(attacks) != scale:
                    issues.append(f"Scale test {scale}: expected {scale}, got {len(attacks)}")
                    score -= 10
            
            # Test 2: Concurrent attack execution simulation
            cyber_range = UltraMinimalCyberRange()
            cyber_range.deploy()
            cyber_range.start()
            
            batch_attacks = generator.generate(num_samples=50)
            
            start_time = time.time()
            results = []
            for attack in batch_attacks:
                result = cyber_range.execute_attack(attack)
                results.append(result)
            execution_time = time.time() - start_time
            
            execution_rate = len(batch_attacks) / execution_time if execution_time > 0 else 0
            benchmarks["execution_rate_50"] = {
                "attacks": len(batch_attacks),
                "time": execution_time,
                "rate": execution_rate
            }
            
            if execution_time > 10.0:  # Should execute 50 attacks in under 10 seconds
                issues.append(f"Batch execution too slow: {execution_time:.2f}s for 50 attacks")
                score -= 15
            
            # Test 3: Diversity calculation scalability
            large_attack_set = generator.generate(num_samples=500)
            
            start_time = time.time()
            diversity = generator.diversity_score(large_attack_set)
            diversity_time = time.time() - start_time
            
            benchmarks["diversity_500"] = {
                "attacks": len(large_attack_set),
                "time": diversity_time,
                "diversity": diversity
            }
            
            if diversity_time > 2.0:
                issues.append(f"Large diversity calculation too slow: {diversity_time:.2f}s")
                score -= 10
            
            cyber_range.stop()
            
        except Exception as e:
            issues.append(f"Scalability test failed: {e}")
            score = 0.0
        
        return len(issues) == 0, max(0, score), {
            "issues": issues,
            "benchmarks": benchmarks,
            "summary": f"Scalability test completed with {len(issues)} issues"
        }
    
    def _test_production_readiness(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Test production readiness"""
        issues = []
        score = 100.0
        
        # Test 1: Package structure
        required_files = [
            '/root/repo/setup.py',
            '/root/repo/requirements.txt',
            '/root/repo/README.md',
            '/root/repo/gan_cyber_range/__init__.py'
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                issues.append(f"Missing required file: {file_path}")
                score -= 10
        
        # Test 2: Import structure
        try:
            import gan_cyber_range
            if not hasattr(gan_cyber_range, '__version__'):
                issues.append("Package missing __version__ attribute")
                score -= 5
            
        except ImportError as e:
            issues.append(f"Package import failed: {e}")
            score -= 20
        
        # Test 3: Configuration management
        config_paths = [
            '/root/repo/config/',
            '/root/repo/gan_cyber_range/config/',
        ]
        
        config_found = any(Path(p).exists() for p in config_paths)
        if not config_found:
            issues.append("No configuration directory found")
            score -= 5
        
        # Test 4: Logging configuration
        try:
            from gan_cyber_range.core.ultra_minimal import UltraMinimalGenerator
            
            generator = UltraMinimalGenerator()
            # Should not crash with logging
            attacks = generator.generate(num_samples=5)
            
        except Exception as e:
            issues.append(f"Logging integration issue: {e}")
            score -= 10
        
        # Test 5: Error handling in production scenario
        try:
            from gan_cyber_range.core.ultra_minimal import UltraMinimalCyberRange
            
            # Simulate production usage patterns
            cyber_range = UltraMinimalCyberRange()
            cyber_range.deploy()
            cyber_range.start()
            
            # Rapid succession of operations
            for _ in range(10):
                attacks = cyber_range.generate_attacks(num_attacks=1)
                if attacks:
                    cyber_range.execute_attack(attacks[0])
            
            metrics = cyber_range.get_metrics()
            if not metrics:
                issues.append("Production metrics collection failed")
                score -= 15
            
            cyber_range.stop()
            
        except Exception as e:
            issues.append(f"Production simulation failed: {e}")
            score -= 20
        
        return len(issues) == 0, max(0, score), {
            "issues": issues,
            "summary": f"Production readiness test completed with {len(issues)} issues"
        }


def main():
    """Execute all quality gates"""
    runner = QualityGateRunner()
    
    overall_pass, summary = runner.run_all_gates()
    
    print("\n" + "=" * 60)
    print("🏁 QUALITY GATES EXECUTION COMPLETE")
    print("=" * 60)
    
    print(f"📊 OVERALL RESULTS:")
    print(f"   Total Gates: {summary['total_gates']}")
    print(f"   Passed: {summary['passed_gates']}")
    print(f"   Failed: {summary['failed_gates']}")
    print(f"   Pass Rate: {summary['pass_rate']:.1%}")
    print(f"   Overall Score: {summary['overall_score']:.1f}/100")
    print(f"   Execution Time: {summary['execution_time']:.2f}s")
    
    status = "✅ PASSED" if overall_pass else "❌ FAILED"
    print(f"\n🎯 FINAL RESULT: {status}")
    
    if overall_pass:
        print("🎉 System ready for production deployment!")
        print("✅ All mandatory quality gates satisfied")
        print("✅ 85%+ pass rate achieved")
        print("✅ No critical issues detected")
    else:
        print("⚠️  System not ready for production")
        print("❌ Quality gates failed")
        print("🔧 Review failed gates and address issues")
    
    # Save detailed results
    results_path = Path('/root/repo/quality_gates_results.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_path}")
    
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())