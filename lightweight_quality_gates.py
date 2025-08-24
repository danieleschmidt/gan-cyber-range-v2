#!/usr/bin/env python3
"""
Lightweight Quality Gates - No External Dependencies
Core quality validation using only standard library
"""

import asyncio
import logging
import time
import sys
import json
import traceback
import os
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import ast
import re

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Quality gate execution result"""
    gate_name: str
    success: bool
    score: float
    execution_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    timestamp: datetime
    overall_success: bool
    overall_score: float
    total_gates: int
    passed_gates: int
    failed_gates: int
    gate_results: List[QualityGateResult]
    execution_time_total_ms: float
    recommendations: List[str] = field(default_factory=list)


class LightweightQualityGates:
    """Lightweight quality gates using only standard library"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
    async def execute_all_gates(self) -> QualityReport:
        """Execute all quality gates"""
        logger.info("🚀 Starting Lightweight Quality Gates Execution")
        start_time = time.time()
        
        gates = [
            ("Import Validation", self._gate_import_validation),
            ("Code Structure", self._gate_code_structure),
            ("Syntax Validation", self._gate_syntax_validation),
            ("Documentation Check", self._gate_documentation),
            ("File Organization", self._gate_file_organization),
            ("Basic Security Check", self._gate_basic_security),
            ("Defensive Components", self._gate_defensive_components),
            ("Requirements Validation", self._gate_requirements)
        ]
        
        results = []
        
        for gate_name, gate_func in gates:
            logger.info(f"🔍 Executing Quality Gate: {gate_name}")
            gate_start = time.time()
            
            try:
                result = await gate_func()
                gate_time = (time.time() - gate_start) * 1000
                
                quality_result = QualityGateResult(
                    gate_name=gate_name,
                    success=result.get('success', True),
                    score=result.get('score', 0),
                    execution_time_ms=gate_time,
                    details=result,
                    warnings=result.get('warnings', []),
                    errors=result.get('errors', []),
                    recommendations=result.get('recommendations', [])
                )
                
                results.append(quality_result)
                
                status = "✅ PASSED" if quality_result.success else "❌ FAILED"
                logger.info(f"{status} {gate_name}: {quality_result.score:.1f}/100 ({gate_time:.1f}ms)")
                
            except Exception as e:
                gate_time = (time.time() - gate_start) * 1000
                error_result = QualityGateResult(
                    gate_name=gate_name,
                    success=False,
                    score=0,
                    execution_time_ms=gate_time,
                    details={'error': str(e)},
                    errors=[str(e)],
                    recommendations=[f"Fix {gate_name} execution error"]
                )
                
                results.append(error_result)
                logger.error(f"❌ FAILED {gate_name}: {e}")
        
        # Generate comprehensive report
        total_time = (time.time() - start_time) * 1000
        passed_gates = sum(1 for r in results if r.success)
        overall_score = sum(r.score for r in results) / len(results) if results else 0
        overall_success = passed_gates >= len(gates) * 0.75  # 75% pass rate
        
        report = QualityReport(
            timestamp=datetime.now(),
            overall_success=overall_success,
            overall_score=overall_score,
            total_gates=len(gates),
            passed_gates=passed_gates,
            failed_gates=len(gates) - passed_gates,
            gate_results=results,
            execution_time_total_ms=total_time,
            recommendations=self._generate_overall_recommendations(results)
        )
        
        return report
    
    async def _gate_import_validation(self) -> Dict[str, Any]:
        """Validate all imports work correctly"""
        try:
            # Test critical autonomous implementations
            critical_modules = [
                'autonomous_defensive_demo',
                'enhanced_defensive_training',
                'robust_defensive_framework',
                'high_performance_defensive_platform'
            ]
            
            successful_imports = 0
            failed_imports = []
            import_details = {}
            
            for module in critical_modules:
                try:
                    # Add current directory to Python path
                    if str(self.project_root) not in sys.path:
                        sys.path.insert(0, str(self.project_root))
                    
                    # Try to import the module
                    imported = __import__(module)
                    successful_imports += 1
                    import_details[module] = "✅ Success"
                    
                except Exception as e:
                    failed_imports.append(f"{module}: {str(e)}")
                    import_details[module] = f"❌ Failed: {str(e)}"
            
            success_rate = successful_imports / len(critical_modules)
            score = success_rate * 100
            
            recommendations = []
            if failed_imports:
                recommendations.append("Fix import errors in autonomous implementations")
            else:
                recommendations.append("All critical imports working correctly")
            
            return {
                'success': success_rate >= 0.75,  # 75% success rate required
                'score': score,
                'successful_imports': successful_imports,
                'failed_imports': failed_imports,
                'import_details': import_details,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'success': False, 
                'score': 0, 
                'error': str(e),
                'recommendations': ['Fix import validation system']
            }
    
    async def _gate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and organization"""
        try:
            # Check for key files and directories
            required_files = [
                'README.md',
                'requirements.txt',
                'setup.py'
            ]
            
            required_dirs = [
                'gan_cyber_range',
                'gan_cyber_range/core',
                'gan_cyber_range/security',
                'gan_cyber_range/training'
            ]
            
            # Check autonomous implementations
            autonomous_files = [
                'autonomous_defensive_demo.py',
                'enhanced_defensive_training.py',
                'robust_defensive_framework.py',
                'high_performance_defensive_platform.py'
            ]
            
            missing_files = []
            missing_dirs = []
            missing_autonomous = []
            
            # Check required files
            for file_path in required_files:
                if not (self.project_root / file_path).exists():
                    missing_files.append(file_path)
            
            # Check required directories
            for dir_path in required_dirs:
                if not (self.project_root / dir_path).is_dir():
                    missing_dirs.append(dir_path)
            
            # Check autonomous implementations
            for file_path in autonomous_files:
                if not (self.project_root / file_path).exists():
                    missing_autonomous.append(file_path)
            
            # Count Python files
            python_files = list(self.project_root.rglob("*.py"))
            
            # Calculate structure score
            structure_score = 100
            structure_score -= len(missing_files) * 10
            structure_score -= len(missing_dirs) * 15
            structure_score -= len(missing_autonomous) * 20  # Higher penalty for missing autonomous files
            
            structure_score = max(0, structure_score)
            
            recommendations = []
            if missing_files:
                recommendations.append(f"Create missing files: {', '.join(missing_files)}")
            if missing_dirs:
                recommendations.append(f"Create missing directories: {', '.join(missing_dirs)}")
            if missing_autonomous:
                recommendations.append(f"Implement missing autonomous components: {', '.join(missing_autonomous)}")
            if structure_score >= 90:
                recommendations.append("Excellent project structure!")
            
            return {
                'success': len(missing_files) == 0 and len(missing_dirs) == 0,
                'score': structure_score,
                'python_files_count': len(python_files),
                'missing_files': missing_files,
                'missing_directories': missing_dirs,
                'missing_autonomous': missing_autonomous,
                'autonomous_implementations': len(autonomous_files) - len(missing_autonomous),
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {'success': False, 'score': 0, 'error': str(e)}
    
    async def _gate_syntax_validation(self) -> Dict[str, Any]:
        """Validate Python syntax in all files"""
        try:
            python_files = list(self.project_root.rglob("*.py"))
            
            if not python_files:
                return {
                    'success': False,
                    'score': 0,
                    'error': 'No Python files found',
                    'recommendations': ['Add Python code to the project']
                }
            
            syntax_errors = []
            valid_files = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    # Parse AST to check syntax
                    ast.parse(source, filename=str(py_file))
                    valid_files += 1
                    
                except SyntaxError as e:
                    syntax_errors.append(f"{py_file}: Line {e.lineno}: {e.msg}")
                except Exception as e:
                    syntax_errors.append(f"{py_file}: {str(e)}")
            
            success_rate = valid_files / len(python_files) if python_files else 0
            score = success_rate * 100
            
            recommendations = []
            if syntax_errors:
                recommendations.append(f"Fix {len(syntax_errors)} syntax errors")
                recommendations.extend(syntax_errors[:3])  # Show first 3 errors
            else:
                recommendations.append("All Python files have valid syntax!")
            
            return {
                'success': len(syntax_errors) == 0,
                'score': score,
                'total_files': len(python_files),
                'valid_files': valid_files,
                'syntax_errors': syntax_errors,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {'success': False, 'score': 0, 'error': str(e)}
    
    async def _gate_documentation(self) -> Dict[str, Any]:
        """Check documentation coverage and quality"""
        try:
            # Check for documentation files
            doc_files = [
                'README.md',
                'CONTRIBUTING.md',
                'LICENSE'
            ]
            
            doc_dirs = [
                'docs'
            ]
            
            existing_docs = []
            missing_docs = []
            
            for doc_file in doc_files:
                doc_path = self.project_root / doc_file
                if doc_path.exists():
                    existing_docs.append(doc_file)
                else:
                    missing_docs.append(doc_file)
            
            # Check for docs directory
            has_docs_dir = (self.project_root / 'docs').is_dir()
            
            # Check README content quality
            readme_path = self.project_root / 'README.md'
            readme_quality_score = 0
            
            if readme_path.exists():
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                
                # Simple quality checks
                if 'installation' in readme_content.lower():
                    readme_quality_score += 25
                if 'usage' in readme_content.lower() or 'quick start' in readme_content.lower():
                    readme_quality_score += 25
                if 'example' in readme_content.lower():
                    readme_quality_score += 25
                if len(readme_content) > 1000:  # Substantial content
                    readme_quality_score += 25
            
            # Check docstrings in Python files
            python_files = list(self.project_root.rglob("*.py"))
            files_with_docstrings = 0
            
            for py_file in python_files[:10]:  # Sample first 10 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Simple docstring detection
                        if '"""' in content or "'''" in content:
                            files_with_docstrings += 1
                except Exception:
                    continue
            
            docstring_coverage = files_with_docstrings / min(10, len(python_files)) if python_files else 0
            doc_file_coverage = len(existing_docs) / len(doc_files)
            
            # Calculate documentation score
            doc_score = (doc_file_coverage * 40) + (docstring_coverage * 30) + (readme_quality_score * 0.3)
            if has_docs_dir:
                doc_score += 10
            
            success = doc_score >= 60
            
            recommendations = []
            if missing_docs:
                recommendations.append(f"Add missing documentation: {', '.join(missing_docs)}")
            if docstring_coverage < 0.5:
                recommendations.append("Add docstrings to Python modules and functions")
            if readme_quality_score < 75:
                recommendations.append("Improve README.md with usage examples and installation instructions")
            if not has_docs_dir:
                recommendations.append("Consider creating a docs/ directory for detailed documentation")
            if doc_score >= 80:
                recommendations.append("Good documentation coverage!")
            
            return {
                'success': success,
                'score': doc_score,
                'existing_docs': existing_docs,
                'missing_docs': missing_docs,
                'has_docs_dir': has_docs_dir,
                'readme_quality_score': readme_quality_score,
                'docstring_coverage': docstring_coverage,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {'success': False, 'score': 0, 'error': str(e)}
    
    async def _gate_file_organization(self) -> Dict[str, Any]:
        """Check file organization and naming conventions"""
        try:
            python_files = list(self.project_root.rglob("*.py"))
            
            # Check naming conventions
            good_names = 0
            bad_names = []
            
            for py_file in python_files:
                filename = py_file.name
                
                # Check for Python naming conventions
                if re.match(r'^[a-z_][a-z0-9_]*\.py$', filename):
                    good_names += 1
                else:
                    bad_names.append(str(py_file))
            
            # Check directory structure
            expected_structure = {
                'gan_cyber_range': 'Main package',
                'tests': 'Test directory',
                'docs': 'Documentation',
                'examples': 'Example code'
            }
            
            existing_structure = {}
            for dirname, description in expected_structure.items():
                dir_path = self.project_root / dirname
                existing_structure[dirname] = dir_path.exists()
            
            # Calculate organization score
            naming_score = (good_names / len(python_files)) * 100 if python_files else 100
            structure_score = (sum(existing_structure.values()) / len(expected_structure)) * 100
            
            organization_score = (naming_score * 0.6) + (structure_score * 0.4)
            success = organization_score >= 70
            
            recommendations = []
            if bad_names:
                recommendations.append(f"Fix naming conventions for: {', '.join(bad_names[:3])}")
            
            missing_dirs = [d for d, exists in existing_structure.items() if not exists]
            if missing_dirs:
                recommendations.append(f"Consider adding directories: {', '.join(missing_dirs)}")
            
            if organization_score >= 85:
                recommendations.append("Excellent file organization!")
            
            return {
                'success': success,
                'score': organization_score,
                'naming_score': naming_score,
                'structure_score': structure_score,
                'good_names': good_names,
                'bad_names': bad_names,
                'existing_structure': existing_structure,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {'success': False, 'score': 0, 'error': str(e)}
    
    async def _gate_basic_security(self) -> Dict[str, Any]:
        """Basic security checks without external tools"""
        try:
            python_files = list(self.project_root.rglob("*.py"))
            
            security_issues = []
            security_warnings = []
            
            # Patterns to check for security issues
            dangerous_patterns = [
                (r'eval\s*\(', 'Use of eval() function'),
                (r'exec\s*\(', 'Use of exec() function'),
                (r'__import__\s*\(', 'Dynamic import usage'),
                (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'Shell injection risk'),
                (r'os\.system\s*\(', 'Use of os.system()'),
                (r'input\s*\([^)]*\)', 'Use of input() - potential security risk'),
            ]
            
            warning_patterns = [
                (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password detected'),
                (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key detected'),
                (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret detected'),
                (r'TODO.*security', 'Security-related TODO found'),
                (r'FIXME.*security', 'Security-related FIXME found'),
            ]
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for dangerous patterns
                    for pattern, description in dangerous_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            security_issues.append(f"{py_file}: {description}")
                    
                    # Check for warning patterns
                    for pattern, description in warning_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            security_warnings.append(f"{py_file}: {description}")
                            
                except Exception:
                    continue
            
            # Calculate security score
            issue_penalty = len(security_issues) * 20
            warning_penalty = len(security_warnings) * 5
            security_score = max(0, 100 - issue_penalty - warning_penalty)
            
            success = len(security_issues) == 0 and security_score >= 80
            
            recommendations = []
            if security_issues:
                recommendations.append(f"Fix {len(security_issues)} security issues immediately")
                recommendations.extend(security_issues[:3])
            if security_warnings:
                recommendations.append(f"Review {len(security_warnings)} security warnings")
            if security_score >= 90:
                recommendations.append("Good basic security practices!")
            
            return {
                'success': success,
                'score': security_score,
                'security_issues': security_issues,
                'security_warnings': security_warnings,
                'files_scanned': len(python_files),
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {'success': False, 'score': 0, 'error': str(e)}
    
    async def _gate_defensive_components(self) -> Dict[str, Any]:
        """Validate defensive cybersecurity components"""
        try:
            # Check for defensive security modules
            defensive_modules = [
                'gan_cyber_range/security',
                'gan_cyber_range/training',
                'gan_cyber_range/blue_team',
                'gan_cyber_range/evaluation'
            ]
            
            existing_modules = []
            missing_modules = []
            
            for module_path in defensive_modules:
                module_dir = self.project_root / module_path
                if module_dir.is_dir():
                    existing_modules.append(module_path)
                else:
                    missing_modules.append(module_path)
            
            # Check for autonomous implementations
            autonomous_files = [
                'autonomous_defensive_demo.py',
                'enhanced_defensive_training.py', 
                'robust_defensive_framework.py',
                'high_performance_defensive_platform.py'
            ]
            
            existing_autonomous = []
            missing_autonomous = []
            
            for file_path in autonomous_files:
                if (self.project_root / file_path).exists():
                    existing_autonomous.append(file_path)
                else:
                    missing_autonomous.append(file_path)
            
            # Check for defensive keywords in code
            python_files = list(self.project_root.rglob("*.py"))
            defensive_keywords = [
                'security', 'threat', 'defense', 'incident', 'malware',
                'forensic', 'training', 'blue_team', 'monitoring'
            ]
            
            files_with_defensive_content = 0
            for py_file in python_files[:20]:  # Sample files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    if any(keyword in content for keyword in defensive_keywords):
                        files_with_defensive_content += 1
                except Exception:
                    continue
            
            # Calculate defensive capabilities score
            module_score = (len(existing_modules) / len(defensive_modules)) * 40
            autonomous_score = (len(existing_autonomous) / len(autonomous_files)) * 40
            content_score = min(20, (files_with_defensive_content / min(20, len(python_files))) * 20) if python_files else 0
            
            defensive_score = module_score + autonomous_score + content_score
            success = defensive_score >= 80
            
            recommendations = []
            if missing_modules:
                recommendations.append(f"Add missing defensive modules: {', '.join(missing_modules)}")
            if missing_autonomous:
                recommendations.append(f"Implement missing autonomous components: {', '.join(missing_autonomous)}")
            if files_with_defensive_content == 0:
                recommendations.append("Add defensive cybersecurity functionality to codebase")
            if defensive_score >= 90:
                recommendations.append("Comprehensive defensive capabilities implemented!")
            
            return {
                'success': success,
                'score': defensive_score,
                'existing_modules': existing_modules,
                'missing_modules': missing_modules,
                'existing_autonomous': existing_autonomous,
                'missing_autonomous': missing_autonomous,
                'files_with_defensive_content': files_with_defensive_content,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {'success': False, 'score': 0, 'error': str(e)}
    
    async def _gate_requirements(self) -> Dict[str, Any]:
        """Validate requirements and dependencies"""
        try:
            requirements_file = self.project_root / "requirements.txt"
            
            if not requirements_file.exists():
                return {
                    'success': False,
                    'score': 0,
                    'error': 'requirements.txt not found',
                    'recommendations': ['Create requirements.txt file with project dependencies']
                }
            
            # Parse requirements
            with open(requirements_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            requirements = []
            comments = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
                elif line.startswith('#'):
                    comments.append(line)
            
            # Check for version pinning
            pinned_versions = 0
            unpinned = []
            
            for req in requirements:
                if any(op in req for op in ['==', '>=', '<=', '~=', '!=']):
                    pinned_versions += 1
                else:
                    unpinned.append(req)
            
            pinning_rate = pinned_versions / len(requirements) if requirements else 0
            
            # Check for essential defensive packages
            defensive_packages = [
                'torch', 'transformers', 'sklearn', 'numpy', 'pandas',
                'fastapi', 'uvicorn', 'sqlalchemy', 'redis', 'cryptography'
            ]
            
            found_defensive_packages = []
            for req in requirements:
                req_lower = req.lower()
                for pkg in defensive_packages:
                    if pkg in req_lower:
                        found_defensive_packages.append(pkg)
                        break
            
            # Calculate requirements score
            requirements_score = 0
            requirements_score += min(40, len(requirements) * 2)  # Up to 40 points for having requirements
            requirements_score += pinning_rate * 30  # 30 points for version pinning
            requirements_score += min(30, len(found_defensive_packages) * 5)  # Up to 30 points for defensive packages
            
            success = requirements_score >= 60
            
            recommendations = []
            if len(requirements) == 0:
                recommendations.append("Add project dependencies to requirements.txt")
            if pinning_rate < 0.8:
                recommendations.append("Pin more dependency versions for reproducible builds")
            if len(found_defensive_packages) < 5:
                recommendations.append("Consider adding more cybersecurity-focused packages")
            if unpinned:
                recommendations.append(f"Pin versions for: {', '.join(unpinned[:3])}")
            if requirements_score >= 80:
                recommendations.append("Good dependency management!")
            
            return {
                'success': success,
                'score': requirements_score,
                'total_requirements': len(requirements),
                'pinned_versions': pinned_versions,
                'unpinned_requirements': unpinned,
                'pinning_rate': pinning_rate,
                'defensive_packages': found_defensive_packages,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {'success': False, 'score': 0, 'error': str(e)}
    
    def _generate_overall_recommendations(self, results: List[QualityGateResult]) -> List[str]:
        """Generate overall improvement recommendations"""
        recommendations = []
        
        # Identify failed gates
        failed_gates = [r for r in results if not r.success]
        low_score_gates = [r for r in results if r.score < 70]
        
        if failed_gates:
            recommendations.append(f"🔴 Priority: Fix failing quality gates: {', '.join(r.gate_name for r in failed_gates)}")
        
        if low_score_gates:
            recommendations.append(f"🟡 Improve low-scoring areas: {', '.join(r.gate_name for r in low_score_gates)}")
        
        # Overall assessment
        overall_score = sum(r.score for r in results) / len(results) if results else 0
        
        if overall_score >= 90:
            recommendations.append("🏆 Excellent overall quality! Ready for production deployment")
        elif overall_score >= 80:
            recommendations.append("✅ Good quality standards met. Minor improvements recommended")
        elif overall_score >= 70:
            recommendations.append("⚠️ Acceptable quality. Some improvements needed before production")
        else:
            recommendations.append("🚨 Quality standards not met. Major improvements required")
        
        # Add specific improvement suggestions
        if any('import' in r.gate_name.lower() for r in failed_gates):
            recommendations.append("🔧 Focus on fixing import dependencies and module structure")
        
        if any('security' in r.gate_name.lower() for r in failed_gates):
            recommendations.append("🛡️ Address security vulnerabilities and improve security practices")
        
        if any('defensive' in r.gate_name.lower() for r in failed_gates):
            recommendations.append("🔒 Complete defensive cybersecurity implementation")
        
        return recommendations


async def main():
    """Main quality gates execution"""
    logger.info("🚀 Starting Lightweight Quality Gates")
    
    project_root = Path.cwd()
    quality_gates = LightweightQualityGates(project_root)
    
    try:
        # Execute all quality gates
        report = await quality_gates.execute_all_gates()
        
        # Display results
        print(f"\n{'='*80}")
        print("🔬 LIGHTWEIGHT QUALITY GATES REPORT")
        print('='*80)
        
        print(f"📊 Overall Score: {report.overall_score:.1f}/100")
        print(f"🎯 Gates Passed: {report.passed_gates}/{report.total_gates}")
        print(f"⏱️  Total Execution Time: {report.execution_time_total_ms:.1f}ms")
        print(f"✅ Overall Status: {'PASSED' if report.overall_success else 'FAILED'}")
        
        print(f"\n📋 QUALITY GATE RESULTS:")
        for result in report.gate_results:
            status_icon = "✅" if result.success else "❌"
            print(f"  {status_icon} {result.gate_name}: {result.score:.1f}/100 ({result.execution_time_ms:.1f}ms)")
            
            # Show key details
            if 'import_details' in result.details:
                for module, status in result.details['import_details'].items():
                    print(f"    📦 {module}: {status}")
            
            if result.errors:
                for error in result.errors[:2]:  # Limit errors
                    print(f"    🚨 {error}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations[:5], 1):  # Top 5 recommendations
            print(f"  {i}. {rec}")
        
        # Display key metrics
        if report.gate_results:
            structure_result = next((r for r in report.gate_results if 'structure' in r.gate_name.lower()), None)
            if structure_result:
                print(f"\n📁 PROJECT STRUCTURE:")
                details = structure_result.details
                print(f"   Python files: {details.get('python_files_count', 0)}")
                print(f"   Autonomous implementations: {details.get('autonomous_implementations', 0)}/4")
                
            defensive_result = next((r for r in report.gate_results if 'defensive' in r.gate_name.lower()), None)
            if defensive_result:
                print(f"\n🛡️ DEFENSIVE CAPABILITIES:")
                details = defensive_result.details
                print(f"   Security modules: {len(details.get('existing_modules', []))}/4")
                print(f"   Autonomous components: {len(details.get('existing_autonomous', []))}/4")
        
        # Save detailed report
        report_file = Path(f"lightweight_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            report_dict = {
                'timestamp': report.timestamp.isoformat(),
                'overall_success': report.overall_success,
                'overall_score': report.overall_score,
                'total_gates': report.total_gates,
                'passed_gates': report.passed_gates,
                'failed_gates': report.failed_gates,
                'execution_time_total_ms': report.execution_time_total_ms,
                'gate_results': [
                    {
                        'gate_name': r.gate_name,
                        'success': r.success,
                        'score': r.score,
                        'execution_time_ms': r.execution_time_ms,
                        'details': r.details,
                        'warnings': r.warnings,
                        'errors': r.errors,
                        'recommendations': r.recommendations
                    }
                    for r in report.gate_results
                ],
                'recommendations': report.recommendations
            }
            
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        # Exit with appropriate code
        if report.overall_success:
            print(f"\n🎉 QUALITY GATES PASSED! System ready for deployment.")
        else:
            print(f"\n⚠️ QUALITY GATES INCOMPLETE - Review recommendations above.")
        
        return report.overall_success
        
    except Exception as e:
        logger.error(f"❌ Quality gates execution failed: {e}")
        print(f"\n💥 EXECUTION FAILED: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Run quality gates
    success = asyncio.run(main())
    sys.exit(0 if success else 1)