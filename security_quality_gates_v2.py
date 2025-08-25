#!/usr/bin/env python3
"""
Security and Quality Gates for GAN Cyber Range Platform (Version 2)
Comprehensive security validation and quality assurance checks
"""

import sys
import os
import time
import json
import logging
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import hashlib
import ast

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecurityGate:
    """Comprehensive security validation gate"""
    
    def __init__(self):
        self.security_findings = []
        self.critical_issues = []
        self.warning_issues = []
        
    def validate_code_security(self, code_paths: List[Path]) -> Dict[str, Any]:
        """Validate code for security vulnerabilities"""
        logger.info("Running security code analysis...")
        
        security_report = {
            "timestamp": datetime.now().isoformat(),
            "files_scanned": 0,
            "critical_issues": [],
            "warning_issues": [],
            "info_issues": [],
            "security_score": 0.0
        }
        
        # Security patterns to check
        security_patterns = {
            "critical": [
                (r"eval\s*\(", "Use of eval() is dangerous"),
                (r"exec\s*\(", "Use of exec() is dangerous"),
                (r"__import__\s*\(", "Dynamic imports can be dangerous"),
                (r"subprocess\.call\([^)]*shell\s*=\s*True", "Shell injection risk"),
                (r"os\.system\s*\(", "Command injection risk"),
                (r"input\s*\([^)]*\)\s*.*exec", "Input directly to exec is dangerous"),
            ],
            "warning": [
                (r"pickle\.loads?\s*\(", "Pickle deserialization can be unsafe"),
                (r"yaml\.load\s*\(", "YAML load without safe_load is risky"),
                (r"subprocess\.Popen\s*\([^)]*shell\s*=\s*True", "Shell subprocess risk"),
                (r"random\.random\s*\(\)", "Use cryptographically secure random for security"),
                (r"hashlib\.md5\s*\(", "MD5 is cryptographically broken"),
                (r"hashlib\.sha1\s*\(", "SHA1 is cryptographically weak"),
            ],
            "info": [
                (r"print\s*\([^)]*password", "Potential password in print statement"),
                (r"print\s*\([^)]*key", "Potential key in print statement"),
                (r"TODO.*security", "Security TODO found"),
                (r"FIXME.*security", "Security FIXME found"),
                (r"XXX.*security", "Security XXX found"),
            ]
        }
        
        for file_path in code_paths:
            if not file_path.suffix == '.py':
                continue
                
            security_report["files_scanned"] += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check security patterns
                for severity, patterns in security_patterns.items():
                    for pattern, description in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                        
                        for match in matches:
                            line_number = content[:match.start()].count('\n') + 1
                            
                            issue = {
                                "file": str(file_path.relative_to(Path.cwd())),
                                "line": line_number,
                                "pattern": pattern,
                                "description": description,
                                "severity": severity,
                                "context": content.split('\n')[line_number-1].strip() if line_number <= len(content.split('\n')) else ""
                            }
                            
                            security_report[f"{severity}_issues"].append(issue)
                            
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")
        
        # Calculate security score
        critical_count = len(security_report["critical_issues"])
        warning_count = len(security_report["warning_issues"])
        info_count = len(security_report["info_issues"])
        
        # Score calculation: 100 - (critical*10 + warning*3 + info*1)
        security_score = max(0, 100 - (critical_count * 10 + warning_count * 3 + info_count * 1))
        security_report["security_score"] = security_score
        
        # Summary
        security_report["summary"] = {
            "total_issues": critical_count + warning_count + info_count,
            "critical_issues": critical_count,
            "warning_issues": warning_count,
            "info_issues": info_count,
            "pass_threshold": 80,
            "passed": security_score >= 80
        }
        
        return security_report
    
    def validate_defensive_usage(self, code_paths: List[Path]) -> Dict[str, Any]:
        """Validate that code is used for defensive purposes only"""
        logger.info("Validating defensive usage patterns...")
        
        defensive_report = {
            "timestamp": datetime.now().isoformat(),
            "files_analyzed": 0,
            "defensive_indicators": [],
            "potential_concerns": [],
            "defensive_score": 0.0
        }
        
        # Defensive patterns (positive indicators)
        defensive_patterns = [
            (r"defensive|defense|protect|secure|monitor", "Defensive terminology"),
            (r"validation|sanitize|filter|escape", "Input validation"),
            (r"logging|audit|monitor|track", "Monitoring and logging"),
            (r"training|education|learn|practice", "Training purposes"),
            (r"simulation|simulate|mock|test", "Simulation environment"),
            (r"blue.*team|defender|analyst", "Blue team references"),
        ]
        
        # Concerning patterns (potential offensive usage)
        concerning_patterns = [
            (r"exploit|attack|payload.*execute", "Potential exploitation"),
            (r"backdoor|rootkit|malware.*deploy", "Malicious software deployment"),
            (r"penetration.*test.*real|pentest.*production", "Real penetration testing"),
            (r"red.*team.*actual|offensive.*operation", "Actual offensive operations"),
        ]
        
        total_defensive_indicators = 0
        total_concerns = 0
        
        for file_path in code_paths:
            if not file_path.suffix == '.py':
                continue
                
            defensive_report["files_analyzed"] += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                # Check defensive patterns
                file_defensive_count = 0
                for pattern, description in defensive_patterns:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    if matches > 0:
                        file_defensive_count += matches
                        total_defensive_indicators += matches
                
                # Check concerning patterns
                file_concern_count = 0
                for pattern, description in concerning_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        file_concern_count += 1
                        total_concerns += 1
                        
                        defensive_report["potential_concerns"].append({
                            "file": str(file_path.relative_to(Path.cwd())),
                            "line": line_number,
                            "pattern": pattern,
                            "description": description,
                            "context": content.split('\n')[line_number-1].strip()[:100]
                        })
                
                if file_defensive_count > 0:
                    defensive_report["defensive_indicators"].append({
                        "file": str(file_path.relative_to(Path.cwd())),
                        "defensive_count": file_defensive_count,
                        "concern_count": file_concern_count
                    })
                    
            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")
        
        # Calculate defensive score
        if total_defensive_indicators > 0:
            defensive_ratio = total_defensive_indicators / max(1, total_defensive_indicators + total_concerns)
            defensive_score = min(100, defensive_ratio * 100 + (total_defensive_indicators / 10))
        else:
            defensive_score = 50  # Neutral if no clear indicators
        
        defensive_report["defensive_score"] = defensive_score
        defensive_report["summary"] = {
            "defensive_indicators": total_defensive_indicators,
            "potential_concerns": total_concerns,
            "defensive_ratio": total_defensive_indicators / max(1, total_defensive_indicators + total_concerns),
            "pass_threshold": 70,
            "passed": defensive_score >= 70 and total_concerns == 0
        }
        
        return defensive_report


def collect_python_files(root_path: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """Collect all Python files in the project"""
    if exclude_patterns is None:
        exclude_patterns = [
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            "env"
        ]
    
    python_files = []
    
    for py_file in root_path.rglob("*.py"):
        # Check if file should be excluded
        exclude = False
        for pattern in exclude_patterns:
            if pattern in str(py_file):
                exclude = True
                break
        
        if not exclude:
            python_files.append(py_file)
    
    return python_files


def run_security_and_quality_gates():
    """Run comprehensive security and quality gates"""
    
    print("🔒 SECURITY AND QUALITY GATES v2")
    print("Comprehensive Platform Validation")
    print("=" * 50)
    
    start_time = time.time()
    project_root = Path.cwd()
    
    # Collect Python files
    print("\n📁 Collecting project files...")
    all_python_files = collect_python_files(project_root)
    test_files = [f for f in all_python_files if 'test' in f.name.lower()]
    source_files = [f for f in all_python_files if 'test' not in f.name.lower()]
    
    print(f"   Found {len(all_python_files)} Python files")
    print(f"   Source files: {len(source_files)}")
    print(f"   Test files: {len(test_files)}")
    
    # Initialize gates
    security_gate = SecurityGate()
    
    # Results tracking
    gate_results = {}
    overall_passed = True
    
    # Run Security Gates
    print(f"\n🛡️ SECURITY VALIDATION")
    print("-" * 30)
    
    # Security code analysis
    security_report = security_gate.validate_code_security(all_python_files)
    gate_results["security_analysis"] = security_report
    
    print(f"   Files scanned: {security_report['files_scanned']}")
    print(f"   Critical issues: {len(security_report['critical_issues'])}")
    print(f"   Warning issues: {len(security_report['warning_issues'])}")
    print(f"   Info issues: {len(security_report['info_issues'])}")
    print(f"   Security score: {security_report['security_score']:.1f}/100")
    
    if security_report["summary"]["passed"]:
        print(f"   ✅ Security gate PASSED")
    else:
        print(f"   ❌ Security gate FAILED")
        overall_passed = False
        
        # Show critical issues
        if security_report["critical_issues"]:
            print(f"   Critical issues found:")
            for issue in security_report["critical_issues"][:3]:
                print(f"     • {issue['file']}:{issue['line']} - {issue['description']}")
    
    # Defensive usage validation
    defensive_report = security_gate.validate_defensive_usage(all_python_files)
    gate_results["defensive_usage"] = defensive_report
    
    print(f"   Defensive indicators: {defensive_report['summary']['defensive_indicators']}")
    print(f"   Potential concerns: {defensive_report['summary']['potential_concerns']}")
    print(f"   Defensive score: {defensive_report['defensive_score']:.1f}/100")
    
    if defensive_report["summary"]["passed"]:
        print(f"   ✅ Defensive usage gate PASSED")
    else:
        print(f"   ❌ Defensive usage gate FAILED")
        overall_passed = False
        
        # Show concerns
        if defensive_report["potential_concerns"]:
            print(f"   Potential concerns:")
            for concern in defensive_report["potential_concerns"][:3]:
                print(f"     • {concern['file']}:{concern['line']} - {concern['description']}")
    
    # Basic quality checks
    print(f"\n📊 QUALITY VALIDATION")
    print("-" * 30)
    
    total_lines = 0
    large_files = 0
    
    for file_path in source_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_lines += lines
                if lines > 1000:
                    large_files += 1
        except:
            pass
    
    avg_lines_per_file = total_lines / len(source_files) if source_files else 0
    
    print(f"   Total source lines: {total_lines}")
    print(f"   Average lines per file: {avg_lines_per_file:.1f}")
    print(f"   Large files (>1000 lines): {large_files}")
    print(f"   Test files present: {len(test_files)}")
    
    # Calculate overall scores
    execution_time = time.time() - start_time
    
    # Security gates (must pass)
    security_gates_passed = (
        security_report["summary"]["passed"] and
        defensive_report["summary"]["passed"]
    )
    
    # Overall assessment
    print(f"\n🎯 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total execution time: {execution_time:.2f}s")
    print(f"Files analyzed: {len(all_python_files)}")
    
    print(f"\n🛡️  SECURITY GATES")
    security_status = "✅ PASSED" if security_gates_passed else "❌ FAILED"
    print(f"   Overall: {security_status}")
    print(f"   Security Score: {security_report['security_score']:.1f}/100")
    print(f"   Defensive Score: {defensive_report['defensive_score']:.1f}/100")
    
    print(f"\n📊 CODE METRICS")
    print(f"   Total Lines: {total_lines}")
    print(f"   Source Files: {len(source_files)}")
    print(f"   Test Files: {len(test_files)}")
    print(f"   Large Files: {large_files}")
    
    # Generate comprehensive report
    comprehensive_report = {
        "timestamp": datetime.now().isoformat(),
        "execution_time": execution_time,
        "project_stats": {
            "total_files": len(all_python_files),
            "source_files": len(source_files),
            "test_files": len(test_files),
            "total_lines": total_lines,
            "avg_lines_per_file": avg_lines_per_file,
            "large_files": large_files
        },
        "security_gates": {
            "overall_passed": security_gates_passed,
            "security_analysis": security_report,
            "defensive_usage": defensive_report
        },
        "overall_result": {
            "security_passed": security_gates_passed,
            "deployment_approved": security_gates_passed,
            "recommendation": "APPROVED FOR DEPLOYMENT" if security_gates_passed else "SECURITY ISSUES MUST BE RESOLVED"
        }
    }
    
    # Save report
    report_file = Path("security_quality_report_v2.json")
    with open(report_file, "w") as f:
        json.dump(comprehensive_report, f, indent=2)
    
    print(f"\n📝 Detailed report saved to: {report_file}")
    
    # Final verdict
    if security_gates_passed:
        print(f"\n🏆 OVERALL RESULT: APPROVED FOR DEPLOYMENT")
        print(f"   Security gates passed - Platform is secure for defensive use")
        print(f"   Code meets security standards for cybersecurity training")
        return 0
    else:
        print(f"\n🚫 OVERALL RESULT: DEPLOYMENT BLOCKED")
        print(f"   Security issues must be resolved before deployment")
        return 1


if __name__ == "__main__":
    exit_code = run_security_and_quality_gates()
    sys.exit(exit_code)