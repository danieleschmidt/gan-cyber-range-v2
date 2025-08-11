"""
Security Scanner for GAN-Cyber-Range-v2

Comprehensive security assessment tool that scans for:
- Code vulnerabilities
- Configuration issues  
- Input validation weaknesses
- Dependency vulnerabilities
- Security best practices compliance
"""

import os
import sys
import re
import json
import ast
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass 
class SecurityIssue:
    severity: SecurityLevel
    category: str
    description: str
    file_path: str
    line_number: int
    recommendation: str
    cwe_id: Optional[str] = None


@dataclass
class ScanResults:
    total_files_scanned: int = 0
    issues_found: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    issues: List[SecurityIssue] = field(default_factory=list)
    scan_duration: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SecurityScanner:
    """Comprehensive security scanner"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.results = ScanResults()
        
        # Security patterns to detect
        self.vulnerability_patterns = {
            "sql_injection": {
                "patterns": [
                    r'execute\s*\(\s*["\'].*%.*["\']',
                    r'cursor\.execute\s*\(\s*["\'].*\+.*["\']',
                    r'query\s*=\s*["\'].*%.*["\']',
                    r'f["\']SELECT.*\{.*\}.*["\']'
                ],
                "severity": SecurityLevel.HIGH,
                "cwe": "CWE-89"
            },
            "command_injection": {
                "patterns": [
                    r'os\.system\s*\(\s*.*\+',
                    r'subprocess\.(call|run|Popen)\s*\(\s*.*\+',
                    r'eval\s*\(',
                    r'exec\s*\('
                ],
                "severity": SecurityLevel.CRITICAL,
                "cwe": "CWE-78"
            },
            "path_traversal": {
                "patterns": [
                    r'open\s*\(\s*.*\+.*["\']\.\.\/["\']',
                    r'os\.path\.join\s*\(\s*.*user.*input',
                    r'\.\.\/.*\.\.\/.*\.\.\/',
                    r'%2e%2e%2f'
                ],
                "severity": SecurityLevel.HIGH,
                "cwe": "CWE-22"
            },
            "hardcoded_secrets": {
                "patterns": [
                    r'password\s*=\s*["\'][^"\']{8,}["\']',
                    r'api_key\s*=\s*["\'][^"\']{16,}["\']',
                    r'secret_key\s*=\s*["\'][^"\']{16,}["\']',
                    r'token\s*=\s*["\'][A-Za-z0-9+/]{20,}["\']'
                ],
                "severity": SecurityLevel.CRITICAL,
                "cwe": "CWE-798"
            },
            "weak_crypto": {
                "patterns": [
                    r'hashlib\.md5\(',
                    r'hashlib\.sha1\(',
                    r'Crypto\.Hash\.MD5',
                    r'DES\.',
                    r'RC4\.'
                ],
                "severity": SecurityLevel.MEDIUM,
                "cwe": "CWE-327"
            },
            "insecure_random": {
                "patterns": [
                    r'random\.random\(',
                    r'random\.randint\(',
                    r'random\.choice\('
                ],
                "severity": SecurityLevel.LOW,
                "cwe": "CWE-330"
            },
            "debug_enabled": {
                "patterns": [
                    r'debug\s*=\s*True',
                    r'DEBUG\s*=\s*True',
                    r'app\.run\s*\(.*debug\s*=\s*True',
                ],
                "severity": SecurityLevel.MEDIUM,
                "cwe": "CWE-489"
            },
            "unsafe_deserialization": {
                "patterns": [
                    r'pickle\.loads\(',
                    r'cPickle\.loads\(',
                    r'yaml\.load\([^,)]*\)',
                    r'marshal\.loads\('
                ],
                "severity": SecurityLevel.HIGH,
                "cwe": "CWE-502"
            }
        }
        
        # File types to scan
        self.scannable_extensions = {'.py', '.yaml', '.yml', '.json', '.txt', '.md'}
        
        # Files/directories to skip
        self.skip_patterns = {
            '__pycache__', '.git', '.pytest_cache', '.mypy_cache',
            'node_modules', 'venv', 'env', '.env'
        }
    
    def scan_project(self) -> ScanResults:
        """Perform comprehensive security scan"""
        start_time = datetime.now()
        
        print("🔍 Starting security scan...")
        print(f"📁 Scanning project: {self.project_root}")
        
        # Scan all files
        for file_path in self._get_scannable_files():
            self._scan_file(file_path)
            self.results.total_files_scanned += 1
            
            if self.results.total_files_scanned % 10 == 0:
                print(f"📄 Scanned {self.results.total_files_scanned} files...")
        
        # Additional scans
        self._scan_dependencies()
        self._scan_configuration()
        self._scan_permissions()
        
        # Calculate metrics
        self._calculate_metrics()
        
        self.results.scan_duration = (datetime.now() - start_time).total_seconds()
        
        return self.results
    
    def _get_scannable_files(self) -> List[Path]:
        """Get list of files to scan"""
        files = []
        
        for root, dirs, filenames in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in self.skip_patterns]
            
            for filename in filenames:
                file_path = Path(root) / filename
                
                # Check extension
                if file_path.suffix.lower() in self.scannable_extensions:
                    files.append(file_path)
        
        return files
    
    def _scan_file(self, file_path: Path) -> None:
        """Scan individual file for vulnerabilities"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Check vulnerability patterns
            for vuln_type, vuln_data in self.vulnerability_patterns.items():
                for pattern in vuln_data['patterns']:
                    for line_num, line in enumerate(lines, 1):
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            issue = SecurityIssue(
                                severity=vuln_data['severity'],
                                category=vuln_type,
                                description=f"Potential {vuln_type.replace('_', ' ')} vulnerability detected",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                recommendation=self._get_recommendation(vuln_type),
                                cwe_id=vuln_data.get('cwe')
                            )
                            self.results.issues.append(issue)
            
            # Additional file-specific checks
            if file_path.suffix == '.py':
                self._scan_python_specific(file_path, content)
            elif file_path.suffix in {'.yaml', '.yml'}:
                self._scan_yaml_specific(file_path, content)
            elif file_path.suffix == '.json':
                self._scan_json_specific(file_path, content)
                
        except Exception as e:
            print(f"⚠️ Error scanning {file_path}: {e}")
    
    def _scan_python_specific(self, file_path: Path, content: str) -> None:
        """Python-specific security checks"""
        try:
            tree = ast.parse(content)
            
            class SecurityVisitor(ast.NodeVisitor):
                def __init__(self, scanner, file_path):
                    self.scanner = scanner
                    self.file_path = file_path
                
                def visit_Import(self, node):
                    # Check for dangerous imports
                    dangerous_modules = {'subprocess', 'os', 'eval', 'exec'}
                    for alias in node.names:
                        if alias.name in dangerous_modules:
                            issue = SecurityIssue(
                                severity=SecurityLevel.LOW,
                                category="dangerous_import",
                                description=f"Import of potentially dangerous module: {alias.name}",
                                file_path=str(self.file_path.relative_to(self.scanner.project_root)),
                                line_number=node.lineno,
                                recommendation="Ensure proper input validation when using this module",
                                cwe_id="CWE-676"
                            )
                            self.scanner.results.issues.append(issue)
                    self.generic_visit(node)
                
                def visit_Call(self, node):
                    # Check for unsafe function calls
                    if isinstance(node.func, ast.Name):
                        if node.func.id in {'eval', 'exec', 'compile'}:
                            issue = SecurityIssue(
                                severity=SecurityLevel.CRITICAL,
                                category="code_injection",
                                description=f"Use of dangerous function: {node.func.id}",
                                file_path=str(self.file_path.relative_to(self.scanner.project_root)),
                                line_number=node.lineno,
                                recommendation="Avoid using eval/exec with user input",
                                cwe_id="CWE-95"
                            )
                            self.scanner.results.issues.append(issue)
                    self.generic_visit(node)
            
            visitor = SecurityVisitor(self, file_path)
            visitor.visit(tree)
            
        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            print(f"⚠️ Error in Python AST analysis for {file_path}: {e}")
    
    def _scan_yaml_specific(self, file_path: Path, content: str) -> None:
        """YAML-specific security checks"""
        # Check for unsafe YAML patterns
        unsafe_patterns = [
            r'!!python/object',
            r'!!python/apply',
            r'load_all\s*\(',
            r'unsafe_load\s*\('
        ]
        
        lines = content.splitlines()
        for line_num, line in enumerate(lines, 1):
            for pattern in unsafe_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issue = SecurityIssue(
                        severity=SecurityLevel.HIGH,
                        category="yaml_injection",
                        description="Unsafe YAML loading pattern detected",
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_num,
                        recommendation="Use safe_load() instead of load() for YAML parsing",
                        cwe_id="CWE-502"
                    )
                    self.results.issues.append(issue)
    
    def _scan_json_specific(self, file_path: Path, content: str) -> None:
        """JSON-specific security checks"""
        try:
            data = json.loads(content)
            
            # Check for potential secrets in JSON
            def check_secrets(obj, path=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        
                        # Check for secret-like keys
                        secret_keys = ['password', 'secret', 'token', 'key', 'api_key']
                        if any(secret_word in key.lower() for secret_word in secret_keys):
                            if isinstance(value, str) and len(value) > 8:
                                issue = SecurityIssue(
                                    severity=SecurityLevel.MEDIUM,
                                    category="hardcoded_secret",
                                    description=f"Potential hardcoded secret in JSON: {current_path}",
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=1,
                                    recommendation="Use environment variables or secure secret management",
                                    cwe_id="CWE-798"
                                )
                                self.results.issues.append(issue)
                        
                        if isinstance(value, (dict, list)):
                            check_secrets(value, current_path)
                
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, (dict, list)):
                            check_secrets(item, f"{path}[{i}]")
            
            check_secrets(data)
            
        except json.JSONDecodeError:
            pass
    
    def _scan_dependencies(self) -> None:
        """Scan dependencies for known vulnerabilities"""
        requirements_files = [
            self.project_root / 'requirements.txt',
            self.project_root / 'Pipfile',
            self.project_root / 'setup.py'
        ]
        
        known_vulnerable = {
            'pyyaml': ['<5.4.0', 'CWE-502'],
            'requests': ['<2.20.0', 'CWE-295'],
            'urllib3': ['<1.24.2', 'CWE-295'],
            'flask': ['<1.0', 'CWE-352'],
            'django': ['<2.2.13', 'CWE-352']
        }
        
        for req_file in requirements_files:
            if req_file.exists():
                try:
                    with open(req_file, 'r') as f:
                        content = f.read()
                    
                    for package, (version, cwe) in known_vulnerable.items():
                        if package in content.lower():
                            issue = SecurityIssue(
                                severity=SecurityLevel.MEDIUM,
                                category="vulnerable_dependency",
                                description=f"Potentially vulnerable dependency: {package}",
                                file_path=str(req_file.relative_to(self.project_root)),
                                line_number=1,
                                recommendation=f"Update {package} to version {version} or later",
                                cwe_id=cwe
                            )
                            self.results.issues.append(issue)
                            
                except Exception as e:
                    print(f"⚠️ Error scanning {req_file}: {e}")
    
    def _scan_configuration(self) -> None:
        """Scan configuration files for security issues"""
        config_files = [
            self.project_root / '.env',
            self.project_root / 'config.py',
            self.project_root / 'settings.py'
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                    
                    # Check for debug mode in production
                    if re.search(r'DEBUG\s*=\s*True', content):
                        issue = SecurityIssue(
                            severity=SecurityLevel.MEDIUM,
                            category="debug_mode",
                            description="Debug mode enabled in configuration",
                            file_path=str(config_file.relative_to(self.project_root)),
                            line_number=1,
                            recommendation="Disable debug mode in production",
                            cwe_id="CWE-489"
                        )
                        self.results.issues.append(issue)
                    
                    # Check for default secrets
                    default_secrets = ['changeme', 'admin', 'password', '123456']
                    for secret in default_secrets:
                        if secret in content.lower():
                            issue = SecurityIssue(
                                severity=SecurityLevel.HIGH,
                                category="default_credentials",
                                description=f"Default or weak credential detected: {secret}",
                                file_path=str(config_file.relative_to(self.project_root)),
                                line_number=1,
                                recommendation="Use strong, unique credentials",
                                cwe_id="CWE-521"
                            )
                            self.results.issues.append(issue)
                            
                except Exception as e:
                    print(f"⚠️ Error scanning {config_file}: {e}")
    
    def _scan_permissions(self) -> None:
        """Check file permissions for security issues"""
        sensitive_files = [
            'private_key', 'id_rsa', '.env', 'secrets.yaml',
            'config.py', 'settings.py'
        ]
        
        for root, dirs, files in os.walk(self.project_root):
            for filename in files:
                file_path = Path(root) / filename
                
                # Check if it's a sensitive file
                is_sensitive = any(sensitive in filename.lower() for sensitive in sensitive_files)
                
                if is_sensitive:
                    try:
                        stat_info = file_path.stat()
                        permissions = oct(stat_info.st_mode)[-3:]
                        
                        # Check if world-readable (permissions ending in 4, 5, 6, 7)
                        if permissions[-1] in '4567':
                            issue = SecurityIssue(
                                severity=SecurityLevel.MEDIUM,
                                category="file_permissions",
                                description=f"Sensitive file is world-readable: {permissions}",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=1,
                                recommendation="Restrict file permissions to owner only (600)",
                                cwe_id="CWE-732"
                            )
                            self.results.issues.append(issue)
                    except Exception:
                        pass
    
    def _get_recommendation(self, vuln_type: str) -> str:
        """Get security recommendation for vulnerability type"""
        recommendations = {
            "sql_injection": "Use parameterized queries or ORM instead of string concatenation",
            "command_injection": "Avoid using user input in system commands, use subprocess with shell=False",
            "path_traversal": "Validate and sanitize file paths, use os.path.abspath()",
            "hardcoded_secrets": "Use environment variables or secure secret management systems",
            "weak_crypto": "Use strong cryptographic algorithms like SHA-256 or better",
            "insecure_random": "Use secrets module for cryptographic purposes",
            "debug_enabled": "Disable debug mode in production environments",
            "unsafe_deserialization": "Use safe serialization formats like JSON, avoid pickle with untrusted data"
        }
        return recommendations.get(vuln_type, "Review and fix the security issue")
    
    def _calculate_metrics(self) -> None:
        """Calculate security metrics"""
        self.results.issues_found = len(self.results.issues)
        
        for issue in self.results.issues:
            if issue.severity == SecurityLevel.CRITICAL:
                self.results.critical_issues += 1
            elif issue.severity == SecurityLevel.HIGH:
                self.results.high_issues += 1
            elif issue.severity == SecurityLevel.MEDIUM:
                self.results.medium_issues += 1
            elif issue.severity == SecurityLevel.LOW:
                self.results.low_issues += 1
    
    def generate_report(self, output_file: str = "security_report.json") -> None:
        """Generate security scan report"""
        report_data = {
            "scan_summary": {
                "timestamp": self.results.timestamp,
                "scan_duration": self.results.scan_duration,
                "files_scanned": self.results.total_files_scanned,
                "issues_found": self.results.issues_found,
                "critical_issues": self.results.critical_issues,
                "high_issues": self.results.high_issues,
                "medium_issues": self.results.medium_issues,
                "low_issues": self.results.low_issues
            },
            "security_score": self._calculate_security_score(),
            "issues": []
        }
        
        # Add issues to report
        for issue in self.results.issues:
            report_data["issues"].append({
                "severity": issue.severity.value,
                "category": issue.category,
                "description": issue.description,
                "file": issue.file_path,
                "line": issue.line_number,
                "recommendation": issue.recommendation,
                "cwe_id": issue.cwe_id
            })
        
        # Sort issues by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        report_data["issues"].sort(key=lambda x: severity_order.get(x["severity"], 4))
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Security report saved to: {output_file}")
    
    def _calculate_security_score(self) -> int:
        """Calculate overall security score (0-100)"""
        if self.results.issues_found == 0:
            return 100
        
        # Weight by severity
        penalty = (
            self.results.critical_issues * 20 +
            self.results.high_issues * 10 +
            self.results.medium_issues * 5 +
            self.results.low_issues * 1
        )
        
        # Base score minus penalties
        score = max(0, 100 - penalty)
        return score
    
    def print_summary(self) -> None:
        """Print scan summary"""
        print("\n" + "="*60)
        print("🛡️ SECURITY SCAN RESULTS")
        print("="*60)
        print(f"📁 Files scanned: {self.results.total_files_scanned}")
        print(f"⏱️ Scan duration: {self.results.scan_duration:.2f}s")
        print(f"🔍 Issues found: {self.results.issues_found}")
        print(f"📊 Security Score: {self._calculate_security_score()}/100")
        print()
        print("Issue Breakdown:")
        print(f"  🔥 Critical: {self.results.critical_issues}")
        print(f"  ⚠️  High:     {self.results.high_issues}")
        print(f"  📋 Medium:   {self.results.medium_issues}")
        print(f"  ℹ️  Low:      {self.results.low_issues}")
        
        if self.results.issues_found > 0:
            print("\nTop Issues:")
            # Show top 5 most severe issues
            sorted_issues = sorted(self.results.issues, 
                                 key=lambda x: ["critical", "high", "medium", "low"].index(x.severity.value))
            
            for issue in sorted_issues[:5]:
                severity_icon = {"critical": "🔥", "high": "⚠️", "medium": "📋", "low": "ℹ️"}
                print(f"  {severity_icon[issue.severity.value]} {issue.category} in {issue.file_path}:{issue.line_number}")
                print(f"     {issue.description}")
        
        print("="*60)


def main():
    """Main security scanner execution"""
    scanner = SecurityScanner()
    
    print("🛡️ GAN-Cyber-Range-v2 Security Scanner")
    print("="*60)
    
    # Perform scan
    results = scanner.scan_project()
    
    # Print summary
    scanner.print_summary()
    
    # Generate detailed report
    scanner.generate_report("security_scan_report.json")
    
    # Return appropriate exit code
    if results.critical_issues > 0 or results.high_issues > 5:
        print("\n❌ Security scan failed due to critical or high-risk issues")
        return 1
    elif results.issues_found == 0:
        print("\n✅ No security issues found!")
        return 0
    else:
        print(f"\n⚠️ Security scan completed with {results.issues_found} issues")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)