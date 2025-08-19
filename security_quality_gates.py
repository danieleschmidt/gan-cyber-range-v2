#!/usr/bin/env python3
"""
Security and Quality Gates for GAN-Cyber-Range-v2.
Comprehensive security scanning and code quality validation.
"""

import os
import sys
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

def scan_for_secrets():
    """Scan codebase for potential secrets and sensitive data"""
    print("🔍 Scanning for secrets and sensitive data...")
    
    secret_patterns = {
        'api_key': r'(?i)(api[_-]?key|apikey)[\'"\s]*[:=][\'"\s]*[a-zA-Z0-9_-]{20,}',
        'password': r'(?i)(password|passwd|pwd)[\'"\s]*[:=][\'"\s]*[\'"][^\'"\s]{8,}[\'"]',
        'token': r'(?i)(token|auth)[\'"\s]*[:=][\'"\s]*[a-zA-Z0-9_-]{20,}',
        'secret': r'(?i)(secret)[\'"\s]*[:=][\'"\s]*[a-zA-Z0-9_-]{20,}',
        'private_key': r'-----BEGIN [A-Z ]+PRIVATE KEY-----',
        'aws_access': r'AKIA[0-9A-Z]{16}',
        'github_token': r'ghp_[a-zA-Z0-9]{36}',
        'slack_token': r'xox[baprs]-[a-zA-Z0-9-]{8,}',
    }
    
    issues = []
    excluded_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'venv', '.venv'}
    
    for root, dirs, files in os.walk('.'):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        for file in files:
            if file.endswith(('.py', '.js', '.ts', '.yaml', '.yml', '.json', '.env', '.conf')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    for pattern_name, pattern in secret_patterns.items():
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            issues.append({
                                'type': 'secret',
                                'pattern': pattern_name,
                                'file': filepath,
                                'line': line_num,
                                'severity': 'HIGH',
                                'match': match.group()[:50] + '...' if len(match.group()) > 50 else match.group()
                            })
                except Exception as e:
                    pass  # Skip files that can't be read
    
    print(f"   Found {len(issues)} potential secret exposures")
    return issues


def scan_security_vulnerabilities():
    """Scan for common security vulnerabilities"""
    print("🔐 Scanning for security vulnerabilities...")
    
    vulnerability_patterns = {
        'sql_injection': r'(SELECT|INSERT|UPDATE|DELETE|DROP)\s+.*\+.*[\'"]',
        'command_injection': r'(os\.system|subprocess\.call|exec|eval)\s*\([^)]*\+',
        'path_traversal': r'(open|file)\s*\([^)]*\.\./.*\)',
        'hardcoded_password': r'(?i)(password|passwd)\s*=\s*[\'"][^\'"\s]{1,50}[\'"]',
        'weak_crypto': r'(md5|sha1|des|rc4)[\(\s]',
        'debug_mode': r'(?i)(debug\s*=\s*true|app\.debug\s*=\s*true)',
        'unsafe_pickle': r'(pickle\.loads?|cPickle\.loads?)\s*\(',
        'dangerous_eval': r'(eval|exec)\s*\([^)]*input',
    }
    
    issues = []
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for vuln_name, pattern in vulnerability_patterns.items():
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            issues.append({
                                'type': 'vulnerability',
                                'pattern': vuln_name,
                                'file': filepath,
                                'line': line_num,
                                'severity': 'MEDIUM',
                                'match': match.group().strip()
                            })
                except Exception as e:
                    pass
    
    print(f"   Found {len(issues)} potential vulnerabilities")
    return issues


def check_code_quality():
    """Check basic code quality metrics"""
    print("📊 Checking code quality...")
    
    issues = []
    metrics = {
        'total_files': 0,
        'total_lines': 0,
        'avg_complexity': 0,
        'long_functions': 0,
        'large_files': 0
    }
    
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    metrics['total_files'] += 1
                    file_lines = len(lines)
                    metrics['total_lines'] += file_lines
                    
                    # Check for large files
                    if file_lines > 1000:
                        metrics['large_files'] += 1
                        issues.append({
                            'type': 'quality',
                            'pattern': 'large_file',
                            'file': filepath,
                            'severity': 'LOW',
                            'message': f'File has {file_lines} lines (>1000)'
                        })
                    
                    # Check for long functions
                    in_function = False
                    function_start = 0
                    function_name = ''
                    
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line.startswith('def ') or line.startswith('async def '):
                            if in_function:
                                # Previous function ended
                                func_length = i - function_start
                                if func_length > 50:
                                    metrics['long_functions'] += 1
                                    issues.append({
                                        'type': 'quality',
                                        'pattern': 'long_function',
                                        'file': filepath,
                                        'line': function_start + 1,
                                        'severity': 'LOW',
                                        'message': f'Function {function_name} has {func_length} lines (>50)'
                                    })
                            
                            in_function = True
                            function_start = i
                            function_name = line.split('(')[0].replace('def ', '').replace('async ', '').strip()
                        elif line.startswith('class ') and in_function:
                            # Function ended at class definition
                            func_length = i - function_start
                            if func_length > 50:
                                metrics['long_functions'] += 1
                                issues.append({
                                    'type': 'quality',
                                    'pattern': 'long_function',
                                    'file': filepath,
                                    'line': function_start + 1,
                                    'severity': 'LOW',
                                    'message': f'Function {function_name} has {func_length} lines (>50)'
                                })
                            in_function = False
                
                except Exception as e:
                    pass
    
    # Calculate averages
    if metrics['total_files'] > 0:
        metrics['avg_lines_per_file'] = metrics['total_lines'] / metrics['total_files']
    
    print(f"   Analyzed {metrics['total_files']} Python files")
    print(f"   Total lines: {metrics['total_lines']}")
    print(f"   Average lines per file: {metrics.get('avg_lines_per_file', 0):.1f}")
    print(f"   Quality issues found: {len(issues)}")
    
    return issues, metrics


def check_dependencies():
    """Check for known vulnerable dependencies"""
    print("📦 Checking dependencies for known vulnerabilities...")
    
    issues = []
    
    # Check requirements.txt
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        # Known vulnerable packages (simplified check)
        vulnerable_patterns = {
            'pillow': r'pillow\s*[<>=]*\s*[0-8]\.', # Old versions
            'django': r'django\s*[<>=]*\s*[0-2]\.', # Very old versions
            'flask': r'flask\s*[<>=]*\s*[0-1]\.', # Very old versions
            'requests': r'requests\s*[<>=]*\s*[0-2]\.19\.', # CVE in old versions
        }
        
        for vuln_name, pattern in vulnerable_patterns.items():
            if re.search(pattern, requirements, re.IGNORECASE):
                issues.append({
                    'type': 'dependency',
                    'pattern': vuln_name,
                    'file': 'requirements.txt',
                    'severity': 'MEDIUM',
                    'message': f'Potentially vulnerable version of {vuln_name}'
                })
    
    print(f"   Found {len(issues)} dependency issues")
    return issues


def check_configuration_security():
    """Check configuration files for security issues"""
    print("⚙️  Checking configuration security...")
    
    issues = []
    config_files = []
    
    # Find configuration files
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.endswith(('.yml', '.yaml', '.json', '.conf', '.cfg', '.ini', '.env')):
                config_files.append(os.path.join(root, file))
    
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for insecure configurations
            insecure_patterns = {
                'debug_enabled': r'(?i)(debug|DEBUG)\s*[:=]\s*(true|True|1|yes)',
                'weak_ssl': r'(?i)(ssl_verify|verify_ssl)\s*[:=]\s*(false|False|0|no)',
                'wildcard_cors': r'(?i)(cors|access-control-allow-origin)\s*[:=]\s*[\'"\*\'"]*',
                'exposed_port': r'(?i)(port|PORT)\s*[:=]\s*(22|23|21|135|139|445|1433|3389)',
                'default_password': r'(?i)(password|passwd)\s*[:=]\s*[\'"]?(admin|password|123456|root)[\'"]?',
            }
            
            for pattern_name, pattern in insecure_patterns.items():
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'type': 'configuration',
                        'pattern': pattern_name,
                        'file': config_file,
                        'line': line_num,
                        'severity': 'MEDIUM',
                        'match': match.group().strip()
                    })
        
        except Exception as e:
            pass
    
    print(f"   Checked {len(config_files)} configuration files")
    print(f"   Found {len(issues)} configuration issues")
    
    return issues


def generate_security_report(all_issues):
    """Generate comprehensive security report"""
    
    # Categorize issues by severity
    critical = [i for i in all_issues if i.get('severity') == 'CRITICAL']
    high = [i for i in all_issues if i.get('severity') == 'HIGH'] 
    medium = [i for i in all_issues if i.get('severity') == 'MEDIUM']
    low = [i for i in all_issues if i.get('severity') == 'LOW']
    
    # Categorize by type
    by_type = {}
    for issue in all_issues:
        issue_type = issue.get('type', 'unknown')
        if issue_type not in by_type:
            by_type[issue_type] = []
        by_type[issue_type].append(issue)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_issues': len(all_issues),
            'critical': len(critical),
            'high': len(high),
            'medium': len(medium),
            'low': len(low)
        },
        'by_type': {k: len(v) for k, v in by_type.items()},
        'issues': all_issues
    }
    
    # Security score calculation
    score = 100
    score -= len(critical) * 20  # Critical issues: -20 points each
    score -= len(high) * 10      # High issues: -10 points each  
    score -= len(medium) * 5     # Medium issues: -5 points each
    score -= len(low) * 1        # Low issues: -1 point each
    score = max(0, score)        # Don't go below 0
    
    report['security_score'] = score
    
    return report


def main():
    """Main security and quality gate runner"""
    print("🛡️  GAN-Cyber-Range-v2 - Security & Quality Gates")
    print("="*65)
    
    all_issues = []
    
    # Run all security checks
    all_issues.extend(scan_for_secrets())
    all_issues.extend(scan_security_vulnerabilities())
    all_issues.extend(check_dependencies())
    all_issues.extend(check_configuration_security())
    
    # Run quality checks
    quality_issues, metrics = check_code_quality()
    all_issues.extend(quality_issues)
    
    # Generate report
    report = generate_security_report(all_issues)
    
    # Display results
    print("\n" + "="*65)
    print("🔍 SECURITY & QUALITY ANALYSIS RESULTS")
    print("="*65)
    
    print(f"\n📊 Summary:")
    print(f"   Total Issues: {report['summary']['total_issues']}")
    print(f"   🔴 Critical: {report['summary']['critical']}")
    print(f"   🟠 High: {report['summary']['high']}")
    print(f"   🟡 Medium: {report['summary']['medium']}")
    print(f"   🟢 Low: {report['summary']['low']}")
    print(f"   🏆 Security Score: {report['security_score']}/100")
    
    # Issues by type
    if report['by_type']:
        print(f"\n📋 Issues by Type:")
        for issue_type, count in report['by_type'].items():
            print(f"   {issue_type.title()}: {count}")
    
    # Show critical and high severity issues
    critical_high = [i for i in all_issues if i.get('severity') in ['CRITICAL', 'HIGH']]
    if critical_high:
        print(f"\n🚨 Critical & High Severity Issues:")
        for issue in critical_high[:10]:  # Show first 10
            file_path = issue.get('file', 'unknown')
            line = issue.get('line', '')
            pattern = issue.get('pattern', issue.get('type', 'unknown'))
            message = issue.get('message', issue.get('match', ''))
            
            line_info = f":{line}" if line else ""
            print(f"   {issue['severity']} - {pattern}")
            print(f"     {file_path}{line_info}")
            print(f"     {message}")
    
    # Quality metrics
    print(f"\n📈 Code Quality Metrics:")
    print(f"   Files Analyzed: {metrics['total_files']}")
    print(f"   Total Lines: {metrics['total_lines']}")
    print(f"   Avg Lines/File: {metrics.get('avg_lines_per_file', 0):.1f}")
    print(f"   Large Files (>1000 lines): {metrics['large_files']}")
    print(f"   Long Functions (>50 lines): {metrics['long_functions']}")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    
    if report['security_score'] >= 90:
        status = "✅ EXCELLENT"
        recommendation = "System is secure and ready for production"
    elif report['security_score'] >= 75:
        status = "🟢 GOOD"
        recommendation = "Minor issues detected, safe for production with monitoring"
    elif report['security_score'] >= 60:
        status = "🟡 ACCEPTABLE"
        recommendation = "Some issues found, recommend addressing before production"
    elif report['security_score'] >= 40:
        status = "🟠 CONCERNING"
        recommendation = "Multiple issues found, security review required"
    else:
        status = "🔴 CRITICAL"
        recommendation = "Significant security issues, immediate attention required"
    
    print(f"   Security Status: {status}")
    print(f"   Recommendation: {recommendation}")
    
    # Save detailed report
    with open('security_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to security_report.json")
    print("="*65)
    
    # Return success if score is acceptable
    return report['security_score'] >= 60


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nSecurity scan interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nSecurity scan failed: {e}")
        sys.exit(1)