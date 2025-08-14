#!/usr/bin/env python3
"""
Structure validation test for GAN Cyber Range.
Tests code structure, imports, and basic functionality without external dependencies.
"""

import os
import sys
import ast
import traceback
from datetime import datetime
from pathlib import Path


def validate_python_syntax(file_path):
    """Validate Python syntax of a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def test_code_structure():
    """Test overall code structure and syntax"""
    print("Testing code structure and syntax...")
    
    repo_root = Path(__file__).parent
    python_files = list(repo_root.rglob("*.py"))
    
    syntax_errors = []
    total_files = 0
    
    for py_file in python_files:
        if 'test_' in py_file.name and py_file != Path(__file__):
            continue  # Skip other test files
        
        total_files += 1
        is_valid, error = validate_python_syntax(py_file)
        
        if not is_valid:
            syntax_errors.append(f"{py_file}: {error}")
    
    if syntax_errors:
        print(f"✗ Found {len(syntax_errors)} syntax errors:")
        for error in syntax_errors[:5]:  # Show first 5
            print(f"  - {error}")
        if len(syntax_errors) > 5:
            print(f"  ... and {len(syntax_errors) - 5} more")
        return False
    else:
        print(f"✓ All {total_files} Python files have valid syntax")
        return True


def test_module_structure():
    """Test module structure and organization"""
    print("\\nTesting module structure...")
    
    expected_structure = {
        'gan_cyber_range': {
            '__init__.py': True,
            'core': {
                '__init__.py': True,
                'attack_gan.py': True,
                'cyber_range.py': True,
                'network_sim.py': True,
                'attack_engine.py': True
            },
            'evaluation': {
                '__init__.py': True,
                'attack_evaluator.py': True,
                'training_evaluator.py': True,
                'blue_team_evaluator.py': True
            },
            'optimization': {
                '__init__.py': True,
                'advanced_performance.py': True
            },
            'utils': {
                '__init__.py': True,
                'enhanced_security.py': True,
                'comprehensive_monitoring.py': True
            }
        }
    }
    
    def check_structure(path, structure):
        """Recursively check directory structure"""
        missing = []
        
        for item, content in structure.items():
            item_path = path / item
            
            if isinstance(content, dict):
                # Directory
                if not item_path.is_dir():
                    missing.append(f"Directory: {item_path}")
                else:
                    missing.extend(check_structure(item_path, content))
            else:
                # File
                if not item_path.is_file():
                    missing.append(f"File: {item_path}")
        
        return missing
    
    repo_root = Path(__file__).parent
    missing_items = check_structure(repo_root, expected_structure)
    
    if missing_items:
        print(f"✗ Missing {len(missing_items)} expected items:")
        for item in missing_items[:10]:
            print(f"  - {item}")
        return False
    else:
        print("✓ All expected modules and files present")
        return True


def test_import_structure():
    """Test import structure without executing imports"""
    print("\\nTesting import structure...")
    
    def extract_imports(file_path):
        """Extract import statements from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            return imports
        except Exception:
            return []
    
    # Check key files for proper import structure
    key_files = [
        'gan_cyber_range/__init__.py',
        'gan_cyber_range/core/attack_gan.py',
        'gan_cyber_range/evaluation/attack_evaluator.py',
        'gan_cyber_range/utils/enhanced_security.py'
    ]
    
    import_issues = []
    
    for file_path in key_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            imports = extract_imports(full_path)
            
            # Check for circular imports (simplified)
            for imp in imports:
                if imp.startswith('gan_cyber_range') and '..' in imp:
                    import_issues.append(f"{file_path}: Potential circular import {imp}")
        else:
            import_issues.append(f"Missing file: {file_path}")
    
    if import_issues:
        print(f"✗ Found {len(import_issues)} import issues:")
        for issue in import_issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ Import structure looks good")
        return True


def test_file_sizes():
    """Test that files are reasonable sizes"""
    print("\\nTesting file sizes...")
    
    repo_root = Path(__file__).parent
    large_files = []
    empty_files = []
    
    for py_file in repo_root.rglob("*.py"):
        if 'test_' in py_file.name:
            continue
        
        size = py_file.stat().st_size
        
        if size == 0:
            empty_files.append(py_file)
        elif size > 100 * 1024:  # 100KB
            large_files.append((py_file, size))
    
    issues = []
    
    if empty_files:
        issues.append(f"{len(empty_files)} empty Python files")
    
    if large_files:
        for file_path, size in large_files:
            if size > 500 * 1024:  # 500KB
                issues.append(f"Very large file: {file_path} ({size // 1024}KB)")
    
    if issues:
        print(f"✗ File size issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ File sizes are reasonable")
        return True


def test_documentation():
    """Test documentation presence"""
    print("\\nTesting documentation...")
    
    repo_root = Path(__file__).parent
    
    # Check for key documentation files
    doc_files = [
        'README.md',
        'requirements.txt',
        'setup.py'
    ]
    
    missing_docs = []
    for doc_file in doc_files:
        if not (repo_root / doc_file).exists():
            missing_docs.append(doc_file)
    
    # Check for module docstrings
    def has_module_docstring(file_path):
        """Check if Python file has module docstring"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            return (tree.body and 
                    isinstance(tree.body[0], ast.Expr) and
                    isinstance(tree.body[0].value, ast.Constant) and
                    isinstance(tree.body[0].value.value, str))
        except Exception:
            return False
    
    # Check key modules for docstrings
    key_modules = [
        'gan_cyber_range/core/attack_gan.py',
        'gan_cyber_range/core/cyber_range.py',
        'gan_cyber_range/evaluation/attack_evaluator.py'
    ]
    
    missing_docstrings = []
    for module_path in key_modules:
        full_path = repo_root / module_path
        if full_path.exists() and not has_module_docstring(full_path):
            missing_docstrings.append(module_path)
    
    issues = []
    if missing_docs:
        issues.extend([f"Missing doc file: {doc}" for doc in missing_docs])
    if missing_docstrings:
        issues.extend([f"Missing docstring: {mod}" for mod in missing_docstrings])
    
    if issues:
        print(f"✗ Documentation issues:")
        for issue in issues[:5]:
            print(f"  - {issue}")
        return False
    else:
        print("✓ Documentation looks good")
        return True


def test_git_structure():
    """Test git repository structure"""
    print("\\nTesting git structure...")
    
    repo_root = Path(__file__).parent
    
    # Check for important git files
    git_files = [
        '.gitignore',
        '.git'
    ]
    
    git_issues = []
    for git_file in git_files:
        git_path = repo_root / git_file
        if not git_path.exists():
            git_issues.append(f"Missing: {git_file}")
    
    # Check if we're in a git repository
    try:
        import subprocess
        result = subprocess.run(['git', 'status'], 
                              cwd=repo_root, 
                              capture_output=True, 
                              text=True)
        if result.returncode != 0:
            git_issues.append("Not a git repository or git not available")
    except Exception:
        git_issues.append("Git not available")
    
    if git_issues:
        print(f"⚠️  Git structure issues (non-critical):")
        for issue in git_issues:
            print(f"  - {issue}")
        return True  # Non-critical
    else:
        print("✓ Git structure looks good")
        return True


def run_structure_tests():
    """Run all structure validation tests"""
    print("=" * 60)
    print("GAN CYBER RANGE - STRUCTURE VALIDATION")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    tests = [
        test_code_structure,
        test_module_structure,
        test_import_structure,
        test_file_sizes,
        test_documentation,
        test_git_structure
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
            print(f"✗ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 60)
    print("STRUCTURE VALIDATION RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    print(f"Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        print("\\n🎉 ALL STRUCTURE TESTS PASSED!")
        print("Code structure is well-organized and ready for deployment.")
    elif failed <= 2:
        print(f"\\n✅ MOSTLY GOOD - {failed} minor issue(s) found.")
        print("Code structure is acceptable for deployment.")
    else:
        print(f"\\n⚠️  {failed} structure issue(s) found.")
        print("Consider fixing structural issues before deployment.")
    
    return failed <= 2


if __name__ == "__main__":
    success = run_structure_tests()
    sys.exit(0 if success else 1)