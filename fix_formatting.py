#!/usr/bin/env python3
"""Fix formatting issues in Python files."""

import os
import re

def fix_file_formatting(file_path):
    """Fix formatting issues in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix escaped newlines in strings
        content = re.sub(r'\\n\s+', '\n            ', content)
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Fixed formatting in {file_path}")
        return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix formatting in all Python files."""
    files_to_fix = [
        '/root/repo/gan_cyber_range/optimization/advanced_performance.py'
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            fix_file_formatting(file_path)

if __name__ == '__main__':
    main()