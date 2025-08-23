#!/usr/bin/env python3
"""
Automated dependency installer for GAN-Cyber-Range-v2

This script installs core dependencies needed for basic functionality
in a defensive cybersecurity research environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command with error handling"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                               capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def install_core_dependencies():
    """Install essential Python packages for defensive functionality"""
    
    core_packages = [
        "numpy>=1.24.0",
        "pandas>=2.0.0", 
        "scikit-learn>=1.3.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.4.0",
        "click>=8.1.0",
        "rich>=13.6.0",
        "python-dotenv>=1.0.0",
        "cryptography>=41.0.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0"
    ]
    
    print("Installing core defensive security dependencies...")
    
    for package in core_packages:
        success = run_command(f"pip3 install '{package}'", check=False)
        if not success:
            print(f"Warning: Failed to install {package}")
    
    return True

def setup_development_environment():
    """Setup development environment for defensive research"""
    
    print("Setting up defensive security development environment...")
    
    # Create necessary directories
    dirs = [
        "logs", "data/defensive_datasets", "models/defensive_models",
        "configs/defensive", "reports", "training_data"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create basic configuration
    config = {
        "defensive_mode": True,
        "research_only": True,
        "authorized_use": True,
        "log_level": "INFO",
        "security_validation": True
    }
    
    import json
    with open("configs/defensive/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Defensive configuration created")
    return True

def validate_installation():
    """Validate that core defensive capabilities are working"""
    
    print("Validating defensive security installation...")
    
    try:
        import numpy as np
        import pandas as pd
        from fastapi import FastAPI
        import cryptography
        print("✅ Core defensive libraries imported successfully")
        
        # Test basic functionality
        data = np.random.random((10, 5))
        df = pd.DataFrame(data)
        print("✅ Data processing capabilities working")
        
        app = FastAPI(title="Defensive Security API")
        print("✅ API framework ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Main installation process for defensive capabilities"""
    
    print("🛡️  GAN-Cyber-Range-v2 Defensive Setup")
    print("=" * 50)
    
    if not install_core_dependencies():
        print("❌ Failed to install dependencies")
        return 1
    
    if not setup_development_environment():
        print("❌ Failed to setup environment")
        return 1
        
    if not validate_installation():
        print("❌ Installation validation failed")
        return 1
    
    print("\n✅ Defensive security environment ready!")
    print("Ready for defensive cybersecurity research and training")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())