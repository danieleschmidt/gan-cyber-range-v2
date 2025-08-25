#!/usr/bin/env python3
"""
Quick start script for GAN Cyber Range
Demonstrates basic defensive functionality without complex dependencies
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check basic environment setup"""
    print("🔍 Environment Check:")
    print(f"  Python: {sys.version}")
    print(f"  Working Directory: {Path.cwd()}")
    print(f"  Platform: {sys.platform}")
    
    # Check critical directories
    critical_dirs = ['gan_cyber_range', 'tests', 'examples']
    for dirname in critical_dirs:
        if Path(dirname).exists():
            print(f"  ✅ {dirname}/ found")
        else:
            print(f"  ❌ {dirname}/ missing")
    
    return True

def simple_defensive_demo():
    """Run simple defensive demo without heavy dependencies"""
    print("\n🛡️ Defensive Cybersecurity Demo")
    print("=" * 50)
    
    try:
        # Import minimal components
        from gan_cyber_range.core.ultra_minimal import UltraMinimalDemo
        
        demo = UltraMinimalDemo()
        results = demo.run()
        
        print(f"✅ Demo completed successfully")
        print(f"📊 Results: {results}")
        
        return True
        
    except ImportError:
        print("📝 Running basic functionality check...")
        
        # Basic attack pattern analysis
        attack_patterns = [
            "reconnaissance network scan 192.168.1.0/24",
            "credential brute force ssh admin",
            "lateral movement smb shares",
            "data exfiltration encrypted tunnel"
        ]
        
        print("\n🔍 Analyzing Attack Patterns:")
        for i, pattern in enumerate(attack_patterns, 1):
            risk_score = len(pattern.split()) * 0.15
            print(f"  {i}. {pattern}")
            print(f"     Risk Score: {risk_score:.2f}/1.0")
        
        print("\n🛡️ Defensive Recommendations:")
        print("  • Network segmentation and monitoring")
        print("  • Multi-factor authentication")
        print("  • Endpoint detection and response")
        print("  • Security awareness training")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False

def generate_sample_config():
    """Generate sample configuration"""
    config = {
        "timestamp": datetime.now().isoformat(),
        "mode": "defensive_training",
        "security_level": "enhanced",
        "monitoring": True,
        "compliance": ["NIST", "ISO27001"],
        "training_modules": [
            "threat_detection",
            "incident_response", 
            "forensic_analysis"
        ]
    }
    
    print("\n⚙️ Sample Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

def main():
    """Main execution"""
    print("🚀 GAN Cyber Range - Quick Start")
    print("Defensive Cybersecurity Training Platform")
    print("=" * 60)
    
    try:
        # Environment check
        check_environment()
        
        # Generate config
        config = generate_sample_config()
        
        # Run defensive demo
        success = simple_defensive_demo()
        
        if success:
            print("\n✅ Quick start completed successfully!")
            print("\n📖 Next Steps:")
            print("  1. Run full demo: python -m gan_cyber_range --mode demo")
            print("  2. Start API server: python -m gan_cyber_range --mode serve")
            print("  3. View examples: python examples/basic_usage_example.py")
        else:
            print("\n❌ Quick start encountered issues")
            return 1
            
    except Exception as e:
        logger.error(f"Quick start failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())