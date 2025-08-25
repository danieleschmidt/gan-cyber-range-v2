#!/usr/bin/env python3
"""
Main entry point for GAN Cyber Range platform
Provides simple CLI interface for core functionality
"""

import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GAN Cyber Range - Defensive Training Platform")
    parser.add_argument('--version', action='version', version='2.0.0')
    parser.add_argument('--mode', choices=['demo', 'train', 'generate', 'serve'], 
                       default='demo', help='Operation mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--port', type=int, default=8000, help='API server port')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'demo':
            from gan_cyber_range.demo import run_defensive_demo
            run_defensive_demo()
        elif args.mode == 'serve':
            from gan_cyber_range.api.main import start_server
            start_server(port=args.port)
        elif args.mode == 'train':
            from gan_cyber_range.core.attack_gan import AttackGAN
            gan = AttackGAN()
            print("Training mode - GAN initialized successfully")
        elif args.mode == 'generate':
            from gan_cyber_range.core.attack_gan import AttackGAN
            gan = AttackGAN()
            print("Generation mode - Ready to generate synthetic attacks")
        else:
            parser.print_help()
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()