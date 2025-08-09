#!/usr/bin/env python3
"""
Setup script for Zero Cost NAS project
Run this script to install all required dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a Python package using pip."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main setup function."""
    print("Setting up Zero Cost NAS environment...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install requirements
    requirements = [
        'numpy>=1.21.0',
        'torch>=1.9.0'
    ]
    
    failed_packages = []
    
    for package in requirements:
        print(f"\nInstalling {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Warning: Failed to install: {', '.join(failed_packages)}")
        print("Please install them manually using:")
        for package in failed_packages:
            print(f"  pip install {package}")
    else:
        print("\n🎉 All dependencies installed successfully!")
        print("\nYou can now run the main script:")
        print("  python main.py")

if __name__ == '__main__':
    main()
