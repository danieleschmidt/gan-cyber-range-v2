"""Setup script for GAN-Cyber-Range-v2"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="gan-cyber-range-v2",
    version="2.0.0",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.com",
    description="Second-Generation Adversarial Cyber Range with GAN-based Attack Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terragonlabs/gan-cyber-range-v2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking",
        "Topic :: Education",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "torch[cu118]>=2.0.0",
            "torchvision[cu118]>=0.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "cyber-range=gan_cyber_range.cli.main:main",
            "gan-trainer=gan_cyber_range.cli.train:main",
            "range-manager=gan_cyber_range.cli.manager:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gan_cyber_range": [
            "data/*.json",
            "data/*.yaml",
            "configs/*.yaml",
            "templates/*.html",
            "static/*",
        ],
    },
    zip_safe=False,
)