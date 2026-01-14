#!/usr/bin/env python3
"""
Setup script for cleanAdS / ads_cft package.

Install in development mode:
    pip install -e .

This creates an ads_cft package that can be imported from anywhere.
"""

from setuptools import setup, find_packages

setup(
    name="ads_cft",
    version="1.0.0",
    description="UV-Stabilized Generative Flow Matching via AdS/CFT",
    author="Your Name",
    packages=["ads_cft"],
    package_dir={"ads_cft": "."},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
        ],
    },
    entry_points={
        "console_scripts": [
            "ads-train=ads_cft.train:main",
            "ads-evaluate=ads_cft.evaluate:main",
        ],
    },
)