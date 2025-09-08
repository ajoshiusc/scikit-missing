#!/usr/bin/env python
"""Setup script for scikit-missing package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scikit-missing",
    version="0.1.0",
    author="Ajay Joshi",
    author_email="ajoshi@usc.edu",
    description="Scikit-learn extension for handling missing features in machine learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajoshiusc/scikit-missing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
)
