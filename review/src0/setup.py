"""
Setup script pour le package Glacier Friction Analysis.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lire le README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="glacier_friction_analysis",
    version="1.0.0",
    author="Glacier Friction Analysis Team",
    description="Analyse des lois de friction pour les glaciers alpins",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "shapely>=1.8.0",
        "adjustText>=0.8",
        "rasterio>=1.2.0",
        "pyproj>=3.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.9",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
