"""
scikit-missing: A scikit-learn extension for handling missing features.

This package provides machine learning algorithms that can work directly 
with missing data without requiring imputation.
"""

__version__ = "0.1.0"
__author__ = "Ajay Joshi"
__email__ = "ajoshi@usc.edu"

# Import modules with error handling for development
try:
    from .svm import mSVM
    from .kernels import (
        BaseMissingKernel,
        ExpectedValueKernel,
        CrossCorrelationKernel,
        LinearMissingKernel,
        PolynomialMissingKernel,
    )
    
    __all__ = [
        "mSVM",
        "BaseMissingKernel",
        "ExpectedValueKernel", 
        "CrossCorrelationKernel",
        "LinearMissingKernel",
        "PolynomialMissingKernel",
    ]
except ImportError as e:
    # Handle import errors during development
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")
    __all__ = []
