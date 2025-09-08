# scikit-missing

A scikit-learn extension for handling missing features in machine learning algorithms.

## ⚠️ Important Note on Citations

**This is NOT a published research paper.** This implementation synthesizes and extends existing research on missing data in Support Vector Machines. See [CITATION_NOTICE.md](CITATION_NOTICE.md) for proper citations of the foundational work this builds upon.

**Key foundational papers:**
- Pelckmans et al. (2005) - "Handling missing values in support vector machine classifiers"
- Chechik et al. (2008) - "Max-margin classification of data with absent features"  
- Williams et al. (2007) - "On classification with incomplete data"

## Overview

This package provides algorithms that can work directly with missing data without requiring imputation. The initial implementation focuses on Support Vector Machines (SVM) with kernels designed to handle missing features.

## Features

- **mSVM (Missing data SVM)**: Support Vector Machine implementation that works with missing features
- **Swappable kernels**: Easy-to-use kernel interface for experimenting with different approaches to missing data
- **Built-in kernels**:
  - Expected value kernel: Uses expected values over known features
  - Cross-correlation kernel: Replaces missing features with Gaussian distributions

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from sklearn.datasets import make_classification
from scikit_missing.svm import mSVM
from scikit_missing.kernels import ExpectedValueKernel, CrossCorrelationKernel

# Create sample data with missing values
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
# Introduce missing values (represented as NaN)
X[X < 0] = np.nan

# Use mSVM with different kernels
svm_ev = mSVM(kernel=ExpectedValueKernel())
svm_cc = mSVM(kernel=CrossCorrelationKernel())

# Fit and predict
svm_ev.fit(X, y)
predictions = svm_ev.predict(X)
```

## License

This project is licensed under the GNU General Public License v2.0 - see the LICENSE file for details.
