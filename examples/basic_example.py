"""
Basic example of using mSVM with missing data.

This example demonstrates how to use the mSVM classifier with different
kernels to handle missing features in the data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add the package to the path for development
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scikit_missing.svm import mSVM
from scikit_missing.kernels import ExpectedValueKernel, CrossCorrelationKernel


def create_missing_data(X, missing_rate=0.2, random_state=42):
    """
    Introduce missing values into the dataset.
    
    Parameters
    ----------
    X : array-like
        Input data matrix.
    missing_rate : float
        Proportion of values to make missing.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    X_missing : array-like
        Data matrix with missing values (NaN).
    """
    np.random.seed(random_state)
    X_missing = X.copy()
    
    # Randomly select entries to make missing
    n_samples, n_features = X.shape
    n_missing = int(missing_rate * n_samples * n_features)
    
    missing_indices = np.random.choice(
        n_samples * n_features, 
        size=n_missing, 
        replace=False
    )
    
    # Convert flat indices to 2D indices
    row_indices = missing_indices // n_features
    col_indices = missing_indices % n_features
    
    # Set selected entries to NaN
    X_missing[row_indices, col_indices] = np.nan
    
    return X_missing


def main():
    """Run the basic mSVM example."""
    print("=== mSVM Basic Example ===\n")
    
    # Generate synthetic dataset
    print("1. Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    
    # Introduce missing values
    print("\n2. Introducing missing values...")
    missing_rates = [0.1, 0.2, 0.3]
    
    for missing_rate in missing_rates:
        print(f"\n   Testing with {missing_rate*100}% missing data:")
        
        # Create missing data
        X_missing = create_missing_data(X, missing_rate=missing_rate)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_missing, y, test_size=0.3, random_state=42
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        print(f"   Missing values in training: {np.isnan(X_train).sum()}")
        print(f"   Missing values in test: {np.isnan(X_test).sum()}")
        
        # Test different kernels
        kernels = {
            'Expected Value': ExpectedValueKernel(gamma=0.1),
            'Cross Correlation': CrossCorrelationKernel(gamma=0.1)
        }
        
        results = {}
        
        for kernel_name, kernel in kernels.items():
            print(f"\n   Training mSVM with {kernel_name} kernel...")
            
            # Create and train mSVM
            clf = mSVM(kernel=kernel, C=1.0)
            clf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[kernel_name] = accuracy
            print(f"   Accuracy: {accuracy:.3f}")
        
        # Compare results
        print(f"\n   Results for {missing_rate*100}% missing data:")
        for kernel_name, accuracy in results.items():
            print(f"   {kernel_name}: {accuracy:.3f}")
    
    print("\n=== Example completed ===")


def demonstrate_kernel_behavior():
    """Demonstrate how different kernels handle missing data."""
    print("\n=== Kernel Behavior Demonstration ===\n")
    
    # Create simple 2D dataset for visualization
    np.random.seed(42)
    n_samples = 100
    
    # Create two classes with different distributions
    class_0 = np.random.multivariate_normal([1, 1], [[0.5, 0.1], [0.1, 0.5]], n_samples//2)
    class_1 = np.random.multivariate_normal([-1, -1], [[0.5, -0.1], [-0.1, 0.5]], n_samples//2)
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Introduce missing values in first feature for some samples
    X_missing = X.copy()
    missing_indices = np.random.choice(n_samples, size=20, replace=False)
    X_missing[missing_indices, 0] = np.nan
    
    print(f"Created dataset with {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Introduced missing values in feature 0 for {len(missing_indices)} samples")
    
    # Test different kernels
    kernels = {
        'Expected Value': ExpectedValueKernel(gamma=1.0),
        'Cross Correlation': CrossCorrelationKernel(gamma=1.0)
    }
    
    for kernel_name, kernel in kernels.items():
        print(f"\nTesting {kernel_name} kernel:")
        
        # Fit kernel and compute kernel matrix
        kernel.fit(X_missing)
        K = kernel.compute_kernel(X_missing)
        
        print(f"   Kernel matrix shape: {K.shape}")
        print(f"   Kernel matrix range: [{K.min():.3f}, {K.max():.3f}]")
        print(f"   Mean kernel value: {K.mean():.3f}")
        
        # Train classifier
        clf = mSVM(kernel=kernel, C=1.0)
        clf.fit(X_missing, y)
        
        # Evaluate
        y_pred = clf.predict(X_missing)
        accuracy = accuracy_score(y, y_pred)
        print(f"   Training accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
    demonstrate_kernel_behavior()
