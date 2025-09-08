"""
Advanced example demonstrating kernel comparison and performance analysis.

This example provides a comprehensive comparison of different kernels
for handling missing data in SVM classification.
"""

import numpy as np
import sys
import os

# Add the package to the path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scikit_missing.svm import mSVM
from scikit_missing.kernels import (
    ExpectedValueKernel, 
    CrossCorrelationKernel,
    LinearMissingKernel,
    PolynomialMissingKernel
)


def generate_synthetic_data(n_samples=500, n_features=10, n_classes=2, noise=0.1, random_state=42):
    """
    Generate synthetic classification data with complex patterns.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Number of features.
    n_classes : int
        Number of classes.
    noise : float
        Noise level in the data.
    random_state : int
        Random seed.
        
    Returns
    -------
    X : ndarray
        Feature matrix.
    y : ndarray
        Target labels.
    """
    np.random.seed(random_state)
    
    # Generate base features
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear decision boundary
    if n_classes == 2:
        # Binary classification with non-linear boundary
        decision_values = (
            X[:, 0]**2 + X[:, 1]**2 - 1 +
            0.5 * np.sin(3 * X[:, 0]) +
            noise * np.random.randn(n_samples)
        )
        y = (decision_values > 0).astype(int)
    else:
        # Multi-class with multiple decision boundaries
        angles = np.linspace(0, 2*np.pi, n_classes+1)[:-1]
        centers = np.column_stack([np.cos(angles), np.sin(angles)]) * 2
        
        y = np.zeros(n_samples)
        for i in range(n_samples):
            distances = np.sum((X[i, :2] - centers)**2, axis=1)
            y[i] = np.argmin(distances)
        
        # Add some noise to class assignments
        noise_mask = np.random.rand(n_samples) < noise
        y[noise_mask] = np.random.randint(0, n_classes, size=np.sum(noise_mask))
    
    return X, y.astype(int)


def introduce_missing_patterns(X, pattern='random', missing_rate=0.2, random_state=42):
    """
    Introduce different patterns of missing data.
    
    Parameters
    ----------
    X : ndarray
        Input data matrix.
    pattern : str
        Type of missing pattern: 'random', 'feature_dependent', 'value_dependent'
    missing_rate : float
        Overall missing rate.
    random_state : int
        Random seed.
        
    Returns
    -------
    X_missing : ndarray
        Data with missing values (NaN).
    """
    np.random.seed(random_state)
    X_missing = X.copy()
    n_samples, n_features = X.shape
    
    if pattern == 'random':
        # Completely random missing
        mask = np.random.rand(n_samples, n_features) < missing_rate
        X_missing[mask] = np.nan
        
    elif pattern == 'feature_dependent':
        # Some features have higher missing rates
        feature_missing_rates = np.random.beta(2, 5, n_features) * missing_rate * 2
        for j in range(n_features):
            mask = np.random.rand(n_samples) < feature_missing_rates[j]
            X_missing[mask, j] = np.nan
            
    elif pattern == 'value_dependent':
        # Missing depends on feature values (MNAR - Missing Not At Random)
        for j in range(n_features):
            # Higher probability of missing for extreme values
            probs = missing_rate * (1 + np.abs(X[:, j]) / np.max(np.abs(X[:, j])))
            mask = np.random.rand(n_samples) < probs
            X_missing[mask, j] = np.nan
    
    return X_missing


def evaluate_kernel_performance():
    """
    Comprehensive evaluation of different kernels under various conditions.
    """
    print("=== Comprehensive Kernel Performance Evaluation ===\n")
    
    # Test parameters
    missing_patterns = ['random', 'feature_dependent', 'value_dependent']
    missing_rates = [0.1, 0.2, 0.3, 0.4]
    datasets = [
        ('Small Dataset', 200, 5, 2),
        ('Medium Dataset', 500, 10, 2),
        ('Large Dataset', 1000, 15, 2),
        ('Multiclass', 500, 10, 3)
    ]
    
    # Initialize kernels
    kernels = {
        'Expected Value (γ=0.1)': ExpectedValueKernel(gamma=0.1),
        'Expected Value (γ=1.0)': ExpectedValueKernel(gamma=1.0),
        'Cross Correlation (γ=0.1)': CrossCorrelationKernel(gamma=0.1),
        'Cross Correlation (γ=1.0)': CrossCorrelationKernel(gamma=1.0),
        'Linear': LinearMissingKernel(),
        'Polynomial (d=2)': PolynomialMissingKernel(degree=2),
        'Polynomial (d=3)': PolynomialMissingKernel(degree=3),
    }
    
    results = []
    
    for dataset_name, n_samples, n_features, n_classes in datasets:
        print(f"\nDataset: {dataset_name}")
        print(f"  Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")
        
        # Generate data
        X, y = generate_synthetic_data(n_samples, n_features, n_classes)
        
        for pattern in missing_patterns:
            print(f"\n  Missing Pattern: {pattern}")
            
            for missing_rate in missing_rates:
                print(f"    Missing Rate: {missing_rate:.1%}")
                
                # Introduce missing data
                X_missing = introduce_missing_patterns(X, pattern, missing_rate)
                
                # Split data
                split_idx = int(0.7 * n_samples)
                X_train, X_test = X_missing[:split_idx], X_missing[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                for kernel_name, kernel in kernels.items():
                    try:
                        # Train model
                        clf = mSVM(kernel=kernel, C=1.0)
                        clf.fit(X_train, y_train)
                        
                        # Evaluate
                        train_acc = clf.score(X_train, y_train)
                        test_acc = clf.score(X_test, y_test)
                        
                        result = {
                            'dataset': dataset_name,
                            'pattern': pattern,
                            'missing_rate': missing_rate,
                            'kernel': kernel_name,
                            'train_accuracy': train_acc,
                            'test_accuracy': test_acc,
                            'n_samples': n_samples,
                            'n_features': n_features,
                            'n_classes': n_classes
                        }
                        results.append(result)
                        
                        print(f"      {kernel_name:25s}: Train={train_acc:.3f}, Test={test_acc:.3f}")
                        
                    except Exception as e:
                        print(f"      {kernel_name:25s}: Error - {str(e)[:50]}...")
    
    return results


def analyze_kernel_properties():
    """
    Analyze mathematical properties of different kernels.
    """
    print("\n=== Kernel Properties Analysis ===\n")
    
    # Create test data with known missing patterns
    np.random.seed(42)
    X = np.random.randn(50, 5)
    
    # Create different missing scenarios
    scenarios = {
        'No Missing': X.copy(),
        'Random 20%': introduce_missing_patterns(X, 'random', 0.2),
        'Feature Dependent': introduce_missing_patterns(X, 'feature_dependent', 0.2),
        'Value Dependent': introduce_missing_patterns(X, 'value_dependent', 0.2),
    }
    
    kernels = {
        'Expected Value': ExpectedValueKernel(gamma=1.0),
        'Cross Correlation': CrossCorrelationKernel(gamma=1.0),
        'Linear': LinearMissingKernel(),
        'Polynomial': PolynomialMissingKernel(degree=2),
    }
    
    for scenario_name, X_scenario in scenarios.items():
        print(f"Scenario: {scenario_name}")
        missing_count = np.isnan(X_scenario).sum()
        total_elements = X_scenario.size
        print(f"  Missing values: {missing_count}/{total_elements} ({missing_count/total_elements:.1%})")
        
        for kernel_name, kernel in kernels.items():
            # Fit kernel
            kernel.fit(X_scenario)
            
            # Compute kernel matrix
            K = kernel.compute_kernel(X_scenario)
            
            # Analyze properties
            eigenvals = np.linalg.eigvals(K)
            
            properties = {
                'min_value': K.min(),
                'max_value': K.max(),
                'mean_value': K.mean(),
                'diagonal_mean': np.mean(np.diag(K)),
                'positive_definite': np.all(eigenvals >= -1e-10),
                'condition_number': np.max(eigenvals) / np.max([np.min(eigenvals), 1e-12])
            }
            
            print(f"  {kernel_name}:")
            for prop, value in properties.items():
                if isinstance(value, bool):
                    print(f"    {prop}: {value}")
                else:
                    print(f"    {prop}: {value:.6f}")
        print()


def kernel_parameter_sensitivity():
    """
    Analyze sensitivity of kernels to their parameters.
    """
    print("\n=== Kernel Parameter Sensitivity Analysis ===\n")
    
    # Generate test data
    X, y = generate_synthetic_data(300, 8, 2)
    X_missing = introduce_missing_patterns(X, 'random', 0.25)
    
    # Split data
    split_idx = int(0.7 * len(X))
    X_train, X_test = X_missing[:split_idx], X_missing[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test RBF kernels with different gamma values
    gamma_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    print("Expected Value Kernel - Gamma Sensitivity:")
    for gamma in gamma_values:
        kernel = ExpectedValueKernel(gamma=gamma)
        clf = mSVM(kernel=kernel, C=1.0)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        print(f"  γ={gamma:6.2f}: Train={train_acc:.3f}, Test={test_acc:.3f}")
    
    print("\nCross Correlation Kernel - Gamma Sensitivity:")
    for gamma in gamma_values:
        kernel = CrossCorrelationKernel(gamma=gamma)
        clf = mSVM(kernel=kernel, C=1.0)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        print(f"  γ={gamma:6.2f}: Train={train_acc:.3f}, Test={test_acc:.3f}")
    
    # Test polynomial kernels with different degrees
    degrees = [1, 2, 3, 4, 5]
    
    print("\nPolynomial Kernel - Degree Sensitivity:")
    for degree in degrees:
        kernel = PolynomialMissingKernel(degree=degree)
        clf = mSVM(kernel=kernel, C=1.0)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        print(f"  d={degree}: Train={train_acc:.3f}, Test={test_acc:.3f}")


def main():
    """Run all advanced examples."""
    try:
        print("Starting comprehensive mSVM analysis...\n")
        
        # Run kernel properties analysis
        analyze_kernel_properties()
        
        # Run parameter sensitivity analysis
        kernel_parameter_sensitivity()
        
        # Run performance evaluation (commented out for speed)
        # Uncomment to run full evaluation
        # results = evaluate_kernel_performance()
        
        print("\n=== Analysis completed successfully ===")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
