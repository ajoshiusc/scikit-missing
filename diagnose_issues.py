"""
Diagnostic script to identify issues with the Cross-Correlation kernel and experimental setup.
"""

import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.dirname(__file__))

from scikit_missing.svm import mSVM
from scikit_missing.kernels import ExpectedValueKernel, CrossCorrelationKernel


def diagnose_issues():
    print("=== Diagnostic Analysis of mSVM Issues ===\n")
    
    # 1. Check data difficulty
    print("1. ANALYZING DATA COMPLEXITY")
    print("-" * 40)
    
    # Generate different difficulty levels
    difficulties = [
        ('Easy', {'class_sep': 2.0, 'n_informative': 8}),
        ('Medium', {'class_sep': 1.0, 'n_informative': 6}),
        ('Hard', {'class_sep': 0.5, 'n_informative': 4})
    ]
    
    for name, params in difficulties:
        X, y = make_classification(
            n_samples=400, n_features=8, n_clusters_per_class=1,
            **params, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Test with no missing data
        from sklearn.svm import SVC
        svm = SVC(kernel='rbf', gamma=1.0, C=1.0)
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        print(f"{name} Problem - Baseline Accuracy: {acc:.3f}")
    
    # 2. Check kernel implementations
    print(f"\n2. ANALYZING KERNEL IMPLEMENTATIONS")
    print("-" * 40)
    
    # Use medium difficulty data
    X, y = make_classification(n_samples=200, n_features=6, class_sep=1.0, 
                              n_informative=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Introduce 20% missing data
    np.random.seed(42)
    missing_mask = np.random.rand(*X_train.shape) < 0.2
    X_train_missing = X_train.copy()
    X_train_missing[missing_mask] = np.nan
    
    missing_mask_test = np.random.rand(*X_test.shape) < 0.2
    X_test_missing = X_test.copy()
    X_test_missing[missing_mask_test] = np.nan
    
    print(f"Training missing rate: {np.isnan(X_train_missing).mean():.1%}")
    print(f"Test missing rate: {np.isnan(X_test_missing).mean():.1%}")
    
    # Test different gamma values
    gamma_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print(f"\nExpected Value Kernel Results:")
    for gamma in gamma_values:
        try:
            kernel = ExpectedValueKernel(gamma=gamma)
            clf = mSVM(kernel=kernel, C=1.0)
            clf.fit(X_train_missing, y_train)
            acc = clf.score(X_test_missing, y_test)
            print(f"  γ={gamma}: {acc:.3f}")
        except Exception as e:
            print(f"  γ={gamma}: ERROR - {e}")
    
    print(f"\nCross-Correlation Kernel Results:")
    for gamma in gamma_values:
        try:
            kernel = CrossCorrelationKernel(gamma=gamma)
            clf = mSVM(kernel=kernel, C=1.0)
            clf.fit(X_train_missing, y_train)
            acc = clf.score(X_test_missing, y_test)
            print(f"  γ={gamma}: {acc:.3f}")
        except Exception as e:
            print(f"  γ={gamma}: ERROR - {e}")
    
    # 3. Check kernel matrix properties
    print(f"\n3. ANALYZING KERNEL MATRIX PROPERTIES")
    print("-" * 40)
    
    small_X = X_train_missing[:50]  # Use small subset for analysis
    
    for kernel_name, kernel_class in [('Expected Value', ExpectedValueKernel), 
                                     ('Cross Correlation', CrossCorrelationKernel)]:
        print(f"\n{kernel_name} Kernel:")
        try:
            kernel = kernel_class(gamma=1.0)
            kernel.fit(small_X)
            K = kernel.compute_kernel(small_X)
            
            print(f"  Shape: {K.shape}")
            print(f"  Diagonal range: [{np.diag(K).min():.3f}, {np.diag(K).max():.3f}]")
            print(f"  Off-diagonal range: [{K[np.triu_indices_from(K, k=1)].min():.3f}, {K[np.triu_indices_from(K, k=1)].max():.3f}]")
            print(f"  Symmetry error: {np.max(np.abs(K - K.T)):.6f}")
            
            # Check positive semi-definiteness
            eigenvals = np.linalg.eigvals(K)
            min_eigenval = np.min(eigenvals)
            print(f"  Min eigenvalue: {min_eigenval:.6f}")
            
            if min_eigenval < -1e-10:
                print(f"  WARNING: Kernel matrix is not positive semi-definite!")
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # 4. Check variance computation in Cross-Correlation kernel
    print(f"\n4. ANALYZING CROSS-CORRELATION KERNEL STATISTICS")
    print("-" * 40)
    
    kernel = CrossCorrelationKernel(gamma=1.0)
    kernel.fit(X_train_missing)
    
    print("Feature statistics:")
    for i in range(min(6, X_train.shape[1])):  # Show first 6 features
        mean_val = kernel.missing_stats_['means'][i]
        var_val = kernel.missing_stats_['vars'][i]
        std_val = np.sqrt(var_val)
        print(f"  Feature {i}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    # Check if variances are reasonable
    high_var_features = np.where(kernel.missing_stats_['vars'] > 10)[0]
    if len(high_var_features) > 0:
        print(f"  WARNING: High variance features detected: {high_var_features}")
    
    low_var_features = np.where(kernel.missing_stats_['vars'] < 0.01)[0]
    if len(low_var_features) > 0:
        print(f"  WARNING: Very low variance features detected: {low_var_features}")


if __name__ == "__main__":
    diagnose_issues()
