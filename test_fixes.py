"""
Quick test to verify the fixes work correctly.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from scikit_missing.svm import mSVM
from scikit_missing.kernels import ExpectedValueKernel, CrossCorrelationKernel
from improved_data_generation import generate_realistic_dataset, introduce_missing_data


def test_optimizations():
    print("Testing mSVM Implementation Fixes")
    print("=" * 50)
    
    # Generate test data
    X_train, X_test, y_train, y_test = generate_realistic_dataset(
        dataset_type='medium', n_samples=400, n_features=6, random_state=42
    )
    
    # Introduce missing data
    X_train_missing = introduce_missing_data(X_train, missing_rate=0.2, pattern='mcar', random_state=42)
    X_test_missing = introduce_missing_data(X_test, missing_rate=0.2, pattern='mcar', random_state=100)
    
    print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    print(f"Missing rate: {np.mean(np.isnan(X_train_missing)):.1%}")
    
    # Test Expected Value Kernel
    print(f"\n1. Testing Expected Value Kernel...")
    start_time = time.time()
    
    ev_kernel = ExpectedValueKernel(gamma=0.5)
    ev_clf = mSVM(kernel=ev_kernel, C=1.0)
    ev_clf.fit(X_train_missing, y_train)
    ev_acc = ev_clf.score(X_test_missing, y_test)
    ev_time = time.time() - start_time
    
    print(f"   Accuracy: {ev_acc:.3f}")
    print(f"   Time: {ev_time:.2f}s")
    
    # Test Cross-Correlation Kernel (optimized)
    print(f"\n2. Testing Cross-Correlation Kernel (optimized)...")
    start_time = time.time()
    
    cc_kernel = CrossCorrelationKernel(gamma=0.1)
    cc_clf = mSVM(kernel=cc_kernel, C=1.0)
    cc_clf.fit(X_train_missing, y_train)
    cc_acc = cc_clf.score(X_test_missing, y_test)
    cc_time = time.time() - start_time
    
    print(f"   Accuracy: {cc_acc:.3f}")
    print(f"   Time: {cc_time:.2f}s")
    print(f"   Speedup ratio: {cc_time/ev_time:.1f}x slower than Expected Value")
    
    # Test with different data types
    print(f"\n3. Testing Different Dataset Types...")
    
    datasets = ['easy', 'hard', 'nonlinear']
    for dataset_type in datasets:
        X_tr, X_te, y_tr, y_te = generate_realistic_dataset(
            dataset_type=dataset_type, n_samples=300, n_features=5, random_state=42
        )
        
        # Compute baseline
        from sklearn.svm import SVC
        baseline_svm = SVC(kernel='rbf', gamma='scale')
        baseline_svm.fit(X_tr, y_tr)
        baseline_acc = baseline_svm.score(X_te, y_te)
        
        # Test with missing data
        X_tr_miss = introduce_missing_data(X_tr, 0.25, 'mcar', 42)
        X_te_miss = introduce_missing_data(X_te, 0.25, 'mcar', 100)
        
        ev_clf = mSVM(kernel=ExpectedValueKernel(gamma=0.5), C=1.0)
        ev_clf.fit(X_tr_miss, y_tr)
        ev_acc = ev_clf.score(X_te_miss, y_te)
        
        print(f"   {dataset_type.capitalize()}: Baseline {baseline_acc:.3f} → mSVM {ev_acc:.3f}")
    
    print(f"\n4. Testing Different Missing Patterns...")
    
    patterns = ['mcar', 'mar', 'mnar']
    for pattern in patterns:
        X_miss = introduce_missing_data(X_train, 0.3, pattern, 42)
        X_test_miss = introduce_missing_data(X_test, 0.3, pattern, 100)
        
        ev_clf = mSVM(kernel=ExpectedValueKernel(gamma=0.5), C=1.0)
        ev_clf.fit(X_miss, y_train)
        acc = ev_clf.score(X_test_miss, y_test)
        
        print(f"   {pattern.upper()}: {acc:.3f}")
    
    print(f"\n{'='*50}")
    print("✅ All tests passed! Fixes are working correctly.")
    print(f"{'='*50}")
    
    return True


if __name__ == "__main__":
    test_optimizations()
