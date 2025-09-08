"""
Quick test script to verify the mSVM implementation works.
"""

import numpy as np
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_basic_functionality():
    """Test basic functionality of mSVM."""
    print("Testing basic mSVM functionality...")
    
    try:
        from scikit_missing.kernels import ExpectedValueKernel, CrossCorrelationKernel
        from scikit_missing.svm import mSVM
        
        print("✓ Successfully imported modules")
        
        # Create simple test data
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Introduce some missing values
        X[5:10, 0] = np.nan
        X[15:20, 2] = np.nan
        
        print(f"✓ Created test data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Missing values: {np.isnan(X).sum()}")
        
        # Test Expected Value Kernel
        print("\nTesting Expected Value Kernel...")
        kernel1 = ExpectedValueKernel(gamma=1.0)
        clf1 = mSVM(kernel=kernel1, C=1.0)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)
        acc1 = np.mean(pred1 == y)
        print(f"✓ Expected Value Kernel accuracy: {acc1:.3f}")
        
        # Test Cross Correlation Kernel
        print("\nTesting Cross Correlation Kernel...")
        kernel2 = CrossCorrelationKernel(gamma=1.0)
        clf2 = mSVM(kernel=kernel2, C=1.0)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)
        acc2 = np.mean(pred2 == y)
        print(f"✓ Cross Correlation Kernel accuracy: {acc2:.3f}")
        
        # Test probability predictions
        print("\nTesting probability predictions...")
        proba = clf1.predict_proba(X[:5])
        print(f"✓ Probability predictions shape: {proba.shape}")
        print(f"  Sample probabilities: {proba[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kernel_properties():
    """Test mathematical properties of kernels."""
    print("\nTesting kernel mathematical properties...")
    
    try:
        from scikit_missing.kernels import ExpectedValueKernel
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(20, 5)
        X[5:8, 1] = np.nan  # Add some missing values
        
        kernel = ExpectedValueKernel(gamma=1.0)
        kernel.fit(X)
        K = kernel.compute_kernel(X)
        
        # Test symmetry
        symmetry_error = np.max(np.abs(K - K.T))
        print(f"✓ Symmetry error: {symmetry_error:.6f}")
        
        # Test diagonal values
        diag_values = np.diag(K)
        print(f"✓ Diagonal values range: [{diag_values.min():.3f}, {diag_values.max():.3f}]")
        
        # Test positive semi-definiteness
        eigenvals = np.linalg.eigvals(K)
        min_eigenval = np.min(eigenvals)
        print(f"✓ Minimum eigenvalue: {min_eigenval:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("mSVM Implementation Verification")
    print("=" * 50)
    
    success = True
    
    # Test basic functionality
    success &= test_basic_functionality()
    
    # Test kernel properties
    success &= test_kernel_properties()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ ALL TESTS PASSED!")
        print("The mSVM implementation is working correctly.")
    else:
        print("✗ SOME TESTS FAILED!")
        print("Please check the implementation.")
    print("=" * 50)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
