"""
Basic tests for scikit-missing package.

This module contains unit tests for the mSVM implementation and kernels.
"""

import unittest
import numpy as np
import sys
import os

# Add the package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scikit_missing.kernels import (
    ExpectedValueKernel,
    CrossCorrelationKernel,
    LinearMissingKernel,
    PolynomialMissingKernel
)
from scikit_missing.svm import mSVM


class TestKernels(unittest.TestCase):
    """Test cases for missing data kernels."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(20, 5)
        self.X_missing = self.X.copy()
        
        # Introduce some missing values
        missing_indices = [(1, 0), (3, 2), (5, 1), (7, 4), (10, 3)]
        for i, j in missing_indices:
            self.X_missing[i, j] = np.nan
    
    def test_expected_value_kernel(self):
        """Test Expected Value Kernel."""
        kernel = ExpectedValueKernel(gamma=1.0)
        kernel.fit(self.X_missing)
        
        # Test kernel computation
        K = kernel.compute_kernel(self.X_missing)
        
        # Check properties
        self.assertEqual(K.shape, (20, 20))
        self.assertTrue(np.allclose(K, K.T))  # Symmetry
        self.assertTrue(np.all(K >= 0))  # Non-negativity for RBF
        self.assertTrue(np.allclose(np.diag(K), 1.0))  # Diagonal should be 1
    
    def test_cross_correlation_kernel(self):
        """Test Cross Correlation Kernel."""
        kernel = CrossCorrelationKernel(gamma=1.0)
        kernel.fit(self.X_missing)
        
        # Test kernel computation
        K = kernel.compute_kernel(self.X_missing)
        
        # Check properties
        self.assertEqual(K.shape, (20, 20))
        self.assertTrue(np.allclose(K, K.T))  # Symmetry
        self.assertTrue(np.all(K >= 0))  # Non-negativity for RBF
        self.assertTrue(np.all(np.diag(K) <= 1.0 + 1e-10))  # Diagonal <= 1
    
    def test_linear_missing_kernel(self):
        """Test Linear Missing Kernel."""
        kernel = LinearMissingKernel()
        kernel.fit(self.X_missing)
        
        # Test kernel computation
        K = kernel.compute_kernel(self.X_missing)
        
        # Check properties
        self.assertEqual(K.shape, (20, 20))
        self.assertTrue(np.allclose(K, K.T))  # Symmetry
    
    def test_polynomial_missing_kernel(self):
        """Test Polynomial Missing Kernel."""
        kernel = PolynomialMissingKernel(degree=2, coef0=1.0)
        kernel.fit(self.X_missing)
        
        # Test kernel computation
        K = kernel.compute_kernel(self.X_missing)
        
        # Check properties
        self.assertEqual(K.shape, (20, 20))
        self.assertTrue(np.allclose(K, K.T))  # Symmetry
        self.assertTrue(np.all(K >= 0))  # Non-negativity for polynomial with coef0 > 0
    
    def test_kernel_with_no_missing_data(self):
        """Test kernels work correctly with no missing data."""
        kernels = [
            ExpectedValueKernel(gamma=1.0),
            CrossCorrelationKernel(gamma=1.0),
            LinearMissingKernel(),
            PolynomialMissingKernel(degree=2)
        ]
        
        for kernel in kernels:
            kernel.fit(self.X)  # No missing data
            K = kernel.compute_kernel(self.X)
            
            self.assertEqual(K.shape, (20, 20))
            self.assertTrue(np.allclose(K, K.T))  # Symmetry
    
    def test_kernel_different_test_data(self):
        """Test kernel computation with different test data."""
        kernel = ExpectedValueKernel(gamma=1.0)
        kernel.fit(self.X_missing)
        
        # Create different test data
        X_test = np.random.randn(10, 5)
        X_test[2, 1] = np.nan
        X_test[5, 3] = np.nan
        
        K = kernel.compute_kernel(self.X_missing, X_test)
        
        self.assertEqual(K.shape, (20, 10))


class TestmSVM(unittest.TestCase):
    """Test cases for mSVM classifier."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create simple 2D classification problem
        n_samples = 100
        self.X = np.random.randn(n_samples, 4)
        
        # Create labels based on a simple decision boundary
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)
        
        # Introduce missing values
        self.X_missing = self.X.copy()
        missing_mask = np.random.rand(n_samples, 4) < 0.15
        self.X_missing[missing_mask] = np.nan
    
    def test_binary_classification(self):
        """Test binary classification."""
        clf = mSVM(kernel=ExpectedValueKernel(gamma=1.0), C=1.0)
        clf.fit(self.X_missing, self.y)
        
        # Check fitted attributes
        self.assertTrue(hasattr(clf, 'classes_'))
        self.assertTrue(hasattr(clf, 'support_'))
        self.assertTrue(hasattr(clf, 'dual_coef_'))
        self.assertTrue(hasattr(clf, 'intercept_'))
        
        # Make predictions
        y_pred = clf.predict(self.X_missing)
        self.assertEqual(len(y_pred), len(self.y))
        self.assertTrue(np.all(np.isin(y_pred, clf.classes_)))
        
        # Check score
        score = clf.score(self.X_missing, self.y)
        self.assertTrue(0 <= score <= 1)
    
    def test_multiclass_classification(self):
        """Test multiclass classification."""
        # Create 3-class problem
        y_multi = np.random.choice([0, 1, 2], size=len(self.y), replace=True)
        
        clf = mSVM(kernel=ExpectedValueKernel(gamma=1.0), C=1.0)
        clf.fit(self.X_missing, y_multi)
        
        # Check fitted attributes
        self.assertEqual(len(clf.classes_), 3)
        self.assertTrue(hasattr(clf, 'binary_classifiers_'))
        
        # Make predictions
        y_pred = clf.predict(self.X_missing)
        self.assertEqual(len(y_pred), len(y_multi))
        self.assertTrue(np.all(np.isin(y_pred, clf.classes_)))
    
    def test_different_kernels(self):
        """Test mSVM with different kernels."""
        kernels = [
            ExpectedValueKernel(gamma=1.0),
            CrossCorrelationKernel(gamma=1.0),
            LinearMissingKernel(),
            PolynomialMissingKernel(degree=2)
        ]
        
        for kernel in kernels:
            clf = mSVM(kernel=kernel, C=1.0)
            clf.fit(self.X_missing, self.y)
            
            y_pred = clf.predict(self.X_missing)
            self.assertEqual(len(y_pred), len(self.y))
    
    def test_predict_proba(self):
        """Test probability prediction."""
        clf = mSVM(kernel=ExpectedValueKernel(gamma=1.0), C=1.0)
        clf.fit(self.X_missing, self.y)
        
        proba = clf.predict_proba(self.X_missing)
        
        # Check shape and properties
        self.assertEqual(proba.shape, (len(self.y), 2))
        self.assertTrue(np.allclose(np.sum(proba, axis=1), 1.0))
        self.assertTrue(np.all(proba >= 0))
        self.assertTrue(np.all(proba <= 1))
    
    def test_decision_function(self):
        """Test decision function."""
        clf = mSVM(kernel=ExpectedValueKernel(gamma=1.0), C=1.0)
        clf.fit(self.X_missing, self.y)
        
        decision = clf.decision_function(self.X_missing)
        self.assertEqual(len(decision), len(self.y))
        
        # Check consistency with predictions
        y_pred = clf.predict(self.X_missing)
        expected_pred = np.where(decision >= 0, clf.classes_[1], clf.classes_[0])
        self.assertTrue(np.array_equal(y_pred, expected_pred))
    
    def test_kernel_string_initialization(self):
        """Test kernel initialization from string."""
        clf = mSVM(kernel='expected_value', C=1.0)
        self.assertIsInstance(clf.kernel, ExpectedValueKernel)
        
        clf = mSVM(kernel='cross_correlation', C=1.0)
        self.assertIsInstance(clf.kernel, CrossCorrelationKernel)
    
    def test_empty_and_edge_cases(self):
        """Test edge cases."""
        # Test with very small dataset
        X_small = self.X_missing[:5]
        y_small = self.y[:5]
        
        clf = mSVM(kernel=ExpectedValueKernel(gamma=1.0), C=1.0)
        clf.fit(X_small, y_small)
        
        y_pred = clf.predict(X_small)
        self.assertEqual(len(y_pred), len(y_small))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self):
        """Test complete machine learning workflow."""
        np.random.seed(42)
        
        # Generate data
        n_samples = 200
        X = np.random.randn(n_samples, 6)
        y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
        
        # Introduce missing values with different patterns
        X_missing = X.copy()
        
        # Random missing
        missing_mask = np.random.rand(n_samples, 6) < 0.2
        X_missing[missing_mask] = np.nan
        
        # Feature-dependent missing
        X_missing[X[:, 0] > 1.5, 1] = np.nan
        X_missing[X[:, 2] < -1.0, 3] = np.nan
        
        # Split data
        split_idx = int(0.7 * n_samples)
        X_train, X_test = X_missing[:split_idx], X_missing[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Test different kernels
        kernels = [
            ('Expected Value', ExpectedValueKernel(gamma=0.5)),
            ('Cross Correlation', CrossCorrelationKernel(gamma=0.5)),
            ('Linear', LinearMissingKernel()),
            ('Polynomial', PolynomialMissingKernel(degree=2))
        ]
        
        results = {}
        
        for kernel_name, kernel in kernels:
            # Train model
            clf = mSVM(kernel=kernel, C=1.0)
            clf.fit(X_train, y_train)
            
            # Evaluate
            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            
            results[kernel_name] = (train_score, test_score)
            
            # Check basic properties
            self.assertTrue(0 <= train_score <= 1)
            self.assertTrue(0 <= test_score <= 1)
            
            # Test predictions
            y_pred = clf.predict(X_test)
            self.assertEqual(len(y_pred), len(y_test))
            self.assertTrue(np.all(np.isin(y_pred, [0, 1])))
        
        # Print results
        print("\nIntegration Test Results:")
        for kernel_name, (train_score, test_score) in results.items():
            print(f"  {kernel_name:15s}: Train={train_score:.3f}, Test={test_score:.3f}")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestKernels))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestmSVM))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
