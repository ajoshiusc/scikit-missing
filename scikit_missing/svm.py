"""
Missing data Support Vector Machine (mSVM) implementation.

This module provides SVM implementations that can work directly with missing
features without requiring imputation.
"""

import numpy as np
from typing import Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.svm import SVC
from scipy.optimize import minimize
import warnings

from .kernels import BaseMissingKernel, ExpectedValueKernel


class mSVM(BaseEstimator, ClassifierMixin):
    """
    Missing data Support Vector Machine.
    
    A Support Vector Machine implementation that can handle missing features
    directly using specialized kernels designed for incomplete data.
    
    Parameters
    ----------
    kernel : BaseMissingKernel or str, default=ExpectedValueKernel()
        Kernel to use for handling missing data. Can be a kernel instance
        or a string specifying a predefined kernel ('expected_value', 
        'cross_correlation').
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C.
    max_iter : int, default=1000
        Maximum number of iterations for the optimization algorithm.
    tol : float, default=1e-3
        Tolerance for stopping criteria.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_features_in_ : int
        Number of features seen during fit.
    support_ : ndarray of shape (n_sv,)
        Indices of support vectors.
    support_vectors_ : ndarray of shape (n_sv, n_features)
        Support vectors.
    dual_coef_ : ndarray of shape (n_classes-1, n_sv)
        Coefficients of the support vector in the decision function.
    intercept_ : ndarray of shape (n_classes * (n_classes-1) / 2,)
        Constants in decision function.
    """
    
    def __init__(
        self,
        kernel: Union[BaseMissingKernel, str] = None,
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: Optional[int] = None
    ):
        self.kernel = kernel if kernel is not None else ExpectedValueKernel()
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Initialize kernel if string is provided
        if isinstance(self.kernel, str):
            self.kernel = self._get_kernel_from_string(self.kernel)
    
    def _get_kernel_from_string(self, kernel_name: str) -> BaseMissingKernel:
        """Convert string kernel name to kernel instance."""
        from .kernels import ExpectedValueKernel, CrossCorrelationKernel
        
        kernel_map = {
            'expected_value': ExpectedValueKernel(),
            'cross_correlation': CrossCorrelationKernel(),
        }
        
        if kernel_name not in kernel_map:
            raise ValueError(f"Unknown kernel '{kernel_name}'. "
                           f"Available kernels: {list(kernel_map.keys())}")
        
        return kernel_map[kernel_name]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'mSVM':
        """
        Fit the mSVM model according to the given training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, may contain NaN for missing features.
        y : array-like of shape (n_samples,)
            Target values (class labels).
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Input validation
        X, y = check_X_y(X, y, accept_sparse=False, force_all_finite=False)
        
        # Store classes and number of features
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Handle binary and multiclass cases
        if len(self.classes_) == 2:
            return self._fit_binary(X, y)
        else:
            return self._fit_multiclass(X, y)
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> 'mSVM':
        """Fit binary SVM."""
        # Convert labels to {-1, +1}
        y_binary = np.where(y == self.classes_[0], -1, 1)
        
        # Fit the kernel
        self.kernel.fit(X)
        
        # Compute kernel matrix
        K = self.kernel.compute_kernel(X, X)
        
        # Solve the dual optimization problem
        alpha = self._solve_dual_problem(K, y_binary)
        
        # Store support vectors and coefficients
        support_mask = alpha > self.tol
        self.support_ = np.where(support_mask)[0]
        self.support_vectors_ = X[support_mask]
        self.dual_coef_ = (alpha[support_mask] * y_binary[support_mask]).reshape(1, -1)
        
        # Compute intercept
        self.intercept_ = self._compute_intercept(K, y_binary, alpha)
        
        return self
    
    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray) -> 'mSVM':
        """Fit multiclass SVM using one-vs-one strategy."""
        n_classes = len(self.classes_)
        
        # Store binary classifiers for each pair of classes
        self.binary_classifiers_ = {}
        
        # Fit kernel
        self.kernel.fit(X)
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                # Create binary problem for classes i and j
                mask = (y == self.classes_[i]) | (y == self.classes_[j])
                X_binary = X[mask]
                y_binary = y[mask]
                
                # Create and fit binary classifier
                binary_clf = mSVM(
                    kernel=self.kernel,
                    C=self.C,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state
                )
                binary_clf.fit(X_binary, y_binary)
                
                self.binary_classifiers_[(i, j)] = binary_clf
        
        return self
    
    def _solve_dual_problem(self, K: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve the SVM dual optimization problem.
        
        minimize: 0.5 * α^T Q α - e^T α
        subject to: 0 ≤ α_i ≤ C, sum(α_i * y_i) = 0
        
        where Q_ij = y_i * y_j * K(x_i, x_j)
        """
        n_samples = len(y)
        
        # Build Q matrix
        Q = np.outer(y, y) * K
        
        # Objective function for minimization
        def objective(alpha):
            return 0.5 * np.dot(alpha, np.dot(Q, alpha)) - np.sum(alpha)
        
        # Gradient of objective function
        def gradient(alpha):
            return np.dot(Q, alpha) - np.ones(n_samples)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y), 'jac': lambda alpha: y}
        ]
        
        # Bounds: 0 <= alpha_i <= C
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        # Initial guess
        alpha0 = np.zeros(n_samples)
        
        # Solve optimization problem
        result = minimize(
            objective,
            alpha0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        return result.x
    
    def _compute_intercept(self, K: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Compute the intercept term."""
        # Find support vectors (0 < alpha < C)
        sv_mask = (alpha > self.tol) & (alpha < self.C - self.tol)
        
        if not np.any(sv_mask):
            # If no support vectors found, use all non-zero alphas
            sv_mask = alpha > self.tol
        
        if not np.any(sv_mask):
            return np.array([0.0])
        
        # Compute intercept using support vectors
        support_indices = np.where(sv_mask)[0]
        intercepts = []
        
        for idx in support_indices:
            decision_value = np.sum(alpha * y * K[idx, :])
            intercept = y[idx] - decision_value
            intercepts.append(intercept)
        
        return np.array([np.mean(intercepts)])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples, may contain NaN for missing features.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        scores = self.decision_function(X)
        
        if len(self.classes_) == 2:
            return np.where(scores >= 0, self.classes_[1], self.classes_[0])
        else:
            return self.classes_[np.argmax(scores, axis=1)]
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the decision function for the samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        decision : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values for the samples.
        """
        X = check_array(X, accept_sparse=False, force_all_finite=False)
        
        if len(self.classes_) == 2:
            return self._binary_decision_function(X)
        else:
            return self._multiclass_decision_function(X)
    
    def _binary_decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function for binary classification."""
        # Compute kernel matrix between X and support vectors
        K = self.kernel.compute_kernel(X, self.support_vectors_)
        
        # Compute decision values
        decision = np.dot(K, self.dual_coef_.ravel()) + self.intercept_[0]
        
        return decision
    
    def _multiclass_decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function for multiclass classification using voting."""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Initialize vote matrix
        votes = np.zeros((n_samples, n_classes))
        
        # Get predictions from all binary classifiers
        for (i, j), clf in self.binary_classifiers_.items():
            scores = clf.decision_function(X)
            
            # Vote for winning class
            class_i_wins = scores >= 0
            votes[class_i_wins, j] += 1  # class j wins when score >= 0
            votes[~class_i_wins, i] += 1  # class i wins when score < 0
        
        return votes
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class probabilities for samples in X.
        
        Note: This is a simple implementation using Platt scaling approximation.
        For more accurate probabilities, consider using proper probability calibration.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        probas : ndarray of shape (n_samples, n_classes)
            Class probabilities for samples in X.
        """
        decision = self.decision_function(X)
        
        if len(self.classes_) == 2:
            # Sigmoid function for binary case
            proba_positive = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba_positive, proba_positive])
        else:
            # Softmax for multiclass case
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        return np.mean(self.predict(X) == y)
