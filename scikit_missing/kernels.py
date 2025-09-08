"""
Kernel implementations for handling missing data in SVM.

This module provides various kernel functions that can work with missing features
represented as NaN values.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union
from scipy.spatial.distance import pdist, squareform


class BaseMissingKernel(ABC):
    """
    Abstract base class for kernels that can handle missing data.
    
    All missing data kernels should inherit from this class and implement
    the compute_kernel method.
    """
    
    def __init__(self):
        self.X_train_ = None
        self.missing_stats_ = None
    
    @abstractmethod
    def compute_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the kernel matrix between X and Y.
        
        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Input data matrix, may contain NaN values for missing features.
        Y : array-like of shape (n_samples_Y, n_features), optional
            Second input data matrix. If None, compute kernel matrix for X with itself.
            
        Returns
        -------
        K : array-like of shape (n_samples_X, n_samples_Y)
            Kernel matrix.
        """
        pass
    
    def fit(self, X: np.ndarray) -> 'BaseMissingKernel':
        """
        Fit the kernel to the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix, may contain NaN values.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_train_ = X.copy()
        self._compute_missing_stats(X)
        return self
    
    def _compute_missing_stats(self, X: np.ndarray) -> None:
        """
        Compute statistics for missing data handling.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data matrix.
        """
        # Compute mean and variance for each feature using available data
        means = np.nanmean(X, axis=0)
        vars = np.nanvar(X, axis=0)
        
        self.missing_stats_ = {
            'means': means,
            'vars': vars,
            'n_features': X.shape[1]
        }
    
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Make the kernel callable."""
        return self.compute_kernel(X, Y)


class ExpectedValueKernel(BaseMissingKernel):
    """
    Expected value kernel for missing data.
    
    This kernel replaces missing values with their expected values (means)
    computed from the training data, then applies a standard RBF kernel.
    
    Parameters
    ----------
    gamma : float, default=1.0
        Kernel coefficient for RBF kernel.
    """
    
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
    
    def compute_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute expected value kernel matrix.
        
        Missing values are replaced with feature means from training data.
        """
        if self.missing_stats_ is None:
            raise ValueError("Kernel must be fitted before computing kernel matrix.")
        
        X_filled = self._fill_missing_with_mean(X)
        
        if Y is None:
            Y_filled = X_filled
        else:
            Y_filled = self._fill_missing_with_mean(Y)
        
        # Compute RBF kernel with filled data
        return self._rbf_kernel(X_filled, Y_filled)
    
    def _fill_missing_with_mean(self, X: np.ndarray) -> np.ndarray:
        """Fill missing values with feature means."""
        X_filled = X.copy()
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if np.any(mask):
                X_filled[mask, i] = self.missing_stats_['means'][i]
        return X_filled
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix."""
        # Compute squared Euclidean distances
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True).T
        distances_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        
        # Apply RBF kernel
        return np.exp(-self.gamma * distances_sq)


class CrossCorrelationKernel(BaseMissingKernel):
    """
    Cross-correlation kernel for missing data.
    
    This kernel handles missing features by modeling them as Gaussian random
    variables with means and variances computed from the training data. The
    kernel computes the expected value of the RBF kernel over the distribution
    of missing features.
    
    Parameters
    ----------
    gamma : float, default=1.0
        Kernel coefficient for RBF kernel.
    """
    
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
    
    def compute_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cross-correlation kernel matrix.
        
        For pairs of samples with missing features, the kernel computes the
        expected value of the RBF kernel assuming missing features follow
        Gaussian distributions.
        """
        if self.missing_stats_ is None:
            raise ValueError("Kernel must be fitted before computing kernel matrix.")
        
        if Y is None:
            Y = X
        
        # Use vectorized computation for better performance
        return self._compute_kernel_vectorized(X, Y)
    
    def _compute_pairwise_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute kernel value between two samples handling missing features.
        
        Parameters
        ----------
        x, y : array-like of shape (n_features,)
            Two samples to compute kernel between.
            
        Returns
        -------
        kernel_value : float
            Kernel value between x and y.
        """
        # Identify missing features in both samples
        x_missing = np.isnan(x)
        y_missing = np.isnan(y)
        both_missing = x_missing & y_missing
        either_missing = x_missing | y_missing
        both_present = ~either_missing
        
        # Compute contribution from features present in both samples
        if np.any(both_present):
            diff_present = x[both_present] - y[both_present]
            dist_sq_present = np.sum(diff_present**2)
        else:
            dist_sq_present = 0.0
        
        # Handle features missing in one or both samples
        expected_dist_sq_missing = 0.0
        
        for k in range(len(x)):
            if both_missing[k]:
                # Both missing: E[(X_k - Y_k)^2] = Var[X_k] + Var[Y_k] = 2*Var[X_k]
                expected_dist_sq_missing += 2 * self.missing_stats_['vars'][k]
            elif x_missing[k] and not y_missing[k]:
                # Only x missing: E[(X_k - y_k)^2] = Var[X_k] + (μ_k - y_k)^2
                mean_k = self.missing_stats_['means'][k]
                var_k = self.missing_stats_['vars'][k]
                expected_dist_sq_missing += var_k + (mean_k - y[k])**2
            elif y_missing[k] and not x_missing[k]:
                # Only y missing: E[(x_k - Y_k)^2] = Var[Y_k] + (x_k - μ_k)^2
                mean_k = self.missing_stats_['means'][k]
                var_k = self.missing_stats_['vars'][k]
                expected_dist_sq_missing += var_k + (x[k] - mean_k)**2
        
        # Total expected squared distance
        total_expected_dist_sq = dist_sq_present + expected_dist_sq_missing
        
        # Apply RBF kernel
        return np.exp(-self.gamma * total_expected_dist_sq)

    def _compute_kernel_vectorized(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Vectorized computation of cross-correlation kernel matrix.
        
        This is much faster than the pairwise approach for larger datasets.
        """
        n_X, n_Y = X.shape[0], Y.shape[0]
        n_features = X.shape[1]
        
        # Create masks for missing values
        X_missing = np.isnan(X)  # (n_X, n_features)
        Y_missing = np.isnan(Y)  # (n_Y, n_features)
        
        # Initialize kernel matrix
        K = np.zeros((n_X, n_Y))
        
        # For better performance, we'll handle the most common cases efficiently
        # and fall back to pairwise computation only when necessary
        
        # Check if we have any missing values at all
        if not np.any(X_missing) and not np.any(Y_missing):
            # No missing values - use standard RBF kernel
            X_norm = np.sum(X**2, axis=1, keepdims=True)
            Y_norm = np.sum(Y**2, axis=1, keepdims=True).T
            distances_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)
            return np.exp(-self.gamma * distances_sq)
        
        # If missing values are sparse, use optimized computation
        missing_rate_X = np.mean(X_missing)
        missing_rate_Y = np.mean(Y_missing)
        
        if missing_rate_X < 0.1 and missing_rate_Y < 0.1:
            # Low missing rate - use optimized computation
            return self._compute_kernel_sparse_missing(X, Y, X_missing, Y_missing)
        else:
            # High missing rate - fall back to pairwise computation
            # but with some optimizations
            return self._compute_kernel_pairwise_optimized(X, Y, X_missing, Y_missing)
    
    def _compute_kernel_sparse_missing(self, X: np.ndarray, Y: np.ndarray, 
                                     X_missing: np.ndarray, Y_missing: np.ndarray) -> np.ndarray:
        """Optimized computation for sparse missing data."""
        n_X, n_Y = X.shape[0], Y.shape[0]
        n_features = X.shape[1]
        
        # Pre-compute variance terms
        vars_array = np.array(self.missing_stats_['vars'])
        means_array = np.array(self.missing_stats_['means'])
        
        K = np.zeros((n_X, n_Y))
        
        # Process in chunks to balance memory usage and speed
        chunk_size = min(100, n_X)
        
        for i_start in range(0, n_X, chunk_size):
            i_end = min(i_start + chunk_size, n_X)
            X_chunk = X[i_start:i_end]
            X_missing_chunk = X_missing[i_start:i_end]
            
            # Vectorized computation for this chunk
            K_chunk = self._compute_chunk_kernel(X_chunk, Y, X_missing_chunk, Y_missing, 
                                               vars_array, means_array)
            K[i_start:i_end] = K_chunk
        
        return K
    
    def _compute_chunk_kernel(self, X_chunk: np.ndarray, Y: np.ndarray,
                            X_missing_chunk: np.ndarray, Y_missing: np.ndarray,
                            vars_array: np.ndarray, means_array: np.ndarray) -> np.ndarray:
        """Compute kernel for a chunk of X against all of Y."""
        n_chunk, n_Y = X_chunk.shape[0], Y.shape[0]
        K_chunk = np.zeros((n_chunk, n_Y))
        
        for i in range(n_chunk):
            for j in range(n_Y):
                # Use the original pairwise computation for now
                # TODO: Further vectorize this if needed
                K_chunk[i, j] = self._compute_pairwise_kernel_optimized(
                    X_chunk[i], Y[j], X_missing_chunk[i], Y_missing[j],
                    vars_array, means_array
                )
        
        return K_chunk
    
    def _compute_kernel_pairwise_optimized(self, X: np.ndarray, Y: np.ndarray,
                                         X_missing: np.ndarray, Y_missing: np.ndarray) -> np.ndarray:
        """Optimized pairwise computation for high missing rates."""
        n_X, n_Y = X.shape[0], Y.shape[0]
        K = np.zeros((n_X, n_Y))
        
        # Pre-compute variance and mean arrays for faster access
        vars_array = np.array(self.missing_stats_['vars'])
        means_array = np.array(self.missing_stats_['means'])
        
        for i in range(n_X):
            for j in range(n_Y):
                K[i, j] = self._compute_pairwise_kernel_optimized(
                    X[i], Y[j], X_missing[i], Y_missing[j],
                    vars_array, means_array
                )
        
        return K
    
    def _compute_pairwise_kernel_optimized(self, x: np.ndarray, y: np.ndarray,
                                         x_missing: np.ndarray, y_missing: np.ndarray,
                                         vars_array: np.ndarray, means_array: np.ndarray) -> float:
        """Optimized pairwise kernel computation."""
        # Identify missing patterns
        both_missing = x_missing & y_missing
        either_missing = x_missing | y_missing
        both_present = ~either_missing
        
        # Compute contribution from features present in both samples
        dist_sq_present = 0.0
        if np.any(both_present):
            diff_present = x[both_present] - y[both_present]
            dist_sq_present = np.sum(diff_present**2)
        
        # Handle features missing in one or both samples
        expected_dist_sq_missing = 0.0
        
        # Vectorized computation for missing features
        if np.any(both_missing):
            # Both missing: E[(X_k - Y_k)^2] = 2*Var[X_k]
            expected_dist_sq_missing += np.sum(2 * vars_array[both_missing])
        
        # Handle x missing, y present
        x_only_missing = x_missing & ~y_missing
        if np.any(x_only_missing):
            # E[(X_k - y_k)^2] = Var[X_k] + (μ_k - y_k)^2
            mean_diff_sq = (means_array[x_only_missing] - y[x_only_missing])**2
            expected_dist_sq_missing += np.sum(vars_array[x_only_missing] + mean_diff_sq)
        
        # Handle y missing, x present
        y_only_missing = y_missing & ~x_missing
        if np.any(y_only_missing):
            # E[(x_k - Y_k)^2] = Var[Y_k] + (x_k - μ_k)^2
            mean_diff_sq = (x[y_only_missing] - means_array[y_only_missing])**2
            expected_dist_sq_missing += np.sum(vars_array[y_only_missing] + mean_diff_sq)
        
        # Total expected squared distance
        total_expected_dist_sq = dist_sq_present + expected_dist_sq_missing
        
        # Apply RBF kernel
        return np.exp(-self.gamma * total_expected_dist_sq)


class LinearMissingKernel(BaseMissingKernel):
    """
    Linear kernel for missing data.
    
    This kernel computes the dot product between samples, handling missing
    features by using expected values.
    """
    
    def __init__(self):
        super().__init__()
    
    def compute_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute linear kernel matrix with missing data handling.
        """
        if self.missing_stats_ is None:
            raise ValueError("Kernel must be fitted before computing kernel matrix.")
        
        X_filled = self._fill_missing_with_mean(X)
        
        if Y is None:
            Y_filled = X_filled
        else:
            Y_filled = self._fill_missing_with_mean(Y)
        
        return np.dot(X_filled, Y_filled.T)
    
    def _fill_missing_with_mean(self, X: np.ndarray) -> np.ndarray:
        """Fill missing values with feature means."""
        X_filled = X.copy()
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if np.any(mask):
                X_filled[mask, i] = self.missing_stats_['means'][i]
        return X_filled


class PolynomialMissingKernel(BaseMissingKernel):
    """
    Polynomial kernel for missing data.
    
    Parameters
    ----------
    degree : int, default=3
        Degree of the polynomial kernel.
    coef0 : float, default=1.0
        Independent term in the polynomial kernel.
    """
    
    def __init__(self, degree: int = 3, coef0: float = 1.0):
        super().__init__()
        self.degree = degree
        self.coef0 = coef0
    
    def compute_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute polynomial kernel matrix with missing data handling.
        """
        if self.missing_stats_ is None:
            raise ValueError("Kernel must be fitted before computing kernel matrix.")
        
        X_filled = self._fill_missing_with_mean(X)
        
        if Y is None:
            Y_filled = X_filled
        else:
            Y_filled = self._fill_missing_with_mean(Y)
        
        # Compute polynomial kernel: (X * Y^T + coef0)^degree
        linear_kernel = np.dot(X_filled, Y_filled.T)
        return (linear_kernel + self.coef0) ** self.degree
    
    def _fill_missing_with_mean(self, X: np.ndarray) -> np.ndarray:
        """Fill missing values with feature means."""
        X_filled = X.copy()
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if np.any(mask):
                X_filled[mask, i] = self.missing_stats_['means'][i]
        return X_filled
