"""
Improved data generation for realistic missing data benchmarks.
"""

import numpy as np
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def generate_realistic_dataset(dataset_type='medium', n_samples=1000, n_features=10, 
                              test_size=0.3, random_state=42):
    """
    Generate realistic datasets with controlled difficulty levels.
    
    Parameters
    ----------
    dataset_type : str
        'easy', 'medium', 'hard', or 'nonlinear'
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    test_size : float
        Fraction of data for testing
    random_state : int
        Random seed
        
    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Split dataset ready for training and testing
    """
    
    if dataset_type == 'easy':
        # Well-separated linear classes
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features-2,
            n_redundant=1,
            n_clusters_per_class=1,
            class_sep=2.0,
            flip_y=0.01,
            random_state=random_state
        )
        
    elif dataset_type == 'medium':
        # Moderately separated classes
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(4, n_features//2),
            n_redundant=2,
            n_clusters_per_class=2,
            class_sep=1.0,
            flip_y=0.02,
            random_state=random_state
        )
        
    elif dataset_type == 'hard':
        # Overlapping classes
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(3, n_features//2),
            n_redundant=min(1, n_features//4),
            n_clusters_per_class=2,
            class_sep=0.5,
            flip_y=0.05,
            random_state=random_state
        )
        
    elif dataset_type == 'nonlinear':
        # Non-linear decision boundary (concentric circles)
        np.random.seed(random_state)
        
        # Generate two concentric distributions
        n_class0 = n_samples // 2
        n_class1 = n_samples - n_class0
        
        # Inner circle (class 0)
        angles0 = np.random.uniform(0, 2*np.pi, n_class0)
        radii0 = np.random.normal(1.0, 0.3, n_class0)
        X_class0 = np.column_stack([
            radii0 * np.cos(angles0),
            radii0 * np.sin(angles0)
        ])
        
        # Outer circle (class 1) 
        angles1 = np.random.uniform(0, 2*np.pi, n_class1)
        radii1 = np.random.normal(3.0, 0.5, n_class1)
        X_class1 = np.column_stack([
            radii1 * np.cos(angles1),
            radii1 * np.sin(angles1)
        ])
        
        # Combine and add extra features
        X_circles = np.vstack([X_class0, X_class1])
        
        # Add additional random features
        if n_features > 2:
            extra_features = np.random.randn(n_samples, n_features - 2)
            X = np.column_stack([X_circles, extra_features])
        else:
            X = X_circles
            
        y = np.hstack([np.zeros(n_class0), np.ones(n_class1)])
        
        # Shuffle
        perm = np.random.permutation(n_samples)
        X, y = X[perm], y[perm]
        
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def introduce_missing_data(X, missing_rate=0.2, pattern='mcar', random_state=None):
    """
    Introduce different patterns of missing data.
    
    Parameters
    ----------
    X : array-like
        Data matrix
    missing_rate : float
        Overall proportion of missing values
    pattern : str
        'mcar' (Missing Completely At Random)
        'mar' (Missing At Random - depends on other features)
        'mnar' (Missing Not At Random - depends on value itself)
    random_state : int
        Random seed
        
    Returns
    -------
    X_missing : array
        Data with missing values (NaN)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X_missing = X.copy()
    n_samples, n_features = X.shape
    
    if pattern == 'mcar':
        # Missing Completely At Random
        missing_mask = np.random.rand(n_samples, n_features) < missing_rate
        X_missing[missing_mask] = np.nan
        
    elif pattern == 'mar':
        # Missing At Random - probability depends on first feature
        # Features are more likely to be missing if first feature is extreme
        first_feature_norm = np.abs(X[:, 0])
        first_feature_norm = first_feature_norm / (np.max(first_feature_norm) + 1e-8)
        
        for i in range(n_samples):
            # Probability increases with extreme values of first feature
            base_prob = missing_rate
            feature_prob = base_prob * (1 + first_feature_norm[i])
            
            for j in range(1, n_features):  # Skip first feature
                if np.random.rand() < feature_prob:
                    X_missing[i, j] = np.nan
                    
    elif pattern == 'mnar':
        # Missing Not At Random - probability depends on the value itself
        for j in range(n_features):
            feature_vals = X[:, j]
            
            # Higher probability for extreme values (high or low)
            feature_normalized = np.abs(feature_vals - np.median(feature_vals))
            feature_normalized = feature_normalized / (np.max(feature_normalized) + 1e-8)
            
            probabilities = missing_rate * (1 + feature_normalized)
            missing_mask = np.random.rand(n_samples) < probabilities
            X_missing[missing_mask, j] = np.nan
            
    else:
        raise ValueError(f"Unknown missing pattern: {pattern}")
    
    return X_missing


def compute_baseline_difficulty(X_train, X_test, y_train, y_test):
    """
    Compute baseline SVM accuracy to assess problem difficulty.
    
    Returns
    -------
    accuracy : float
        Baseline accuracy without missing data
    """
    from sklearn.svm import SVC
    
    baseline_svm = SVC(kernel='rbf', gamma='scale', C=1.0)
    baseline_svm.fit(X_train, y_train)
    accuracy = baseline_svm.score(X_test, y_test)
    
    return accuracy
