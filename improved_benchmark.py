"""
Improved benchmark script with more realistic data and better experimental design.

This version addresses several issues:
1. More complex/realistic synthetic data
2. Different missing data patterns
3. Hyperparameter tuning
4. Better experimental design
"""

import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

from scikit_missing.svm import mSVM
from scikit_missing.kernels import ExpectedValueKernel, CrossCorrelationKernel


def generate_realistic_data(n_samples=1000, n_features=10, n_informative=6, 
                          n_redundant=2, n_clusters_per_class=2, 
                          class_sep=0.8, random_state=42):
    """Generate more realistic classification data with controlled difficulty."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,  # Lower = harder problem
        flip_y=0.02,  # Add some label noise
        random_state=random_state
    )
    return X, y


def introduce_missing_patterns(X, missing_rate, pattern='mcar', random_state=None):
    """Introduce different patterns of missing data."""
    if random_state is not None:
        np.random.seed(random_state)
    
    X_missing = X.copy()
    n_samples, n_features = X.shape
    
    if pattern == 'mcar':
        # Missing Completely At Random
        missing_mask = np.random.rand(n_samples, n_features) < missing_rate
        X_missing[missing_mask] = np.nan
        
    elif pattern == 'mar':
        # Missing At Random - probability depends on other observed features
        # Higher chance of missing if first feature is extreme
        feature_probs = np.abs(X[:, 0]) / (np.abs(X[:, 0]).max() + 1e-8)
        feature_probs = feature_probs * missing_rate * 2  # Scale to desired rate
        
        for i in range(n_samples):
            for j in range(1, n_features):  # Skip first feature (it's always observed)
                if np.random.rand() < feature_probs[i]:
                    X_missing[i, j] = np.nan
                    
    elif pattern == 'mnar':
        # Missing Not At Random - probability depends on the value itself
        for j in range(n_features):
            feature_vals = X[:, j]
            # More likely to be missing if value is extreme (high or low)
            probs = np.abs(feature_vals - np.median(feature_vals))
            probs = probs / (probs.max() + 1e-8) * missing_rate * 2
            
            missing_indices = np.random.rand(n_samples) < probs
            X_missing[missing_indices, j] = np.nan
    
    return X_missing


def tune_hyperparameters(X_train, y_train, kernel_class, param_grid, cv=3):
    """Simple grid search for hyperparameter tuning."""
    best_score = -1
    best_params = None
    
    for gamma in param_grid.get('gamma', [1.0]):
        for C in param_grid.get('C', [1.0]):
            try:
                kernel = kernel_class(gamma=gamma)
                clf = mSVM(kernel=kernel, C=C)
                
                # Use cross-validation for more robust evaluation
                scores = []
                for fold in range(cv):
                    # Simple train/validation split
                    train_idx = np.random.choice(len(X_train), size=int(len(X_train)*0.8), replace=False)
                    val_idx = np.setdiff1d(np.arange(len(X_train)), train_idx)
                    
                    clf.fit(X_train[train_idx], y_train[train_idx])
                    score = clf.score(X_train[val_idx], y_train[val_idx])
                    scores.append(score)
                
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'gamma': gamma, 'C': C}
                    
            except Exception as e:
                print(f"Error with gamma={gamma}, C={C}: {e}")
                continue
    
    return best_params if best_params else {'gamma': 1.0, 'C': 1.0}


def evaluate_methods_improved(X_train, X_test, y_train, y_test, missing_rate, 
                            missing_pattern='mcar', n_runs=5, tune_params=True):
    """Improved evaluation with hyperparameter tuning and better experimental design."""
    
    results = {
        'Expected Value': [],
        'Cross Correlation': [],
        'Standard SVM + Mean Imputation': []
    }
    
    # Hyperparameter grids
    param_grid = {
        'gamma': [0.1, 0.5, 1.0, 2.0, 5.0],
        'C': [0.1, 1.0, 10.0]
    }
    
    # Tune hyperparameters on clean data first
    best_params_ev = {'gamma': 1.0, 'C': 1.0}
    best_params_cc = {'gamma': 1.0, 'C': 1.0}
    
    if tune_params:
        print(f"  Tuning hyperparameters for {missing_rate:.0%} missing data...")
        # Use a small sample of missing data for tuning
        X_tune_missing = introduce_missing_patterns(X_train[:200], missing_rate, missing_pattern, 42)
        y_tune = y_train[:200]
        
        best_params_ev = tune_hyperparameters(X_tune_missing, y_tune, ExpectedValueKernel, param_grid)
        best_params_cc = tune_hyperparameters(X_tune_missing, y_tune, CrossCorrelationKernel, param_grid)
        
        print(f"    Best EV params: {best_params_ev}")
        print(f"    Best CC params: {best_params_cc}")
    
    for run in range(n_runs):
        # Introduce missing values with different random seeds
        X_train_missing = introduce_missing_patterns(X_train, missing_rate, missing_pattern, 42+run)
        X_test_missing = introduce_missing_patterns(X_test, missing_rate, missing_pattern, 100+run)
        
        try:
            # Expected Value Kernel
            kernel_ev = ExpectedValueKernel(gamma=best_params_ev['gamma'])
            clf_ev = mSVM(kernel=kernel_ev, C=best_params_ev['C'])
            clf_ev.fit(X_train_missing, y_train)
            acc_ev = clf_ev.score(X_test_missing, y_test)
            results['Expected Value'].append(acc_ev)
            
            # Cross Correlation Kernel
            kernel_cc = CrossCorrelationKernel(gamma=best_params_cc['gamma'])
            clf_cc = mSVM(kernel=kernel_cc, C=best_params_cc['C'])
            clf_cc.fit(X_train_missing, y_train)
            acc_cc = clf_cc.score(X_test_missing, y_test)
            results['Cross Correlation'].append(acc_cc)
            
            # Standard SVM with Mean Imputation
            imputer = SimpleImputer(strategy='mean')
            X_train_imputed = imputer.fit_transform(X_train_missing)
            X_test_imputed = imputer.transform(X_test_missing)
            
            clf_standard = SVC(kernel='rbf', gamma=best_params_ev['gamma'], C=best_params_ev['C'])
            clf_standard.fit(X_train_imputed, y_train)
            acc_standard = clf_standard.score(X_test_imputed, y_test)
            results['Standard SVM + Mean Imputation'].append(acc_standard)
            
        except Exception as e:
            print(f"    Error in run {run}: {e}")
            continue
    
    # Calculate mean and std for each method
    summary = {}
    for method, scores in results.items():
        if scores:  # Only if we have valid scores
            summary[method] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        else:
            summary[method] = {
                'mean': np.nan,
                'std': np.nan,
                'scores': []
            }
    
    return summary


def run_improved_benchmark():
    """Run the improved benchmark with more realistic conditions."""
    print("=== Improved mSVM Benchmark: Realistic Missing Data Performance ===")
    print()
    
    # Test different data complexities and missing patterns
    test_configs = [
        {'class_sep': 1.2, 'pattern': 'mcar', 'name': 'Easy Problem (MCAR)'},
        {'class_sep': 0.8, 'pattern': 'mcar', 'name': 'Medium Problem (MCAR)'},
        {'class_sep': 0.5, 'pattern': 'mcar', 'name': 'Hard Problem (MCAR)'},
        {'class_sep': 0.8, 'pattern': 'mar', 'name': 'Medium Problem (MAR)'},
        {'class_sep': 0.8, 'pattern': 'mnar', 'name': 'Medium Problem (MNAR)'},
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        # Generate data with specified difficulty
        X, y = generate_realistic_data(
            n_samples=800, 
            n_features=8, 
            class_sep=config['class_sep'],
            random_state=42
        )
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        print(f"Baseline accuracy (no missing data): ", end="")
        
        # Check baseline performance
        baseline_clf = SVC(kernel='rbf', gamma=1.0, C=1.0)
        baseline_clf.fit(X_train, y_train)
        baseline_acc = baseline_clf.score(X_test, y_test)
        print(f"{baseline_acc:.3f}")
        
        # Missing rates to evaluate
        missing_rates = [0.1, 0.2, 0.3, 0.4]
        
        print(f"\n| Missing Rate | Expected Value | Cross Correlation | Standard SVM + Mean Imputation |")
        print(f"|--------------|----------------|-------------------|--------------------------------|")
        
        for missing_rate in missing_rates:
            results = evaluate_methods_improved(
                X_train, X_test, y_train, y_test, 
                missing_rate, 
                missing_pattern=config['pattern'], 
                n_runs=3,  # Fewer runs for speed
                tune_params=(missing_rate == 0.1)  # Only tune for first rate
            )
            
            ev_result = results['Expected Value']
            cc_result = results['Cross Correlation']
            std_result = results['Standard SVM + Mean Imputation']
            
            ev_str = f"{ev_result['mean']:.2f} ± {ev_result['std']:.2f}" if not np.isnan(ev_result['mean']) else "Failed"
            cc_str = f"{cc_result['mean']:.2f} ± {cc_result['std']:.2f}" if not np.isnan(cc_result['mean']) else "Failed"
            std_str = f"{std_result['mean']:.2f} ± {std_result['std']:.2f}" if not np.isnan(std_result['mean']) else "Failed"
            
            print(f"| {missing_rate:.0%} | {ev_str} | {cc_str} | {std_str} |")


if __name__ == "__main__":
    try:
        run_improved_benchmark()
        print(f"\n{'='*60}")
        print("Improved benchmark completed!")
        print(f"{'='*60}")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
