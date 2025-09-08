"""
Benchmark script to reproduce the results table from the technical documentation.

This script systematically evaluates different kernels with varying missing rates
to generate the performance comparison table.
"""

import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

from scikit_missing.svm import mSVM
from scikit_missing.kernels import ExpectedValueKernel, CrossCorrelationKernel


def generate_synthetic_data(n_samples=1000, n_features=10, random_state=42):
    """Generate synthetic classification data."""
    np.random.seed(random_state)
    
    # Create two classes with different characteristics
    # Class 0: centered around origin
    class0_size = n_samples // 2
    X_class0 = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features) * 0.5,
        size=class0_size
    )
    
    # Class 1: shifted
    class1_size = n_samples - class0_size
    mean_class1 = np.ones(n_features) * 1.5
    X_class1 = np.random.multivariate_normal(
        mean=mean_class1,
        cov=np.eye(n_features) * 0.8,
        size=class1_size
    )
    
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(class0_size), np.ones(class1_size)])
    
    return X, y


def introduce_missing_values(X, missing_rate, random_state=None):
    """Introduce missing values completely at random (MCAR)."""
    if random_state is not None:
        np.random.seed(random_state)
    
    X_missing = X.copy()
    n_samples, n_features = X.shape
    
    # Create random mask for missing values
    missing_mask = np.random.rand(n_samples, n_features) < missing_rate
    X_missing[missing_mask] = np.nan
    
    return X_missing


def evaluate_methods(X_train, X_test, y_train, y_test, missing_rate, n_runs=5):
    """Evaluate different methods and return mean ± std performance."""
    
    results = {
        'Expected Value': [],
        'Cross Correlation': [],
        'Standard SVM + Mean Imputation': []
    }
    
    for run in range(n_runs):
        # Introduce missing values with different random seeds
        X_train_missing = introduce_missing_values(X_train, missing_rate, random_state=42+run)
        X_test_missing = introduce_missing_values(X_test, missing_rate, random_state=100+run)
        
        try:
            # Expected Value Kernel
            kernel_ev = ExpectedValueKernel(gamma=1.0)
            clf_ev = mSVM(kernel=kernel_ev, C=1.0)
            clf_ev.fit(X_train_missing, y_train)
            acc_ev = clf_ev.score(X_test_missing, y_test)
            results['Expected Value'].append(acc_ev)
            
            # Cross Correlation Kernel
            kernel_cc = CrossCorrelationKernel(gamma=1.0)
            clf_cc = mSVM(kernel=kernel_cc, C=1.0)
            clf_cc.fit(X_train_missing, y_train)
            acc_cc = clf_cc.score(X_test_missing, y_test)
            results['Cross Correlation'].append(acc_cc)
            
            # Standard SVM with Mean Imputation
            imputer = SimpleImputer(strategy='mean')
            X_train_imputed = imputer.fit_transform(X_train_missing)
            X_test_imputed = imputer.transform(X_test_missing)
            
            clf_standard = SVC(kernel='rbf', gamma=1.0, C=1.0)
            clf_standard.fit(X_train_imputed, y_train)
            acc_standard = clf_standard.score(X_test_imputed, y_test)
            results['Standard SVM + Mean Imputation'].append(acc_standard)
            
        except Exception as e:
            print(f"Error in run {run} with missing rate {missing_rate}: {e}")
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


def run_benchmark():
    """Run the complete benchmark and generate results table."""
    print("=== mSVM Benchmark: Missing Data Performance Comparison ===")
    print()
    
    # Generate synthetic data
    print("Generating synthetic dataset...")
    X, y = generate_synthetic_data(n_samples=1000, n_features=10, random_state=42)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    print()
    
    # Missing rates to evaluate
    missing_rates = [0.1, 0.2, 0.3, 0.4]
    
    # Store all results
    all_results = {}
    
    print("Running benchmarks...")
    for missing_rate in missing_rates:
        print(f"\nEvaluating with {missing_rate:.0%} missing data...")
        
        results = evaluate_methods(X_train, X_test, y_train, y_test, missing_rate, n_runs=5)
        all_results[missing_rate] = results
        
        # Print results for this missing rate
        for method, stats in results.items():
            if not np.isnan(stats['mean']):
                print(f"  {method}: {stats['mean']:.3f} ± {stats['std']:.3f}")
            else:
                print(f"  {method}: Failed")
    
    # Generate final table
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    print()
    print("| Missing Rate | Expected Value | Cross Correlation | Standard SVM + Mean Imputation |")
    print("|--------------|----------------|-------------------|--------------------------------|")
    
    for missing_rate in missing_rates:
        results = all_results[missing_rate]
        
        ev_result = results['Expected Value']
        cc_result = results['Cross Correlation']
        std_result = results['Standard SVM + Mean Imputation']
        
        ev_str = f"{ev_result['mean']:.2f} ± {ev_result['std']:.2f}" if not np.isnan(ev_result['mean']) else "Failed"
        cc_str = f"{cc_result['mean']:.2f} ± {cc_result['std']:.2f}" if not np.isnan(cc_result['mean']) else "Failed"
        std_str = f"{std_result['mean']:.2f} ± {std_result['std']:.2f}" if not np.isnan(std_result['mean']) else "Failed"
        
        print(f"| {missing_rate:.0%} | {ev_str} | {cc_str} | {std_str} |")
    
    print()
    print("Note: Results are mean ± standard deviation over 5 runs with different missing patterns")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    try:
        results = run_benchmark()
        print("\nBenchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
