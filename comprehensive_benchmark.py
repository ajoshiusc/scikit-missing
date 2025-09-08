"""
Comprehensive benchmark for mSVM with proper experimental design.
"""

import numpy as np
import sys
import os
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import time

# Add the package to the path
sys.path.insert(0, os.path.dirname(__file__))

from scikit_missing.svm import mSVM
from scikit_missing.kernels import ExpectedValueKernel, CrossCorrelationKernel
from improved_data_generation import (
    generate_realistic_dataset, 
    introduce_missing_data, 
    compute_baseline_difficulty
)


def tune_hyperparameters(X_train, y_train, kernel_class, missing_rate=0.2, cv=3):
    """
    Tune hyperparameters using cross-validation.
    
    Returns
    -------
    best_params : dict
        Best hyperparameters found
    best_score : float
        Best cross-validation score
    """
    # Parameter grid
    if kernel_class == ExpectedValueKernel:
        param_grid = {
            'gamma': [0.01, 0.1, 0.5, 1.0, 2.0],
            'C': [0.1, 1.0, 10.0]
        }
    else:  # CrossCorrelationKernel
        param_grid = {
            'gamma': [0.01, 0.05, 0.1, 0.2, 0.5],
            'C': [0.1, 1.0, 10.0]
        }
    
    best_score = -1
    best_params = None
    
    # Introduce missing data for tuning
    X_train_missing = introduce_missing_data(X_train, missing_rate, 'mcar', 42)
    
    # Grid search
    for gamma in param_grid['gamma']:
        for C in param_grid['C']:
            try:
                # Cross-validation
                scores = []
                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                
                for train_idx, val_idx in skf.split(X_train_missing, y_train):
                    X_train_fold = X_train_missing[train_idx]
                    X_val_fold = X_train_missing[val_idx]
                    y_train_fold = y_train[train_idx]
                    y_val_fold = y_train[val_idx]
                    
                    # Train model
                    kernel = kernel_class(gamma=gamma)
                    clf = mSVM(kernel=kernel, C=C)
                    clf.fit(X_train_fold, y_train_fold)
                    
                    # Evaluate
                    score = clf.score(X_val_fold, y_val_fold)
                    scores.append(score)
                
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'gamma': gamma, 'C': C}
                    
            except Exception as e:
                # Skip this parameter combination if it fails
                continue
    
    return best_params if best_params else {'gamma': 1.0, 'C': 1.0}, best_score


def evaluate_method(method_name, X_train, X_test, y_train, y_test, 
                   missing_rates, missing_patterns, n_runs=5, tune_params=True):
    """
    Evaluate a single method across different conditions.
    
    Returns
    -------
    results : dict
        Nested dictionary with results
    """
    results = {}
    
    for pattern in missing_patterns:
        results[pattern] = {}
        
        for missing_rate in missing_rates:
            print(f"  {method_name} - {pattern} - {missing_rate:.0%}")
            
            scores = []
            times = []
            
            for run in range(n_runs):
                try:
                    # Generate missing data
                    X_train_missing = introduce_missing_data(
                        X_train, missing_rate, pattern, random_state=42+run
                    )
                    X_test_missing = introduce_missing_data(
                        X_test, missing_rate, pattern, random_state=100+run
                    )
                    
                    start_time = time.time()
                    
                    if method_name == 'Expected Value':
                        if tune_params and run == 0:  # Tune only on first run
                            best_params, _ = tune_hyperparameters(
                                X_train, y_train, ExpectedValueKernel, missing_rate
                            )
                        else:
                            best_params = {'gamma': 0.1, 'C': 1.0}
                        
                        kernel = ExpectedValueKernel(gamma=best_params['gamma'])
                        clf = mSVM(kernel=kernel, C=best_params['C'])
                        clf.fit(X_train_missing, y_train)
                        score = clf.score(X_test_missing, y_test)
                        
                    elif method_name == 'Cross Correlation':
                        if tune_params and run == 0:  # Tune only on first run
                            best_params, _ = tune_hyperparameters(
                                X_train, y_train, CrossCorrelationKernel, missing_rate
                            )
                        else:
                            best_params = {'gamma': 0.1, 'C': 1.0}
                        
                        kernel = CrossCorrelationKernel(gamma=best_params['gamma'])
                        clf = mSVM(kernel=kernel, C=best_params['C'])
                        clf.fit(X_train_missing, y_train)
                        score = clf.score(X_test_missing, y_test)
                        
                    elif method_name == 'Standard SVM + Mean Imputation':
                        # Impute missing values
                        imputer = SimpleImputer(strategy='mean')
                        X_train_imputed = imputer.fit_transform(X_train_missing)
                        X_test_imputed = imputer.transform(X_test_missing)
                        
                        # Standard SVM
                        clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
                        clf.fit(X_train_imputed, y_train)
                        score = clf.score(X_test_imputed, y_test)
                    
                    elif method_name == 'Standard SVM + Median Imputation':
                        # Impute missing values
                        imputer = SimpleImputer(strategy='median')
                        X_train_imputed = imputer.fit_transform(X_train_missing)
                        X_test_imputed = imputer.transform(X_test_missing)
                        
                        # Standard SVM
                        clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
                        clf.fit(X_train_imputed, y_train)
                        score = clf.score(X_test_imputed, y_test)
                    
                    elapsed_time = time.time() - start_time
                    
                    scores.append(score)
                    times.append(elapsed_time)
                    
                except Exception as e:
                    print(f"    Error in run {run}: {e}")
                    continue
            
            if scores:
                results[pattern][missing_rate] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores,
                    'mean_time': np.mean(times),
                    'std_time': np.std(times)
                }
            else:
                results[pattern][missing_rate] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'scores': [],
                    'mean_time': np.nan,
                    'std_time': np.nan
                }
    
    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmark with proper experimental design."""
    print("=" * 80)
    print("COMPREHENSIVE mSVM BENCHMARK")
    print("=" * 80)
    
    # Test configurations
    datasets = ['easy', 'medium', 'hard', 'nonlinear']
    missing_rates = [0.1, 0.2, 0.3, 0.4]
    missing_patterns = ['mcar', 'mar', 'mnar']
    methods = [
        'Expected Value',
        'Cross Correlation', 
        'Standard SVM + Mean Imputation',
        'Standard SVM + Median Imputation'
    ]
    
    all_results = {}
    
    for dataset_type in datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_type.upper()}")
        print(f"{'='*60}")
        
        # Generate dataset
        X_train, X_test, y_train, y_test = generate_realistic_dataset(
            dataset_type=dataset_type,
            n_samples=800,
            n_features=8,
            random_state=42
        )
        
        # Compute baseline difficulty
        baseline_acc = compute_baseline_difficulty(X_train, X_test, y_train, y_test)
        print(f"Baseline accuracy (no missing data): {baseline_acc:.3f}")
        
        # Evaluate each method
        dataset_results = {}
        
        for method in methods:
            print(f"\nEvaluating {method}...")
            
            method_results = evaluate_method(
                method, X_train, X_test, y_train, y_test,
                missing_rates, missing_patterns, n_runs=3, tune_params=True
            )
            
            dataset_results[method] = method_results
        
        all_results[dataset_type] = {
            'baseline_accuracy': baseline_acc,
            'results': dataset_results
        }
        
        # Print summary table for this dataset
        print_dataset_summary(dataset_type, dataset_results, missing_rates)
    
    # Print overall summary
    print_overall_summary(all_results, datasets, missing_rates)
    
    return all_results


def print_dataset_summary(dataset_type, results, missing_rates):
    """Print summary table for a single dataset."""
    print(f"\n{dataset_type.upper()} DATASET SUMMARY (MCAR Pattern)")
    print("-" * 80)
    print(f"{'Missing Rate':<12} {'Expected Value':<15} {'Cross Correlation':<18} {'Mean Imputation':<15} {'Median Imputation':<16}")
    print("-" * 80)
    
    for missing_rate in missing_rates:
        row = f"{missing_rate:.0%}           "
        
        for method in ['Expected Value', 'Cross Correlation', 'Standard SVM + Mean Imputation', 'Standard SVM + Median Imputation']:
            try:
                mean_acc = results[method]['mcar'][missing_rate]['mean']
                std_acc = results[method]['mcar'][missing_rate]['std']
                if not np.isnan(mean_acc):
                    if method == 'Expected Value':
                        row += f"{mean_acc:.3f}±{std_acc:.3f}    "
                    elif method == 'Cross Correlation':
                        row += f"{mean_acc:.3f}±{std_acc:.3f}       "
                    elif method == 'Standard SVM + Mean Imputation':
                        row += f"{mean_acc:.3f}±{std_acc:.3f}    "
                    else:  # Median Imputation
                        row += f"{mean_acc:.3f}±{std_acc:.3f}     "
                else:
                    if method == 'Expected Value':
                        row += "Failed        "
                    elif method == 'Cross Correlation':
                        row += "Failed           "
                    elif method == 'Standard SVM + Mean Imputation':
                        row += "Failed        "
                    else:  # Median Imputation
                        row += "Failed         "
            except:
                if method == 'Expected Value':
                    row += "Error         "
                elif method == 'Cross Correlation':
                    row += "Error            "
                elif method == 'Standard SVM + Mean Imputation':
                    row += "Error         "
                else:  # Median Imputation
                    row += "Error          "
        
        print(row)


def print_overall_summary(all_results, datasets, missing_rates):
    """Print overall summary across all datasets."""
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    for dataset_type in datasets:
        baseline = all_results[dataset_type]['baseline_accuracy']
        print(f"\n{dataset_type.upper()} Dataset (Baseline: {baseline:.3f})")
        print_dataset_summary(dataset_type, all_results[dataset_type]['results'], missing_rates)


if __name__ == "__main__":
    try:
        results = run_comprehensive_benchmark()
        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
