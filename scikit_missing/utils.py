"""
Utility functions for scikit-missing package.

This module provides helper functions for data preprocessing, 
missing data analysis, and performance evaluation.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


def missing_data_summary(X: np.ndarray) -> Dict[str, float]:
    """
    Provide summary statistics about missing data patterns.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix with potential missing values (NaN).
        
    Returns
    -------
    summary : dict
        Dictionary containing missing data statistics.
    """
    n_samples, n_features = X.shape
    total_elements = n_samples * n_features
    
    # Count missing values
    missing_mask = np.isnan(X)
    total_missing = np.sum(missing_mask)
    
    # Per-sample missing counts
    missing_per_sample = np.sum(missing_mask, axis=1)
    
    # Per-feature missing counts  
    missing_per_feature = np.sum(missing_mask, axis=0)
    
    # Calculate statistics
    summary = {
        'total_missing': int(total_missing),
        'missing_rate': total_missing / total_elements,
        'samples_with_missing': int(np.sum(missing_per_sample > 0)),
        'samples_with_missing_rate': np.sum(missing_per_sample > 0) / n_samples,
        'features_with_missing': int(np.sum(missing_per_feature > 0)),
        'features_with_missing_rate': np.sum(missing_per_feature > 0) / n_features,
        'max_missing_per_sample': int(np.max(missing_per_sample)),
        'mean_missing_per_sample': float(np.mean(missing_per_sample)),
        'max_missing_per_feature': int(np.max(missing_per_feature)),
        'mean_missing_per_feature': float(np.mean(missing_per_feature)),
    }
    
    return summary


def visualize_missing_pattern(X: np.ndarray, max_samples: int = 100) -> str:
    """
    Create a text-based visualization of missing data patterns.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.
    max_samples : int, default=100
        Maximum number of samples to display.
        
    Returns
    -------
    visualization : str
        Text representation of missing data pattern.
    """
    n_samples, n_features = X.shape
    
    # Limit samples for display
    display_samples = min(max_samples, n_samples)
    X_display = X[:display_samples]
    
    # Create visualization
    lines = []
    lines.append(f"Missing Data Pattern (showing {display_samples}/{n_samples} samples)")
    lines.append("=" * (n_features + 10))
    
    # Header
    header = "Sample  " + "".join([f"{i:>3}" for i in range(n_features)])
    lines.append(header)
    lines.append("-" * len(header))
    
    # Data rows
    for i, row in enumerate(X_display):
        row_str = f"{i:6d}  "
        for val in row:
            if np.isnan(val):
                row_str += "  ."
            else:
                row_str += "  ●"
        lines.append(row_str)
    
    # Legend
    lines.append("-" * len(header))
    lines.append("Legend: ● = observed, . = missing")
    
    return "\n".join(lines)


def create_missing_data_report(X: np.ndarray, feature_names: Optional[List[str]] = None) -> str:
    """
    Generate a comprehensive missing data report.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.
    feature_names : list of str, optional
        Names of features. If None, generic names will be used.
        
    Returns
    -------
    report : str
        Formatted missing data report.
    """
    n_samples, n_features = X.shape
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Get summary statistics
    summary = missing_data_summary(X)
    
    # Calculate per-feature statistics
    missing_mask = np.isnan(X)
    missing_per_feature = np.sum(missing_mask, axis=0)
    
    # Start building report
    lines = []
    lines.append("MISSING DATA ANALYSIS REPORT")
    lines.append("=" * 50)
    
    # Overall statistics
    lines.append("\nOVERALL STATISTICS:")
    lines.append(f"  Dataset shape: {n_samples} samples × {n_features} features")
    lines.append(f"  Total missing values: {summary['total_missing']:,}")
    lines.append(f"  Missing rate: {summary['missing_rate']:.2%}")
    lines.append(f"  Samples with missing data: {summary['samples_with_missing']}/{n_samples} ({summary['samples_with_missing_rate']:.1%})")
    lines.append(f"  Features with missing data: {summary['features_with_missing']}/{n_features} ({summary['features_with_missing_rate']:.1%})")
    
    # Per-sample statistics
    lines.append("\nPER-SAMPLE STATISTICS:")
    lines.append(f"  Maximum missing per sample: {summary['max_missing_per_sample']}")
    lines.append(f"  Average missing per sample: {summary['mean_missing_per_sample']:.1f}")
    
    # Per-feature statistics
    lines.append("\nPER-FEATURE STATISTICS:")
    lines.append(f"{'Feature':<20} {'Missing Count':<15} {'Missing Rate':<15}")
    lines.append("-" * 50)
    
    for i, (name, count) in enumerate(zip(feature_names, missing_per_feature)):
        rate = count / n_samples
        lines.append(f"{name:<20} {count:<15} {rate:<15.2%}")
    
    # Pattern analysis
    lines.append("\nMISSING PATTERNS:")
    
    # Complete cases
    complete_cases = np.sum(~np.any(missing_mask, axis=1))
    lines.append(f"  Complete cases: {complete_cases}/{n_samples} ({complete_cases/n_samples:.1%})")
    
    # Cases with specific number of missing features
    missing_counts = np.sum(missing_mask, axis=1)
    unique_counts, count_frequencies = np.unique(missing_counts, return_counts=True)
    
    lines.append(f"  Cases by number of missing features:")
    for missing_count, frequency in zip(unique_counts, count_frequencies):
        lines.append(f"    {missing_count} missing: {frequency} cases ({frequency/n_samples:.1%})")
    
    return "\n".join(lines)


def compare_kernel_performance(results: List[Dict]) -> str:
    """
    Generate a performance comparison report for different kernels.
    
    Parameters
    ----------
    results : list of dict
        List of result dictionaries from kernel evaluations.
        
    Returns
    -------
    report : str
        Formatted performance comparison report.
    """
    if not results:
        return "No results to compare."
    
    lines = []
    lines.append("KERNEL PERFORMANCE COMPARISON")
    lines.append("=" * 60)
    
    # Group results by missing rate and dataset
    grouped_results = {}
    for result in results:
        key = (result['dataset'], result['missing_rate'], result['pattern'])
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Generate comparison for each group
    for (dataset, missing_rate, pattern), group_results in grouped_results.items():
        lines.append(f"\nDataset: {dataset}, Missing Rate: {missing_rate:.1%}, Pattern: {pattern}")
        lines.append("-" * 60)
        lines.append(f"{'Kernel':<25} {'Train Acc':<12} {'Test Acc':<12} {'Generalization':<15}")
        lines.append("-" * 60)
        
        for result in group_results:
            kernel = result['kernel']
            train_acc = result['train_accuracy']
            test_acc = result['test_accuracy']
            generalization = test_acc - train_acc
            
            lines.append(f"{kernel:<25} {train_acc:<12.3f} {test_acc:<12.3f} {generalization:<15.3f}")
    
    return "\n".join(lines)


def recommend_kernel(X: np.ndarray, task_type: str = 'classification') -> Dict[str, str]:
    """
    Recommend appropriate kernels based on data characteristics.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.
    task_type : str, default='classification'
        Type of machine learning task.
        
    Returns
    -------
    recommendations : dict
        Dictionary with kernel recommendations and justifications.
    """
    n_samples, n_features = X.shape
    summary = missing_data_summary(X)
    
    recommendations = {}
    
    # Analyze data characteristics
    missing_rate = summary['missing_rate']
    samples_with_missing_rate = summary['samples_with_missing_rate']
    features_with_missing_rate = summary['features_with_missing_rate']
    
    # Primary recommendation based on missing data characteristics
    if missing_rate < 0.1:
        primary = "ExpectedValueKernel"
        justification = "Low missing rate - simple mean imputation in kernel should work well"
    elif missing_rate < 0.3:
        if features_with_missing_rate > 0.5:
            primary = "CrossCorrelationKernel"
            justification = "Moderate missing rate with many affected features - probabilistic approach recommended"
        else:
            primary = "ExpectedValueKernel"
            justification = "Moderate missing rate - expected value approach should be sufficient"
    else:
        primary = "CrossCorrelationKernel"
        justification = "High missing rate - probabilistic kernel needed to handle uncertainty"
    
    recommendations['primary'] = primary
    recommendations['primary_justification'] = justification
    
    # Secondary recommendations
    secondary_options = []
    
    if n_features <= 10:
        secondary_options.append(("PolynomialMissingKernel", "Small feature space - polynomial interactions may be beneficial"))
    
    if n_samples >= 1000:
        secondary_options.append(("LinearMissingKernel", "Large dataset - linear kernel for computational efficiency"))
    
    if missing_rate > 0.2:
        if primary != "CrossCorrelationKernel":
            secondary_options.append(("CrossCorrelationKernel", "High missing rate alternative"))
    
    recommendations['secondary'] = secondary_options
    
    # Parameter recommendations
    param_recommendations = {}
    
    if n_features > 20:
        param_recommendations['gamma'] = "Try smaller values (0.01-0.1) for high-dimensional data"
    elif n_features < 5:
        param_recommendations['gamma'] = "Try larger values (1.0-10.0) for low-dimensional data"
    else:
        param_recommendations['gamma'] = "Start with default values (0.1-1.0)"
    
    param_recommendations['C'] = "Start with C=1.0, tune based on cross-validation"
    
    recommendations['parameters'] = param_recommendations
    
    return recommendations


def generate_kernel_recommendation_report(X: np.ndarray, feature_names: Optional[List[str]] = None) -> str:
    """
    Generate a comprehensive kernel recommendation report.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data matrix.
    feature_names : list of str, optional
        Names of features.
        
    Returns
    -------
    report : str
        Formatted recommendation report.
    """
    # Get data summary
    summary = missing_data_summary(X)
    recommendations = recommend_kernel(X)
    
    lines = []
    lines.append("KERNEL RECOMMENDATION REPORT")
    lines.append("=" * 50)
    
    # Data overview
    lines.append("\nDATA OVERVIEW:")
    n_samples, n_features = X.shape
    lines.append(f"  Samples: {n_samples:,}")
    lines.append(f"  Features: {n_features}")
    lines.append(f"  Missing rate: {summary['missing_rate']:.2%}")
    lines.append(f"  Features with missing data: {summary['features_with_missing']}/{n_features}")
    
    # Primary recommendation
    lines.append("\nPRIMARY RECOMMENDATION:")
    lines.append(f"  Kernel: {recommendations['primary']}")
    lines.append(f"  Reason: {recommendations['primary_justification']}")
    
    # Secondary recommendations
    if recommendations['secondary']:
        lines.append("\nSECONDARY OPTIONS:")
        for kernel, reason in recommendations['secondary']:
            lines.append(f"  • {kernel}: {reason}")
    
    # Parameter recommendations
    lines.append("\nPARAMETER TUNING SUGGESTIONS:")
    for param, suggestion in recommendations['parameters'].items():
        lines.append(f"  {param}: {suggestion}")
    
    # General advice
    lines.append("\nGENERAL ADVICE:")
    lines.append("  • Use cross-validation to compare kernel performance")
    lines.append("  • Consider computational cost vs. accuracy trade-offs")
    lines.append("  • Validate results on held-out test data")
    
    if summary['missing_rate'] > 0.3:
        lines.append("  • High missing rate detected - consider data collection improvements")
    
    return "\n".join(lines)
