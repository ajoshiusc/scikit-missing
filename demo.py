"""
Demonstration script for mSVM with missing data.

This script demonstrates the basic functionality of the mSVM implementation
and can be run to verify that everything works correctly.
"""

import numpy as np
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scikit_missing.svm import mSVM
from scikit_missing.kernels import ExpectedValueKernel, CrossCorrelationKernel
from scikit_missing.utils import missing_data_summary, generate_kernel_recommendation_report


def create_synthetic_dataset():
    """Create a synthetic dataset with missing values for demonstration."""
    print("Creating synthetic dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate base data
    n_samples = 300
    n_features = 8
    
    # Create two classes with different characteristics
    # Class 0: centered around origin
    class0_size = n_samples // 2
    X_class0 = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features) * 0.5,
        size=class0_size
    )
    
    # Class 1: shifted and with different covariance
    class1_size = n_samples - class0_size
    mean_class1 = np.ones(n_features) * 1.5
    cov_class1 = np.eye(n_features) * 0.8
    # Add some correlation
    cov_class1[0, 1] = cov_class1[1, 0] = 0.3
    
    X_class1 = np.random.multivariate_normal(
        mean=mean_class1,
        cov=cov_class1,
        size=class1_size
    )
    
    # Combine classes
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(class0_size), np.ones(class1_size)])
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    return X, y


def introduce_missing_values(X, missing_rate=0.25):
    """Introduce missing values with realistic patterns."""
    print(f"\nIntroducing missing values (rate: {missing_rate:.1%})...")
    
    X_missing = X.copy()
    n_samples, n_features = X.shape
    
    # Pattern 1: Completely random missing (MCAR)
    random_missing_rate = missing_rate * 0.6
    random_mask = np.random.rand(n_samples, n_features) < random_missing_rate
    
    # Pattern 2: Feature-dependent missing (MAR)
    # Some features are more likely to be missing
    feature_missing_probs = np.random.beta(2, 8, n_features) * missing_rate * 0.8
    for j in range(n_features):
        feature_mask = np.random.rand(n_samples) < feature_missing_probs[j]
        random_mask[feature_mask, j] = True
    
    # Pattern 3: Value-dependent missing (MNAR)
    # Extreme values are more likely to be missing
    for j in range(min(3, n_features)):  # Only for first few features
        feature_values = X[:, j]
        # Standardize
        feature_std = (feature_values - np.mean(feature_values)) / np.std(feature_values)
        # Higher probability for extreme values
        extreme_probs = missing_rate * 0.2 * (1 + np.abs(feature_std) / 3)
        extreme_mask = np.random.rand(n_samples) < extreme_probs
        random_mask[extreme_mask, j] = True
    
    # Apply missing mask
    X_missing[random_mask] = np.nan
    
    # Print missing data summary
    summary = missing_data_summary(X_missing)
    print(f"Total missing values: {summary['total_missing']}")
    print(f"Actual missing rate: {summary['missing_rate']:.2%}")
    print(f"Samples with missing data: {summary['samples_with_missing']}/{n_samples}")
    
    return X_missing


def demonstrate_kernels(X_train, X_test, y_train, y_test):
    """Demonstrate different kernels on the dataset."""
    print("\n" + "="*60)
    print("KERNEL COMPARISON")
    print("="*60)
    
    # Define kernels to test
    kernels = {
        'Expected Value (γ=0.1)': ExpectedValueKernel(gamma=0.1),
        'Expected Value (γ=1.0)': ExpectedValueKernel(gamma=1.0),
        'Cross Correlation (γ=0.1)': CrossCorrelationKernel(gamma=0.1),
        'Cross Correlation (γ=1.0)': CrossCorrelationKernel(gamma=1.0),
    }
    
    results = {}
    
    for kernel_name, kernel in kernels.items():
        print(f"\nTesting {kernel_name}...")
        
        try:
            # Create and train classifier
            clf = mSVM(kernel=kernel, C=1.0, max_iter=500)
            clf.fit(X_train, y_train)
            
            # Evaluate performance
            train_accuracy = clf.score(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)
            
            # Get additional info
            n_support = len(clf.support_)
            
            results[kernel_name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'n_support_vectors': n_support,
                'generalization_gap': train_accuracy - test_accuracy
            }
            
            print(f"  Train accuracy: {train_accuracy:.3f}")
            print(f"  Test accuracy:  {test_accuracy:.3f}")
            print(f"  Support vectors: {n_support}")
            print(f"  Generalization gap: {train_accuracy - test_accuracy:+.3f}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results[kernel_name] = None
    
    return results


def analyze_results(results):
    """Analyze and summarize the results."""
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if v is not None}
    
    if not successful_results:
        print("No successful results to analyze.")
        return
    
    # Find best performers
    best_test = max(successful_results.items(), key=lambda x: x[1]['test_accuracy'])
    best_generalization = min(successful_results.items(), 
                            key=lambda x: abs(x[1]['generalization_gap']))
    
    print(f"\nBest test accuracy: {best_test[0]}")
    print(f"  Test accuracy: {best_test[1]['test_accuracy']:.3f}")
    
    print(f"\nBest generalization: {best_generalization[0]}")
    print(f"  Generalization gap: {best_generalization[1]['generalization_gap']:+.3f}")
    
    # Summary table
    print(f"\nSUMMARY TABLE:")
    print(f"{'Kernel':<25} {'Train':<8} {'Test':<8} {'Gap':<8} {'#SV':<6}")
    print("-" * 55)
    
    for kernel_name, result in successful_results.items():
        if result:
            print(f"{kernel_name:<25} "
                  f"{result['train_accuracy']:<8.3f} "
                  f"{result['test_accuracy']:<8.3f} "
                  f"{result['generalization_gap']:+<8.3f} "
                  f"{result['n_support_vectors']:<6d}")


def demonstrate_probability_predictions(X_train, X_test, y_train, y_test):
    """Demonstrate probability predictions and decision functions."""
    print("\n" + "="*60)
    print("PROBABILITY PREDICTIONS DEMO")
    print("="*60)
    
    # Use best performing kernel for demonstration
    kernel = ExpectedValueKernel(gamma=1.0)
    clf = mSVM(kernel=kernel, C=1.0)
    clf.fit(X_train, y_train)
    
    # Get predictions for a few test samples
    n_demo = min(10, len(X_test))
    X_demo = X_test[:n_demo]
    y_demo = y_test[:n_demo]
    
    # Get different types of predictions
    y_pred = clf.predict(X_demo)
    y_proba = clf.predict_proba(X_demo)
    decision_values = clf.decision_function(X_demo)
    
    print(f"\nPredictions for {n_demo} test samples:")
    print(f"{'Sample':<8} {'True':<6} {'Pred':<6} {'P(Class 0)':<12} {'P(Class 1)':<12} {'Decision':<10}")
    print("-" * 60)
    
    for i in range(n_demo):
        print(f"{i:<8} "
              f"{int(y_demo[i]):<6} "
              f"{int(y_pred[i]):<6} "
              f"{y_proba[i, 0]:<12.3f} "
              f"{y_proba[i, 1]:<12.3f} "
              f"{decision_values[i]:<10.3f}")


def main():
    """Run the complete demonstration."""
    print("mSVM Demonstration Script")
    print("=" * 50)
    
    try:
        # Step 1: Create dataset
        X, y = create_synthetic_dataset()
        
        # Step 2: Introduce missing values
        X_missing = introduce_missing_values(X, missing_rate=0.2)
        
        # Step 3: Generate recommendation report
        print("\n" + "="*60)
        print("KERNEL RECOMMENDATION")
        print("="*60)
        report = generate_kernel_recommendation_report(X_missing)
        print(report)
        
        # Step 4: Split data
        print(f"\nSplitting data...")
        split_idx = int(0.7 * len(X))
        X_train, X_test = X_missing[:split_idx], X_missing[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Step 5: Demonstrate kernels
        results = demonstrate_kernels(X_train, X_test, y_train, y_test)
        
        # Step 6: Analyze results
        analyze_results(results)
        
        # Step 7: Demonstrate probability predictions
        demonstrate_probability_predictions(X_train, X_test, y_train, y_test)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nKey takeaways:")
        print("• mSVM can handle missing data directly without imputation")
        print("• Different kernels have different strengths")
        print("• Cross-correlation kernel handles uncertainty better with more missing data")
        print("• Expected value kernel is simpler and often performs well")
        print("• Parameter tuning (especially gamma) is important for performance")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
