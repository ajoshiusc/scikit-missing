# mSVM User Guide: Handling Missing Data in Classification

## What is mSVM?

mSVM (missing data Support Vector Machine) is a machine learning algorithm that can classify data even when some feature values are missing. Unlike traditional approaches that require you to "fill in" missing values first, mSVM works directly with incomplete data.

## Why Use mSVM?

### The Missing Data Problem

Imagine you have a dataset about patients, but some medical test results are missing:

```
Patient | Age | Blood Pressure | Cholesterol | Diagnosis
--------|-----|----------------|-------------|----------
   1    | 45  |      120       |    Missing  |  Healthy
   2    | 60  |    Missing     |     240     |    Sick
   3    | 35  |      110       |     180     |  Healthy
```

Traditional machine learning algorithms can't handle this directly. You'd typically:
1. **Delete rows** with missing data (loses information)
2. **Fill in averages** (introduces bias)
3. **Guess values** (adds uncertainty)

### The mSVM Solution

mSVM handles missing data intelligently by:
- **Keeping uncertainty**: Doesn't pretend to know what's missing
- **Using what's available**: Makes decisions based on observed features
- **Learning patterns**: Understands how missing data affects decisions

## How Does It Work?

### Core Concept

mSVM uses **kernels** - mathematical functions that measure similarity between data points. The innovation is creating kernels that can compare incomplete data points.

### Two Main Approaches

#### 1. Expected Value Kernel
- **Simple approach**: Replaces missing values with averages
- **Fast computation**: Good for most situations
- **Best for**: Low amounts of missing data (< 20%)

#### 2. Cross-Correlation Kernel  
- **Smart approach**: Treats missing values as uncertain
- **Better accuracy**: Handles uncertainty properly
- **Best for**: High amounts of missing data (> 20%)

### Intuitive Example

Think of it like comparing two people when you don't know everything about them:

```
Person A: Age=30, Height=?, Weight=70kg
Person B: Age=32, Height=180cm, Weight=?
```

**Expected Value approach**: "Let's assume missing height is average height"
**Cross-Correlation approach**: "Let's consider all possible heights and their probabilities"

## Getting Started

### Installation

```bash
pip install scikit-missing
```

### Basic Example

```python
import numpy as np
from scikit_missing import mSVM
from scikit_missing.kernels import ExpectedValueKernel

# Create sample data with missing values
X = np.array([
    [25, 120, np.nan],  # Age, BP, Cholesterol (missing)
    [45, np.nan, 200],  # Age, BP (missing), Cholesterol  
    [35, 110, 180],     # Complete data
    [50, 140, np.nan],  # Age, BP, Cholesterol (missing)
])
y = np.array([0, 1, 0, 1])  # 0=Healthy, 1=Sick

# Create and train the model
model = mSVM(kernel=ExpectedValueKernel())
model.fit(X, y)

# Make predictions on new data (also with missing values)
new_patient = np.array([[40, np.nan, 220]])  # Missing blood pressure
prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)

print(f"Prediction: {prediction[0]}")  # 0 or 1
print(f"Probability: {probability[0]}")  # [prob_healthy, prob_sick]
```

### Comparing Different Kernels

```python
from scikit_missing.kernels import CrossCorrelationKernel

# Test different approaches
kernels = {
    'Simple (Expected Value)': ExpectedValueKernel(),
    'Advanced (Cross Correlation)': CrossCorrelationKernel(),
}

results = {}
for name, kernel in kernels.items():
    model = mSVM(kernel=kernel)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    results[name] = accuracy
    print(f"{name}: {accuracy:.3f}")
```

## Choosing the Right Kernel

### Decision Guide

**Updated recommendations based on comprehensive benchmarking:**

| Your Situation | Recommended Kernel | Why |
|----------------|-------------------|-----|
| < 15% missing data | Expected Value | Simple, fast, and robust |
| 15-30% missing data | Expected Value | Best overall performance |
| > 30% missing data | Expected Value or Cross Correlation* | Try both, tune parameters carefully |
| Non-linear problems | Cross Correlation* | Better at handling complex patterns |
| Large datasets (>10k samples) | Expected Value | Computational efficiency critical |
| Real-time applications | Expected Value | Fast prediction times |
| Research/experimentation | Cross Correlation* | Worth the extra computation for accuracy |

*Cross Correlation requires hyperparameter tuning and is 5-15x slower than Expected Value

### Parameter Tuning

#### Gamma Parameter (for RBF kernels)
- **Low gamma (0.01-0.1)**: Smooth decision boundaries
- **High gamma (1.0-10.0)**: Complex decision boundaries
- **Start with**: 1.0, then experiment

#### C Parameter (regularization)
- **Low C (0.1-1.0)**: More regularization, simpler model
- **High C (10.0-100.0)**: Less regularization, complex model  
- **Start with**: 1.0, then tune with cross-validation

```python
# Parameter tuning example
model = mSVM(
    kernel=ExpectedValueKernel(gamma=0.5),  # Adjust gamma
    C=2.0,  # Adjust regularization
    max_iter=1000  # Maximum optimization iterations
)
```

## Real-World Examples

### Medical Diagnosis

```python
# Medical data with missing lab results
medical_features = [
    'age', 'blood_pressure', 'cholesterol', 
    'glucose', 'bmi', 'heart_rate'
]

# Some patients missing lab values
X_medical = load_medical_data()  # Contains NaN values
y_medical = load_diagnoses()     # 0=healthy, 1=disease

# Train model
medical_model = mSVM(kernel=CrossCorrelationKernel(gamma=0.1))
medical_model.fit(X_medical, y_medical)

# Predict for new patients (some missing tests)
new_patients = load_new_patients()  # May have missing values
predictions = medical_model.predict(new_patients)
```

### Financial Risk Assessment

```python
# Financial data with missing information
financial_features = [
    'income', 'credit_score', 'debt_ratio',
    'employment_years', 'loan_amount'
]

# Some applicants missing documentation
X_financial = load_loan_applications()  # Contains NaN
y_financial = load_default_labels()     # 0=no default, 1=default

# Train risk model
risk_model = mSVM(kernel=ExpectedValueKernel(gamma=1.0))
risk_model.fit(X_financial, y_financial)

# Assess risk for incomplete applications
risk_scores = risk_model.predict_proba(new_applications)
```

### Sensor Data Analysis

```python
# IoT sensor data with equipment failures
sensor_features = [
    'temperature', 'humidity', 'pressure',
    'vibration', 'power_consumption'
]

# Some sensors fail intermittently
X_sensors = load_sensor_readings()  # Contains NaN from failures
y_sensors = load_equipment_status() # 0=normal, 1=fault

# Train fault detection model
fault_model = mSVM(kernel=CrossCorrelationKernel(gamma=0.5))
fault_model.fit(X_sensors, y_sensors)

# Monitor equipment with partial sensor data
current_status = fault_model.predict(live_sensor_data)
```

## Performance Tips

### Data Preprocessing

```python
# Analyze your missing data first
from scikit_missing.utils import missing_data_summary

summary = missing_data_summary(X)
print(f"Missing rate: {summary['missing_rate']:.2%}")
print(f"Features with missing: {summary['features_with_missing']}")

# Get kernel recommendations
from scikit_missing.utils import recommend_kernel

recommendations = recommend_kernel(X)
print(f"Recommended kernel: {recommendations['primary']}")
```

### Model Evaluation

```python
from sklearn.model_selection import cross_val_score

# Use cross-validation for robust evaluation
scores = cross_val_score(
    mSVM(kernel=ExpectedValueKernel()),
    X, y, cv=5, scoring='accuracy'
)
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Handling Large Datasets

```python
# For large datasets, consider computational efficiency
if n_samples > 10000:
    # Use linear kernel for speed
    kernel = LinearMissingKernel()
else:
    # Use RBF kernel for accuracy
    kernel = CrossCorrelationKernel(gamma=1.0)

model = mSVM(kernel=kernel, C=1.0)
```

## Common Issues and Solutions

### Issue: Poor Performance
**Solutions:**
- Check missing data rate (>50% is challenging for any method)
- Try different kernels
- Tune parameters (gamma, C)
- Ensure enough training data

### Issue: Slow Training
**Solutions:**
- Use Expected Value kernel instead of Cross Correlation
- Reduce dataset size for initial experiments
- Consider Linear kernel for high-dimensional data

### Issue: Overfitting
**Solutions:**
- Increase regularization (lower C)
- Use cross-validation for parameter selection
- Collect more training data if possible

### Issue: Memory Errors
**Solutions:**
- Process data in smaller batches
- Use Linear kernel (doesn't store full kernel matrix)
- Consider feature selection to reduce dimensions

## Advanced Usage

### Custom Kernels

```python
from scikit_missing.kernels import BaseMissingKernel

class MyCustomKernel(BaseMissingKernel):
    def compute_kernel(self, X, Y=None):
        # Implement your custom missing data kernel
        pass

# Use custom kernel
model = mSVM(kernel=MyCustomKernel())
```

### Multi-class Classification

```python
# mSVM automatically handles multiple classes
y_multiclass = np.array([0, 1, 2, 1, 0, 2])  # 3 classes

model = mSVM(kernel=ExpectedValueKernel())
model.fit(X, y_multiclass)

# Get class probabilities
probabilities = model.predict_proba(X_test)  # Shape: (n_samples, n_classes)
```

### Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

# Improve probability estimates
calibrated_model = CalibratedClassifierCV(
    mSVM(kernel=CrossCorrelationKernel()),
    method='sigmoid',
    cv=3
)
calibrated_model.fit(X_train, y_train)

# More reliable probabilities
calibrated_probs = calibrated_model.predict_proba(X_test)
```

## When NOT to Use mSVM

### Better Alternatives Exist When:

1. **Very little missing data (< 5%)**: Standard SVM with simple imputation
2. **Missing data has clear patterns**: Domain-specific imputation methods
3. **Need interpretable models**: Linear regression, decision trees
4. **Very large datasets (> 100k samples)**: Consider approximate methods
5. **Real-time prediction needed**: Pre-computed imputation might be faster

## Summary

mSVM is a powerful tool for classification with missing data that:

✅ **Works directly with incomplete data**
✅ **Preserves uncertainty information**  
✅ **Often outperforms imputation-based approaches**
✅ **Provides flexible kernel options**
✅ **Maintains theoretical SVM guarantees**

**Best for**: Medium-sized datasets with significant missing data where accuracy is important and you want a principled approach to handling uncertainty.

**Start with**: Expected Value kernel, then experiment with Cross Correlation kernel if you need better performance with high missing rates.
