# Contributing to scikit-missing

Thank you for your interest in contributing to scikit-missing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ajoshiusc/scikit-missing.git
   cd scikit-missing
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting: `black .`
- Run flake8 for linting: `flake8 .`
- Use type hints where appropriate

## Testing

- Write unit tests for new functionality
- Run tests with: `python -m pytest tests/`
- Ensure all tests pass before submitting

## Adding New Kernels

When adding a new kernel for handling missing data:

1. **Inherit from `BaseMissingKernel`:**
   ```python
   from scikit_missing.kernels import BaseMissingKernel
   
   class MyKernel(BaseMissingKernel):
       def compute_kernel(self, X, Y=None):
           # Implementation here
           pass
   ```

2. **Key requirements:**
   - Handle NaN values in input data
   - Implement proper kernel mathematics
   - Ensure symmetry and positive semi-definiteness when appropriate
   - Include proper parameter validation

3. **Add tests:**
   - Test kernel properties (symmetry, positive definiteness)
   - Test with various missing data patterns
   - Test edge cases (no missing data, all missing data)

4. **Update documentation:**
   - Add docstrings with mathematical description
   - Include usage examples
   - Update README if appropriate

## Submitting Changes

1. **Create a new branch:**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes and commit:**
   ```bash
   git add .
   git commit -m "Add new feature: description"
   ```

3. **Run tests and ensure they pass:**
   ```bash
   python -m pytest tests/
   ```

4. **Push to your fork and create a Pull Request**

## Code Review Process

- All submissions require review before merging
- Ensure code follows style guidelines
- Include appropriate tests and documentation
- Be responsive to feedback and suggestions

## Questions?

Feel free to open an issue for questions or discussions about contributing.
