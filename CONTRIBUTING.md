# Contributing to NDSwin-JAX

Thank you for your interest in contributing to NDSwin-JAX! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and inclusive in all interactions. We welcome contributors from all backgrounds.

## Getting Started

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ndswin-jax.git
   cd ndswin-jax
   ```

2. **Create a virtual environment:**
   ```bash
   conda create -n ndswin-dev python=3.10
   conda activate ndswin-dev
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=ndswin --cov-report=html
```

### Code Style

We use `ruff` for linting and formatting:

```bash
# Check code
ruff check src/ndswin tests

# Format code
ruff format src/ndswin tests

# Fix issues
ruff check --fix src/ndswin tests
```

### Type Checking

```bash
mypy src/ndswin --ignore-missing-imports
```

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

Example:
```
feat(attention): add support for flash attention

Implemented flash attention for improved memory efficiency
on long sequences.

Closes #123
```

### Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** and commit

3. **Push to your fork:**
   ```bash
   git push origin feature/my-feature
   ```

4. **Open a Pull Request** with:
   - Clear description of changes
   - Link to related issues
   - Screenshots/examples if applicable

5. **Address review feedback**

6. **Merge** after approval

## Project Structure

```
ndswin-jax/
├── src/ndswin/          # Main package
│   ├── core/            # Core components (attention, blocks)
│   ├── models/          # Model implementations
│   ├── training/        # Training infrastructure
│   ├── inference/       # Inference utilities
│   └── utils/           # Helper utilities
├── tests/               # Test suite
├── docs/                # Documentation
└── train/               # Training scripts
```

## Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Test both success and failure cases
- Include edge cases

Example:
```python
def test_window_attention_output_shape():
    """Test that WindowAttention produces correct output shape."""
    attention = WindowAttention(dim=64, num_heads=4, window_size=(4, 4))
    x = jnp.ones((4, 16, 64))
    
    rng = jax.random.PRNGKey(0)
    variables = attention.init(rng, x, training=False)
    output = attention.apply(variables, x, training=False)
    
    assert output.shape == x.shape
```

### Test Coverage

Aim for high coverage:
- Core modules: >90%
- Models: >85%
- Training: >80%
- Utils: >75%

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def partition_windows(x: Array, window_size: Tuple[int, ...]) -> Array:
    """Partition input tensor into non-overlapping windows.
    
    Args:
        x: Input tensor of shape (B, *spatial, C).
        window_size: Size of each window dimension.
        
    Returns:
        Windows tensor of shape (num_windows * B, window_prod, C).
        
    Raises:
        ValueError: If spatial dimensions are not divisible by window size.
        
    Example:
        >>> x = jnp.ones((1, 8, 8, 64))
        >>> windows = partition_windows(x, (4, 4))
        >>> windows.shape
        (4, 16, 64)
    """
```

### Documentation Updates

When adding features:
1. Update docstrings
2. Update relevant docs in `docs/`
3. Add examples if appropriate
4. Update API reference

## Performance Considerations

### JAX Best Practices

- Use `jax.jit` for compilation
- Avoid Python loops in hot paths
- Use `jax.vmap` for batching
- Profile with `jax.profiler`

### Memory Efficiency

- Use gradient checkpointing for large models
- Consider mixed precision (bfloat16)
- Test on various batch sizes

## Questions?

- Open an issue for bugs or features
- Use discussions for questions
- Tag maintainers for urgent issues

Thank you for contributing!
