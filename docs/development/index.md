# Development

Resources for contributors and developers working with Chronopt.

## Getting Started with Development

<div class="grid cards" markdown>

-   :material-source-pull:{ .lg .middle } __Contributing__

    ---

    Guidelines for contributing code, documentation, and bug reports.

    [:octicons-arrow-right-24: Contributing Guide](contributing.md)

-   :material-file-tree:{ .lg .middle } __Architecture__

    ---

    Understanding Chronopt's Rust core and PyO3 bindings design.

    [:octicons-arrow-right-24: Architecture Overview](architecture.md)

</div>

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/bradyplanden/chronopt.git
cd chronopt

# Create Python environment
uv sync

# Build Rust extension with Python bindings
uv run maturin develop

# Run tests
uv run pytest -v      # Python tests
cargo test            # Rust tests
```

## Development Workflow

### 1. Make Changes

Edit Rust source in `rust/src/` or Python bindings in `python/`.

### 2. Rebuild

```bash
uv run maturin develop  # For Python binding changes
```

### 3. Test

```bash
# Python tests
uv run pytest -v

# Rust tests
cargo test

# Both
cargo test && uv run pytest -v
```

### 4. Update Stubs

If you modified Python bindings:

```bash
uv run cargo run -p chronopt-py --no-default-features --features stubgen --bin generate_stubs
```

### 5. Format and Lint

```bash
# Rust
cargo fmt
cargo clippy

# Python
uv run ruff check .
uv run ruff format .
```

## Project Structure

```
chronopt/
├── rust/                   # Rust core implementation
│   ├── src/
│   │   ├── builders/       # Problem builders
│   │   ├── optimisers/     # Optimisation algorithms
│   │   ├── sampler/        # MCMC and nested sampling
│   │   ├── cost/           # Cost metrics
│   │   └── problem/        # Problem types
│   ├── Cargo.toml
│   └── tests/              # Rust tests
├── python/                 # Python bindings
│   ├── src/chronopt/
│   │   ├── __init__.py
│   │   └── _chronopt.pyi   # Generated type stubs
│   └── chronopt/           # PyO3 bindings source
├── examples/               # Example scripts
├── tests/                  # Python tests
├── docs/                   # Documentation (this site)
├── pyproject.toml          # Python package config
└── README.md
```

## Key Technologies

- **Rust**: Core algorithms, high performance
- **PyO3**: Python bindings with minimal overhead
- **Maturin**: Build system for Rust Python extensions
- **uv**: Fast Python package management
- **MkDocs Material**: Documentation site

## Testing Strategy

### Unit Tests

Rust unit tests alongside code:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // ...
    }
}
```

### Integration Tests

Python integration tests in `tests/`:

```python
def test_optimisation():
    builder = chron.ScalarBuilder().with_objective(func)
    # ...
    assert result.success
```

### Continuous Integration

GitHub Actions runs:
- Rust tests and clippy
- Python tests on multiple versions
- Type checking with mypy
- Linting with ruff
- Documentation builds

## Documentation

### Rust Docs

```bash
cargo doc --open --no-deps
```

### Python Docs

Edit markdown files in `docs/`:

```bash
mkdocs serve  # Live preview
mkdocs build  # Build static site
```

### Docstrings

Use NumPy-style docstrings:

```python
def example(x, y):
    """
    Brief description.

    Parameters
    ----------
    x : float
        Description of x.
    y : float
        Description of y.

    Returns
    -------
    float
        Description of return value.
    """
```

## Code Style

### Rust

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` (automatic formatting)
- Pass `cargo clippy` (linting)

### Python

- Follow PEP 8
- Use type hints
- Format with `ruff format`

## Performance Profiling

### Rust

```bash
cargo install flamegraph
cargo flamegraph --example your_example
```

### Python

```bash
uv pip install py-spy
py-spy record --native -- python examples/your_example.py
```

## See Also

- [Architecture](architecture.md) - System design
- [Contributing](contributing.md) - Contribution guidelines
- [API Reference](../api-reference/index.md) - API documentation
