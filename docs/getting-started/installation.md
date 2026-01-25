# Installation

Diffid is available as a Python package with pre-built wheels for most platforms.

## Installation Methods

=== "uv"

    [uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.

    ```bash
    uv pip install diffid
    ```

=== "pip"

    ```bash
    pip install diffid
    ```

### Optional Dependencies

Diffid has optional plotting support via matplotlib:

=== "uv"

    ```bash
    uv pip install "diffid[plotting]"
    ```

=== "pip"

    ```bash
    pip install "diffid[plotting]"
    ```

## Verifying Installation

After installation, verify that Diffid is working correctly:

```python
import diffid
import numpy as np

# Simple test
def test_func(x):
    return np.asarray([(x[0] - 1.0) ** 2])

builder = diffid.ScalarBuilder().with_objective(test_func).with_parameter("x", 0.0)
problem = builder.build()
result = problem.optimise()

print(f"Success: {result.success}")
print(f"Optimal x: {result.x[0]:.3f}")
```

Expected output:
```
Success: True
Optimal x: 1.000
```

## Platform-Specific Notes

### Linux

Pre-built wheels are available for x86_64 and aarch64 architectures. No additional setup required.

### macOS

Pre-built wheels are available for both Intel (x86_64) and Apple Silicon (arm64) Macs.

If you're using Apple Silicon and encounter issues, ensure you're using a native arm64 Python installation rather than running under Rosetta.

### Windows

Windows builds are marked experimental. Pre-built wheels are available but don't currently support diffsol gradients due to LLVM integration issues.

If you encounter issues consider using [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install)

## Building from Source

If you need to build from source (for development or if pre-built wheels aren't available):

### Prerequisites

- [Rust](https://rustup.rs/) >= 1.70
- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended)

### Build Steps

```bash
# Clone the repository
git clone https://github.com/bradyplanden/diffid.git
cd diffid

# Create Python environment
uv sync

# Build Rust extension with Python bindings
uv run maturin develop

# Run tests
uv run pytest -v
```

For additional troubleshooting, see the [Troubleshooting Guide](../guides/troubleshooting.md).

## Next Steps

Now that Diffid is installed, proceed to the [5-Minute Quickstart](quickstart.md) to run your first optimisation.
