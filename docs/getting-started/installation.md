# Installation

Chronopt is available as a Python package with pre-built wheels for most platforms.

## Requirements

- **Python**: >= 3.11
- **Operating Systems**:
    - Linux (x86_64, aarch64)
    - macOS (x86_64, Apple Silicon)
    - Windows (experimental)

## Installation Methods

=== "pip"

    ```bash
    pip install chronopt
    ```

=== "uv (recommended)"

    [uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.

    ```bash
    uv pip install chronopt
    ```

### Optional Dependencies

Chronopt has optional plotting support via matplotlib:

=== "pip"

    ```bash
    pip install "chronopt[plotting]"
    ```

=== "uv"

    ```bash
    uv pip install "chronopt[plotting]"
    ```

## Verifying Installation

After installation, verify that Chronopt is working correctly:

```python
import chronopt as chron
import numpy as np

# Simple test
def test_func(x):
    return np.asarray([(x[0] - 1.0) ** 2])

builder = chron.ScalarBuilder().with_callable(test_func).with_parameter("x", 0.0)
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

### Windows (Experimental)

Windows builds are currently experimental. Pre-built wheels are available but may have limitations.

If you encounter issues:

1. Ensure you have the latest [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) installed
2. Consider using [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install)

## Building from Source

If you need to build from source (for development or if pre-built wheels aren't available):

### Prerequisites

- [Rust](https://rustup.rs/) >= 1.70
- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended)

### Build Steps

```bash
# Clone the repository
git clone https://github.com/bradyplanden/chronopt.git
cd chronopt

# Create Python environment
uv sync

# Build Rust extension with Python bindings
uv run maturin develop

# Run tests
uv run pytest -v
```

For more details, see [Building from Source](../development/building.md).

## Troubleshooting

### Import Error: No module named 'chronopt'

**Cause**: Chronopt is not installed in your current Python environment.

**Solution**:
1. Verify your Python environment is active
2. Re-run the installation command
3. Check that `pip list` shows chronopt

### ImportError: DLL load failed (Windows)

**Cause**: Missing Visual C++ runtime libraries.

**Solution**: Install [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

### Wheel Not Available for Your Platform

**Cause**: Pre-built wheels might not be available for your specific Python version or platform.

**Solution**: [Build from source](#building-from-source) or open an issue on [GitHub](https://github.com/bradyplanden/chronopt/issues)

### Installation Succeeds but Import Fails

**Cause**: Binary incompatibility or corrupted installation.

**Solution**:
```bash
pip uninstall chronopt
pip cache purge  # Clear pip cache
pip install chronopt
```

For additional troubleshooting, see the [Troubleshooting Guide](../guides/troubleshooting.md).

## Next Steps

Now that Chronopt is installed, proceed to the [5-Minute Quickstart](quickstart.md) to run your first optimisation.
