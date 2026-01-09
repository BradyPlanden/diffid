# Building from Source

!!! info "Coming Soon"
    Detailed build instructions are being written.

## Quick Build

```bash
# Clone repository
git clone https://github.com/bradyplanden/chronopt.git
cd chronopt

# Set up environment
uv sync

# Build
uv run maturin develop

# Test
uv run pytest -v
cargo test
```

## Prerequisites

- Rust >= 1.70
- Python >= 3.11
- uv (recommended)

## Platform-Specific Notes

### Linux

No special requirements.

### macOS

Works on both Intel and Apple Silicon.

### Windows

Experimental support. May require Visual C++ Build Tools.

## See Also

- [Contributing](contributing.md)
- [Architecture](architecture.md)
- [Installation](../getting-started/installation.md)
