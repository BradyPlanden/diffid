# Architecture

!!! info "Coming Soon"
    Detailed architecture documentation is being written.

## High-Level Overview

```
┌─────────────────────────────────────┐
│         Python API Layer            │
│   (PyO3 bindings, type stubs)       │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│          Rust Core                  │
│  (Optimisers, Samplers, Builders)   │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      External Libraries             │
│  (Diffsol, ndarray, rayon)          │
└─────────────────────────────────────┘
```

## Key Design Patterns

- **Builder Pattern**: Fluent API for problem construction
- **Zero-Copy**: Efficient data handling between Python and Rust
- **Trait-Based**: Extensible algorithm interfaces

## See Also

- [Building from Source](building.md)
- [Contributing](contributing.md)
- [Rust API](../api-reference/rust/index.md)
