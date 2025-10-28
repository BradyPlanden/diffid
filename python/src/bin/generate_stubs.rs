// python/src/bin/generate_stubs.rs
use chronopt::stub_info;
use pyo3_stub_gen::Result;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    // Change to workspace root so pyo3_stub_gen reads the correct pyproject.toml
    let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = binding;

    env::set_current_dir(workspace_root)
        .expect("Failed to change to workspace root");

    let stub = stub_info()?;
    stub.generate()?;
    Ok(())
}