// python/src/bin/generate_stubs.rs
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use chronopt::{stub_info, stub_info_from};
use clap::Parser;
use pyo3_stub_gen::Result;

/// Generators for Python type stubs produced from PyO3 bindings
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Explicit path to the pyproject.toml describing the Python package
    #[arg(long, value_name = "FILE")]
    pyproject: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    ensure_python_paths();
    let args = Args::parse();

    // Allow overriding pyproject.toml location if the build tool provides one.
    if let Some(pyproject) = args.pyproject {
        stub_info_from(pyproject)?.generate()
    } else {
        stub_info()?.generate()
    }
}

fn ensure_python_paths() {
    if env::var_os("PYTHONHOME").is_some() {
        return;
    }

    let venv_root = env::var_os("VIRTUAL_ENV")
        .or_else(|| env::var_os("UV_ACTIVE_VENV"))
        .or_else(|| env::var_os("UV_PROJECT_ENVIRONMENT"))
        .map(PathBuf::from);

    let Some(root) = venv_root else {
        return;
    };

    if !root.join("pyvenv.cfg").exists() {
        return;
    }

    let mut site_packages = collect_site_packages(&root);

    let mut stdlib_paths = Vec::new();
    if let Some(base_prefix) = resolve_base_prefix(&root) {
        env::set_var("PYTHONHOME", &base_prefix);
        env::remove_var("VIRTUAL_ENV");
        env::remove_var("UV_ACTIVE_VENV");
        env::remove_var("UV_PROJECT_ENVIRONMENT");

        stdlib_paths.extend(collect_stdlib_paths(&base_prefix));
        site_packages.extend(collect_site_packages(&base_prefix));
    }

    if stdlib_paths.is_empty() && site_packages.is_empty() {
        return;
    }

    let mut paths: Vec<PathBuf> = env::var_os("PYTHONPATH")
        .map(|os| env::split_paths(&os).collect())
        .unwrap_or_default();

    for site in site_packages {
        prepend_unique(&mut paths, site);
    }

    for stdlib in stdlib_paths {
        prepend_unique(&mut paths, stdlib);
    }

    if let Ok(joined) = env::join_paths(paths) {
        env::set_var("PYTHONPATH", joined);
    }
}

fn prepend_unique(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.iter().any(|existing| existing == &candidate) {
        paths.insert(0, candidate);
    }
}

fn resolve_base_prefix(root: &Path) -> Option<PathBuf> {
    let cfg = fs::read_to_string(root.join("pyvenv.cfg")).ok()?;
    for line in cfg.lines() {
        let mut parts = line.splitn(2, '=');
        let key = parts.next()?.trim();
        let value = parts.next()?.trim();
        if key == "home" {
            let base_bin = PathBuf::from(value);
            return base_bin.parent().map(Path::to_path_buf);
        }
    }
    None
}

fn collect_site_packages(root: &Path) -> Vec<PathBuf> {
    let mut site_paths = Vec::new();
    let lib_root = root.join("lib");
    if let Ok(entries) = fs::read_dir(&lib_root) {
        for entry in entries.flatten() {
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if !ft.is_dir() {
                continue;
            }
            let name = entry.file_name();
            if !name.to_string_lossy().starts_with("python") {
                continue;
            }
            let site = entry.path().join("site-packages");
            if site.exists() {
                site_paths.push(site);
            }
        }
    }
    site_paths
}

fn collect_stdlib_paths(prefix: &Path) -> Vec<PathBuf> {
    let mut libraries = Vec::new();
    let lib_root = prefix.join("lib");
    if let Ok(entries) = fs::read_dir(&lib_root) {
        for entry in entries.flatten() {
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            let path = entry.path();

            if ft.is_dir() && name_str.starts_with("python") {
                libraries.push(path.clone());

                let mut zip_name: String = name_str.chars().filter(|c| *c != '.').collect();
                zip_name.push_str(".zip");
                let zip_path = lib_root.join(&zip_name);
                if zip_path.exists() {
                    libraries.push(zip_path);
                }
            } else if ft.is_file()
                && name_str.starts_with("python")
                && path.extension().and_then(|ext| ext.to_str()) == Some("zip")
            {
                libraries.push(path);
            }
        }
    }
    libraries
}
