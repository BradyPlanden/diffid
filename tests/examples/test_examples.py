import os
import pathlib
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def is_ci() -> bool:
    """Detect CI environment."""
    ci_env_vars = ("CI", "GITHUB_ACTIONS")
    return any(os.environ.get(var) for var in ci_env_vars)


@dataclass(frozen=True)
class ExampleConfig:
    """Configuration for a specific example pattern."""

    pattern: str
    skip_condition: Callable[[], bool] | None = None
    skip_reason: str = ""
    timeout: float = 120.0
    required_import: str | None = None


# Define special handling for specific example patterns
EXAMPLE_CONFIGS: list[ExampleConfig] = [
    ExampleConfig(
        pattern="predator_prey",
        timeout=240.0,
        required_import="diffrax",
    ),
    ExampleConfig(
        pattern="diffeqpy",
        skip_condition=is_ci,
        skip_reason="Skipping Julia/diffeqpy example in CI (Julia installation too heavy)",
    ),
    ExampleConfig(
        pattern="bouncy_ball",
        skip_condition=lambda: sys.platform.startswith("win"),
        skip_reason="Skipping sensitivity-based example on Windows (missing support)",
    ),
]


def _get_config_for_path(relative_path: pathlib.Path) -> ExampleConfig | None:
    """Find matching config for an example path."""
    path_str = str(relative_path)
    for config in EXAMPLE_CONFIGS:
        if config.pattern in path_str:
            return config
    return None


def _run_example(script: pathlib.Path, timeout: float = 120.0) -> None:
    """Run an example script and fail the test if it returns non-zero."""
    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=script.parent,  # Run from script's directory for relative imports
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.returncode != 0:
        # Truncate very long output
        max_output = 2000
        stdout = result.stdout[:max_output] + (
            "..." if len(result.stdout) > max_output else ""
        )
        stderr = result.stderr[:max_output] + (
            "..." if len(result.stderr) > max_output else ""
        )
        pytest.fail(
            f"Example {script.name} failed with code {result.returncode}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )


def _discover_example_scripts() -> list[pathlib.Path]:
    """Discover all example scripts, excluding private modules."""
    if not EXAMPLES.exists():
        return []

    scripts = [
        path.relative_to(EXAMPLES)
        for path in EXAMPLES.rglob("*.py")
        if not path.name.startswith("_") and "__pycache__" not in path.parts
    ]
    return sorted(scripts)


EXAMPLE_SCRIPTS = _discover_example_scripts()


@pytest.mark.parametrize(
    "relative_path",
    EXAMPLE_SCRIPTS,
    ids=lambda p: str(p).replace("/", "::").replace("\\", "::"),
)
def test_example_script(relative_path: pathlib.Path) -> None:
    """Run each discovered example script and verify it exits successfully."""
    config = _get_config_for_path(relative_path)

    if config:
        # Check for required imports
        if config.required_import:
            pytest.importorskip(
                config.required_import,
                reason=f"{config.required_import} not installed for {relative_path}",
            )

        # Check skip conditions
        if config.skip_condition and config.skip_condition():
            pytest.skip(config.skip_reason)

        timeout = config.timeout
    else:
        timeout = 120.0

    _run_example(EXAMPLES / relative_path, timeout=timeout)
