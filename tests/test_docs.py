"""Documentation quality tests.

Validates that:
1. Documentation builds without errors
2. Notebooks are valid and well-structured
"""

import json
import pathlib
import subprocess

import pytest

# Project root
ROOT = pathlib.Path(__file__).parent.parent
DOCS_DIR = ROOT / "docs"
MKDOCS_YML = ROOT / "mkdocs.yml"


class TestDocumentationBuild:
    """Test that documentation builds successfully."""

    def test_mkdocs_build_succeeds(self):
        """Test that mkdocs builds without errors.

        mkdocs strict mode validates:
        - All internal links resolve
        - No broken references
        - Proper navigation structure
        """
        result = subprocess.run(
            ["mkdocs", "build", "--strict"],
            check=False,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"mkdocs build failed:\n{result.stdout}\n{result.stderr}"
        )


class TestNotebookStructure:
    """Test that notebooks are valid and well-structured."""

    def test_all_notebooks_are_valid_json(self):
        """Verify all notebooks are valid JSON with proper structure."""
        notebooks = list((DOCS_DIR / "tutorials" / "notebooks").glob("*.ipynb"))

        assert len(notebooks) > 0, "No notebooks found"

        for notebook_path in notebooks:
            with notebook_path.open(encoding="utf-8") as f:
                nb_data = json.load(f)

            # Validate basic notebook structure
            assert "cells" in nb_data, f"{notebook_path.name} missing cells"
            assert "metadata" in nb_data, f"{notebook_path.name} missing metadata"
            assert len(nb_data["cells"]) > 0, f"{notebook_path.name} has no cells"


def test_readme_not_in_docs():
    """Ensure README doesn't conflict with index.md."""
    readme = DOCS_DIR / "README.md"
    assert not readme.exists(), "docs/README.md conflicts with index.md"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
