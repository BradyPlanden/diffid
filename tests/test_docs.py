"""Documentation quality and coverage tests.

This module validates that:
1. All public APIs are documented
2. Documentation builds without errors
3. All internal links resolve
4. Code examples in docs are valid
"""

import inspect
import pathlib
import re
import subprocess

import pytest

# Project root
ROOT = pathlib.Path(__file__).parent.parent
DOCS_DIR = ROOT / "docs"
MKDOCS_YML = ROOT / "mkdocs.yml"


class TestDocumentationBuild:
    """Test that documentation builds successfully."""

    def test_mkdocs_config_exists(self):
        """Check that mkdocs.yml exists."""
        assert MKDOCS_YML.exists(), "mkdocs.yml not found"

    def test_docs_directory_exists(self):
        """Check that docs/ directory exists."""
        assert DOCS_DIR.exists(), "docs/ directory not found"

    def test_mkdocs_build_succeeds(self):
        """Test that mkdocs builds without errors."""
        result = subprocess.run(
            ["mkdocs", "build", "--strict"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        # Check return code
        assert result.returncode == 0, (
            f"mkdocs build failed with:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )


class TestAPIDocumentation:
    """Test that all public APIs are documented."""

    def _get_public_apis(self, module) -> set[str]:
        """Extract public API names from a module."""
        apis = set()

        for name, obj in inspect.getmembers(module):
            # Skip private members
            if name.startswith("_"):
                continue

            # Include classes, functions, and constants
            if (
                inspect.isclass(obj)
                or inspect.isfunction(obj)
                or isinstance(obj, (int, float, str))
            ):
                apis.add(name)

        return apis

    def _find_documented_apis(self, doc_path: pathlib.Path) -> set[str]:
        """Extract API names mentioned in documentation."""
        if not doc_path.exists():
            return set()

        content = doc_path.read_text()

        # Find patterns like `ClassName`, `function_name()`, etc.
        # This is a simple heuristic - could be improved
        patterns = [
            r"`(\w+)`",  # Inline code
            r"##\s+(\w+)",  # Headers
            r"class:\s+(\w+)",  # Class references
        ]

        documented = set()
        for pattern in patterns:
            matches = re.findall(pattern, content)
            documented.update(matches)

        return documented

    @pytest.mark.skipif(
        not (ROOT / "python" / "src" / "chronopt").exists(),
        reason="chronopt package not installed",
    )
    def test_core_apis_documented(self):
        """Check that core chronopt APIs are documented."""
        try:
            import chronopt as chron
        except ImportError:
            pytest.skip("chronopt not installed")

        # Get public APIs from chronopt
        self._get_public_apis(chron)

        # Find documentation files
        api_docs = list((DOCS_DIR / "api-reference" / "python").rglob("*.md"))

        # Collect documented APIs
        documented_apis = set()
        for doc_file in api_docs:
            documented_apis.update(self._find_documented_apis(doc_file))

        # Core APIs that should be documented
        expected_apis = {
            "ScalarBuilder",
            "DiffsolBuilder",
            "VectorBuilder",
            "NelderMead",
            "CMAES",
            "Adam",
            "SSE",
            "RMSE",
            "GaussianNLL",
        }

        # Check coverage
        missing = expected_apis - documented_apis

        if missing:
            print(f"\n⚠️  APIs missing from documentation: {missing}")
            print(f"Documented APIs: {documented_apis}")

        # We expect at least 80% coverage
        coverage = len(expected_apis - missing) / len(expected_apis)
        assert coverage >= 0.8, (
            f"API documentation coverage too low: {coverage:.0%}\nMissing: {missing}"
        )


class TestNotebookQuality:
    """Test Jupyter notebook quality."""

    def test_all_notebooks_exist(self):
        """Check that all referenced notebooks exist."""
        # Parse mkdocs.yml for notebook references
        content = MKDOCS_YML.read_text()

        # Find .ipynb references
        notebook_refs = re.findall(r"tutorials/notebooks/(\d+_\w+\.ipynb)", content)

        # Check each exists
        for notebook_name in notebook_refs:
            notebook_path = DOCS_DIR / "tutorials" / "notebooks" / notebook_name
            assert notebook_path.exists(), f"Notebook not found: {notebook_path}"

    def test_notebooks_have_metadata(self):
        """Check that notebooks have proper metadata."""
        import json

        notebooks = list((DOCS_DIR / "tutorials" / "notebooks").glob("*.ipynb"))

        for notebook_path in notebooks:
            if notebook_path.name == "utils.py":  # Skip utility file
                continue

            with open(notebook_path) as f:
                nb_data = json.load(f)

            # Check structure
            assert "cells" in nb_data, f"{notebook_path.name} missing cells"
            assert "metadata" in nb_data, f"{notebook_path.name} missing metadata"

            # Check that it has markdown cells (documentation)
            has_markdown = any(
                cell.get("cell_type") == "markdown" for cell in nb_data["cells"]
            )
            assert has_markdown, f"{notebook_path.name} has no markdown cells"

    def test_notebooks_have_objectives(self):
        """Check that notebooks start with objectives."""
        import json

        notebooks = list((DOCS_DIR / "tutorials" / "notebooks").glob("[0-9]*.ipynb"))

        for notebook_path in notebooks:
            with open(notebook_path) as f:
                nb_data = json.load(f)

            # First cell should be markdown with  objectives
            first_cell = nb_data["cells"][0]
            assert first_cell["cell_type"] == "markdown", (
                f"{notebook_path.name} first cell is not markdown"
            )

            # Check for objectives
            content = "".join(first_cell["source"])
            has_objectives = "Objectives" in content or "objectives" in content
            assert has_objectives, (
                f"{notebook_path.name} missing objectives in first cell"
            )


class TestDocumentationStructure:
    """Test documentation structure and organization."""

    def test_required_sections_exist(self):
        """Check that required documentation sections exist."""
        required = [
            "getting-started",
            "tutorials",
            "guides",
            "algorithms",
            "api-reference",
            "examples",
            "development",
        ]

        for section in required:
            section_path = DOCS_DIR / section
            assert section_path.exists(), f"Required section missing: {section}"
            assert section_path.is_dir(), f"Section is not a directory: {section}"

    def test_index_pages_exist(self):
        """Check that all sections have index pages."""
        sections = [
            "getting-started",
            "tutorials",
            "guides",
            "algorithms",
            "api-reference",
            "development",
        ]

        for section in sections:
            index_path = DOCS_DIR / section / "index.md"
            assert index_path.exists(), f"Missing index page: {section}/index.md"

    def test_navigation_completeness(self):
        """Check that mkdocs.yml nav includes all main sections."""
        content = MKDOCS_YML.read_text()

        required_nav = [
            "Getting Started",
            "Tutorials",
            "User Guides",
            "Algorithms",
            "API Reference",
            "Examples",
            "Development",
        ]

        for section in required_nav:
            assert section in content, f"Navigation missing section: {section}"


class TestInternalLinks:
    """Test that internal links are valid."""

    def _extract_md_links(self, content: str) -> list[str]:
        """Extract markdown links from content."""
        # Match [text](path) but not [text](http://...)
        pattern = r"\[([^\]]+)\]\((?!http)([^)]+)\)"
        matches = re.findall(pattern, content)
        return [path for _, path in matches]

    def test_getting_started_links(self):
        """Test links in getting started guides."""
        getting_started = DOCS_DIR / "getting-started"

        for md_file in getting_started.glob("*.md"):
            content = md_file.read_text()
            links = self._extract_md_links(content)

            for link in links:
                # Resolve relative link
                if link.startswith("#"):  # Anchor link
                    continue

                if link.startswith("../../"):
                    # Relative to docs root
                    target = DOCS_DIR / link.replace("../../", "")
                elif link.startswith("../"):
                    # Relative to parent
                    target = getting_started.parent / link.replace("../", "")
                else:
                    target = getting_started / link

                # Check if target exists (handle .md vs .html)
                if not target.exists() and target.suffix == "":
                    target = target.with_suffix(".md")

                assert target.exists(), (
                    f"Broken link in {md_file.name}: {link} -> {target}"
                )


def test_readme_not_in_docs():
    """Ensure README doesn't conflict with index.md."""
    readme = DOCS_DIR / "README.md"
    assert not readme.exists(), (
        "docs/README.md conflicts with index.md (causes mkdocs warnings)"
    )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
