"""Runs every docs/user-guides/*.py example so a `--8<--` snippet in the docs always reflects real, currently-working code."""

import runpy
from pathlib import Path

import pytest

_DOCS_DIR = Path(__file__).parent.parent / "docs" / "user-guides"
_EXAMPLE_SCRIPTS = sorted(_DOCS_DIR.glob("*.py"))

assert _EXAMPLE_SCRIPTS, f"No example scripts found in {_DOCS_DIR}"


@pytest.mark.parametrize("script", _EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_doc_example_runs_without_error(script: Path) -> None:
    """Executes the script top to bottom; any raised exception fails the test."""
    runpy.run_path(str(script), run_name="__main__")
