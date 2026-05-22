"""Smoke test -- every example script runs end to end without raising.

Keeps ``examples/`` from rotting as the API evolves. ``benchmark.py`` is
excluded: it is deliberately slow.
"""
import runpy
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
_SCRIPTS = sorted(
    p.name for p in _EXAMPLES_DIR.glob("*.py") if p.name != "benchmark.py"
)


@pytest.mark.parametrize("script", _SCRIPTS)
def test_example_runs(script):
    """examples/<script> runs without raising."""
    runpy.run_path(str(_EXAMPLES_DIR / script), run_name="__main__")
