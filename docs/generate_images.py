"""Regenerate the tutorial chart images embedded under docs/images/.

Run from the repo root with the project venv::

    .venv/bin/python docs/generate_images.py

It renders the chapter 10 charts -- the BEL/RA/CSM liability trajectory and
the CSM analysis-of-change waterfall -- from the built-in sample portfolio,
so the committed images stay in step with the engine and the plot styling.
Re-run it whenever the sample data, the engine output, or plots.py changes.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless: render straight to file, no display needed

import fastcashflow as fcf

_IMAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
_DPI = 150


def _save(ax, name: str) -> None:
    path = os.path.join(_IMAGES, name)
    ax.figure.savefig(path, dpi=_DPI, bbox_inches="tight")
    print(f"wrote {path}")


def main() -> None:
    mps = fcf.load_sample_model_points()
    asmp = fcf.load_sample_assumptions()
    m = fcf.measure(mps, asmp)

    # 10.1 -- BEL/RA/CSM trajectories over the contract's life
    _save(fcf.plot_liability(m), "liability-trajectory.png")

    # 10.2 -- CSM analysis-of-change waterfall, first reporting period
    recon = fcf.reconcile(fcf.roll_forward(m, period_months=12))
    _save(fcf.plot_analysis_of_change(recon[0]), "analysis-of-change.png")


if __name__ == "__main__":
    main()
