"""Smoke tests for the plotting helpers (the viz extra)."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes

import fastcashflow as fcf

_SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_basis.xlsx"


@pytest.fixture(scope="module")
def book():
    """A small measured book: model points, assumptions, measurement."""
    asmp = fcf.read_assumptions(_SAMPLE)
    mps = fcf.ModelPointSet(
        issue_age=np.array([40, 45]),
        death_benefit=np.array([1e8, 8e7]),
        monthly_premium=np.array([25_000.0, 22_000.0]),
        term_months=np.array([120, 120]),
    )
    return mps, asmp, fcf.measure(mps, asmp)


def test_plot_liability_returns_axes(book):
    _, _, m = book
    assert isinstance(fcf.plot_liability(m), Axes)


def test_plot_csm_runoff_returns_axes(book):
    _, _, m = book
    assert isinstance(fcf.plot_csm_runoff(m), Axes)


def test_plot_cashflows_returns_axes(book):
    _, _, m = book
    assert isinstance(fcf.plot_cashflows(m), Axes)


def test_plot_cashflows_rejects_bad_period(book):
    _, _, m = book
    with pytest.raises(ValueError):
        fcf.plot_cashflows(m, period_months=0)


def test_plot_analysis_of_change_returns_axes(book):
    _, _, m = book
    recon = fcf.reconcile(fcf.roll_forward(m, period_months=12))[0]
    assert isinstance(fcf.plot_analysis_of_change(recon), Axes)


def test_plot_analysis_of_change_rejects_bad_component(book):
    _, _, m = book
    recon = fcf.reconcile(fcf.roll_forward(m, period_months=12))[0]
    with pytest.raises(ValueError):
        fcf.plot_analysis_of_change(recon, component="xxx")


def test_plot_stochastic_returns_axes(book):
    mps, asmp, _ = book
    dist = fcf.value_stochastic(mps, asmp, np.array([0.02, 0.03, 0.04]))
    assert isinstance(fcf.plot_stochastic(dist), Axes)


def test_plot_stochastic_rejects_bad_line(book):
    mps, asmp, _ = book
    dist = fcf.value_stochastic(mps, asmp, np.array([0.02, 0.03, 0.04]))
    with pytest.raises(ValueError):
        fcf.plot_stochastic(dist, line="xxx")
