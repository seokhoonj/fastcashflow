"""Smoke tests for the plotting helpers (the viz extra)."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes

import fastcashflow as fcf

@pytest.fixture(scope="module")
def book():
    """A small measured book: model points, assumptions, measurement."""
    asmp = next(iter(fcf.load_sample_assumptions().values()))
    mps = fcf.load_sample_model_points()
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


def test_plot_stochastic_without_kde(book):
    mps, asmp, _ = book
    dist = fcf.value_stochastic(mps, asmp, np.array([0.02, 0.03, 0.04]))
    assert isinstance(fcf.plot_stochastic(dist, kde=False), Axes)


def test_plot_stochastic_rejects_bad_line(book):
    mps, asmp, _ = book
    dist = fcf.value_stochastic(mps, asmp, np.array([0.02, 0.03, 0.04]))
    with pytest.raises(ValueError):
        fcf.plot_stochastic(dist, line="xxx")


def test_plot_risk_adjustment_returns_axes(book):
    _, asmp, m = book
    assert isinstance(fcf.plot_risk_adjustment(m, asmp), Axes)


def test_plot_risk_adjustment_rejects_cost_of_capital(book):
    _, asmp, m = book
    coc = replace(asmp, ra_method="cost_of_capital")
    with pytest.raises(ValueError):
        fcf.plot_risk_adjustment(m, coc)
