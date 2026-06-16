"""Smoke tests for the plotting helpers (the viz extra)."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes

import fastcashflow as fcf

@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test opened -- keeps the figure count bounded."""
    yield
    matplotlib.pyplot.close("all")


@pytest.fixture(scope="module")
def book():
    """A small measured book: model points, basis, measurement."""
    basis = next(iter(fcf.samples.basis().segments.values()))
    mps = fcf.samples.model_points()
    return mps, basis, fcf.gmm.measure(mps, basis)


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
    mps, basis, _ = book
    dist = fcf.gmm.stochastic(mps, basis, np.array([0.02, 0.03, 0.04]))
    assert isinstance(fcf.plot_stochastic(dist), Axes)


def test_plot_stochastic_without_kde(book):
    mps, basis, _ = book
    dist = fcf.gmm.stochastic(mps, basis, np.array([0.02, 0.03, 0.04]))
    assert isinstance(fcf.plot_stochastic(dist, kde=False), Axes)


def test_plot_stochastic_rejects_bad_line(book):
    mps, basis, _ = book
    dist = fcf.gmm.stochastic(mps, basis, np.array([0.02, 0.03, 0.04]))
    with pytest.raises(ValueError):
        fcf.plot_stochastic(dist, line="xxx")


def test_plot_risk_adjustment_returns_axes(book):
    _, basis, m = book
    assert isinstance(fcf.plot_risk_adjustment(m, basis), Axes)


def test_plot_risk_adjustment_rejects_cost_of_capital(book):
    _, basis, m = book
    coc = replace(basis, ra_method="cost_of_capital")
    with pytest.raises(ValueError):
        fcf.plot_risk_adjustment(m, coc)


def test_plot_risk_adjustment_works_headline_only(book):
    """The RA fan only needs the headline figures, so full=False suffices."""
    mps, basis, _ = book
    m = fcf.gmm.measure(mps, basis, full=False)
    assert isinstance(fcf.plot_risk_adjustment(m, basis), Axes)


# ---------------------------------------------------------------------------
# Model dispatch -- VFA / PAA / reinsurance arms
# ---------------------------------------------------------------------------
def _flat_basis() -> fcf.Basis:
    """A tiny flat basis a single synthetic model point can be measured on."""
    death_fn = lambda sex, issue_age, duration: np.full(issue_age.shape, 0.012)
    return fcf.Basis(
        mortality_annual=death_fn,
        lapse_annual=lambda sex, issue_age, duration: np.full(duration.shape, 0.05),
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.10,
        investment_return=0.06,
        fund_fee=0.015,
        coverages=(fcf.CoverageRate("DEATH", death_fn),),
    )


@pytest.fixture(scope="module")
def vfa_book():
    """A VFA-measured account-value contract (expense_cv drives its RA)."""
    vfa_basis = replace(
        _flat_basis(), expense_cv=0.10,
        expense_items=(fcf.ExpenseItem("maintenance", "gamma_fixed", 60_000.0),))
    m = fcf.vfa.measure(
        fcf.ModelPoints.single(40, 0.0, 60, account_value=1e8,
            calculation_methods={"DEATH": fcf.CalculationMethod.DEATH}), vfa_basis)
    return vfa_basis, m


@pytest.fixture(scope="module")
def paa_m():
    return fcf.paa.measure(
        fcf.ModelPoints.single(40, 50_000.0, 12, benefits={"DEATH": 1e8},
            calculation_methods={"DEATH": fcf.CalculationMethod.DEATH}),
        _flat_basis())


@pytest.fixture(scope="module")
def reins_m():
    return fcf.reinsurance.measure(
        fcf.ModelPoints.single(40, 50_000.0, 120, benefits={"DEATH": 1e8},
            calculation_methods={"DEATH": fcf.CalculationMethod.DEATH}),
        _flat_basis(), treaty=fcf.reinsurance.QuotaShare(0.5))


def test_plot_liability_dispatches_per_model(vfa_book, paa_m, reins_m):
    _, vfa_m = vfa_book
    for m in (vfa_m, paa_m, reins_m):
        assert isinstance(fcf.plot_liability(m), Axes)


def test_plot_cashflows_dispatches_per_model(vfa_book, paa_m, reins_m):
    _, vfa_m = vfa_book
    for m in (vfa_m, paa_m, reins_m):
        assert isinstance(fcf.plot_cashflows(m), Axes)


def test_plot_csm_runoff_vfa_and_reinsurance(vfa_book, reins_m):
    _, vfa_m = vfa_book
    assert isinstance(fcf.plot_csm_runoff(vfa_m), Axes)
    assert isinstance(fcf.plot_csm_runoff(reins_m), Axes)


def test_plot_csm_runoff_rejects_paa(paa_m):
    """The PAA carries no CSM -- the runoff chart points at plot_liability."""
    with pytest.raises(TypeError, match="no CSM"):
        fcf.plot_csm_runoff(paa_m)


def test_plot_risk_adjustment_vfa_and_reinsurance(vfa_book, reins_m):
    vfa_basis, vfa_m = vfa_book
    assert isinstance(fcf.plot_risk_adjustment(vfa_m, vfa_basis), Axes)
    assert isinstance(fcf.plot_risk_adjustment(reins_m, _flat_basis()), Axes)


def test_plot_risk_adjustment_rejects_paa(paa_m):
    with pytest.raises(TypeError, match="no explicit risk adjustment"):
        fcf.plot_risk_adjustment(paa_m, _flat_basis())


def test_plot_risk_adjustment_margin_direction(book, reins_m):
    """A direct RA adds to the liability (markers right of the BEL); the
    reinsurance-held RA reduces the net cost (markers left of the BEL)."""
    _, basis, m = book
    ax = fcf.plot_risk_adjustment(m, basis)
    bands = [l for l in ax.lines if l.get_linestyle() == "--"]
    assert bands and all(l.get_xdata()[0] > float(m.bel.sum()) for l in bands)

    ax = fcf.plot_risk_adjustment(reins_m, _flat_basis())
    bands = [l for l in ax.lines if l.get_linestyle() == "--"]
    assert bands and all(
        l.get_xdata()[0] < float(reins_m.bel.sum()) for l in bands)


def test_plot_analysis_of_change_dispatches_per_model(vfa_book, paa_m, reins_m):
    _, vfa_m = vfa_book
    for m in (vfa_m, paa_m, reins_m):
        recon = fcf.reconcile(fcf.roll_forward(m, period_months=12))[0]
        assert isinstance(fcf.plot_analysis_of_change(recon), Axes)


def test_plot_analysis_of_change_paa_components(paa_m):
    """The PAA waterfall draws each paragraph-100 block."""
    recon = fcf.reconcile(fcf.roll_forward(paa_m, period_months=12))[0]
    for component in ("lrc", "loss_component", "lic_path"):
        assert isinstance(
            fcf.plot_analysis_of_change(recon, component=component), Axes)
    with pytest.raises(ValueError, match="lrc"):
        fcf.plot_analysis_of_change(recon, component="csm")


def test_plot_rejects_unknown_measurement():
    with pytest.raises(TypeError, match="GMM, PAA, VFA or reinsurance"):
        fcf.plot_liability(object())


def test_plot_rejects_portfolio_with_slot_hint(book):
    """A portfolio container is refused with a pointer at the model slots."""
    mps, _, _ = book
    pm = fcf.portfolio.measure(mps, fcf.samples.basis())
    with pytest.raises(TypeError, match="model slot"):
        fcf.plot_liability(pm)


def test_plot_liability_requires_full(book):
    mps, basis, _ = book
    m = fcf.gmm.measure(mps, basis, full=False)
    with pytest.raises(ValueError, match="full=True"):
        fcf.plot_liability(m)
