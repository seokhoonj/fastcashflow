"""Embedded value -- VNB = PVFP - CoC - TVOG, hand-calc anchors.

The value of new business assembles already-computed pieces: the present value of
a profit signature (PVFP), the frictional cost of holding required capital (CoC),
and the time value of guarantees (TVOG). The anchors pin each piece down --
including that the CoC arithmetic reproduces the engine's own cost-of-capital risk
adjustment for the same capital path (a no-rebuild check).
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import pricing
from fastcashflow.embedded_value import EmbeddedValue, embedded_value
from fastcashflow.profit import ProfitSignature
from fastcashflow.curves import discount_monthly_curve
from fastcashflow.numerics import _cost_of_capital_ra

from conftest import make_death_basis, PATTERNS


def _sig(profit=(100.0, 80.0)):
    profit = np.asarray(profit, dtype=np.float64)
    month_end = (np.arange(1, profit.shape[0] + 1) * 12).astype(np.int64)
    return ProfitSignature(period_months=12, month_end=month_end, profit=profit)


def test_zero_capital_zero_tvog_is_pvfp():
    """With no capital charge and no TVOG, the value is just the PVFP."""
    sig = _sig()
    ev = embedded_value(sig, reference_rate=0.03)
    assert isinstance(ev, EmbeddedValue)
    assert ev.cost_of_capital == 0.0
    assert ev.tvog == 0.0
    assert np.isclose(ev.pvfp, sig.present_value(0.03))
    assert np.isclose(ev.value, sig.present_value(0.03))


def test_cost_of_capital_flat_closed_form():
    """A flat capital C over n months at annual spread s costs, per month,
    (s/12) * C, present-valued: (s/12) * C * sum(df_bom)."""
    n = 60
    r_m = (1.03) ** (1.0 / 12.0) - 1.0
    dm = np.full(n, r_m)
    C, s = 1_000_000.0, 0.06
    rc = np.full(n, C)
    df_bom = np.concatenate([[1.0], np.cumprod(1.0 / (1.0 + dm))[:-1]])  # (n,)
    expected = (s / 12.0) * C * float(df_bom.sum())

    ev = embedded_value(_sig(), reference_rate=0.03, discount_monthly=dm,
                        required_capital=rc, frictional_spread=s)
    assert np.isclose(ev.cost_of_capital, expected)
    assert np.isclose(ev.value, ev.pvfp - expected)


def test_cost_of_capital_reproduces_engine_coc_ra():
    """The CoC sum reproduces the engine's cost-of-capital RA inception value for
    the same capital path -- the confidence-level ra_path IS that capital."""
    basis = make_death_basis(mortality_q=0.001, lapse_q=0.003, discount_annual=0.03,
                             mortality_cv=0.10)
    mp = fcf.ModelPoints.single(40, 60_000.0, 60, benefits={"DEATH": 1e8},
                                calculation_methods=PATTERNS)
    cl = fcf.gmm.measure(mp, basis, full=True)          # confidence-level RA path
    n_time = cl.ra_path.shape[1] - 1
    dm = discount_monthly_curve(basis, n_time)
    rc = cl.ra_path.sum(axis=0)                         # (n_time+1,) portfolio capital
    coc_rate = 0.06

    ev = embedded_value(_sig(), reference_rate=0.03, discount_monthly=dm,
                        required_capital=rc, frictional_spread=coc_rate)
    expected = float(_cost_of_capital_ra(cl.ra_path, dm, coc_rate)[:, 0].sum())
    assert np.isclose(ev.cost_of_capital, expected)


def test_tvog_subtracts():
    sig = _sig()
    ev0 = embedded_value(sig, reference_rate=0.03)
    evt = embedded_value(sig, reference_rate=0.03, tvog=25.0)
    assert evt.tvog == 25.0
    assert np.isclose(evt.value, ev0.value - 25.0)


def test_scalar_capital_factor_matches_explicit_array():
    """A scalar required_capital is a factor on the reserve -- equal to passing the
    pre-multiplied array."""
    n = 48
    dm = np.full(n, (1.025) ** (1.0 / 12.0) - 1.0)
    V = np.linspace(0.0, 5_000_000.0, n + 1)            # (n+1,) reserve path
    kw = dict(reference_rate=0.025, discount_monthly=dm, frictional_spread=0.05)

    ev_scalar = embedded_value(_sig(), required_capital=0.04, reserve=V, **kw)
    ev_array = embedded_value(_sig(), required_capital=0.04 * V, **kw)
    assert np.isclose(ev_scalar.cost_of_capital, ev_array.cost_of_capital)
    assert ev_scalar.cost_of_capital > 0.0


def test_scalar_capital_needs_reserve():
    with pytest.raises(ValueError, match="reserve="):
        embedded_value(_sig(), reference_rate=0.03, discount_monthly=np.full(12, 0.002),
                       required_capital=0.04, frictional_spread=0.05)
