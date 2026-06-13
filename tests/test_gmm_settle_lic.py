"""gmm.settle -- the liability-for-incurred-claims (LIC) block under a
settlement_pattern basis (paragraphs 40(b) / 42 / 103(b)).

Authoritative skeleton (P-5c pattern). Anchors from dev/lic-settle-gate.md and
the hand-calc oracle dev/scratch_lic_settle_gate.py.

gmm.settle previously REJECTED a settlement_pattern basis (no LIC movement
lines). The LIC trajectory is already built by the measure
(``_settlement_lic``, GMMMeasurement.lic) and the BEL already discounts claims
to their payment dates; this adds the four LIC movement lines, mirroring
paa.settle exactly::

    lic_closing == lic_opening + claims_incurred - claims_paid

entirely at the expected scale (k_exp), claims_paid the residual. The LIC is
the undiscounted incurred-but-unpaid balance, reconstructed from the unit
projection each period (no prior_lic on the state).
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, CalculationMethod, CoverageRate, InforceState, ModelPoints)
from fastcashflow.movement import GMMSettlementMovement

settle = getattr(fcf.gmm, "settle", None)
_HAS_LIC = (settle is not None
            and "lic_closing" in GMMSettlementMovement.__dataclass_fields__)
pytestmark = pytest.mark.skipif(
    not _HAS_LIC,
    reason="gmm.settle LIC block not implemented yet (the skeleton activates "
           "unchanged once the four LIC movement lines land)")

CM = {"DEATH": CalculationMethod.DEATH}
PATTERN = np.array([0.6, 0.3, 0.1])   # 60% paid on incurrence, 30% +1m, 10% +2m


def _flat(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _basis(*, discount=0.03, settlement=True):
    return Basis(
        mortality_annual=_flat(0.012), lapse_annual=_flat(0.05),
        discount_annual=discount, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat(0.012)),),
        settlement_pattern=PATTERN if settlement else None,
    )


def _book(basis, *, em_open=12, period=12, term=36, scale=1000.0,
          prior_csm=5_000.0):
    em_close = em_open + period
    surv = fcf.gmm.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([100.0]),
                    term_months=np.array([term]), benefits={0: np.array([1e6])},
                    count=np.array([1.0]), calculation_methods=CM),
        basis, full=True).cashflows.inforce[0]
    prior_count = scale * surv[em_open]
    count_close = scale * surv[em_close]
    ids = np.array(["P0"])
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([term]), benefits={0: np.array([1e6])},
        count=np.array([count_close]), elapsed_months=np.array([em_close]),
        mp_id=ids, product=np.array(["A"]), calculation_methods=CM)
    state = InforceState(
        mp_id=ids, elapsed_months=np.array([em_close]),
        count=np.array([count_close]), prior_csm=np.array([prior_csm]),
        lock_in_rate=basis.discount_annual,
        prior_count=np.array([prior_count]))
    return mp, state


# ---------------------------------------------------------------------------
# the settlement_pattern basis is accepted; the block reconciles
# ---------------------------------------------------------------------------

def test_settlement_pattern_basis_is_accepted_and_carries_lic_lines():
    basis = _basis()
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=12)
    for nm in ("lic_opening", "claims_incurred", "claims_paid", "lic_closing"):
        assert hasattr(mv, nm)


def test_lic_block_identity():
    basis = _basis()
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_allclose(
        mv.lic_opening + mv.claims_incurred - mv.claims_paid,
        mv.lic_closing, rtol=1e-10)


def test_settlement_pattern_leaves_an_outstanding_lic():
    """A real payment lag means claims are still outstanding at both dates:
    the LIC opening and closing are strictly positive on a claims-paying
    in-force book mid-coverage."""
    basis = _basis()
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=12)
    assert np.all(mv.lic_opening > 0.0)
    assert np.all(mv.lic_closing > 0.0)
    assert np.all(mv.claims_incurred > 0.0)
    assert np.all(mv.claims_paid > 0.0)


def test_lic_opening_is_k_exp_times_the_unit_trajectory():
    """The LIC opening is the unit LIC trajectory at em_open scaled by k_exp --
    the same reconstruction paa.settle uses."""
    basis = _basis()
    em_open, period, term, scale = 12, 12, 36, 1000.0
    mp, state = _book(basis, em_open=em_open, period=period, term=term,
                      scale=scale)
    mv = settle(mp, state, basis, period_months=period)
    unit = fcf.gmm.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([100.0]),
                    term_months=np.array([term]), benefits={0: np.array([1e6])},
                    count=np.array([1.0]), calculation_methods=CM),
        basis, full=True)
    surv_open = unit.cashflows.inforce[0][em_open]
    k_exp = (scale * surv_open) / surv_open      # == scale (prior_count/surv)
    np.testing.assert_allclose(
        mv.lic_opening[0], k_exp * unit.lic[0][em_open], rtol=1e-9)


# ---------------------------------------------------------------------------
# no settlement pattern => zero LIC balance; existing lines unchanged
# ---------------------------------------------------------------------------

def test_no_pattern_has_zero_lic_balance_and_claims_paid_equals_incurred():
    """settlement_pattern=None: claims are paid as incurred, so the LIC balance
    is zero at both dates and claims_paid == claims_incurred. The LRC lines are
    the pre-feature settle (the existing suite pins those)."""
    basis = _basis(settlement=False)
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=12)
    np.testing.assert_array_equal(mv.lic_opening, 0.0)
    np.testing.assert_array_equal(mv.lic_closing, 0.0)
    np.testing.assert_allclose(mv.claims_paid, mv.claims_incurred, rtol=1e-12)


# ---------------------------------------------------------------------------
# the LIC lines thread through the reconciliation and the aggregate
# ---------------------------------------------------------------------------

def test_reconciliation_and_aggregate_carry_the_lic_lines():
    basis = _basis()
    mp, state = _book(basis)
    mv = settle(mp, state, basis, period_months=12)
    recon = fcf.reconcile([mv])[0]
    assert hasattr(recon, "lic_closing")
    # claims_paid is a run-off, stored negative by display convention (mirrors
    # the PAA reconciliation).
    np.testing.assert_allclose(
        recon.claims_paid, -float(mv.claims_paid.sum()), rtol=1e-10)
    agg = fcf.gmm.settle_aggregate(mp, state, basis, period_months=12)
    np.testing.assert_allclose(
        agg.lic_closing, float(mv.lic_closing.sum()), rtol=1e-9)
    np.testing.assert_allclose(
        agg.claims_incurred, float(mv.claims_incurred.sum()), rtol=1e-9)
