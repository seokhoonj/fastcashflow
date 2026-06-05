"""IFRS 17 contract boundary (Sec. 34) -- ``ModelPoints.contract_boundary_months``.

The projection stops at the boundary; cash flows past it leave the current
contract, and the maturity benefit is paid only when the boundary reaches the
coverage term. ``None`` (the default) keeps the boundary at ``term_months`` --
no cut, the historical behaviour.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.basis import Basis
from fastcashflow.modelpoints import ModelPoints

_MORT = lambda s, a, d: np.full(np.shape(a), 0.01)
_ZERO = lambda s, a, d: np.full(np.shape(d), 0.0)


def _basis():
    return Basis(mortality_annual=_MORT, lapse_annual=_ZERO,
                 discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
                 coverages=(fcf.CoverageRate("DEATH", _MORT),))


def _mp(boundary):
    return ModelPoints(
        issue_age=np.array([40], dtype=np.int64),
        benefits={0: np.array([100_000.0])},
        premium=np.array([100.0]),
        term_months=np.array([24], dtype=np.int64),
        maturity_benefit=np.array([1000.0]),
        contract_boundary_months=(None if boundary is None
                                  else np.array([boundary], dtype=np.int64)),
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})


# ---------------------------------------------------------------------------
# CB0 -- default boundary == term is bit-identical to no boundary
# ---------------------------------------------------------------------------
def test_default_boundary_is_term():
    none = fcf.gmm.measure(_mp(None), _basis())
    explicit = fcf.gmm.measure(_mp(24), _basis())     # boundary = term
    assert none.bel[0] == pytest.approx(explicit.bel[0], rel=1e-12)
    # full=False (fast path) agrees too when there is no cut
    assert (fcf.gmm.measure(_mp(None), _basis(), full=False).bel[0]
            == pytest.approx(none.bel[0], rel=1e-9))


# ---------------------------------------------------------------------------
# CB1 -- a boundary short of the term cuts cash flows and withholds maturity
# ---------------------------------------------------------------------------
def test_boundary_cut_withholds_maturity_and_cash_flows():
    full = fcf.gmm.measure(_mp(24), _basis())
    cut = fcf.gmm.measure(_mp(12), _basis())
    # maturity (paid at term=24) is past the 12-month boundary -> not paid
    assert full.cashflows.maturity_cf[0] > 0.0
    assert cut.cashflows.maturity_cf[0] == pytest.approx(0.0)
    # premium / claim only within the boundary
    assert int(np.sum(full.cashflows.premium_cf[0] > 0)) == 24
    assert int(np.sum(cut.cashflows.premium_cf[0] > 0)) == 12
    assert int(np.sum(cut.cashflows.claim_cf[0] > 0)) == 12
    # the cut excludes future claims + the maturity -> a different BEL
    assert cut.bel[0] != pytest.approx(full.bel[0])


# ---------------------------------------------------------------------------
# CB2 -- validation
# ---------------------------------------------------------------------------
def test_boundary_cannot_exceed_term():
    with pytest.raises(ValueError, match="must not exceed term_months"):
        _mp(36)                                       # boundary 36 > term 24


def test_boundary_must_be_positive():
    with pytest.raises(ValueError, match=">= 1"):
        _mp(0)


# ---------------------------------------------------------------------------
# CB3 -- the fast path rejects a boundary cut (until it carries the boundary)
# ---------------------------------------------------------------------------
def test_fast_path_rejects_boundary_cut():
    with pytest.raises(NotImplementedError, match="contract boundary"):
        fcf.gmm.measure(_mp(12), _basis(), full=False)
    # but the default (no cut) is fine on the fast path
    fcf.gmm.measure(_mp(None), _basis(), full=False)
