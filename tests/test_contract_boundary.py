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
from fastcashflow.model_points import ModelPoints

_MORT = lambda s, a, d: np.full(np.shape(a), 0.01)
_ZERO = lambda s, a, d: np.full(np.shape(d), 0.0)


def _basis():
    return Basis(mortality_annual=_MORT, lapse_annual=_ZERO,
                 discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.10,
                 coverages=(fcf.CoverageRate("DEATH", _MORT),))


def _mp(boundary):
    return ModelPoints(
        issue_age=np.array([40], dtype=np.int64),
        benefits={"DEATH": np.array([100_000.0])},
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
    assert int(np.sum(cut.cashflows.mortality_cf[0] > 0)) == 12
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
# CB3 -- the fast path applies the same boundary cut as the full path
# ---------------------------------------------------------------------------
def test_fast_path_matches_full_under_boundary_cut():
    full = fcf.gmm.measure(_mp(12), _basis(), full=True)
    fast = fcf.gmm.measure(_mp(12), _basis(), full=False)
    assert fast.bel[0] == pytest.approx(full.bel[0], rel=1e-9)
    # and a no-cut policy still agrees
    assert (fcf.gmm.measure(_mp(None), _basis(), full=False).bel[0]
            == pytest.approx(fcf.gmm.measure(_mp(None), _basis()).bel[0], rel=1e-9))


# ---------------------------------------------------------------------------
# CB4 -- the file reader wires contract_boundary_months through (not dropped
# into ModelPoints.attributes), so a policies.csv column actually cuts the
# projection. Same hand-checked case as CB1, read from disk.
# ---------------------------------------------------------------------------
def test_reader_applies_contract_boundary_months(tmp_path):
    (tmp_path / "policies.csv").write_text(
        "mp_id,issue_age,sex,term_months,premium,contract_boundary_months\n"
        "1,40,0,24,100,12\n"
    )
    (tmp_path / "coverages.csv").write_text(
        "mp_id,coverage,amount\n1,DEATH,100000\n"
    )
    (tmp_path / "methods.csv").write_text(
        "coverage,calculation_method\nDEATH,DEATH\n"
    )
    mp = fcf.read_model_points(
        tmp_path / "policies.csv",
        coverages=tmp_path / "coverages.csv",
        calculation_methods=tmp_path / "methods.csv",
    )
    # the column is the engine field, not a stray grouping attribute
    assert mp.contract_boundary_months is not None
    assert int(mp.contract_boundary_months[0]) == 12
    assert not (mp.attributes or {}).get("contract_boundary_months") is not None
    # and the cut actually bites: the read MP matches the direct-construction
    # boundary=12 BEL, and differs from the no-cut boundary
    read_bel = fcf.gmm.measure(mp, _basis()).bel[0]
    assert read_bel == pytest.approx(fcf.gmm.measure(_mp(12), _basis()).bel[0],
                                     rel=1e-12)
    assert read_bel != pytest.approx(fcf.gmm.measure(_mp(None), _basis()).bel[0])


# ---------------------------------------------------------------------------
# CB5 -- an in-force as-of date past the boundary is a clear ValueError, not
# the IndexError the bel_path slice would otherwise raise (the trajectory is
# only contract_boundary_months wide, not term_months).
# ---------------------------------------------------------------------------
def _inforce_mp(term, boundary, elapsed):
    return ModelPoints(
        issue_age=np.array([40], dtype=np.int64),
        benefits={"DEATH": np.array([100_000.0])},
        premium=np.array([100.0]),
        term_months=np.array([term], dtype=np.int64),
        contract_boundary_months=np.array([boundary], dtype=np.int64),
        elapsed_months=np.array([elapsed], dtype=np.int64),
        count=np.array([1.0]),
        calculation_methods={"DEATH": fcf.CalculationMethod.DEATH})


def test_inforce_past_boundary_raises_value_error():
    from fastcashflow.gmm._engine import _measure_inforce_fast, _measure_inforce_full
    # term 120, boundary 12, elapsed 24 -- inside the term but past the
    # boundary, the case that used to IndexError off the 12-wide trajectory
    mp = _inforce_mp(term=120, boundary=12, elapsed=24)
    with pytest.raises(ValueError, match="contract_boundary_months"):
        _measure_inforce_fast(mp, _basis())
    # the full path (settlement / roll_forward) guards the same way
    with pytest.raises(ValueError, match="contract_boundary_months"):
        _measure_inforce_full(mp, _basis(), prior_csm=np.array([0.0]),
                              lock_in_rate=0.0, period_months=12)


def test_inforce_within_boundary_is_fine():
    from fastcashflow.gmm._engine import _measure_inforce_fast
    mp = _inforce_mp(term=120, boundary=12, elapsed=6)
    v = _measure_inforce_fast(mp, _basis())
    assert np.isfinite(v.bel[0])
