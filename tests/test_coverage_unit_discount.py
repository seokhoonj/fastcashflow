"""B119 coverage-unit discounting accounting-policy choice -- hand-calc anchor.

IFRS 17 B119 leaves to judgement whether future coverage units are discounted
when allocating the CSM release. ``Basis.coverage_unit_discount`` exposes the
choice (default False = undiscounted). The two choices reallocate the release
over time (same total, different timing) whenever the discount rate is non-zero;
this test derives both tails by an independent backward recursion and checks the
engine matches each.
"""
import numpy as np

from fastcashflow import ModelPoints
from fastcashflow.gmm import measure
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


def _profitable(**basis_overrides):
    kw = dict(
        issue_age=35, benefits={"DEATH": 50_000_000.0}, premium=80_000.0,
        term_months=4, calculation_methods=PATTERNS,
    )
    over = dict(
        mortality_annual=lambda s, a, d: np.full(a.shape, _annual(0.001)),
        lapse_annual=lambda s, a, d: np.full(d.shape, _annual(0.01)),
        discount_annual=0.03,
    )
    over.update(basis_overrides)
    return measure(ModelPoints.single(**kw), make_death_basis(**over))


def test_coverage_unit_discount_matches_hand_calc():
    base = _profitable()                                # undiscounted (default)
    disc = _profitable(coverage_unit_discount=True)     # discounted units

    # Inception CSM_0 is the floor max(0, -FCF); unaffected by the choice.
    assert base.csm_path[0, 0] > 0.0
    assert np.isclose(disc.csm_path[0, 0], base.csm_path[0, 0])
    csm0 = base.csm_path[0, 0]

    # Coverage units = in-force, identical for both runs (projection is the
    # same; only the CSM release tail differs).
    cu = base.cashflows.inforce[0]
    i_m = (1.0 + 0.03) ** (1.0 / 12.0) - 1.0
    accreted = csm0 * (1.0 + i_m)                       # one month's accretion

    # t=1 undiscounted release: tail = sum of coverage units.
    rel_undisc = accreted * cu[0] / cu.sum()
    assert np.isclose(base.csm_path[0, 1], accreted - rel_undisc)

    # t=1 discounted release: tail built back at the locked-in rate, exactly
    # as the kernel does: dt[s] = cu[s] + dt[s+1] / (1 + i_m).
    n = len(cu)
    dt = np.empty(n)
    dt[-1] = cu[-1]
    for s in range(n - 2, -1, -1):
        dt[s] = cu[s] + dt[s + 1] / (1.0 + i_m)
    rel_disc = accreted * cu[0] / dt[0]
    assert np.isclose(disc.csm_path[0, 1], accreted - rel_disc)

    # The choice actually changes the release (discounting shrinks the tail,
    # so the first period's share rises).
    assert not np.isclose(base.csm_path[0, 1], disc.csm_path[0, 1])
    assert disc.csm_path[0, 1] < base.csm_path[0, 1]

    # Same total: both fully release the CSM to ~0 by end of term.
    assert np.isclose(base.csm_path[0, -1], 0.0, atol=1e-6)
    assert np.isclose(disc.csm_path[0, -1], 0.0, atol=1e-6)


def test_coverage_unit_discount_default_is_undiscounted():
    """Default basis (no flag) reproduces the undiscounted release exactly."""
    a = _profitable()
    b = _profitable(coverage_unit_discount=False)
    assert np.allclose(a.csm_path, b.csm_path)


def test_settle_honors_coverage_unit_discount():
    """The period-close (settlement) B119 release responds to the policy too --
    the path the close pack uses, not just the inception measurement."""
    import pytest
    import fastcashflow as fcf
    from fastcashflow import (Basis, CalculationMethod, CoverageRate,
                              InforceState, ModelPoints)
    settle = getattr(fcf.gmm, "settle", None)
    if settle is None:
        pytest.skip("gmm.settle not available")

    CM = {"DEATH": CalculationMethod.DEATH}
    flat = lambda v: (lambda s, a, d: np.full(d.shape, v, float))

    def mkbasis(du):
        return Basis(
            mortality_annual=flat(0.012), lapse_annual=flat(0.05),
            discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
            coverages=(CoverageRate("DEATH", flat(0.012)),),
            coverage_unit_discount=du)

    def book(basis, em_open=12, period=12, scale=1000.0, term=36, prior_csm=5000.0):
        unit = ModelPoints(
            issue_age=np.array([40]), premium=np.array([100.0]),
            term_months=np.array([term]), benefits={"DEATH": np.array([1e6])},
            count=np.array([1.0]), calculation_methods=CM)
        surv = fcf.gmm.measure(unit, basis, full=True).cashflows.inforce[0]
        em_close = em_open + period
        ids = np.array(["P0"])
        cc = scale * (surv[em_close] if em_close < surv.shape[0] else 0.0)
        mp = ModelPoints(
            issue_age=np.array([40]), premium=np.array([100.0]),
            term_months=np.array([term]), benefits={"DEATH": np.array([1e6])},
            count=np.array([cc]), elapsed_months=np.array([em_close]),
            mp_id=ids, product=np.array(["A"]), calculation_methods=CM)
        st = InforceState(
            mp_id=ids, elapsed_months=np.array([em_close]), count=np.array([cc]),
            prior_csm=np.array([prior_csm]), lock_in_rate=basis.discount_annual,
            prior_count=np.array([scale * surv[em_open]]))
        return mp, st

    bf, bt = mkbasis(False), mkbasis(True)
    mp, st = book(bf)
    mv_f = settle(mp, st, bf, period_months=12)
    mv_t = settle(mp, st, bt, period_months=12)
    rel_f = float(np.sum(mv_f.csm_release))
    rel_t = float(np.sum(mv_t.csm_release))
    # Discounting the future coverage units shrinks the tail, so the provided
    # period's release share rises.
    assert not np.isclose(rel_f, rel_t)
    assert rel_t > rel_f
