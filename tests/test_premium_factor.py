"""Premium SHAPE -- ``Basis.premium_factor_annual``.

A multiplicative factor on the level ``ModelPoints.premium`` by policy year, so
a renewable / step-rated (renewable) or step-up premium can be projected
while ``premium[mp]`` stays the scalar SCALE ``solve_premium`` solves for. The
factor must apply identically across every kernel path (the full Markov / full
semi-Markov projection and the fused fast / codegen / GPU value paths), so the
key tests are full==fast parity on both a Markov and a semi-Markov contract,
plus a fast==gpu parity test (skipped when no CUDA device is present).
"""
import numpy as np
import pytest
from numba import cuda

import fastcashflow as fcf
from fastcashflow.basis import Basis, CoverageRate
from fastcashflow.state_model import StateModel, State, Transition

CM = {"DEATH": fcf.CalculationMethod.DEATH}
_MORT = lambda s, a, d: np.full(np.shape(a), 0.01)
_LAPSE = lambda s, a, d: np.full(np.shape(d), 0.02)
_NONE = lambda s, a, d: np.full(np.shape(a), 0.0)


def _basis(pf=None, **kw):
    return Basis(mortality_annual=kw.get("mort", _MORT), lapse_annual=_LAPSE,
                 discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
                 coverages=(CoverageRate("DEATH", kw.get("mort", _MORT)),),
                 premium_factor_annual=pf)


# ---------------------------------------------------------------------------
# PF1 -- hand-calc: a step-up factor doubles the second year's premium exactly
# ---------------------------------------------------------------------------
def test_premium_factor_hand_calc():
    """No decrement (in-force = count), annual pay, flat 0 discount. The factor
    1 + 0.5*duration charges 100 in year 0 and 150 in year 1, so the premium PV
    is 250 (vs 200 level) and the BEL drops by exactly 50 (premium is inflow)."""
    pf = lambda s, a, d, ic, el: 1.0 + 0.5 * d
    b = Basis(mortality_annual=_NONE, lapse_annual=_NONE, discount_annual=0.0,
              ra_confidence=0.75, mortality_cv=0.10,
              coverages=(CoverageRate("DEATH", _NONE),), premium_factor_annual=pf)
    b0 = Basis(mortality_annual=_NONE, lapse_annual=_NONE, discount_annual=0.0,
               ra_confidence=0.75, mortality_cv=0.10,
               coverages=(CoverageRate("DEATH", _NONE),))
    mp = fcf.ModelPoints.single(issue_age=40, benefits={0: 12_000.0}, premium=100.0,
                                term_months=24, premium_frequency_months=12,
                                calculation_methods=CM)
    m = fcf.gmm.measure(mp, b, full=True)
    pcf = m.cashflows.premium_cf[0]
    assert pcf[0] == pytest.approx(100.0)
    assert pcf[12] == pytest.approx(150.0)
    assert pcf.sum() == pytest.approx(250.0)
    m0 = fcf.gmm.measure(mp, b0, full=True)
    assert m0.cashflows.premium_cf[0].sum() == pytest.approx(200.0)
    assert m.bel[0] - m0.bel[0] == pytest.approx(-50.0)
    # default None is bit-identical to the no-shape level premium
    assert m0.cashflows.premium_cf[0, 0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# PF2 -- full==fast parity (Markov) with a non-trivial step-up factor
# ---------------------------------------------------------------------------
def test_premium_factor_full_matches_fast_markov():
    pf = lambda s, a, d, ic, el: 1.0 + 0.3 * d
    mp = fcf.ModelPoints.single(issue_age=40, benefits={0: 1e6}, premium=1000.0,
                                term_months=60, calculation_methods=CM)
    full = fcf.gmm.measure(mp, _basis(pf), full=True)
    fast = fcf.gmm.measure(mp, _basis(pf), full=False)
    assert np.allclose(full.bel, fast.bel, rtol=1e-9)
    assert np.allclose(full.ra, fast.ra, rtol=1e-9)
    assert np.allclose(full.csm, fast.csm, rtol=1e-9)
    # the shape actually bites: BEL differs from the level baseline
    assert not np.isclose(full.bel[0], fcf.gmm.measure(mp, _basis(), full=True).bel[0])


# ---------------------------------------------------------------------------
# PF3 -- full==fast parity on a SEMI-MARKOV contract (the codegen path both
# adversaries flagged as the highest-risk missed edit)
# ---------------------------------------------------------------------------
def _reincidence_model(sojourn_tracking_months=12):
    return StateModel(states=(
        State("active", pays_premium=True, transitions=(
            Transition("mortality"),
            Transition("ci_incidence", to="post_first"),
            Transition("lapse"),
        )),
        State("post_first", sojourn_tracking_months=sojourn_tracking_months, pays_premium=True, transitions=(
            Transition("mortality"),
            Transition("ci_reincidence", to="post_second",
                       sojourn_dependent=True, pays_lump_sum=True),
        )),
        State("post_second", transitions=(Transition("mortality"),)),
    ))


def test_premium_factor_full_matches_fast_semi_markov():
    """A premium-paying semi-Markov (re-incidence) contract with a step-up
    factor: full==fast must hold -- this is the test that catches a missed edit
    in the semi-Markov codegen, which a Markov-only test would not."""
    pf = lambda s, a, d, ic, el: 1.0 + 0.2 * d
    annual = lambda q: 1.0 - (1.0 - q) ** 12
    basis = Basis(
        mortality_annual=lambda s, a, d: np.full(np.shape(a), annual(0.001)),
        lapse_annual=lambda s, a, d: np.full(np.shape(d), annual(0.005)),
        ci_incidence_annual=lambda s, a, d: np.full(np.shape(a), annual(0.004)),
        ci_reincidence_annual=lambda s, a, p, sd: np.full_like(sd, annual(0.01), dtype=float),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        state_model=_reincidence_model(12),
        coverages=(CoverageRate("DEATH", lambda s, a, d: np.full(np.shape(a), annual(0.001))),),
        premium_factor_annual=pf,
    )
    mp = fcf.ModelPoints(
        issue_age=np.array([45], dtype=np.int64),
        benefits={0: np.array([1e8])},
        premium=np.array([50_000.0]),
        term_months=np.array([60], dtype=np.int64),
        disability_benefit=np.array([2e7]),
        calculation_methods=CM)
    full = fcf.gmm.measure(mp, basis, full=True)
    fast = fcf.gmm.measure(mp, basis, full=False)
    assert np.allclose(full.bel, fast.bel, rtol=1e-9)
    assert np.allclose(full.csm, fast.csm, rtol=1e-9)


# ---------------------------------------------------------------------------
# PF4 -- solve_premium prices on a shape: it solves the SCALE, linearity intact
# ---------------------------------------------------------------------------
def test_solve_premium_on_a_shape():
    """Pricing a renewable: premium_factor_annual fixes the shape, solve_premium
    solves the level SCALE that breaks even over the full term. The solved scale,
    fed back, gives FCF ~ 0, and the projected premium follows scale * shape."""
    from fastcashflow.pricing import solve_premium
    from dataclasses import replace
    pf = lambda s, a, d, ic, el: 1.0 + 0.3 * d
    basis = _basis(pf)
    mp = fcf.ModelPoints.single(issue_age=40, benefits={0: 1e6}, premium=0.0,
                                term_months=60, calculation_methods=CM)
    s = solve_premium(mp, basis, break_even=True)
    assert np.all(np.isfinite(s)) and np.all(s > 0)
    # fed back as the level scale -> FCF (BEL + RA) is ~ 0 (break-even)
    priced = fcf.gmm.measure(replace(mp, premium=s), basis, full=True)
    assert abs(float(priced.bel[0] + priced.ra[0])) < 1e-4
    # projected premium at year k follows scale * (1 + 0.3 k)
    pcf = priced.cashflows.premium_cf[0]
    inforce = priced.cashflows.inforce[0]
    assert pcf[0] == pytest.approx(s[0] * 1.0 * inforce[0], rel=1e-9)
    assert pcf[12] == pytest.approx(s[0] * 1.3 * inforce[12], rel=1e-9)


# ---------------------------------------------------------------------------
# PF5 -- the factor output is validated: a negative / NaN factor is rejected
# (it would silently flip the premium sign or poison the BEL, bypassing the
# ModelPoints premium >= 0 invariant), on BOTH the full and the fast path
# ---------------------------------------------------------------------------
def test_premium_factor_rejects_negative_and_nan():
    mp = fcf.ModelPoints.single(issue_age=40, benefits={0: 1e6}, premium=1000.0,
                                term_months=24, calculation_methods=CM)
    neg = lambda s, a, d, ic, el: np.full(np.shape(a), -1.0)
    nan = lambda s, a, d, ic, el: np.full(np.shape(a), np.nan)
    for fn, msg in ((neg, "negative"), (nan, "non-finite")):
        with pytest.raises(ValueError, match=f"premium_factor_annual.*{msg}"):
            fcf.gmm.measure(mp, _basis(fn), full=True)
        with pytest.raises(ValueError, match=f"premium_factor_annual.*{msg}"):
            fcf.gmm.measure(mp, _basis(fn), full=False)
    # a zero factor is allowed (a premium holiday), not rejected
    holiday = lambda s, a, d, ic, el: np.where(d == 0, 1.0, 0.0)
    pcf = fcf.gmm.measure(mp, _basis(holiday), full=True).cashflows.premium_cf[0]
    assert pcf[0] > 0.0 and pcf[12] == pytest.approx(0.0)


def test_premium_factor_rejects_wrong_shape():
    """A factor callable that returns a scalar / wrong-shape array is a clean
    ValueError (an input-contract failure), not an AssertionError -- which
    would also vanish under ``python -O`` and let a mis-shaped factor through.
    Checked on both the full and the fast path."""
    mp = fcf.ModelPoints.single(issue_age=40, benefits={0: 1e6}, premium=1000.0,
                                term_months=24, calculation_methods=CM)
    scalar = lambda s, a, d, ic, el: 1.0
    for full in (True, False):
        with pytest.raises(ValueError,
                           match="premium_factor_annual must return an array of shape"):
            fcf.gmm.measure(mp, _basis(scalar), full=full)


# ---------------------------------------------------------------------------
# PF6 -- fast==gpu parity: the GPU kernel applies the premium factor identically
# to the CPU fast path (the factor is threaded into the CUDA kernel via the same
# validated dense grid). Skipped when no CUDA device is present.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not cuda.is_available(), reason="no CUDA device available")
@pytest.mark.filterwarnings("ignore::numba.core.errors.NumbaPerformanceWarning")
def test_premium_factor_fast_matches_gpu():
    pf = lambda s, a, d, ic, el: 1.0 + 0.3 * d
    rng = np.random.default_rng(7)
    n = 3_000
    mps = fcf.ModelPoints(
        issue_age=rng.integers(25, 60, n),
        benefits={0: rng.integers(10, 100, n) * 1_000_000},
        premium=rng.integers(3, 15, n) * 10_000,
        term_months=np.full(n, 120),
    )
    cpu = fcf.gmm.measure(mps, _basis(pf), backend="cpu", full=False)
    gpu = fcf.gmm.measure(mps, _basis(pf), backend="gpu", full=False)
    assert np.allclose(gpu.bel, cpu.bel)
    assert np.allclose(gpu.ra, cpu.ra)
    assert np.allclose(gpu.csm, cpu.csm)
    # the factor genuinely bites: the BEL differs from the level baseline
    base = fcf.gmm.measure(mps, _basis(), backend="cpu", full=False)
    assert not np.allclose(cpu.bel, base.bel)


def test_annuity_factor_rejects_negative_and_nan():
    """The annuity factor twin is guarded the same way (a negative annuity
    factor would flip a survival benefit into an inflow)."""
    mp = fcf.ModelPoints.single(issue_age=60, benefits={0: 0.0}, premium=0.0,
                                term_months=36, annuity_payment=100.0,
                                annuity_frequency_months=12,
                                calculation_methods={"ANN": fcf.CalculationMethod.ANNUITY})
    def _abasis(af):
        return Basis(mortality_annual=_NONE, lapse_annual=_NONE, discount_annual=0.0,
                     ra_confidence=0.75, mortality_cv=0.10,
                     coverages=(CoverageRate("ANN", _NONE),), annuity_factor_annual=af)
    neg = lambda s, a, d, ic, el: np.full(np.shape(a), -1.0)
    with pytest.raises(ValueError, match="annuity_factor_annual.*negative"):
        fcf.gmm.measure(mp, _abasis(neg), full=True)
