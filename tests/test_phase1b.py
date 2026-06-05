"""Phase 1b validation -- select-and-ultimate mortality and duration-based lapse.

Both basis step at the first policy-year boundary, so a 24-month case
exercises the select period, the ultimate period and the duration switch in
both mortality and lapse. The in-force recursion is recomputed independently
in plain Python as the correctness anchor.
"""
import numpy as np

from fastcashflow import ModelPoints
from fastcashflow.gmm import measure
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


SELECT_Q = 0.005      # monthly mortality, policy year 0 (select)
ULT_Q = 0.02          # monthly mortality, policy year 1+ (ultimate)
SELECT_LAPSE = 0.03   # monthly lapse, policy year 0
ULT_LAPSE = 0.01      # monthly lapse, policy year 1+


def _mortality(sex, issue_age, duration):
    """Select for the first policy year, ultimate thereafter."""
    return np.where(duration < 1, _annual(SELECT_Q), _annual(ULT_Q))


def _lapse(sex, issue_age, duration):
    """Higher lapse in the first policy year, lower thereafter."""
    return np.where(duration < 1, _annual(SELECT_LAPSE), _annual(ULT_LAPSE))


def _assumptions(**overrides):
    kw = dict(
        mortality_annual = _mortality,
        lapse_annual     = _lapse,
        discount_annual  = 0.0,
        ra_confidence    = 0.75,
        mortality_cv     = 0.0,
    )
    kw.update(overrides)
    return make_death_basis(**kw)


def test_select_ultimate_and_duration_lapse():
    """Mortality and lapse both step at the policy-year boundary -- hand-checked."""
    death_benefit = 1_000_000.0
    premium = 10_000.0
    term = 24

    res = measure(
        ModelPoints.single(
            issue_age=40, benefits={0: death_benefit},
            premium=premium, term_months=term,
            calculation_methods=PATTERNS,
        ),
        _assumptions(),
    )

    # Independent recomputation of the in-force / death recursion.
    inforce = np.empty(term)
    deaths = np.empty(term)
    inforce[0] = 1.0
    for t in range(term):
        q = SELECT_Q if t < 12 else ULT_Q
        lapse = SELECT_LAPSE if t < 12 else ULT_LAPSE
        deaths[t] = inforce[t] * q
        if t + 1 < term:
            inforce[t + 1] = inforce[t] * (1.0 - q) * (1.0 - lapse)

    assert np.allclose(res.cashflows.inforce[0], inforce)
    assert np.allclose(res.cashflows.deaths[0], deaths)

    # The select -> ultimate step is visible at the year boundary.
    select_factor = (1.0 - SELECT_Q) * (1.0 - SELECT_LAPSE)
    assert np.isclose(res.cashflows.inforce[0, 12], select_factor ** 12)
    assert np.isclose(
        res.cashflows.inforce[0, 13],
        select_factor ** 12 * (1.0 - ULT_Q) * (1.0 - ULT_LAPSE),
    )
    # mortality switches from select to ultimate at month 12
    assert np.isclose(
        res.cashflows.deaths[0, 11] / res.cashflows.inforce[0, 11], SELECT_Q
    )
    assert np.isclose(
        res.cashflows.deaths[0, 12] / res.cashflows.inforce[0, 12], ULT_Q
    )

    # BEL = PV(claims) - PV(premiums); zero discount, zero expenses
    pv_claims = death_benefit * deaths.sum()
    pv_premiums = premium * inforce.sum()
    assert np.isclose(res.bel_path[0, 0], pv_claims - pv_premiums)


def test_value_matches_run_phase1b():
    """The fast path reproduces measure() under duration-varying basis."""
    rng = np.random.default_rng(11)
    n = 500
    mps = ModelPoints(
        issue_age=rng.integers(25, 55, n),
        benefits={0: rng.integers(10, 100, n) * 1_000_000},
        premium=rng.integers(3, 15, n) * 10_000,
        term_months=rng.integers(13, 36, n),
        calculation_methods=PATTERNS,
    )
    basis = _assumptions(mortality_cv=0.10, discount_annual=0.03)

    fast = measure(mps, basis, full=False)
    detailed = measure(mps, basis)

    assert np.allclose(fast.bel, detailed.bel_path[:, 0])
    assert np.allclose(fast.ra, detailed.ra_path[:, 0])
    assert np.allclose(fast.csm, detailed.csm_path[:, 0])
    assert np.allclose(fast.loss_component, detailed.loss_component)
