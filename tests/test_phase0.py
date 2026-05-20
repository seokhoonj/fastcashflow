"""Core engine validation -- output vs an independent hand calculation.

The main test uses a deliberately tiny case (1 policy, 2-month term, flat
rates, zero discount, zero expenses) so every figure can be derived by hand.
This is the engine's correctness anchor.
"""
import numpy as np

from fastcashflow import Assumptions, ModelPointSet, measure

# Standard-normal 75th percentile -- a known mathematical constant, used so
# the RA check does not depend on the engine's own quantile code.
Z_75 = 0.6744897501960817


def _assumptions(**overrides) -> Assumptions:
    """Build an Assumptions with simple defaults, overridable per test."""
    base = dict(
        mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.01),
        lapse_monthly=lambda duration: np.full(duration.shape, 0.02),
        discount_annual=0.0,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.75,
        claims_cv=0.10,
    )
    base.update(overrides)
    return Assumptions(**base)


def test_hand_calculation():
    death_benefit = 1_000_000.0
    premium = 12_000.0
    term = 2

    res = measure(
        ModelPointSet.single(
            issue_age=40, death_benefit=death_benefit,
            monthly_premium=premium, term_months=term,
        ),
        _assumptions(),
    )

    # in force: inforce[0] = 1.0 ; inforce[1] = 1 * (1-0.01) * (1-0.02) = 0.9702
    inforce = [1.0, 0.99 * 0.98]
    assert np.isclose(res.cashflows.inforce[0, 1], inforce[1])

    # cash flows (discount factors are all 1 -- zero discount)
    deaths = [inforce[0] * 0.01, inforce[1] * 0.01]
    premium_cf = [inforce[0] * premium, inforce[1] * premium]
    claim_cf = [deaths[0] * death_benefit, deaths[1] * death_benefit]
    pv_claims = sum(claim_cf)        # 10000 + 9702 = 19702
    pv_premiums = sum(premium_cf)    # 12000 + 11642.4 = 23642.4

    # BEL = PV(claims) + PV(expenses) - PV(premiums); expenses = 0 here
    bel = pv_claims - pv_premiums
    assert np.isclose(res.bel[0, 0], bel)
    assert np.isclose(res.bel[0, 0], -3940.4)

    # RA = z(0.75) * claims_cv * PV(claims)
    ra = Z_75 * 0.10 * pv_claims
    assert np.isclose(res.ra[0, 0], ra)

    # FCF = BEL + RA ; CSM_0 = max(0, -FCF)
    fcf = bel + ra
    assert np.isclose(res.csm[0, 0], max(0.0, -fcf))
    assert np.isclose(res.loss_component[0], max(0.0, fcf))

    # CSM roll-forward (zero discount, coverage units = in force):
    #   t=1: release = CSM_0 * cu[0] / (cu[0] + cu[1]) ; CSM[1] = CSM_0 - release
    release0 = res.csm[0, 0] * inforce[0] / (inforce[0] + inforce[1])
    assert np.isclose(res.csm[0, 1], res.csm[0, 0] - release0)
    #   t=2: the remaining CSM is fully released
    assert np.isclose(res.csm[0, 2], 0.0)


def test_onerous_contract():
    """Premium far too low -> onerous -> CSM floored at 0, loss component > 0."""
    res = measure(
        ModelPointSet.single(
            issue_age=40, death_benefit=1_000_000.0,
            monthly_premium=100.0, term_months=12,
        ),
        _assumptions(
            mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.05)
        ),
    )
    assert res.csm[0, 0] == 0.0
    assert res.loss_component[0] > 0.0


def test_csm_fully_releases():
    """A profitable contract's CSM must run off to ~0 by the end of term."""
    res = measure(
        ModelPointSet.single(
            issue_age=35, death_benefit=50_000_000.0,
            monthly_premium=80_000.0, term_months=60,
        ),
        _assumptions(
            mortality_monthly=lambda issue_age, duration: np.full(issue_age.shape, 0.001),
            lapse_monthly=lambda duration: np.full(duration.shape, 0.01),
            discount_annual=0.03,
        ),
    )
    assert res.csm[0, 0] > 0.0
    assert np.isclose(res.csm[0, -1], 0.0, atol=1e-6)
