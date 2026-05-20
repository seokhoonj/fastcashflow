"""Phase 0 validation -- engine output vs an independent hand calculation.

The main test uses a deliberately tiny case (1 policy, 2-month term, flat
rates, zero discount) so every figure can be derived by hand and checked
inline. This is the Phase 0 correctness anchor before any scale-up.
"""
import numpy as np

from fastcashflow import Assumptions, ModelPointSet, run


def test_phase0_hand_calculation():
    # --- Inputs -----------------------------------------------------------
    q_m = 0.01            # flat monthly mortality
    w_m = 0.02            # flat monthly lapse
    sum_assured = 1_000_000.0
    premium = 12_000.0
    term = 2
    ra_rate = 0.05

    asmp = Assumptions(
        mortality_monthly=lambda ages: np.full(ages.shape, q_m),
        lapse_monthly=w_m,
        discount_annual=0.0,          # zero discount -> all discount factors = 1
        ra_rate=ra_rate,
    )
    mps = ModelPointSet.single(
        issue_age=40,
        sum_assured=sum_assured,
        monthly_premium=premium,
        term_months=term,
    )
    res = run(mps, asmp)

    # --- Hand calculation -------------------------------------------------
    # in force: inforce[0] = 1.0
    #           inforce[1] = 1 * (1 - 0.01) * (1 - 0.02) = 0.9702
    inforce = [1.0, 0.99 * 0.98]
    assert np.isclose(res.projection.inforce[0, 1], inforce[1])

    # deaths[t] = inforce[t] * q_m
    deaths = [inforce[0] * q_m, inforce[1] * q_m]
    premium_cf = [inforce[0] * premium, inforce[1] * premium]
    claim_cf = [deaths[0] * sum_assured, deaths[1] * sum_assured]

    # discount factors are all 1 (zero discount)
    # BEL = sum(claim_cf) - sum(premium_cf)
    #     = (10000 + 9702) - (12000 + 11642.4) = 19702 - 23642.4 = -3940.4
    bel = sum(claim_cf) - sum(premium_cf)
    assert np.isclose(res.bel[0], bel)
    assert np.isclose(res.bel[0], -3940.4)

    # RA = ra_rate * PV(claims) = 0.05 * 19702 = 985.1
    ra = ra_rate * sum(claim_cf)
    assert np.isclose(res.ra[0], ra)
    assert np.isclose(res.ra[0], 985.1)

    # FCF = BEL + RA = -2955.3 ; CSM_0 = max(0, -FCF) = 2955.3
    fcf = bel + ra
    assert np.isclose(res.csm0[0], max(0.0, -fcf))
    assert np.isclose(res.csm0[0], 2955.3)
    assert np.isclose(res.loss_component[0], 0.0)

    # CSM roll-forward (zero discount, coverage units = in force):
    #   t=1: accreted = 2955.3
    #        cu_remaining = 1.0 + 0.9702 = 1.9702
    #        release    = 2955.3 * 1.0 / 1.9702 = 1500.0
    #        CSM[1]     = 2955.3 - 1500.0 = 1455.3
    cu_remaining = inforce[0] + inforce[1]
    release0 = res.csm0[0] * inforce[0] / cu_remaining
    assert np.isclose(release0, 1500.0)
    assert np.isclose(res.csm[0, 1], res.csm0[0] - release0)
    assert np.isclose(res.csm[0, 1], 1455.3)

    #   t=2: the whole remaining CSM is released -> CSM[2] = 0
    assert np.isclose(res.csm[0, 2], 0.0)
    assert np.isclose(res.csm[0, term], 0.0)


def test_phase0_onerous_contract():
    """Premium far too low -> onerous -> CSM is floored at 0, loss component > 0."""
    asmp = Assumptions(
        mortality_monthly=lambda ages: np.full(ages.shape, 0.05),
        lapse_monthly=0.0,
        discount_annual=0.0,
        ra_rate=0.05,
    )
    mps = ModelPointSet.single(
        issue_age=40,
        sum_assured=1_000_000.0,
        monthly_premium=100.0,        # absurdly low
        term_months=12,
    )
    res = run(mps, asmp)

    assert res.csm0[0] == 0.0
    assert res.loss_component[0] > 0.0


def test_phase0_csm_fully_releases():
    """For any profitable contract the CSM must run off to ~0 by end of term."""
    asmp = Assumptions(
        mortality_monthly=lambda ages: np.full(ages.shape, 0.001),
        lapse_monthly=0.01,
        discount_annual=0.03,
        ra_rate=0.05,
    )
    # Expected monthly claim cost per policy = 0.001 * 50,000,000 = 50,000.
    # The premium must clear that (plus margin) for the contract to be profitable.
    mps = ModelPointSet.single(
        issue_age=35,
        sum_assured=50_000_000.0,
        monthly_premium=80_000.0,
        term_months=60,
    )
    res = run(mps, asmp)

    assert res.csm0[0] > 0.0
    assert np.isclose(res.csm[0, -1], 0.0, atol=1e-6)
