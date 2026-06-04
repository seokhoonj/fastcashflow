"""Hand-calc anchors for amount-based surrender value (해약환급금) -- S1.

The current engine post-computes ``surrender_cf = lapse_flow x cum_premium
x factor`` -- a factor on cumulative premium, which is path-dependent on
pre-valuation premiums and so cannot be re-based exactly for an in-force
book. S1 reframes the surrender value as a contractual amount by policy
duration::

    surrender_value_basis = "amount_per_policy"
    surrender_cf[t] = lapse_flow[t] * surrender_value_curve[t]
                    = inforce[t] * lapse_monthly * amount[t]

The curve holds the per-policy surrender amount at absolute policy-duration
``t`` (months since inception). Being linear in the in-force, the in-force
``count / inforce[elapsed]`` rescale (High1) makes the settlement figure
exact, and the sample-grade ``UserWarning`` no longer applies.

These anchors are written before the engine branch exists: they pin the
target behaviour and currently FAIL (the engine still applies the
cum_premium formula and warns regardless of the basis).
"""
import warnings

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, CalculationMethod, ModelPoints, CoverageRate
from fastcashflow.engine import _measure_inforce_fast
from fastcashflow.gmm import measure
from conftest import PATTERNS


def _flat_rate(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _amount_basis(lapse_rate, amount_curve, discount=0.0):
    """A surrender-only basis: no mortality, no claim, amount-mode surrender.

    Isolating surrender makes the BEL the surrender PV alone, so the
    hand calculation has a single moving part.
    """
    return Basis(
        mortality_annual=_flat_rate(0.0),
        lapse_annual=_flat_rate(lapse_rate),
        discount_annual=discount,
        ra_confidence=0.75,
        mortality_cv=0.0,
        surrender_value_curve=amount_curve,
        surrender_value_basis="amount_per_policy",
        coverages=(CoverageRate("DEATH", _flat_rate(0.0)),),
    )


def test_amount_per_policy_per_month_hand_calc():
    """``surrender_cf[t] = inforce[t] * lapse_monthly * amount[t]`` -- the base
    is the contractual amount at duration t, not cumulative premium. A
    distinct amount per duration catches a wrong (off-by-month) index."""
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        premium=10_000.0, term_months=12,
        calculation_methods=PATTERNS,
    )
    lapse_annual = 0.12
    amount = np.array([1_000.0 * (t + 1) for t in range(13)])
    m = measure(mp, _amount_basis(lapse_annual, amount))

    lapse_monthly = 1.0 - (1.0 - lapse_annual) ** (1.0 / 12.0)
    inforce = m.cashflows.inforce[0]
    expected = inforce[:12] * lapse_monthly * amount[:12]
    assert np.allclose(m.cashflows.surrender_cf[0, :12], expected)


def test_amount_per_policy_independent_of_premium():
    """The amount-mode surrender value does not scale with premium -- the
    sharp discriminator against the cum_premium base, which does."""
    amount = np.full(13, 5_000.0)

    def surr(premium):
        mp = ModelPoints.single(
            issue_age=40, benefits={0: 100_000_000.0},
            premium=premium, term_months=12,
            calculation_methods=PATTERNS,
        )
        return measure(mp, _amount_basis(0.1, amount)).cashflows.surrender_cf[0]

    assert np.allclose(surr(10_000.0), surr(20_000.0))


def test_inforce_amount_surrender_rescale_is_exact():
    """In amount mode the in-force surrender is exact under the existing
    ``count / inforce[E]`` rescale. surrender_cf is linear in the in-force,
    so the as-of figure equals the ground truth

        c * sum_{u>=E} survival(E->u) * lapse_monthly * amount[u]

    computed independently from the engine's own survival curve (inforce of
    the fresh count=1 projection). premium = 0 so the BEL is surrender only;
    discount = 0 so the PV is the plain (mid-month factor 1) sum."""
    lapse_annual = 0.10
    amount = np.array([100.0 * (t + 1) for t in range(60)])
    basis = _amount_basis(lapse_annual, amount, discount=0.0)
    CM = {"DEATH": CalculationMethod.DEATH}

    # fresh, count = 1, from inception -> engine survival = inforce_fresh
    fresh = ModelPoints(
        issue_age=np.array([40]), premium=np.array([0.0]),
        term_months=np.array([48]), benefits={0: np.array([1e6])},
        count=np.array([1.0]), calculation_methods=CM,
    )
    m = measure(fresh, basis)
    inforce_fresh = m.cashflows.inforce[0]

    E, c = 12, 0.8
    inforce_mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([0.0]),
        term_months=np.array([48]), benefits={0: np.array([1e6])},
        count=np.array([c]), elapsed_months=np.array([E]),
        calculation_methods=CM,
    )
    bel_inforce = _measure_inforce_fast(inforce_mp, basis).bel[0]

    lapse_monthly = 1.0 - (1.0 - lapse_annual) ** (1.0 / 12.0)
    u = np.arange(E, inforce_fresh.shape[0])
    surv_E_to_u = inforce_fresh[u] / inforce_fresh[E]
    expected = c * np.sum(surv_E_to_u * lapse_monthly * amount[u])
    assert np.isclose(bel_inforce, expected, rtol=1e-9)


def test_amount_per_unit_raises_until_wired():
    """amount_per_unit needs a per-MP base amount that is not wired yet; it
    raises on both paths rather than silently dropping the base."""
    amount = np.full(13, 5_000.0)
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 100_000_000.0},
        premium=10_000.0, term_months=12,
        calculation_methods=PATTERNS,
    )
    basis = Basis(
        mortality_annual=_flat_rate(0.0),
        lapse_annual=_flat_rate(0.1),
        discount_annual=0.0,
        ra_confidence=0.75,
        mortality_cv=0.0,
        surrender_value_curve=amount,
        surrender_value_basis="amount_per_unit",
        coverages=(CoverageRate("DEATH", _flat_rate(0.0)),),
    )
    with pytest.raises(NotImplementedError, match="amount_per_unit"):
        measure(mp, basis, full=False)
    with pytest.raises(NotImplementedError, match="amount_per_unit"):
        measure(mp, basis, full=True)


def test_inforce_amount_emits_no_surrender_warning():
    """With an amount-mode surrender curve the in-force figure is exact, so
    measure_inforce no longer emits the sample-grade surrender warning."""
    CM = {"DEATH": CalculationMethod.DEATH}
    amount = np.full(60, 1_000.0)
    basis = _amount_basis(0.1, amount)
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([0.0]),
        term_months=np.array([48]), benefits={0: np.array([1e6])},
        count=np.array([1.0]), elapsed_months=np.array([12]),
        calculation_methods=CM,
    )
    state = fcf.InforceState(
        mp_id=np.array(["A"]),
        elapsed_months=np.array([12], dtype=np.int64),
        count=np.array([1.0]),
        prior_csm=np.array([0.0]),
        lock_in_rate=0.0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fcf.gmm.measure_inforce(mp, basis, state, full=False)
    assert not [w for w in caught if "surrender" in str(w.message).lower()]
