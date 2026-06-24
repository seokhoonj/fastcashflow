"""Hand-calc anchors for amount-based surrender value -- S1.

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
from fastcashflow.gmm._engine import _measure_inforce_fast
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


@pytest.mark.parametrize("em", [0, 12])
def test_inforce_surrender_value_amount_per_policy(em):
    """``inforce_surrender_value = count * amount[em]``: the per-policy surrender
    amount at the valuation duration, times the as-of count. The inception-run
    survival to ``em`` cancels against the count rebase, leaving exactly
    ``count * amount[em]`` for the amount-per-policy basis."""
    from dataclasses import replace
    from fastcashflow._measurement.inforce import inforce_surrender_value
    amount = np.array([1_000.0 * (t + 1) for t in range(61)])
    basis = _amount_basis(0.12, amount)
    mp = ModelPoints.single(issue_age=40, benefits={"DEATH": 1e8},
                            premium=10_000.0, term_months=60, count=1_000.0,
                            calculation_methods=PATTERNS)
    mp = replace(mp, elapsed_months=np.array([em], dtype=np.int64))
    sv = inforce_surrender_value(mp, basis)
    assert np.isclose(sv[0], 1_000.0 * amount[em])


def _ul_surrender_basis(surr_charge, **overrides):
    """A minimal universal-life basis -- account-backed DEATH coverage, no COI /
    expenses / lapse, flat surrender charge. With zero credit and no charges the
    account value stays at ``av0``, so ``av_mid == av0`` and the surrender value
    is a clean ``av0 * (1 - surr_charge)``."""
    kw = dict(
        mortality_annual=_flat_rate(0.0),
        lapse_annual=_flat_rate(0.0),
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.0,
        investment_return=0.0,                 # r_m = 0 -> credit = 0
        surrender_charge_annual=(lambda s, a, d, ic, el:
                                 np.full(s.shape, surr_charge, dtype=np.float64)),
        coverages=(CoverageRate("DEATH", _flat_rate(0.0), funds_from_account=True,
                                pays_account_balance=True),),
    )
    kw.update(overrides)
    return Basis(**kw)


@pytest.mark.parametrize("em", [0, 12, 24])
def test_inforce_surrender_value_account_backed_hand_calc(em):
    """Universal-life surrender value = ``count * av0 * (1 - surr_charge)``.

    With zero credit / COI / expenses the account stays flat at ``av0``, so the
    per-policy surrender value is ``av0 * (1 - surr_charge)`` at every duration,
    paid to the full as-of count. Independent of the elapsed months (flat av)."""
    from dataclasses import replace
    from fastcashflow._measurement.inforce import inforce_surrender_value
    av0, surr_charge, count = 1_000.0, 0.10, 500.0
    basis = _ul_surrender_basis(surr_charge)
    mp = ModelPoints.single(issue_age=40, benefits={"DEATH": 1e6},
                            premium=0.0, term_months=60, count=count,
                            account_value=av0, minimum_death_benefit=1e6,
                            calculation_methods={"DEATH": CalculationMethod.DEATH})
    mp = replace(mp, elapsed_months=np.array([em], dtype=np.int64))
    sv = inforce_surrender_value(mp, basis)
    assert np.isclose(sv[0], count * av0 * (1.0 - surr_charge))


def test_inforce_surrender_value_account_backed_matches_av_mid():
    """The UL surrender value reads the account roll's own ``av_mid`` net of the
    charge -- cross-check against the measured trajectory for a credited account
    (non-flat av), so the wiring is exact, not just the flat-av hand calc."""
    from dataclasses import replace
    from fastcashflow._measurement.inforce import inforce_surrender_value
    av0, surr_charge, count, em = 1_000.0, 0.08, 300.0, 18
    basis = _ul_surrender_basis(surr_charge, investment_return=0.05)  # credited -> av grows
    mp = ModelPoints.single(issue_age=40, benefits={"DEATH": 1e6},
                            premium=0.0, term_months=60, count=count,
                            account_value=av0, minimum_death_benefit=1e6,
                            calculation_methods={"DEATH": CalculationMethod.DEATH})
    mp = replace(mp, elapsed_months=np.array([em], dtype=np.int64))
    m = fcf.vfa.measure(mp, basis)
    av_mid_em = m.cashflows.account.av_mid[0, em]
    sv = inforce_surrender_value(mp, basis)
    assert av_mid_em > av0                                    # the account did grow
    assert np.isclose(sv[0], count * av_mid_em * (1.0 - surr_charge))


def test_inforce_surrender_value_zero_without_curve():
    """No ``surrender_value_curve`` -> zero (lapse removes the contract with no
    payment); the helper short-circuits before projecting."""
    from fastcashflow._measurement.inforce import inforce_surrender_value
    basis = Basis(
        mortality_annual=_flat_rate(0.001), lapse_annual=_flat_rate(0.01),
        discount_annual=0.0, ra_confidence=0.75, mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", _flat_rate(0.001)),),
    )
    mp = ModelPoints.single(issue_age=40, benefits={"DEATH": 1e8},
                            premium=10_000.0, term_months=60, count=1_000.0,
                            calculation_methods=PATTERNS)
    assert np.allclose(inforce_surrender_value(mp, basis), 0.0)


def test_amount_per_policy_per_month_hand_calc():
    """``surrender_cf[t] = inforce[t] * lapse_monthly * amount[t]`` -- the base
    is the contractual amount at duration t, not cumulative premium. A
    distinct amount per duration catches a wrong (off-by-month) index."""
    mp = ModelPoints.single(
        issue_age=40, benefits={"DEATH": 100_000_000.0},
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
            issue_age=40, benefits={"DEATH": 100_000_000.0},
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
        term_months=np.array([48]), benefits={"DEATH": np.array([1e6])},
        count=np.array([1.0]), calculation_methods=CM,
    )
    m = measure(fresh, basis)
    inforce_fresh = m.cashflows.inforce[0]

    E, c = 12, 0.8
    inforce_mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([0.0]),
        term_months=np.array([48]), benefits={"DEATH": np.array([1e6])},
        count=np.array([c]), elapsed_months=np.array([E]),
        calculation_methods=CM,
    )
    bel_inforce = _measure_inforce_fast(inforce_mp, basis).bel[0]

    lapse_monthly = 1.0 - (1.0 - lapse_annual) ** (1.0 / 12.0)
    u = np.arange(E, inforce_fresh.shape[0])
    surv_E_to_u = inforce_fresh[u] / inforce_fresh[E]
    expected = c * np.sum(surv_E_to_u * lapse_monthly * amount[u])
    assert np.isclose(bel_inforce, expected, rtol=1e-9)


def _per_unit_basis(lapse_rate, amount_curve, discount=0.0):
    """A surrender-only basis in amount_per_unit mode."""
    return Basis(
        mortality_annual=_flat_rate(0.0),
        lapse_annual=_flat_rate(lapse_rate),
        discount_annual=discount,
        ra_confidence=0.75,
        mortality_cv=0.0,
        surrender_value_curve=amount_curve,
        surrender_value_basis="amount_per_unit",
        coverages=(CoverageRate("DEATH", _flat_rate(0.0)),),
    )


def test_amount_per_unit_scales_by_base():
    """amount_per_unit: surrender_cf[t] = inforce[t] * lapse_monthly *
    amount[t] * surrender_base_amount[mp]. Two otherwise-identical MPs with
    bases in a 1:2 ratio surrender in a 1:2 ratio; MP 0 is hand-checked."""
    amount = np.full(13, 100.0)
    mp = ModelPoints(
        issue_age=np.array([40, 40]),
        premium=np.array([0.0, 0.0]),
        term_months=np.array([12, 12]),
        benefits={"DEATH": np.array([1e6, 1e6])},
        count=np.array([1.0, 1.0]),
        surrender_base_amount=np.array([1_000.0, 2_000.0]),
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    m = measure(mp, _per_unit_basis(0.1, amount))
    s = m.cashflows.surrender_cf
    lapse_monthly = 1.0 - (1.0 - 0.1) ** (1.0 / 12.0)
    inforce0 = m.cashflows.inforce[0]
    assert np.allclose(s[0, :12],
                       inforce0[:12] * lapse_monthly * amount[:12] * 1_000.0)
    assert np.allclose(s[1], 2.0 * s[0])


def test_amount_per_unit_requires_base():
    """amount_per_unit without surrender_base_amount raises -- no default
    base is inferred (it differs by product)."""
    amount = np.full(13, 100.0)
    mp = ModelPoints.single(
        issue_age=40, benefits={"DEATH": 100_000_000.0},
        premium=10_000.0, term_months=12,
        calculation_methods=PATTERNS,
    )
    with pytest.raises(ValueError, match="surrender_base_amount"):
        measure(mp, _per_unit_basis(0.1, amount))


def test_inforce_amount_emits_no_surrender_warning():
    """With an amount-mode surrender curve the in-force figure is exact, so
    measure_inforce no longer emits the sample-grade surrender warning."""
    CM = {"DEATH": CalculationMethod.DEATH}
    amount = np.full(60, 1_000.0)
    basis = _amount_basis(0.1, amount)
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([0.0]),
        term_months=np.array([48]), benefits={"DEATH": np.array([1e6])},
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
        fcf.gmm.measure_inforce(mp, state, basis, full=False)
    assert not [w for w in caught if "surrender" in str(w.message).lower()]
