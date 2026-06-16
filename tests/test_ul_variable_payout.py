"""Universal-life variable (실적배당) annuity payout -- hand-calc anchor.

A variable-payout annuity re-floats the phase-2 income each elapsed month by the
ratio of the realised fund return to an assumed interest rate (AIR, 예정이율):

    payment_k = locked_annuity_payment * ((1 + fund) / (1 + air))^k,  k = t - A

This is the annuity-unit method: the unit count is fixed at conversion, the unit
value floats with the fund. Under VFA (discount = fund return) the fund cancels,
so BEL(phase 2) = the initial payment valued as a FIXED annuity at the AIR -- the
investment risk passes through to the policyholder, only longevity stays with the
insurer. A variable payout is a direct-participation feature: it is measured
through ``vfa.measure`` and rejected on ``gmm.measure``.

The payout is turned on per model point by a finite ``annuity_air_annual``; NaN
(the default) keeps the fixed GAO payout. The crediting floor is set to
NO_GUARANTEE here so the credited rate equals the fund return and the
cancellation is exact (the floor's value is a separate, deferred guarantee).
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
from fastcashflow.model_points import NO_GUARANTEE_RATE


def _basis(**overrides):
    """A minimal UL-annuity basis -- account-backed DEATH coverage (COI 0)."""
    kw = dict(
        mortality_annual=0.0,
        lapse_annual=0.0,
        discount_annual=0.0,
        ra_confidence=0.75,
        mortality_cv=0.1,
        longevity_cv=0.0,
        investment_return=0.0,
        coi_annual=0.0,
    )
    kw.update(overrides)
    kw["coverages"] = (
        CoverageRate("DEATH", 0.0, funds_from_account=True,
                     pays_account_balance=True),
    )
    return Basis(**kw)


def _mp(face, **fields):
    face_arr = np.atleast_1d(np.asarray(face, dtype=float))
    fields["minimum_death_benefit"] = face_arr
    fields.setdefault("benefits", {"DEATH": face_arr})
    fields.setdefault("calculation_methods", {"DEATH": CalculationMethod.DEATH})
    return ModelPoints(**fields)


def _annuity_mp(av0, rate, air_annual, inv_return_fields):
    """A single-premium account that converts at month 1, term 4 -> payout months
    1, 2, 3 (three annuity-due payments), no mortality / lapse / COI / load."""
    return _mp(
        10_000_000.0,                         # face registers the coverage; COI 0
        issue_age=np.array([60.0]),
        premium=np.array([0.0]),
        term_months=np.array([4]),
        premium_term_months=np.array([0]),    # single-premium
        account_value=np.array([av0]),
        annuitization_months=np.array([1]),
        annuitization_rate=np.array([rate]),
        annuity_air_annual=np.array([air_annual]),
        minimum_crediting_rate=np.array([NO_GUARANTEE_RATE]),  # credit = fund
        sex=np.array([0]),
        **inv_return_fields,
    )


# ---------------------------------------------------------------------------
# Hand-calc: the payout re-floats by (1+fund)/(1+air); BEL == the AIR-reserve.
# ---------------------------------------------------------------------------

def test_variable_payout_refloats_and_air_reserve():
    fund_m = 0.005
    air_m = 0.002
    inv_return = (1.0 + fund_m) ** 12 - 1.0
    air_annual = (1.0 + air_m) ** 12 - 1.0
    rate = 0.01
    av0 = 1_000_000.0

    mp = _annuity_mp(av0, rate, air_annual, {})
    basis = _basis(investment_return=inv_return)
    m = fcf.vfa.measure(mp, basis, full=True)
    proj = m.cashflows

    # Conversion at month 1: av[1] = av0 * (1+fund) (credit = fund, no COI/admin).
    converted = av0 * (1.0 + fund_m)
    locked = converted * rate
    growth = (1.0 + fund_m) / (1.0 + air_m)
    # Annuity-due months 1, 2, 3; in-force 1 (no decrement); re-floats by growth^k.
    assert np.allclose(
        proj.annuity_cf[0],
        [0.0, locked, locked * growth, locked * growth ** 2])

    # Under VFA the fund cancels: pv(annuity) = locked * (1+fund)^-A *
    # sum_{k=0}^{n-1} (1+air)^-k -- the initial payment as a fixed annuity at the
    # AIR. BEL = pv(annuity) - account_value0 (no premium, no other claims).
    A, n = 1, 3
    air_reserve = locked * (1.0 + fund_m) ** (-A) * sum(
        (1.0 + air_m) ** (-k) for k in range(n))
    assert np.isclose(m.bel[0], air_reserve - av0)


# ---------------------------------------------------------------------------
# air == fund -> growth 1 -> the variable payout is exactly the level GAO.
# ---------------------------------------------------------------------------

def test_variable_payout_level_when_air_equals_fund():
    fund_m = 0.004
    inv_return = (1.0 + fund_m) ** 12 - 1.0
    rate = 0.01
    av0 = 1_000_000.0

    # AIR set equal to the fund return -> (1+fund)/(1+air) = 1 -> level payment.
    mp = _annuity_mp(av0, rate, inv_return, {})
    basis = _basis(investment_return=inv_return)
    m = fcf.vfa.measure(mp, basis, full=True)

    converted = av0 * (1.0 + fund_m)
    locked = converted * rate
    assert np.allclose(m.cashflows.annuity_cf[0], [0.0, locked, locked, locked])


# ---------------------------------------------------------------------------
# A finite air with a higher fund return raises the payout above the level GAO;
# a fixed (NaN) payout stays level under the same fund.
# ---------------------------------------------------------------------------

def test_variable_vs_fixed_payout_differ():
    fund_m = 0.006
    air_m = 0.002
    inv_return = (1.0 + fund_m) ** 12 - 1.0
    air_annual = (1.0 + air_m) ** 12 - 1.0
    rate = 0.01
    av0 = 1_000_000.0

    var = fcf.vfa.measure(
        _annuity_mp(av0, rate, air_annual, {}),
        _basis(investment_return=inv_return), full=True)
    fixed = fcf.vfa.measure(
        _annuity_mp(av0, rate, float("nan"), {}),
        _basis(investment_return=inv_return), full=True)

    var_cf = var.cashflows.annuity_cf[0]
    fixed_cf = fixed.cashflows.annuity_cf[0]
    # First payment identical (growth^0 = 1); later payments rise above the level.
    assert np.isclose(var_cf[1], fixed_cf[1])
    assert var_cf[2] > fixed_cf[2]
    assert var_cf[3] > fixed_cf[3]
    assert np.allclose(fixed_cf, [0.0, fixed_cf[1], fixed_cf[1], fixed_cf[1]])


# ---------------------------------------------------------------------------
# The variable payout is a direct-participation feature: gmm.measure rejects it,
# vfa.measure measures it. A fixed (NaN) annuitizing book still works on gmm.
# ---------------------------------------------------------------------------

def test_gmm_rejects_variable_payout_vfa_accepts():
    fund_m = 0.004
    air_m = 0.002
    inv_return = (1.0 + fund_m) ** 12 - 1.0
    air_annual = (1.0 + air_m) ** 12 - 1.0

    var_mp = _annuity_mp(1_000_000.0, 0.01, air_annual, {})
    basis = _basis(investment_return=inv_return)

    with pytest.raises((ValueError, NotImplementedError)):
        fcf.gmm.measure(var_mp, basis, full=True)

    # vfa accepts the variable payout.
    m = fcf.vfa.measure(var_mp, basis, full=True)
    assert np.isfinite(m.bel[0])

    # A fixed (NaN air) annuitizing book is unaffected -- still measures on gmm.
    fixed_mp = _annuity_mp(1_000_000.0, 0.01, float("nan"), {})
    mf = fcf.gmm.measure(fixed_mp, basis, full=True)
    assert np.isfinite(mf.bel[0])


# ---------------------------------------------------------------------------
# The variable payout still carries longevity risk in the RA (the AIR-reserve is
# a survival-contingent stream): longevity_cv > 0 lifts the RA above zero.
# ---------------------------------------------------------------------------

def test_variable_payout_carries_longevity_ra():
    fund_m = 0.005
    air_m = 0.002
    inv_return = (1.0 + fund_m) ** 12 - 1.0
    air_annual = (1.0 + air_m) ** 12 - 1.0
    mp = _annuity_mp(1_000_000.0, 0.01, air_annual, {})

    no_ra = fcf.vfa.measure(mp, _basis(investment_return=inv_return,
                                       longevity_cv=0.0), full=True)
    with_ra = fcf.vfa.measure(mp, _basis(investment_return=inv_return,
                                         longevity_cv=0.15), full=True)
    assert np.isclose(no_ra.ra[0], 0.0)
    assert with_ra.ra[0] > 0.0


# ---------------------------------------------------------------------------
# A non-finite (inf) AIR is a data error -- rejected at construction, so it can
# never reach the kernel's variable-payout gate (where it would zero the income).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_air", [np.inf, -np.inf, -0.01])
def test_non_finite_or_negative_air_rejected(bad_air):
    with pytest.raises(ValueError):
        _annuity_mp(1_000_000.0, 0.01, bad_air, {})


# ---------------------------------------------------------------------------
# The bundled "ul-var-annuity" sample (a mixed variable / fixed book).
# ---------------------------------------------------------------------------

def test_ul_var_annuity_sample_measures():
    assert "ul-var-annuity" in fcf.samples.templates()
    mp = fcf.samples.model_points("ul-var-annuity")
    basis = fcf.samples.basis("ul-var-annuity")

    m = fcf.vfa.measure(mp, basis, full=True)
    assert np.all(np.isfinite(m.bel))
    # The variable (contract 0) and fixed (contract 1) rows both pay an annuity.
    ac = m.cashflows.annuity_cf
    assert ac[0, 180] > 0.0 and ac[1, 120] > 0.0

    # gmm rejects the variable book; vfa accepts it.
    with pytest.raises(NotImplementedError):
        fcf.gmm.measure(mp, basis)
