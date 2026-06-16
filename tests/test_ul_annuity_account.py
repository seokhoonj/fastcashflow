"""Universal-life annuitization (2-phase whole-life annuity) -- hand-calc anchor.

A universal-life annuity accumulates an account value (phase 1, the ordinary UL
roll) and at ``annuitization_months`` converts the balance to a guaranteed
survival-contingent income (phase 2): ``locked_annuity_payment =
max(account, GMAB) * annuitization_rate``, paid annuity-due on the surviving
in-force, with no further premium / COI / surrender and no maturity lump (the
balance was already converted -- paying a lump too would double-count). The
payout decrements by mortality only (a life annuity in payment cannot lapse).

These pin the conversion arithmetic, the phase-2 cash-flow hygiene, the
mortality-only payout decrement, the longevity risk adjustment on the payout,
and the conversion gain/loss landing in the CSM -- the load-bearing details the
critique panel flagged.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
from fastcashflow.basis import annual_to_monthly
from fastcashflow.curves import discount_monthly_curve, discount_factors_from_curve


def _annuity_basis(coi_annual_value=0.0, **overrides):
    """A minimal UL-annuity basis -- account-backed DEATH coverage, flat rates."""
    kw = dict(
        mortality_annual=0.0,
        lapse_annual=0.0,
        discount_annual=0.0,
        ra_confidence=0.75,
        mortality_cv=0.1,
        longevity_cv=0.15,
        investment_return=0.0,
        coi_annual=coi_annual_value,
    )
    kw.update(overrides)
    kw["coverages"] = (
        CoverageRate("DEATH", coi_annual_value, funds_from_account=True,
                     pays_account_balance=True),
    )
    return Basis(**kw)


def _annuity_mp(face, **fields):
    face_arr = np.atleast_1d(np.asarray(face, dtype=float))
    fields["minimum_death_benefit"] = face_arr
    fields.setdefault("benefits", {"DEATH": face_arr})
    fields.setdefault("calculation_methods", {"DEATH": CalculationMethod.DEATH})
    return ModelPoints(**fields)


# ---------------------------------------------------------------------------
# Conversion + phase-2 cash flows + the conversion gain in the CSM (the crux).
# ---------------------------------------------------------------------------

def test_ul_annuity_conversion_and_cashflows_hand_calc():
    # Single-premium 1e6 account, convert at month 1, then a 0.01/month guaranteed
    # income on the surviving in-force. No mortality / lapse / COI / credit / load
    # / discount -- so the whole thing is hand-derivable. term = 4 (boundary 4),
    # so the payout runs months 1, 2, 3 (three annuity-due payments).
    rate = 0.01
    mp = _annuity_mp(
        10_000_000.0,                          # face registers the coverage; COI is 0
        issue_age=np.array([60.0]),
        premium=np.array([0.0]),
        term_months=np.array([4]),
        premium_term_months=np.array([0]),     # single-premium; no premium <= A
        account_value=np.array([1_000_000.0]),
        annuitization_months=np.array([1]),
        annuitization_rate=np.array([rate]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _annuity_basis(0.0, longevity_cv=0.0)   # COI 0; isolate the conversion gain
    m = fcf.gmm.measure(mp, basis, full=True)
    proj = m.cashflows
    acct = proj.account

    # Conversion: balance carried into month 1 = the single premium (no charge,
    # no credit) = 1e6; locked income = 1e6 * 0.01 = 10_000.
    assert np.isclose(acct.av[0, 1], 1_000_000.0)
    locked = 1_000_000.0 * rate
    # Annuity-due from month 1, in-force = 1 (no decrement): three payments.
    assert np.allclose(proj.annuity_cf[0], [0.0, locked, locked, locked])
    # Phase-2 hygiene: no account-death claim, no surrender, no maturity lump.
    assert np.allclose(proj.mortality_cf[0], 0.0)
    assert np.allclose(proj.surrender_cf[0], 0.0)
    assert np.isclose(proj.maturity_cf[0], 0.0)
    # Account is spent at conversion -- av is not carried past A (av[2:] == 0).
    assert np.allclose(acct.av[0, 2:], 0.0)

    # The crux: an off-market guaranteed rate. The insurer took 1e6, pays out
    # 3 * 10_000 = 30_000 of income (zero discount), keeps the rest. With the
    # seed-netting BEL = PV(annuity) - PV(premium) - account_value0:
    #   BEL = 30_000 - 0 - 1_000_000 = -970_000, a profit -> CSM = 970_000.
    assert np.isclose(m.bel[0], 30_000.0 - 1_000_000.0)
    assert np.isclose(m.csm[0], 970_000.0)
    assert np.isclose(m.loss_component[0], 0.0)
    assert isinstance(m, fcf.GMMMeasurement)


def test_ul_annuity_gmab_floor_at_conversion():
    # The conversion floors the balance at the GMAB (minimum_accumulation_benefit).
    # account 1e6 but GMAB 1.5e6 -> converted_balance = 1.5e6, income on the floor.
    rate = 0.02
    gmab = 1_500_000.0
    mp = _annuity_mp(
        10_000_000.0,
        issue_age=np.array([60.0]),
        premium=np.array([0.0]),
        term_months=np.array([3]),
        premium_term_months=np.array([0]),
        account_value=np.array([1_000_000.0]),
        minimum_accumulation_benefit=np.array([gmab]),
        annuitization_months=np.array([1]),
        annuitization_rate=np.array([rate]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _annuity_basis(0.0)
    m = fcf.gmm.measure(mp, basis, full=True)
    # Account (1e6) < GMAB (1.5e6) -> the floor bites: income = 1.5e6 * 0.02.
    locked = gmab * rate
    assert np.allclose(m.cashflows.annuity_cf[0], [0.0, locked, locked])


# ---------------------------------------------------------------------------
# Mortality-only payout decrement -- lapse is suppressed in the payout phase.
# ---------------------------------------------------------------------------

def test_ul_annuity_payout_decrements_by_mortality_only():
    # A large lapse in accumulation; in the payout phase lapse must be suppressed
    # (a life annuity in payment cannot surrender), so the in-force decays by
    # mortality ALONE. Assert: in payout inforce[t+1] == inforce[t] - deaths[t]
    # exactly (no lapse loss); in accumulation lapse removes strictly more.
    mp = _annuity_mp(
        10_000_000.0,
        issue_age=np.array([60.0]),
        premium=np.array([200_000.0]),
        term_months=np.array([6]),
        premium_term_months=np.array([2]),
        account_value=np.array([0.0]),
        annuitization_months=np.array([2]),
        annuitization_rate=np.array([0.01]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _annuity_basis(0.001, mortality_annual=0.02, lapse_annual=0.3)
    m = fcf.gmm.measure(mp, basis, full=True)
    proj = m.cashflows
    inforce = proj.inforce[0]
    deaths = proj.deaths[0]
    A = 2

    # Payout months (t >= A, and t+1 still inside the horizon): mortality-only.
    for t in range(A, len(inforce) - 1):
        assert np.isclose(inforce[t + 1], inforce[t] - deaths[t]), t
    # Accumulation months: lapse removes more than mortality alone.
    for t in range(A):
        assert inforce[t + 1] < inforce[t] - deaths[t] - 1e-12, t
    # The annuity is paid on the surviving in-force at each payout month.
    locked = m.cashflows.account.av[0, A] * 0.01
    for t in range(A, len(inforce)):
        assert np.isclose(proj.annuity_cf[0, t], inforce[t] * locked), t


def test_ul_annuity_no_surrender_or_account_death_in_payout():
    # Explicit phase-2 hygiene with decrements on: zero account-death claim and
    # zero surrender for every payout month (the whole account benefit block is
    # bypassed once annuitized -- otherwise the converted balance would be paid
    # twice).
    mp = _annuity_mp(
        10_000_000.0,
        issue_age=np.array([60.0]),
        premium=np.array([200_000.0]),
        term_months=np.array([6]),
        premium_term_months=np.array([2]),
        account_value=np.array([0.0]),
        annuitization_months=np.array([2]),
        annuitization_rate=np.array([0.01]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _annuity_basis(0.001, mortality_annual=0.02, lapse_annual=0.3)
    proj = fcf.gmm.measure(mp, basis, full=True).cashflows
    A = 2
    assert np.allclose(proj.mortality_cf[0, A:], 0.0)
    assert np.allclose(proj.surrender_cf[0, A:], 0.0)


# ---------------------------------------------------------------------------
# Longevity RA on the payout; conversion gain via an independent forward PV.
# ---------------------------------------------------------------------------

def test_ul_annuity_payout_bears_longevity_ra():
    # The payout income bears longevity risk; the RA must carry a
    # longevity_cv * pv(annuity) term. Compare an annuitizing book with
    # longevity_cv on vs off -- the difference isolates that term (and proves it
    # is non-zero, i.e. the account RA branch was extended).
    def build(longevity_cv):
        mp = _annuity_mp(
            10_000_000.0,
            issue_age=np.array([60.0]),
            premium=np.array([0.0]),
            term_months=np.array([6]),
            premium_term_months=np.array([0]),
            account_value=np.array([1_000_000.0]),
            annuitization_months=np.array([1]),
            annuitization_rate=np.array([0.03]),
            minimum_crediting_rate=np.array([0.0]),
            sex=np.array([0]),
        )
        basis = _annuity_basis(0.0, mortality_annual=0.02, longevity_cv=longevity_cv)
        return fcf.gmm.measure(mp, basis, full=True)

    ra_on = build(0.15).ra[0]
    ra_off = build(0.0).ra[0]
    assert ra_on > ra_off > -1e-9            # the longevity term adds positive RA
    assert ra_on - ra_off > 0.0


def test_ul_annuity_full_matches_independent_forward_pv():
    # The engine BEL must equal an independent forward PV of the projected cash
    # flows, netted by the seed account value: a cross-check of the whole
    # assembly (conversion, payout, mortality-only decrement, fund netting).
    mp = _annuity_mp(
        20_000_000.0,
        issue_age=np.array([55.0]),
        premium=np.array([300_000.0]),
        term_months=np.array([12]),
        premium_term_months=np.array([3]),
        account_value=np.array([0.0]),
        annuitization_months=np.array([3]),
        annuitization_rate=np.array([0.02]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _annuity_basis(
        0.002, mortality_annual=0.015, lapse_annual=0.1, discount_annual=0.03,
        investment_return=0.0)
    m = fcf.gmm.measure(mp, basis, full=True)
    proj = m.cashflows

    # Independent forward PV at inception, off the SAME locked-in discount curve
    # the engine uses (the curve itself is tested elsewhere; here we cross-check
    # the cash-flow -> BEL assembly: conversion, payout, mortality-only decrement,
    # fund netting). Premiums and annuity payments discount at the start of their
    # month (discount_factor_bom), claims / morbidity / disability / expense / surrender
    # mid-month (discount_factor_mid); the maturity lump rolls to the boundary column.
    bom, mid = discount_factors_from_curve(discount_monthly_curve(basis, proj.n_time))
    bom_t = bom[:proj.n_time]
    mid_legs = (proj.mortality_cf[0] + proj.morbidity_cf[0] + proj.disability_cf[0]
                + proj.expense_cf[0] + proj.surrender_cf[0])
    pv = (float(np.dot(proj.annuity_cf[0], bom_t))
          - float(np.dot(proj.premium_cf[0], bom_t))
          + float(np.dot(mid_legs, mid)))
    pv += proj.maturity_cf[0] * bom[int(mp.contract_boundary_months[0])]
    pv -= mp.account_value[0]                 # seed-netting (account_value0)
    assert np.isclose(m.bel[0], pv, rtol=1e-9, atol=1e-6)


# ---------------------------------------------------------------------------
# Engine-entry guards (need the coverage flags / the boundary).
# ---------------------------------------------------------------------------

def test_annuitization_on_non_account_book_rejected():
    # annuitization_months set on a plain (non-account-backed) DEATH coverage --
    # there is no balance to convert. The account-backed cross-check needs the
    # coverage flags, so it fires at measurement, not construction.
    face = np.array([1_000_000.0])
    mp = ModelPoints(
        issue_age=np.array([60.0]), premium=np.array([0.0]),
        term_months=np.array([6]), premium_term_months=np.array([0]),
        annuitization_months=np.array([2]), annuitization_rate=np.array([0.01]),
        minimum_death_benefit=face, benefits={"DEATH": face},
        calculation_methods={"DEATH": CalculationMethod.DEATH})
    # Plain DEATH coverage -- no funds_from_account / pays_account_balance flag.
    basis = Basis(mortality_annual=0.0, lapse_annual=0.0, discount_annual=0.0,
                  ra_confidence=0.75, mortality_cv=0.1,
                  coverages=(CoverageRate("DEATH", 0.001),))
    with pytest.raises(ValueError, match="no account-backed coverage"):
        fcf.gmm.measure(mp, basis, full=True)


def test_annuitization_month_must_be_below_boundary():
    # A == boundary (== term here) is never reached by the t = 0..boundary-1
    # loop, so it would pay neither the stream nor a lump -- rejected.
    mp = _annuity_mp(
        10_000_000.0,
        issue_age=np.array([60.0]),
        premium=np.array([0.0]),
        term_months=np.array([4]),
        premium_term_months=np.array([0]),
        account_value=np.array([1_000_000.0]),
        annuitization_months=np.array([4]),       # == term == boundary
        annuitization_rate=np.array([0.01]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _annuity_basis(0.0)
    with pytest.raises(ValueError, match="< contract_boundary_months"):
        fcf.gmm.measure(mp, basis, full=True)
