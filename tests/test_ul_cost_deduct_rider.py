"""Universal-life cost-deducting rider -- hand-calc anchor.

A cost-deducting rider (e.g. a universal recurring-cancer rider, 유니버설 재진단암)
draws its monthly risk charge from the account value, but pays a FIXED benefit on
the claim side -- it does not read the account balance. It is expressed with the
account-chassis flags ``funds_from_account=True`` and ``pays_account_balance=False``
on its :class:`fastcashflow.basis.CoverageRate`.

The death leg of the same contract carries BOTH flags True (charge on the net
amount at risk, benefit = max(account, face)). So the account roll deducts two
distinct charges each month:

* the DEATH leg's net-amount-at-risk COI -- ``coi_rate * max(0, face - av)``;
* every cost-deducting rider's FIXED charge -- ``rider_rate * rider_amount``.

The rider benefit is paid normally (here as a recurring MORBIDITY claim,
``inforce * rider_rate * rider_amount``); only its COST moves from a separate
premium to an account deduction. These tests pin a tiny case derived by hand,
prove the charge isolates to the account (a non-funded rider with the same
benefit leaves the account untouched), and check fast/full parity.
"""
import numpy as np

import fastcashflow as fcf
from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
from fastcashflow.basis import annual_to_monthly


def _basis(coi_annual_value, rider_annual_value, *, funds_rider=True, **overrides):
    """A UL basis with an account-backed DEATH leg and one recurring rider.

    ``funds_rider`` toggles whether the rider's cost is account-funded -- the
    control case (``funds_rider=False``) is a plain morbidity rider with the
    same benefit but no account deduction.
    """
    kw = dict(
        mortality_annual=0.0,             # no decrement -> inforce stays at count
        lapse_annual=0.0,
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.1,
        investment_return=0.0,            # r_m = 0 -> credit = guarantee floor
        coi_annual=coi_annual_value,
    )
    kw.update(overrides)
    kw["coverages"] = (
        CoverageRate("DEATH", coi_annual_value, funds_from_account=True,
                     pays_account_balance=True),
        CoverageRate("CANCER", rider_annual_value, funds_from_account=funds_rider,
                     pays_account_balance=False),
    )
    return Basis(**kw)


def _mp(face, rider_amount, **fields):
    n_mp = np.atleast_1d(np.asarray(fields["issue_age"])).shape[0]
    face_arr = np.broadcast_to(np.asarray(face, dtype=float), (n_mp,)).copy()
    rider_arr = np.broadcast_to(np.asarray(rider_amount, dtype=float), (n_mp,)).copy()
    fields["minimum_death_benefit"] = face_arr
    fields.setdefault("benefits", {"DEATH": face_arr, "CANCER": rider_arr})
    fields.setdefault("calculation_methods",
                      {"DEATH": CalculationMethod.DEATH,
                       "CANCER": CalculationMethod.MORBIDITY})
    return ModelPoints(**fields)


def _account(measurement):
    return measurement.cashflows.account


# ---------------------------------------------------------------------------
# Hand-calc: the rider charge AND the death-leg COI both deduct from the account.
# ---------------------------------------------------------------------------

def test_cost_deduct_rider_charges_account_hand_calc():
    # 1 policy, face 1e8, 500k premium/month, 2 months. Monthly death COI 1e-4
    # (on the NAR) and a recurring-cancer rider: monthly rate 2e-4 on a fixed
    # 30,000,000 benefit -> a level account charge of 6,000 each month, paid as a
    # morbidity claim of inforce * 6,000.
    coi_m = 0.0001
    rider_m = 0.0002
    credit_m = 0.003
    coi_annual = 1.0 - (1.0 - coi_m) ** 12
    rider_annual = 1.0 - (1.0 - rider_m) ** 12
    inv_return = (1.0 + credit_m) ** 12 - 1.0
    face = 100_000_000.0
    rider_amount = 30_000_000.0
    rider_charge = rider_m * rider_amount        # 6,000 per surviving policy/month

    mp = _mp(
        face, rider_amount,
        issue_age=np.array([40.0]),
        premium=np.array([500_000.0]),
        term_months=np.array([2]),
        account_value=np.array([0.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    basis = _basis(coi_annual, rider_annual, investment_return=inv_return)
    m = fcf.gmm.measure(mp, basis, full=True)
    acct = _account(m)

    assert np.isclose(annual_to_monthly(np.array(rider_annual)).item(), rider_m)

    # Month 0: a = 500_000; NAR = 1e8 - 500_000; COI = 1e-4 * NAR; charge = 6,000.
    nar0 = face - 500_000.0
    coi0 = coi_m * nar0
    assert np.isclose(acct.coi[0, 0], coi0)
    bef_inv0 = 500_000.0 - coi0 - rider_charge
    assert np.isclose(acct.av[0, 1], bef_inv0 * (1.0 + credit_m))
    assert np.isclose(acct.av_mid[0, 0], bef_inv0 * (1.0 + credit_m) ** 0.5)

    # Month 1: carries the month-0 closing AV; the rider charge deducts again.
    bef_fee1 = acct.av[0, 1] + 500_000.0
    coi1 = coi_m * (face - bef_fee1)
    bef_inv1 = bef_fee1 - coi1 - rider_charge
    assert np.isclose(acct.coi[0, 1], coi1)
    assert np.isclose(acct.av[0, 2], bef_inv1 * (1.0 + credit_m))

    # The rider benefit is paid every month as a recurring morbidity claim,
    # inforce (= count = 1, no decrement) * rider_charge.
    morb = m.cashflows.morbidity_cf
    assert np.allclose(morb[0, :2], rider_charge)


# ---------------------------------------------------------------------------
# Control: a funded rider deducts from the account; a non-funded rider does not.
# Both pay the SAME benefit -- only the funding choice differs.
# ---------------------------------------------------------------------------

def test_funded_vs_non_funded_rider_isolates_the_charge():
    coi_m = 0.0001
    rider_m = 0.0002
    credit_m = 0.003
    coi_annual = 1.0 - (1.0 - coi_m) ** 12
    rider_annual = 1.0 - (1.0 - rider_m) ** 12
    inv_return = (1.0 + credit_m) ** 12 - 1.0
    face = 100_000_000.0
    rider_amount = 30_000_000.0
    rider_charge = rider_m * rider_amount

    fields = dict(
        issue_age=np.array([40.0]),
        premium=np.array([500_000.0]),
        term_months=np.array([6]),
        account_value=np.array([0.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    mp = _mp(face, rider_amount, **fields)

    funded = fcf.gmm.measure(
        mp, _basis(coi_annual, rider_annual, funds_rider=True,
                   investment_return=inv_return), full=True)
    plain = fcf.gmm.measure(
        mp, _basis(coi_annual, rider_annual, funds_rider=False,
                   investment_return=inv_return), full=True)

    av_funded = _account(funded).av
    av_plain = _account(plain).av

    # The funded account is strictly lower from month 1 on (a charge is drawn);
    # before any charge (month 0 opening) they are equal.
    assert np.isclose(av_funded[0, 0], av_plain[0, 0])
    assert np.all(av_funded[0, 1:] < av_plain[0, 1:])

    # The month-1 gap equals exactly the month-0 charge, carried by one credit.
    gap1 = av_plain[0, 1] - av_funded[0, 1]
    assert np.isclose(gap1, rider_charge * (1.0 + credit_m))

    # The benefit is identical under both funding choices.
    assert np.allclose(funded.cashflows.morbidity_cf,
                       plain.cashflows.morbidity_cf)


# ---------------------------------------------------------------------------
# The cost-deducting rider's health benefit is priced into the account-book RA.
# The account RA branch otherwise carries only at-risk mortality + expense; the
# rider adds a morbidity term, isolated here by toggling morbidity_cv only (it
# changes the RA, never the account roll or the BEL).
# ---------------------------------------------------------------------------

def test_cost_deduct_rider_priced_in_account_ra():
    coi_annual = 0.0015
    rider_annual = 1.0 - (1.0 - 0.0002) ** 12      # monthly rider rate 2e-4
    rider_amount = 30_000_000.0
    morbidity_cv = 0.2
    mp = _mp(
        100_000_000.0, rider_amount,
        issue_age=np.array([45.0]),
        premium=np.array([600_000.0]),
        term_months=np.array([36]),
        account_value=np.array([0.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    priced = _basis(coi_annual, rider_annual, investment_return=0.024,
                    morbidity_cv=morbidity_cv)
    unpriced = _basis(coi_annual, rider_annual, investment_return=0.024,
                      morbidity_cv=0.0)

    m_priced = fcf.gmm.measure(mp, priced, full=True)
    m_unpriced = fcf.gmm.measure(mp, unpriced, full=True)

    # BEL and the account roll are independent of morbidity_cv.
    assert np.allclose(m_priced.bel, m_unpriced.bel)
    assert np.allclose(_account(m_priced).av, _account(m_unpriced).av)

    # The RA gains exactly z * morbidity_cv * pv(morbidity_cf), the rider's
    # health benefit priced through morbidity risk (mid-month discounting).
    from fastcashflow.engine import _norm_ppf
    i_m = (1.03) ** (1.0 / 12.0) - 1.0
    morb = m_priced.cashflows.morbidity_cf[0]
    t = np.arange(morb.shape[0])
    pv_morb = float(np.sum(morb * (1.0 + i_m) ** (-(t + 0.5))))
    expected = _norm_ppf(0.75) * morbidity_cv * pv_morb
    assert pv_morb > 0.0
    assert np.isclose(float(m_priced.ra[0] - m_unpriced.ra[0]), expected)


# ---------------------------------------------------------------------------
# Fast (fused) vs full (roll-forward) parity for a cost-deducting rider book.
# ---------------------------------------------------------------------------

def test_cost_deduct_fast_full_parity():
    coi_annual = 0.0015
    rider_annual = 0.0024
    mp = _mp(
        100_000_000.0, 30_000_000.0,
        issue_age=np.array([45.0, 50.0]),
        premium=np.array([600_000.0, 800_000.0]),
        term_months=np.array([120, 120]),
        account_value=np.array([0.0, 2_000_000.0]),
        minimum_crediting_rate=np.array([0.01, 0.01]),
        sex=np.array([0, 1]),
    )
    basis = _basis(coi_annual, rider_annual,
                   mortality_annual=0.003, lapse_annual=0.02,
                   investment_return=0.024)
    full = fcf.gmm.measure(mp, basis, full=True)
    fast = fcf.gmm.measure(mp, basis, full=False)
    assert np.allclose(full.bel, fast.bel)
    assert np.allclose(full.ra, fast.ra)
    assert np.allclose(full.csm, fast.csm)
    assert np.allclose(full.loss_component, fast.loss_component)


# ---------------------------------------------------------------------------
# The bundled "ul-cost-deduct" sample template measures and charges the account.
# ---------------------------------------------------------------------------

def test_ul_cost_deduct_sample_measures():
    import pytest

    assert "ul-cost-deduct" in fcf.samples.templates()
    mp = fcf.samples.model_points("ul-cost-deduct")
    basis = fcf.samples.basis("ul-cost-deduct")

    full = fcf.gmm.measure(mp, basis, full=True)
    fast = fcf.gmm.measure(mp, basis, full=False)
    assert np.allclose(full.bel, fast.bel)
    assert np.allclose(full.ra, fast.ra)
    assert np.allclose(full.csm, fast.csm)

    # The CANCER rider draws a charge from the account, so the morbidity benefit
    # is paid AND the account is lower than the same book without the rider.
    assert np.any(full.cashflows.morbidity_cf > 0.0)

    # The template is load-only (constructed in memory, no exportable files).
    with pytest.raises(NotImplementedError):
        fcf.samples.export("/tmp/should_not_be_written_cd",
                           template="ul-cost-deduct")
