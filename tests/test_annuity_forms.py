"""Plain-annuity payout forms -- deferred start / term-certain / guaranteed period.

These three IntArray ModelPoints fields shape the ``annuity_payment`` income:

* ``annuity_start_months``     -- deferred payout start (0 = from inception)
* ``annuity_term_months``      -- term-certain payout count (0 = unlimited/life)
* ``annuity_guarantee_months`` -- certain-and-life guarantee window (0 = pure life)

This module holds the input-layer validation (the behaviour hand-calcs land with
each form's kernel stage). All three default to 0, so an existing book is a no-op.
"""
import numpy as np
import pytest

from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
from fastcashflow.gmm import measure


def _flat(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


_Q = 0.012
_CM = {"DEATH": CalculationMethod.DEATH}


def _payout_basis():
    """Flat mortality, no lapse, zero discount, RA off -- every figure by hand."""
    return Basis(
        mortality_annual=_flat(_Q), lapse_annual=_flat(0.0), discount_annual=0.0,
        ra_confidence=0.75, mortality_cv=0.0,
        coverages=(CoverageRate("DEATH", _flat(_Q)),))


def _surv(term):
    """Deterministic in-force from inception under the flat monthly mortality."""
    mq = 1.0 - (1.0 - _Q) ** (1.0 / 12.0)
    return (1.0 - mq) ** np.arange(term)


def _annuity_payout_mp(term=24, payment=100.0, **forms):
    return ModelPoints.single(issue_age=40, premium=0.0, term_months=term,
                              annuity_payment=payment, calculation_methods=_CM,
                              **forms)


def _annuity_mp(**overrides):
    kw = dict(issue_age=40, premium=0.0, term_months=120, annuity_payment=10.0)
    kw.update(overrides)
    return ModelPoints.single(**kw)


def test_annuity_form_fields_default_zero():
    """An ordinary book leaves all three forms at 0 (the historical behaviour)."""
    mp = _annuity_mp()
    assert mp.annuity_start_months[0] == 0
    assert mp.annuity_term_months[0] == 0
    assert mp.annuity_guarantee_months[0] == 0


def test_annuity_form_fields_set_and_subset():
    """The fields round-trip through ``.single`` and survive ``subset``."""
    mp = _annuity_mp(annuity_start_months=12, annuity_term_months=60,
                     annuity_guarantee_months=24)
    assert (mp.annuity_start_months[0], mp.annuity_term_months[0],
            mp.annuity_guarantee_months[0]) == (12, 60, 24)
    sub = mp.subset([0])
    assert sub.annuity_start_months[0] == 12
    assert sub.annuity_term_months[0] == 60
    assert sub.annuity_guarantee_months[0] == 24


def test_negative_form_field_raises():
    for field in ("annuity_start_months", "annuity_term_months",
                  "annuity_guarantee_months"):
        with pytest.raises(ValueError, match=">= 0"):
            _annuity_mp(**{field: -1})


def test_guarantee_exceeds_term_raises():
    """The guarantee window cannot outlast the payout term."""
    with pytest.raises(ValueError, match="guarantee_months must be <="):
        _annuity_mp(annuity_term_months=12, annuity_guarantee_months=24)


def test_start_at_or_past_boundary_raises():
    """A deferred start must leave at least one payout month in the horizon."""
    with pytest.raises(ValueError, match="start_months must be <"):
        _annuity_mp(term_months=120, annuity_start_months=120)


def test_new_form_with_annuitization_raises():
    """The plain-annuity forms are not yet supported with UL annuitization."""
    with pytest.raises(NotImplementedError, match="annuitization"):
        ModelPoints.single(
            issue_age=40, premium=0.0, term_months=120, account_value=1000.0,
            premium_term_months=0, annuitization_months=60,
            annuitization_rate=0.004, annuity_guarantee_months=24)


# ---------------------------------------------------------------------------
# Cash-flow behaviour -- hand-calc anchors (zero discount, flat mortality)
# ---------------------------------------------------------------------------

def test_deferred_start_handcalc():
    """A deferred annuity pays nothing before the start month; the BEL is the
    undiscounted PV of the survival-contingent income from the start onward."""
    S, term = 12, 24
    m = measure(_annuity_payout_mp(term=term, annuity_start_months=S), _payout_basis())
    acf = m.cashflows.annuity_cf[0]
    assert np.allclose(acf[:S], 0.0)
    inforce = _surv(term)
    assert np.isclose(m.bel[0], float(np.sum(inforce[S:] * 100.0)), rtol=1e-9)


def test_term_certain_handcalc():
    """A term-certain annuity pays exactly N payouts then stops."""
    N, term = 6, 24
    m = measure(_annuity_payout_mp(term=term, annuity_term_months=N), _payout_basis())
    acf = m.cashflows.annuity_cf[0]
    assert np.all(acf[:N] > 0) and np.allclose(acf[N:], 0.0)
    inforce = _surv(term)
    assert np.isclose(m.bel[0], float(np.sum(inforce[:N] * 100.0)), rtol=1e-9)


def test_guaranteed_period_handcalc():
    """The guarantee window is paid with CERTAINTY on the payout-start count
    (here 1 policy, so a flat level even as in-force decays), then the income
    reverts to survival-contingent. BEL = PV(certain G payments) + PV(tail)."""
    G, term = 6, 24
    m = measure(_annuity_payout_mp(term=term, annuity_guarantee_months=G), _payout_basis())
    acf = m.cashflows.annuity_cf[0]
    inforce = _surv(term)
    # first G payments are the level (the count is 1), NOT inforce-decayed
    assert np.allclose(acf[:G], 100.0)
    assert not np.allclose(acf[1:G], inforce[1:G] * 100.0)   # would decay if contingent
    assert np.allclose(acf[G:], inforce[G:] * 100.0)         # contingent tail
    bel_hand = float(np.sum(np.ones(G) * 100.0) + np.sum(inforce[G:] * 100.0))
    assert np.isclose(m.bel[0], bel_hand, rtol=1e-9)


def test_immediate_annuity_unchanged():
    """All-zero forms reproduce the level whole-life-from-inception income."""
    term = 24
    m = measure(_annuity_payout_mp(term=term), _payout_basis())
    inforce = _surv(term)
    assert np.isclose(m.bel[0], float(np.sum(inforce * 100.0)), rtol=1e-9)


def test_annuity_forms_route_to_full_on_fast_path():
    """measure(full=False) routes a forms book to the full kernel (same BEL)."""
    mp = _annuity_payout_mp(annuity_start_months=12)
    basis = _payout_basis()
    assert np.isclose(measure(mp, basis, full=False).bel[0], measure(mp, basis).bel[0])


# ---------------------------------------------------------------------------
# Guaranteed-period longevity-RA split -- the certain payments bear no risk
# ---------------------------------------------------------------------------

def _ra_basis():
    """Longevity RA on, mortality RA off, zero discount -- isolates longevity."""
    return Basis(
        mortality_annual=_flat(_Q), lapse_annual=_flat(0.0), discount_annual=0.0,
        ra_confidence=0.75, mortality_cv=0.0, longevity_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat(_Q)),))


def test_guaranteed_period_ra_excludes_certain_payments():
    """The longevity RA prices ONLY the survival-contingent tail (t >= G); the
    guaranteed payments are certain, so they carry no longevity risk and the BEL
    still includes them."""
    from fastcashflow._numerics import _norm_ppf
    G, term = 6, 24
    basis = _ra_basis()
    m = measure(_annuity_payout_mp(term=term, annuity_guarantee_months=G), basis)
    inforce = _surv(term)
    z = _norm_ppf(0.75)
    ra_hand = z * 0.10 * float(np.sum(inforce[G:] * 100.0))   # contingent tail only
    assert np.isclose(m.ra[0], ra_hand, rtol=1e-9)
    # the BEL still carries the certain payments (only the RA drops them)
    bel_hand = float(np.sum(np.ones(G) * 100.0) + np.sum(inforce[G:] * 100.0))
    assert np.isclose(m.bel[0], bel_hand, rtol=1e-9)


def test_guarantee_ra_below_pure_life_ra():
    """A guaranteed annuity's longevity RA is below the same pure-life annuity's
    (the certain portion is removed from the risk-bearing PV)."""
    basis = _ra_basis()
    mg = measure(_annuity_payout_mp(annuity_guarantee_months=6), basis)
    ml = measure(_annuity_payout_mp(), basis)
    assert mg.ra[0] < ml.ra[0]


def test_no_guarantee_ra_unchanged():
    """Without a guarantee, the longevity RA prices the whole survival stream."""
    from fastcashflow._numerics import _norm_ppf
    term = 24
    basis = _ra_basis()
    m = measure(_annuity_payout_mp(term=term), basis)
    inforce = _surv(term)
    z = _norm_ppf(0.75)
    assert np.isclose(m.ra[0], z * 0.10 * float(np.sum(inforce * 100.0)), rtol=1e-9)
