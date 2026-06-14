"""Every GMM cookbook mechanic measured together in one basket.

A heterogeneous portfolio -- one model point per mechanic, each routed to a
segment with its own state model and basis -- is measured in a single
``measure(mp, basis_dict)`` call. Two invariants must hold:

* **basket == standalone** -- a segment's headline is identical whether it is
  measured in the mixed portfolio or on its own, i.e. segment stitching never
  lets one segment corrupt a neighbour.
* **full == fast** -- the fused, headline-only fast path (``full=False``)
  agrees with the trajectory path (``full=True``) on the whole mix.

The mechanics exercised side by side: a death + maturity contract, a depleting
diagnosis pool, a repeated-payout morbidity rider, a survival annuity, waiver
to paid-up, semi-Markov reincidence, semi-Markov disability income with
recovery, semi-Markov long-term care (a monthly-benefit cap and an elevated
in-state mortality), a surrender value, and an IFRS 17 contract-boundary cut.

The routing keys are the bare ``product`` / ``channel``; a coverage maps to one
calculation method portfolio-wide, so the diagnosis and the inpatient cancer
riders carry distinct codes (``CANCER_DIAGNOSIS`` vs ``CANCER_INPATIENT``).
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.basis import BasisRouter
from fastcashflow import State, Transition, StateModel
from fastcashflow.state_model import STATE_MODELS

CM = fcf.CalculationMethod

# --- flat toy rates (production uses experience tables) ----------------------
_death     = lambda s, a, d:     np.full(np.shape(a), 0.01)
_lapse     = lambda s, a, d:     np.full(np.shape(d), 0.03)
_care_mort = lambda s, a, d:     np.full(np.shape(a), 0.20)   # elevated in care
_inc       = lambda s, a, d:     np.full(np.shape(a), 0.02)   # incidence
_cancer    = lambda s, a, d:     np.full(np.shape(a), 0.005)
_ci1       = lambda s, a, d:     np.full(np.shape(a), 0.05)   # first diagnosis
_ci2       = lambda s, a, d, sd: np.where(sd < 2, 0.0, 0.20)  # reincidence, waiting
_di_rec    = lambda s, a, d, sd: np.where(sd < 12, 0.45, 0.10)  # recovery, sojourn

# --- one state model per mechanic --------------------------------------------
_M_DEATH = StateModel(states=(
    State("active", pays_premium=True, transitions=(
        Transition("mortality"), Transition("lapse"))),
), seating=(0,))

_M_DI = StateModel(states=(
    State("active", pays_premium=True, transitions=(
        Transition("mortality"),
        Transition("waiver_incidence", to="disabled"),
        Transition("lapse"))),
    State("disabled", pays_periodic_benefit=True, sojourn_tracking_months=24, transitions=(
        Transition("mortality"),
        Transition("disability_recovery", to="active", sojourn_dependent=True))),
), seating=(0, 1))

_M_LTC = StateModel(states=(
    State("active", pays_premium=True, transitions=(
        Transition("mortality"), Transition("lapse"),
        Transition("waiver_incidence", to="care", pays_lump_sum=True))),
    State("care", pays_periodic_benefit=True, sojourn_tracking_months=60, periodic_benefit_term_months=36,
          mortality_rate_name="dth_care", transitions=(
          Transition("mortality"),)),
), seating=(0, 1))

_M_REINCID = StateModel(states=(
    State("healthy", pays_premium=True, transitions=(
        Transition("mortality"),
        Transition("ci_incidence", to="post_first", pays_lump_sum=True),
        Transition("lapse"))),
    State("post_first", sojourn_tracking_months=12, transitions=(
        Transition("mortality"),
        Transition("ci_reincidence", to="post_second",
                   pays_lump_sum=True, sojourn_dependent=True))),
    State("post_second", transitions=(Transition("mortality"),)),
), seating=(0, 1, 2))


def _basis(**kw):
    base = dict(
        mortality_annual = _death, lapse_annual = _lapse, discount_annual = 0.03,
        ra_confidence    = 0.75,   mortality_cv = 0.10,   morbidity_cv    = 0.15,
        disability_cv    = 0.20,   coverages    = (fcf.CoverageRate("DEATH", _death),),
    )
    base.update(kw)
    return fcf.Basis(**base)


def _bases():
    """One basis (with its own state model) per product segment."""
    return {
        "DEATH_A":   _basis(state_model=_M_DEATH),
        "DIAG_A":    _basis(state_model=_M_DEATH,
                            coverages=(fcf.CoverageRate("CANCER_DIAGNOSIS", _cancer),)),
        "MORB_A":    _basis(state_model=_M_DEATH,
                            coverages=(fcf.CoverageRate("CANCER_INPATIENT", _cancer),)),
        "ANN_A":     _basis(state_model=_M_DEATH),
        "WAIVER_A":  _basis(state_model=STATE_MODELS["WAIVER"],
                            waiver_incidence_annual=_inc),
        "REINCID_A": _basis(state_model=_M_REINCID,
                            ci_incidence_annual=_ci1, ci_reincidence_annual=_ci2),
        "DI_A":      _basis(state_model=_M_DI,
                            waiver_incidence_annual=_inc, disability_recovery_annual=_di_rec),
        "LTC_A":     _basis(state_model=_M_LTC,
                            waiver_incidence_annual=_inc,
                            state_mortality_annual={"dth_care": _care_mort}),
        "SURR_A":    _basis(state_model=_M_DEATH,
                            surrender_value_curve=np.full(240, 0.8),
                            surrender_value_basis="cum_premium_factor"),
        "RENEW_A":   _basis(state_model=_M_DEATH,
                            coverages=(fcf.CoverageRate("CANCER_INPATIENT", _cancer),)),
    }


# Per-row spec: product, issue_age, benefit, premium, term, disability_income,
# disability_benefit, boundary, maturity, annuity_payment, methods (standalone).
_ROWS = [
    ("DEATH_A",   40, 100_000_000., 50_000, 240,         0.,          0., 240, 5_000_000.,         0., {"DEATH": CM.DEATH}),
    ("DIAG_A",    45,  30_000_000., 28_000, 240,         0.,          0., 240,         0.,         0., {"CANCER_DIAGNOSIS": CM.DIAGNOSIS}),
    ("MORB_A",    45,   1_000_000., 12_000, 240,         0.,          0., 240,         0.,         0., {"CANCER_INPATIENT": CM.MORBIDITY}),
    ("ANN_A",     60,           0.,     0., 240,         0.,          0., 240,         0., 1_000_000., {}),
    ("WAIVER_A",  40, 100_000_000., 45_000, 240,         0.,          0., 240,         0.,         0., {"DEATH": CM.DEATH}),
    ("REINCID_A", 40,     100_000.,     0.,   4,         0.,  1_000_000.,   4,         0.,         0., {"DEATH": CM.DEATH}),
    ("DI_A",      45,           0., 30_000, 120, 1_000_000.,          0., 120,         0.,         0., {"DEATH": CM.DEATH}),
    ("LTC_A",     60,           0., 90_000, 360, 1_000_000., 20_000_000., 360,         0.,         0., {"DEATH": CM.DEATH}),
    ("SURR_A",    40,  50_000_000., 40_000, 240,         0.,          0., 240,         0.,         0., {"DEATH": CM.DEATH}),
    ("RENEW_A",   40,  30_000_000., 25_000, 480,         0.,          0., 120,         0.,         0., {"CANCER_INPATIENT": CM.MORBIDITY}),
]

# Portfolio-wide taxonomy -- one method per coverage code.
_METHODS = {"DEATH": CM.DEATH,
            "CANCER_DIAGNOSIS": CM.DIAGNOSIS,
            "CANCER_INPATIENT": CM.MORBIDITY}


def _standalone_mp(spec):
    p, age, ben, prem, term, di, db, bdy, mat, ann, methods = spec
    kw = dict(
        issue_age                = np.array([age], dtype=np.int64),
        benefits                 = {0: np.array([ben])},
        premium                  = np.array([float(prem)]),
        term_months              = np.array([term], dtype=np.int64),
        disability_income        = np.array([di]),
        disability_benefit       = np.array([db]),
        contract_boundary_months = np.array([bdy], dtype=np.int64),
        maturity_benefit         = np.array([mat]),
        state                    = np.array([0], dtype=np.int64),
        calculation_methods      = methods,
    )
    if ann > 0:
        kw["annuity_payment"]          = np.array([ann])
        kw["annuity_frequency_months"] = np.array([12], dtype=np.int64)
    return fcf.ModelPoints(**kw)


def _basket_mp():
    col = lambda i: [r[i] for r in _ROWS]
    return fcf.ModelPoints(
        issue_age                = np.array(col(1), dtype=np.int64),
        benefits                 = {0: np.array(col(2))},
        premium                  = np.array([float(x) for x in col(3)]),
        term_months              = np.array(col(4), dtype=np.int64),
        product                  = np.array(col(0)),
        channel                  = np.array(["FC"] * len(_ROWS)),
        disability_income        = np.array(col(5)),
        disability_benefit       = np.array(col(6)),
        contract_boundary_months = np.array(col(7), dtype=np.int64),
        maturity_benefit         = np.array(col(8)),
        annuity_payment          = np.array(col(9)),
        annuity_frequency_months = np.full(len(_ROWS), 12, dtype=np.int64),
        state                    = np.zeros(len(_ROWS), dtype=np.int64),
        calculation_methods      = _METHODS,
    )


def _basket_basis():
    bases = _bases()
    return BasisRouter({(r[0], "FC"): bases[r[0]] for r in _ROWS})
def test_basket_headline_matches_standalone():
    """Each segment in the mixed portfolio equals the same policy measured
    alone -- segment stitching does not corrupt a neighbour."""
    bases = _bases()
    solo = {r[0]: fcf.gmm.measure(_standalone_mp(r), bases[r[0]], full=True)
            for r in _ROWS}
    basket = fcf.gmm.measure(_basket_mp(), _basket_basis(), full=True)
    for i, r in enumerate(_ROWS):
        product = r[0]
        assert basket.bel[i] == pytest.approx(solo[product].bel[0], rel=1e-7), product
        assert basket.csm[i] == pytest.approx(solo[product].csm[0], rel=1e-7), product


def test_basket_fast_equals_full():
    """The fused fast path agrees with the trajectory path on the whole mix."""
    mp, basis = _basket_mp(), _basket_basis()
    full = fcf.gmm.measure(mp, basis, full=True)
    fast = fcf.gmm.measure(mp, basis, full=False)
    assert fast.bel == pytest.approx(full.bel, rel=1e-7)
    assert fast.csm == pytest.approx(full.csm, rel=1e-7)
    assert fast.ra  == pytest.approx(full.ra,  rel=1e-7)
