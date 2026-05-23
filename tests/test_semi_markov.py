"""Hand-calculation validation of the Phase (c) semi-Markov path.

Tests are intentionally tiny -- one contract, a couple of months, simple
rates -- so each BEL can be derived by hand and matched to ``value()``.
"""
from __future__ import annotations

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.statemodel import StateModel, State, Transition


def _annual(monthly: float) -> float:
    return 1.0 - (1.0 - monthly) ** 12


def _cancer_reincidence_model(duration_max: int) -> StateModel:
    return StateModel(states=(
        State("healthy", premium=True, transitions=(
            Transition("mortality"),
            Transition("ci_incidence", to="post_first"),
            Transition("lapse"),
        )),
        State("post_first", duration_max=duration_max, transitions=(
            Transition("mortality"),
            Transition("ci_reincidence", to="post_second",
                       lump_sum=True, duration_dependent=True),
        )),
        State("post_second", transitions=(
            Transition("mortality"),
        )),
    ), seating=(0, 1, 2))


def _flat_assumptions(*, ci_reincidence_fn) -> fcf.Assumptions:
    return fcf.Assumptions(
        mortality_annual=lambda s, a, d: np.full(d.shape, _annual(0.001)),
        lapse_annual=lambda s, a, d: np.full(d.shape, 0.0),
        ci_incidence_annual=lambda s, a, d: np.full(d.shape, _annual(0.005)),
        ci_reincidence_annual=ci_reincidence_fn,
        discount_annual=0.0,
        expense_acquisition=0.0,
        expense_maintenance_annual=0.0,
        expense_inflation=0.0,
        ra_confidence=0.5,
        mortality_cv=0.0,
        state_model=_cancer_reincidence_model(12),
    )


def _single_contract(term_months: int, *, death_benefit: float = 10_000_000.0,
                     reincidence_benefit: float = 5_000_000.0) -> fcf.ModelPoints:
    return fcf.ModelPoints(
        issue_age=np.array([40], dtype=np.int64),
        death_benefit=np.array([death_benefit]),
        level_premium=np.array([0.0]),
        term_months=np.array([term_months], dtype=np.int64),
        disability_benefit=np.array([reincidence_benefit]),
    )


def test_one_month_only_death_claim():
    """Term = 1 month, reincidence rate = 0 everywhere: the BEL collapses to
    one month of pure death-claim cost. Hand calculation:

        in-force = 1.0 at t = 0
        claim rate per unit IF = mortality_monthly * death_benefit
                               = 0.001 * 10_000_000 = 10_000
        pc = 1.0 * 10_000 * dm  (dm = mid-month discount factor at t=0)

    With discount_annual = 0, dm = 1, so pc = 10_000.
    No other PV components fire (no premium, no annuity, no reincidence,
    no maturity), so bel = pc = 10_000.
    """
    asmp = _flat_assumptions(
        ci_reincidence_fn=lambda s, a, p, sd: np.zeros_like(sd, dtype=float),
    )
    v = fcf.value(_single_contract(1), asmp)
    assert np.isclose(v.bel[0], 10_000.0), v.bel[0]


def test_one_month_with_reincidence_in_exclusion():
    """At t = 0 only the healthy state has in-force; the reincidence
    transition operates on post_first occupancy which is still zero.
    Even with a nonzero reincidence rate the first-month BEL must equal
    the pure-death-claim value of the prior test.
    """
    asmp = _flat_assumptions(
        ci_reincidence_fn=lambda s, a, p, sd: np.full_like(sd, _annual(0.02),
                                                           dtype=float),
    )
    v = fcf.value(_single_contract(1), asmp)
    assert np.isclose(v.bel[0], 10_000.0), v.bel[0]


def test_two_month_first_diagnosis_no_reincidence():
    """Two months. Reincidence rate = 0 throughout, so post_first cohorts
    only drain via mortality. Hand calculation:

      t = 0: occ = {h:1.0, p1:0, p2:0}
              claim PV += 1.0 * 0.001 * 10M = 10_000

              edges:
                healthy -> post_first  prob = 0.999 * 0.005 = 0.004995
                healthy stays          prob = 0.999 * 0.995 = 0.994005
              after step:
                h          = 0.994005
                p1[tau=0]  = 0.004995

      t = 1: ift = 0.994005 + 0.004995 = 0.999
              claim PV += 0.999 * 0.001 * 10M = 9_990
              (mortality rides every state via the DEATH coverage)

    Total bel = 10_000 + 9_990 = 19_990.
    """
    asmp = _flat_assumptions(
        ci_reincidence_fn=lambda s, a, p, sd: np.zeros_like(sd, dtype=float),
    )
    v = fcf.value(_single_contract(2), asmp)
    assert np.isclose(v.bel[0], 19_990.0), v.bel[0]


def test_one_month_reincidence_active_via_seating():
    """Seat the contract directly on post_first (ss = 1, cohort 0) and place
    the reincidence rate outside its exclusion window. Term = 1 month.

      t = 0: occ = {h:0, p1[tau=0]:1.0, p2:0}
        Death-claim PV (mortality on the whole portfolio) = 1.0 * 0.001 * 10M
                                                          = 10_000.
        Reincidence rate at cohort 0 is set to monthly 0.02 (after the
        prior mortality is taken in competing-decrement order):
            flow = 0.999 * 0.02 = 0.01998
        Reincidence lump-sum PV = flow * reincidence_benefit * dm
                               = 0.01998 * 5_000_000 = 99_900.
      bel = 10_000 + 99_900 = 109_900.
    """
    # Reincidence rate = 0.02 monthly = _annual(0.02) annual, but only at
    # cohort 0; later cohorts unused in a one-month term.
    def ci_rein(s, a, p, sd):
        return np.full_like(sd, _annual(0.02), dtype=float)

    asmp = _flat_assumptions(ci_reincidence_fn=ci_rein)
    mp = fcf.ModelPoints(
        issue_age=np.array([40], dtype=np.int64),
        death_benefit=np.array([10_000_000.0]),
        level_premium=np.array([0.0]),
        term_months=np.array([1], dtype=np.int64),
        disability_benefit=np.array([5_000_000.0]),
        state=np.array([1], dtype=np.int64),    # seat on post_first
    )
    v = fcf.value(mp, asmp)
    assert np.isclose(v.bel[0], 109_900.0), v.bel[0]


def test_reincidence_rate_zero_in_exclusion_window():
    """Seat on post_first cohort 0. With a 12-month exclusion (rate = 0 for
    sd < 12), the first month must look identical to the case where the
    rate is zero everywhere.
    """
    def ci_rein_with_excl(s, a, p, sd):
        return np.where(sd < 12, 0.0, _annual(0.02))

    def ci_rein_all_zero(s, a, p, sd):
        return np.zeros_like(sd, dtype=float)

    mp = fcf.ModelPoints(
        issue_age=np.array([40], dtype=np.int64),
        death_benefit=np.array([10_000_000.0]),
        level_premium=np.array([0.0]),
        term_months=np.array([1], dtype=np.int64),
        disability_benefit=np.array([5_000_000.0]),
        state=np.array([1], dtype=np.int64),
    )
    v_excl = fcf.value(mp, _flat_assumptions(ci_reincidence_fn=ci_rein_with_excl))
    v_zero = fcf.value(mp, _flat_assumptions(ci_reincidence_fn=ci_rein_all_zero))
    assert np.isclose(v_excl.bel[0], v_zero.bel[0])
