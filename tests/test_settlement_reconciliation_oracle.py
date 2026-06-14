"""Phase-0 refactor test net: the settlement reconciliation surface oracle.

The four settlement Movement / Reconciliation / Aggregate families and their
reconcile / write dispatch are exercised only incidentally inside the per-feature
settle test modules (test_gmm_settle*, test_vfa_settle*, ...). Nothing pins the
SHARED invariants the reporting-layer single-source (the `_LINE_META` line spine
+ to_frame + write_reconciliation) will rest on. This module is that pin -- it
must be green on today's code BEFORE any spine / serializer refactor.

Two oracles here:

* the line spine is one source -- ``set(_X_SETTLEMENT_LINES)`` equals the
  per-MP array (``FloatArray``) fields of the matching settlement Movement
  dataclass, in EVERY family. This is what lets the writers (and to_frame) be
  driven from the tuples; if a future edit adds a movement line but forgets the
  tuple (or vice versa) this fails at collection time.
* the display-negation SIGN convention is absolute, not merely round-trip
  consistent. ``reconcile`` stores release / reversed / paid / revenue lines
  NEGATED so opening + every row foots to closing; an independent sign oracle
  catches a uniform-flip that ``reconcile(aggregate)==reconcile(per-MP)`` would
  mask. (The sign + identity oracles that need a constructed movement are added
  alongside the per-model settle fixtures; this file pins the structural spine,
  which needs no projection.)
"""
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, CoverageRate, ExpenseItem, InforceState, ModelPoints)
from fastcashflow.movement import (
    GMMSettlementMovement, PAASettlementMovement,
    ReinsuranceSettlementMovement, VFASettlementMovement,
    _GMM_SETTLEMENT_LINES, _PAA_SETTLEMENT_LINES,
    _REINSURANCE_SETTLEMENT_LINES, _VFA_SETTLEMENT_LINES,
)
from conftest import PATTERNS, make_death_basis


def _reconciled(mv, agg):
    """(per-MP movement, aggregate, reconcile(per-MP), reconcile(aggregate))."""
    return mv, agg, fcf.reconcile([mv])[0], fcf.reconcile(agg)


@pytest.fixture
def gmm_settlement():
    # onerous (prior loss component) + settlement pattern + discount, so the
    # bel/ra releases, loss-component amortisation and claims_paid are all nonzero.
    basis = make_death_basis(
        mortality_q=0.02, lapse_q=0.03, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10,
        settlement_pattern=np.array([0.6, 0.4]))
    surv = fcf.gmm.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([100.0]),
                    term_months=np.array([36]), benefits={0: np.array([1e6])},
                    count=np.array([1.0]), calculation_methods=PATTERNS),
        basis, full=True).cashflows.inforce[0]
    eo, p, scale = 12, 12, 1000.0
    ec = eo + p
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([36]), benefits={0: np.array([1e6])},
        count=np.array([scale * surv[ec]]), elapsed_months=np.array([ec]),
        mp_id=np.array(["P0"]), product=np.array(["A"]),
        calculation_methods=PATTERNS)
    st = InforceState(
        mp_id=np.array(["P0"]), elapsed_months=np.array([ec]),
        count=np.array([scale * surv[ec]]), prior_csm=np.array([0.0]),
        lock_in_rate=0.03, prior_count=np.array([scale * surv[eo]]),
        prior_loss_component=np.array([200_000.0]))
    mv = fcf.gmm.settle(mp, st, basis, period_months=12)
    agg = fcf.gmm.settle_aggregate(mp, st, basis, period_months=12)
    return _reconciled(mv, agg)


@pytest.fixture
def paa_settlement():
    basis = make_death_basis(
        mortality_q=0.02, lapse_q=0.0, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10,
        settlement_pattern=np.array([0.6, 0.4]))
    surv = fcf.paa.measure(
        ModelPoints(issue_age=np.array([40]), premium=np.array([60.0]),
                    term_months=np.array([12]), premium_term_months=np.array([1]),
                    benefits={0: np.array([6000.0])}, count=np.array([1.0]),
                    calculation_methods=PATTERNS),
        basis, full=True).cashflows.inforce[0]
    eo, ec = 3, 6
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([60.0]),
        term_months=np.array([12]), premium_term_months=np.array([1]),
        benefits={0: np.array([6000.0])}, count=np.array([surv[ec]]),
        elapsed_months=np.array([ec]), mp_id=np.array(["PA0"]),
        product=np.array(["ACC"]), calculation_methods=PATTERNS)
    st = InforceState(
        mp_id=np.array(["PA0"]), elapsed_months=np.array([ec]),
        count=np.array([surv[ec]]), prior_csm=np.array([0.0]),
        lock_in_rate=0.0, prior_count=np.array([surv[eo]]))
    mv = fcf.paa.settle(mp, st, basis, period_months=3)
    agg = fcf.paa.settle_aggregate(mp, st, basis, period_months=3)
    return _reconciled(mv, agg)


@pytest.fixture
def reinsurance_settlement():
    basis = make_death_basis(
        mortality_q=0.002, lapse_q=0.005, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.10)
    treaty = fcf.reinsurance.QuotaShare(0.4)
    unit = ModelPoints.single(40, 400_000.0, 240, benefits={0: 1e8},
                              calculation_methods=PATTERNS)
    m = fcf.reinsurance.measure(unit, basis, treaty=treaty)
    surv = m.cashflows.inforce[0]
    eo, ec, scale = 24, 36, 1000.0
    csm_seed = float(m.csm_path[0, eo])
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([400_000.0]),
        term_months=np.array([240]), benefits={0: np.array([1e8])},
        count=np.array([scale * surv[ec]]), elapsed_months=np.array([ec]),
        mp_id=np.array(["R0"]), calculation_methods=PATTERNS)
    st = InforceState(
        mp_id=np.array(["R0"]), elapsed_months=np.array([ec]),
        count=np.array([scale * surv[ec]]), prior_csm=np.array([csm_seed * scale]),
        lock_in_rate=0.03, prior_count=np.array([scale * surv[eo]]))
    mv = fcf.reinsurance.settle(mp, st, basis, treaty=treaty, period_months=12)
    agg = fcf.reinsurance.settle_aggregate(mp, st, basis, treaty=treaty,
                                           period_months=12)
    return _reconciled(mv, agg)


@pytest.fixture
def vfa_settlement():
    death_fn = lambda s, ia, d: np.full(s.shape, 0.012)
    basis = Basis(
        mortality_annual=death_fn,
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=0.05, ra_confidence=0.75, mortality_cv=0.0,
        expense_cv=0.10, investment_return=0.05, fund_fee=0.015,
        expense_items=(ExpenseItem("maintenance", "gamma_fixed", 1_000.0),),
        settlement_pattern=np.array([0.6, 0.4]),
        coverages=(CoverageRate("DEATH", death_fn),))
    mp0 = ModelPoints.single(40, 100.0, 24, account_value=1e6,
                             minimum_crediting_rate=0.08)
    m0 = fcf.vfa.measure(mp0, basis)
    inforce = m0.cashflows.inforce
    eo, p = 6, 6
    ec = eo + p
    r_m = (1.05) ** (1.0 / 12.0) - 1.0
    f_m = (1.015) ** (1.0 / 12.0) - 1.0
    growth = (1.0 + r_m) * (1.0 - f_m)
    av_open = 1e6 * growth ** eo
    av_close = av_open * growth ** p
    boundary = np.asarray(mp0.contract_boundary_months)
    pad = np.concatenate([inforce, np.zeros((1, 1))], axis=1)
    count_close = pad[np.arange(1), np.minimum(ec, boundary)]
    mp = replace(mp0, mp_id=np.array(["P0"]),
                 elapsed_months=np.full(1, ec, dtype=np.int64), count=count_close)
    st = InforceState(
        mp_id=np.array(["P0"]), elapsed_months=np.full(1, ec, dtype=np.int64),
        count=count_close, prior_csm=m0.csm_path[np.arange(1), eo],
        lock_in_rate=0.0, account_value=np.array([av_close]),
        prior_count=inforce[np.arange(1), eo],
        prior_account_value=np.array([av_open]),
        prior_loss_component=np.array([20_000.0]))
    mv = fcf.vfa.settle(mp, st, basis, period_months=6)
    agg = fcf.vfa.settle_aggregate(mp, st, basis, period_months=6)
    return _reconciled(mv, agg)


# (lines tuple, movement class) for each of the four settlement families.
_FAMILIES = (
    ("gmm", _GMM_SETTLEMENT_LINES, GMMSettlementMovement),
    ("vfa", _VFA_SETTLEMENT_LINES, VFASettlementMovement),
    ("reinsurance", _REINSURANCE_SETTLEMENT_LINES, ReinsuranceSettlementMovement),
    ("paa", _PAA_SETTLEMENT_LINES, PAASettlementMovement),
)


def _float_array_fields(cls):
    """The per-MP array lines of a settlement Movement dataclass -- the fields
    annotated FloatArray (movement.py uses ``from __future__ import annotations``
    so the annotation is the string 'FloatArray'). The scalar / reference fields
    (period_months, lock_in_rate, model_points, measurement_basis, revenue_basis)
    carry their own rules and are NOT line-spine entries."""
    return {name for name, f in cls.__dataclass_fields__.items()
            if str(f.type) == "FloatArray"}


def test_settlement_lines_tuple_equals_movement_float_array_fields():
    """The line spine is one source: every _X_SETTLEMENT_LINES tuple equals the
    set of FloatArray fields on its Movement dataclass. This is the invariant
    that lets the writers and to_frame be driven from the tuples (refactor
    Delta 1); a drift between tuple and dataclass fails here, at collection."""
    for key, lines, cls in _FAMILIES:
        tuple_set = set(lines)
        field_set = _float_array_fields(cls)
        assert len(lines) == len(tuple_set), f"{key}: _SETTLEMENT_LINES has duplicates"
        missing = field_set - tuple_set
        extra = tuple_set - field_set
        assert not missing, f"{key}: FloatArray fields not in the lines tuple: {sorted(missing)}"
        assert not extra, f"{key}: lines-tuple entries that are not FloatArray fields: {sorted(extra)}"


def test_settlement_lines_are_ordered_and_nonempty():
    """Each family declares a non-empty, hashable, string line spine (the
    ordering is the canonical display / serialization order the writers and the
    _LINE_META registry will key on)."""
    for key, lines, _cls in _FAMILIES:
        assert lines, f"{key}: empty _SETTLEMENT_LINES"
        assert all(isinstance(n, str) for n in lines), f"{key}: non-str line name"


# ---------------------------------------------------------------------------
# the display-negation SIGN oracle + the aggregate==per-MP identity
# ---------------------------------------------------------------------------
# The run-off / draw-down lines are stored NEGATED in the reconciliation so that
# opening + every row foots to closing. These sets are the lines each
# _reconcile_*_settlement body wraps in float(-m.x.sum()); pinning recon.x ==
# -mv.x.sum() (not merely reconcile(agg)==reconcile(per-MP)) catches a uniform
# negate-flip that the round-trip identity would mask.
_NEGATED = {
    "gmm": {"bel_release", "ra_release", "loss_component_amortised",
            "loss_component_reversed", "csm_release", "claims_paid"},
    "vfa": {"bel_release", "ra_release", "loss_component_amortised",
            "loss_component_reversed", "csm_release", "claims_paid"},
    "reinsurance": {"bel_release", "ra_release", "csm_release",
                    "loss_recovery_reversed"},
    "paa": {"revenue", "loss_component_reversed", "claims_paid"},
}


def _assert_sign_convention(model, mv, recon):
    """Every reconciliation float field that mirrors a movement FloatArray line
    equals +/- that line's portfolio sum -- minus for the draw-down lines, plus
    for the build-up lines. Independent of the reconcile code's own negation."""
    mv_lines = _float_array_fields(type(mv))
    negated = _NEGATED[model]
    checked = 0
    for name in type(recon).__dataclass_fields__:
        val = getattr(recon, name)
        if name in mv_lines and isinstance(val, float):
            sign = -1.0 if name in negated else 1.0
            np.testing.assert_allclose(
                val, sign * float(getattr(mv, name).sum()),
                rtol=1e-9, atol=1e-6,
                err_msg=f"{model}.{name}: wrong display sign/magnitude")
            checked += 1
    assert checked >= 4, f"{model}: oracle only checked {checked} lines (too few)"


def _assert_some_negated_nonzero(model, mv):
    """Guard against a vacuous sign test: at least two draw-down lines must be
    materially non-zero on the fixture."""
    nz = sum(1 for n in _NEGATED[model]
             if hasattr(mv, n) and abs(float(getattr(mv, n).sum())) > 1.0)
    assert nz >= 2, f"{model}: fixture exercises only {nz} non-zero negated lines"


def test_gmm_settlement_reconciliation_signs_and_identity(gmm_settlement):
    mv, agg, recon, recon_agg = gmm_settlement
    _assert_some_negated_nonzero("gmm", mv)
    _assert_sign_convention("gmm", mv, recon)
    _assert_reconcile_aggregate_matches(recon, recon_agg)


def test_vfa_settlement_reconciliation_signs_and_identity(vfa_settlement):
    mv, agg, recon, recon_agg = vfa_settlement
    _assert_some_negated_nonzero("vfa", mv)
    _assert_sign_convention("vfa", mv, recon)
    _assert_reconcile_aggregate_matches(recon, recon_agg)


def test_paa_settlement_reconciliation_signs_and_identity(paa_settlement):
    mv, agg, recon, recon_agg = paa_settlement
    _assert_some_negated_nonzero("paa", mv)
    _assert_sign_convention("paa", mv, recon)
    _assert_reconcile_aggregate_matches(recon, recon_agg)


def test_reinsurance_settlement_reconciliation_signs_and_identity(reinsurance_settlement):
    mv, agg, recon, recon_agg = reinsurance_settlement
    _assert_some_negated_nonzero("reinsurance", mv)
    _assert_sign_convention("reinsurance", mv, recon)
    _assert_reconcile_aggregate_matches(recon, recon_agg)


def _assert_reconcile_aggregate_matches(recon, recon_agg):
    """reconcile(aggregate) reproduces reconcile(per-MP) fieldwise -- the
    bounded-memory path is numerically identical to the per-MP path."""
    for name in type(recon).__dataclass_fields__:
        a, b = getattr(recon, name), getattr(recon_agg, name)
        if isinstance(a, float):
            np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-6, err_msg=name)
        else:
            assert b == a, f"{name}: {b!r} != {a!r}"


# -- write_measurement output schema characterization -------------------------
# The exact column order write_measurement emits for each settlement family.
# Pinned so the writer-collapse refactor (driving the columns from
# _X_SETTLEMENT_LINES) stays byte-identical, and so a later line addition is a
# deliberate, visible schema change rather than a silent drift.
import os
import tempfile

import polars as pl


def _written_columns(mv):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.csv")
        fcf.write_measurement(mv, path)
        return pl.read_csv(path).columns


_GMM_WRITE_COLUMNS = [
    "bel_opening", "bel_interest", "bel_release", "bel_experience", "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_experience_unlocking",
    "csm_premium_experience", "csm_investment_experience",
    "claims_experience", "expense_experience", "finance_wedge",
    "premium_experience_revenue", "csm_release", "csm_closing",
    "loss_component_opening", "loss_component_finance", "loss_component_amortised",
    "loss_component_reversed", "loss_component_recognised", "loss_component_closing",
    "coverage_units_provided", "coverage_units_future",
    "lic_opening", "claims_incurred", "lic_finance", "claims_paid", "lic_closing",
    "lock_in_rate", "measurement_basis", "elapsed_months", "count",
]
_PAA_WRITE_COLUMNS = [
    "lrc_opening", "premiums", "revenue", "lrc_experience", "lrc_closing",
    "loss_component_opening", "loss_component_recognised", "loss_component_reversed",
    "loss_component_closing", "lic_opening", "claims_incurred", "lic_finance",
    "claims_paid", "lic_closing", "claims_experience", "expense_experience",
    "revenue_basis", "measurement_basis", "elapsed_months", "count",
]
_REINSURANCE_WRITE_COLUMNS = [
    "bel_opening", "bel_interest", "bel_release", "bel_experience", "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_experience_unlocking", "finance_wedge",
    "csm_release", "csm_closing", "loss_recovery_opening", "loss_recovery_recognised",
    "loss_recovery_reversed", "loss_recovery_closing",
    "coverage_units_provided", "coverage_units_future",
    "lock_in_rate", "measurement_basis", "elapsed_months", "count",
]
_VFA_WRITE_COLUMNS = [
    "bel_opening", "bel_interest", "bel_release", "bel_experience", "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_fv_share", "csm_future_service",
    "csm_premium_experience", "premium_experience_revenue", "csm_investment_experience",
    "claims_experience", "expense_experience", "csm_release", "csm_closing",
    "loss_component_opening", "loss_component_finance", "loss_component_amortised",
    "loss_component_reversed", "loss_component_recognised", "loss_component_closing",
    "variable_fee_closing", "account_value_closing",
    "coverage_units_provided", "coverage_units_future",
    "lic_opening", "claims_incurred", "lic_finance", "claims_paid", "lic_closing",
    "lock_in_rate", "measurement_basis", "elapsed_months", "count",
]


def test_gmm_write_measurement_column_order(gmm_settlement):
    assert _written_columns(gmm_settlement[0]) == _GMM_WRITE_COLUMNS


def test_paa_write_measurement_column_order(paa_settlement):
    assert _written_columns(paa_settlement[0]) == _PAA_WRITE_COLUMNS


def test_reinsurance_write_measurement_column_order(reinsurance_settlement):
    assert _written_columns(reinsurance_settlement[0]) == _REINSURANCE_WRITE_COLUMNS


def test_vfa_write_measurement_column_order(vfa_settlement):
    assert _written_columns(vfa_settlement[0]) == _VFA_WRITE_COLUMNS
