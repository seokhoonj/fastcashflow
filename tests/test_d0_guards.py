"""D0 -- the measurement time-basis discriminator and its consumer guards.

An in-force result's headline is an as-of valuation-date figure while its
trajectory fields stay on the inception axis (and its CSM is carried, not
derivable from column-0 FCF). Before D0, feeding such a result to the
inception-axis consumers silently produced wrong numbers -- the pinned trap
below showed group() re-flooring the carried CSM away with no warning. D0
stamps ``measurement_basis`` on every measurement and a single predicate
(:func:`fastcashflow._measurement_basis._require_inception`) rejects
non-inception results from group / group_of_contracts / roll_forward /
report / transition / plot_*; ``write_measurement`` stays allowed but adds
marker columns so the file output is distinguishable too.
"""
import matplotlib

matplotlib.use("Agg")

import dataclasses

import numpy as np
import polars as pl
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, BasisRouter, CalculationMethod, CoverageRate, ExpenseItem,
    InforceState, ModelPoints,
)
from fastcashflow.engine import _measure_inforce_fast, _measure_inforce_full
from fastcashflow._vfa import (
    CSM_BASIS_CARRY_ONLY, CSM_BASIS_INITIAL, CSM_BASIS_PARAGRAPH_45,
    CSM_BASIS_PROJECTED_RUNOFF, Measurement,
)

CM = {"DEATH": CalculationMethod.DEATH}

# A few guard tests call the deprecated carry bridge (measure_inforce); silence
# only its own deprecation notice.
pytestmark = pytest.mark.filterwarnings(
    "ignore:reinsurance.measure_inforce:DeprecationWarning")


def _flat_rate(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


def _basis(**kw):
    return Basis(
        mortality_annual=_flat_rate(0.012), lapse_annual=_flat_rate(0.05),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat_rate(0.012)),), **kw,
    )


def _carry_book(n=3, full=True):
    """The pinned trap fixture: a 3-MP settlement-carry book (prior CSM on
    rows 0 and 2). Before D0, group() on this returned CSM 470,511 where the
    carry headline summed to 441,345 -- silently wrong (session-measured;
    re-derived here as engine output, the guard now makes it unreachable)."""
    ids = np.array([f"P{i}" for i in range(n)])
    mp = ModelPoints(
        issue_age=np.full(n, 40), premium=np.full(n, 100.0),
        term_months=np.full(n, 24), benefits={"DEATH": np.full(n, 1e6)},
        count=np.full(n, 1000.0), elapsed_months=np.full(n, 12),
        mp_id=ids, product=np.full(n, "A"), calculation_methods=CM,
    )
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(n, 12, dtype=np.int64),
        count=np.full(n, 1000.0),
        prior_csm=np.array([3e5, 0.0, 2e5])[:n],
        lock_in_rate=0.03,
    )
    m = fcf.gmm.measure_inforce(mp, state, _basis(), period_months=12,
                                full=full)
    return mp, state, m


# ---------------------------------------------------------------------------
# stamping
# ---------------------------------------------------------------------------

def test_inception_measurement_is_tagged_inception():
    mp = ModelPoints.single(40, 50_000.0, 120, benefits={"DEATH": 1e8},
                            calculation_methods=CM)
    assert fcf.gmm.measure(mp, _basis()).measurement_basis == "inception"
    assert fcf.gmm.measure(mp, _basis(), full=False).measurement_basis == "inception"


def test_measure_inforce_is_tagged_settlement_carry():
    _, _, m_full = _carry_book(full=True)
    _, _, m_fast = _carry_book(full=False)
    assert m_full.measurement_basis == "settlement_carry"
    assert m_fast.measurement_basis == "settlement_carry"


def test_segmented_measure_inforce_keeps_the_tag():
    """The stitch helper is shared with the new-business segmented path and
    constructs a default measurement -- the in-force route must re-tag."""
    n = 2
    ids = np.array(["A0", "B0"])
    mp = ModelPoints(
        issue_age=np.full(n, 40), premium=np.full(n, 100.0),
        term_months=np.full(n, 24), benefits={"DEATH": np.full(n, 1e6)},
        count=np.full(n, 10.0), elapsed_months=np.full(n, 12), mp_id=ids,
        calculation_methods=CM, product=np.array(["A", "B"]),
        channel=np.array(["GA", "GA"]),
    )
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(n, 12, dtype=np.int64),
        count=np.full(n, 10.0), prior_csm=np.array([1e4, 0.0]),
        lock_in_rate=0.03,
    )
    router = BasisRouter({("A", "GA"): _basis(), ("B", "GA"): _basis()})
    for full in (True, False):
        m = fcf.gmm.measure_inforce(mp, state, router, period_months=12,
                                    full=full)
        assert m.measurement_basis == "settlement_carry"


def test_hypothetical_mode_is_tagged():
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([24]), benefits={"DEATH": np.array([1e6])},
        count=np.array([1.0]), elapsed_months=np.array([12]),
        calculation_methods=CM,
    )
    assert _measure_inforce_fast(mp, _basis()).measurement_basis == "hypothetical"
    assert _measure_inforce_full(mp, _basis()).measurement_basis == "hypothetical"


def test_paa_and_reinsurance_inforce_are_tagged():
    ids = np.array(["P0"])
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([1200.0]),
        term_months=np.array([12]), benefits={"DEATH": np.array([1e6])},
        count=np.array([1.0]), elapsed_months=np.array([6]), mp_id=ids,
        calculation_methods=CM,
    )
    state = InforceState(
        mp_id=ids, elapsed_months=np.array([6], dtype=np.int64),
        count=np.array([1.0]), prior_csm=np.array([0.0]), lock_in_rate=0.03,
    )
    for full in (True, False):
        assert (fcf.paa.measure_inforce(mp, state, _basis(), full=full)
                .measurement_basis == "settlement_carry")
        assert (fcf.reinsurance.measure_inforce(
                    mp, state, _basis(),
                    treaty=fcf.reinsurance.QuotaShare(cession=0.4),
                    period_months=6, full=full)
                .measurement_basis == "settlement_carry")


def test_vfa_measurement_basis_is_derived_from_csm_basis():
    z = np.zeros(1)
    expected = {CSM_BASIS_INITIAL: "inception",
                CSM_BASIS_PROJECTED_RUNOFF: "inception",
                CSM_BASIS_CARRY_ONLY: "settlement_carry",
                CSM_BASIS_PARAGRAPH_45: "settlement"}
    for csm_basis, want in expected.items():
        m = Measurement(bel=z, ra=z, csm=z, variable_fee=z, time_value=z,
                           loss_component=z, csm_basis=csm_basis)
        assert m.measurement_basis == want


# ---------------------------------------------------------------------------
# the guard matrix
# ---------------------------------------------------------------------------

def _gmm_consumers(basis):
    return [
        ("group", lambda m: fcf.group(m, "product")),
        ("group_of_contracts", lambda m: fcf.group_of_contracts(m)),
        ("roll_forward", lambda m: fcf.roll_forward(m, 12)),
        ("report", lambda m: fcf.report(m)),
        ("transition",
         lambda m: fcf.transition(m, np.zeros(m.bel.shape[0]))),
        # non-dispatch consumers enumerated explicitly (plots guard inside
        # each registered arm; transition is a plain function)
        ("plot_liability", lambda m: fcf.plot_liability(m)),
        ("plot_cashflows", lambda m: fcf.plot_cashflows(m)),
        ("plot_csm_runoff", lambda m: fcf.plot_csm_runoff(m)),
        ("plot_risk_adjustment", lambda m: fcf.plot_risk_adjustment(m, basis)),
    ]


def test_guard_matrix_rejects_settlement_carry_gmm():
    basis = _basis()
    _, _, m = _carry_book(full=True)
    for name, call in _gmm_consumers(basis):
        with pytest.raises(ValueError, match="measurement_basis"):
            call(m)


def test_guard_matrix_rejects_hypothetical_gmm():
    basis = _basis()
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([100.0]),
        term_months=np.array([24]), benefits={"DEATH": np.array([1e6])},
        count=np.array([1.0]), elapsed_months=np.array([12]),
        product=np.array(["A"]), calculation_methods=CM,
    )
    m = _measure_inforce_full(mp, basis)
    for name, call in _gmm_consumers(basis):
        with pytest.raises(ValueError, match="measurement_basis"):
            call(m)


def test_guards_reject_paa_and_reinsurance_carry():
    ids = np.array(["P0"])
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([1200.0]),
        term_months=np.array([12]), benefits={"DEATH": np.array([1e6])},
        count=np.array([1.0]), elapsed_months=np.array([6]), mp_id=ids,
        product=np.array(["A"]), calculation_methods=CM,
    )
    state = InforceState(
        mp_id=ids, elapsed_months=np.array([6], dtype=np.int64),
        count=np.array([1.0]), prior_csm=np.array([0.0]), lock_in_rate=0.03,
    )
    paa = fcf.paa.measure_inforce(mp, state, _basis(), full=True)
    reins = fcf.reinsurance.measure_inforce(
        mp, state, _basis(), treaty=fcf.reinsurance.QuotaShare(cession=0.4),
        period_months=6, full=True)
    for m in (paa, reins):
        with pytest.raises(ValueError, match="measurement_basis"):
            fcf.group(m, "product")
        with pytest.raises(ValueError, match="measurement_basis"):
            fcf.group_of_contracts(m)
        with pytest.raises(ValueError, match="measurement_basis"):
            fcf.roll_forward(m, 6)
        with pytest.raises(ValueError, match="measurement_basis"):
            fcf.report(m)
        with pytest.raises(ValueError, match="measurement_basis"):
            fcf.plot_liability(m)
        with pytest.raises(ValueError, match="measurement_basis"):
            fcf.plot_cashflows(m)
    # PAA has no CSM chart / RA fan by design (TypeError there) -- the basis
    # guard applies to the reinsurance arms only.
    with pytest.raises(ValueError, match="measurement_basis"):
        fcf.plot_csm_runoff(reins)
    with pytest.raises(ValueError, match="measurement_basis"):
        fcf.plot_risk_adjustment(reins, _basis())


def test_vfa_carry_only_is_rejected_by_plots():
    """plot_risk_adjustment reads the headline only, so a carry-only VFA
    result would reach it without the trajectory check -- every VFA plot arm
    guards on csm_basis (Codex review P1)."""
    z = np.ones(1)
    carry = Measurement(bel=z, ra=z, csm=z, variable_fee=z, time_value=z,
                           loss_component=z, csm_basis=CSM_BASIS_CARRY_ONLY)
    for call in (lambda m: fcf.plot_liability(m),
                 lambda m: fcf.plot_cashflows(m),
                 lambda m: fcf.plot_csm_runoff(m),
                 lambda m: fcf.plot_risk_adjustment(m, _basis())):
        with pytest.raises(ValueError, match="carry_only"):
            call(carry)


def test_write_measurement_marks_paa_and_reinsurance_inforce(tmp_path):
    ids = np.array(["P0"])
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([1200.0]),
        term_months=np.array([12]), benefits={"DEATH": np.array([1e6])},
        count=np.array([1.0]), elapsed_months=np.array([6]), mp_id=ids,
        calculation_methods=CM,
    )
    state = InforceState(
        mp_id=ids, elapsed_months=np.array([6], dtype=np.int64),
        count=np.array([1.0]), prior_csm=np.array([0.0]), lock_in_rate=0.03,
    )
    paa = fcf.paa.measure_inforce(mp, state, _basis(), full=False)
    reins = fcf.reinsurance.measure_inforce(
        mp, state, _basis(), treaty=fcf.reinsurance.QuotaShare(cession=0.4),
        period_months=6, full=False)
    for name, m in (("paa", paa), ("reins", reins)):
        out = tmp_path / f"{name}.parquet"
        fcf.write_measurement(m, out)
        df = pl.read_parquet(out)
        assert df["measurement_basis"].to_list() == ["settlement_carry"]
        assert df["elapsed_months"].to_list() == [6]


def test_the_pinned_trap_is_now_unreachable():
    """The original silently-wrong path: group() on a settlement-carry book
    re-floored at inception (CSM 470,511 vs the carry headline sum 441,345
    in the session fixture). It must now raise, loudly, pointing at settle."""
    _, _, m = _carry_book(full=True)
    with pytest.raises(ValueError, match="settle"):
        fcf.group(m, "product")


def test_inception_paths_still_flow_end_to_end():
    """The default path is untouched: measure -> group / roll_forward /
    report all run, and the dataclass carries the new field with its
    default (the asdict-level targeted assertion)."""
    mp = ModelPoints.single(40, 50_000.0, 120, benefits={"DEATH": 1e8},
                            calculation_methods=CM)
    basis = Basis(
        mortality_annual=_flat_rate(0.005), lapse_annual=_flat_rate(0.05),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        morbidity_cv=0.10, expense_inflation=0.02,
        expense_items=(ExpenseItem("maintenance", "gamma_fixed", 1_000.0),),
        coverages=(CoverageRate("DEATH", _flat_rate(0.005)),),
    )
    m = fcf.gmm.measure(mp, basis, full=True)
    field_names = {f.name for f in dataclasses.fields(m)}
    assert "measurement_basis" in field_names
    assert m.measurement_basis == "inception"
    # asdict-level targeted assertion: the discriminator is a stored field
    # with its default, not a property or a derived attribute.
    assert dataclasses.asdict(m)["measurement_basis"] == "inception"
    fcf.roll_forward(m, 12)
    fcf.report(m)


# ---------------------------------------------------------------------------
# write_measurement markers
# ---------------------------------------------------------------------------

def test_write_measurement_marks_inforce_output(tmp_path):
    _, _, m = _carry_book(full=False)
    out = tmp_path / "carry.parquet"
    fcf.write_measurement(m, out)
    df = pl.read_parquet(out)
    assert df["measurement_basis"].to_list() == ["settlement_carry"] * 3
    assert df["elapsed_months"].to_list() == [12, 12, 12]


def test_write_measurement_inception_schema_is_unchanged(tmp_path):
    mp = ModelPoints.single(40, 50_000.0, 120, benefits={"DEATH": 1e8},
                            calculation_methods=CM)
    m = fcf.gmm.measure(mp, _basis(), full=False)
    out = tmp_path / "new.parquet"
    fcf.write_measurement(m, out)
    df = pl.read_parquet(out)
    assert "measurement_basis" not in df.columns
    assert "elapsed_months" not in df.columns


# ---------------------------------------------------------------------------
# entry seals
# ---------------------------------------------------------------------------

def test_measure_inforce_rejects_a_mixed_model_router():
    ids = np.array(["A0", "B0"])
    mp = ModelPoints(
        issue_age=np.full(2, 40), premium=np.full(2, 100.0),
        term_months=np.full(2, 24), benefits={"DEATH": np.full(2, 1e6)},
        count=np.full(2, 1.0), elapsed_months=np.full(2, 12), mp_id=ids,
        calculation_methods=CM, product=np.array(["A", "B"]),
        channel=np.array(["GA", "GA"]),
    )
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(2, 12, dtype=np.int64),
        count=np.full(2, 1.0), prior_csm=np.zeros(2), lock_in_rate=0.03,
    )
    router = BasisRouter({("A", "GA"): _basis(), ("B", "GA"): _basis()},
                         measurement_models={("B", "GA"): "PAA"})
    with pytest.raises(ValueError, match="portfolio"):
        fcf.gmm.measure_inforce(mp, state, router)


def test_paa_group_full_false_message_names_full_true():
    mp = ModelPoints.single(40, 1200.0, 12, benefits={"DEATH": 1e6},
                            calculation_methods=CM)
    m = fcf.paa.measure(mp, _basis(), full=False)
    with pytest.raises(ValueError, match=r"full=True"):
        fcf.group(m, np.array(["g"]))
