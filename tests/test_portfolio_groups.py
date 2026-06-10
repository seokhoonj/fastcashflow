"""Contract skeleton for the per-group portfolio aggregate -- P-5c.

The **scalable form of group_of_contracts**: the IFRS 17 unit-of-account
aggregation (portfolio x annual cohort x profitability) computed in bounded
memory, so it works where holding the per-model-point measure(full=True) would
OOM. Unlike measure_aggregate (which sums each contract's already-floored CSM),
this **re-floors on the group's fulfilment cash flows** -- max(0, -sum FCF) per
group, applied once on the fully-accumulated group, never per chunk.

THE PIVOTAL FACT this skeleton pins:
under any IFRS 17 SecParagraph16-compliant profitability split, a group never mixes
inception-FCF signs, so at INITIAL RECOGNITION the re-floor and the per-MP-floor
sum give the IDENTICAL headline -- measure_group_of_contracts totals EQUAL measure_aggregate
totals whenever the group of contracts respects SecParagraph16. The re-floor changes the number only
for a deliberately COARSER, sign-mixing grouping (legitimate within-group
mutualisation) or in SUBSEQUENT measurement (SecParagraph44, out of scope). So
measure_group_of_contracts's value at inception is STRUCTURAL (per-group rows for disclosure /
roll-forward / the SecParagraph44 foundation), not a different number.

This file is the contract, written before the implementation (skeleton-first,
Codex's order). It skips cleanly until the entry points exist, so the suite stays
green; the implementation then activates it unchanged.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, ModelPoints, CoverageRate, group, group_of_contracts, report,
    roll_forward, reconcile)
from fastcashflow.basis import BasisRouter
from fastcashflow.portfolio import (
    measure, measure_aggregate, PortfolioReport, PortfolioMovements,
    PortfolioReconciliation)

# Contract-first: the per-group entry points do not exist yet. Skip the whole
# module until they land, so committing this spec does not break the suite.
import fastcashflow.portfolio as _pf                          # noqa: E402
if not hasattr(_pf, "measure_group_of_contracts"):
    pytest.skip(
        "per-group aggregate (measure_group_of_contracts / measure_groups / PortfolioGroups) "
        "is a contract skeleton, not yet implemented",
        allow_module_level=True)

from fastcashflow.portfolio import (                          # noqa: E402
    measure_group_of_contracts, measure_groups, PortfolioGroups)


def _flat_basis(discount=0.05, investment_return=0.0):
    return Basis(
        mortality_annual=lambda s, ia, d: np.full(s.shape, 0.001),
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=discount, ra_confidence=0.75, mortality_cv=0.0,
        investment_return=investment_return, fund_fee=0.015,
        coverages=(CoverageRate("DEATH", lambda s, ia, d: np.full(s.shape, 0.001)),))


def _mixed_book():
    """A mixed book whose contracts span profitable and onerous, two cohorts.

    GMM rows 0 (profitable: high premium) and 1 (onerous: zero premium), PAA
    row 2, VFA row 3 -- each model present, two issue years.
    """
    router = BasisRouter(
        {("G", "GA"): _flat_basis(),
         ("P", "GA"): _flat_basis(),
         ("V", "GA"): _flat_basis(investment_return=0.04)},
        measurement_models={("P", "GA"): "PAA", ("V", "GA"): "VFA"})
    mp = ModelPoints(
        issue_age=np.full(4, 40),
        premium=np.array([5000.0, 0.0, 1200.0, 0.0]),       # row0 profitable, row1 onerous
        term_months=np.full(4, 60),
        benefits={0: np.full(4, 1e4)},
        account_value=np.array([0.0, 0.0, 0.0, 1e6]),
        product=np.array(["G", "G", "P", "V"]),
        channel=np.array(["GA", "GA", "GA", "GA"]),
        issue_date=np.array(["2026-02-01", "2027-02-01", "2026-02-01", "2026-02-01"],
                            dtype="datetime64[D]"))
    return mp, router


def _two_gmm_same_cohort():
    """Two GMM contracts in the SAME product AND SAME cohort -- one profitable,
    one onerous. So profitability is the ONLY axis that distinguishes them, which
    is what isolates the re-floor effect from the cohort axis."""
    router = BasisRouter({("G", "GA"): _flat_basis()})        # single GMM segment
    mp = ModelPoints(
        issue_age=np.full(2, 40),
        premium=np.array([5000.0, 0.0]),                      # profitable, onerous
        term_months=np.full(2, 60),
        benefits={0: np.full(2, 1e4)},
        product=np.array(["G", "G"]),
        channel=np.array(["GA", "GA"]),
        issue_date=np.array(["2026-02-01", "2026-02-01"],     # one cohort
                            dtype="datetime64[D]"))
    return mp, router


# ===========================================================================
# 1) MASTER invariant -- the scalable aggregate == the in-memory group_of_contracts
# ===========================================================================
def test_equals_in_memory_group_of_contracts_per_model():
    """For a book that fits in memory, measure_group_of_contracts per model equals running
    group_of_contracts on the per-MP full measurement -- byte-for-byte the same
    native grouped measurement. The anchor: the chunked aggregate is
    group_of_contracts made memory-bounded, nothing more, so it must reproduce the
    full grouped result -- not just the headline but the trajectories, the group
    sizes and the grouped cash flows that roll_forward / report consume."""
    mp, router = _mixed_book()
    pg = measure_group_of_contracts(mp, router, chunk_size=10_000)
    full = measure(mp, router, full=True)            # full=True per-MP, in memory

    for model in ("gmm", "paa", "vfa"):
        agg_m = getattr(pg, model)
        ref = group_of_contracts(getattr(full, model).measurement)
        assert np.array_equal(agg_m.group_labels, ref.group_labels)
        assert np.array_equal(agg_m.group_sizes, ref.group_sizes)
        if model == "paa":
            assert np.allclose(agg_m.lrc, ref.lrc)
            assert np.allclose(agg_m.lrc_path, ref.lrc_path)
            assert np.allclose(agg_m.lic, ref.lic)
        else:
            assert np.allclose(agg_m.bel, ref.bel)
            assert np.allclose(agg_m.csm, ref.csm)
            assert np.allclose(agg_m.bel_path, ref.bel_path)
            assert np.allclose(agg_m.ra_path, ref.ra_path)
            assert np.allclose(agg_m.csm_path, ref.csm_path)
        assert np.allclose(agg_m.loss_component, ref.loss_component)
        # the grouped cash flows must match too -- roll_forward / report read them
        assert np.allclose(agg_m.cashflows.inforce, ref.cashflows.inforce)


def test_ragged_terms_same_curve_group():
    """Two contracts on the SAME curve but DIFFERENT terms in ONE group of contracts, chunked one
    per block. Pins the two implementation risks the contract flags: the global
    horizon (the shorter contract's path adds into the leading slice of the longer
    horizon) and the per-group representative curve (the longest-horizon row may
    arrive in a later chunk). The grouped result must still equal the in-memory
    group_of_contracts."""
    router = BasisRouter({("G", "GA"): _flat_basis()})        # single curve
    mp = ModelPoints(
        issue_age=np.full(2, 40),
        premium=np.array([5000.0, 5200.0]),                   # both profitable -> one group of contracts
        term_months=np.array([36, 60]),                       # ragged, same curve
        benefits={0: np.full(2, 1e4)},
        product=np.array(["G", "G"]), channel=np.array(["GA", "GA"]),
        issue_date=np.array(["2026-02-01", "2026-02-01"], dtype="datetime64[D]"))
    pg = measure_group_of_contracts(mp, router, chunk_size=1)                # each ragged row its own block
    ref = group_of_contracts(measure(mp, router, full=True).gmm.measurement)
    assert pg.gmm.bel.shape[0] == 1                           # one group (same curve, same cohort, both profitable)
    assert np.allclose(pg.gmm.bel_path, ref.bel_path)
    assert np.allclose(pg.gmm.csm_path, ref.csm_path)
    assert np.allclose(pg.gmm.cashflows.inforce, ref.cashflows.inforce)


def test_two_segments_same_curve_late_representative():
    """The 2-D _per_group_bom SUCCESS path (contract risk #1). Two routing
    segments of the SAME product (so one group of contracts) on the SAME discount curve but
    different terms, the short-term segment routed/chunked FIRST and the long-term
    one LATER. Two segments make the stitched discount_bom 2-D, so the per-group
    representative must be the longest-horizon row -- which only arrives in a later
    chunk -- and same-curve different-term rows must reconcile, not be
    mis-rejected. chunk_size=1 forces the late arrival. (The 1-D ragged test above
    cannot exercise this: a single segment keeps discount_bom 1-D.)"""
    router = BasisRouter(
        {("G", "A"): _flat_basis(discount=0.04),       # short term, routed first
         ("G", "B"): _flat_basis(discount=0.04)},      # long term, SAME curve, later
        measurement_models={})                          # both GMM
    p = np.zeros(4, dtype=int)                          # force one profitability class
    mp = ModelPoints(
        issue_age=np.full(4, 40), premium=np.full(4, 5000.0),
        term_months=np.array([24, 24, 60, 60]),         # A short, B long
        benefits={0: np.full(4, 1e4)},
        product=np.full(4, "G"), channel=np.array(["A", "A", "B", "B"]),
        issue_date=np.array(["2026-02-01"] * 4, dtype="datetime64[D]"))
    pg = measure_group_of_contracts(mp, router, profitability=p, chunk_size=1)
    ref = group_of_contracts(
        measure(mp, router, full=True).gmm.measurement, profitability=p)
    assert pg.gmm.bel.shape[0] == 1                      # one group of contracts across both segments
    assert np.allclose(pg.gmm.bel_path, ref.bel_path)
    assert np.allclose(pg.gmm.csm_path, ref.csm_path)    # representative curve chosen right
    assert np.allclose(pg.gmm.cashflows.inforce, ref.cashflows.inforce)


# ===========================================================================
# 2) THE PIVOTAL FACT -- under the SecParagraph16 split, the re-floor headline
#    EQUALS measure_aggregate's per-MP-floor-sum headline (not a different number)
# ===========================================================================
def test_equals_measure_aggregate_under_secparagraph16_split():
    """A correctly-keyed group of contracts never mixes inception-FCF signs, so CSM(sum FCF) ==
    sum CSM(FCF). measure_group_of_contracts and measure_aggregate report the SAME totals at
    inception under the SecParagraph16 onerous split."""
    mp, router = _mixed_book()
    pg = measure_group_of_contracts(mp, router)
    agg = measure_aggregate(mp, router)
    # GMM total CSM over all its groups of contracts == the per-MP-floor-sum aggregate
    assert np.isclose(pg.gmm.csm.sum(), agg.gmm.csm)
    assert np.isclose(pg.gmm.loss_component.sum(), agg.gmm.loss_component)
    assert np.isclose(pg.loss_component_total(), agg.loss_component_total())


# ===========================================================================
# 3) ...but the re-floor DOES differ for a coarse, sign-mixing grouping
#    (this is the only thing that distinguishes the two at inception)
# ===========================================================================
def test_refloors_on_group_fcf_not_per_mp_sum():
    """measure_groups(by="product") puts a profitable and an onerous GMM contract
    (same product, same cohort) in ONE group with no profitability axis, so the
    floor nets them -- the group CSM is max(0, -(FCF_profit + FCF_onerous)),
    strictly less than summing each contract's floored CSM. This is the re-floor;
    measure_aggregate cannot show it."""
    mp, router = _two_gmm_same_cohort()
    grouped = measure_groups(mp, router, by="product")        # coarse: mixes signs
    agg = measure_aggregate(mp, router)
    assert grouped.gmm.bel.shape[0] == 1                      # both contracts in one group
    # netting the profitable contract's surplus before the floor lowers the CSM
    assert grouped.gmm.csm.sum() < agg.gmm.csm


def test_nets_within_a_group_not_across():
    """Splitting by profitability (the SecParagraph16 axis, derived internally by
    measure_group_of_contracts) stops the netting -- the onerous contract stands alone, restoring
    the per-MP-floor total; the coarse product-only grouping absorbs its loss into
    the profitable contract's surplus."""
    mp, router = _two_gmm_same_cohort()
    coarse = measure_groups(mp, router, by="product")         # signs mixed -> netted
    split = measure_group_of_contracts(mp, router)                           # adds the onerous axis
    # the onerous contract isolated by profitability keeps its full loss
    assert np.isclose(split.gmm.loss_component.sum(),
                      measure_aggregate(mp, router).gmm.loss_component)
    # mixing absorbs that loss into the profitable contract's surplus
    assert coarse.gmm.loss_component.sum() < split.gmm.loss_component.sum()


# ===========================================================================
# 4) chunk-invariance even when a group of contracts spans chunks -- floor ONCE on the
#    accumulated group, never per chunk (the core correctness pin)
# ===========================================================================
def test_floors_once_on_accumulated_group_across_chunks():
    """chunk_size=1 puts every contract in its own block, so every group of contracts spans many
    blocks. The result must equal the single-block computation: the floor is
    applied once on the fully-accumulated group FCF, not per chunk."""
    mp, router = _mixed_book()
    a = measure_groups(mp, router, by="product", chunk_size=1)
    b = measure_groups(mp, router, by="product", chunk_size=10_000)
    assert np.array_equal(a.gmm.group_labels, b.gmm.group_labels)
    assert np.allclose(a.gmm.csm, b.gmm.csm)
    assert np.allclose(a.gmm.bel, b.gmm.bel)
    assert np.allclose(a.gmm.loss_component, b.gmm.loss_component)


# ===========================================================================
# 5) cohort -- reject missing issue_date (no silent single-cohort fallback),
#    never substitute issue_age/term (FLAGGED decision #3: diverges from preset)
# ===========================================================================
def test_rejects_missing_issue_date_no_age_substitution():
    """Unlike group_of_contracts (which silently falls back to one cohort), the
    scalable aggregate REJECTS a missing issue_date for the default cohort -- a
    silent collapse to one annual cohort would mutualise across cohorts (SecParagraph22)
    invisibly at settlement scale. issue_age / term_months are never used as a
    cohort substitute."""
    router = BasisRouter({("G", "GA"): _flat_basis()})
    mp = ModelPoints(                                  # no issue_date
        issue_age=np.array([40, 50]),                  # ages present, but not a cohort
        premium=np.array([5000.0, 0.0]), term_months=np.full(2, 60),
        benefits={0: np.full(2, 1e4)},
        product=np.array(["G", "G"]), channel=np.array(["GA", "GA"]))
    with pytest.raises(ValueError, match="issue_date|cohort"):
        measure_group_of_contracts(mp, router)


# ===========================================================================
# 6) profitability -- per-MP standalone inception-FCF sign, ONE rule across
#    models (via loss_component); not a per-model headline, not post-floor
# ===========================================================================
def test_profitability_is_per_mp_inception_fcf_sign():
    """The default profitability axis is each contract's standalone onerous test
    (loss_component > 0 at inception), the SAME field for GMM / PAA / VFA -- so no
    per-model definition can drift. The onerous GMM contract (row 1, zero premium)
    lands in an 'onerous' group of contracts; the profitable one does not."""
    mp, router = _mixed_book()
    pg = measure_group_of_contracts(mp, router)
    # row 1 is onerous standalone -> its group of contracts carries a positive loss component;
    # the profitable row's group of contracts carries none. Exactly one onerous GMM group of contracts here.
    assert (pg.gmm.loss_component > 0.0).sum() == 1
    # and it matches the per-MP standalone classification
    full = measure(mp, router, full=True)
    assert (full.gmm.measurement.loss_component > 0.0).sum() == 1


# ===========================================================================
# 7) container -- each model in its own slot, no cross-model group of contracts; summary /
#    loss_component_total mirror PortfolioAggregate
# ===========================================================================
def test_keeps_each_model_in_its_own_container():
    mp, router = _mixed_book()
    pg = measure_group_of_contracts(mp, router)
    assert isinstance(pg, PortfolioGroups)
    # native grouped measurement per slot (rows = that model's groups of contracts)
    assert pg.gmm is not None and pg.paa is not None and pg.vfa is not None
    # no flat field where a BEL and an LRC could be added
    assert not hasattr(pg, "bel")
    s = pg.summary()
    assert "loss_component_total" in s
    assert set(s["paa"]) == {"lrc", "loss_component"}     # LRC, never BEL/CSM
    assert "bel" in s["gmm"] and "bel" in s["vfa"]


def test_omits_absent_models():
    router = BasisRouter({("G", "GA"): _flat_basis()})
    mp = ModelPoints(
        issue_age=np.full(2, 40), premium=np.array([5000.0, 0.0]),
        term_months=np.full(2, 60), benefits={0: np.full(2, 1e4)},
        product=np.array(["G", "G"]), channel=np.array(["GA", "GA"]),
        issue_date=np.array(["2026-02-01", "2026-02-01"], dtype="datetime64[D]"))
    pg = measure_group_of_contracts(mp, router)
    assert pg.gmm is not None and pg.paa is None and pg.vfa is None
    assert set(pg.summary()) == {"loss_component_total", "gmm"}


# ===========================================================================
# 7b) in-memory composition -- group / group_of_contracts accept the container
#     (PortfolioMeasurement) and return a PortfolioGroups, like the leaf models
# ===========================================================================
def test_group_of_contracts_on_container_equals_scalable():
    """For a book that fits in memory, group_of_contracts on the measured
    PortfolioMeasurement equals the chunked measure_group_of_contracts on the same
    model points and router -- the in-memory composition and the scalable fused
    path are the same grouping, just different memory profiles.

    They agree only because this book sets issue_date: with the default cohort and
    no issue_date the two paths diverge by design -- the in-memory group_of_contracts
    falls back to a single annual cohort, while the scalable path rejects it (no
    silent cross-cohort mutualisation at settlement scale)."""
    mp, router = _mixed_book()
    composed = group_of_contracts(measure(mp, router, full=True))
    scalable = measure_group_of_contracts(mp, router)
    assert isinstance(composed, PortfolioGroups)
    for model in ("gmm", "paa", "vfa"):
        a, b = getattr(composed, model), getattr(scalable, model)
        assert np.array_equal(a.group_labels, b.group_labels)
        assert np.array_equal(a.group_sizes, b.group_sizes)
        assert np.allclose(a.loss_component, b.loss_component)
        if model == "paa":
            assert np.allclose(a.lrc, b.lrc)
        else:
            assert np.allclose(a.bel, b.bel)
            assert np.allclose(a.csm, b.csm)
            assert np.allclose(a.csm_path, b.csm_path)


def test_group_on_container_returns_portfolio_groups():
    """The general group(by=...) arm: a PortfolioMeasurement in -> PortfolioGroups
    out, each slot grouped on its own native measurement (no BEL/LRC pooling)."""
    mp, router = _mixed_book()
    pg = group(measure(mp, router, full=True), by="product")
    assert isinstance(pg, PortfolioGroups)
    assert pg.gmm is not None and pg.paa is not None and pg.vfa is not None
    # one product per model slot here -> one group row each
    assert pg.gmm.bel.shape[0] == 1 and pg.paa.lrc.shape[0] == 1


def test_group_on_container_subsets_precomputed_array_by_slot():
    """A precomputed (n_mp,) by-array is subset to each model slot's rows, so the
    GMM slot (rows 0,1) is split into two groups by a per-row label."""
    mp, router = _mixed_book()
    labels = np.array(["a", "b", "a", "a"])          # full-portfolio (n_mp,)
    pg = group(measure(mp, router, full=True), by=labels)
    assert pg.gmm.bel.shape[0] == 2                  # rows 0,1 -> labels a,b


def test_report_on_portfolio_containers():
    """fcf.report dispatches the portfolio containers -> PortfolioReport, one
    Report per model present, each equal to the leaf report on that slot. Works
    on both PortfolioMeasurement (per-MP) and PortfolioGroups (grouped)."""
    mp, router = _mixed_book()
    pm = measure(mp, router, full=True)
    rep = report(pm)
    assert isinstance(rep, PortfolioReport)
    assert rep.gmm is not None and rep.paa is not None and rep.vfa is not None
    assert np.allclose(rep.gmm.insurance_revenue,
                       report(pm.gmm.measurement).insurance_revenue)
    assert np.allclose(rep.paa.insurance_revenue,
                       report(pm.paa.measurement).insurance_revenue)
    # also accepts the grouped container
    pg = measure_group_of_contracts(mp, router)
    rep2 = report(pg)
    assert isinstance(rep2, PortfolioReport)
    assert np.allclose(rep2.gmm.insurance_revenue, report(pg.gmm).insurance_revenue)


def test_roll_forward_and_reconcile_on_portfolio_containers():
    """roll_forward(container) -> PortfolioMovements (per-model movement lists);
    reconcile(PortfolioMovements) -> PortfolioReconciliation. Each slot equals the
    leaf roll_forward / reconcile on that slot, and a plain list still reconciles
    through the base dispatch."""
    mp, router = _mixed_book()
    pm = measure(mp, router, full=True)
    mv = roll_forward(pm, period_months=12)
    assert isinstance(mv, PortfolioMovements)
    assert mv.gmm is not None and mv.paa is not None and mv.vfa is not None
    leaf_gmm = roll_forward(pm.gmm.measurement, period_months=12)
    assert len(mv.gmm) == len(leaf_gmm)

    rec = reconcile(mv)
    assert isinstance(rec, PortfolioReconciliation)
    assert rec.gmm is not None and len(rec.gmm) == len(reconcile(leaf_gmm))
    # the base (list) dispatch is unchanged
    assert isinstance(reconcile(leaf_gmm), list)


def test_roll_forward_container_rejects_gmm_only_options():
    """The revision / experience options need a single GMM measurement (and a
    matching revised book), so the container rejects them -- roll the gmm slot
    directly for those."""
    mp, router = _mixed_book()
    pm = measure(mp, router, full=True)
    with pytest.raises(ValueError, match="revision / experience|single GMM"):
        roll_forward(pm, revised=pm)


# ===========================================================================
# 8) guards
# ===========================================================================
def test_rejects_non_positive_chunk_size():
    mp, router = _mixed_book()
    with pytest.raises(ValueError, match="chunk_size"):
        measure_group_of_contracts(mp, router, chunk_size=0)


def test_requires_a_basis_router():
    """A single Basis cannot route a mixed portfolio -- use fcf.group_of_contracts
    on a single-model measurement instead."""
    with pytest.raises(TypeError, match="BasisRouter"):
        measure_group_of_contracts(ModelPoints(
            issue_age=np.full(2, 40), premium=np.zeros(2), term_months=np.full(2, 60),
            benefits={0: np.full(2, 1e4)}), _flat_basis())


# ===========================================================================
# 9) discount-curve uniformity -- a group of contracts sits in one portfolio = one curve
# ===========================================================================
def test_rejects_mixed_discount_curves_within_a_group():
    """Two segments of the same product but different discount curves, forced into
    one group of contracts, must be rejected -- a group must sit in one basis (the same
    uniformity check group() enforces). Pins the incremental curve-reconciliation
    risk flagged in the contract."""
    router = BasisRouter(
        {("G", "A"): _flat_basis(discount=0.03),
         ("G", "B"): _flat_basis(discount=0.06)},           # same product, diff curve
        measurement_models={})                               # both GMM
    mp = ModelPoints(
        issue_age=np.full(2, 40), premium=np.array([5000.0, 5000.0]),
        term_months=np.full(2, 60), benefits={0: np.full(2, 1e4)},
        product=np.array(["G", "G"]), channel=np.array(["A", "B"]),
        issue_date=np.array(["2026-02-01", "2026-02-01"], dtype="datetime64[D]"))
    with pytest.raises(ValueError, match="discount curve|one portfolio|one basis"):
        measure_group_of_contracts(mp, router)        # product "G" groups both -> mixed curves


def test_curve_uses_live_horizon_not_contract_boundary():
    """The representative-curve choice must use each contract's LIVE horizon
    (cashflows.inforce > 0), exactly as the in-memory _per_group_bom -- not the
    contract boundary. A count=0 contract is never in force, so its discount curve
    is irrelevant even when its boundary is the longest. Here a longer-boundary,
    count=0 segment on a DIFFERENT curve shares the product-level group with a
    shorter live segment; a boundary-based choice would falsely reject (or adopt
    the dead curve), so this pins the live-horizon selection against
    group(by='product')."""
    router = BasisRouter(
        {("G", "A"): _flat_basis(discount=0.04),     # live rows
         ("G", "B"): _flat_basis(discount=0.09)},    # count=0, longer term, other curve
        measurement_models={})
    mp = ModelPoints(
        issue_age=np.full(4, 40), premium=np.full(4, 5000.0),
        term_months=np.array([36, 36, 60, 60]),      # B has the longer boundary
        benefits={0: np.full(4, 1e4)},
        count=np.array([1.0, 1.0, 0.0, 0.0]),        # B never in force
        product=np.full(4, "G"), channel=np.array(["A", "A", "B", "B"]),
        issue_date=np.array(["2026-02-01"] * 4, dtype="datetime64[D]"))
    grouped = measure_groups(mp, router, by="product")   # must NOT reject
    ref = group(measure(mp, router, full=True).gmm.measurement, by="product")
    assert grouped.gmm.bel.shape[0] == 1
    assert np.allclose(grouped.gmm.bel_path, ref.bel_path)
    assert np.allclose(grouped.gmm.csm_path, ref.csm_path)    # live curve, not the dead one
    assert np.allclose(grouped.gmm.cashflows.inforce, ref.cashflows.inforce)


def test_all_dead_group_keeps_first_rows_real_curve():
    """A group with no live row (every contract count=0) must keep its lowest-index
    contract's REAL discount curve, exactly as _per_group_bom (whose argmax returns
    the first row when all live horizons are -1) -- not a flat placeholder. The CSM
    is 0 either way, but the public discount_bom and the downstream report /
    roll-forward input must match group_of_contracts, so the master invariant holds
    in full."""
    router = BasisRouter({("G", "GA"): _flat_basis(discount=0.04)})
    mp = ModelPoints(
        issue_age=np.full(2, 40), premium=np.full(2, 5000.0),
        term_months=np.full(2, 60), benefits={0: np.full(2, 1e4)},
        count=np.zeros(2),                                # nobody in force
        product=np.array(["G", "G"]), channel=np.array(["GA", "GA"]),
        issue_date=np.array(["2026-02-01", "2026-02-01"], dtype="datetime64[D]"))
    pg = measure_group_of_contracts(mp, router)
    ref = group_of_contracts(measure(mp, router, full=True).gmm.measurement)
    assert pg.gmm.bel.shape[0] == 1
    assert np.allclose(pg.gmm.discount_bom, ref.discount_bom)   # real curve, not flat
    assert np.allclose(pg.gmm.csm, ref.csm)                     # 0 either way


def test_rejects_wrong_length_group_array():
    """A precomputed group-label array must be (n_mp,) -- a too-long array would
    silently drop its tail, a too-short one error obscurely deep inside. Reject it
    up front, as fcf.group does."""
    mp, router = _mixed_book()
    with pytest.raises(ValueError, match="one entry per model point|group ids"):
        measure_groups(mp, router, by=np.zeros(mp.n_mp + 1, dtype=int))


def test_rejects_wrong_length_profitability_array():
    """A precomputed profitability array must be (n_mp,) too -- same guard."""
    mp, router = _mixed_book()
    with pytest.raises(ValueError, match="one entry per model point|profitability"):
        measure_group_of_contracts(mp, router, profitability=np.zeros(mp.n_mp + 1, dtype=int))


def test_rejects_short_array_in_by_list_no_broadcast():
    """A short precomputed array inside a list ``by`` must be rejected before the
    join -- otherwise a length-1 array broadcasts in np.char.add and silently tags
    every row with one label (passing the final (n_mp,) check)."""
    mp, router = _mixed_book()
    with pytest.raises(ValueError, match="one entry per model point|group axis"):
        measure_groups(mp, router, by=["product", np.zeros(1, dtype=int)])


def test_vfa_two_segments_same_return_reconcile():
    """The VFA 2-D discount_bom SUCCESS path (contract risk #4). Two VFA routing
    segments with the SAME investment_return but ragged terms, grouped into one
    group of contracts. The bom.ndim == 2 branch of the VFA group must reconcile segments on the
    same underlying-items return (not mis-reject), with the longest-horizon
    representative arriving in a later chunk under chunk_size=1. Mirrors the GMM
    late-representative test for the VFA curve."""
    router = BasisRouter(
        {("V", "A"): _flat_basis(investment_return=0.04),   # short, routed first
         ("V", "B"): _flat_basis(investment_return=0.04)},  # long, SAME return, later
        measurement_models={("V", "A"): "VFA", ("V", "B"): "VFA"})
    p = np.zeros(4, dtype=int)                               # force one group of contracts
    mp = ModelPoints(
        issue_age=np.full(4, 40), premium=np.zeros(4),
        term_months=np.array([24, 24, 60, 60]),             # ragged, short first
        benefits={0: np.full(4, 1e4)},
        account_value=np.full(4, 1e6),
        product=np.full(4, "V"), channel=np.array(["A", "A", "B", "B"]),
        issue_date=np.array(["2026-02-01"] * 4, dtype="datetime64[D]"))
    pg = measure_group_of_contracts(mp, router, profitability=p, chunk_size=1)
    ref = group_of_contracts(
        measure(mp, router, full=True).vfa.measurement, profitability=p)
    assert pg.vfa.bel.shape[0] == 1                          # one group of contracts across both VFA segments
    assert np.allclose(pg.vfa.bel_path, ref.bel_path)
    assert np.allclose(pg.vfa.csm_path, ref.csm_path)        # 2-D return curve reconciled
    assert np.allclose(pg.vfa.cashflows.inforce, ref.cashflows.inforce)


def test_portfolio_trace_routes_each_row_to_its_model():
    """portfolio.trace renders a row with its segment's model tracer: a GMM row
    via the GMM tracer, a PAA row via the PAA tracer, a VFA row via the VFA
    tracer -- a non-GMM row is never traced as GMM."""
    import io

    mp, router = _mixed_book()                     # GMM rows 0,1; PAA row 2; VFA row 3

    def render(row):
        buf = io.StringIO()
        fcf.portfolio.trace(row, mp, router, file=buf)
        return buf.getvalue()

    gmm, paa, vfa = render(0), render(2), render(3)
    assert "PAA" not in gmm and "VFA" not in gmm   # GMM row -> GMM tracer (no model tag)
    assert "PAA" in paa                            # PAA row -> PAA tracer
    assert "VFA" in vfa                            # VFA row -> VFA tracer


def test_portfolio_trace_requires_router_and_valid_index():
    """portfolio.trace needs a BasisRouter (a routed book) and an in-range row."""
    mp, router = _mixed_book()
    with pytest.raises(TypeError, match="BasisRouter"):
        fcf.portfolio.trace(0, mp, router.resolve(("G", "GA")))   # a single Basis
    with pytest.raises(IndexError):
        fcf.portfolio.trace(99, mp, router)


def test_portfolio_trace_diff_routes_each_row_to_its_model():
    """portfolio.trace_diff renders a row's shock diff with its segment's model
    diff tracer: GMM row -> GMM diff, PAA row -> PAA diff, VFA row -> VFA diff."""
    import io
    from dataclasses import replace

    mp, router = _mixed_book()
    router_b = BasisRouter(
        {k: replace(router.resolve(k), mortality_cv=0.20) for k in router.segments},
        segment_axes=router.segment_axes,
        measurement_models={k: router.measurement_model_of(k)
                            for k in router.segments})

    def render(row):
        buf = io.StringIO()
        fcf.portfolio.trace_diff(row, mp, router, router_b, file=buf)
        return buf.getvalue()

    gmm, paa, vfa = render(0), render(2), render(3)
    assert gmm.startswith("diff mp[0]")            # GMM diff tracer (no model tag)
    assert "diff-paa" in paa                        # PAA diff tracer
    assert "diff-vfa" in vfa                         # VFA diff tracer


def test_portfolio_trace_diff_requires_routers():
    """Both bases must be a BasisRouter (a routed book)."""
    mp, router = _mixed_book()
    single = router.resolve(("G", "GA"))
    with pytest.raises(TypeError, match="BasisRouter"):
        fcf.portfolio.trace_diff(0, mp, single, router)
    with pytest.raises(TypeError, match="BasisRouter"):
        fcf.portfolio.trace_diff(0, mp, router, single)
