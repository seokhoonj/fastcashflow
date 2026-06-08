"""Contract skeleton for the per-GIC portfolio aggregate -- P-5c.

The **scalable form of group_of_contracts**: the IFRS 17 unit-of-account
aggregation (portfolio x annual cohort x profitability) computed in bounded
memory, so it works where holding the per-model-point measure(full=True) would
OOM. Unlike measure_aggregate (which sums each contract's already-floored CSM),
this **re-floors on the group's fulfilment cash flows** -- max(0, -sum FCF) per
group, applied once on the fully-accumulated group, never per chunk.

THE PIVOTAL FACT this skeleton pins (see dev/p5c-per-gic-aggregate-contract.md):
under any IFRS 17 SecParagraph16-compliant profitability split, a group never mixes
inception-FCF signs, so at INITIAL RECOGNITION the re-floor and the per-MP-floor
sum give the IDENTICAL headline -- measure_gic totals EQUAL measure_aggregate
totals whenever the GIC respects SecParagraph16. The re-floor changes the number only
for a deliberately COARSER, sign-mixing grouping (legitimate within-GIC
mutualisation) or in SUBSEQUENT measurement (SecParagraph44, out of scope). So
measure_gic's value at inception is STRUCTURAL (per-GIC rows for disclosure /
roll-forward / the SecParagraph44 foundation), not a different number.

This file is the contract, written before the implementation (skeleton-first,
Codex's order). It skips cleanly until the entry points exist, so the suite stays
green; the implementation then activates it unchanged.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, ModelPoints, CoverageRate, group, group_of_contracts
from fastcashflow.basis import BasisRouter
from fastcashflow.portfolio import measure, measure_aggregate

# Contract-first: the per-GIC entry points do not exist yet. Skip the whole
# module until they land, so committing this spec does not break the suite.
import fastcashflow.portfolio as _pf                          # noqa: E402
if not hasattr(_pf, "measure_gic"):
    pytest.skip(
        "per-GIC aggregate (measure_gic / measure_groups / PortfolioGroups) "
        "is a contract skeleton, not yet implemented",
        allow_module_level=True)

from fastcashflow.portfolio import (                          # noqa: E402
    measure_gic, measure_groups, PortfolioGroups)


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
def test_gic_equals_in_memory_group_of_contracts_per_model():
    """For a book that fits in memory, measure_gic per model equals running
    group_of_contracts on the per-MP full measurement -- byte-for-byte the same
    native grouped measurement. The anchor: the chunked aggregate is
    group_of_contracts made memory-bounded, nothing more, so it must reproduce the
    full grouped result -- not just the headline but the trajectories, the group
    sizes and the grouped cash flows that roll_forward / report consume."""
    mp, router = _mixed_book()
    pg = measure_gic(mp, router, chunk_size=10_000)
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


def test_gic_ragged_terms_same_curve_group():
    """Two contracts on the SAME curve but DIFFERENT terms in ONE GIC, chunked one
    per block. Pins the two implementation risks the contract flags: the global
    horizon (the shorter contract's path adds into the leading slice of the longer
    horizon) and the per-group representative curve (the longest-horizon row may
    arrive in a later chunk). The grouped result must still equal the in-memory
    group_of_contracts."""
    router = BasisRouter({("G", "GA"): _flat_basis()})        # single curve
    mp = ModelPoints(
        issue_age=np.full(2, 40),
        premium=np.array([5000.0, 5200.0]),                   # both profitable -> one GIC
        term_months=np.array([36, 60]),                       # ragged, same curve
        benefits={0: np.full(2, 1e4)},
        product=np.array(["G", "G"]), channel=np.array(["GA", "GA"]),
        issue_date=np.array(["2026-02-01", "2026-02-01"], dtype="datetime64[D]"))
    pg = measure_gic(mp, router, chunk_size=1)                # each ragged row its own block
    ref = group_of_contracts(measure(mp, router, full=True).gmm.measurement)
    assert pg.gmm.bel.shape[0] == 1                           # one group (same curve, same cohort, both profitable)
    assert np.allclose(pg.gmm.bel_path, ref.bel_path)
    assert np.allclose(pg.gmm.csm_path, ref.csm_path)
    assert np.allclose(pg.gmm.cashflows.inforce, ref.cashflows.inforce)


def test_gic_two_segments_same_curve_late_representative():
    """The 2-D _per_group_bom SUCCESS path (contract risk #1). Two routing
    segments of the SAME product (so one GIC) on the SAME discount curve but
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
    pg = measure_gic(mp, router, profitability=p, chunk_size=1)
    ref = group_of_contracts(
        measure(mp, router, full=True).gmm.measurement, profitability=p)
    assert pg.gmm.bel.shape[0] == 1                      # one GIC across both segments
    assert np.allclose(pg.gmm.bel_path, ref.bel_path)
    assert np.allclose(pg.gmm.csm_path, ref.csm_path)    # representative curve chosen right
    assert np.allclose(pg.gmm.cashflows.inforce, ref.cashflows.inforce)


# ===========================================================================
# 2) THE PIVOTAL FACT -- under the SecParagraph16 split, the re-floor headline
#    EQUALS measure_aggregate's per-MP-floor-sum headline (not a different number)
# ===========================================================================
def test_gic_equals_measure_aggregate_under_secparagraph16_split():
    """A correctly-keyed GIC never mixes inception-FCF signs, so CSM(sum FCF) ==
    sum CSM(FCF). measure_gic and measure_aggregate report the SAME totals at
    inception under the SecParagraph16 onerous split."""
    mp, router = _mixed_book()
    pg = measure_gic(mp, router)
    agg = measure_aggregate(mp, router)
    # GMM total CSM over all its GICs == the per-MP-floor-sum aggregate
    assert np.isclose(pg.gmm.csm.sum(), agg.gmm.csm)
    assert np.isclose(pg.gmm.loss_component.sum(), agg.gmm.loss_component)
    assert np.isclose(pg.loss_component_total(), agg.loss_component_total())


# ===========================================================================
# 3) ...but the re-floor DOES differ for a coarse, sign-mixing grouping
#    (this is the only thing that distinguishes the two at inception)
# ===========================================================================
def test_gic_refloors_on_group_fcf_not_per_mp_sum():
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


def test_gic_nets_within_a_group_not_across():
    """Splitting by profitability (the SecParagraph16 axis, derived internally by
    measure_gic) stops the netting -- the onerous contract stands alone, restoring
    the per-MP-floor total; the coarse product-only grouping absorbs its loss into
    the profitable contract's surplus."""
    mp, router = _two_gmm_same_cohort()
    coarse = measure_groups(mp, router, by="product")         # signs mixed -> netted
    split = measure_gic(mp, router)                           # adds the onerous axis
    # the onerous contract isolated by profitability keeps its full loss
    assert np.isclose(split.gmm.loss_component.sum(),
                      measure_aggregate(mp, router).gmm.loss_component)
    # mixing absorbs that loss into the profitable contract's surplus
    assert coarse.gmm.loss_component.sum() < split.gmm.loss_component.sum()


# ===========================================================================
# 4) chunk-invariance even when a GIC spans chunks -- floor ONCE on the
#    accumulated group, never per chunk (the core correctness pin)
# ===========================================================================
def test_gic_floors_once_on_accumulated_group_across_chunks():
    """chunk_size=1 puts every contract in its own block, so every GIC spans many
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
def test_gic_rejects_missing_issue_date_no_age_substitution():
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
        measure_gic(mp, router)


# ===========================================================================
# 6) profitability -- per-MP standalone inception-FCF sign, ONE rule across
#    models (via loss_component); not a per-model headline, not post-floor
# ===========================================================================
def test_gic_profitability_is_per_mp_inception_fcf_sign():
    """The default profitability axis is each contract's standalone onerous test
    (loss_component > 0 at inception), the SAME field for GMM / PAA / VFA -- so no
    per-model definition can drift. The onerous GMM contract (row 1, zero premium)
    lands in an 'onerous' GIC; the profitable one does not."""
    mp, router = _mixed_book()
    pg = measure_gic(mp, router)
    # row 1 is onerous standalone -> its GIC carries a positive loss component;
    # the profitable row's GIC carries none. Exactly one onerous GMM GIC here.
    assert (pg.gmm.loss_component > 0.0).sum() == 1
    # and it matches the per-MP standalone classification
    full = measure(mp, router, full=True)
    assert (full.gmm.measurement.loss_component > 0.0).sum() == 1


# ===========================================================================
# 7) container -- each model in its own slot, no cross-model GIC; summary /
#    loss_component_total mirror PortfolioAggregate
# ===========================================================================
def test_gic_keeps_each_model_in_its_own_container():
    mp, router = _mixed_book()
    pg = measure_gic(mp, router)
    assert isinstance(pg, PortfolioGroups)
    # native grouped measurement per slot (rows = that model's GICs)
    assert pg.gmm is not None and pg.paa is not None and pg.vfa is not None
    # no flat field where a BEL and an LRC could be added
    assert not hasattr(pg, "bel")
    s = pg.summary()
    assert "loss_component_total" in s
    assert set(s["paa"]) == {"lrc", "loss_component"}     # LRC, never BEL/CSM
    assert "bel" in s["gmm"] and "bel" in s["vfa"]


def test_gic_omits_absent_models():
    router = BasisRouter({("G", "GA"): _flat_basis()})
    mp = ModelPoints(
        issue_age=np.full(2, 40), premium=np.array([5000.0, 0.0]),
        term_months=np.full(2, 60), benefits={0: np.full(2, 1e4)},
        product=np.array(["G", "G"]), channel=np.array(["GA", "GA"]),
        issue_date=np.array(["2026-02-01", "2026-02-01"], dtype="datetime64[D]"))
    pg = measure_gic(mp, router)
    assert pg.gmm is not None and pg.paa is None and pg.vfa is None
    assert set(pg.summary()) == {"loss_component_total", "gmm"}


# ===========================================================================
# 8) guards
# ===========================================================================
def test_gic_rejects_non_positive_chunk_size():
    mp, router = _mixed_book()
    with pytest.raises(ValueError, match="chunk_size"):
        measure_gic(mp, router, chunk_size=0)


def test_gic_requires_a_basis_router():
    """A single Basis cannot route a mixed portfolio -- use fcf.group_of_contracts
    on a single-model measurement instead."""
    with pytest.raises(TypeError, match="BasisRouter"):
        measure_gic(ModelPoints(
            issue_age=np.full(2, 40), premium=np.zeros(2), term_months=np.full(2, 60),
            benefits={0: np.full(2, 1e4)}), _flat_basis())


# ===========================================================================
# 9) discount-curve uniformity -- a GIC sits in one portfolio = one curve
# ===========================================================================
def test_gic_rejects_mixed_discount_curves_within_a_group():
    """Two segments of the same product but different discount curves, forced into
    one GIC, must be rejected -- a group must sit in one basis (the same
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
        measure_gic(mp, router)        # product "G" groups both -> mixed curves


def test_gic_curve_uses_live_horizon_not_contract_boundary():
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


def test_gic_all_dead_group_keeps_first_rows_real_curve():
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
    pg = measure_gic(mp, router)
    ref = group_of_contracts(measure(mp, router, full=True).gmm.measurement)
    assert pg.gmm.bel.shape[0] == 1
    assert np.allclose(pg.gmm.discount_bom, ref.discount_bom)   # real curve, not flat
    assert np.allclose(pg.gmm.csm, ref.csm)                     # 0 either way


def test_gic_rejects_wrong_length_group_array():
    """A precomputed group-label array must be (n_mp,) -- a too-long array would
    silently drop its tail, a too-short one error obscurely deep inside. Reject it
    up front, as fcf.group does."""
    mp, router = _mixed_book()
    with pytest.raises(ValueError, match="one entry per model point|group ids"):
        measure_groups(mp, router, by=np.zeros(mp.n_mp + 1, dtype=int))


def test_gic_rejects_wrong_length_profitability_array():
    """A precomputed profitability array must be (n_mp,) too -- same guard."""
    mp, router = _mixed_book()
    with pytest.raises(ValueError, match="one entry per model point|profitability"):
        measure_gic(mp, router, profitability=np.zeros(mp.n_mp + 1, dtype=int))


def test_gic_vfa_two_segments_same_return_reconcile():
    """The VFA 2-D discount_bom SUCCESS path (contract risk #4). Two VFA routing
    segments with the SAME investment_return but ragged terms, grouped into one
    GIC. The bom.ndim == 2 branch of the VFA group must reconcile segments on the
    same underlying-items return (not mis-reject), with the longest-horizon
    representative arriving in a later chunk under chunk_size=1. Mirrors the GMM
    late-representative test for the VFA curve."""
    router = BasisRouter(
        {("V", "A"): _flat_basis(investment_return=0.04),   # short, routed first
         ("V", "B"): _flat_basis(investment_return=0.04)},  # long, SAME return, later
        measurement_models={("V", "A"): "VFA", ("V", "B"): "VFA"})
    p = np.zeros(4, dtype=int)                               # force one GIC
    mp = ModelPoints(
        issue_age=np.full(4, 40), premium=np.zeros(4),
        term_months=np.array([24, 24, 60, 60]),             # ragged, short first
        benefits={0: np.full(4, 1e4)},
        account_value=np.full(4, 1e6),
        product=np.full(4, "V"), channel=np.array(["A", "A", "B", "B"]),
        issue_date=np.array(["2026-02-01"] * 4, dtype="datetime64[D]"))
    pg = measure_gic(mp, router, profitability=p, chunk_size=1)
    ref = group_of_contracts(
        measure(mp, router, full=True).vfa.measurement, profitability=p)
    assert pg.vfa.bel.shape[0] == 1                          # one GIC across both VFA segments
    assert np.allclose(pg.vfa.bel_path, ref.bel_path)
    assert np.allclose(pg.vfa.csm_path, ref.csm_path)        # 2-D return curve reconciled
    assert np.allclose(pg.vfa.cashflows.inforce, ref.cashflows.inforce)
