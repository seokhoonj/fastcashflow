"""portfolio.settle_group_of_contracts -- VFA per-GoC settlement (skeleton).

Authoritative skeleton (settle-family pattern): written before the VFA
implementation and activated unchanged once it lands. Anchor facts from the
G-gate (dev/vfa-goc-settle-gate.md) and its handcalc
(dev/scratch_vfa_goc_settle_gate.py, re-run here through the public entry).

The gate's resolution (the flagged "k_obs re-summation"): VFA per-GoC settle
is a STRUCTURAL MIRROR of the GMM per-GoC settle. On the per-MP pre-floor
lines of ``vfa.settle``, the LINEAR lines are group-summed -- including
``csm_fv_share`` (45(b)) and ``csm_future_service`` (45(c)), each carrying its
own ``v_half``/``k_obs``, so the group fv_share is the SUM of the per-MP
fv_shares, NOT a re-derivation from a re-summed group account value. The
NON-LINEAR step -- the paragraph-48/50(b) algebra and the B119 release -- is
applied ONCE at group grain:

    x_g         = sum(csm_fv_share_i + csm_future_service_i) = -sum(bel_exp+ra_exp)
    csm_after_g = algebra(sum csm_open_i + sum accretion_i, x_g, sum lc_open_i)
    frac_g      = cu_provided_g / (cu_provided_g + cu_future_g)   (0 denom -> 0)
    csm_release_g = csm_after_g * frac_g

Heterogeneous rows mutualise inside the group CSM (per-MP floor sum != group
floor), exactly as in GMM. Scope: VFA-only book -> VFAGoCSettlement; a mixed
GMM+VFA book is rejected WHOLE (a group sits in one product, hence one model);
PAA rejected (no CSM/floor).
"""
import numpy as np
import pytest
from dataclasses import replace

import fastcashflow as fcf
from fastcashflow import (
    Basis, CalculationMethod, CoverageRate, ExpenseItem, InforceState,
    ModelPoints)
from fastcashflow.basis import BasisRouter

import fastcashflow.vfa as _vfa_ns
if not hasattr(_vfa_ns, "GoCSettlement"):
    pytest.skip(
        "VFA per-GoC settle (fcf.vfa.GoCSettlement) is a contract skeleton, not "
        "yet implemented (v1.1; activates this module unchanged once it lands)",
        allow_module_level=True)

from fastcashflow.portfolio import settle_group_of_contracts
from fastcashflow.vfa import GoCSettlement
from fastcashflow.numerics import _csm_loss_component_step


# ---------------------------------------------------------------------------
# VFA basis + routing fixtures
# ---------------------------------------------------------------------------
def make_vfa_basis(*, investment_return=0.05, fund_fee=0.015):
    death_fn = lambda s, ia, d: np.full(s.shape, 0.012)
    return Basis(
        mortality_annual=death_fn,
        lapse_annual=lambda s, ia, d: np.full(s.shape, 0.05),
        discount_annual=investment_return, ra_confidence=0.75,
        mortality_cv=0.0, expense_cv=0.10,
        investment_return=investment_return, fund_fee=fund_fee,
        expense_items=(ExpenseItem("maintenance", "gamma_fixed", 1_000.0),),
        coverages=(CoverageRate("DEATH", death_fn),))


VFA_BASIS = make_vfa_basis()
ROUTER = BasisRouter({("VA_A", "GA"): VFA_BASIS},
                     measurement_models={("VA_A", "GA"): "VFA"})


def _growth(b, mp):
    r_m = (1.0 + b.investment_return) ** (1.0 / 12.0) - 1.0
    f_m = (1.0 + b.fund_fee) ** (1.0 / 12.0) - 1.0
    return (1.0 + r_m) * (1.0 - f_m)


def vfa_book(*, em_open=6, period=6, mdb=(0.0, 2.0e6),
             av_factor=(1.10, 0.50), prior_csm=(600.0, 12.0), term=(24, 24)):
    """Two VFA MPs in one group (VA_A / GA / one cohort). Defaults reproduce
    the gate handcalc: MP0 favourable (+10% AV, no guarantee), MP1 ITM GMDB
    with a halving AV and a thin CSM -> onerous per-MP, mutualised in group."""
    n = len(mdb)
    em_close = em_open + period
    mp0 = ModelPoints(
        issue_age=np.full(n, 40), premium=np.zeros(n),
        term_months=np.asarray(term, dtype=np.int64),
        minimum_death_benefit=np.asarray(mdb, dtype=np.float64),
        account_value=np.full(n, 1.0e6),
        product=np.full(n, "VA_A"), channel=np.full(n, "GA"),
        issue_date=np.array(["2026-02-01"] * n, dtype="datetime64[D]"),
        benefits={"DEATH": np.zeros(n)}, count=np.ones(n),
        calculation_methods={"DEATH": CalculationMethod.DEATH})
    m0 = fcf.vfa.measure(mp0, VFA_BASIS)
    inforce = m0.cashflows.inforce
    rows = np.arange(n)
    g = _growth(VFA_BASIS, mp0)
    av_open = mp0.account_value * g ** em_open
    av_close = av_open * g ** period * np.asarray(av_factor, dtype=np.float64)
    boundary = np.asarray(mp0.contract_boundary_months)
    pad = np.concatenate([inforce, np.zeros((n, 1))], axis=1)
    count_open = inforce[rows, em_open]
    count_close = pad[rows, np.minimum(em_close, boundary)]
    ids = np.array([f"P{i}" for i in range(n)])
    mp = replace(mp0, mp_id=ids,
                 elapsed_months=np.full(n, em_close, dtype=np.int64),
                 count=count_close)
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(n, em_close, dtype=np.int64),
        count=count_close, prior_csm=np.asarray(prior_csm, dtype=np.float64),
        lock_in_rate=0.0, account_value=av_close, prior_count=count_open,
        prior_account_value=av_open)
    return mp, state


def settle_goc(mp, state, *, period=6, coverage_units="count", **kw):
    n = mp.n_mp
    kw.setdefault("profitability", np.zeros(n, dtype=np.int64))
    return settle_group_of_contracts(
        mp, state, ROUTER, period_months=period,
        coverage_units=coverage_units, **kw)


# ---------------------------------------------------------------------------
# Oracle: the group aggregation from per-MP VFASettlementMovement lines
# ---------------------------------------------------------------------------
_VFA_LINEAR = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_fv_share", "csm_future_service", "csm_opening", "csm_accretion",
    "variable_fee_closing", "account_value_closing", "loss_component_opening",
)
_VFA_NONLINEAR = (
    "csm_release", "csm_closing", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
)


def vfa_goc_oracle(mv, w):
    csm_open = float(mv.csm_opening.sum())
    accretion = float(mv.csm_accretion.sum())
    x = float(mv.csm_fv_share.sum()) + float(mv.csm_future_service.sum())
    lc_open = float(mv.loss_component_opening.sum())
    cu_p = float((w * mv.coverage_units_provided).sum())
    cu_f = float((w * mv.coverage_units_future).sum())
    csm_after, lc_rev, lc_rec, lc_close = _csm_loss_component_step(
        csm_open + accretion, x, lc_open)
    denom = cu_p + cu_f
    frac = cu_p / denom if denom > 0 else 0.0
    release = csm_after * frac
    return {"csm_release": release, "csm_closing": csm_after - release,
            "loss_component_reversed": lc_rev,
            "loss_component_recognised": lc_rec,
            "loss_component_closing": lc_close, "frac": frac}


def assert_group_matches_oracle(goc, mv, w, row=0):
    ref = vfa_goc_oracle(mv, w)
    for name in _VFA_LINEAR:
        np.testing.assert_allclose(
            getattr(goc, name)[row], float(getattr(mv, name).sum()),
            rtol=1e-11, atol=1e-9, err_msg=name)
    for name in ("coverage_units_provided", "coverage_units_future"):
        np.testing.assert_allclose(
            getattr(goc, name)[row], float((w * getattr(mv, name)).sum()),
            rtol=1e-11, atol=1e-9, err_msg=name)
    for name in _VFA_NONLINEAR:
        np.testing.assert_allclose(
            getattr(goc, name)[row], ref[name], rtol=1e-11, atol=1e-9,
            err_msg=name)


# ===========================================================================
# the gate's pinned mutualisation case
# ===========================================================================
def test_gate_mutualisation_case_pins_the_group_floor():
    mp, state = vfa_book()
    mv = fcf.vfa.settle(mp, state, VFA_BASIS, period_months=6)
    goc = settle_goc(mp, state, period=6)

    assert goc.csm_closing.shape == (1,)             # one group row
    assert_group_matches_oracle(goc, mv, w=np.ones(2))

    # the discriminating numbers (dev/scratch_vfa_goc_settle_gate.py)
    np.testing.assert_allclose(mv.csm_fv_share,
                               [1388.53197188, -6942.65985942], rtol=1e-6)
    np.testing.assert_allclose(mv.csm_closing,
                               [1314.30053031, 0.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mv.loss_component_closing,
                               [0.0, 12583.34023984], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(goc.csm_closing[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(goc.loss_component_closing[0], 10579.991222,
                               rtol=1e-6)
    # genuine mutualisation: per-MP floor sum != group floor
    assert abs(float(mv.csm_closing.sum()) - goc.csm_closing[0]) > 1.0
    assert abs(float(mv.loss_component_closing.sum())
               - goc.loss_component_closing[0]) > 1.0


# ===========================================================================
# k_obs re-summation (gate part b): group fv_share = sum of per-MP fv_shares
# ===========================================================================
def test_k_obs_resummation_group_fv_share_is_per_mp_sum():
    mp, state = vfa_book()
    mv = fcf.vfa.settle(mp, state, VFA_BASIS, period_months=6)
    goc = settle_goc(mp, state, period=6)
    np.testing.assert_allclose(
        goc.csm_fv_share[0], float(mv.csm_fv_share.sum()), rtol=1e-11)
    np.testing.assert_allclose(
        goc.csm_future_service[0], float(mv.csm_future_service.sum()),
        rtol=1e-11)
    # the two legs partition the total future-service change additively
    np.testing.assert_allclose(
        goc.csm_fv_share[0] + goc.csm_future_service[0],
        -float(mv.bel_experience.sum() + mv.ra_experience.sum()), rtol=1e-10)


# ===========================================================================
# account value / variable fee are group-summed echoes (linear)
# ===========================================================================
def test_account_value_and_fee_are_group_summed_echoes():
    mp, state = vfa_book()
    mv = fcf.vfa.settle(mp, state, VFA_BASIS, period_months=6)
    goc = settle_goc(mp, state, period=6)
    np.testing.assert_allclose(
        goc.account_value_closing[0], float(state.account_value.sum()),
        rtol=1e-11)
    np.testing.assert_allclose(
        goc.variable_fee_closing[0], float(mv.variable_fee_closing.sum()),
        rtol=1e-11)


# ===========================================================================
# on-track CSM-positive book: every line matches the oracle, release alive
# ===========================================================================
def test_on_track_positive_book_matches_oracle():
    mp, state = vfa_book(mdb=(0.0, 0.0), av_factor=(1.0, 1.0),
                         prior_csm=(600.0, 500.0))
    mv = fcf.vfa.settle(mp, state, VFA_BASIS, period_months=6)
    goc = settle_goc(mp, state, period=6)
    assert_group_matches_oracle(goc, mv, w=np.ones(2))
    assert goc.csm_closing[0] > 0.0
    assert goc.csm_release[0] > 0.0


# ===========================================================================
# weighted coverage units move the group B119 release fraction
# ===========================================================================
def test_weighted_units_change_release_fraction():
    # different terms -> different tail coverage-unit profiles, so weighting
    # the two rows differently genuinely moves the group B119 fraction
    mp, state = vfa_book(mdb=(0.0, 0.0), av_factor=(1.02, 0.98),
                         prior_csm=(600.0, 500.0), term=(24, 18))
    mv = fcf.vfa.settle(mp, state, VFA_BASIS, period_months=6)
    w = np.array([1.0, 5.0])
    g_cnt = settle_goc(mp, state, period=6, coverage_units="count")
    g_wt = settle_goc(mp, state, period=6, coverage_units=w)
    assert_group_matches_oracle(g_cnt, mv, w=np.ones(2))
    assert_group_matches_oracle(g_wt, mv, w=w)
    assert abs(g_cnt.csm_release[0] - g_wt.csm_release[0]) > 1e-6


# ===========================================================================
# closing_inputs chains: allocated group balances sum back to the group
# ===========================================================================
def test_closing_inputs_allocates_group_balances():
    mp, state = vfa_book(mdb=(0.0, 0.0), av_factor=(1.0, 1.0),
                         prior_csm=(600.0, 500.0))
    goc = settle_goc(mp, state, period=6)
    next_mp, next_state = goc.closing_inputs()
    np.testing.assert_allclose(float(next_state.prior_csm.sum()),
                               goc.csm_closing[0], rtol=1e-10)
    np.testing.assert_allclose(
        float(np.asarray(next_state.account_value).sum()),
        goc.account_value_closing[0], rtol=1e-10)


# ===========================================================================
# rejection seams: mixed GMM+VFA rejected whole; the result type is VFA
# ===========================================================================
def test_vfa_only_book_returns_vfa_goc_settlement():
    mp, state = vfa_book()
    goc = settle_goc(mp, state, period=6)
    assert isinstance(goc, GoCSettlement)


def test_mixed_gmm_vfa_book_rejected_whole():
    from conftest import make_death_basis
    gmm_basis = make_death_basis(mortality_q=0.002, lapse_q=0.005,
                                 discount_annual=0.05, ra_confidence=0.75,
                                 mortality_cv=0.10)
    mp, state = vfa_book()
    # relabel one row to a GMM-routed product
    prod = np.asarray(mp.product).copy(); prod[0] = "PROT_A"
    mp = replace(mp, product=prod)
    router = BasisRouter({("VA_A", "GA"): VFA_BASIS,
                          ("PROT_A", "GA"): gmm_basis},
                         measurement_models={("VA_A", "GA"): "VFA"})
    with pytest.raises(ValueError, match="mixed|single model|one model|GMM"):
        settle_group_of_contracts(
            mp, state, router, period_months=6, coverage_units="count",
            profitability=np.zeros(mp.n_mp, dtype=np.int64))
