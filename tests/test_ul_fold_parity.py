"""Universal-life fold -- the account roll lives in the shared kernels.

A universal-life contract carrying ``CoverageRate("DEATH", coi_annual,
funds_from_account=True, pays_account_balance=True)`` is measured through the
shared projection (``gmm.measure`` / ``vfa.measure``): the death leg is routed
through the recursive account roll, the account benefits stay in-band on
:class:`Cashflows` (``mortality_cf`` / ``surrender_cf`` / ``maturity_cf``), and the
account-state trajectory is exposed on the ``cashflows.account`` sidecar.

These tests pin the SELF-CONSISTENCY of that fold -- the fused fast path
(``full=False``) against the full roll-forward (``full=True``), the Step-4 routing
that decides which kernel a book runs, the Step-3.5 gate that rejects an account
book on the raw-consumer paths, and the sidecar population -- without an external
reference. (The hand-calc anchor lives in ``test_ul_account_value.py``.)
"""
from dataclasses import replace

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints


def _ul_basis():
    """The UL basis -- the DEATH coverage (rate = coi_annual) carries the
    account-chassis flags, so the shared projection folds the account roll."""
    coi = 0.0015
    return Basis(
        mortality_annual=0.004,
        lapse_annual=0.03,
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.1,
        investment_return=0.024,
        coi_annual=coi,
        premium_load=0.08,
        coverages=(
            CoverageRate("DEATH", coi, funds_from_account=True,
                         pays_account_balance=True),
        ),
    )


def _single_mp():
    return ModelPoints(
        issue_age=np.array([40.0]),
        premium=np.array([500_000.0]),
        term_months=np.array([36]),
        account_value=np.array([0.0]),
        minimum_death_benefit=np.array([80_000_000.0]),
        minimum_accumulation_benefit=np.array([0.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
        # The DEATH coverage's amount is unused (the account death reads the
        # account balance, not coverage_amount); set it to the face for clarity.
        benefits={"DEATH": np.array([80_000_000.0])},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )


def _two_mp():
    return ModelPoints(
        issue_age=np.array([40.0, 55.0]),
        premium=np.array([500_000.0, 300_000.0]),
        term_months=np.array([36, 24]),
        account_value=np.array([0.0, 1_000_000.0]),
        minimum_death_benefit=np.array([80_000_000.0, 50_000_000.0]),
        minimum_accumulation_benefit=np.array([0.0, 0.0]),
        minimum_crediting_rate=np.array([0.0, 0.01]),
        sex=np.array([0, 1]),
        benefits={"DEATH": np.array([80_000_000.0, 50_000_000.0])},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )


def _assert_self_consistent(make_mp):
    mp = make_mp()
    basis = _ul_basis()

    # full=False (the fused fast path) carries the account roll in the SCALAR
    # fused kernel itself (Step 4) -- an account book no longer routes to
    # _measure_full, it runs the account-aware scalar kernel directly. Confirm
    # the routing actually exercises the scalar fast path (requires_full is now
    # False for an account book; the account check was removed from it).
    from fastcashflow.engine import requires_full
    from fastcashflow._measurement.account import _portfolio_has_account
    assert _portfolio_has_account(mp, basis)
    assert not requires_full(mp, basis), (
        "Step 4: an account book must run the scalar fast path, not auto-route "
        "to the full kernel via requires_full")

    # The scalar fused kernel accumulates present values FORWARD (pv += cf x
    # discount factor), whereas _measure_full runs the BACKWARD roll-forward
    # recursion (bel[t] = cf x half[t] + bel[t+1] x full[t]). The two are
    # mathematically equal but differ in floating-point summation order, so the
    # fused path is numerically equal -- NOT bit-identical -- to the backward
    # roll, exactly as every other fused (full=False) book is to its full=True
    # counterpart. So this is np.allclose, not np.array_equal.
    got_h = fcf.gmm.measure(mp, basis, full=False)
    got_full = fcf.gmm.measure(mp, basis, full=True)
    for name in ("bel", "ra", "csm", "loss_component"):
        g = getattr(got_h, name)
        f = getattr(got_full, name)
        assert np.allclose(g, f), (
            f"{name} full=False (scalar) vs full=True (fold): "
            f"max abs delta {np.abs(g - f).max()}")


def test_ul_fold_self_consistent_single_policy():
    _assert_self_consistent(_single_mp)


def test_ul_fold_self_consistent_two_policy():
    _assert_self_consistent(_two_mp)


def test_vfa_ul_self_consistent():
    # Variable UL is the recursive account roll discounted at the underlying-items
    # return (the only thing the VFA model changes). The fused headline matches
    # the full roll, and a UL book has no asset-based fee / guarantee TVOG (v1).
    for make_mp in (_single_mp, _two_mp):
        mp = make_mp()
        basis = _ul_basis()
        got_full = fcf.vfa.measure(mp, basis, full=True)
        got_h = fcf.vfa.measure(mp, basis, full=False)
        for name in ("bel", "ra", "csm", "loss_component"):
            g = getattr(got_h, name)
            f = getattr(got_full, name)
            assert np.allclose(g, f), (
                f"{name} VFA full=False vs full=True: "
                f"max abs delta {np.abs(g - f).max()}")
        # A universal-life book has no asset-based fee / guarantee TVOG (v1).
        assert np.array_equal(got_full.variable_fee, np.zeros_like(got_full.bel))
        assert np.array_equal(got_full.time_value, np.zeros_like(got_full.bel))


def test_ul_settlement_pattern_rejected():
    # An account book settles its benefit at exit, not over a settlement
    # pattern; the measurement's settlement factor would mis-discount it (GMM
    # in-year rate, also wrong under VFA), so it is rejected on both paths.
    basis = Basis(
        mortality_annual=0.004, lapse_annual=0.03, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.1, investment_return=0.024,
        coi_annual=0.0015, premium_load=0.08,
        settlement_pattern=np.array([0.6, 0.4]),
        coverages=(CoverageRate("DEATH", 0.0015, funds_from_account=True,
                                pays_account_balance=True),))
    mp = _single_mp()
    with pytest.raises(NotImplementedError):
        fcf.gmm.measure(mp, basis, full=True)
    with pytest.raises(NotImplementedError):
        fcf.vfa.measure(mp, basis)


def test_vfa_ul_guarantee_time_value():
    # The UL guarantee time value is now computed: re-roll the account under the
    # return scenarios, price the GMDB/GMAB floors, mean cost less the central
    # intrinsic. Zero-volatility scenarios carry no time value; volatile ones do,
    # and the CSM absorbs it (FCF = BEL + RA + TVOG).
    mp, basis = _single_mp(), _ul_basis()
    n_time = int(np.asarray(mp.contract_boundary_months).max())
    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    zv = fcf.vfa.measure(mp, basis, return_scenarios=np.full((8, n_time), r_m))
    np.testing.assert_allclose(zv.time_value, 0.0, atol=1e-6)   # no volatility -> no TVOG
    rng = np.random.default_rng(0)
    m = fcf.vfa.measure(
        mp, basis, return_scenarios=rng.normal(r_m, 0.04, (64, n_time)))
    assert np.isfinite(m.time_value).all()
    np.testing.assert_allclose(                                 # CSM absorbs the TVOG
        m.csm, np.maximum(0.0, -(m.bel + m.ra + m.time_value)), rtol=1e-9)
    with pytest.raises(ValueError):                             # width != horizon
        fcf.vfa.measure(mp, basis, return_scenarios=np.zeros((4, n_time - 3)))


def test_ul_fold_account_sidecar_populated():
    # The folded projection exposes the account trajectory as a nested sidecar.
    got = fcf.gmm.measure(_two_mp(), _ul_basis(), full=True)
    acct = got.cashflows.account
    assert acct is not None
    n_mp, n_time = got.cashflows.mortality_cf.shape
    assert acct.av.shape == (n_mp, n_time + 1)
    assert acct.av_mid.shape == (n_mp, n_time)
    assert acct.coi.shape == (n_mp, n_time)
    assert acct.fund.shape == (n_mp, n_time + 1)


def test_callable_coi_rate_keeps_account_flags():
    # A callable COI rate triggers Basis.__post_init__'s rate-arity rebuild of
    # the coverage tuple. The account-chassis flags MUST survive that rebuild --
    # a rebuild that kept only (code, rate) silently dropped them, disabling UL
    # routing for every realistic (age-varying) COI rate. Regression guard.
    coi_fn = lambda s, a, d: np.full(np.asarray(a).shape, 0.0015)
    common = dict(
        mortality_annual=0.004, lapse_annual=0.03, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.1, investment_return=0.024,
        coi_annual=coi_fn, premium_load=0.08)
    folded = Basis(coverages=(CoverageRate(
        "DEATH", coi_fn, funds_from_account=True, pays_account_balance=True),),
        **common)
    # The flags survived the rate-arity rebuild -> the account roll is active.
    assert folded.coverages[0].funds_from_account is True
    assert folded.coverages[0].pays_account_balance is True
    m = fcf.gmm.measure(_single_mp(), folded, full=True)
    assert m.cashflows.account is not None


def test_account_coc_routes_full_false_to_full():
    # Step 4 routing: an account book with a cost-of-capital RA cannot run the
    # confidence-level-only fused kernel, so full=False auto-routes to the full
    # measurement -> the headline is bit-identical to full=True.
    coc = Basis(
        mortality_annual=0.004, lapse_annual=0.03, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.1, investment_return=0.024,
        coi_annual=0.0015, premium_load=0.08,
        ra_method="cost_of_capital", cost_of_capital_rate=0.06,
        coverages=(CoverageRate("DEATH", 0.0015, funds_from_account=True,
                                pays_account_balance=True),))
    mp = _single_mp()
    fast = fcf.gmm.measure(mp, coc, full=False)
    full = fcf.gmm.measure(mp, coc, full=True)
    for name in ("bel", "ra", "csm", "loss_component"):
        assert np.array_equal(getattr(fast, name), getattr(full, name))


def test_account_boundary_cut_routes_full_false_to_full():
    # Step 4 routing: a contract boundary shorter than the term pays the boundary
    # survivors a terminal surrender that the scalar fold does not handle, so a
    # boundary-cut account book routes full=False -> full (bit-identical headline).
    basis = _ul_basis()
    mp = ModelPoints(
        issue_age=np.array([40.0]),
        premium=np.array([500_000.0]),
        term_months=np.array([36]),
        contract_boundary_months=np.array([24]),   # boundary < term
        account_value=np.array([0.0]),
        minimum_death_benefit=np.array([80_000_000.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
        benefits={"DEATH": np.array([80_000_000.0])},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )
    fast = fcf.gmm.measure(mp, basis, full=False)
    full = fcf.gmm.measure(mp, basis, full=True)
    for name in ("bel", "ra", "csm", "loss_component"):
        assert np.array_equal(getattr(fast, name), getattr(full, name))


def test_account_book_gated_on_raw_consumers():
    # Step 3.5 -- paths that read the benefit cash flows raw (no account fund
    # netting) must REJECT a universal-life book rather than double-count it.
    mp = _two_mp()
    basis = _ul_basis()
    n_time = fcf.gmm.measure(mp, basis, full=True).bel_path.shape[1] - 1

    # The standalone vfa.tvog diagnostic is NOT gated: it values the CREDITED-RATE
    # floor for a universal-life book (a different guarantee from the GMDB/GMAB
    # account-value floors, whose time value comes through vfa.measure --
    # test_vfa_ul_guarantee_time_value; the credited-rate floor is
    # test_ul_credit_rate_tvog_*). _two_mp carries a per-MP-varying crediting rate
    # (0.0 vs 0.01), so it raises for THAT reason (scalar guarantee in v1), not an
    # account gate -- a uniform-rate UL book is measured (test below).
    with pytest.raises(NotImplementedError, match="uniform"):
        fcf.vfa.tvog(mp, basis,
                     np.tile(np.linspace(-0.01, 0.03, 8)[:, None], (1, n_time)))
    # reinsurance.measure is NOT gated: a universal-life book cedes the net amount
    # at risk (not the gross account death), netted before the cession (see
    # test_reinsurance_universal_life_cedes_nar).
    # gmm.stochastic is NOT gated: a universal-life book skips the raw fast kernel
    # and falls to the per-scenario measure() loop, which nets the account (see
    # test_stochastic_universal_life_distribution).
    # roll_forward is NOT gated: it reads only the account-netted bel / ra / csm
    # paths and the in-force count, never the raw benefit cash flows, so it
    # supports an account book (see
    # test_movement.test_roll_forward_universal_life_account_reconciles).


def test_stochastic_universal_life_distribution():
    # A universal-life account book runs through measure_stochastic via the
    # per-scenario measure() fallback (which routes the account book to the full
    # measurement and nets it), so the liability distribution under interest-rate
    # scenarios is produced -- finite and monotone (higher discount -> lower PV).
    mp, basis = _two_mp(), _ul_basis()
    r = fcf.gmm.stochastic(mp, basis, np.array([0.02, 0.03, 0.04, 0.05]))
    assert r.bel.shape == (4,) and np.isfinite(r.bel).all()
    np.testing.assert_array_less(np.diff(r.bel), 1e-6)   # decreasing in the rate


def test_reinsurance_universal_life_cedes_nar():
    # A universal-life account book reinsures the NET AMOUNT AT RISK, not the
    # gross account death benefit -- the account-value part of the death benefit
    # is the policyholder's deposit, not reinsured risk. The death claim is netted
    # (mortality_cf - deaths * av_mid) before the treaty cession.
    mp, basis = _two_mp(), _ul_basis()
    r = fcf.reinsurance.measure(mp, basis, treaty=fcf.samples.treaty())
    assert np.isfinite(np.atleast_1d(r.bel)).all()
    assert np.isfinite(np.atleast_1d(r.ra)).all()


# --- credited-rate floor TVOG for a universal-life book (standalone vfa.tvog) ---
#
# The minimum-crediting-rate guarantee credits max(return, floor) each month; the
# entity funds the shortfall, and that funded extra account value is paid out on
# the account exits. Unlike the GMDB / GMAB floors (a put on the account, valued
# through vfa.measure(return_scenarios)), the credited-rate floor is the account
# LIFT itself, so the account is re-rolled floored vs bare and the exit-payout
# difference is the cost. measure_tvog routes a universal-life book here.

def _credit_mp(g, term=24, av0=2_000_000.0, premium=400_000.0,
               face=60_000_000.0, gmab=0.0):
    return ModelPoints(
        issue_age=np.array([45.0]),
        premium=np.array([premium]),
        term_months=np.array([term]),
        account_value=np.array([av0]),
        minimum_death_benefit=np.array([face]),
        minimum_accumulation_benefit=np.array([0.0]),
        maturity_benefit=np.array([gmab]),
        minimum_crediting_rate=np.array([g]),
        sex=np.array([0]),
        benefits={"DEATH": np.array([face])},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
    )


def _flat(r_annual: float, n_time: int, rows: int = 1):
    r_m = (1.0 + r_annual) ** (1.0 / 12.0) - 1.0
    return np.full((rows, n_time), r_m)


def test_ul_credit_rate_tvog_zero_vol_and_intrinsic():
    # Zero volatility -> the time value vanishes (mean cost == central). When the
    # central flat return sits BELOW the crediting floor the floor bites even
    # deterministically (intrinsic > 0); above it the deterministic account is
    # never floored (intrinsic == 0), and only volatility creates a time value.
    basis = _ul_basis()                                  # investment_return = 0.024
    n_time = 24
    # central floors: r (0.024) < g (0.04)
    mp_lo = _credit_mp(g=0.04, term=n_time)
    res = fcf.vfa.tvog(mp_lo, basis, _flat(0.024, n_time, rows=8))
    assert res.intrinsic_value > 0.0
    np.testing.assert_allclose(res.time_value, 0.0, atol=1e-6)
    # central out-of-the-money: r (0.024) > g (0.0) -> no deterministic floor cost
    mp_hi = _credit_mp(g=0.0, term=n_time)
    res_hi = fcf.vfa.tvog(mp_hi, basis, _flat(0.024, n_time, rows=8))
    np.testing.assert_allclose(res_hi.intrinsic_value, 0.0, atol=1e-6)
    np.testing.assert_allclose(res_hi.time_value, 0.0, atol=1e-6)


def test_ul_credit_rate_tvog_hand_roll():
    # Correctness anchor: an INDEPENDENT dual roll of the account (floored at g vs
    # bare return), differenced on the death / surrender / maturity exits and
    # discounted at the path's own return, reproduces both the intrinsic (central
    # path) and a single scenario's guarantee cost to floating-point.
    from fastcashflow._measurement.account import _account_roll_inputs
    from fastcashflow.projection import project_cashflows
    from fastcashflow.tvog import credited_monthly_rate

    g, r = 0.03, 0.024
    basis = _ul_basis()
    mp = _credit_mp(g=g, term=24)
    n_time = int(np.asarray(mp.contract_boundary_months).max())
    r_m = (1.0 + r) ** (1.0 / 12.0) - 1.0

    proj = project_cashflows(mp, basis)
    deaths, inforce = proj.deaths, proj.inforce
    n_mp = inforce.shape[0]
    boundary_idx = mp.contract_boundary_months - 1
    within = (mp.term_months - 1) <= boundary_idx
    term_idx = np.where(within, mp.term_months - 1, boundary_idx)
    matsurv = np.where(within, proj.maturity_survivors, 0.0)
    pad = np.concatenate([inforce, np.zeros((n_mp, 1))], axis=1)
    surr = pad[:, :-1] - pad[:, 1:] - deaths
    surr[np.arange(n_mp), term_idx] -= matsurv
    (av0, face, prem, coi, admin, charge, gmab, _g, sc) = _account_roll_inputs(mp, basis)
    bound = np.asarray(mp.contract_boundary_months, np.int64)

    def hand(returns):
        cr_f = credited_monthly_rate(returns, g)
        disc = np.ones(n_time + 1)
        disc[1:] = np.cumprod(1.0 / (1.0 + returns))
        disc_mid = disc[:n_time] * (1.0 + returns) ** (-0.5)
        a_f = av0.astype(float).copy(); a_b = av0.astype(float).copy()
        avt_f = a_f.copy(); avt_b = a_b.copy()
        cost = np.zeros(n_mp)
        for t in range(n_time):
            act = t < bound
            ch = admin[t] + charge[:, t]
            a_f = a_f + np.where(act, prem[:, t], 0.0)
            a_f = a_f - np.where(act, ch + coi[:, t] * np.maximum(0.0, face - a_f), 0.0)
            a_f = np.maximum(0.0, a_f)
            am_f = np.where(act, a_f * (1.0 + cr_f[t]) ** 0.5, 0.0)
            a_b = a_b + np.where(act, prem[:, t], 0.0)
            a_b = a_b - np.where(act, ch + coi[:, t] * np.maximum(0.0, face - a_b), 0.0)
            a_b = np.maximum(0.0, a_b)
            am_b = np.where(act, a_b * (1.0 + returns[t]) ** 0.5, 0.0)
            keep = 1.0 - sc[:, t]
            dl = np.maximum(am_f, face) - np.maximum(am_b, face)
            sl = np.maximum(0.0, am_f * keep) - np.maximum(0.0, am_b * keep)
            cost += (deaths[:, t] * dl + surr[:, t] * sl) * disc_mid[t]
            a_f = np.where(act, a_f * (1.0 + cr_f[t]), a_f); avt_f = np.where(act, a_f, avt_f)
            a_b = np.where(act, a_b * (1.0 + returns[t]), a_b); avt_b = np.where(act, a_b, avt_b)
        ml = np.maximum(avt_f, gmab) - np.maximum(avt_b, gmab)
        cost += matsurv * ml * disc[np.minimum(bound, n_time)]
        return float(cost.sum())

    scen = r_m + np.tile(np.array([0.01, -0.02, 0.005, -0.03, 0.0, 0.02, -0.01, 0.015]),
                         n_time // 8 + 1)[:n_time]
    res = fcf.vfa.tvog(mp, basis, scen[None, :])
    np.testing.assert_allclose(res.intrinsic_value, hand(np.full(n_time, r_m)), rtol=1e-12)
    np.testing.assert_allclose(res.guarantee_cost[0], hand(scen), rtol=1e-12)


def test_ul_credit_rate_tvog_monotone_in_g():
    # A higher crediting floor is worth at least as much (max(return, g) is
    # non-decreasing in g), and every scenario cost is non-negative (the floored
    # account never falls below the bare account).
    basis = _ul_basis()
    n_time = 24
    rng = np.random.default_rng(7)
    r_m = (1.024) ** (1.0 / 12.0) - 1.0
    scen = rng.normal(r_m, 0.03, (256, n_time))
    tv = [fcf.vfa.tvog(_credit_mp(g=g, term=n_time), basis, scen) for g in (0.0, 0.02, 0.04)]
    assert tv[2].total_value >= tv[1].total_value >= tv[0].total_value >= -1e-6
    for res in tv:
        assert np.all(res.guarantee_cost >= -1e-6)


def test_ul_credit_rate_tvog_rejects():
    # NO_GUARANTEE_RATE has no credited-rate floor (ValueError); a per-MP varying
    # rate is a scalar-only v1 restriction (NotImplementedError); an annuitizing
    # book uses a different conversion floor (NotImplementedError); the scenario
    # width must match the horizon.
    basis = _ul_basis()
    n_time = 24
    scen = np.full((6, n_time), (1.024) ** (1.0 / 12.0) - 1.0)
    with pytest.raises(ValueError, match="guarantee"):
        fcf.vfa.tvog(_credit_mp(g=fcf.NO_GUARANTEE_RATE, term=n_time), basis, scen)
    with pytest.raises(NotImplementedError, match="uniform"):
        fcf.vfa.tvog(_two_mp(), basis,
                     np.full((6, int(_two_mp().contract_boundary_months.max())),
                             (1.024) ** (1.0 / 12.0) - 1.0))
    annz = _credit_mp(g=0.02, term=n_time)
    object.__setattr__(annz, "annuitization_months", np.array([12]))
    with pytest.raises(NotImplementedError, match="annuitizing"):
        fcf.vfa.tvog(annz, basis, scen)
    with pytest.raises(ValueError, match="columns"):
        fcf.vfa.tvog(_credit_mp(g=0.02, term=n_time), basis, np.full((6, 7), 0.002))


# ----------------------- universal-life surrender charge -----------------------
#
# A surrender pays the account value net of a surrender charge (a fraction of the
# account withheld to recover acquisition costs, declining by policy year). The
# charge scales only the surrender PAYOUT -- the account roll and the decrements
# are unchanged -- so surrender_cf scales exactly by (1 - rate).

def test_ul_surrender_charge_scales_payout_and_lowers_bel():
    mp = _single_mp()
    base = _ul_basis()
    charged = replace(base, surrender_charge_annual=0.10)   # flat 10% withheld
    m0 = fcf.gmm.measure(mp, base, full=True)
    m1 = fcf.gmm.measure(mp, charged, full=True)
    s0 = np.asarray(m0.cashflows.surrender_cf)
    s1 = np.asarray(m1.cashflows.surrender_cf)
    # surrender payout scales exactly by (1 - 0.10)
    np.testing.assert_allclose(s1, 0.90 * s0, rtol=1e-12)
    # the account roll itself is untouched by the charge (it hits the payout only)
    np.testing.assert_array_equal(
        np.asarray(m0.cashflows.account.av), np.asarray(m1.cashflows.account.av))
    # less surrender outflow -> a lower (more profitable) BEL
    assert float(np.atleast_1d(m1.bel)[0]) < float(np.atleast_1d(m0.bel)[0])


def test_ul_surrender_charge_declining_schedule_hand_calc():
    # A declining schedule max(0.10 - 0.02*year, 0): year 0 -> 10%, year 1 -> 8%.
    mp = _single_mp()                                       # 36-month term
    sched = lambda s, a, d, ic, el: np.maximum(0.10 - 0.02 * d, 0.0)
    charged = replace(_ul_basis(), surrender_charge_annual=sched)
    m0 = fcf.gmm.measure(mp, _ul_basis(), full=True)
    m1 = fcf.gmm.measure(mp, charged, full=True)
    s0 = np.asarray(m0.cashflows.surrender_cf)[0]
    s1 = np.asarray(m1.cashflows.surrender_cf)[0]
    keep = np.where(np.arange(s0.shape[0]) // 12 == 0, 0.90, 0.0)
    keep = np.where(np.arange(s0.shape[0]) // 12 == 1, 0.92, keep)
    keep = np.where(np.arange(s0.shape[0]) // 12 >= 2, 0.94, keep)
    np.testing.assert_allclose(s1, keep * s0, rtol=1e-12)


def test_ul_surrender_charge_fast_matches_full():
    mp = _single_mp()
    charged = replace(_ul_basis(), surrender_charge_annual=0.10)
    fast = fcf.gmm.measure(mp, charged, full=False)
    full = fcf.gmm.measure(mp, charged, full=True)
    np.testing.assert_allclose(
        np.atleast_1d(fast.bel), np.atleast_1d(full.bel), rtol=1e-9)


def test_ul_surrender_charge_reduces_credit_floor_tvog():
    # The credited-rate floor's surrender leg pays av_mid * (1 - charge), so a
    # surrender charge reduces that leg (death / maturity legs are unchanged).
    mp = _credit_mp(g=0.05, term=24)
    base = _ul_basis()
    charged = replace(base, surrender_charge_annual=0.10)
    rng = np.random.default_rng(3)
    scen = rng.normal((1.024) ** (1.0 / 12.0) - 1.0, 0.03, (200, 24))
    t0 = fcf.vfa.tvog(mp, base, scen)
    t1 = fcf.vfa.tvog(mp, charged, scen)
    assert t1.time_value < t0.time_value
    assert t1.intrinsic_value <= t0.intrinsic_value + 1e-6
    # A full (100%) charge zeroes the surrender leg; a >100% charge cannot drive it
    # negative -- the leg is the marginal of max(0, av_mid*(1-charge)), so charge
    # 1.0 and 1.5 give the SAME (surrender-free) time value.
    t_full = fcf.vfa.tvog(mp, replace(base, surrender_charge_annual=1.0), scen)
    t_over = fcf.vfa.tvog(mp, replace(base, surrender_charge_annual=1.5), scen)
    np.testing.assert_allclose(t_over.time_value, t_full.time_value, rtol=1e-12)
    assert t_full.time_value <= t1.time_value + 1e-6


# ------------------ combined guarantee TVOG (credited floor + GMDB/GMAB) ------------------

def test_guarantee_tvog_is_additive():
    # guarantee_tvog sums the credited-rate floor (vfa.tvog) and the GMDB/GMAB
    # account-value floors (vfa.measure.time_value, over the model points) -- the
    # two disjoint guarantees -- so total equals the two separate calls.
    mp = _credit_mp(g=0.05, term=24, gmab=12_000_000.0)   # both guarantees live
    basis = _ul_basis()
    rng = np.random.default_rng(5)
    scen = rng.normal((1.024) ** (1.0 / 12.0) - 1.0, 0.04, (300, 24))
    g = fcf.vfa.guarantee_tvog(mp, basis, scen)
    cr = fcf.vfa.tvog(mp, basis, scen).time_value
    af = float(np.sum(fcf.vfa.measure(mp, basis, return_scenarios=scen).time_value))
    np.testing.assert_allclose(g.credited_rate_floor, cr, rtol=1e-12)
    np.testing.assert_allclose(g.account_floor, af, rtol=1e-12)
    np.testing.assert_allclose(g.total, cr + af, rtol=1e-12)


def test_guarantee_tvog_no_crediting_guarantee_is_zero_not_raise():
    # A book with no crediting guarantee contributes a zero crediting floor (rather
    # than raising as the standalone vfa.tvog does) and still reports the GMDB/GMAB
    # account floor.
    mp = _credit_mp(g=fcf.NO_GUARANTEE_RATE, term=24, gmab=12_000_000.0)
    basis = _ul_basis()
    rng = np.random.default_rng(6)
    scen = rng.normal((1.024) ** (1.0 / 12.0) - 1.0, 0.04, (200, 24))
    g = fcf.vfa.guarantee_tvog(mp, basis, scen)
    assert g.credited_rate_floor == 0.0
    af = float(np.sum(fcf.vfa.measure(mp, basis, return_scenarios=scen).time_value))
    np.testing.assert_allclose(g.account_floor, af, rtol=1e-12)
    assert g.total == g.account_floor


def test_non_account_portfolio_has_no_account_sidecar():
    # A plain protection portfolio (no account coverage) gets account=None.
    mp = ModelPoints.single(
        issue_age=40, benefits={"DEATH": 1_000_000.0}, premium=12_000.0,
        term_months=60, calculation_methods={"DEATH": CalculationMethod.DEATH})
    basis = Basis(
        mortality_annual=0.005, lapse_annual=0.01, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.1,
        coverages=(CoverageRate("DEATH", lambda s, a, d: np.full(a.shape, 0.005)),))
    m = fcf.gmm.measure(mp, basis, full=True)
    assert m.cashflows.account is None
