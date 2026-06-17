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
    from fastcashflow.engine import requires_full, _portfolio_has_account
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

    with pytest.raises(NotImplementedError):
        fcf.reinsurance.measure(mp, basis, treaty=fcf.samples.treaty())
    # stochastic fast branch (confidence RA + no settlement_pattern -- the UL
    # basis defaults) reads mortality_cf raw.
    with pytest.raises(NotImplementedError):
        fcf.gmm.stochastic(mp, basis, np.linspace(0.01, 0.05, 8))
    with pytest.raises(NotImplementedError):
        fcf.vfa.tvog(mp, basis,
                     np.tile(np.linspace(-0.01, 0.03, 8)[:, None], (1, n_time)))
    # roll_forward is NOT gated: it reads only the account-netted bel / ra / csm
    # paths and the in-force count, never the raw benefit cash flows, so it
    # supports an account book (see
    # test_movement.test_roll_forward_universal_life_account_reconciles).


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
