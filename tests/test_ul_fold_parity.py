"""Universal-life fold parity -- gmm.measure == measure_ul (the bit-identity
target of Step 2+3).

The standalone ``measure_ul`` path (``_ul.py``) rolls the account value in a
separate kernel and measures it. Step 2+3 folds that roll into the shared
projection kernel (``_project_kernel``) behind the per-coverage account-chassis
flags, so a UL contract carrying ``CoverageRate("DEATH", coi_annual,
funds_from_account=True, pays_account_balance=True)`` measured through
``gmm.measure(full=True)`` must reproduce ``measure_ul(..., "GMM", full=True)``
exactly.

The two portfolios differ only in expression: the ``measure_ul`` portfolio
carries the COI / load / crediting assumptions on the Basis with NO coverage
list (the old separate path); the folded portfolio additionally registers the
DEATH coverage (rate = ``coi_annual``) with the account flags so the shared
projection routes the death leg through the account roll.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
from fastcashflow._ul import measure_ul


def _ul_basis(*, with_coverage: bool):
    """The UL basis. ``with_coverage=True`` adds the DEATH coverage (rate =
    coi_annual) with the account-chassis flags, so the shared projection folds
    the account roll; ``False`` is the standalone ``measure_ul`` basis."""
    coi = 0.0015
    kw = dict(
        mortality_annual=0.004,
        lapse_annual=0.03,
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.1,
        investment_return=0.024,
        coi_annual=coi,
        premium_load=0.08,
    )
    if with_coverage:
        kw["coverages"] = (
            CoverageRate("DEATH", coi, funds_from_account=True,
                         pays_account_balance=True),
        )
    return Basis(**kw)


def _single_mp(*, with_coverage: bool):
    common = dict(
        issue_age=np.array([40.0]),
        premium=np.array([500_000.0]),
        term_months=np.array([36]),
        account_value=np.array([0.0]),
        minimum_death_benefit=np.array([80_000_000.0]),
        minimum_accumulation_benefit=np.array([0.0]),
        minimum_crediting_rate=np.array([0.0]),
        sex=np.array([0]),
    )
    if with_coverage:
        # The DEATH coverage's amount is unused (the account death reads the
        # account balance, not coverage_amount); set it to the face for clarity.
        common["benefits"] = {"DEATH": np.array([80_000_000.0])}
        common["calculation_methods"] = {"DEATH": CalculationMethod.DEATH}
    return ModelPoints(**common)


def _two_mp(*, with_coverage: bool):
    common = dict(
        issue_age=np.array([40.0, 55.0]),
        premium=np.array([500_000.0, 300_000.0]),
        term_months=np.array([36, 24]),
        account_value=np.array([0.0, 1_000_000.0]),
        minimum_death_benefit=np.array([80_000_000.0, 50_000_000.0]),
        minimum_accumulation_benefit=np.array([0.0, 0.0]),
        minimum_crediting_rate=np.array([0.0, 0.01]),
        sex=np.array([0, 1]),
    )
    if with_coverage:
        common["benefits"] = {"DEATH": np.array([80_000_000.0, 50_000_000.0])}
        common["calculation_methods"] = {"DEATH": CalculationMethod.DEATH}
    return ModelPoints(**common)


def _assert_parity(make_mp):
    ref = measure_ul(make_mp(with_coverage=False), _ul_basis(with_coverage=False),
                     measurement_model="GMM", full=True)
    got = fcf.gmm.measure(make_mp(with_coverage=True),
                          _ul_basis(with_coverage=True), full=True)
    for name in ("bel", "ra", "csm", "loss_component"):
        r = getattr(ref, name)
        g = getattr(got, name)
        # The fold reuses the SAME shared roll-forward / CSM kernels on the
        # SAME inputs as measure_ul, so parity is bit-identical, not merely
        # within tolerance.
        assert np.array_equal(g, r), (
            f"{name} parity: gmm.measure={g} vs measure_ul={r} "
            f"(max abs delta {np.abs(g - r).max()})")
    # The full-path trajectories match bit-for-bit too.
    assert np.array_equal(got.bel_path, ref.bel_path)
    assert np.array_equal(got.ra_path, ref.ra_path)
    assert np.array_equal(got.csm_path, ref.csm_path)

    # full=False (the fused fast path) now carries the account roll in the
    # SCALAR fused kernel itself (Step 4) -- an account book no longer routes to
    # _measure_full, it runs the account-aware scalar kernel directly. Confirm
    # the routing actually exercises the scalar fast path (requires_full is now
    # False for an account book; the account check was removed from it).
    from fastcashflow.engine import requires_full, _portfolio_has_account
    mp_c = make_mp(with_coverage=True)
    basis_c = _ul_basis(with_coverage=True)
    assert _portfolio_has_account(mp_c, basis_c)
    assert not requires_full(mp_c, basis_c), (
        "Step 4: an account book must run the scalar fast path, not auto-route "
        "to the full kernel via requires_full")

    # The scalar fused kernel accumulates present values FORWARD (pv += cf x
    # discount factor), whereas measure_ul / _measure_full run the BACKWARD
    # roll-forward recursion (bel[t] = cf x half[t] + bel[t+1] x full[t]). The
    # two are mathematically equal but differ in floating-point summation order,
    # so the fused path is numerically equal -- NOT bit-identical -- to the
    # backward roll, exactly as every other fused (full=False) book is to its
    # full=True counterpart. So this is np.allclose, not np.array_equal.
    ref_h = measure_ul(make_mp(with_coverage=False), _ul_basis(with_coverage=False),
                       measurement_model="GMM", full=False)
    got_h = fcf.gmm.measure(mp_c, basis_c, full=False)
    for name in ("bel", "ra", "csm", "loss_component"):
        g = getattr(got_h, name)
        r = getattr(ref_h, name)
        assert np.allclose(g, r), (
            f"{name} full=False parity (scalar kernel) vs measure_ul: "
            f"max abs delta {np.abs(g - r).max()}")

    # Explicit gmm.measure(full=False) == gmm.measure(full=True): this now
    # exercises the SCALAR account kernel against the full-path account fold.
    # Same forward-vs-backward floating-point story -> np.allclose, the same
    # fast-vs-full relationship a plain protection book has.
    got_full = fcf.gmm.measure(mp_c, basis_c, full=True)
    for name in ("bel", "ra", "csm", "loss_component"):
        g = getattr(got_h, name)
        f = getattr(got_full, name)
        assert np.allclose(g, f), (
            f"{name} full=False (scalar) vs full=True (fold): "
            f"max abs delta {np.abs(g - f).max()}")


def test_ul_fold_parity_single_policy():
    _assert_parity(_single_mp)


def test_ul_fold_parity_two_policy():
    _assert_parity(_two_mp)


def test_vfa_ul_parity():
    # Step 5: vfa.measure of a universal-life book == measure_ul(...,"VFA")
    # bit-identical -- variable UL is the recursive account roll discounted at
    # the underlying-items return (the only thing the VFA model changes).
    for make_mp in (_single_mp, _two_mp):
        ref = measure_ul(make_mp(with_coverage=False),
                         _ul_basis(with_coverage=False),
                         measurement_model="VFA", full=True)
        got = fcf.vfa.measure(make_mp(with_coverage=True),
                              _ul_basis(with_coverage=True))
        for name in ("bel", "ra", "csm", "loss_component"):
            assert np.array_equal(getattr(got, name), getattr(ref, name)), (
                f"{name} VFA parity: vfa.measure={getattr(got, name)} vs "
                f"measure_ul={getattr(ref, name)}")
        assert np.array_equal(got.bel_path, ref.bel_path)
        assert np.array_equal(got.ra_path, ref.ra_path)
        assert np.array_equal(got.csm_path, ref.csm_path)
        # A universal-life book has no asset-based fee / guarantee TVOG (v1).
        assert np.array_equal(got.variable_fee, np.zeros_like(got.bel))
        assert np.array_equal(got.time_value, np.zeros_like(got.bel))
        # full=False headline matches too.
        got_h = fcf.vfa.measure(make_mp(with_coverage=True),
                                _ul_basis(with_coverage=True), full=False)
        ref_h = measure_ul(make_mp(with_coverage=False),
                           _ul_basis(with_coverage=False),
                           measurement_model="VFA", full=False)
        for name in ("bel", "ra", "csm", "loss_component"):
            assert np.array_equal(getattr(got_h, name), getattr(ref_h, name))


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
    mp = _single_mp(with_coverage=True)
    with pytest.raises(NotImplementedError):
        fcf.gmm.measure(mp, basis, full=True)
    with pytest.raises(NotImplementedError):
        fcf.vfa.measure(mp, basis)


def test_vfa_ul_return_scenarios_rejected():
    # UL guarantee time value under VFA is deferred -- return_scenarios raises.
    with pytest.raises(NotImplementedError):
        fcf.vfa.measure(_single_mp(with_coverage=True),
                        _ul_basis(with_coverage=True),
                        return_scenarios=np.zeros((4, 36)))


def test_ul_fold_account_sidecar_populated():
    # The folded projection exposes the account trajectory as a nested sidecar.
    got = fcf.gmm.measure(_two_mp(with_coverage=True),
                          _ul_basis(with_coverage=True), full=True)
    acct = got.cashflows.account
    assert acct is not None
    n_mp, n_time = got.cashflows.claim_cf.shape
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
    m = fcf.gmm.measure(_single_mp(with_coverage=True), folded, full=True)
    assert m.cashflows.account is not None
    # And it still matches the standalone measure_ul (same callable COI).
    ref = measure_ul(_single_mp(with_coverage=False), Basis(**common),
                     measurement_model="GMM", full=True)
    assert np.array_equal(m.bel, ref.bel) and np.array_equal(m.csm, ref.csm)


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
    mp = _single_mp(with_coverage=True)
    fast = fcf.gmm.measure(mp, coc, full=False)
    full = fcf.gmm.measure(mp, coc, full=True)
    for name in ("bel", "ra", "csm", "loss_component"):
        assert np.array_equal(getattr(fast, name), getattr(full, name))


def test_account_boundary_cut_routes_full_false_to_full():
    # Step 4 routing: a contract boundary shorter than the term pays the boundary
    # survivors a terminal surrender that the scalar fold does not handle, so a
    # boundary-cut account book routes full=False -> full (bit-identical headline).
    basis = _ul_basis(with_coverage=True)
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
    mp = _two_mp(with_coverage=True)
    basis = _ul_basis(with_coverage=True)
    n_time = fcf.gmm.measure(mp, basis, full=True).bel_path.shape[1] - 1

    with pytest.raises(NotImplementedError):
        fcf.reinsurance.measure(mp, basis, treaty=fcf.samples.treaty())
    # stochastic fast branch (confidence RA + no settlement_pattern -- the UL
    # basis defaults) reads claim_cf raw.
    with pytest.raises(NotImplementedError):
        fcf.gmm.stochastic(mp, basis, np.linspace(0.01, 0.05, 8))
    with pytest.raises(NotImplementedError):
        fcf.vfa.tvog(mp, basis,
                     np.tile(np.linspace(-0.01, 0.03, 8)[:, None], (1, n_time)))
    # roll_forward reads claim_cf as incurred claims.
    m = fcf.gmm.measure(mp, basis, full=True)
    with pytest.raises(NotImplementedError):
        fcf.roll_forward(m, 12)


def test_non_account_portfolio_has_no_account_sidecar():
    # A plain protection portfolio (no account coverage) gets account=None.
    mp = ModelPoints.single(
        issue_age=40, benefits={0: 1_000_000.0}, premium=12_000.0,
        term_months=60, calculation_methods={"DEATH": CalculationMethod.DEATH})
    basis = Basis(
        mortality_annual=0.005, lapse_annual=0.01, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.1,
        coverages=(CoverageRate("DEATH", lambda s, a, d: np.full(a.shape, 0.005)),))
    m = fcf.gmm.measure(mp, basis, full=True)
    assert m.cashflows.account is None
