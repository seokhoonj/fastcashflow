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

    # full=False (the fused fast path) must produce the same headline: an
    # account book auto-routes to the full measurement via requires_full (the
    # fused account carrier is a deferred Step 4), so the headline is correct,
    # not silently wrong on the account-unaware fast kernel.
    ref_h = measure_ul(make_mp(with_coverage=False), _ul_basis(with_coverage=False),
                       measurement_model="GMM", full=False)
    got_h = fcf.gmm.measure(make_mp(with_coverage=True),
                            _ul_basis(with_coverage=True), full=False)
    for name in ("bel", "ra", "csm", "loss_component"):
        assert np.array_equal(getattr(got_h, name), getattr(ref_h, name)), (
            f"{name} full=False parity failed -- account book did not route to "
            f"the full measurement")


def test_ul_fold_parity_single_policy():
    _assert_parity(_single_mp)


def test_ul_fold_parity_two_policy():
    _assert_parity(_two_mp)


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
