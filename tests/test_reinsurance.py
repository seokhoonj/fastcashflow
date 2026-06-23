"""Reinsurance-held validation -- a quota-share treaty over a direct portfolio.

The cedant cedes a fraction of its claims (recovered) and the same fraction
of its premiums (paid to the reinsurer). The CSM carries the net cost or
gain of the cover -- it may be negative, and there is no loss component.
"""
import fastcashflow as fcf
import numpy as np
import pytest

from fastcashflow import ModelPoints
from fastcashflow.numerics import _norm_ppf
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis

# Some tests here exercise the deprecated carry bridge (measure_inforce) as a
# reference; silence only its own deprecation notice.
pytestmark = pytest.mark.filterwarnings(
    "ignore:reinsurance.measure_inforce:DeprecationWarning")


Q = 0.002          # flat monthly mortality
LAPSE = 0.005      # flat monthly lapse
MORTALITY_CV = 0.10


def _basis():
    return make_death_basis(
        mortality_q     = Q,
        lapse_q         = LAPSE,
        discount_annual = 0.03,
        ra_confidence   = 0.75,
        mortality_cv    = MORTALITY_CV,
    )


def test_reinsurance_hand_calc():
    """Single quota-share treaty -- hand-checked BEL, RA and CSM."""
    basis = _basis()
    death_benefit, premium, term, cession = 1e8, 80_000.0, 60, 0.4
    res = fcf.reinsurance.measure(
        ModelPoints.single(40, premium, term, benefits={"DEATH": death_benefit}, calculation_methods=PATTERNS),
        basis, treaty=fcf.reinsurance.QuotaShare(cession=cession)
    )

    i = basis.discount_monthly
    surv = (1.0 - Q) * (1.0 - LAPSE)
    half = (1.0 + i) ** (-0.5)
    full = 1.0 / (1.0 + i)
    geom = float(np.sum((surv * full) ** np.arange(term)))

    pv_recovery = cession * Q * death_benefit * half * geom
    pv_reinsurance_premium = cession * premium * geom
    bel = pv_reinsurance_premium - pv_recovery
    ra = _norm_ppf(basis.ra_confidence) * MORTALITY_CV * pv_recovery

    assert np.isclose(res.bel[0], bel)
    assert np.isclose(res.ra[0], ra)
    assert np.isclose(res.csm_path[0, 0], -(bel - ra))


def test_reinsurance_csm_can_be_negative():
    """Ceding a profitable book has a net cost -- a negative CSM, no loss component."""
    res = fcf.reinsurance.measure(
        ModelPoints.single(40, 300_000.0, 60, benefits={"DEATH": 1e8}, calculation_methods=PATTERNS),
        _basis(), treaty=fcf.reinsurance.QuotaShare(cession=0.5)
    )
    assert res.bel[0] > 0.0           # reinsurance premiums ceded exceed recoveries
    assert res.csm_path[0, 0] < 0.0        # the net cost is carried as a negative CSM


def test_reinsurance_csm_analysis_of_change_reconciles():
    """The reinsurance CSM waterfall reconciles opening to closing."""
    res = fcf.reinsurance.measure(
        ModelPoints.single(40, 80_000.0, 120, benefits={"DEATH": 1e8}, calculation_methods=PATTERNS),
        _basis(), treaty=fcf.reinsurance.QuotaShare(cession=0.3)
    )
    assert np.allclose(
        res.csm_path[:, :-1] + res.csm_accretion - res.csm_release, res.csm_path[:, 1:]
    )


def test_reinsurance_zero_cession_is_nothing():
    """A zero cession rate cedes nothing -- every figure is zero."""
    res = fcf.reinsurance.measure(
        ModelPoints.single(40, 80_000.0, 60, benefits={"DEATH": 1e8}, calculation_methods=PATTERNS),
        _basis(), treaty=fcf.reinsurance.QuotaShare(cession=0.0)
    )
    assert np.allclose(res.bel, 0.0)
    assert np.allclose(res.ra, 0.0)
    assert np.allclose(res.csm, 0.0)
    assert np.allclose(res.recovery, 0.0)


def test_reinsurance_measure_treaty_is_keyword_only_and_full_false_drops_paths():
    mp = ModelPoints.single(40, 80_000.0, 60, benefits={"DEATH": 1e8},
                            calculation_methods=PATTERNS)
    treaty = fcf.reinsurance.QuotaShare(cession=0.5)

    with pytest.raises(TypeError):
        fcf.reinsurance.measure(mp, _basis(), treaty)   # positional treaty rejected

    full = fcf.reinsurance.measure(mp, _basis(), treaty=treaty)
    head = fcf.reinsurance.measure(mp, _basis(), treaty=treaty, full=False)
    assert np.allclose(head.bel, full.bel)
    assert np.allclose(head.ra, full.ra)
    assert np.allclose(head.csm, full.csm)
    for name in ("bel_path", "ra_path", "csm_path", "csm_accretion",
                 "csm_release", "recovery", "reinsurance_premium", "cashflows",
                 "discount_factor_bom"):
        assert getattr(head, name) is None


def test_reinsurance_rejects_bad_cession_rate():
    """A cession rate outside [0, 1] is an error."""
    with pytest.raises(ValueError, match="cession"):
        fcf.reinsurance.measure(
            ModelPoints.single(40, 80_000.0, 60, benefits={"DEATH": 1e8}, calculation_methods=PATTERNS),
            _basis(), treaty=fcf.reinsurance.QuotaShare(cession=1.5)
        )


def test_reinsurance_trace_renders_and_matches_measure():
    """reinsurance.trace prints a tree whose headline BEL / RA / CSM match the
    measure -- the tree is a faithful view of the same computation."""
    import io

    basis = _basis()
    treaty = fcf.reinsurance.QuotaShare(cession=0.4)
    mp = ModelPoints.single(40, 80_000.0, 60, benefits={"DEATH": 1e8},
                            calculation_methods=PATTERNS)
    m = fcf.reinsurance.measure(mp, basis, treaty=treaty)

    buf = io.StringIO()
    fcf.reinsurance.trace(0, mp, basis, treaty=treaty, file=buf)
    text = buf.getvalue()

    assert "Reinsurance" in text
    assert "Treaty / inputs" in text
    assert "CSM roll-forward" in text
    # the headline figures in the tree equal the measure's (no drift)
    assert f"{float(m.bel[0]):>15,.2f}" in text
    assert f"{float(m.ra[0]):>15,.2f}" in text
    assert f"{float(m.csm[0]):>15,.2f}" in text


def test_reinsurance_trace_routes_a_dict_basis():
    """A dict / BasisRouter basis routes by (product, channel), like show_trace."""
    import io

    mp = fcf.samples.model_points()
    basis = fcf.samples.basis()
    buf = io.StringIO()
    fcf.reinsurance.trace(0, mp, basis, treaty=fcf.reinsurance.QuotaShare(0.5), file=buf)
    assert "Reinsurance" in buf.getvalue()


def test_reinsurance_trace_rejects_bad_index():
    basis = _basis()
    mp = ModelPoints.single(40, 80_000.0, 60, benefits={"DEATH": 1e8},
                            calculation_methods=PATTERNS)
    with pytest.raises(IndexError):
        fcf.reinsurance.trace(9, mp, basis, treaty=fcf.reinsurance.QuotaShare(0.5))


def test_reinsurance_inforce_carries_csm_and_rebases_bel():
    """In-force subsequent measurement (Sec. 44): the prior reinsurance CSM is
    carried forward (accreted at lock-in, released over coverage units) and the
    BEL / RA are the inception slice re-based to the valuation-date count.

    Pinned two ways: (a) with prior_csm taken from the inception CSM trajectory
    at E - period and lock_in = the current discount, rolling one period must
    reproduce that trajectory's CSM at E (the CSM is scale-invariant); (b) the
    BEL equals the PV at E of the remaining ceded flows (re-derived here from the
    measure's own recovery / reinsurance_premium streams), re-based by 1/inforce[E].
    """
    from fastcashflow import InforceState
    from fastcashflow.curves import discount_factors

    basis = _basis()
    treaty = fcf.reinsurance.QuotaShare(cession=0.4)
    mp_new = ModelPoints.single(40, 80_000.0, 240, benefits={"DEATH": 1e8},
                                calculation_methods=PATTERNS)
    m = fcf.reinsurance.measure(mp_new, basis, treaty=treaty)
    elapsed, period = 36, 12
    prior_csm = m.csm_path[:, elapsed - period]

    state = InforceState(
        mp_id=np.array(["R1"]), elapsed_months=np.array([elapsed]),
        count=np.array([1.0]), prior_csm=prior_csm,
        lock_in_rate=basis.discount_annual,
    )
    mp_inf = ModelPoints(
        issue_age=np.array([40]), premium=np.array([80_000.0]),
        term_months=np.array([240]), benefits={"DEATH": np.array([1e8])},
        calculation_methods=PATTERNS, mp_id=np.array(["R1"]),
        elapsed_months=np.array([elapsed]), count=np.array([1.0]),
    )
    v = fcf.reinsurance.measure_inforce(mp_inf, state, basis, treaty=treaty,
                                        period_months=period)

    # (a) CSM carry reproduces the inception trajectory's CSM at E
    assert np.isclose(v.csm[0], m.csm_path[0, elapsed])

    # (b) BEL = PV-at-E of the remaining ceded flows, re-based to count = 1
    bom, mid = discount_factors(basis, m.cashflows.n_time)
    rp, rec = m.reinsurance_premium[0], m.recovery[0]
    pv_at_E = ((rp[elapsed:] * bom[elapsed:-1]).sum()
               - (rec[elapsed:] * mid[elapsed:]).sum()) / bom[elapsed]
    rescale = 1.0 / m.cashflows.inforce[0, elapsed]
    assert np.isclose(v.bel[0], pv_at_E * rescale)
    assert v.bel_path is not None
    assert v.ra_path is not None
    assert np.isclose(v.bel_path[0, 0], m.bel[0])
    assert np.isclose(v.ra_path[0, 0], m.ra[0])

    head = fcf.reinsurance.measure_inforce(
        mp_inf, state, basis, treaty=treaty, period_months=period, full=False)
    assert np.allclose(head.bel, v.bel)
    assert np.allclose(head.ra, v.ra)
    assert np.allclose(head.csm, v.csm)
    for name in ("bel_path", "ra_path", "csm_path", "csm_accretion",
                 "csm_release", "recovery", "reinsurance_premium", "cashflows",
                 "discount_factor_bom"):
        assert getattr(head, name) is None


def test_reinsurance_inforce_rejects_non_positive_period():
    from fastcashflow import InforceState

    basis = _basis()
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([80_000.0]),
        term_months=np.array([60]), benefits={"DEATH": np.array([1e8])},
        calculation_methods=PATTERNS, mp_id=np.array(["R1"]),
        elapsed_months=np.array([12]), count=np.array([1.0]),
    )
    state = InforceState(
        mp_id=np.array(["R1"]), elapsed_months=np.array([12]),
        count=np.array([1.0]), prior_csm=np.array([0.0]),
        lock_in_rate=basis.discount_annual,
    )
    with pytest.raises(ValueError, match="period_months"):
        fcf.reinsurance.measure_inforce(
            mp, state, basis, treaty=fcf.reinsurance.QuotaShare(0.5), period_months=0)


def test_reinsurance_inforce_rejects_runoff():
    """An as-of date at or past the contract boundary (no remaining coverage)
    is rejected -- there is nothing left to value."""
    from fastcashflow import InforceState

    basis = _basis()
    state = InforceState(
        mp_id=np.array(["R1"]), elapsed_months=np.array([60]),
        count=np.array([1.0]), prior_csm=np.array([0.0]),
        lock_in_rate=basis.discount_annual,
    )
    mp_inf = ModelPoints(
        issue_age=np.array([40]), premium=np.array([80_000.0]),
        term_months=np.array([60]), benefits={"DEATH": np.array([1e8])},
        calculation_methods=PATTERNS, mp_id=np.array(["R1"]),
        elapsed_months=np.array([60]), count=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="no remaining coverage"):
        fcf.reinsurance.measure_inforce(mp_inf, state, basis,
                                        treaty=fcf.reinsurance.QuotaShare(0.5))


def test_reinsurance_aggregate_sums_per_mp_and_is_chunk_invariant():
    """measure_aggregate is the scalable sum of the per-model-point measure:
    BEL / RA / CSM totals and the aggregate csm_path / recovery /
    reinsurance_premium equal the full result summed over the model-point axis,
    and the chunk size does not change the totals."""
    basis = _basis()
    treaty = fcf.reinsurance.QuotaShare(0.4)
    mp = ModelPoints(
        issue_age=np.array([35, 40, 45, 50, 55]),
        premium=np.array([60_000.0, 70_000.0, 80_000.0, 90_000.0, 100_000.0]),
        term_months=np.array([120, 180, 240, 120, 60]),
        benefits={"DEATH": np.array([1e8, 8e7, 1.2e8, 5e7, 9e7])},
        calculation_methods=PATTERNS,
    )
    agg = fcf.reinsurance.measure_aggregate(mp, basis, treaty=treaty)
    full = fcf.reinsurance.measure(mp, basis, treaty=treaty)

    assert np.isclose(agg.bel, full.bel.sum())
    assert np.isclose(agg.ra, full.ra.sum())
    assert np.isclose(agg.csm, full.csm.sum())
    assert np.allclose(agg.bel_path, full.bel_path.sum(axis=0))
    assert np.allclose(agg.ra_path, full.ra_path.sum(axis=0))
    assert np.allclose(agg.csm_path, full.csm_path.sum(axis=0))
    assert np.allclose(agg.recovery, full.recovery.sum(axis=0))
    assert np.allclose(agg.reinsurance_premium, full.reinsurance_premium.sum(axis=0))

    agg1 = fcf.reinsurance.measure_aggregate(mp, basis, treaty=treaty, chunk_size=1)
    assert np.isclose(agg1.bel, agg.bel)
    assert np.allclose(agg1.bel_path, agg.bel_path)
    assert np.allclose(agg1.csm_path, agg.csm_path)


def test_reinsurance_aggregate_rejects_bad_chunk_size():
    basis = _basis()
    mp = ModelPoints.single(40, 80_000.0, 60, benefits={"DEATH": 1e8},
                            calculation_methods=PATTERNS)
    with pytest.raises(ValueError, match="chunk_size"):
        fcf.reinsurance.measure_aggregate(mp, basis,
                                          treaty=fcf.reinsurance.QuotaShare(0.5), chunk_size=0)


def test_reinsurance_trace_diff_renders_assumption_and_headline():
    """trace_diff shows the changed assumption and the headline BEL/RA/CSM move."""
    import io, dataclasses

    b1 = _basis()
    b2 = dataclasses.replace(b1, mortality_cv=0.20)
    mp = ModelPoints.single(40, 80_000.0, 120, benefits={"DEATH": 1e8},
                            calculation_methods=PATTERNS)
    buf = io.StringIO()
    fcf.reinsurance.trace_diff(0, mp, b1, b2, treaty=fcf.reinsurance.QuotaShare(0.4),
                               file=buf)
    t = buf.getvalue()
    assert "diff-reinsurance" in t
    assert "mortality_cv" in t                 # the changed assumption surfaces
    assert "RA" in t and "CSM" in t            # the headline deltas

    # a no-change baseline reports no changes; the shocked diff must not, and at
    # least one headline metric must show a non-zero numeric delta (not just the
    # metric label being present).
    base = io.StringIO()
    fcf.reinsurance.trace_diff(0, mp, b1, b1, treaty=fcf.reinsurance.QuotaShare(0.4), file=base)
    assert "(no changes in tracked fields)" in base.getvalue()
    assert "(no changes in tracked fields)" not in t
    moved = False
    for line in t.splitlines():
        if "->" in line and "(" in line and "=" not in line:
            try:
                lo = float(line.split("->")[0].split()[-1].replace(",", ""))
                hi = float(line.split("->")[1].split()[0].replace(",", ""))
                moved = moved or abs(hi - lo) > 1e-9
            except (ValueError, IndexError):
                pass
    assert moved   # the shocked assumption moved at least one headline metric


def test_reinsurance_measure_stream_matches_in_memory(tmp_path):
    """Streaming a parquet ceded book chunk by chunk gives the same per-policy
    CSM as the in-memory measure (low-benefit out-of-core path)."""
    import polars as pl

    basis = _basis()
    treaty = fcf.reinsurance.QuotaShare(0.4)
    pol = pl.DataFrame({"mp_id": ["A", "B", "C"], "issue_age": [40, 45, 50],
                        "term_months": [12, 12, 12],
                        "premium_term_months": [12, 12, 12],
                        "count": [1.0, 1.0, 1.0]})
    cov = pl.DataFrame({"mp_id": ["A", "B", "C"], "coverage": ["DEATH"] * 3,
                        "amount": [1e8, 1e8, 1e8],
                        "premium": [10_000.0, 11_000.0, 12_000.0]})
    pp, cp, od = tmp_path / "pol.parquet", tmp_path / "cov.parquet", tmp_path / "out"
    pol.write_parquet(pp)
    cov.write_parquet(cp)
    n = fcf.reinsurance.measure_stream(pp, od, basis, treaty=treaty, coverages=cp,
                                       calculation_methods=PATTERNS, chunk_size=2)
    assert n == 3
    parts = pl.concat([pl.read_parquet(p) for p in sorted(od.glob("part-*.parquet"))])
    mp = fcf.read_model_points(pp, coverages=cp, calculation_methods=PATTERNS)
    ref = fcf.reinsurance.measure(mp, basis, treaty=treaty)
    assert np.allclose(sorted(parts["csm"].to_list()), sorted(ref.csm.tolist()))


def test_pv_path_reverse_cumsum_anchors():
    """_pv_path is the reverse-cumsum PV-at-t of a month stream: column 0 is the
    inception PV (the whole-stream sum at unit discount), the terminal column is
    0 (nothing left), and with positive flows it decays monotonically. A
    non-unit discount re-anchors each column by its bom factor."""
    from fastcashflow._reinsurance import _pv_path

    month_pv = np.array([[4.0, 3.0, 2.0, 1.0]])      # (1 mp, 4 months)
    p = _pv_path(month_pv, np.ones(5))               # unit discount, n_time+1
    assert p.shape == (1, 5)
    assert np.isclose(p[0, 0], 10.0)                 # inception PV = whole sum
    assert np.isclose(p[0, 4], 0.0)                  # nothing past the horizon
    assert np.all(np.diff(p[0]) <= 1e-12)            # monotonic non-increasing

    bom = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    raw = np.array([10.0, 6.0, 3.0, 1.0, 0.0])       # reverse-cumsum, padded
    assert np.allclose(_pv_path(month_pv, bom)[0], raw / bom)


def test_reinsurance_inforce_high_lapse_stays_finite():
    """With very high lapse the in-force surviving to the valuation date is
    near zero; the count / inforce[elapsed] rescale is huge, but the sliced
    trajectory is proportionally tiny, so BEL / RA / CSM stay finite (the
    rescale and the slice cancel) -- the near-zero-inforce guard at the rescale."""
    from fastcashflow import InforceState

    basis = make_death_basis(mortality_q=0.002, lapse_q=0.5,      # 50%/month lapse
                             discount_annual=0.03, ra_confidence=0.75,
                             mortality_cv=0.10)
    treaty = fcf.reinsurance.QuotaShare(0.4)
    elapsed = 24                                                   # < term 120
    state = InforceState(
        mp_id=np.array(["R1"]), elapsed_months=np.array([elapsed]),
        count=np.array([100.0]), prior_csm=np.array([0.0]),
        lock_in_rate=basis.discount_annual)
    mp = ModelPoints(
        issue_age=np.array([40]), premium=np.array([80_000.0]),
        term_months=np.array([120]), benefits={"DEATH": np.array([1e8])},
        calculation_methods=PATTERNS, mp_id=np.array(["R1"]),
        elapsed_months=np.array([elapsed]), count=np.array([100.0]))
    v = fcf.reinsurance.measure_inforce(mp, state, basis, treaty=treaty, period_months=12)
    assert np.isfinite(v.bel[0]) and np.isfinite(v.ra[0]) and np.isfinite(v.csm[0])


def test_reinsurance_inforce_per_mp_varying_elapsed_carries_each_csm():
    """Two ceded contracts valued in-force at DIFFERENT elapsed_months each carry
    their own prior CSM -- the per-MP ``prior_t = elapsed - period`` gather must
    pick the right column per row. With each prior_csm taken from that contract's
    inception csm_path[elapsed-period] and lock_in = the discount, rolling one
    period must reproduce each contract's csm_path[elapsed] (CSM is
    scale-invariant). This pins the per-row gather the single-MP test cannot."""
    from fastcashflow import InforceState

    basis = _basis()
    treaty = fcf.reinsurance.QuotaShare(0.4)
    age = np.array([40, 50]); prem = np.array([80_000.0, 60_000.0])
    term = np.array([240, 180]); ben = np.array([1e8, 7e7])
    m = fcf.reinsurance.measure(
        ModelPoints(issue_age=age, premium=prem, term_months=term,
                    benefits={"DEATH": ben}, calculation_methods=PATTERNS),
        basis, treaty=treaty)

    elapsed = np.array([36, 24]); period = 12
    prior_csm = np.array([m.csm_path[0, elapsed[0] - period],
                          m.csm_path[1, elapsed[1] - period]])
    state = InforceState(
        mp_id=np.array(["R1", "R2"]), elapsed_months=elapsed,
        count=np.array([1.0, 1.0]), prior_csm=prior_csm,
        lock_in_rate=basis.discount_annual)
    mp_inf = ModelPoints(
        issue_age=age, premium=prem, term_months=term, benefits={"DEATH": ben},
        calculation_methods=PATTERNS, mp_id=np.array(["R1", "R2"]),
        elapsed_months=elapsed, count=np.array([1.0, 1.0]))
    v = fcf.reinsurance.measure_inforce(mp_inf, state, basis, treaty=treaty,
                                        period_months=period)
    # each row carries its own CSM, sliced at its own elapsed
    assert np.isclose(v.csm[0], m.csm_path[0, elapsed[0]])
    assert np.isclose(v.csm[1], m.csm_path[1, elapsed[1]])


def test_reinsurance_roll_forward_and_reconcile_balance():
    """roll_forward slices a reinsurance measurement into BEL/RA/CSM period
    movements; reconcile aggregates them. Each per-MP movement and each
    portfolio reconciliation block balances (opening + finance/accretion -
    release == closing), and the CSM runs off to ~0 (no loss component)."""
    basis = _basis()
    treaty = fcf.reinsurance.QuotaShare(0.4)
    mp = ModelPoints(
        issue_age=np.array([40, 50]), premium=np.array([80_000.0, 60_000.0]),
        term_months=np.array([60, 60]), benefits={"DEATH": np.array([1e8, 7e7])},
        calculation_methods=PATTERNS)
    m = fcf.reinsurance.measure(mp, basis, treaty=treaty)

    movs = fcf.roll_forward(m, 12)
    recs = fcf.reconcile(movs)
    assert all(isinstance(x, fcf.reinsurance.PeriodMovement) for x in movs)
    assert all(isinstance(x, fcf.reinsurance.Reconciliation) for x in recs)

    for mv in movs:                                   # per-MP blocks balance
        assert np.allclose(mv.bel_opening + mv.bel_interest - mv.bel_release,
                           mv.bel_closing)
        assert np.allclose(mv.csm_opening + mv.csm_accretion - mv.csm_release,
                           mv.csm_closing)
    for r in recs:                                    # portfolio totals balance
        for comp in ("bel", "ra", "csm"):
            o = getattr(r, f"{comp}_opening"); fin = getattr(r, f"{comp}_finance")
            rel = getattr(r, f"{comp}_release"); c = getattr(r, f"{comp}_closing")
            assert np.isclose(o + fin + rel, c, atol=1e-4)
    assert np.isclose(recs[-1].csm_closing, 0.0, atol=1e-3)   # fully amortised
