"""PAA validation -- the Premium Allocation Approach measurement.

The PAA measures the Liability for Remaining Coverage as an unearned
premium: premiums build it up, insurance revenue (allocated by coverage
units) releases it. Total revenue equals total premium, so the service
result is just premiums less claims and expenses -- the underwriting profit.
"""
import fastcashflow as fcf
import numpy as np
import pytest

from fastcashflow import ExpenseItem, ModelPoints
from conftest import PATTERNS, annual_from_monthly as _annual, make_death_basis


Q = 0.002          # flat monthly mortality
LAPSE = 0.005      # flat monthly lapse


def _basis(**overrides):
    kw = dict(
        mortality_q     = Q,
        lapse_q         = LAPSE,
        discount_annual = 0.03,
        ra_confidence   = 0.75,
        mortality_cv    = 0.10,
    )
    kw.update(overrides)
    return make_death_basis(**kw)


def test_paa_revenue_equals_total_premium():
    """Insurance revenue recognised over the contract equals total premium."""
    res = fcf.paa.measure(ModelPoints.single(40, 50_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS), _basis())
    assert np.isclose(res.revenue.sum(), res.cashflows.premium_cf.sum())


def test_paa_lrc_hand_calc():
    """Single-premium contract -- the LRC is the textbook pro-rata UPR."""
    basis = _basis()
    single, term = 1_000_000.0, 12
    res = fcf.paa.measure(
        ModelPoints.single(40, single, term, benefits={0: 1e8}, premium_term_months=1, calculation_methods=PATTERNS), basis
    )

    # straight-line earning: the premium spread evenly over the coverage period
    assert np.allclose(res.revenue[0], single / term)
    # LRC = premium * remaining coverage / total coverage (unearned premium)
    lrc = np.empty(term + 1)
    lrc[0] = 0.0
    lrc[1:] = single * (term - np.arange(1, term + 1)) / term
    assert np.allclose(res.lrc_path[0], lrc)
    assert np.isclose(res.lrc_path[0, -1], 0.0)     # fully earned by the term end


def test_paa_lrc_builds_and_releases():
    """The LRC builds from zero and releases back to zero over the term."""
    res = fcf.paa.measure(ModelPoints.single(35, 40_000.0, 24, benefits={0: 5e7}, calculation_methods=PATTERNS), _basis())
    assert np.isclose(res.lrc_path[0, 0], 0.0)        # builds from zero
    assert np.isclose(res.lrc_path[0, -1], 0.0)       # releases back to zero
    assert np.all(res.lrc_path[0] >= -1e-6)           # a liability, never negative
    assert res.lrc_path[0].max() > 0.0                # genuinely non-trivial between


def test_paa_service_result_is_the_underwriting_profit():
    """Total service result = premiums - claims - expenses."""
    basis = _basis(expense_items=(
        ExpenseItem("acquisition",  "alpha_fixed",    100_000.0),
        ExpenseItem("maintenance",  "gamma_fixed",  12_000.0),
    ))
    res = fcf.paa.measure(ModelPoints.single(45, 60_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS), basis)
    cf = res.cashflows
    profit = (cf.premium_cf.sum() - cf.claim_cf.sum()
              - cf.morbidity_cf.sum() - cf.expense_cf.sum())
    assert np.isclose(res.service_result.sum(), profit)


def test_paa_onerous_contract_carries_a_loss():
    """A contract whose claims exceed its premiums is flagged onerous."""
    profitable = fcf.paa.measure(
        ModelPoints.single(40, 500_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS), _basis()
    )
    onerous = fcf.paa.measure(
        ModelPoints.single(40, 1_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS), _basis()
    )
    assert np.allclose(profitable.loss_component, 0.0)
    assert onerous.loss_component[0] > 0.0


def test_paa_onerous_test_honours_cost_of_capital_ra():
    """The PAA onerous test used to hardcode the confidence-level RA, silently
    ignoring ra_method='cost_of_capital'. It now routes through the shared RA
    helper, so the cost-of-capital basis gives a different (non-zero) RA and
    hence a different loss component than the confidence-level basis."""
    mp = ModelPoints.single(40, 1_000.0, 24, benefits={0: 1e8},
                            calculation_methods=PATTERNS)
    cl = fcf.paa.measure(mp, _basis(ra_method="confidence_level"))
    coc = fcf.paa.measure(mp, _basis(
        ra_method="cost_of_capital", cost_of_capital_rate=0.06))
    assert coc.loss_component[0] > 0.0
    # the two RA methods give materially different onerous losses (before the
    # fix the cost-of-capital basis silently produced the confidence-level loss)
    assert not np.isclose(coc.loss_component[0], cl.loss_component[0])


def test_paa_revenue_basis_claims():
    """B126(b): revenue allocated by the expected timing of incurred claims."""
    basis = _basis(expense_items=(
        ExpenseItem("acquisition", "alpha_fixed", 500_000.0),
    ))
    mps = ModelPoints.single(40, 50_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS)
    by_time = fcf.paa.measure(mps, basis, revenue_basis="time")
    by_claims = fcf.paa.measure(mps, basis, revenue_basis="claims")

    total_premium = by_claims.cashflows.premium_cf.sum()
    assert np.isclose(by_claims.revenue.sum(), total_premium)   # still totals premium

    se = by_claims.service_expense[0]
    assert np.allclose(by_claims.revenue[0], total_premium * se / se.sum())
    # the t=0 acquisition spike makes the claims basis differ from passage of time
    assert not np.allclose(by_time.revenue[0], by_claims.revenue[0])


def test_paa_rejects_unknown_revenue_basis():
    """An unrecognised revenue basis is an error."""
    with pytest.raises(ValueError, match="revenue_basis"):
        fcf.paa.measure(ModelPoints.single(40, 50_000.0, 12, benefits={0: 1e8}, calculation_methods=PATTERNS),
                    _basis(), revenue_basis="weekly")


# ---------------------------------------------------------------------------
# full=False headline contract (the chunked-portfolio building block) + guards
# ---------------------------------------------------------------------------
def _paa_mp():
    return ModelPoints.single(40, 1_000_000.0, 12, benefits={0: 1e8},
                              calculation_methods=PATTERNS)


def test_paa_full_false_matches_full_headline():
    """full=False fills the same headline (lrc / loss_component / fcf) as
    full=True and leaves every trajectory and the cash flows None."""
    basis = _basis()
    mp = _paa_mp()
    full = fcf.paa.measure(mp, basis)
    head = fcf.paa.measure(mp, basis, full=False)
    assert np.allclose(head.lrc, full.lrc)
    assert np.allclose(head.loss_component, full.loss_component)
    assert np.allclose(head.fcf, full.fcf)
    assert head.lrc_path is None and head.revenue is None
    assert head.service_expense is None and head.lic is None
    assert head.cashflows is None


def test_paa_headline_only_rejected_by_consumers():
    """A headline-only PAA measurement gives a clear error in group / roll /
    report -- not an AttributeError on a None trajectory (PAA has no bel_path,
    so the guard checks lrc_path)."""
    head = fcf.paa.measure(_paa_mp(), _basis(), full=False)
    with pytest.raises(ValueError, match="full=True PAA"):
        fcf.roll_forward(head)
    with pytest.raises(ValueError, match="full=True PAA"):
        fcf.report(head)
    with pytest.raises(ValueError, match="full PAA measurement"):
        fcf.group(head, np.zeros(1, dtype=int))


def test_paa_rejects_bad_revenue_basis_even_on_headline():
    """revenue_basis is validated up front, so a typo is caught on the headline
    path too (where the revenue allocation it selects is never computed)."""
    with pytest.raises(ValueError, match="revenue_basis"):
        fcf.paa.measure(_paa_mp(), _basis(), revenue_basis="nope", full=False)


def test_paa_onerous_matches_gmm_with_settlement_discount():
    """The PAA onerous test discounts claims to their settlement dates exactly
    as the GMM does, so the loss component matches GMM for identical claims.

    With both a settlement pattern and a non-zero discount, claims paid later
    are worth less; the PAA onerous test reuses the GMM fulfilment cash flows
    and so applies the same _settlement_factor. The LIC stays undiscounted, but
    the onerous-test FCF / loss component must equal GMM's.
    """
    from dataclasses import replace
    basis = replace(_basis(mortality_q=0.02, discount_annual=0.06),
                    settlement_pattern=np.array([0.4, 0.3, 0.2, 0.1]))
    mp = ModelPoints.single(40, 5_000.0, 24, benefits={0: 5e8},
                            calculation_methods=PATTERNS)
    gmm = fcf.gmm.measure(mp, basis)
    paa = fcf.paa.measure(mp, basis)
    assert paa.loss_component[0] > 0.0                          # genuinely onerous
    assert np.isclose(paa.loss_component[0], gmm.loss_component[0])
    assert np.isclose(paa.fcf[0], gmm.bel[0] + gmm.ra[0])


def test_paa_trace_diff_renders_assumption_and_headline():
    """trace_diff shows the changed assumption and the headline LRC / LIC move."""
    import io, dataclasses

    b1 = _basis()
    b2 = dataclasses.replace(b1, discount_annual=0.05)
    mp = ModelPoints.single(40, 10_000.0, 12, benefits={0: 1e6},
                            calculation_methods=PATTERNS)
    buf = io.StringIO()
    fcf.paa.trace_diff(0, mp, b1, b2, file=buf)
    t = buf.getvalue()
    assert "diff-paa" in t and "discount_annual" in t and "LRC" in t


def test_paa_measure_stream_matches_in_memory(tmp_path):
    """Streaming a parquet book chunk by chunk gives the same per-policy
    headline as the in-memory measure (low-benefit out-of-core path)."""
    import polars as pl

    basis = _basis()
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
    n = fcf.paa.measure_stream(pp, od, basis, coverages=cp,
                               calculation_methods=PATTERNS, chunk_size=2)
    assert n == 3
    parts = pl.concat([pl.read_parquet(p) for p in sorted(od.glob("part-*.parquet"))])
    mp = fcf.read_model_points(pp, coverages=cp, calculation_methods=PATTERNS)
    ref = fcf.paa.measure(mp, basis)
    assert np.allclose(sorted(parts["loss_component"].to_list()),
                       sorted(ref.loss_component.tolist()))
