"""paa.settle_aggregate / paa.settle_stream -- the PAA settlement scale
variants (skeleton).

Authoritative skeleton: written before the implementation and activated
unchanged once it lands. The PAA counterparts of gmm.settle_aggregate /
gmm.settle_stream: every paragraph-55(b) settlement line (LRC / loss component
/ LIC) is additive across contracts, so the aggregate is the per-MP settle sum
in bounded memory, and the stream writes one part-NNNNN.parquet per chunk.
"""
import numpy as np
import pytest
import polars as pl

import fastcashflow as fcf
from fastcashflow import InforceState, ModelPoints
from fastcashflow._measurement.movement import reconcile
from conftest import PATTERNS, make_death_basis

settle_aggregate = getattr(fcf.paa, "settle_aggregate", None)
settle_stream = getattr(fcf.paa, "settle_stream", None)
pytestmark = pytest.mark.skipif(
    settle_aggregate is None or settle_stream is None,
    reason="paa.settle_aggregate / paa.settle_stream not implemented yet "
           "(skeleton activates unchanged once they land)")

_PAA_LINES = (
    "lrc_opening", "premiums", "revenue", "lrc_experience", "lrc_closing",
    "loss_component_opening", "loss_component_recognised",
    "loss_component_reversed", "loss_component_closing",
    "lic_opening", "claims_incurred", "claims_paid", "lic_closing",
)


def _basis(**overrides):
    kw = dict(mortality_q=0.0, lapse_q=0.0, discount_annual=0.0,
              ra_confidence=0.75, mortality_cv=0.10)
    kw.update(overrides)
    return make_death_basis(**kw)


def _multi_book(*, premiums, benefits, em_close=6, count_factor=None,
                term=12):
    """A heterogeneous multi-MP single-premium accident book seated at
    em_close. ``count_factor`` scales the closing count off the expected
    survival (None = on-track)."""
    n = len(premiums)
    ids = np.array([f"PA{i}" for i in range(n)])
    premiums = np.asarray(premiums, dtype=np.float64)
    benefits = np.asarray(benefits, dtype=np.float64)
    factor = np.ones(n) if count_factor is None else np.asarray(count_factor,
                                                                dtype=np.float64)
    prior_count = np.ones(n)
    count = factor * 1.0
    mp = ModelPoints(
        issue_age=np.full(n, 40, dtype=np.int64), premium=premiums,
        term_months=np.full(n, term, dtype=np.int64),
        premium_term_months=np.full(n, 1, dtype=np.int64),
        benefits={"DEATH": benefits}, count=count,
        elapsed_months=np.full(n, em_close, dtype=np.int64), mp_id=ids,
        product=np.full(n, "ACC"), calculation_methods=PATTERNS)
    state = InforceState(
        mp_id=ids, elapsed_months=np.full(n, em_close, dtype=np.int64),
        count=count, prior_csm=np.zeros(n), lock_in_rate=0.0,
        prior_count=prior_count)
    return mp, state


# Profitable + onerous rows in one book, so the loss-component lines are alive.
PREMS = [120.0, 60.0, 120.0]
BENS = [480.0, 6000.0, 480.0]


# ---------------------------------------------------------------------------
# settle_aggregate: the per-MP settle sum, every line
# ---------------------------------------------------------------------------
def test_settle_aggregate_equals_per_mp_settle_sum():
    basis = _basis()
    mp, state = _multi_book(premiums=PREMS, benefits=BENS,
                            count_factor=[0.95, 0.97, 1.0])
    agg = settle_aggregate(mp, state, basis, period_months=3)
    per = fcf.paa.settle(mp, state, basis, period_months=3)
    for name in _PAA_LINES:
        np.testing.assert_allclose(
            getattr(agg, name), float(getattr(per, name).sum()),
            rtol=1e-10, atol=1e-9, err_msg=name)
    assert agg.measurement_basis == "settlement"
    assert agg.period_months == 3


def test_aggregate_chunk_size_is_a_numerical_noop():
    basis = _basis()
    mp, state = _multi_book(premiums=PREMS, benefits=BENS,
                            count_factor=[0.95, 0.97, 1.0])
    one = settle_aggregate(mp, state, basis, period_months=3, chunk_size=1)
    big = settle_aggregate(mp, state, basis, period_months=3, chunk_size=999)
    for name in _PAA_LINES:
        np.testing.assert_allclose(getattr(one, name), getattr(big, name),
                                   rtol=1e-12, err_msg=name)


def test_aggregate_reconcile_matches_per_mp_table():
    basis = _basis()
    mp, state = _multi_book(premiums=PREMS, benefits=BENS,
                            count_factor=[0.95, 0.97, 1.0])
    agg = settle_aggregate(mp, state, basis, period_months=3)
    per = fcf.paa.settle(mp, state, basis, period_months=3)
    r_agg = reconcile(agg)
    r_per = reconcile([per])[0]
    for f in ("lrc_opening", "premiums", "revenue", "lrc_closing",
              "loss_component_recognised", "loss_component_reversed",
              "lic_opening", "claims_incurred", "claims_paid", "lic_closing"):
        np.testing.assert_allclose(getattr(r_agg, f), getattr(r_per, f),
                                   rtol=1e-10, atol=1e-9, err_msg=f)


def test_aggregate_cannot_chain():
    basis = _basis()
    mp, state = _multi_book(premiums=PREMS, benefits=BENS)
    agg = settle_aggregate(mp, state, basis, period_months=3)
    with pytest.raises(ValueError, match="chain|closing_inputs"):
        agg.closing_inputs()


def test_aggregate_handles_settlement_pattern_book():
    """PAA accepts a settlement_pattern book (unlike gmm/vfa settle), so the
    LIC lines are alive in the aggregate. Mortality > 0 so claims are incurred
    and the settlement pattern carries an outstanding LIC."""
    basis = _basis(mortality_q=0.002, discount_annual=0.03,
                   settlement_pattern=np.array([0.6, 0.4]))
    mp, state = _multi_book(premiums=[60.0, 60.0], benefits=[6000.0, 6000.0],
                            count_factor=[0.97, 1.0])
    agg = settle_aggregate(mp, state, basis, period_months=3)
    per = fcf.paa.settle(mp, state, basis, period_months=3)
    assert abs(agg.lic_closing) > 0.0
    np.testing.assert_allclose(agg.lic_closing,
                               float(per.lic_closing.sum()), rtol=1e-10)


# ---------------------------------------------------------------------------
# settle_stream: out-of-core, matches the in-memory settle
# ---------------------------------------------------------------------------
def _write_paa_files(mp, state, tmp_path, *, combined):
    tmp_path.mkdir(parents=True, exist_ok=True)
    n = mp.n_mp
    spec = {
        "mp_id": np.asarray(mp.mp_id).astype(str),
        "issue_age": np.asarray(mp.issue_age),
        "premium": np.asarray(mp.premium),
        "term_months": np.asarray(mp.term_months),
        "premium_term_months": np.asarray(mp.premium_term_months),
        "product": np.asarray(mp.product).astype(str),
    }
    st = {
        "mp_id": np.asarray(state.mp_id).astype(str),
        "elapsed_months": np.asarray(state.elapsed_months),
        "count": np.asarray(state.count),
        # The settlement-stream state schema is uniform across models; the PAA
        # ignores prior_csm / lock_in_rate, so they ride as neutral values.
        "prior_csm": np.zeros(n),
        "lock_in_rate": np.zeros(n),
        "prior_count": np.asarray(state.prior_count),
    }
    cov = pl.DataFrame({
        "mp_id": spec["mp_id"], "coverage": ["DEATH"] * n,
        "amount": np.asarray(mp.benefits["DEATH"], dtype=np.float64),
    })
    cp = tmp_path / "coverages.parquet"
    cov.write_parquet(cp)
    if combined:
        ip = tmp_path / "inforce.parquet"
        pl.DataFrame({**spec, **{k: v for k, v in st.items()
                                 if k != "mp_id"}}).write_parquet(ip)
        return ip, cp, None
    ip = tmp_path / "policies.parquet"
    pl.DataFrame(spec).write_parquet(ip)
    sp = tmp_path / "state.parquet"
    pl.DataFrame(st).write_parquet(sp)
    return ip, cp, sp


def _parts(out_dir):
    return pl.read_parquet(str(out_dir / "part-*.parquet")).sort("id")


@pytest.mark.parametrize("combined", [True, False])
def test_settle_stream_matches_in_memory(tmp_path, combined):
    basis = _basis()
    mp, state = _multi_book(premiums=PREMS, benefits=BENS,
                            count_factor=[0.95, 0.97, 1.0])
    ip, cp, sp = _write_paa_files(mp, state, tmp_path, combined=combined)
    out = tmp_path / ("out_c" if combined else "out_s")
    n = settle_stream(ip, out, basis, coverages=cp,
                      calculation_methods=PATTERNS, state_path=sp,
                      period_months=3, chunk_size=2)
    assert n == mp.n_mp
    parts = _parts(out)
    mv = fcf.paa.settle(mp, state, basis, period_months=3)
    order = {str(i): k for k, i in enumerate(np.asarray(mp.mp_id).astype(str))}
    idx = [order[i] for i in parts["id"].to_list()]
    np.testing.assert_allclose(parts["lrc_closing"].to_numpy(),
                               np.asarray(mv.lrc_closing)[idx], rtol=1e-9,
                               atol=1e-9)
    np.testing.assert_allclose(parts["lic_closing"].to_numpy(),
                               np.asarray(mv.lic_closing)[idx], rtol=1e-9,
                               atol=1e-9)
