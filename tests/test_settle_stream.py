"""gmm/vfa.settle_stream -- out-of-core period-close settlement (skeleton).

Authoritative skeleton (P-5c pattern): written before the implementation and
activated unchanged by it. The anchor facts, from dev/inforce-redesign-FINAL.md
(stage 4, decisions B1/B2 adopted from the superseded scale contract):

* Input layouts: ONE combined file (policies spec + closing-state columns,
  the industry period-close snapshot -- primary) or TWO files (policies
  parquet + state parquet, per-chunk mp_id semi-join). Both give identical
  output.
* A semi-join hides BOTH missing and surplus rows, so the two-file path
  guards the GLOBAL id sets bidirectionally up front; a duplicate mp_id in
  the state file is rejected like one in the policies file.
* Output: one part-NNNNN.parquet per chunk through the settlement write
  arms -- every movement line plus the markers (measurement_basis,
  elapsed_months) plus the closing-state columns (count, lock_in_rate,
  and for the VFA account_value_closing) so the NEXT period's state can be
  assembled from the parts alone: the disk side of the closing_inputs()
  chain.
* Part rows concatenated over chunks equal the in-memory per-MP movement
  row for row; lock_in_rate must be uniform across the whole book (v1
  scalar), validated globally, not per chunk.
"""
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

import fastcashflow as fcf
from fastcashflow import (
    Basis, CalculationMethod, CoverageRate, InforceState, ModelPoints)

pytestmark = pytest.mark.skipif(
    getattr(fcf.gmm, "settle_stream", None) is None
    or getattr(fcf.vfa, "settle_stream", None) is None,
    reason="settle_stream not implemented yet (redesign stage 4; skeleton "
           "activates unchanged once it lands)")

CM = {"DEATH": CalculationMethod.DEATH}


def _flat(value):
    def fn(sex, issue_age, duration):
        return np.full(duration.shape, value, dtype=np.float64)
    return fn


_GMM_LINES = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_experience_unlocking",
    "finance_wedge", "csm_release", "csm_closing",
    "loss_component_opening", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
    "coverage_units_provided", "coverage_units_future",
)

_VFA_LINES = (
    "bel_opening", "bel_interest", "bel_release", "bel_experience",
    "bel_closing",
    "ra_opening", "ra_interest", "ra_release", "ra_experience", "ra_closing",
    "csm_opening", "csm_accretion", "csm_fv_share", "csm_future_service",
    "csm_release", "csm_closing",
    "loss_component_opening", "loss_component_reversed",
    "loss_component_recognised", "loss_component_closing",
    "variable_fee_closing", "account_value_closing",
    "coverage_units_provided", "coverage_units_future",
)


# ---------------------------------------------------------------------------
# GMM fixtures -- a 7-row heterogeneous in-force book, in memory and on disk
# ---------------------------------------------------------------------------
def _gmm_basis():
    return Basis(
        mortality_annual=_flat(0.012), lapse_annual=_flat(0.05),
        discount_annual=0.03, ra_confidence=0.75, mortality_cv=0.10,
        coverages=(CoverageRate("DEATH", _flat(0.012)),),
    )


def _gmm_book(n=7, *, em_close=24, lock_in=0.03):
    """In-memory (mp, state): heterogeneous balances, CSM xor LC per row."""
    rng = np.random.default_rng(7)
    ids = np.array([f"P{i}" for i in range(n)])
    rep = lambda v: np.full(n, v)
    prior_count = rng.uniform(500.0, 2000.0, n).round(1)
    count_close = (prior_count * rng.uniform(0.85, 0.99, n)).round(1)
    prior_csm = np.where(np.arange(n) % 3 == 1, 0.0,
                         rng.uniform(1_000.0, 30_000.0, n).round(2))
    lc_open = np.where(prior_csm == 0.0,
                       rng.uniform(500.0, 5_000.0, n).round(2), 0.0)
    mp = ModelPoints(
        issue_age=rng.integers(30, 55, n), premium=rep(100.0),
        term_months=rep(36).astype(np.int64), benefits={0: rep(1e6)},
        count=count_close, elapsed_months=rep(em_close).astype(np.int64),
        mp_id=ids, product=np.full(n, "A"), calculation_methods=CM,
    )
    state = InforceState(
        mp_id=ids, elapsed_months=rep(em_close).astype(np.int64),
        count=count_close, prior_csm=prior_csm, lock_in_rate=lock_in,
        prior_count=prior_count, prior_loss_component=lc_open,
    )
    return mp, state


def _write_gmm_files(mp, state, tmp_path, *, combined, shuffle_state=False,
                     state_tweak=None, lock_override=None):
    """Write the book to disk; returns (input_path, coverages_path,
    state_path-or-None)."""
    n = mp.n_mp
    spec = {
        "mp_id": np.asarray(mp.mp_id).astype(str),
        "issue_age": np.asarray(mp.issue_age),
        "premium": np.asarray(mp.premium),
        "term_months": np.asarray(mp.term_months),
    }
    lock = np.full(n, state.lock_in_rate if lock_override is None
                   else np.nan)
    if lock_override is not None:
        lock = np.asarray(lock_override, dtype=np.float64)
    st = {
        "mp_id": np.asarray(state.mp_id).astype(str),
        "elapsed_months": np.asarray(state.elapsed_months),
        "count": np.asarray(state.count),
        "prior_csm": np.asarray(state.prior_csm),
        "lock_in_rate": lock,
        "prior_count": np.asarray(state.prior_count),
        "prior_loss_component": np.asarray(state.prior_loss_component),
    }
    if state_tweak is not None:
        st = state_tweak(st)
    cov = pl.DataFrame({
        "mp_id": spec["mp_id"], "coverage": ["DEATH"] * n,
        "amount": np.full(n, 1e6),
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
    sdf = pl.DataFrame(st)
    if shuffle_state:
        sdf = sdf.reverse()
    sp = tmp_path / "state.parquet"
    sdf.write_parquet(sp)
    return ip, cp, sp


def _parts(out_dir):
    return pl.read_parquet(str(out_dir / "part-*.parquet")).sort("id")


# ---------------------------------------------------------------------------
# VFA fixtures -- a 5-row account-value book (no coverages frame)
# ---------------------------------------------------------------------------
def _vfa_basis():
    return Basis(
        mortality_annual=_flat(0.012), lapse_annual=_flat(0.05),
        discount_annual=0.05, ra_confidence=0.75, mortality_cv=0.0,
        expense_cv=0.10, investment_return=0.05, fund_fee=0.015,
    )


def _vfa_book(n=5, *, em_close=18):
    rng = np.random.default_rng(11)
    ids = np.array([f"V{i}" for i in range(n)])
    rep = lambda v: np.full(n, v)
    prior_count = rng.uniform(0.7, 1.0, n).round(3)
    count_close = (prior_count * rng.uniform(0.85, 0.99, n)).round(3)
    av_open = rng.uniform(5e5, 2e6, n).round(0)
    av_close = (av_open * rng.uniform(0.95, 1.15, n)).round(0)
    prior_csm = np.where(np.arange(n) % 2 == 1, 0.0,
                         rng.uniform(1_000.0, 20_000.0, n).round(2))
    lc_open = np.where(prior_csm == 0.0,
                       rng.uniform(500.0, 3_000.0, n).round(2), 0.0)
    mp = ModelPoints(
        issue_age=rng.integers(35, 55, n), premium=rep(0.0),
        term_months=rep(36).astype(np.int64), count=count_close,
        elapsed_months=rep(em_close).astype(np.int64), mp_id=ids,
        account_value=av_close, calculation_methods=CM,
    )
    state = InforceState(
        mp_id=ids, elapsed_months=rep(em_close).astype(np.int64),
        count=count_close, prior_csm=prior_csm, lock_in_rate=0.0,
        account_value=av_close, prior_count=prior_count,
        prior_account_value=av_open, prior_loss_component=lc_open,
    )
    return mp, state


def _write_vfa_files(mp, state, tmp_path, *, combined):
    n = mp.n_mp
    spec = {
        "mp_id": np.asarray(mp.mp_id).astype(str),
        "issue_age": np.asarray(mp.issue_age),
        "term_months": np.asarray(mp.term_months),
    }
    st = {
        "mp_id": spec["mp_id"],
        "elapsed_months": np.asarray(state.elapsed_months),
        "count": np.asarray(state.count),
        "prior_csm": np.asarray(state.prior_csm),
        "lock_in_rate": np.full(n, 0.0),
        "account_value": np.asarray(state.account_value),
        "prior_count": np.asarray(state.prior_count),
        "prior_account_value": np.asarray(state.prior_account_value),
        "prior_loss_component": np.asarray(state.prior_loss_component),
    }
    if combined:
        ip = tmp_path / "vfa_inforce.parquet"
        pl.DataFrame({**spec, **{k: v for k, v in st.items()
                                 if k != "mp_id"}}).write_parquet(ip)
        return ip, None
    ip = tmp_path / "vfa_policies.parquet"
    # the two-file spec side still carries the account-value column (a
    # policies column); the OBSERVED closing value rides on the state file
    pl.DataFrame({**spec,
                  "account_value": np.asarray(state.account_value),
                  }).write_parquet(ip)
    sp = tmp_path / "vfa_state.parquet"
    pl.DataFrame(st).write_parquet(sp)
    return ip, sp


# ---------------------------------------------------------------------------
# part concat == in-memory, with markers and chain columns
# ---------------------------------------------------------------------------
def test_gmm_stream_matches_in_memory(tmp_path):
    basis = _gmm_basis()
    mp, state = _gmm_book()
    mv = fcf.gmm.settle(mp, state, basis, period_months=12)
    ip, cp, _ = _write_gmm_files(mp, state, tmp_path, combined=True)
    out = tmp_path / "out"
    n = fcf.gmm.settle_stream(ip, out, basis, coverages=cp,
                              calculation_methods=CM, period_months=12,
                              chunk_size=3)
    assert n == mp.n_mp
    assert len(sorted(out.glob("part-*.parquet"))) == 3   # 3 + 3 + 1
    df = _parts(out)
    assert df["id"].to_list() == sorted(np.asarray(mp.mp_id).tolist())
    order = np.argsort(np.asarray(mp.mp_id).astype(str))
    for name in _GMM_LINES:
        np.testing.assert_allclose(df[name].to_numpy(),
                                   getattr(mv, name)[order],
                                   rtol=1e-12, err_msg=name)
    # markers + the closing-state chain columns
    assert df["measurement_basis"].to_list() == ["settlement"] * mp.n_mp
    np.testing.assert_array_equal(df["elapsed_months"].to_numpy(),
                                  np.asarray(mp.elapsed_months)[order])
    np.testing.assert_allclose(df["count"].to_numpy(),
                               np.asarray(mp.count)[order], rtol=1e-12)
    np.testing.assert_allclose(df["lock_in_rate"].to_numpy(),
                               state.lock_in_rate, rtol=1e-12)


def test_vfa_stream_matches_in_memory(tmp_path):
    basis = _vfa_basis()
    mp, state = _vfa_book()
    mv = fcf.vfa.settle(mp, state, basis, period_months=12)
    ip, _ = _write_vfa_files(mp, state, tmp_path, combined=True)
    out = tmp_path / "out"
    n = fcf.vfa.settle_stream(ip, out, basis, period_months=12, chunk_size=2)
    assert n == mp.n_mp
    df = _parts(out)
    order = np.argsort(np.asarray(mp.mp_id).astype(str))
    for name in _VFA_LINES:
        np.testing.assert_allclose(df[name].to_numpy(),
                                   getattr(mv, name)[order],
                                   rtol=1e-12, err_msg=name)
    assert df["measurement_basis"].to_list() == ["settlement"] * mp.n_mp
    np.testing.assert_allclose(df["count"].to_numpy(),
                               np.asarray(mp.count)[order], rtol=1e-12)


# ---------------------------------------------------------------------------
# two-file == one-file (the state file in its own row order)
# ---------------------------------------------------------------------------
def test_gmm_two_file_equals_one_file(tmp_path):
    basis = _gmm_basis()
    mp, state = _gmm_book()
    ip1, cp, _ = _write_gmm_files(mp, state, tmp_path / "a", combined=True)
    out1 = tmp_path / "a" / "out"
    fcf.gmm.settle_stream(ip1, out1, basis, coverages=cp,
                          calculation_methods=CM, period_months=12,
                          chunk_size=3)
    ip2, cp2, sp = _write_gmm_files(mp, state, tmp_path / "b", combined=False,
                                    shuffle_state=True)
    out2 = tmp_path / "b" / "out"
    fcf.gmm.settle_stream(ip2, out2, basis, coverages=cp2,
                          calculation_methods=CM, state_path=sp,
                          period_months=12, chunk_size=3)
    one, two = _parts(out1), _parts(out2)
    assert one.columns == two.columns
    for name in one.columns:
        if name in ("id", "measurement_basis"):
            assert one[name].to_list() == two[name].to_list()
        else:
            np.testing.assert_allclose(one[name].to_numpy(),
                                       two[name].to_numpy(),
                                       rtol=1e-12, err_msg=name)


def test_vfa_two_file_equals_one_file(tmp_path):
    basis = _vfa_basis()
    mp, state = _vfa_book()
    ip1, _ = _write_vfa_files(mp, state, tmp_path / "a", combined=True)
    out1 = tmp_path / "a" / "out"
    fcf.vfa.settle_stream(ip1, out1, basis, period_months=12, chunk_size=2)
    ip2, sp = _write_vfa_files(mp, state, tmp_path / "b", combined=False)
    out2 = tmp_path / "b" / "out"
    fcf.vfa.settle_stream(ip2, out2, basis, state_path=sp, period_months=12,
                          chunk_size=2)
    one, two = _parts(out1), _parts(out2)
    assert one.columns == two.columns
    for name in one.columns:
        if name in ("id", "measurement_basis"):
            assert one[name].to_list() == two[name].to_list()
        else:
            np.testing.assert_allclose(one[name].to_numpy(),
                                       two[name].to_numpy(),
                                       rtol=1e-12, err_msg=name)


# ---------------------------------------------------------------------------
# guards: bidirectional id sets, duplicate state ids, uniform lock-in,
# fresh output directory
# ---------------------------------------------------------------------------
def test_state_id_set_must_match_bidirectionally(tmp_path):
    basis = _gmm_basis()
    mp, state = _gmm_book()
    # missing: drop one state row -- the semi-join would silently starve it
    ip, cp, sp = _write_gmm_files(
        mp, state, tmp_path / "miss", combined=False,
        state_tweak=lambda st: {k: v[:-1] for k, v in st.items()})
    with pytest.raises(ValueError, match="state"):
        fcf.gmm.settle_stream(ip, tmp_path / "miss" / "out", basis,
                              coverages=cp, calculation_methods=CM,
                              state_path=sp, period_months=12)
    # surplus: add a state row for a contract not in the policies file --
    # the semi-join would silently ignore it
    def add_row(st):
        return {k: np.concatenate([v, v[-1:]]) if k != "mp_id"
                else np.concatenate([v, np.array(["GHOST"])])
                for k, v in st.items()}
    ip, cp, sp = _write_gmm_files(mp, state, tmp_path / "extra",
                                  combined=False, state_tweak=add_row)
    with pytest.raises(ValueError, match="state"):
        fcf.gmm.settle_stream(ip, tmp_path / "extra" / "out", basis,
                              coverages=cp, calculation_methods=CM,
                              state_path=sp, period_months=12)


def test_duplicate_state_mp_id_rejected(tmp_path):
    basis = _gmm_basis()
    mp, state = _gmm_book()

    def dup_row(st):
        return {k: np.concatenate([v, v[-1:]]) for k, v in st.items()}

    ip, cp, sp = _write_gmm_files(mp, state, tmp_path, combined=False,
                                  state_tweak=dup_row)
    with pytest.raises(ValueError, match="duplicate"):
        fcf.gmm.settle_stream(ip, tmp_path / "out", basis, coverages=cp,
                              calculation_methods=CM, state_path=sp,
                              period_months=12)


def test_lock_in_rate_must_be_uniform_across_the_whole_book(tmp_path):
    """v1 scalar lock-in: validated GLOBALLY (a per-chunk check would pass a
    book whose rates differ only across chunks)."""
    basis = _gmm_basis()
    mp, state = _gmm_book()
    lock = np.full(mp.n_mp, 0.03)
    lock[-1] = 0.04                     # different rate in the LAST row --
    ip, cp, _ = _write_gmm_files(       # only a later chunk would see it
        mp, state, tmp_path, combined=True, lock_override=lock)
    with pytest.raises(NotImplementedError, match="lock_in_rate"):
        fcf.gmm.settle_stream(ip, tmp_path / "out", basis, coverages=cp,
                              calculation_methods=CM, period_months=12,
                              chunk_size=3)


def test_rejects_existing_output_parts(tmp_path):
    basis = _gmm_basis()
    mp, state = _gmm_book()
    ip, cp, _ = _write_gmm_files(mp, state, tmp_path, combined=True)
    out = tmp_path / "out"
    fcf.gmm.settle_stream(ip, out, basis, coverages=cp,
                          calculation_methods=CM, period_months=12)
    with pytest.raises(ValueError, match="part"):
        fcf.gmm.settle_stream(ip, out, basis, coverages=cp,
                              calculation_methods=CM, period_months=12)


# ---------------------------------------------------------------------------
# the disk side of the closing_inputs() chain
# ---------------------------------------------------------------------------
def test_part_columns_seed_the_next_period(tmp_path):
    """A part file carries every column the next period's state needs: the
    state assembled from the part (plus the next observation) produces the
    SAME second-period movement as chaining in memory via closing_inputs()."""
    basis = _gmm_basis()
    mp, state = _gmm_book()
    ip, cp, _ = _write_gmm_files(mp, state, tmp_path, combined=True)
    out = tmp_path / "out"
    fcf.gmm.settle_stream(ip, out, basis, coverages=cp,
                          calculation_methods=CM, period_months=6,
                          chunk_size=3)
    df = _parts(out)

    # in-memory chain: settle 6m, closing_inputs, advance on-track 6m more
    mv1 = fcf.gmm.settle(mp, state, basis, period_months=6)
    mp_mid, state_mid = mv1.closing_inputs()
    em_next = np.asarray(mp.elapsed_months) + 6
    count_next = np.asarray(mp.count) * 0.95          # observed next close
    mp_next = replace(mp_mid, elapsed_months=em_next, count=count_next)
    state_next = replace(state_mid, elapsed_months=em_next, count=count_next)
    mv2_mem = fcf.gmm.settle(mp_next, state_next, basis, period_months=6)

    # disk chain: the SAME advance, but the prior balances read off the part
    order = np.argsort(np.asarray(mp.mp_id).astype(str))
    back = np.argsort(order)                          # part order -> mp order
    state_disk = InforceState(
        mp_id=df["id"].to_numpy()[back],
        elapsed_months=em_next,
        count=count_next,
        prior_csm=df["csm_closing"].to_numpy()[back],
        lock_in_rate=float(df["lock_in_rate"][0]),
        prior_count=df["count"].to_numpy()[back],
        prior_loss_component=df["loss_component_closing"].to_numpy()[back],
    )
    mv2_disk = fcf.gmm.settle(mp_next, state_disk, basis, period_months=6)
    for name in _GMM_LINES:
        np.testing.assert_allclose(getattr(mv2_disk, name),
                                   getattr(mv2_mem, name),
                                   rtol=1e-12, err_msg=name)
