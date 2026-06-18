"""fcf.samples.* -- the single packaged-sample surface.

Replaces the old per-file ``save_sample_*`` helpers: :func:`samples.export`
writes a template's starter files (in a chosen format) to a directory, and the
loaders return assembled objects. A round-trip through ``read_*`` must
reproduce the bundled in-memory sample.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow.gmm import measure


def test_templates_lists_available():
    assert fcf.samples.templates() == [
        "gmm", "vfa", "paa", "ul", "ul-annuity", "ul-cost-deduct",
        "ul-var-annuity", "annuity"]


def test_export_gmm_round_trips(tmp_path):
    """export writes the gmm set; reading it back reproduces the bundled
    sample's measurement."""
    fcf.samples.export(tmp_path, template="gmm")
    for name in ("basis.xlsx", "policies.csv", "coverages.csv",
                 "calculation_methods.csv", "inforce_state.csv",
                 "inforce_policies.csv"):
        assert (tmp_path / name).exists(), name
    mp = fcf.read_model_points(
        tmp_path / "policies.csv",
        coverages=tmp_path / "coverages.csv",
        calculation_methods=tmp_path / "calculation_methods.csv")
    basis = fcf.read_basis(tmp_path / "basis.xlsx")
    a = measure(mp, basis, full=False)
    b = measure(fcf.samples.model_points(), fcf.samples.basis(), full=False)
    assert np.allclose(a.bel, b.bel) and np.allclose(a.csm, b.csm)


def test_export_combined_inforce_round_trips(tmp_path):
    """The combined inforce_policies file reads back via read_inforce_policies
    with the period-close state folded in."""
    fcf.samples.export(tmp_path, template="gmm")
    mp, state = fcf.read_inforce_policies(
        tmp_path / "inforce_policies.csv",
        coverages=tmp_path / "coverages.csv",
        calculation_methods=tmp_path / "calculation_methods.csv")
    assert mp.n_mp == state.elapsed_months.shape[0]
    assert np.all(np.asarray(mp.elapsed_months) > 0)  # state folded in


@pytest.mark.parametrize("fmt,ext", [("csv", ".csv"), ("parquet", ".parquet"),
                                     ("feather", ".feather"), ("xlsx", ".xlsx")])
def test_export_format_picks_data_extension(tmp_path, fmt, ext):
    """format= sets the data-file extension; the basis stays .xlsx; reads back."""
    fcf.samples.export(tmp_path, template="gmm", format=fmt)
    assert (tmp_path / "basis.xlsx").exists()
    assert (tmp_path / f"policies{ext}").exists()
    mp = fcf.read_model_points(
        tmp_path / f"policies{ext}",
        coverages=tmp_path / f"coverages{ext}",
        calculation_methods=tmp_path / f"calculation_methods{ext}")
    assert mp.n_mp == 11


def test_export_vfa(tmp_path):
    fcf.samples.export(tmp_path, template="vfa")
    assert (tmp_path / "basis.xlsx").exists()
    assert (tmp_path / "policies.csv").exists()


def test_export_returns_directory(tmp_path):
    out = fcf.samples.export(tmp_path / "fresh", template="gmm")
    assert out == tmp_path / "fresh" and out.is_dir()


def test_export_rejects_unknown_template_and_format(tmp_path):
    with pytest.raises(ValueError, match="template must be one of"):
        fcf.samples.export(tmp_path, template="gi")
    with pytest.raises(ValueError, match="format must be one of"):
        fcf.samples.export(tmp_path, format="json")
    with pytest.raises(ValueError, match="template must be one of"):
        fcf.samples.basis(template="gi")


def test_export_paa(tmp_path):
    fcf.samples.export(tmp_path, template="paa")
    assert (tmp_path / "basis.xlsx").exists()
    assert (tmp_path / "policies.csv").exists()
    assert (tmp_path / "coverages.csv").exists()


def test_sample_supports_group_of_contracts_cohorts():
    """The bundled sample carries issue_date, so group_of_contracts splits by
    annual cohort (IFRS 17 Sec. 22) -- 2025 and 2026 here -- on top of the
    portfolio (product) and the derived onerous / remaining profitability."""
    m = measure(fcf.samples.model_points(), fcf.samples.basis())
    g = fcf.group_of_contracts(m)
    cohorts = {str(lab).split("|")[1] for lab in g.group_labels}
    assert cohorts == {"2025", "2026"}


def test_sample_return_scenarios_drive_tvog():
    """The toy fund-return scenarios feed vfa.measure and wake up the guarantee
    time value: the deterministic measure has TVOG 0, with scenarios positive.
    Deterministic across calls (fixed seed)."""
    scen = fcf.samples.return_scenarios()
    mp = fcf.samples.model_points("vfa")
    assert scen.ndim == 2 and scen.shape[0] == 1000          # 1,000 toy paths
    assert scen.shape[1] == int(np.asarray(mp.term_months).max())
    assert np.allclose(scen, fcf.samples.return_scenarios())  # reproducible
    basis = fcf.samples.basis("vfa")
    det = fcf.vfa.measure(mp, basis)
    sto = fcf.vfa.measure(mp, basis, return_scenarios=scen)
    assert np.allclose(det.time_value, 0.0)
    assert np.all(sto.time_value > 0.0)


def test_sample_return_scenarios_reject_non_vfa():
    """return_scenarios are a variable-contract input -- non-vfa is rejected."""
    with pytest.raises(ValueError, match="VFA"):
        fcf.samples.return_scenarios(template="gmm")


def test_sample_rate_scenarios_drive_stochastic():
    """The toy discount-rate scenarios feed gmm.stochastic and produce a BEL
    distribution across rates (the rate counterpart to return_scenarios)."""
    rates = fcf.samples.rate_scenarios()
    assert rates.ndim == 1 and rates.shape[0] == 1000        # 1,000 flat rates
    assert np.all(rates > 0.0)
    assert np.allclose(rates, fcf.samples.rate_scenarios())   # reproducible
    mp = fcf.samples.model_points()
    basis = fcf.samples.basis().resolve(("TERM_LIFE_A", "GA"))
    res = fcf.gmm.stochastic(mp, basis, rates)
    assert res.bel.shape[0] == 1000                          # one BEL per scenario
    assert np.all(np.isfinite(res.bel))


def test_treaty_is_a_bundled_quota_share():
    """samples.treaty() returns the bundled quota-share treaty (a parameter
    object, not a data file); cession defaults to 30% and is overridable."""
    t = fcf.samples.treaty()
    assert isinstance(t, fcf.reinsurance.QuotaShare)
    assert t.cession == pytest.approx(0.30)
    assert fcf.samples.treaty(0.5).cession == pytest.approx(0.50)


def test_close_pack_nets_reinsurance_on_the_bundled_book():
    """A quota-share cession of a bundled segment, settled alongside the issued
    book, reduces the close pack's net carrying amount -- the reinsurance
    recoverable is added in the one signed frame (net == issued + reins)."""
    import polars as pl
    from fastcashflow import InforceState

    basis  = fcf.samples.basis()
    book   = fcf.samples.model_points()
    state  = fcf.samples.inforce_state()
    treaty = fcf.samples.treaty()
    segment   = ("TERM_LIFE_A", "FC")
    seg_basis = basis.resolve(segment)
    rows = np.flatnonzero((book.product == segment[0]) & (book.channel == segment[1]))
    mp   = book.subset(rows)
    st   = state.subset(np.flatnonzero(np.isin(state.mp_id, mp.mp_id)))
    valued = fcf.apply_inforce_state(mp, st)
    period = 12

    issued = fcf.reconcile([fcf.gmm.settle(valued, st, seg_basis, period_months=period)])[0]
    reins_m = fcf.reinsurance.measure(mp, seg_basis, treaty=treaty)
    opening = np.asarray(st.elapsed_months) - period
    re_state = InforceState(
        mp_id=st.mp_id, elapsed_months=st.elapsed_months, count=st.count,
        prior_csm=reins_m.csm_path[np.arange(mp.mp_id.shape[0]), opening],
        lock_in_rate=st.lock_in_rate, prior_count=st.prior_count)
    held = fcf.reconcile([fcf.reinsurance.settle(
        valued, re_state, seg_basis, treaty=treaty, period_months=period)])[0]

    pack = fcf.close([issued, held], group_ids=["issued", "reins"])
    sofp = pack.sofp

    def total(kind):
        r = sofp.filter((pl.col("kind") == kind) & (pl.col("component") == "Total"))
        return float(r["closing"][0])

    iss = total("Insurance contracts issued")
    rei = total("Reinsurance contracts held")
    net = total("Net")
    # net is the signed sum (the P0 fix: add, do not subtract)
    assert net == pytest.approx(iss + rei)
    # this cession is a recoverable (negative carrying), so it lowers the net
    assert rei < 0.0
    assert net < iss
