"""Phase 3b validation -- polars file I/O for model points and results."""
import fastcashflow as fcf
import numpy as np
import polars as pl
import pytest

from fastcashflow import ExpenseItem, ModelPoints, read_model_points, sample_data_dir, write_measurement
from fastcashflow.gmm import measure
from conftest import (PATTERNS, annual_from_monthly as _annual,
                      make_death_basis, mp_to_frames)


def _portfolio(n: int = 400) -> ModelPoints:
    rng = np.random.default_rng(3)
    return ModelPoints(
        issue_age=rng.integers(25, 60, n),
        benefits={0: rng.integers(10, 100, n) * 1_000_000},
        premium=rng.integers(3, 15, n) * 10_000,
        term_months=rng.integers(60, 180, n),
        calculation_methods=PATTERNS,
    )


def _death_benefits(mps: ModelPoints) -> np.ndarray:
    """Reconstruct the per-policy death benefit from the CSR -- the
    post-(B) replacement for the previous ``ModelPoints.death_benefit``
    field. Sums every coverage_amount whose coverage_index is 0 (the
    DEATH coverage in these tests)."""
    mp_of_cov = np.repeat(np.arange(mps.n_mp), np.diff(mps.coverage_offset))
    mask = mps.coverage_index == 0
    return np.bincount(mp_of_cov[mask], weights=mps.coverage_amount[mask],
                       minlength=mps.n_mp)


def _assumptions():
    return make_death_basis(
        mortality_q       = 0.001,
        lapse_q           = 0.01,
        discount_annual   = 0.03,
        expense_inflation = 0.02,
        expense_items     = (
            ExpenseItem("acquisition",  "alpha_fixed",    200_000.0),
            ExpenseItem("maintenance",  "gamma_fixed",  48_000.0),
        ),
        ra_confidence     = 0.85,
        mortality_cv      = 0.10,
    )


def _write_model_points(mps: ModelPoints, tmp_path, suffix: str = ".parquet", **policy_extra):
    """Write ``mps`` as a policies + coverages pair; return the two paths.

    The portfolio carries a single DEATH coverage (the only code registered in
    these tests). ``policy_extra`` adds optional policy columns (e.g. count).
    """
    n = mps.n_mp
    tmp_path.mkdir(parents=True, exist_ok=True)
    policies = pl.DataFrame({
        "mp_id": np.arange(n),
        "issue_age": mps.issue_age,
        "premium": mps.premium,
        "term_months": mps.term_months,
        **policy_extra,
    })
    coverages = pl.DataFrame({
        "mp_id": np.arange(n),
        "coverage": ["DEATH"] * n,
        "amount": _death_benefits(mps),
    })
    pp = tmp_path / f"policies{suffix}"
    cp = tmp_path / f"coverages{suffix}"
    if suffix == ".parquet":
        policies.write_parquet(pp); coverages.write_parquet(cp)
    else:
        policies.write_csv(pp); coverages.write_csv(cp)
    return pp, cp


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_model_points_round_trip(tmp_path, suffix):
    """read_model_points reconstructs a ModelPoints written to disk."""
    mps = _portfolio()
    pp, cp = _write_model_points(mps, tmp_path, suffix)

    loaded = read_model_points(pp, coverages=cp, calculation_methods=PATTERNS)
    assert loaded.n_mp == mps.n_mp
    assert np.allclose(loaded.issue_age, mps.issue_age)
    assert np.allclose(_death_benefits(loaded), _death_benefits(mps))
    assert np.allclose(loaded.premium, mps.premium)
    assert np.allclose(loaded.term_months, mps.term_months)


def test_read_model_points_reads_count(tmp_path):
    """read_model_points reads an optional count column, else defaults to one."""
    mps = _portfolio(50)
    counts = np.arange(1, mps.n_mp + 1, dtype=float)
    pp, cp = _write_model_points(mps, tmp_path / "with", ".parquet", count=counts)
    assert np.allclose(
        read_model_points(pp, coverages=cp, calculation_methods=PATTERNS).count, counts
    )

    pp2, cp2 = _write_model_points(mps, tmp_path / "no", ".parquet")
    assert np.allclose(
        read_model_points(pp2, coverages=cp2, calculation_methods=PATTERNS).count,
        np.ones(mps.n_mp),
    )


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_write_measurement_round_trip(tmp_path, suffix):
    """write_measurement persists the four valuation columns with an id column."""
    mps = _portfolio()
    val = measure(mps, _assumptions(), full=False)
    ids = np.arange(mps.n_mp)

    path = tmp_path / f"out{suffix}"
    write_measurement(val, path, ids=ids)
    df = pl.read_parquet(path) if suffix == ".parquet" else pl.read_csv(path)

    assert df.columns == ["id", "bel", "ra", "csm", "loss_component"]
    assert np.allclose(df["id"].to_numpy(), ids)
    assert np.allclose(df["bel"].to_numpy(), val.bel)
    assert np.allclose(df["ra"].to_numpy(), val.ra)
    assert np.allclose(df["csm"].to_numpy(), val.csm)
    assert np.allclose(df["loss_component"].to_numpy(), val.loss_component)


def test_write_measurement_without_ids(tmp_path):
    """ids are optional -- omitting them writes just the four result columns."""
    val = measure(_portfolio(50), _assumptions(), full=False)
    path = tmp_path / "out.parquet"
    write_measurement(val, path)
    assert pl.read_parquet(path).columns == ["bel", "ra", "csm", "loss_component"]


def test_write_measurement_dispatches_per_model(tmp_path):
    """write_measurement is singledispatch: PAA writes lrc, VFA adds
    variable_fee / time_value, and an unsupported type errors loudly."""
    import fastcashflow as fcf
    import numpy as np
    mp = fcf.samples.model_points()
    b = fcf.samples.basis()[("HEALTH_A", "FC")]
    idx = np.where((np.asarray(mp.product) == "HEALTH_A") &
                   (np.asarray(mp.channel) == "FC"))[0]
    sub = mp.subset(idx)

    fcf.write_measurement(fcf.paa.measure(sub, b), tmp_path / "paa.parquet")
    assert pl.read_parquet(tmp_path / "paa.parquet").columns == [
        "lrc", "loss_component"]

    vmp = fcf.samples.model_points(template="vfa")
    vb = fcf.samples.basis(template="vfa")
    fcf.write_measurement(fcf.vfa.measure(vmp, vb), tmp_path / "vfa.parquet")
    assert pl.read_parquet(tmp_path / "vfa.parquet").columns == [
        "bel", "ra", "csm", "variable_fee", "time_value", "loss_component"]

    with pytest.raises(TypeError, match="does not handle"):
        fcf.write_measurement("not a measurement", tmp_path / "x.csv")


def test_measurement_equality_is_identity():
    """Measurement results carry numpy arrays, for which the default dataclass
    == / hash raise (ambiguous truth value / unhashable). They are eq=False, so
    equality and hashing fall back to identity -- no raise, usable in a set."""
    import fastcashflow as fcf
    import numpy as np
    mp = fcf.samples.model_points()
    b = fcf.samples.basis()[("HEALTH_A", "FC")]
    idx = np.where((np.asarray(mp.product) == "HEALTH_A") &
                   (np.asarray(mp.channel) == "FC"))[0]
    m = fcf.gmm.measure(mp.subset(idx), b)
    m2 = fcf.gmm.measure(mp.subset(idx), b)
    assert m == m and m != m2          # identity, not array compare (no ValueError)
    assert len({m, m2}) == 2           # hashable by identity


def test_read_ignores_extra_columns_and_flags_missing(tmp_path):
    """Extra columns are ignored; a missing required column is an error."""
    mps = _portfolio(50)
    pp, cp = _write_model_points(mps, tmp_path, ".parquet")
    # extra columns on the policies frame are ignored
    pl.read_parquet(pp).with_columns(pl.lit(1).alias("junk")).write_parquet(pp)
    assert read_model_points(
        pp, coverages=cp, calculation_methods=PATTERNS
    ).n_mp == mps.n_mp

    # a coverages frame missing a required column is an error
    pl.DataFrame({"mp_id": [0], "coverage": ["DEATH"]}).write_parquet(
        tmp_path / "bad_cov.parquet")
    with pytest.raises(ValueError, match="missing required column"):
        read_model_points(pp, coverages=tmp_path / "bad_cov.parquet",
                           calculation_methods=PATTERNS)


def test_file_workflow_matches_in_memory(tmp_path):
    """A file round-trip produces the same valuation as the in-memory path."""
    mps = _portfolio()
    basis = _assumptions()
    pp, cp = _write_model_points(mps, tmp_path, ".parquet")

    from_file = measure(read_model_points(pp, coverages=cp, calculation_methods=PATTERNS), basis, full=False)
    in_memory = measure(mps, basis, full=False)

    assert np.allclose(from_file.bel, in_memory.bel)
    assert np.allclose(from_file.ra, in_memory.ra)
    assert np.allclose(from_file.csm, in_memory.csm)
    assert np.allclose(from_file.loss_component, in_memory.loss_component)


def test_measure_stream_streaming_matches_in_memory(tmp_path):
    """Chunked file-to-file valuation equals the in-memory valuation exactly."""
    rng = np.random.default_rng(5)
    n = 1000
    mps = ModelPoints(
        issue_age=rng.integers(25, 60, n),
        benefits={0: rng.integers(10, 100, n) * 1_000_000},
        premium=rng.integers(3, 15, n) * 10_000,
        term_months=rng.integers(60, 180, n),
        calculation_methods=PATTERNS,
    )
    basis = _assumptions()

    pp, cp = _write_model_points(mps, tmp_path, ".parquet")

    out_dir = tmp_path / "results"
    processed = fcf.gmm.measure_stream(pp, out_dir, basis, coverages=cp, chunk_size=300,
                           calculation_methods=PATTERNS)
    assert processed == n
    assert len(sorted(out_dir.glob("part-*.parquet"))) == 4  # 300+300+300+100

    results = pl.read_parquet(str(out_dir / "part-*.parquet")).sort("id")
    in_memory = measure(mps, basis, full=False)
    assert np.array_equal(results["id"].to_numpy(), np.arange(n))
    assert np.array_equal(results["bel"].to_numpy(), in_memory.bel)
    assert np.array_equal(results["ra"].to_numpy(), in_memory.ra)
    assert np.array_equal(results["csm"].to_numpy(), in_memory.csm)
    assert np.array_equal(results["loss_component"].to_numpy(), in_memory.loss_component)


def test_measure_stream_rejects_existing_output(tmp_path):
    """A directory that already holds part files is rejected."""
    mps = _portfolio(100)
    pp, cp = _write_model_points(mps, tmp_path, ".parquet")
    out_dir = tmp_path / "results"

    fcf.gmm.measure_stream(pp, out_dir, _assumptions(), coverages=cp, calculation_methods=PATTERNS)
    with pytest.raises(ValueError, match="already contains part"):
        fcf.gmm.measure_stream(pp, out_dir, _assumptions(), coverages=cp, calculation_methods=PATTERNS)


def test_measure_stream_id_column(tmp_path):
    """measure_stream writes the result id from id_column (a business key) when
    given, not always mp_id; an unknown id_column is a clear error up front."""
    n = 50
    mps = _portfolio(n)
    policy_no = np.array([f"PN{i:04d}" for i in range(n)], dtype=object)
    pp, cp = _write_model_points(mps, tmp_path, ".parquet", policy_no=policy_no)

    fcf.gmm.measure_stream(pp, tmp_path / "r1", _assumptions(), coverages=cp,
                           calculation_methods=PATTERNS, id_column="policy_no")
    ids = pl.read_parquet(str(tmp_path / "r1" / "part-*.parquet"))["id"].to_list()
    assert set(ids) == set(policy_no.tolist())

    with pytest.raises(ValueError, match=r"id_column 'nope' is not a column"):
        fcf.gmm.measure_stream(pp, tmp_path / "r2", _assumptions(), coverages=cp,
                               calculation_methods=PATTERNS, id_column="nope")


def test_measure_stream_rejects_global_duplicate_mp_id(tmp_path):
    """A duplicate mp_id straddling two chunks (invisible to a per-chunk read)
    is rejected up front -- the uniqueness read_model_points enforces in memory
    -- unless validate_unique_mp_id=False opts out."""
    pl.DataFrame({"mp_id": ["A", "B", "A"], "issue_age": [40, 50, 60],
                  "term_months": [12, 12, 12], "premium": [0.0, 0.0, 0.0]}
                 ).write_parquet(tmp_path / "p.parquet")
    pl.DataFrame({"mp_id": ["A", "B", "A"], "coverage": ["DEATH"] * 3,
                  "amount": [1e6] * 3}).write_parquet(tmp_path / "c.parquet")

    with pytest.raises(ValueError, match="duplicate mp_id"):
        fcf.gmm.measure_stream(tmp_path / "p.parquet", tmp_path / "o1",
                               _assumptions(), coverages=tmp_path / "c.parquet",
                               calculation_methods=PATTERNS, chunk_size=2)
    # opt-out runs the duplicate through (the caller's explicit choice)
    fcf.gmm.measure_stream(tmp_path / "p.parquet", tmp_path / "o2",
                           _assumptions(), coverages=tmp_path / "c.parquet",
                           calculation_methods=PATTERNS, chunk_size=2,
                           validate_unique_mp_id=False)
    assert sorted((tmp_path / "o2").glob("part-*.parquet"))


def test_load_sample_data_runs():
    """The bundled sample data loads and values without error."""
    mps = fcf.samples.model_points()
    basis = next(iter(fcf.samples.basis().values()))
    assert mps.n_mp > 0
    val = measure(mps, basis, full=False)
    assert val.bel.shape == (mps.n_mp,)
    assert val.csm.shape == (mps.n_mp,)


def test_sample_data_dir_exposes_bundled_files():
    """sample_data_dir() points at the directory holding the bundled inputs."""
    d = sample_data_dir()
    assert d.is_dir()
    names = {p.name for p in d.iterdir()}
    assert {"sample_basis.xlsx",
            "sample_policies.csv",
            "sample_coverages.csv"}.issubset(names)


def test_describe_basis_renders_both_shapes(capsys):
    """describe_basis prints a tree for a Basis and for a dict."""
    from fastcashflow import describe_basis
    basis_dict = fcf.samples.basis()
    seg_basis = next(iter(basis_dict.values()))

    describe_basis(seg_basis)
    out_one = capsys.readouterr().out
    assert out_one.startswith("Basis")
    assert "상태 전이율" in out_one
    assert "state_model" in out_one
    assert "coverages" in out_one

    describe_basis(basis_dict)
    out_dict = capsys.readouterr().out
    assert "(7 segments)" in out_dict
    # every segment unfolded -- both ('TERM_LIFE_A', 'GA') and ('TERM_LIFE_A', 'FC') appear
    assert "('TERM_LIFE_A', 'GA')" in out_dict
    assert "('TERM_LIFE_A', 'FC')" in out_dict


def test_long_form_round_trips(tmp_path):
    """A policies + coverages pair written out and re-read through
    read_model_points reproduces the valuation."""

    basis = next(iter(fcf.samples.basis().values()))
    patterns = fcf.samples.calculation_methods()
    mps = fcf.samples.model_points()
    policies, coverages = mp_to_frames(mps, basis)
    policies.write_csv(tmp_path / "pol.csv")
    coverages.write_csv(tmp_path / "cov.csv")
    back = read_model_points(tmp_path / "pol.csv",
                             coverages=tmp_path / "cov.csv",
                             calculation_methods=patterns)
    a, b = measure(mps, basis, full=False), measure(back, basis, full=False)
    assert np.allclose(a.bel, b.bel)
    assert np.allclose(a.csm, b.csm)


def test_measure_stream_streams_frames(tmp_path):
    """gmm.measure_stream streams a policies + coverages pair in chunks."""
    
    basis = next(iter(fcf.samples.basis().values()))
    patterns = fcf.samples.calculation_methods()
    mps = fcf.samples.model_points()
    policies, coverages = mp_to_frames(mps, basis)
    policies.write_parquet(tmp_path / "pol.parquet")
    coverages.write_parquet(tmp_path / "cov.parquet")

    out_dir = tmp_path / "results"
    processed = fcf.gmm.measure_stream(
        tmp_path / "pol.parquet", out_dir, basis,
        coverages=tmp_path / "cov.parquet",
        calculation_methods=patterns, chunk_size=3,
    )
    assert processed == mps.n_mp

    results = pl.read_parquet(str(out_dir / "part-*.parquet")).sort("id")
    in_memory = measure(mps, basis, full=False)
    assert np.allclose(results["bel"].to_numpy(), in_memory.bel)
    assert np.allclose(results["csm"].to_numpy(), in_memory.csm)


def test_measure_stream_routes_a_basis_dict(tmp_path):
    """measure_stream routes each chunk's model points by a per-segment basis dict."""

    basis_dict = fcf.samples.basis()                 # {(product, channel): Basis}
    patterns   = fcf.samples.calculation_methods()
    mps        = fcf.samples.model_points()
    policies, coverages = mp_to_frames(mps, next(iter(basis_dict.values())))
    # the dict path routes on the segment keys, so they must ride on the policies frame
    policies = policies.with_columns(
        pl.Series("product", mps.product),
        pl.Series("channel", mps.channel),
    )
    policies.write_parquet(tmp_path / "pol.parquet")
    coverages.write_parquet(tmp_path / "cov.parquet")

    out_dir = tmp_path / "results"
    processed = fcf.gmm.measure_stream(
        tmp_path / "pol.parquet", out_dir, basis_dict,
        coverages=tmp_path / "cov.parquet",
        calculation_methods=patterns, chunk_size=3,
    )
    assert processed == mps.n_mp

    results = pl.read_parquet(str(out_dir / "part-*.parquet")).sort("id")
    in_memory = measure(mps, basis_dict, full=False)   # same per-segment routing
    assert np.allclose(results["bel"].to_numpy(), in_memory.bel)
    assert np.allclose(results["csm"].to_numpy(), in_memory.csm)


def test_measure_stream_dict_needs_segment_keys(tmp_path):
    """A basis dict with no product / channel on the policies is a clear error."""

    basis_dict = fcf.samples.basis()
    patterns   = fcf.samples.calculation_methods()
    mps        = fcf.samples.model_points()
    policies, coverages = mp_to_frames(mps, next(iter(basis_dict.values())))
    policies.write_parquet(tmp_path / "pol.parquet")     # segment keys deliberately absent
    coverages.write_parquet(tmp_path / "cov.parquet")

    with pytest.raises(ValueError, match="product"):
        fcf.gmm.measure_stream(
            tmp_path / "pol.parquet", tmp_path / "results", basis_dict,
            coverages=tmp_path / "cov.parquet",
            calculation_methods=patterns, chunk_size=3,
        )


def test_model_points_repr_and_str_are_compact():
    """ModelPoints repr / str summarise the portfolio, not dump every array."""
    mp = fcf.samples.model_points()
    r = repr(mp)
    assert r.startswith("<ModelPoints") and "model point" in r
    assert "array(" not in r and len(r) < 200          # not the raw dataclass dump
    s = str(mp)
    for field in ("products", "coverages", "count"):
        assert field in s
    assert "array(" not in s

    # A VFA book carries no coverages -> 'account-value' noted, no coverages line.
    v = fcf.samples.model_points("vfa")
    assert "account-value" in repr(v)
    assert "account" in str(v) and "coverages" not in str(v)


def test_measure_stream_requires_mp_id_column(tmp_path):
    """A policies file with no mp_id is a clear ValueError, not a leaked polars
    ColumnNotFoundError -- even when id_column names a different result id."""
    pl.DataFrame({"policy_no": ["P1"], "issue_age": [40], "term_months": [12],
                  "premium": [0.0]}).write_parquet(tmp_path / "p.parquet")
    pl.DataFrame({"mp_id": ["P1"], "coverage": ["DEATH"], "amount": [1e6]}
                 ).write_parquet(tmp_path / "c.parquet")
    with pytest.raises(ValueError, match="no 'mp_id' column"):
        fcf.gmm.measure_stream(tmp_path / "p.parquet", tmp_path / "o",
                               _assumptions(), coverages=tmp_path / "c.parquet",
                               calculation_methods=PATTERNS, id_column="policy_no")
