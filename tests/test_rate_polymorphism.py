"""Polymorphic rate input -- a Basis rate slot (and CoverageRate.rate) accepts
a scalar, a per-policy-year array, a polars / pandas DataFrame, or a RateFn
callable. All are normalised to a RateFn by ``basis._as_rate_fn`` and must give
results identical to the equivalent callable.
"""
import numpy as np
import polars as pl
import pytest

import fastcashflow as fcf

RATE = 1 - (1 - 0.01) ** 12  # 1%/month, annualised


def _bel(mort, cov, *, term=2, sex=0):
    mp = fcf.ModelPoints.single(
        issue_age=40, sex=sex, benefits={0: 12_000}, premium=100, term_months=term
    )
    basis = fcf.Basis(
        mortality_annual=mort, lapse_annual=0.0, discount_annual=1.005 ** 12 - 1,
        ra_confidence=0.75, mortality_cv=0.10,
        coverages=(fcf.CoverageRate("DEATH", cov),),
    )
    return float(fcf.gmm.measure(mp, basis).bel[0])


def test_scalar_equals_callable():
    fn = lambda s, a, d: np.full(a.shape, RATE)
    assert _bel(RATE, RATE) == pytest.approx(_bel(fn, fn))


def test_flat_array_equals_scalar():
    # term 36mo -> 3 policy years; a flat per-year array == the scalar
    assert _bel([RATE, RATE, RATE], [RATE, RATE, RATE], term=36) == pytest.approx(
        _bel(RATE, RATE, term=36)
    )


def test_array_indexes_by_policy_year():
    # Rate concentrated in year 0 vs year 1 -- timing must change the result,
    # proving arr[duration] indexing (not a flat average).
    early = _bel([0.20, 0.0, 0.0], [0.20, 0.0, 0.0], term=36)
    late = _bel([0.0, 0.20, 0.0], [0.0, 0.20, 0.0], term=36)
    assert early != pytest.approx(late)


def test_array_too_short_raises():
    # term 36mo needs 3 policy years; a length-1 array cannot cover it.
    with pytest.raises(ValueError, match="cover the contract term"):
        _bel([RATE], [RATE], term=36)


def test_polars_dataframe_selects_by_sex():
    # sex 0 high, sex 1 low -> the table must select per model point's sex.
    df = pl.DataFrame({"sex": [0, 1], "age": [40, 40], "rate": [0.05, 0.01]})
    bel_m = _bel(df, df, sex=0)
    bel_f = _bel(df, df, sex=1)
    assert bel_m != pytest.approx(bel_f)
    # and each matches the equivalent scalar
    assert bel_m == pytest.approx(_bel(0.05, 0.05, sex=0))
    assert bel_f == pytest.approx(_bel(0.01, 0.01, sex=1))


def test_pandas_dataframe_matches_polars():
    pd = pytest.importorskip("pandas")
    cols = {"sex": [0, 1], "age": [40, 40], "rate": [0.05, 0.01]}
    pdf = pd.DataFrame(cols)
    pldf = pl.DataFrame(cols)
    assert _bel(pdf, pdf, sex=0) == pytest.approx(_bel(pldf, pldf, sex=0))


def test_coveragerate_coerces_scalar_to_callable():
    cr = fcf.CoverageRate("DEATH", 0.012)
    assert callable(cr.rate)  # scalar normalised to a RateFn at construction


def test_basis_slot_coerces_scalar_to_callable():
    basis = fcf.Basis(
        mortality_annual=0.012, lapse_annual=0.0, discount_annual=0.0,
        ra_confidence=0.75, mortality_cv=0.0,
    )
    assert callable(basis.mortality_annual)


def test_bad_rate_type_raises():
    with pytest.raises(TypeError):
        fcf.CoverageRate("DEATH", np.zeros((2, 2)))  # 2-D array -> use a DataFrame
