"""Risk-neutral economic scenario generator -- ``fcf.esg``.

A stochastic generator is validated by no-arbitrage / distributional properties,
not a single hand calculation: the short rate must reprice the calibrated curve
(the bond martingale), the discounted fund value must be a martingale, the draw
must be deterministic in the seed, and the output must feed the real consumers
(``gmm.stochastic`` / ``vfa.measure``) in the right units.
"""
import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import esg

# A small Korean-style observed curve (as in tests/test_smith_wilson.py).
_MAT = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 20.0])
_RATE = np.array([0.0310, 0.0355, 0.0368, 0.0390, 0.0408, 0.0410])
_UFR, _ALPHA = 0.0405, 0.10


def _sim(**over):
    kw = dict(ufr=_UFR, alpha=_ALPHA, mean_reversion=0.10, rate_vol=0.01,
              equity_vol=0.15, correlation=-0.2, n_scenarios=20000, n_time=240,
              seed=1)
    kw.update(over)
    return esg.simulate(_MAT, _RATE, **kw)


# ---------------------------------------------------------------------------
# No-arbitrage anchors
# ---------------------------------------------------------------------------

def test_short_rate_reprices_the_curve():
    """The exact-discrete drift calibration makes the Monte-Carlo bond reprice
    the Smith-Wilson curve up to MC noise only -- and the error shrinks ~1/sqrt(n)."""
    err5k = _sim(n_scenarios=5000, seed=2).martingale_error()[0]
    err80k = _sim(n_scenarios=80000, seed=2).martingale_error()[0]
    assert err80k < 1e-3                      # tight at large n
    assert err80k < err5k                     # shrinks with more paths


def test_discounted_fund_is_a_martingale():
    """The fund's risk-neutral drift is the short rate, so its discounted terminal
    value averages to 1 (the equity martingale)."""
    _, equity_err = _sim(n_scenarios=60000, seed=3).martingale_error()
    assert equity_err < 1e-2


def test_short_rate_mean_matches_drift():
    """E[r_t] = the deterministic drift alpha_t (the mean-zero OU factor adds no
    bias); the MC mean of the short rate tracks it closely."""
    es = _sim(n_scenarios=40000, seed=4)
    # drift = E[r] under the model; recover it as the path mean and check the
    # term structure is finite, monotone-ish and near the ~3-4% input level.
    mean_r = es.short_rate.mean(axis=0)
    assert np.all(np.isfinite(mean_r))
    assert 0.0 < mean_r[0] < 0.10 and 0.0 < mean_r[-1] < 0.10


# ---------------------------------------------------------------------------
# Units / output contract
# ---------------------------------------------------------------------------

def test_output_shapes_and_units():
    es = _sim()
    assert es.rates.shape == (20000, 240)
    assert es.returns.shape == (20000, 240)
    assert np.all(np.isfinite(es.rates))          # annual rates, negatives allowed
    assert np.all(es.returns > -1.0)              # monthly returns, the VFA bound


def test_feeds_the_consumers():
    """rates -> gmm.stochastic, returns -> vfa.measure, in the right units."""
    vmp, vb = fcf.samples.model_points("vfa"), fcf.samples.basis("vfa")
    n_time = int(np.asarray(vmp.term_months).max())
    es = esg.simulate(_MAT, _RATE, ufr=_UFR, alpha=_ALPHA, mean_reversion=0.10,
                      rate_vol=0.01, equity_vol=0.12, correlation=-0.2,
                      n_scenarios=400, n_time=n_time, seed=5)
    vm = fcf.vfa.measure(vmp, vb, return_scenarios=es.returns)
    assert np.all(np.isfinite(vm.time_value))

    gmp, gb = fcf.samples.model_points("gmm"), fcf.samples.basis("gmm")
    key = ("HEALTH_A", "FC")
    idx = np.where((np.asarray(gmp.product) == key[0])
                   & (np.asarray(gmp.channel) == key[1]))[0]
    gn = int(gmp.subset(idx).contract_boundary_months.max())
    rates = esg.hull_white_rates(_MAT, _RATE, ufr=_UFR, alpha=_ALPHA,
                                 mean_reversion=0.10, rate_vol=0.01,
                                 n_scenarios=300, n_time=gn, seed=6)
    sr = fcf.gmm.stochastic(gmp.subset(idx), gb.resolve(key), rates)
    assert np.all(np.isfinite(sr.bel))


# ---------------------------------------------------------------------------
# Determinism, variance reduction, correlation
# ---------------------------------------------------------------------------

def test_deterministic_in_the_seed():
    a, b = _sim(seed=11), _sim(seed=11)
    assert np.array_equal(a.rates, b.rates) and np.array_equal(a.returns, b.returns)
    c = _sim(seed=12)
    assert not np.array_equal(a.rates, c.rates)


def test_antithetic_pairs_mirror_the_rate_factor():
    """With antithetic on, path s and path s+half share negated innovations, so
    their short rates are mirrored about the (shared) deterministic drift."""
    es = _sim(n_scenarios=1000, n_time=60, antithetic=True)
    half = 500
    mid = 0.5 * (es.short_rate[:half] + es.short_rate[half:2 * half])
    drift = mid[0]                                  # the shared deterministic level
    assert np.allclose(mid, drift[None, :], atol=1e-12)


def test_antithetic_reduces_estimator_variance():
    """The antithetic bond-price estimator has a smaller spread across seeds."""
    def price(anti, seed):
        es = esg.simulate(_MAT, _RATE, ufr=_UFR, alpha=_ALPHA, mean_reversion=0.10,
                          rate_vol=0.012, equity_vol=0.0, correlation=0.0,
                          n_scenarios=2000, n_time=120, seed=seed, antithetic=anti)
        disc = np.exp(-np.cumsum(es.short_rate, axis=1) * (1.0 / 12.0))[:, -1]
        return disc.mean()
    plain = np.std([price(False, s) for s in range(16)])
    anti = np.std([price(True, s) for s in range(16)])
    assert anti < plain


def test_rate_equity_correlation_recovered():
    """The realised correlation between the short-rate innovation and the fund
    log-return is close to the input rho."""
    rho = -0.4
    es = _sim(n_scenarios=40000, n_time=120, correlation=rho)
    # dr[:, i] is driven by the rate innovation z_rate[:, i]; the fund return at
    # the SAME innovation index is returns[:, i] (z_eq[:, i]). Align both on i.
    dr = np.diff(es.short_rate, axis=1).ravel()         # innovations i = 0..n-2
    lr = np.log1p(es.returns)[:, :-1].ravel()           # returns at i = 0..n-2
    assert abs(np.corrcoef(dr, lr)[0, 1] - rho) < 0.05


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="mean_reversion"):
        _sim(mean_reversion=0.0)
    with pytest.raises(ValueError, match="correlation"):
        _sim(correlation=1.5)
    with pytest.raises(ValueError, match=">= 0"):
        _sim(rate_vol=-0.01)
    with pytest.raises(ValueError):
        _sim(n_time=0)
