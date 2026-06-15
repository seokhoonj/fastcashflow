"""Step 0 golden-master oracle for the universal-life funding-mechanism refactor.

This pins the numerical output of every measurement path BEFORE the UL account
roll is folded into the shared projection kernel (design
``dev/UL-SEPARATION-DESIGN.md`` v2, Step 0). The refactor is a pure refactor: it
must keep every number bit-identical. This module captures a bit-exact digest of
every measurement result on a fixed deterministic seed portfolio and asserts it
against a committed snapshot -- ``np.array_equal`` semantics, localised to the
field that diverged.

Mechanism: each numpy array is fingerprinted by ``sha256(shape|dtype|bytes)`` --
bit-identity (stricter than ``allclose``, and NaN-robust, since identical NaN bit
patterns hash equal). On first run (no snapshot) the digests are WRITTEN and the
test is skipped with a notice; thereafter the test recomputes and compares,
reporting exactly which ``path.field`` changed.

To regenerate intentionally (only when a number is *meant* to change): delete
``tests/_ul_oracle_snapshot.json`` and re-run.
"""
import dataclasses
import hashlib
import json
import os

import numpy as np
import pytest

import fastcashflow as fcf
from fastcashflow import Basis, CalculationMethod, CoverageRate, ModelPoints
from fastcashflow._ul import measure_ul

SNAPSHOT = os.path.join(os.path.dirname(__file__), "_ul_oracle_snapshot.json")


# ---------------------------------------------------------------------------
# Bit-exact digest helpers
# ---------------------------------------------------------------------------

#: Input echoes carried on result objects -- not refactor outputs; capturing
#: them only adds noise (and object-dtype string columns hash non-deterministically).
_SKIP_FIELDS = {"model_points"}


def _digest_array(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(f"{a.shape}|{a.dtype}|".encode())
    if a.dtype.kind in "OSU":
        # object / string arrays: ``tobytes`` serialises pointer addresses, not
        # content -- hash a content-based repr so the digest is deterministic.
        h.update(repr(a.tolist()).encode())
    else:
        h.update(a.tobytes())
    return h.hexdigest()


def _capture(out: dict, label: str, obj, depth: int = 2) -> None:
    """Fingerprint every ndarray / numeric field of a result object, recursing
    one level into nested dataclasses (e.g. ``cashflows``) so the projected
    arrays the refactor touches are pinned too."""
    if obj is None:
        out[label] = "None"
        return
    if isinstance(obj, np.ndarray):
        out[label] = _digest_array(obj)
        return
    if isinstance(obj, (int, float, np.integer, np.floating, bool, str)):
        out[label] = f"scalar:{obj!r}"
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            if f.name in _SKIP_FIELDS:
                continue
            v = getattr(obj, f.name, None)
            child = f"{label}.{f.name}"
            if isinstance(v, np.ndarray):
                out[child] = _digest_array(v)
            elif isinstance(v, (int, float, np.integer, np.floating, bool, str)):
                out[child] = f"scalar:{v!r}"
            elif v is None:
                out[child] = "None"
            elif dataclasses.is_dataclass(v) and depth > 0:
                _capture(out, child, v, depth - 1)
        return
    # lists of movements / reconciliations
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _capture(out, f"{label}[{i}]", v, depth)
        return


# ---------------------------------------------------------------------------
# Deterministic seed inputs (no RNG -- fixed sample files + inline constructions)
# ---------------------------------------------------------------------------

def _ul_mp():
    """A small recurring-premium UL portfolio (the death leg the refactor folds)."""
    return ModelPoints(
        issue_age=np.array([40.0, 55.0]),
        premium=np.array([500_000.0, 300_000.0]),
        term_months=np.array([36, 24]),
        account_value=np.array([0.0, 1_000_000.0]),
        minimum_death_benefit=np.array([80_000_000.0, 50_000_000.0]),
        minimum_accumulation_benefit=np.array([0.0, 0.0]),
        minimum_crediting_rate=np.array([0.0, 0.01]),
        sex=np.array([0, 1]),
    )


def _ul_basis():
    return Basis(
        mortality_annual=0.004,
        lapse_annual=0.03,
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.1,
        investment_return=0.024,
        coi_annual=0.0015,
        premium_load=0.08,
    )


def _fixed_return_scenarios(n_time: int, n_scen: int = 8):
    """Deterministic monthly-return paths (no RNG)."""
    base = np.linspace(-0.01, 0.03, n_scen)[:, None]
    drift = (np.arange(n_time)[None, :] % 12) * 0.0005
    return np.ascontiguousarray(base + drift)


def _fixed_rate_scenarios(n_scen: int = 8):
    return np.ascontiguousarray(np.linspace(0.01, 0.05, n_scen))


def _gmm_single_mp():
    return ModelPoints.single(
        issue_age=40, benefits={0: 1_000_000.0}, premium=12_000.0,
        term_months=60, calculation_methods={"DEATH": CalculationMethod.DEATH})


def _gmm_single_basis():
    return Basis(
        mortality_annual=0.005, lapse_annual=0.01, discount_annual=0.03,
        ra_confidence=0.75, mortality_cv=0.1,
        coverages=(CoverageRate("DEATH", lambda s, a, d: np.full(a.shape, 0.005)),))


def _collect() -> dict:
    out: dict = {}

    # ----- UL reference path (the bit-identity target for the refactor) -----
    ul_mp, ul_basis = _ul_mp(), _ul_basis()
    for model in ("GMM", "VFA"):
        for full in (True, False):
            tag = f"ul.{model}.full{full}"
            _capture(out, tag, measure_ul(ul_mp, ul_basis,
                                          measurement_model=model, full=full))

    # ----- GMM protection portfolio (samples are file-backed, deterministic) -----
    g_mp = fcf.samples.model_points("gmm")
    g_basis = fcf.samples.basis("gmm")
    _capture(out, "gmm.full", fcf.gmm.measure(g_mp, g_basis, full=True))
    _capture(out, "gmm.fast", fcf.gmm.measure(g_mp, g_basis, full=False))
    _capture(out, "gmm.aggregate", fcf.gmm.measure_aggregate(g_mp, g_basis))

    # ----- VFA account portfolio -----
    v_mp = fcf.samples.model_points("vfa")
    v_basis = fcf.samples.basis("vfa")
    _capture(out, "vfa.full", fcf.vfa.measure(v_mp, v_basis, full=True))
    _capture(out, "vfa.fast", fcf.vfa.measure(v_mp, v_basis, full=False))

    # ----- PAA short-duration portfolio -----
    p_mp = fcf.samples.model_points("paa")
    p_basis = fcf.samples.basis("paa")
    _capture(out, "paa.full", fcf.paa.measure(p_mp, p_basis, full=True))
    _capture(out, "paa.fast", fcf.paa.measure(p_mp, p_basis, full=False))

    # ----- single-basis GMM portfolio: the consumers that fold project_cashflows
    # but take a single Basis (settle/inforce need a matching InforceState and are
    # a documented follow-up below). Pinned on a NON-account portfolio so the
    # refactor must leave them bit-identical. -----
    s_mp, s_basis = _gmm_single_mp(), _gmm_single_basis()
    _capture(out, "gmm_single.full", fcf.gmm.measure(s_mp, s_basis, full=True))
    _capture(out, "reinsurance.measure",
             fcf.reinsurance.measure(s_mp, s_basis, treaty=fcf.samples.treaty()))
    _capture(out, "stochastic",
             fcf.gmm.stochastic(s_mp, s_basis, _fixed_rate_scenarios()))
    m_for_roll = fcf.gmm.measure(s_mp, s_basis, full=True)
    _capture(out, "roll_forward", fcf.roll_forward(m_for_roll, 12))
    _capture(out, "reconcile", fcf.reconcile(fcf.roll_forward(m_for_roll, 12)))

    # ----- VFA time value of guarantee (return-scenario consumer) -----
    n_time = fcf.vfa.measure(v_mp, v_basis, full=True).bel_path.shape[1] - 1
    _capture(out, "tvog",
             fcf.vfa.tvog(v_mp, v_basis, _fixed_return_scenarios(n_time)))

    # PENDING oracle expansion (state / grouping / parquet plumbing; tracked, NOT
    # silently dropped -- close before Step 3.5 gates land): measure_inforce,
    # settle, settle_aggregate, group_of_contracts / settle_group_of_contracts,
    # measure_stream / settle_stream (all families). Each consumes
    # project_cashflows and must be pinned on a non-account portfolio.
    return out


# ---------------------------------------------------------------------------
# The oracle test
# ---------------------------------------------------------------------------

def test_ul_refactor_oracle():
    current = _collect()
    if not os.path.exists(SNAPSHOT):
        with open(SNAPSHOT, "w") as fh:
            json.dump(current, fh, indent=1, sort_keys=True)
        pytest.skip(
            f"oracle snapshot written ({len(current)} fields) -- re-run to verify")
    with open(SNAPSHOT) as fh:
        reference = json.load(fh)

    cur_keys, ref_keys = set(current), set(reference)
    assert cur_keys == ref_keys, (
        f"oracle field set changed: only-now={sorted(cur_keys - ref_keys)[:10]} "
        f"only-ref={sorted(ref_keys - cur_keys)[:10]}")
    diverged = [k for k in reference if current[k] != reference[k]]
    assert not diverged, (
        f"{len(diverged)} field(s) diverged from the golden master (the refactor "
        f"changed a number it must not): {diverged[:20]}")
