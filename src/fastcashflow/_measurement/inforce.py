"""In-force valuation-date primitives shared across the measurement models.

:func:`_inforce_rescale` re-bases an inception-run projection to the valuation
date; :func:`inforce_surrender_value` is the one-time outflow a mass-lapse shock
pays the leaving policies; :func:`_reconcile_state` aligns an ``InforceState`` to
model-points order. All model-neutral (the in-force decrement is identical across
GMM / VFA / PAA), so they live in the shared measurement layer.
"""
from __future__ import annotations

import numpy as np

from fastcashflow._typing import FloatArray
from fastcashflow.basis import Basis, SURRENDER_VALUE_BASES
from fastcashflow.model_points import ModelPoints
from fastcashflow.projection import project_cashflows
from fastcashflow._measurement.account import _portfolio_has_account


def _inforce_rescale(inforce, model_points, em, rows) -> FloatArray:
    """Per-MP factor that re-bases an inception-run projection to the valuation
    date: ``count / inforce[em] = 1 / survival(0->em)``.

    The in-force projection runs from inception, so ``inforce[em] = count x
    survival(0->em)`` -- it decrements the as-of ``count`` again from inception.
    Scaling the sliced ``bel`` / ``ra`` by this factor makes the as-of in-force
    exactly the input ``count``; it is exact for every cash flow linear in the
    in-force. Where ``inforce[em]`` is zero (a fully run-off cohort) the bel is
    already zero, so the factor is 1 (a no-op).

    ``inforce`` is the ``(n_mp, n_time)`` start-of-month in-force trajectory.
    """
    inforce_em = inforce[rows, em]
    safe = np.where(inforce_em > 0.0, inforce_em, 1.0)
    count = np.asarray(model_points.count, dtype=np.float64)
    return np.where(inforce_em > 0.0, count / safe, 1.0)


def inforce_surrender_value(model_points: ModelPoints, basis: Basis) -> FloatArray:
    """Per-MP surrender value of the in-force at its valuation date -- ``(n_mp,)``.

    For each model point, the total surrender value that would be paid if its
    entire as-of in-force (``count``) surrendered at the valuation date (the
    moment ``elapsed_months`` after inception): the engine's own curve-based
    surrender value, read at the valuation-date duration and re-based to the
    as-of ``count`` (see :func:`_inforce_rescale`). This is the one-time outflow
    a mass-lapse shock pays the leaving policies -- the strain the future-cash-
    flow projection alone does not carry.

    Zero where the basis prices no ``surrender_value_curve`` (lapse removes the
    contract with no payment).

    Account-backed (universal-life) books are surrendered for the account value
    net of the surrender charge, ``max(0, av_mid x (1 - surr_charge_rate))``, read
    at the valuation-date duration -- the figure a surrender exit is paid in
    :func:`fastcashflow.projection.project_cashflows`. ``project_cashflows``
    rejects a mixed account / term book, so an account portfolio is homogeneous;
    ``av_mid`` is a per-policy level (independent of the in-force decrement), so
    the total paid if the whole as-of ``count`` surrenders is ``per-policy value
    x count`` (no ``_inforce_rescale``).

    The curve (non-account) per-mode value mirrors the ``surrender_cf`` block in
    ``project_cashflows``: with ``surrender_cf = (lapse_flow / inforce) x V``,
    this returns ``V`` (the total in-force surrender value) at the valuation
    slice, re-based to ``count``.
    """
    em = np.asarray(model_points.elapsed_months, dtype=np.int64)
    n_mp = em.shape[0]
    rows = np.arange(n_mp)
    if _portfolio_has_account(model_points, basis):
        proj = project_cashflows(model_points, basis)
        count = (np.ones(n_mp) if model_points.count is None
                 else np.asarray(model_points.count, dtype=np.float64))
        return proj.account.surr_value[rows, em] * count
    curve = basis.surrender_value_curve
    if curve is None:
        return np.zeros(n_mp)
    proj = project_cashflows(model_points, basis)
    inforce = proj.inforce
    c = np.asarray(curve, dtype=np.float64)
    value_em = c[np.minimum(em, c.shape[0] - 1)]      # curve held flat past its end
    mode = basis.surrender_value_basis
    if mode == "cum_premium_factor":
        cum_premium = np.cumsum(proj.premium_cf, axis=1)
        v_total = cum_premium[rows, em] * value_em
    elif mode == "amount_per_policy":
        v_total = inforce[rows, em] * value_em
    elif mode == "amount_per_unit":
        base = model_points.surrender_base_amount
        if base is None:
            raise ValueError(
                "surrender_value_basis='amount_per_unit' requires "
                "ModelPoints.surrender_base_amount (no default base is inferred).")
        v_total = inforce[rows, em] * value_em * np.asarray(base, dtype=np.float64)
    else:
        raise ValueError(
            f"unknown surrender_value_basis {mode!r}; expected one of "
            f"{SURRENDER_VALUE_BASES}.")
    return v_total * _inforce_rescale(inforce, model_points, em, rows)


def _reconcile_state(model_points: ModelPoints,
                     state: "InforceState") -> "InforceState":
    """Return ``state`` row-aligned to ``model_points`` (by mp_id), after
    checking the model points were already reconciled with it.

    Two jobs in one place. (1) The measurement reads each contract's as-of
    duration / size from ``model_points``; a model_points whose elapsed_months
    / count disagree with ``state`` was not reconciled (``apply_inforce_state``)
    and is rejected, so a stale snapshot cannot borrow a fresh state's CSM.
    (2) ``state.prior_csm`` is per-MP and must enter the measurement in
    model-points order, not the state file's order -- the returned state is
    reordered by mp_id so prior_csm lines up with the rows it belongs to.
    A reconciled, same-order pair passes through unchanged."""
    from fastcashflow.model_points import align_inforce_state
    # align_inforce_state does the mp_id join (and rejects mismatched id sets)
    # and reorders every per-MP field -- crucially prior_csm -- to mp order.
    state = align_inforce_state(model_points, state)
    em_ok = np.array_equal(
        np.asarray(state.elapsed_months, dtype=np.int64),
        np.asarray(model_points.elapsed_months, dtype=np.int64),
    )
    cnt_ok = model_points.count is not None and np.array_equal(
        np.asarray(state.count, dtype=np.float64),
        np.asarray(model_points.count, dtype=np.float64),
    )
    if not (em_ok and cnt_ok):
        raise ValueError(
            "measure_inforce: model_points elapsed_months / count do not match "
            "the InforceState. Reconcile them first -- "
            "model_points = apply_inforce_state(model_points, state) -- so the "
            "as-of duration and size come from the same period-close snapshot."
        )
    return state
