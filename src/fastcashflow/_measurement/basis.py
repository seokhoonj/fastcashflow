"""Measurement time-basis discriminator and its consumer guard.

A measurement result is either an *inception* measurement (column 0 of every
trajectory is the contract's inception, headline == column 0) or an in-force
result whose headline is an as-of valuation-date figure while the trajectory
fields stay on the inception axis (and, in settlement-carry mode, carry a
prior-period CSM that must not be re-floored or re-rolled from inception).
Consumers that read trajectories from column 0 -- group / group_of_contracts /
roll_forward / report / transition / plot_* -- are only correct on the former;
:func:`_require_inception` is the single predicate they all call.

The VFA keeps its richer ``csm_basis`` field (see ``_vfa.CSM_BASES``) and
derives ``measurement_basis`` from it as a read-only property.
"""

MEASUREMENT_BASIS_INCEPTION = "inception"
# A what-if valuation of a seasoned book with no prior balances: the engine
# seats the contracts mid-life and reports the trajectory a freshly issued
# contract would produce under the current basis (engine-internal mode).
MEASUREMENT_BASIS_HYPOTHETICAL = "hypothetical"
# In-force diagnostic: BEL / RA re-based to the valuation date (current-rate
# remeasurement), CSM carried from the prior close without paragraph 44 unlocking
# (loss component hard zero). Not settlement-grade; the settle family is.
MEASUREMENT_BASIS_SETTLEMENT_CARRY = "settlement_carry"
# Paragraph-44/45 settlement output (the settle family's closing figures).
MEASUREMENT_BASIS_SETTLEMENT = "settlement"

MEASUREMENT_BASES = (
    MEASUREMENT_BASIS_INCEPTION,
    MEASUREMENT_BASIS_HYPOTHETICAL,
    MEASUREMENT_BASIS_SETTLEMENT_CARRY,
    MEASUREMENT_BASIS_SETTLEMENT,
)


def _require_inception(measurement, operation: str) -> None:
    """Reject a non-inception measurement from an inception-axis consumer.

    ``operation`` names the caller in the error (e.g. ``"group()"``). The
    check reads ``measurement_basis`` defensively so measurement types that
    predate the field (or third-party results) pass through unchanged.
    """
    basis = getattr(measurement, "measurement_basis",
                    MEASUREMENT_BASIS_INCEPTION)
    if basis == MEASUREMENT_BASIS_INCEPTION:
        return
    raise ValueError(
        f"{operation} reads the trajectories from inception (column 0) and "
        f"would silently mis-handle this measurement_basis={basis!r} result: "
        "its headline is an as-of valuation-date figure while the trajectory "
        "fields stay on the inception axis, and a carried CSM must not be "
        "re-floored or re-rolled from inception. For a period close use the "
        "settle family (gmm.settle / vfa.settle) and run this operation on "
        "its movement output."
    )


def _inforce_marker_columns(measurement, n: int) -> dict:
    """Marker columns for ``write_measurement`` on a non-inception result.

    On disk a settlement-carry headline is byte-compatible with new-business
    output; the ``measurement_basis`` column (plus ``elapsed_months`` when the
    source model points are stamped) keeps the two distinguishable at the file
    boundary. Inception output is unchanged (empty dict).
    """
    import numpy as np

    basis = getattr(measurement, "measurement_basis",
                    MEASUREMENT_BASIS_INCEPTION)
    if basis == MEASUREMENT_BASIS_INCEPTION:
        return {}
    cols = {"measurement_basis": [basis] * n}
    mp = getattr(measurement, "model_points", None)
    if mp is not None and getattr(mp, "elapsed_months", None) is not None:
        cols["elapsed_months"] = np.asarray(mp.elapsed_months, dtype=np.int64)
    return cols


_AGGREGATE_NO_CHAIN = (
    "an aggregate cannot seed the next period: chaining needs the per-MP "
    "closing balances, which the sums no longer carry. Chain through the "
    "per-MP movement's closing_inputs() instead (settle the book in row "
    "blocks if it does not fit in memory)."
)
