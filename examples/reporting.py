"""Reporting -- the IFRS 17 report, the analysis of change, and the close pack.

The inputs are the bundled sample portfolio (``fcf.samples``).

    python examples/reporting.py
"""
import tempfile
from pathlib import Path

import numpy as np

import fastcashflow as fcf


def main() -> None:
    basis = fcf.samples.basis()
    book = fcf.samples.model_points()
    m = fcf.gmm.measure(book, basis)

    # -- Inception reporting --------------------------------------------------
    # The IFRS 17 report -- insurance revenue, service expense, service result.
    print(fcf.report(m))
    print()

    # The period-close analysis of change, first reporting year.
    recon = fcf.reconcile(fcf.roll_forward(m, period_months=12))
    print(recon[0])
    print()

    # Aggregation to the IFRS 17 unit of account -- here, two age groups.
    group_ids = (book.issue_age >= 45).astype(int)
    grouped = fcf.group(m, group_ids)
    for g, label in enumerate(("under 45", "45 and over")):
        print(f"group '{label}'  CSM {grouped.csm_path[g, 0]:>14,.0f}")
    print()

    # -- Period close ---------------------------------------------------------
    # The reporting-period close: settle each group of contracts (one per
    # product / channel segment) from its in-force snapshot, then assemble the
    # IFRS 17 close pack -- the statement of financial position, the finance
    # statement and the per-group reconciliation detail.
    state = fcf.samples.inforce_state()
    recons, group_labels = [], []
    for segment in basis.segments:
        rows = np.flatnonzero((book.product == segment[0]) & (book.channel == segment[1]))
        if rows.size == 0:
            continue
        mp = book.subset(rows)
        st = state.subset(np.flatnonzero(np.isin(state.mp_id, mp.mp_id)))
        valued = fcf.apply_inforce_state(mp, st)              # as-of the close date
        movement = fcf.gmm.settle(valued, st, basis.resolve(segment), period_months=12)
        recons.append(fcf.reconcile([movement])[0])
        group_labels.append("/".join(segment))

    package = fcf.close(recons, group_ids=group_labels)
    print(package)
    print()

    # Serialise the multi-sheet close-pack workbook (the auditable artifact).
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "close_pack.xlsx"
        fcf.write_close_pack(package, out)
        print(f"close pack written -- {out.name}  ({len(package.to_frames())} sheets)")


if __name__ == "__main__":
    main()
