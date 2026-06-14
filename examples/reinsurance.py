"""Reinsurance -- a quota-share treaty held over a direct portfolio.

The inputs are the bundled sample portfolio (``fcf.samples``).
``reinsurance.measure`` / ``reinsurance.settle`` take a single :class:`Basis`,
so this cedes one segment of the book.

    python examples/reinsurance.py
"""
import numpy as np

import fastcashflow as fcf
from fastcashflow import InforceState


def main() -> None:
    basis  = fcf.samples.basis()
    book   = fcf.samples.model_points()
    state  = fcf.samples.inforce_state()
    treaty = fcf.samples.treaty()                  # 30% quota share (bundled)

    segment   = ("TERM_LIFE_A", "FC")
    seg_basis = basis.resolve(segment)
    rows      = np.flatnonzero((book.product == segment[0]) & (book.channel == segment[1]))
    mp        = book.subset(rows)

    # -- Inception: the reinsurance-held position -----------------------------
    reins = fcf.reinsurance.measure(mp, seg_basis, treaty=treaty)
    print("reinsurance held -- 30% quota share (inception)")
    print(f"  BEL (PV premiums - recoveries)  {reins.bel.sum():>16,.0f}")
    print(f"  RA  (risk transferred)          {reins.ra.sum():>16,.0f}")
    print(f"  CSM (net cost/gain of cover)    {reins.csm_path[:, 0].sum():>16,.0f}")
    print()

    # -- Period close: the close pack, net of reinsurance ---------------------
    # Settle the segment's issued book and its quota-share cession over a
    # reporting period, then assemble the close pack. The Net row is the issued
    # liability less the reinsurance recoverable (IFRS 17 paragraph 78).
    st     = state.subset(np.flatnonzero(np.isin(state.mp_id, mp.mp_id)))
    valued = fcf.apply_inforce_state(mp, st)
    period = 12

    issued = fcf.reconcile([fcf.gmm.settle(valued, st, seg_basis, period_months=period)])[0]

    # The reinsurance held carries its OWN prior CSM (not the gross CSM); a real
    # close reads it from the reinsurance in-force state. Here it is seeded from
    # the inception measure at the opening duration.
    opening  = np.asarray(st.elapsed_months) - period
    re_state = InforceState(
        mp_id=st.mp_id,
        elapsed_months=st.elapsed_months,
        count=st.count,
        prior_csm=reins.csm_path[np.arange(mp.mp_id.shape[0]), opening],
        lock_in_rate=st.lock_in_rate,
        prior_count=st.prior_count,
    )
    held = fcf.reconcile([fcf.reinsurance.settle(
        valued, re_state, seg_basis, treaty=treaty, period_months=period)])[0]

    pack = fcf.close([issued, held],
                     group_ids=["TERM_LIFE_A/FC issued", "TERM_LIFE_A/FC reins 30% QS"])
    print(pack)


if __name__ == "__main__":
    main()
