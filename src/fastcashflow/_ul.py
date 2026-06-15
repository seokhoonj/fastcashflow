"""Universal-life (account-value) mechanics -- the recursive account-value roll.

A universal-life / cash-value contract carries a policyholder account value (AV)
that, each month, takes in premium (net of a load), has a maintenance fee and a
cost-of-insurance (COI) charge deducted, and is then credited interest at the
declared rate (floored at any guaranteed minimum). The death benefit is
``max(sum_assured, AV)``, so the insurer's true exposure is the net amount at
risk ``NAR = max(0, sum_assured - AV)``; the COI charges that NAR at a
contractual cost-of-insurance rate that is distinct from the best-estimate
mortality used to value actual claims (their spread is the mortality margin).

Because the COI depends on the NAR, which depends on the AV, which the COI in
turn reduces, the account value is genuinely path-dependent and cannot be the
closed-form geometric roll of the variable-fee (VFA) account. It is rolled
forward month by month here, vectorised over the model-point axis and sequential
over time -- the engine's standard hot-loop shape.

Within each month the events occur in a fixed order (the order matters because
the COI is charged on the post-premium, pre-credit balance):

    AV[t]
      + premium net of load        (account before fee)
      - maintenance fee - COI      (account before crediting; COI = rate * NAR)
      x (1 + credited rate)        (account at month end = AV[t+1])

Death and lapse are assumed mid-month and settle on the half-month-credited
balance. The COI is an internal deduction from the policyholder's account, not a
separate insurer cash flow: it shapes the account value and hence the benefits
paid, and the mortality margin emerges in the fulfilment cash flows as the COI
withheld against the much smaller expected NAR claim.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _ul_av_kernel(av0, prem_to_av, sum_assured, coi_rate_m, admin_fee_m, credit_m):
    """Recursive universal-life account-value roll-forward -- raw arrays only.

    Per model point (run in parallel across cores), roll the account value
    month by month with the within-month event order above.

    Parameters (all per model point):
    - ``av0``           ``(n_mp,)``           account value at the projection start
    - ``prem_to_av``    ``(n_mp, n_time)``    premium credited each month, net of load
    - ``sum_assured``   ``(n_mp,)``           death-benefit face amount
    - ``coi_rate_m``    ``(n_mp, n_time)``    monthly cost-of-insurance (charge) rate
    - ``admin_fee_m``   ``(n_mp, n_time)``    monthly per-policy maintenance fee
    - ``credit_m``      ``(n_mp, n_time)``    monthly credited rate (already floored
                                              at the guaranteed minimum)

    Returns:
    - ``av``      ``(n_mp, n_time + 1)``  account value at each month start (``av[:, 0] = av0``)
    - ``coi``     ``(n_mp, n_time)``      COI charged each month
    - ``av_mid``  ``(n_mp, n_time)``      half-month-credited account value (death / lapse basis)
    - ``nar``     ``(n_mp, n_time)``      net amount at risk used for the COI each month
    """
    n_mp, n_time = prem_to_av.shape
    av = np.zeros((n_mp, n_time + 1))
    coi = np.zeros((n_mp, n_time))
    av_mid = np.zeros((n_mp, n_time))
    nar = np.zeros((n_mp, n_time))

    for mp in prange(n_mp):
        a = av0[mp]
        av[mp, 0] = a
        sa = sum_assured[mp]
        for t in range(n_time):
            a += prem_to_av[mp, t]                 # account before fee (premium credited)
            risk = sa - a                          # net amount at risk on the BEF_FEE balance
            if risk < 0.0:
                risk = 0.0
            c = coi_rate_m[mp, t] * risk
            nar[mp, t] = risk
            coi[mp, t] = c
            a -= admin_fee_m[mp, t] + c            # account before crediting (fee + COI out)
            if a < 0.0:
                a = 0.0                            # a depleted account does not go negative
            cr = credit_m[mp, t]
            av_mid[mp, t] = a * (1.0 + cr) ** 0.5  # mid-month value for death / lapse
            a = a * (1.0 + cr)                     # month end: full crediting
            av[mp, t + 1] = a

    return av, coi, av_mid, nar
