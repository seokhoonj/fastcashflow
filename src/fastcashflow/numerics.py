"""Shared numerical primitives -- numpy arrays in, numpy arrays out.

These functions know nothing about ``Basis`` / ``Cashflows`` / any
other domain dataclass. The orchestration layer (engine, PAA, VFA, movement)
unpacks domain objects to raw arrays + scalars and calls in here. The split
keeps these primitives numba-friendly and unit-testable on bare numpy.

The orchestration-specific ``@njit`` kernels (``_project_kernel`` in
projection.py, ``_fast_kernel_scalar`` and the codegen fast kernel in
engine.py) stay next to their callers -- they have only one call site each.
The primitives below are the ones that genuinely cross modules.

Contents:

* settlement-pattern helpers (``_settlement_lic``, ``_settlement_factor``)
* the standard-normal inverse CDF (``_norm_ppf``)
* the cost-of-capital RA accumulator (``_cost_of_capital_ra``)
* the BEL / RA / CSM time-loop kernels (``_rollforward_kernel``,
  ``_csm_kernel``)
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray


def _settlement_lic(
    incurred: FloatArray, settlement_pattern: FloatArray
) -> FloatArray:
    """Liability for incurred claims over a claims settlement pattern.

    ``incurred`` is the ``(n_mp, n_time)`` claims incurred each month;
    ``settlement_pattern`` is the run-off pattern, summing to 1. Returns the
    ``(n_mp, n_time+1)`` LIC trajectory -- claims build it up as incurred and
    run it off as paid -- held undiscounted.

    When the run-off tail extends past the projection horizon (a claim
    incurred near ``n_time`` whose settlement would fall after it), the unpaid
    tail is retained in the terminal column ``lic[:, -1]``: it stays
    outstanding rather than being dropped -- an incurred claim is a liability
    for incurred claims regardless of whether the coverage period has ended
    (IFRS 17). ``lic[:, 0]`` is always 0, so the LIC never double-counts the
    BEL, which values the full incurred claim separately (the caller applies
    the settlement discount factor).
    """
    incurred = np.asarray(incurred, dtype=np.float64)
    pattern = np.asarray(settlement_pattern, dtype=np.float64)
    if not np.isclose(pattern.sum(), 1.0):
        raise ValueError(f"settlement_pattern must sum to 1, got {pattern.sum()}")
    n_mp, n_time = incurred.shape
    paid = np.zeros_like(incurred)
    for k, weight in enumerate(pattern):
        # Lags k >= n_time settle PAST the projection horizon. Leaving them
        # un-subtracted from `paid` keeps their weight in cumsum(incurred -
        # paid), so it parks at the terminal column lic[:, -1] -- the LIC for
        # claims whose settlement runs beyond the horizon. Do NOT 'simplify'
        # this guard away: a wrapping / truncating convolution would silently
        # drop the tail and understate the liability, and for a pattern longer
        # than n_time the stop index incurred[:, :n_time - k] would go negative
        # and broadcast-error.
        if k < n_time:
            paid[:, k:] += weight * incurred[:, :n_time - k]
    lic = np.zeros((n_mp, n_time + 1))
    lic[:, 1:] = np.cumsum(incurred - paid, axis=1)
    # Set the terminal residual -- claims whose settlement runs past the horizon
    # -- from the exact analytic tail rather than the cumsum, so a book that
    # fully settles within the horizon lands on exact zero (not float run-off
    # dust). A claim incurred at month i still owes the settlement weight that
    # pays at or after the horizon, sum(pattern[n_time - i:]); tail_weight[m] is
    # that reverse-cumulative sum (0 past the pattern length).
    tail_weight = np.concatenate([np.cumsum(pattern[::-1])[::-1], [0.0]])
    j = np.clip(n_time - np.arange(n_time), 0, pattern.size)
    lic[:, -1] = incurred @ tail_weight[j]
    return lic


def _settlement_lic_discounted(
    incurred: FloatArray, settlement_pattern: FloatArray,
    monthly_rate: float,
) -> FloatArray:
    """Discounted liability for incurred claims (Sec. 40(b) / 42(c)).

    The present-value mirror of :func:`_settlement_lic`: a claim incurred ``j``
    months ago contributes the present value -- at the current month boundary --
    of the part of its run-off still outstanding::

        LIC_disc[:, d] = sum_{1 <= j <= n_pat} incurred[:, d - j] x w[j]
        w[j] = sum_{k >= j} pattern[k] x v^(k - j),  v = 1 / (1 + monthly_rate)

    ``w[j]`` is the PV (at the claim's age ``j``) of the outstanding run-off; it
    values every payment still owed, including those settling past the
    projection horizon, so the terminal column ``LIC_disc[:, -1]`` retains the
    discounted tail exactly as :func:`_settlement_lic` retains the undiscounted
    one. With ``monthly_rate == 0`` this reduces to ``_settlement_lic``.

    ``monthly_rate`` is a flat per-month scalar: the LIC run-off is short (at
    most the pattern length) so a flat rate over it is a close approximation to
    the full discount curve used for the BEL. ``LIC_disc[:, 0]`` is 0 -- a claim
    incurred at ``d`` is not yet in the LIC at ``d`` -- so the LIC never
    double-counts the BEL.
    """
    incurred = np.asarray(incurred, dtype=np.float64)
    pattern = np.asarray(settlement_pattern, dtype=np.float64)
    if not np.isclose(pattern.sum(), 1.0):
        raise ValueError(f"settlement_pattern must sum to 1, got {pattern.sum()}")
    n_mp, n_time = incurred.shape
    n_pat = pattern.size
    r = float(monthly_rate)
    v = 1.0 / (1.0 + r)
    # w[j], j = 1 .. n_pat: PV at age j of the outstanding run-off. w[n_pat] = 0
    # (a claim older than the pattern is fully settled), so the loop below stops
    # contributing once a claim has run off.
    w = np.zeros(n_pat + 1)
    for j in range(1, n_pat + 1):
        ks = np.arange(j, n_pat)
        w[j] = float(np.sum(pattern[ks] * v ** (ks - j)))
    lic = np.zeros((n_mp, n_time + 1))
    # Lag j shifts the incurred claims forward by j months: incurred[:, s] (the
    # claim incurred at s) lands in LIC[:, s + j] with weight w[j]. The largest
    # boundary is n_time, where each recently-incurred claim sits at age
    # n_time - s -- captured by the matching j -- so the terminal column holds
    # the discounted outstanding tail without separate parking.
    for j in range(1, min(n_pat, n_time) + 1):
        if w[j] == 0.0:
            continue
        lic[:, j:n_time + 1] += w[j] * incurred[:, : n_time + 1 - j]
    return lic


def _carry_lic_residual(lic, idx, t, n_time, seg_lic):
    """Flat-fill a stitched segment's parked LIC residual to the global terminal.

    When a portfolio is measured per segment, each segment is measured on its
    own horizon ``t`` and scattered into the global ``(n_mp, n_time+1)`` array
    by ``lic[idx, :t+1] = seg_lic``. A claim whose settlement tail runs past the
    segment horizon leaves an outstanding liability for incurred claims at the
    segment's terminal column ``seg_lic[:, -1]`` (IFRS 17 -- an incurred claim
    is a liability whether or not the coverage period has ended). Carry it flat
    across the padded tail -- mirroring the discount-tail flat-fill and the
    single-segment path -- so the residual does not vanish (which would
    understate the closing liability and book a phantom claims_paid in the
    roll-forward). Unlike BEL / RA / CSM, which are legitimately zero past term.

    The carried residual ``seg_lic[:, -1]`` is the exact analytic tail computed
    by ``_settlement_lic`` (not the float-dusty cumsum), so a fully-settled
    segment carries exact zero -- pads like its BEL / RA / CSM siblings -- and a
    genuine residual, however small, carries exactly. No-op when ``t == n_time``
    (the horizon-defining segment: nothing to pad) and when the segment has no
    settlement residual (``seg_lic[:, -1]`` is zero).
    """
    if t >= n_time:
        return
    lic[idx, t + 1:] = seg_lic[:, -1:]


def _settlement_factor(
    settlement_pattern: FloatArray, monthly_rate: float | FloatArray
) -> float | FloatArray:
    """Present-value factor for a claim spread over a settlement pattern.

    The present value, at the month a claim is incurred, of paying a unit
    claim over ``settlement_pattern`` -- discounted at ``monthly_rate``.
    A pattern that pays everything immediately gives 1.

    ``monthly_rate`` may be either:

    * a scalar -- the run-off is discounted at a flat per-month rate and
      the result is a single scalar factor (the legacy behaviour, kept
      for callers that need one number);
    * a per-month rate curve of shape ``(n_time,)`` -- the result is an
      ``(n_time,)`` factor whose element ``t`` discounts the run-off
      starting at month ``t`` using ``monthly_rate[t:]``. The tail past
      ``n_time`` is held flat at the last curve value, so a settlement
      pattern with more lags than the curve still terminates.

    The curve form is the right reference under a discount curve (Sec. 40
    / B71 -- the rate at the month of incurrence). Callers may continue
    to pass a scalar where a representative single factor is desired (the
    fused fast path, in particular, multiplies a per-policy coverage
    amount that is not month-indexed).
    """
    pattern = np.asarray(settlement_pattern, dtype=np.float64)
    if not np.isclose(pattern.sum(), 1.0):
        raise ValueError(f"settlement_pattern must sum to 1, got {pattern.sum()}")

    rate = np.asarray(monthly_rate, dtype=np.float64)
    n_pat = pattern.shape[0]
    if rate.ndim == 0:
        months = np.arange(n_pat)
        return float(np.sum(pattern / (1.0 + float(rate)) ** months))

    if rate.ndim != 1:
        raise ValueError(
            f"monthly_rate must be a scalar or a 1-D curve, got shape {rate.shape}"
        )
    n_time = rate.shape[0]
    # Hold the curve flat past its end so the run-off can extend into the
    # tail when the pattern is longer than ``n_time - t``.
    ext = np.concatenate([rate, np.full(n_pat - 1, rate[-1])])
    # ``disc[t, k]`` is the cumulative discount from month ``t`` to ``t + k``,
    # built one lag at a time so the operation stays vectorised over ``t``.
    disc = np.ones((n_time, n_pat))
    for k in range(1, n_pat):
        disc[:, k] = disc[:, k - 1] / (1.0 + ext[k - 1 : k - 1 + n_time])
    return (disc * pattern[None, :]).sum(axis=1)


# Coefficients of Acklam's rational approximation of the standard-normal
# inverse CDF -- the published constants of the algorithm.
_ACKLAM_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_ACKLAM_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01)
_ACKLAM_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_ACKLAM_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00)


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (quantile function).

    Rational-approximation algorithm (Acklam) followed by one Halley step.
    Acklam alone is ~1e-9, which degrades in the extreme tails (p < 1e-7
    or symmetric upper); the Halley refinement on the standard-normal CDF
    -- accessible via :func:`math.erfc` -- restores essentially full
    double precision across the whole open interval. Avoids a scipy
    dependency for a value the engine needs only once per run.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in the open interval (0, 1)")

    a, b, c, d = _ACKLAM_A, _ACKLAM_B, _ACKLAM_C, _ACKLAM_D
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
             / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0))
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = ((((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
             / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0))
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = (-(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
             / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0))

    # Halley step: with f(x) = Phi(x) - p, f'(x) = phi(x), f''(x) = -x phi(x),
    # the update simplifies to x - err / (phi + 0.5 * x * err) where
    # err = Phi(x) - p. Compute err directly from erfc on the side that
    # keeps the small quantity small -- avoids the catastrophic 1 - tiny
    # cancellation that would otherwise wreck precision in the upper tail.
    sqrt2 = math.sqrt(2.0)
    if p < 0.5:
        err = 0.5 * math.erfc(-x / sqrt2) - p
    else:
        err = (1.0 - p) - 0.5 * math.erfc(x / sqrt2)
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return x - err / (pdf + 0.5 * x * err)


def _cost_of_capital_ra(
    confidence_margin: FloatArray, monthly_rate: FloatArray, coc_rate: float
) -> FloatArray:
    """Cost-of-capital RA -- the cost of holding the confidence-level margin
    as non-financial-risk capital over the contract's run-off.

    The capital required at each future month is taken as the confidence-
    level margin there; the RA at month ``t`` is the cost-of-capital rate
    times the present value, at ``t``, of that capital over months ``t``
    onward. ``monthly_rate`` is the per-month rate curve, shape
    ``(n_time,)``; a flat rate and a yield curve share the same form.
    """
    full = 1.0 / (1.0 + monthly_rate)             # (n_time,)
    cap_pv = np.empty_like(confidence_margin)
    cap_pv[:, -1] = confidence_margin[:, -1]
    for t in range(confidence_margin.shape[1] - 2, -1, -1):
        cap_pv[:, t] = confidence_margin[:, t] + full[t] * cap_pv[:, t + 1]
    return coc_rate * cap_pv


def _risk_adjustment(basis, pv_claims, pv_morbidity, pv_disability,
                     pv_survival, monthly_rate):
    """The risk adjustment per ``basis.ra_method``.

    The confidence-level margin is ``z(ra_confidence)`` times the sum over
    risks of each coefficient-of-variation x its claim PV. ``ra_method`` then
    selects whether that margin IS the RA (``"confidence_level"``) or seeds the
    cost-of-capital run-off (``"cost_of_capital"``, which discounts the future
    capital at ``monthly_rate``). Shared by the GMM and PAA measurements so the
    two cannot diverge on the method -- the PAA onerous test used to hardcode
    the confidence-level form and silently ignore ``cost_of_capital``. The
    ``pv_*`` may be the full ``(n_mp, n_time+1)`` trajectory (GMM, PAA) -- the
    cost-of-capital branch needs the trajectory; the caller slices what it needs.
    """
    if basis.expense_cv != 0.0:
        raise NotImplementedError(
            "expense_cv is not included in the GMM / PAA risk adjustment -- only "
            "the mortality / morbidity / disability / longevity risks are priced "
            "(there is no expense-risk PV in this RA). Set expense_cv=0 for a "
            "GMM / PAA measurement. (The VFA RA does price expense_cv.)"
        )
    z = _norm_ppf(basis.ra_confidence)
    confidence_margin = z * (basis.mortality_cv * pv_claims
                     + basis.morbidity_cv * pv_morbidity
                     + basis.disability_cv * pv_disability
                     + basis.longevity_cv * pv_survival)
    if basis.ra_method == "confidence_level":
        return confidence_margin
    if basis.ra_method == "cost_of_capital":
        return _cost_of_capital_ra(confidence_margin, monthly_rate,
                                   basis.cost_of_capital_rate)
    raise ValueError(
        "ra_method must be 'confidence_level' or 'cost_of_capital', "
        f"got {basis.ra_method!r}"
    )


@njit(parallel=True, cache=True)
def _rollforward_kernel(claim_cf, morbidity_cf, disability_cf, expense_cf,
                        premium_cf, annuity_cf, maturity_cf, surrender_cf,
                        contract_boundary_months, monthly_rate):
    """Backward pass -- the BEL and the four RA present-value trajectories.

    ``BEL[t]`` is the present value, at month boundary ``t``, of the cash
    flows from month ``t`` onward, built by a backward recursion. Premiums
    and annuity payments fall at the start of the month, claims, expenses
    and surrender mid-month::

        BEL[t] = annuity[t] - premium[t]
                 + (claim[t] + morbidity[t] + disability[t]
                    + expense[t] + surrender[t]) * (1+i[t])^-0.5
                 + BEL[t+1] * (1+i[t])^-1

    ``monthly_rate`` is the per-month rate curve, shape ``(n_time,)``, so
    the locked-in rate can be flat or a yield curve. The maturity benefit
    is a single payment at ``t = term``, so it seeds ``BEL[term]``.
    ``BEL[:, 0]`` is then the inception BEL.

    Four more trajectories feed the Risk Adjustment, one per risk class:
    ``pv_claims`` (death claims -- mortality risk), ``pv_morbidity`` (health
    claims -- morbidity risk), ``pv_disability`` (disability income and the
    lump sum -- disability risk) and ``pv_survival`` (annuity payments and
    the maturity benefit -- longevity risk). All five trajectories have shape
    ``(n_mp, n_time+1)``.
    """
    n_mp, n_time = claim_cf.shape
    bel = np.zeros((n_mp, n_time + 1))
    pv_claims = np.zeros((n_mp, n_time + 1))
    pv_morbidity = np.zeros((n_mp, n_time + 1))
    pv_disability = np.zeros((n_mp, n_time + 1))
    pv_survival = np.zeros((n_mp, n_time + 1))

    half = (1.0 + monthly_rate) ** (-0.5)
    full = 1.0 / (1.0 + monthly_rate)

    for mp in prange(n_mp):
        # The backward pass runs from the Sec. 34 contract boundary (= term
        # when there is no boundary cut). maturity_cf is already 0 when the
        # boundary is short of the term (the projection withheld it), so
        # seeding at the boundary is correct either way.
        boundary = contract_boundary_months[mp]
        bel[mp, boundary] = maturity_cf[mp]
        pv_survival[mp, boundary] = maturity_cf[mp]
        for t in range(boundary - 1, -1, -1):
            claim = claim_cf[mp, t]
            morbidity = morbidity_cf[mp, t]
            disability = disability_cf[mp, t]
            annuity = annuity_cf[mp, t]
            surrender = surrender_cf[mp, t]
            bel[mp, t] = (
                annuity - premium_cf[mp, t]
                + (claim + morbidity + disability
                   + expense_cf[mp, t] + surrender) * half[t]
                + bel[mp, t + 1] * full[t]
            )
            pv_claims[mp, t] = claim * half[t] + pv_claims[mp, t + 1] * full[t]
            pv_morbidity[mp, t] = morbidity * half[t] + pv_morbidity[mp, t + 1] * full[t]
            pv_disability[mp, t] = disability * half[t] + pv_disability[mp, t + 1] * full[t]
            pv_survival[mp, t] = annuity + pv_survival[mp, t + 1] * full[t]

    return bel, pv_claims, pv_morbidity, pv_disability, pv_survival


@njit(parallel=True, cache=True)
def _csm_kernel(csm0, coverage_units, monthly_rate, discount_units):
    """Compiled CSM roll-forward kernel -- raw numpy arrays only.

    Per model point (run in parallel across cores): interest accretion at the
    locked-in rate -- a per-month curve ``monthly_rate`` of length
    ``n_time``, so flat scalar and yield curve share the kernel -- then
    release proportional to coverage units. The coverage-unit tail sum is
    built in a single backward pass so the roll-forward stays linear in
    time. Monthly interest and release are returned too, so the roll-forward
    is fully decomposable: ``csm[t+1] = csm[t] + accretion[t] - release[t]``.

    The release is the IFRS 17 B119 coverage-unit allocation: the
    end-of-period (i.e. post-accretion) CSM is spread over the coverage units
    provided this period and expected in future, and this period's share is
    recognised::

        release[t] = csm_accreted[t] * coverage_units[t] / sum_{s>=t} coverage_units[s]

    Two choices B119 leaves to judgement are fixed here:

    * **coverage units = in-force.** The caller passes ``proj.inforce`` as
      ``coverage_units``, so one coverage unit is one in-force contract-month
      (a fixed-benefit-protection convention; for a contract whose benefit
      amount varies over time the quantity of coverage would differ from raw
      in-force).
    * **future coverage units: discounted or not is an accounting-policy
      choice (B119), exposed via ``discount_units``.** When ``False`` (the
      default) the tail sum ``sum_{s>=t}`` carries no discount factor. When
      ``True`` each future period's coverage units are discounted back to the
      current period at the same locked-in ``monthly_rate`` used for accretion,
      so the tail becomes ``sum_{s>=t} coverage_units[s] * v(t->s)`` with
      ``v`` the locked-in discount factor -- the choice many entities make.
    """
    n_mp, n_time = coverage_units.shape
    csm = np.zeros((n_mp, n_time + 1))
    accretion = np.zeros((n_mp, n_time))
    release = np.zeros((n_mp, n_time))

    for mp in prange(n_mp):
        csm[mp, 0] = csm0[mp]

        cu_tail = np.empty(n_time)          # cu_tail[s] = (discounted) sum of coverage_units[mp, s:]
        if discount_units:
            # Discount future coverage units back to each period at the
            # locked-in rate: cu_tail[s] = cu[s] + cu_tail[s+1] / (1 + rate[s]).
            cu_tail[n_time - 1] = coverage_units[mp, n_time - 1]
            for s in range(n_time - 2, -1, -1):
                cu_tail[s] = (coverage_units[mp, s]
                              + cu_tail[s + 1] / (1.0 + monthly_rate[s]))
        else:
            running = 0.0
            for s in range(n_time - 1, -1, -1):
                running += coverage_units[mp, s]
                cu_tail[s] = running

        # Epsilon (not exact > 0) so the rounding residual of a reverse
        # cumulative sum near the end of the run-off cannot produce a
        # denormal ``cu_remaining`` and a runaway release. Scaled to the
        # total coverage units so a portfolio in tiny units (e.g. per-policy
        # with sub-unit inforce) does not trip the guard for every contract.
        eps = 1e-12 * cu_tail[0] if cu_tail[0] > 0.0 else 1e-12

        for t in range(1, n_time + 1):
            interest = csm[mp, t - 1] * monthly_rate[t - 1]
            accreted = csm[mp, t - 1] + interest
            cu_remaining = cu_tail[t - 1]
            if cu_remaining > eps:
                rel = accreted * coverage_units[mp, t - 1] / cu_remaining
            else:
                rel = 0.0
            accretion[mp, t - 1] = interest
            release[mp, t - 1] = rel
            csm[mp, t] = accreted - rel

    return csm, accretion, release


@njit(parallel=True, cache=True)
def _csm_kernel_permp(csm0, coverage_units, monthly_rate, discount_units):
    """CSM roll-forward with a **per-model-point** rate -- ``monthly_rate`` is
    ``(n_mp, n_time)`` rather than the shared ``(n_time,)`` of
    :func:`_csm_kernel`. Used when a portfolio's model points discount on
    different curves (a segmented measurement, where each row carries its own
    segment's rate). Identical roll-forward, only the rate is indexed by row.
    ``discount_units`` is the B119 coverage-unit discounting choice (see
    :func:`_csm_kernel`); when ``True`` the per-row locked-in rate is used.
    """
    n_mp, n_time = coverage_units.shape
    csm = np.zeros((n_mp, n_time + 1))
    accretion = np.zeros((n_mp, n_time))
    release = np.zeros((n_mp, n_time))

    for mp in prange(n_mp):
        csm[mp, 0] = csm0[mp]

        cu_tail = np.empty(n_time)
        if discount_units:
            cu_tail[n_time - 1] = coverage_units[mp, n_time - 1]
            for s in range(n_time - 2, -1, -1):
                cu_tail[s] = (coverage_units[mp, s]
                              + cu_tail[s + 1] / (1.0 + monthly_rate[mp, s]))
        else:
            running = 0.0
            for s in range(n_time - 1, -1, -1):
                running += coverage_units[mp, s]
                cu_tail[s] = running
        eps = 1e-12 * cu_tail[0] if cu_tail[0] > 0.0 else 1e-12

        for t in range(1, n_time + 1):
            interest = csm[mp, t - 1] * monthly_rate[mp, t - 1]
            accreted = csm[mp, t - 1] + interest
            cu_remaining = cu_tail[t - 1]
            if cu_remaining > eps:
                rel = accreted * coverage_units[mp, t - 1] / cu_remaining
            else:
                rel = 0.0
            accretion[mp, t - 1] = interest
            release[mp, t - 1] = rel
            csm[mp, t] = accreted - rel

    return csm, accretion, release


def _csm_roll(csm0, coverage_units, monthly_rate, discount_units=False):
    """Roll the CSM with a shared ``(n_time,)`` or per-MP ``(n_mp, n_time)`` rate.

    A single-basis portfolio shares one discount curve (1-D rate); a segmented
    (per-basis-dict) one discounts each row on its own curve (2-D rate). Picks
    the matching kernel so callers do not branch. ``discount_units`` is the
    B119 coverage-unit discounting accounting-policy choice (default ``False``
    -- undiscounted; see :func:`_csm_kernel`).
    """
    if np.asarray(monthly_rate).ndim == 2:
        return _csm_kernel_permp(csm0, coverage_units,
                                 np.ascontiguousarray(monthly_rate),
                                 discount_units)
    return _csm_kernel(csm0, coverage_units, monthly_rate, discount_units)


def _paragraph45_csm_algebra(accreted, x, lc_open):
    """The paragraph-45/48/50(b) CSM and loss-component step, branchless.

    ``accreted`` is the opening CSM accreted to the adjustment date, ``x`` the
    favourable(+)/unfavourable(-) future-service change, ``lc_open`` the
    opening loss component. A favourable change reverses the loss component
    before rebuilding the CSM (50(b)); an unfavourable change beyond the CSM
    falls into the loss component (48). Returns
    ``(csm_after, lc_reversed, lc_recognised, lc_closing)`` satisfying the
    conservation identity ``(csm_after - accreted) - (lc_closing - lc_open)
    == x`` in every sign case. Shared by the VFA (paragraph 45) and GMM
    (paragraph 44) settlement movements -- the loss-component algebra is the
    same in both models.
    """
    lc_reversed = np.minimum(lc_open, np.maximum(x, 0.0))
    csm_adj = accreted + x - lc_reversed
    csm_after = np.maximum(csm_adj, 0.0)
    lc_recognised = np.maximum(-csm_adj, 0.0)
    lc_closing = lc_open - lc_reversed + lc_recognised
    return csm_after, lc_reversed, lc_recognised, lc_closing
