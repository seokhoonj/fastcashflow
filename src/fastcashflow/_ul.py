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

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray, IntArray
from fastcashflow.basis import Basis, _single_basis, annual_to_monthly, validate_factor
from fastcashflow.curves import discount_monthly_curve
from fastcashflow.model_points import ModelPoints
from fastcashflow.numerics import (
    _cost_of_capital_ra, _csm_kernel, _norm_ppf, _rollforward_kernel)
from fastcashflow.projection import Cashflows, project_cashflows, _expense_kernel_args
from fastcashflow.state_model import resolve_state_model
from fastcashflow.tvog import credited_monthly_rate


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


def _ul_benefits(av_end, av_mid, deaths, lapses, maturity_survivors, term_idx,
                 sum_assured, surr_charge, minimum_accumulation_benefit):
    """Universal-life benefit cash flows from the account-value trajectory.

    The account value determines every benefit:
    - **death** (mid-month) pays ``max(account value, sum_assured)`` -- the
      account is returned and, where the face exceeds it, the net amount at risk
      tops it up. Settled on the half-month-credited ``av_mid``.
    - **surrender** (mid-month lapse) pays the account value less the duration's
      surrender charge, floored at zero: ``max(0, av_mid - surr_charge)``.
    - **maturity** pays ``max(matured account value, guaranteed accumulation
      benefit)`` on the month-end value at the contract's term.

    ``deaths`` / ``lapses`` are ``(n_mp, n_time)`` mid-month head-counts;
    ``maturity_survivors`` is ``(n_mp,)`` -- the count reaching the contract's
    maturity; ``term_idx`` is each contract's maturity column index.
    ``av_end`` is ``(n_mp, n_time + 1)`` (month-end values incl. the matured
    value at ``term_idx + 1``); ``av_mid`` / ``surr_charge`` are ``(n_mp, n_time)``.

    Returns ``(benefit_cf, death_cf, surrender_cf, maturity_cf)`` -- benefit_cf is
    the combined ``(n_mp, n_time)`` outflow with maturity entered nominally at
    ``term_idx``.
    """
    n_mp, n_time = av_mid.shape
    rows = np.arange(n_mp)
    death_cf = deaths * np.maximum(av_mid, sum_assured[:, None])
    surrender_cf = lapses * np.maximum(0.0, av_mid - surr_charge)
    av_at_maturity = av_end[rows, term_idx + 1]
    maturity_cf = maturity_survivors * np.maximum(
        av_at_maturity, minimum_accumulation_benefit)

    benefit_cf = death_cf + surrender_cf
    benefit_cf[rows, term_idx] += maturity_cf
    return benefit_cf, death_cf, surrender_cf, maturity_cf


@dataclass(frozen=True, slots=True)
class _ULProjection:
    """Universal-life projection -- decrements and the recursive account value
    woven into benefit cash flows. The building blocks ``measure_ul`` discounts
    into the BEL / RA / CSM (and ``ul.settle`` re-anchors at a valuation date),
    independent of the VFA-vs-GMM discounting choice it has yet to make.

    Shapes: trajectories are ``(n_mp, n_time)`` over the projection horizon;
    ``av`` / ``fund`` carry the extra month-end column (``n_time + 1``).
    ``maturity_cf`` / ``term_idx`` / ``maturity_survivors`` are ``(n_mp,)``.
    """

    cashflows: "Cashflows"          # the underlying decrement projection
    inforce: FloatArray             # (n_mp, n_time) policies in force at month start
    av: FloatArray                  # (n_mp, n_time+1) per-policy account value (month start)
    av_mid: FloatArray              # (n_mp, n_time) half-month-credited value (death / lapse)
    coi: FloatArray                 # (n_mp, n_time) COI charged to the account
    nar: FloatArray                 # (n_mp, n_time) net amount at risk
    fund: FloatArray                # (n_mp, n_time+1) in-force-weighted account value held
    benefit_cf: FloatArray          # (n_mp, n_time) combined benefit outflow (maturity at term_idx)
    death_cf: FloatArray            # (n_mp, n_time) death benefit, max(av_mid, face)
    surrender_cf: FloatArray        # (n_mp, n_time) surrender value on lapse
    maturity_cf: FloatArray         # (n_mp,) maturity benefit, max(matured av, GMAB)
    term_idx: IntArray              # (n_mp,) maturity column index (boundary-clamped)
    maturity_survivors: FloatArray  # (n_mp,) in-force reaching maturity


def _ul_project(
    model_points: ModelPoints,
    basis: Basis,
    *,
    _proj: "Cashflows | None" = None,
) -> "_ULProjection":
    """Project universal-life cash flows from the recursive account value.

    Runs the standard decrement projection (:func:`project_cashflows`), rolls the
    per-policy account value forward through :func:`_ul_av_kernel` on the basis'
    COI / premium-load / crediting assumptions, and weaves the resulting account
    value into the death / surrender / maturity benefits via
    :func:`_ul_benefits`. The decrements come from the occupancy projection (they
    depend on policy duration, not the fund); the account value and the benefits
    it drives are layered on top.

    The cost-of-insurance is charged on the net amount at risk against the model
    point's ``minimum_death_benefit`` (the UL face). The premium credited to the
    account is the contractual premium net of ``basis.premium_load``; the monthly
    maintenance fee deducted from the account is the per-policy ``gamma_fixed``
    expense; crediting is the declared ``investment_return`` floored at each
    contract's ``minimum_crediting_rate`` guarantee. No surrender penalty is
    applied in v1 (the kernel takes a per-duration charge; here it is zero, so a
    surrender pays the account value).

    ``_proj`` (private) lets a caller share one decrement projection across two
    account-value legs (an expected / observed pair), as ``vfa.settle`` does.
    """
    basis = _single_basis(basis, entry="measure_ul")
    # The UL death money is paid on the occupancy decrement (deaths * max(av,
    # face)); a state-conditioned death benefit or a deterministic sojourn exit
    # would be silently ignored by that flow, so reject them rather than
    # mis-measure -- the same guards the VFA account-value path applies.
    state_model = resolve_state_model(basis)
    if any(s.death_benefit_factor != 1.0 for s in state_model.states):
        raise NotImplementedError(
            "state-conditioned death benefit (State.death_benefit_factor) is "
            "not supported on the UL path; the account-value death benefit pays "
            "max(account value, face) on the occupancy decrement.")
    if any(tr.after_sojourn_months
           for s in state_model.states for tr in s.transitions):
        raise NotImplementedError(
            "a deterministic transition (Transition.after_sojourn_months) is not "
            "supported on the UL path.")

    proj = _proj if _proj is not None else project_cashflows(model_points, basis)
    inforce = proj.inforce
    n_mp, n_time = inforce.shape
    n_years = (n_time + 11) // 12

    # Per-year rate grid -- the unified (sex, issue_age, duration, issue_class,
    # elapsed) shape every Basis rate is read on (see project_cashflows). The
    # COI is the one rate the UL roll needs; elapsed = 0 (no semi-Markov axis).
    durations = np.arange(n_years)
    sex_grid, _ = np.meshgrid(model_points.sex, durations, indexing="ij")
    issue_age_grid, duration_grid = np.meshgrid(
        model_points.issue_age, durations, indexing="ij")
    issue_class_grid, _ = np.meshgrid(
        model_points.issue_class, durations, indexing="ij")
    elapsed_grid = np.zeros_like(duration_grid)
    year_of_month = np.arange(n_time) // 12   # a year's rate holds across its 12 months

    # COI monthly charge rate, year-expanded to (n_mp, n_time). None -> no COI
    # (a pure-accumulation account: NAR-charge zero everywhere).
    if basis.coi_annual is None:
        coi_rate_m = np.zeros((n_mp, n_time))
    else:
        coi_monthly = annual_to_monthly(basis.coi_annual(
            sex_grid, issue_age_grid, duration_grid,
            issue_class_grid, elapsed_grid))           # (n_mp, n_years)
        coi_rate_m = np.ascontiguousarray(coi_monthly[:, year_of_month])

    # Per-policy premium schedule, net of the premium load. This is the
    # contractual premium each paying month (NOT in-force weighted -- the kernel
    # rolls a single policy's account; in-force enters at the fund / benefit
    # aggregation). A premium-shape factor scales the level premium by year.
    if basis.premium_factor_annual is None:
        premium_factor = np.ones((n_mp, n_years))
    else:
        premium_factor = validate_factor(
            basis.premium_factor_annual(
                sex_grid, issue_age_grid, duration_grid,
                issue_class_grid, elapsed_grid),
            "premium_factor_annual", (n_mp, n_years))
    pf_m = premium_factor[:, year_of_month]            # (n_mp, n_time)
    t_idx = np.arange(n_time)[None, :]
    premium_term = model_points.premium_term_months[:, None]
    prem_freq = model_points.premium_frequency_months[:, None]
    paying = (t_idx < premium_term) & (t_idx % prem_freq == 0)
    premium_per_month = model_points.premium[:, None] * pf_m * paying
    prem_to_av = np.ascontiguousarray(
        premium_per_month * (1.0 - basis.premium_load))

    # Account charges and crediting. The maintenance fee deducted from the
    # account is the per-policy monthly gamma_fixed expense (v1: the account
    # admin charge equals the insurer's maintenance expense). Crediting is the
    # declared return floored at each contract's minimum guarantee; no fund fee
    # (UL revenue is the COI / load spreads, not an asset-based fee).
    _, _, _, gamma_fixed, _ = _expense_kernel_args(basis, n_time)
    admin_fee_m = np.ascontiguousarray(
        np.broadcast_to(gamma_fixed, (n_mp, n_time)))
    r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
    credit_monthly = credited_monthly_rate(
        r_m, model_points.minimum_crediting_rate)      # (n_mp,)
    credit_m = np.ascontiguousarray(
        np.broadcast_to(credit_monthly[:, None], (n_mp, n_time)))

    # The UL face is the model point's minimum_death_benefit (death pays
    # max(account value, face); NAR = max(0, face - account value)). av0 is the
    # account value at issue. Both are materialised on ModelPoints (zeros when
    # absent), so neither is None here.
    sum_assured = model_points.minimum_death_benefit
    av0 = model_points.account_value

    av, coi, av_mid, nar = _ul_av_kernel(
        av0, prem_to_av, sum_assured, coi_rate_m, admin_fee_m, credit_m)

    # Decrements -> benefits, reusing the VFA exit / maturity machinery. Every
    # policy eventually exits with its account value; deaths take max(av, face),
    # surrenders the account value, and the survivors reaching maturity
    # max(matured av, GMAB).
    inforce_pad = np.concatenate([inforce, np.zeros((n_mp, 1))], axis=1)
    exits = inforce_pad[:, :-1] - inforce_pad[:, 1:]   # (n_mp, n_time)
    deaths = proj.deaths
    rows = np.arange(n_mp)
    # Maturity is realised only when the term falls within that contract's own
    # Sec. 34 boundary; clamp the index past a boundary cut (zero weight makes
    # the clamped cell harmless). Identical to the VFA maturity handling.
    boundary_idx = model_points.contract_boundary_months - 1
    within = (model_points.term_months - 1) <= boundary_idx
    term_idx = np.where(within, model_points.term_months - 1, boundary_idx)
    maturity_survivors = np.where(within, proj.maturity_survivors, 0.0)
    # Mid-month surrenders (lapses) are the non-maturity, non-death exits.
    non_maturity_exits = exits - deaths
    non_maturity_exits[rows, term_idx] -= maturity_survivors

    # No surrender penalty in v1 -- the kernel takes a per-duration surrender
    # charge, but a UL surrender-charge schedule is a follow-up; zero here means
    # a surrender pays the (half-month-credited) account value, as the VFA does.
    surr_charge = np.zeros((n_mp, n_time))
    benefit_cf, death_cf, surrender_cf, maturity_cf = _ul_benefits(
        av, av_mid, deaths, non_maturity_exits, maturity_survivors, term_idx,
        sum_assured, surr_charge, model_points.minimum_accumulation_benefit)

    # The account value the entity holds for the policies in force (the
    # underlying items), in-force weighted -- the VFA's fund quantity.
    fund = inforce_pad * av
    return _ULProjection(
        cashflows=proj, inforce=inforce, av=av, av_mid=av_mid, coi=coi, nar=nar,
        fund=fund, benefit_cf=benefit_cf, death_cf=death_cf,
        surrender_cf=surrender_cf, maturity_cf=maturity_cf, term_idx=term_idx,
        maturity_survivors=maturity_survivors)


#: The IFRS 17 measurement models a universal-life contract may be measured
#: under -- VFA for a participating (return-share) account, GMM for a fixed /
#: declared-rate account. UL is not a fourth model; the choice selects only the
#: discounting / CSM-accretion rate (the account mechanics are identical).
UL_MEASUREMENT_MODELS = ("GMM", "VFA")


@dataclass(frozen=True, slots=True, eq=False)
class ULMeasurement:
    """Universal-life measurement of an account-value portfolio.

    The headline ``bel`` / ``ra`` / ``csm`` / ``loss_component`` are ``(n_mp,)``
    as-of figures at inception. The BEL is reported net of the account value the
    entity holds and of the present value of future premiums::

        BEL = PV(benefit_cf + expense_cf) - PV(premium_cf) - fund

    -- a generalisation of the VFA's ``PV(benefits + expenses) - fund`` to a
    recurring-premium account (a single-premium account, with ``premium_cf = 0``,
    reduces to the VFA form). The RA is a confidence-level (or cost-of-capital)
    margin on the mortality risk borne on the NET AMOUNT AT RISK -- the part of
    the death benefit above the account value, the only insurance-risk exposure
    -- plus expense risk. The CSM is ``max(0, -(BEL + RA))``, accreted at the
    measurement model's rate and released by coverage units::

        csm_path[:, t+1] = csm_path[:, t] + csm_accretion[:, t] - csm_release[:, t]

    ``measurement_model`` records which discounting basis was used: ``"GMM"``
    (locked-in ``discount_annual``) or ``"VFA"`` (the underlying-items return
    ``investment_return``). The account credits at the declared rate either way.

    The full path adds the ``(n_mp, n_time+1)`` trajectories ``bel_path`` /
    ``ra_path`` / ``csm_path`` / ``account_value_path`` (column 0 the as-of
    figure), ``None`` on the headline-only (``full=False``) path.
    """

    # headline -- always present, shape (n_mp,)
    bel: FloatArray
    ra: FloatArray
    csm: FloatArray
    loss_component: FloatArray
    measurement_model: str
    # trajectory -- full only (None on the headline-only path)
    bel_path: FloatArray | None = None            # (n_mp, n_time+1)
    ra_path: FloatArray | None = None             # (n_mp, n_time+1)
    csm_path: FloatArray | None = None            # (n_mp, n_time+1)
    account_value_path: FloatArray | None = None  # (n_mp, n_time+1)
    csm_accretion: FloatArray | None = None       # (n_mp, n_time)
    csm_release: FloatArray | None = None         # (n_mp, n_time)
    discount_bom: FloatArray | None = None        # (n_time+1,) start-of-month discount
    cashflows: "Cashflows | None" = None
    model_points: "ModelPoints | None" = None

    def _columns(self):
        return [("BEL", self.bel), ("RA", self.ra), ("CSM", self.csm),
                ("loss", self.loss_component)]

    def __repr__(self) -> str:
        from fastcashflow._display import measurement_repr
        return measurement_repr("ULMeasurement", self._columns())

    def __str__(self) -> str:
        from fastcashflow._display import measurement_str
        return measurement_str("ULMeasurement", self._columns())


def measure_ul(
    model_points: ModelPoints,
    basis: Basis,
    *,
    measurement_model: str = "GMM",
    full: bool = True,
) -> ULMeasurement:
    """Measure a universal-life (account-value) portfolio.

    Rolls the recursive account value (:func:`_ul_project`) and measures the
    contract liability. The death benefit pays ``max(account value, face)``, a
    surrender the account value, and the survivors reaching maturity
    ``max(matured account value, GMAB)``; the account is built from premium net
    of ``basis.premium_load``, charged a cost-of-insurance on the net amount at
    risk (``basis.coi_annual``) and a maintenance fee, and credited the declared
    ``investment_return`` floored at each contract's ``minimum_crediting_rate``.

    The BEL nets the present value of benefits and expenses against the present
    value of future premiums and the account value the entity holds
    (``PV(benefit + expense) - PV(premium) - fund``). The RA prices the mortality
    risk on the net amount at risk (the death benefit above the account) plus
    expense risk; the CSM is ``max(0, -(BEL + RA))``.

    ``measurement_model`` selects the discounting / CSM-accretion basis -- the
    only thing that varies between a participating and a fixed-rate UL:

    * ``"GMM"`` (default) -- the locked-in ``discount_annual`` curve (Sec. 36).
      A fixed / declared-rate (interest-sensitive) account.
    * ``"VFA"`` -- the underlying-items return ``investment_return``. A
      participating (unit-linked / with-profits) account.

    ``full=True`` (default) returns the BEL / RA / CSM / account-value
    trajectories; ``full=False`` fills only the headline figures (the inception
    CSM is ``csm0``, so the release kernel is skipped) and leaves the trajectory
    fields ``None``. ``basis`` must resolve to a single :class:`Basis`.
    """
    basis = _single_basis(basis, entry="measure_ul")
    if measurement_model not in UL_MEASUREMENT_MODELS:
        raise ValueError(
            f"measurement_model must be one of {UL_MEASUREMENT_MODELS}, got "
            f"{measurement_model!r}")

    p = _ul_project(model_points, basis)
    inforce = p.inforce
    n_mp, n_time = inforce.shape
    proj = p.cashflows

    # Discount / CSM-accretion rate -- the locked-in curve for GMM, the flat
    # underlying-items return for VFA. The account credits at the declared rate
    # either way (that is set inside _ul_project, independent of this choice).
    if measurement_model == "VFA":
        r_m = (1.0 + basis.investment_return) ** (1.0 / 12.0) - 1.0
        disc_monthly = np.full(n_time, r_m)
    else:
        disc_monthly = discount_monthly_curve(basis, n_time)
    boundary = model_points.contract_boundary_months
    zeros_t = np.zeros((n_mp, n_time))
    zeros_mp = np.zeros(n_mp)

    # BEL backward pass (reuse the GMM roll-forward kernel): death + surrender +
    # expense settle mid-month, premium at the start of month, maturity at the
    # contract boundary. fund is the account value held -- subtracted to report
    # the BEL net of the deposit (and of the premiums that build it).
    bel_pre_fund, *_ = _rollforward_kernel(
        np.ascontiguousarray(p.death_cf), zeros_t, zeros_t,
        proj.expense_cf, proj.premium_cf, zeros_t,
        np.ascontiguousarray(p.maturity_cf),
        np.ascontiguousarray(p.surrender_cf), boundary, disc_monthly)
    bel = bel_pre_fund - p.fund

    # Risk-adjustment present values. The insurance risk is the mortality borne
    # on the NET AMOUNT AT RISK -- the death benefit above the account
    # (``deaths * max(0, face - av_mid)``); the account portion returns the
    # policyholder's own money and bears no insurance risk. Run the at-risk
    # claim and the expense through one kernel pass (both discounted mid-month).
    face = model_points.minimum_death_benefit
    nar_claim = np.ascontiguousarray(
        proj.deaths * np.maximum(0.0, face[:, None] - p.av_mid))
    _, pv_nar, pv_expense, *_ = _rollforward_kernel(
        nar_claim, proj.expense_cf, zeros_t, zeros_t, zeros_t, zeros_t,
        zeros_mp, zeros_t, boundary, disc_monthly)
    z = _norm_ppf(basis.ra_confidence)
    confidence_margin = z * (basis.mortality_cv * pv_nar
                             + basis.expense_cv * pv_expense)
    if basis.ra_method == "cost_of_capital":
        ra = _cost_of_capital_ra(
            confidence_margin, disc_monthly, basis.cost_of_capital_rate)
    else:
        ra = confidence_margin

    fcf = bel[:, 0] + ra[:, 0]
    loss_component = np.maximum(0.0, fcf)
    csm0 = np.maximum(0.0, -fcf)

    full_factor = 1.0 / (1.0 + disc_monthly)
    discount_bom = np.concatenate([[1.0], np.cumprod(full_factor)])

    if not full:
        return ULMeasurement(
            bel=bel[:, 0], ra=ra[:, 0], csm=csm0,
            loss_component=loss_component, measurement_model=measurement_model,
            model_points=model_points)

    # CSM accretes at the measurement-model rate, released by coverage units.
    csm, csm_accretion, csm_release = _csm_kernel(
        csm0, inforce, disc_monthly, basis.coverage_unit_discount)

    return ULMeasurement(
        bel=bel[:, 0],
        ra=ra[:, 0],
        csm=csm[:, 0],
        loss_component=loss_component,
        measurement_model=measurement_model,
        bel_path=bel,
        ra_path=ra,
        csm_path=csm,
        account_value_path=p.av,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        discount_bom=discount_bom,
        cashflows=proj,
        model_points=model_points)
