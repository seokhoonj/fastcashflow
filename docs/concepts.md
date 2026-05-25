# Concepts

This section explains the IFRS 17 ideas the engine implements. It is the home
for the educational material and will grow over time.

## The measurement models

IFRS 17 measures insurance contracts under three models; fastcashflow
implements all three.

### General measurement model (GMM)

The default model. The liability is the fulfilment cash flows -- the best
estimate liability plus the risk adjustment -- plus the contractual service
margin. The GMM is the building block the other two models specialise.

### Premium allocation approach (PAA)

A simplified model for short-coverage contracts. The liability for remaining
coverage is tracked as an unearned-premium-style balance rather than an
explicit cash flow projection, while the liability for incurred claims is
still measured explicitly.

### Variable fee approach (VFA)

The model for contracts with direct participation features -- where the
policyholder shares in the return on a pool of underlying items. The CSM
absorbs the entity's variable fee and the financial variability of that fee.

## The building blocks

### Best estimate liability (BEL)

The probability-weighted present value of the future cash flows within the
contract boundary -- premiums, claims, expenses -- discounted at a rate that
reflects the time value of money.

### Risk adjustment (RA)

The compensation the entity requires for bearing the uncertainty in the
amount and timing of the non-financial-risk cash flows. fastcashflow offers a
confidence-level method and a cost-of-capital method.

### Contractual service margin (CSM)

The unearned profit in the contract. A profitable contract recognises no
day-one gain: the profit is held as the CSM and released to the income
statement as insurance service is provided. An onerous contract has no CSM;
its expected loss is recognised immediately as a loss component.

## The period-close reporting cycle

Each reporting period the liability is rolled forward and the movement is
decomposed into an analysis of change: the opening balance, the interest
accreted, the effect of current-period service, changes in assumptions, the
release to profit or loss, and the closing balance -- each column reconciling
exactly.

## Initial recognition vs subsequent measurement

The default `value` / `measure` path treats every model point as a *new
contract at inception* (initial recognition, IFRS 17 Sec. 38). Each
contract's BEL, RA, and CSM are reported as of its issue date.

`value_in_force(model_points, assumptions)` returns the same quantities at
each contract's **valuation date**. Set `ModelPoints.elapsed_months[mp]` to
the number of months between inception and the valuation date for that
contract (different contracts in the same portfolio can have different
elapsed values -- the engine slices each one independently). With
`elapsed_months = 0` the result collapses to `value`. With `elapsed_months
= E` the result is the PV of future cash flows from month `E` forward --
the trajectory slice at duration `E`.

Two modes:

* **Hypothetical** (default, `prior_csm=None`). The CSM returned is the
  one a freshly issued contract would have at duration `E` under the
  current basis -- useful for inspection, not a production-settlement CSM
  (the real-world CSM is path-dependent: locked-in discount rate,
  accumulated unlocking and experience adjustments).
* **Settlement carry-forward** (`prior_csm=...`, `lock_in_rate=...`).
  Implements Sec. 44: the prior period's closing CSM is accreted at the
  locked-in rate and released over the coverage units forward to the
  valuation date. `prior_csm` is the closing CSM at month `elapsed - period_months`,
  `lock_in_rate` is the annual locked-in discount rate, `period_months`
  defaults to 12. v1 covers interest accretion and coverage-unit release
  only; assumption-change unlocking and experience adjustments go via
  `roll_forward` with full prior and current measurements.

## Calibrated rates and stateful product patterns

`rate_tables` holds **current best estimate** rates (Sec. 33) -- not
pricing-basis rates, and not raw industry incidence either. The number
the engine multiplies is the *calibrated* rate at the valuation date,
and that calibration is where stateful product mechanics -- claim
limits, waiting periods, reset windows, severity caps -- are absorbed.
The engine itself is deterministic and works with per-period mean rates;
distribution / variance information that a stateful mechanic depends on
is not visible to the kernel, so the mechanic's effect has to live in
the rate itself.

### The mapping to the calibration layers

A common Korean calibration framework lays out the final rate as a
product of layers, e.g. one large insurer's published shape:

> 보험금 = 가입금액 × **최적위험률 × A/E × 선택효과 × Trend**

fastcashflow's four assumption layers carry the same structure: the
base `rate_tables`, an `ae_factors` multiplier, the duration axis of
the base table (select-and-ultimate is expressed there), and an
`improvement_tables` trend. A stateful mechanic's effect lands on one
of these layers -- most often on `rate_tables` or `ae_factors` -- before
the engine ever sees the number.

### Example: a per-coverage claim-day limit

Inpatient indemnity and inpatient-daily-cash coverages in Korea
typically carry a per-period cap on claim days, sometimes with a reset
window after a cap is reached. The cap is per-policy and depends on the
individual claim path, but the engine only sees the per-period mean
incidence.

* **Uncalibrated**: pass the raw incidence rate. Expected claim cost
  ignores the cap, so the BEL is overstated for portfolios that
  routinely hit the cap.
* **Calibrated**: the experience study replaces the raw incidence with
  an *effective* incidence that already nets out the cap effect --
  typically by measuring actual paid days against the uncapped
  expectation in the cell (sex × age × duration × product) and folding
  the resulting ratio into `ae_factors`, or by replacing the raw rate
  itself with the post-cap empirical rate. Either flow keeps the
  engine's kernel unchanged; the rate the kernel multiplies already has
  the cap baked in.

The calibration itself -- experience study, the A/E construction, the
credibility-weighted blending against an industry reference, the
expert-committee sign-off -- is an actuarial workflow outside
fastcashflow. The engine takes the result and projects.

### What this means -- and does not mean

* **Portfolio-mean BEL is accurate** when the rate is properly
  cap-adjusted: the projection's expected claim cost matches the
  calibration's post-cap expectation.
* **Per-policy distribution / tail is not exposed.** A stress scenario
  that depends on the conditional shape of the claim distribution
  (e.g., what fraction of policies hit the cap under deterioration)
  needs more than a calibrated mean rate -- a stochastic scenario set
  (`value_stochastic`) or a future per-coverage state extension.
* **The engine does not validate the calibration.** The `rate(...)`
  callable returns a number; the engine multiplies and accumulates.
  Whether that number is a cap-adjusted best estimate or a raw pricing
  rate is the caller's responsibility.

This is the same principle as the existing waiting / reduced-benefit
periods, which are expressed structurally on the coverage (because the
duration clock is deterministic and visible to the kernel), and as the
semi-Markov cohort path for reincidence (the sojourn-time effect is
made visible by the cohort axis). State whose evolution depends on the
per-policy claim history -- and so on the distribution rather than the
mean -- is the part that lives in the calibration.
