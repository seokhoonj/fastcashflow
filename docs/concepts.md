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

This is an MVP for subsequent measurement (Sec. 40-52). The CSM returned is
the one a freshly issued contract would have at duration `E` under the
current basis; it does not yet carry forward the prior period's CSM with
experience adjustments. Period-close roll-forward (`roll_forward`,
`reconcile`) is the path for that.
