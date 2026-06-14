# Changelog

All notable changes are listed here. Until the 0.1.0 release the on-disk
public API is treated as unstable -- deprecation paths are provided but
the project does not guarantee that the deprecated form will survive the
next minor release.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- **Period-close reporting -- the IFRS 17 close pack.** `close(...)` assembles a
  reporting period's settlement reconciliations (one per group of contracts) into
  the disclosure statements -- the statement of financial position
  (`assemble_sofp`, paragraphs 78 / 99-101), the insurance finance statement
  (`assemble_finance`, B130-B136) and, when reports are supplied, the insurance
  service result (`assemble_service_result`, B120-B124) -- as a `ClosePackage`.
  The net statement adds reinsurance held to contracts issued in one signed frame
  (a recoverable lowers the net, paragraph 78). `write_close_pack` serialises the
  package to a multi-sheet `.xlsx`, keeping the per-model-point movement in a
  parquet sidecar; `reconciliation_to_frame` / `write_reconciliation` expose the
  per-reconciliation tidy frame and `line_metadata` the audit registry (line
  code, IFRS 17 paragraph, memo flag, sort order). `report.by_period` buckets a
  report on the elapsed or calendar basis. The four settlement reconciliations
  print a readable blocked table driven from that shared line spec.
- **Settlement family extended to every model.** `settle` / `settle_aggregate` /
  `settle_stream` now cover VFA (paragraph-45 CSM, account-value-linked LIC), PAA
  (paragraph-55(b) LRC roll) and reinsurance held (paragraph-66, no floor, with
  the 66A-66B loss-recovery component) alongside GMM, plus per-group-of-contracts
  settlement for the CSM-bearing models. `fcf.samples.treaty()` returns a bundled
  quota-share treaty so the reinsurance and close-pack examples run on the
  packaged book.
- **`gmm.settle` -- the IFRS 17 paragraph-44 settlement movement.** The
  period-close opening -> closing movement of a GMM in-force book: BEL / RA
  at current rates (B72(1)), CSM accretion and the paragraph-44(c)
  future-service unlocking at the locked-in rate (B72(b)/(c)) with the
  current-vs-locked-in gap carried as a named `finance_wedge` line (B97(a),
  insurance finance income/expense -- outside the CSM block), the
  paragraph-48/50(b) loss-component algebra (shared with `vfa.settle`), and
  a single period-end B119 coverage-unit release that telescopes exactly to
  `measure_inforce`'s monthly carry when experience is on-track. Returns a
  `GMMSettlementMovement` whose blocks reconcile by construction and whose
  `closing_inputs()` seeds the next period; `reconcile` and
  `write_measurement` accept it. Final settlement (a zero closing snapshot
  at the contract boundary) and mid-period full surrender derecognise the
  whole CSM through the release line.
- **Measurement time-basis discriminator (`measurement_basis`).** Every
  measurement now states what its numbers are: `'inception'` (new business),
  `'hypothetical'` (a what-if of a seasoned book), `'settlement_carry'` (the
  in-force diagnostic from `measure_inforce` -- as-of headline over
  inception-axis trajectories, prior CSM carried without unlocking) or
  `'settlement'` (the settle family). The VFA derives it from its existing
  `csm_basis`. The inception-axis consumers -- `group`, `group_of_contracts`,
  `roll_forward`, `report`, `transition` and the `plot_*` charts -- now reject
  a non-inception result with a pointer to the settle family instead of
  silently re-flooring a carried CSM at inception; `write_measurement` keeps
  accepting in-force output and adds `measurement_basis` / `elapsed_months`
  marker columns so the files stay distinguishable from new-business output.
  `gmm.measure_inforce` also rejects a mixed-model `BasisRouter` up front
  (route through `fcf.portfolio.measure_inforce`).
- **Time-varying premium and annuity (`Basis.premium_factor_annual` /
  `Basis.annuity_factor_annual`).** A per-policy-year multiplicative factor on
  the level premium / annuity, so a renewable / step-rated or escalating
  (체증형) premium or annuity can be projected while the scalar `premium` /
  `annuity_payment` stays the SCALE `solve_premium` solves for. Applies across
  every kernel path (full, fast, GPU). `None` is the level cash flow.
- **Per-coverage escalating benefits (체증형 보험금 / 간병비).** New coverages
  columns `step_month` / `step_factor` (a benefit step at a duration) and
  `escalation_annual` / `escalation_cap` (annual compounding growth, capped) --
  the bidirectional partner of the reduction rule. Full path; a new
  escalating-benefits cookbook chapter walks through all three shapes.
- **State-conditioned death benefit and true occupancy exit.**
  `State.death_benefit_factor` scales the death-coverage benefit paid for lives
  in a state (occupancy-weighted; default 1.0 is unchanged), and
  `State.exit_after` removes a cohort from the in-force set at a sojourn
  boundary (semi-Markov). Full path; the fast / VFA paths reject a non-default
  value rather than mis-measure it.
- **Insurance finance expense disaggregated by source (IFRS 17 B130-B136).**
  `Report` now carries `bel_finance_expense` / `ra_finance_expense` /
  `csm_finance_expense` alongside the aggregate `insurance_finance_expense`
  (the structural basis for a later P&L / OCI allocation).
- **`fcf.samples.export()` prints a tree of what it wrote** -- the files
  dropped in the directory, with `basis.xlsx` expanded into its assumption
  sheets, so it is clear at a glance what landed where. Pass `quiet=True` to
  suppress (e.g. in scripts).
- **`group` aggregator and `group_of_contracts` preset.** `group(m, by=...)`
  aggregates a `full=True` GMM measurement to any axis -- a single axis name, a
  list of names and/or precomputed `(n_mp,)` label arrays, or a bare label
  array -- re-deriving the CSM and loss component on the group aggregate so the
  floor nets within a group, not across. `group_of_contracts(m)` is the IFRS 17
  preset (portfolio x annual cohort x profitability, paragraphs 14/22/16):
  `portfolio` (default `product`) and `cohort` (default `issue_year`,
  derived from `issue_date`) name columns; `profitability` defaults to the
  engine-derived onerous / remaining split (it is an output, not a known
  input) and accepts an array or a column-name override. Both `group` and
  `group_of_contracts` dispatch on the measurement type via `singledispatch`
  and support all four models -- `GMMMeasurement`, `VFAMeasurement`,
  `ReinsuranceMeasurement` and `PAAMeasurement`. The VFA CSM re-derivation
  accretes at the underlying-items return (paragraph 45); reinsurance held has
  no loss component or floor (paragraph 65), so its grouped CSM is the sum of
  the contract CSMs and `group_of_contracts` splits its profitability by the
  net gain at initial recognition (paragraph 61) rather than the onerous test;
  the PAA has no CSM (paragraphs 53-59) -- the LRC, revenue, service expense
  and LIC sum, and only the onerous loss (paragraph 57) re-floors on the group
  aggregate. `VFAMeasurement`, `ReinsuranceMeasurement` and `PAAMeasurement` now
  carry the model points (and reinsurance the discount curve) so axis names
  resolve and the grouped result re-derives. A grouped result exposes
  `group_labels` -- the composite label of each row -- so a caller can map a
  group back to its key (e.g. `"|"`-split a `group_of_contracts` label into
  portfolio / cohort / profitability) without rebuilding the keys, and
  `group_sizes` -- the number of model points in each group.
- **Phase (c) semi-Markov in-force projection.** Tracks per-cohort
  occupancy in any state declared with `duration_max > 0`, so
  transition rates can depend on sojourn time. Powers the two flagship
  Korean-market use cases: cancer reincidence (`ci_reincidence_annual`,
  `lump_sum`-on-transition, exclusion window via the rate function)
  and disability-income recovery (`disability_recovery_annual`,
  duration-since-disablement axis).
- `STATE_MODELS` registry mapping a string key to a bundled `StateModel`.
  Currently a single entry, `"WAIVER"`. `read_assumptions` resolves
  the `state_model` column in the `segments` sheet against this
  registry, raising `ValueError` with a hint when the key is unknown.
- `state_model` column in the assumptions workbook's `segments` sheet
  (optional). Documented in `docs/assumptions-format.md` Section 4.
- `Assumptions.waiver_incidence_annual` (canonical name), plus
  `ci_incidence_annual`, `ci_reincidence_annual`,
  `disability_recovery_annual` fields. The 4-argument shape (one extra
  `state_duration` argument) is captured by a new `DurationRateFn`
  type alias.
- `_codegen_value_kernel_source_semi_markov` -- per-topology generated
  semi-Markov kernel, disk-cached. The coverage-rule and diagnosis
  passes reuse the main pass's saved in-force trajectory (no second
  state-machine walk), so combining cohort tracking with coverages is
  near-free.
- `Cookbook` documentation track (work in progress) -- product-by-
  product recipes for practicing actuaries.

### Changed

- **VFA `minimum_crediting_rate == 0.0` is now a real 0% floor, not "no
  guarantee" (breaking semantic correction).** A zero crediting rate now means
  the account is credited `max(return, 0)` -- principal protection that carries
  a time value -- and `vfa.measure(..., return_scenarios)` and `vfa.tvog` agree
  on it (`tvog` previously rejected zero while `measure` priced it; the two are
  now consistent). "No crediting guarantee" (the account follows the bare
  return, which may be negative) is the new default and is expressed by the
  exported `fcf.NO_GUARANTEE_RATE` sentinel, which `ModelPoints` fills when
  `minimum_crediting_rate` is omitted; `vfa.tvog` rejects it (there is no
  credited-rate time value to measure). A stray negative rate that is neither
  the sentinel nor `>= 0` is now rejected as a data error. The deterministic
  BEL is unchanged whenever the central return is non-negative; only the
  stochastic time value (and a negative-return BEL) shifts. A blank
  `minimum_crediting_rate` cell in a VFA policies file reads as the sentinel.
- **Routing / grouping axes are now the bare keys `product`, `channel`,
  `coverage`** (was `product_code`, `channel_code`, `coverage_code`). The
  engine treats them as opaque join keys -- whatever the input puts there
  (a code, a name, a custom analysis group) is equally valid -- so the
  `_code` suffix, which presumed "a code", is dropped. Affects the
  `ModelPoints` fields, the `measure` segment-routing default, the
  `policies` / `coverages` / `calculation_methods` columns and the
  `basis.xlsx` `segments` / `coverages` sheets. `read_basis` and
  `read_model_points` raise a rename hint when they find an old `_code`
  column. The `ModelPoints.coverage_codes` tuple (the pinned rate-driven
  order) keeps its name -- it is a distinct internal construct, not the
  per-row routing key. Display-only `*_name` label columns are no longer
  carried in the sample data; a workbook may still include them and the
  engine ignores them.
- Codegen value kernel is now the default dispatch for every multi-state
  model with `n_states >= 2`. The Markov-only closure factory and the
  hand-unrolled `n=2` / `n=3` kernels have been removed (their work is
  subsumed by the codegen path; git history preserves the earlier
  shape).
- README + tutorial chapters 08/09/10 quickstart examples now reflect
  the live API (`mortality_annual` / `lapse_annual` / `premium`).
  Older copies in the wild that used `mortality_monthly` /
  `lapse_monthly` / `monthly_premium` no longer work.

### Removed

- **`single_premium` field.** A single premium is now expressed as
  `premium` with `premium_term_months=1` (the premium is collected once,
  at inception) -- a single, uniform premium model rather than a separate
  level / single split. `ModelPoints.single_premium`, the `single_premium`
  argument to `ModelPoints.single`, and the `single_premium` policies
  column are gone. (The previous form additionally allowed a one-off
  premium *on top of* a level premium; that combination is no longer
  representable.)
- `level_premium` was renamed to `premium` in the same release (its single
  occurrence on `ModelPoints`, the reader columns, and every example).

### Fixed

- **Premium / annuity / coverage rate callables are validated at the kernel
  boundary.** A factor (`premium_factor_annual` / `annuity_factor_annual`) or a
  coverage rate callable that returns the wrong shape, a non-finite value, or a
  negative factor is now a clear `ValueError` (naming the offending coverage /
  factor) at the point its output is materialised -- on every kernel path. The
  shape check was previously an `assert`, which raised the wrong error type and
  was stripped under `python -O`, letting a mis-shaped grid mis-index the
  kernel; a negative premium / annuity factor could flip a cash-flow sign and
  bypass the `premium >= 0` invariant. The new CSR coverage-rule arrays
  (waiting / reduction / step / escalation) are likewise length- and
  sign-checked at construction.
- **In-force BEL / RA are re-based to the valuation date.** The in-force
  projection runs from each contract's inception, so the sliced
  `inforce[elapsed]` had decremented the as-of `count` again from inception and
  the BEL / RA understated the as-of figures by the inception-to-valuation
  survival. The sliced BEL / RA are now scaled by `count / inforce[elapsed]`,
  which is exact for every cash flow linear in the in-force (premium, claim,
  morbidity, expense, maturity, annuity); the CSM is scale-invariant and
  unchanged. The one remaining approximation is the **surrender value** -- it
  still uses the sample-grade `lapse x cum_premium x factor` base (no
  contractual surrender table, no pre-valuation premiums); `measure_inforce`
  now emits a `UserWarning` for that only when the basis carries a surrender
  curve and any `elapsed_months > 0`.
- `measure_inforce(..., full=False)` applied its `period_months` default
  inconsistently -- the `full=True` path defaulted a missing `period_months`
  to 12 but the `full=False` path raised on `None`. Both now default to 12.

### Deprecated

These names still work but emit `DeprecationWarning`; they will be
removed in **0.1.0**. Update on the first edit:

- `Assumptions.waiver_inception_annual` -> `waiver_incidence_annual`
- `Transition(rate="waiver_inception", ...)` -> `Transition(rate="waiver_incidence", ...)`

The deprecation routes the legacy form to the canonical one
automatically (the legacy field is then cleared, the legacy rate
name is normalised inside `compile_state_model`), so the only
user-visible change is the warning.

### Performance

- Markov n=2 / n=3 codegen kernels run at ~80-85 ms / 1M model points
  on a Ryzen 3700X with the warm disk cache; n=6 (LTC-like) at ~117
  ms. Numbers are stable post-cleanup.
- Semi-Markov 1M MP scales linearly in cohort depth `D`: D=12 -> ~280
  ms, D=60 -> ~1.34 s, D=120 -> ~2.6 s. Adding a coverage with a
  waiting/reduction rule or a diagnosis-pool depletion costs ~5%
  baseline overhead and ~0-10% per coverage thanks to the in-force
  trajectory cache.
