# Changelog

All notable changes are listed here. Until the 0.1.0 release the on-disk
public API is treated as unstable -- deprecation paths are provided but
the project does not guarantee that the deprecated form will survive the
next minor release.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

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
