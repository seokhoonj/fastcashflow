# Changelog

All notable changes are listed here. Until the 0.1.0 release the on-disk
public API is treated as unstable -- deprecation paths are provided but
the project does not guarantee that the deprecated form will survive the
next minor release.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- **`group` aggregator and `group_of_contracts` preset.** `group(m, by=...)`
  aggregates a `full=True` GMM measurement to any axis -- a single axis name, a
  list of names and/or precomputed `(n_mp,)` label arrays, or a bare label
  array -- re-deriving the CSM and loss component on the group aggregate so the
  floor nets within a group, not across. `group_of_contracts(m)` is the IFRS 17
  preset (portfolio x annual cohort x profitability, paragraphs 14/22/16):
  `portfolio` (default `product_code`) and `cohort` (default `issue_year`,
  derived from `issue_date`) name columns; `profitability` defaults to the
  engine-derived onerous / remaining split (it is an output, not a known
  input) and accepts an array or a column-name override. Dispatches on the
  measurement type via `singledispatch`.
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

- Codegen value kernel is now the default dispatch for every multi-state
  model with `n_states >= 2`. The Markov-only closure factory and the
  hand-unrolled `n=2` / `n=3` kernels have been removed (their work is
  subsumed by the codegen path; git history preserves the earlier
  shape).
- README + tutorial chapters 08/09/10 quickstart examples now reflect
  the live API (`mortality_annual` / `lapse_annual` / `level_premium`).
  Older copies in the wild that used `mortality_monthly` /
  `lapse_monthly` / `monthly_premium` no longer work.

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
