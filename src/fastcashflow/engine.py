"""Engine entry points.

Two paths:

* ``measure`` -- detailed: full monthly cash flow and CSM trajectories. Use it
  for inspection, validation and movement analysis.
* ``value``   -- fast: a single fused, parallel kernel producing only the
  headline valuation (BEL, RA, CSM, loss component) per model point. It
  materialises no per-month arrays, so it is memory-minimal and the fastest
  path for large-scale valuation.

Both paths share the same arithmetic, so ``value`` reproduces ``measure``'s
headline numbers exactly (cross-checked in the tests).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange

from fastcashflow._typing import FloatArray
from fastcashflow.assumptions import Assumptions, annual_to_monthly
from fastcashflow.curves import (
    discount_factors,
    discount_factors_from_curve,
    discount_monthly_curve,
    inflation_index,
    maintenance_monthly_curve,
)
from fastcashflow.numerics import (
    _cost_of_capital_ra,
    _csm_kernel,
    _norm_ppf,
    _rollforward_kernel,
    _settlement_factor,
    _settlement_lic,
)
from fastcashflow.coverage import coverage_arrays, coverage_rates
from fastcashflow.modelpoints import ModelPoints
from fastcashflow.projection import Cashflows, project_cashflows
from fastcashflow.statemodel import (
    WAIVER_MODEL,
    compile_state_model,
    compile_state_model_with_duration,
    is_semi_markov,
)


# ---------------------------------------------------------------------------
# Detailed path
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Measurement:
    """Detailed measurement -- BEL, RA and CSM rolled forward over time.

    ``bel``, ``ra`` and ``csm`` are ``(n_mp, n_time+1)`` trajectories; column
    0 is the inception measurement. The CSM roll-forward decomposes as
    ``csm[:, t+1] = csm[:, t] + csm_accretion[:, t] - csm_release[:, t]``.
    ``lic`` is the liability for incurred claims -- zero unless a claims
    settlement pattern is set, which also discounts claims to their payment
    dates in the BEL.
    """

    bel: FloatArray              # (n_mp, n_time+1) -- BEL trajectory
    ra: FloatArray               # (n_mp, n_time+1) -- RA trajectory
    csm: FloatArray              # (n_mp, n_time+1) -- CSM trajectory
    csm_accretion: FloatArray    # (n_mp, n_time)   -- CSM interest accreted each month
    csm_release: FloatArray      # (n_mp, n_time)   -- CSM released each month
    loss_component: FloatArray   # (n_mp,)          -- loss component at inception
    lic: FloatArray              # (n_mp, n_time+1) -- liability for incurred claims
    cashflows: Cashflows
    discount_start: FloatArray   # (n_time,) -- start-of-month discount factors
    discount_mid: FloatArray     # (n_time,) -- mid-month discount factors


def _compute_csm(bel0, ra0, inforce, monthly_rate):
    """CSM at initial recognition (Sec. 38) and deterministic roll-forward (Sec. 44).

    Pure-array orchestration: fulfilment cash flows ``FCF = BEL + RA``,
    initial CSM = ``max(0, -FCF)``, loss component = ``max(0, FCF)``, then
    the CSM is rolled forward in :func:`_csm_kernel` (interest accretion at
    the locked-in monthly rate, release proportional to coverage units --
    in-force here).

    ``inforce`` is ``(n_mp, n_time)`` (the coverage-unit series), ``bel0`` /
    ``ra0`` are ``(n_mp,)``. Returns
    ``(csm, accretion, release, loss_component)``.
    """
    fcf = bel0 + ra0
    csm0 = np.maximum(0.0, -fcf)
    loss_component = np.maximum(0.0, fcf)
    csm, accretion, release = _csm_kernel(csm0, inforce, monthly_rate)
    return csm, accretion, release, loss_component


def measure(model_points: ModelPoints, assumptions: Assumptions) -> Measurement:
    """Detailed GMM measurement: BEL, RA and CSM rolled forward over time."""
    proj = project_cashflows(model_points, assumptions)
    claim_cf, morbidity_cf = proj.claim_cf, proj.morbidity_cf
    monthly_rate = discount_monthly_curve(assumptions, proj.n_time)
    if assumptions.settlement_pattern is None:
        lic = np.zeros((claim_cf.shape[0], proj.n_time + 1))
    else:
        lic = _settlement_lic(claim_cf + morbidity_cf, assumptions.settlement_pattern)
        # Claims are paid over the pattern, not at incurrence -- discount
        # them to their payment dates in the fulfilment cash flows. With a
        # discount curve we use the in-year scalar (Sec. 40 / B71 -- the
        # rate at the month of incurrence is the right reference); the
        # full-curve treatment would require a time-varying settlement
        # factor inside the kernel, deferred.
        factor = _settlement_factor(assumptions.settlement_pattern, assumptions.discount_monthly)
        claim_cf = claim_cf * factor
        morbidity_cf = morbidity_cf * factor
    discount_start, discount_mid = discount_factors_from_curve(monthly_rate)

    bel, pv_claims, pv_morbidity, pv_disability, pv_survival = _rollforward_kernel(
        claim_cf, morbidity_cf, proj.disability_cf, proj.expense_cf,
        proj.premium_cf, proj.annuity_cf, proj.maturity_cf,
        model_points.term_months, monthly_rate,
    )
    z = _norm_ppf(assumptions.ra_confidence)
    cl_margin = z * (assumptions.mortality_cv * pv_claims
                     + assumptions.morbidity_cv * pv_morbidity
                     + assumptions.disability_cv * pv_disability
                     + assumptions.longevity_cv * pv_survival)
    if assumptions.ra_method == "confidence_level":
        ra = cl_margin
    elif assumptions.ra_method == "cost_of_capital":
        ra = _cost_of_capital_ra(
            cl_margin, monthly_rate, assumptions.cost_of_capital_rate
        )
    else:
        raise ValueError(
            "ra_method must be 'confidence_level' or 'cost_of_capital', "
            f"got {assumptions.ra_method!r}"
        )
    csm, csm_accretion, csm_release, loss_component = _compute_csm(
        bel[:, 0], ra[:, 0], proj.inforce, monthly_rate,
    )

    return Measurement(
        bel=bel,
        ra=ra,
        csm=csm,
        csm_accretion=csm_accretion,
        csm_release=csm_release,
        loss_component=loss_component,
        lic=lic,
        cashflows=proj,
        discount_start=discount_start,
        discount_mid=discount_mid,
    )


# ---------------------------------------------------------------------------
# Fast path -- fused, memory-minimal valuation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Valuation:
    """Headline IFRS 17 GMM valuation. Each array has shape ``(n_mp,)``."""

    bel: FloatArray             # Best Estimate of Liability
    ra: FloatArray              # Risk Adjustment
    csm: FloatArray             # CSM at initial recognition
    loss_component: FloatArray  # loss component at inception (onerous contracts)


# ---------------------------------------------------------------------------
# Codegen specialisation -- the multi-state value kernel
# ---------------------------------------------------------------------------
#
# The multi-state CPU value() kernel is generated per StateModel: the state
# count, the full edge topology, the lump-sum flags, and which states pay
# premium or a benefit all become part of the generated Python source so
# numba's compiled inner loop has no array indirections left, only scalar
# arithmetic on register-resident occupancy. One source file lives on disk
# per unique topology (see _codegen_cache_dir), and numba caches its native
# code next to it -- compile-once per topology across Python processes.
#
# Earlier iterations of this engine kept a Markov-only closure factory plus
# hand-unrolled n_states=2 and n_states=3 kernels alongside the codegen
# path. They became dead code once the codegen path was extended to all
# n_states>=2 (commit e5e8e83) and have been removed; the git history
# carries them for anyone who wants to read the earlier shape.


def _codegen_value_kernel_source(n_states, edge_from, edge_to, edge_lump_sum,
                                 premium_state, benefit_state) -> str:
    """Generate the Python source of a fully-specialised value kernel.

    All structural parameters (n_states, edge topology, lump-sum flags,
    premium- and benefit-paying states) are baked into the source as
    literals. The returned text is intended for ``exec`` in a namespace
    that exposes ``np``, ``njit`` and ``prange``.
    """
    n_edges = len(edge_from)
    edge_from = [int(x) for x in edge_from]
    edge_to = [int(x) for x in edge_to]
    edge_lump_sum = [bool(x) for x in edge_lump_sum]
    premium_state = [bool(x) for x in premium_state]
    benefit_state = [bool(x) for x in benefit_state]

    sum_all = " + ".join(f"occ_{i}" for i in range(n_states))
    sum_prem = " + ".join(f"occ_{i}" for i in range(n_states)
                          if premium_state[i]) or "0.0"
    sum_ben = " + ".join(f"occ_{i}" for i in range(n_states)
                         if benefit_state[i]) or "0.0"

    L: list[str] = []

    def line(indent: int, text: str) -> None:
        L.append(" " * indent + text)

    # Module-level prologue. The generated text is written to a real .py
    # file under the on-disk cache so numba's @njit(cache=True) can anchor
    # its own compile cache to that file -- without a source file numba's
    # cache silently no-ops and every Python process pays the JIT cost.
    line(0, '"""Auto-generated by fastcashflow.engine.'
            '_codegen_value_kernel_source -- do not edit."""')
    line(0, "import numpy as np")
    line(0, "from numba import njit, prange")
    line(0, "")
    line(0, "")

    def emit_init(indent: int) -> None:
        for i in range(n_states):
            line(indent, f"occ_{i} = 0.0")
        line(indent, "if ss == 0:")
        line(indent + 4, "occ_0 = cnt")
        for i in range(1, n_states):
            line(indent, f"elif ss == {i}:")
            line(indent + 4, f"occ_{i} = cnt")

    def emit_edge_step(indent: int, scale: str = "",
                       include_lump: bool = True) -> None:
        for i in range(n_states):
            line(indent, f"occ_next_{i} = 0.0")
        for e in range(n_edges):
            ef, et, ls = edge_from[e], edge_to[e], edge_lump_sum[e]
            line(indent,
                 f"flow_{e} = occ_{ef}{scale} * edge_prob[sx, ridx, year, {e}]")
            line(indent, f"occ_next_{et} += flow_{e}")
            if include_lump and ls:
                line(indent,
                     f"pd += flow_{e} * disability_benefit[mp] * dm")
        for i in range(n_states):
            line(indent, f"occ_{i} = occ_next_{i}")

    line(0, "@njit(parallel=True, cache=True)")
    line(0, "def kernel(edge_from, edge_to, edge_prob, edge_lump_sum,")
    line(0, "           premium_state, benefit_state, start_state, "
            "issue_index, sex,")
    line(0, "           term_months, count, level_premium, single_premium,")
    line(0, "           premium_term_months, premium_frequency, "
            "annuity_frequency,")
    line(0, "           cov_kind, cov_amount, cov_offset, cov_rates, "
            "cov_risk,")
    line(0, "           cov_is_diagnosis, maturity_benefit, "
            "annuity_payment,")
    line(0, "           disability_income, disability_benefit,")
    line(0, "           expense_acquisition, maint_inflated_monthly,")
    line(0, "           discount_start, discount_mid, mortality_factor,")
    line(0, "           morbidity_factor, longevity_factor, "
            "disability_factor,")
    line(0, "           cov_waiting, cov_reduction_end, "
            "cov_reduction_factor):")
    line(4, "n_mp = issue_index.shape[0]")
    line(4, "bel = np.empty(n_mp)")
    line(4, "ra = np.empty(n_mp)")
    line(4, "csm = np.empty(n_mp)")
    line(4, "loss_component = np.empty(n_mp)")

    line(4, "for mp in prange(n_mp):")
    line(8, "term = term_months[mp]")
    line(8, "premium_term = premium_term_months[mp]")
    line(8, "prem_freq = premium_frequency[mp]")
    line(8, "ann_freq = annuity_frequency[mp]")
    line(8, "ridx = issue_index[mp]")
    line(8, "sx = sex[mp]")
    line(8, "cnt = count[mp]")
    line(8, "premium = level_premium[mp]")
    line(8, "annuity = annuity_payment[mp]")
    line(8, "c_start = cov_offset[mp]")
    line(8, "c_end = cov_offset[mp + 1]")
    line(8, "ss = start_state[mp]")
    emit_init(8)
    line(8, "pc = 0.0")
    line(8, "pcm = 0.0")
    line(8, "pd = 0.0")
    line(8, "pp = 0.0")
    line(8, "pe = 0.0")
    line(8, "pa = 0.0")
    line(8, "last_year = -1")
    line(8, "claim_rate = 0.0")
    line(8, "morb_rate = 0.0")
    line(8, "prem_due = 0")
    line(8, "ann_due = 0")
    line(8, "prem_left = premium_term")

    # Main t loop
    line(8, "for t in range(term):")
    line(12, "year = t // 12")
    line(12, "if year != last_year:")
    line(16, "claim_rate = 0.0")
    line(16, "morb_rate = 0.0")
    line(16, "for k in range(c_start, c_end):")
    line(20, "kind = cov_kind[k]")
    line(20, "if cov_is_diagnosis[kind]:")
    line(24, "continue")
    line(20, "if cov_waiting[k] != 0 or cov_reduction_end[k] != 0:")
    line(24, "continue")
    line(20, "rate = cov_rates[kind, sx, ridx, year] * cov_amount[k]")
    line(20, "if cov_risk[kind] == 0:")
    line(24, "claim_rate += rate")
    line(20, "else:")
    line(24, "morb_rate += rate")
    line(16, "last_year = year")
    line(12, f"ift = {sum_all}")
    line(12, f"prem_occ = {sum_prem}")
    line(12, f"benefit_occ = {sum_ben}")
    line(12, "ds = discount_start[t]")
    line(12, "dm = discount_mid[t]")
    line(12, "single = prem_occ * single_premium[mp] if t == 0 else 0.0")
    line(12, "if prem_due == 0 and prem_left > 0:")
    line(16, "level = prem_occ * premium")
    line(16, "prem_due = prem_freq - 1")
    line(12, "else:")
    line(16, "level = 0.0")
    line(16, "prem_due -= 1")
    line(12, "prem_left -= 1")
    line(12, "pp += (level + single) * ds")
    line(12, "pc += ift * claim_rate * dm")
    line(12, "pcm += ift * morb_rate * dm")
    line(12, "if ann_due == 0:")
    line(16, "pa += ift * annuity * ds")
    line(16, "ann_due = ann_freq - 1")
    line(12, "else:")
    line(16, "ann_due -= 1")
    line(12, "pd += benefit_occ * disability_income[mp] * dm")
    line(12, "acquisition = cnt * expense_acquisition if t == 0 else 0.0")
    line(12, "pe += (acquisition + ift * maint_inflated_monthly[t]) * dm")
    emit_edge_step(12, scale="", include_lump=True)

    line(8, f"total = {sum_all}")
    line(8, "pm = total * maturity_benefit[mp] * discount_start[term]")

    # Coverage-rule pass
    line(8, "for k in range(c_start, c_end):")
    line(12, "kind = cov_kind[k]")
    line(12, "if cov_is_diagnosis[kind]:")
    line(16, "continue")
    line(12, "wait = cov_waiting[k]")
    line(12, "red_end = cov_reduction_end[k]")
    line(12, "if wait == 0 and red_end == 0:")
    line(16, "continue")
    line(12, "benefit = cov_amount[k]")
    line(12, "red_factor = cov_reduction_factor[k]")
    line(12, "mortality_risk = cov_risk[kind] == 0")
    emit_init(12)
    line(12, "for t in range(term):")
    line(16, "year = t // 12")
    line(16, "if t >= wait:")
    line(20, "mult = red_factor if t < red_end else 1.0")
    line(20, f"inf = {sum_all}")
    line(20, "contrib = (inf * cov_rates[kind, sx, ridx, year]")
    line(20, "           * benefit * mult * discount_mid[t])")
    line(20, "if mortality_risk:")
    line(24, "pc += contrib")
    line(20, "else:")
    line(24, "pcm += contrib")
    emit_edge_step(16, scale="", include_lump=False)

    # Diagnosis pass
    line(8, "for k in range(c_start, c_end):")
    line(12, "kind = cov_kind[k]")
    line(12, "if not cov_is_diagnosis[kind]:")
    line(16, "continue")
    line(12, "benefit = cov_amount[k]")
    line(12, "wait = cov_waiting[k]")
    line(12, "red_end = cov_reduction_end[k]")
    line(12, "red_factor = cov_reduction_factor[k]")
    emit_init(12)
    line(12, "d_year = -1")
    line(12, "d_rate = 0.0")
    line(12, "for t in range(term):")
    line(16, "year = t // 12")
    line(16, "if year != d_year:")
    line(20, "d_rate = cov_rates[kind, sx, ridx, year]")
    line(20, "d_year = year")
    line(16, "if t >= wait:")
    line(20, "mult = red_factor if t < red_end else 1.0")
    line(20, f"healthy = {sum_all}")
    line(20, "pcm += healthy * d_rate * benefit * mult * discount_mid[t]")
    line(16, "undiag = 1.0 - d_rate")
    emit_edge_step(16, scale=" * undiag", include_lump=False)

    line(8, "bel_mp = pc + pcm + pd + pm + pa + pe - pp")
    line(8, "ra_mp = (mortality_factor * pc + morbidity_factor * pcm")
    line(8, "         + disability_factor * pd "
            "+ longevity_factor * (pm + pa))")
    line(8, "fcf = bel_mp + ra_mp")
    line(8, "bel[mp] = bel_mp")
    line(8, "ra[mp] = ra_mp")
    line(8, "csm[mp] = max(0.0, -fcf)")
    line(8, "loss_component[mp] = max(0.0, fcf)")
    line(4, "return bel, ra, csm, loss_component")

    return "\n".join(L)


_VALUE_KERNEL_CODEGEN_CACHE: dict = {}


def _codegen_cache_dir() -> Path:
    """Return the on-disk directory holding generated kernel source files.

    Honours ``XDG_CACHE_HOME`` and falls back to ``~/.cache``; the
    fastcashflow-private subdirectory is created on demand.
    """
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    cache = root / "fastcashflow" / "codegen"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    Two processes racing on the same topology hash produce byte-identical
    sources (codegen is deterministic), so the final file content is the
    same regardless of who wins -- the atomic replace just avoids a
    partially-written file being observed.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def _get_value_kernel_codegen(n_states, edge_from, edge_to, edge_lump_sum,
                              premium_state, benefit_state):
    """Return a codegen-specialised kernel for the given state machine.

    Two-level cache:

    * Process-local dict keyed on the topology -- a repeat lookup in the
      same Python process returns the already-imported function with no
      filesystem touch.
    * On-disk ``.py`` file under the codegen cache directory, named by a
      hash of the generated source. The first call for a given topology
      generates the source and writes the file atomically; ``@njit(cache
      =True)`` inside the generated module then persists the compiled
      bytecode in its ``__pycache__`` so subsequent Python processes pay
      no JIT cost. Hashing the *source* automatically invalidates the
      cache when the codegen logic itself changes.
    """
    key = (
        int(n_states),
        tuple(int(x) for x in edge_from),
        tuple(int(x) for x in edge_to),
        tuple(bool(x) for x in edge_lump_sum),
        tuple(bool(x) for x in premium_state),
        tuple(bool(x) for x in benefit_state),
    )
    cached = _VALUE_KERNEL_CODEGEN_CACHE.get(key)
    if cached is not None:
        return cached

    src = _codegen_value_kernel_source(
        n_states, edge_from, edge_to, edge_lump_sum,
        premium_state, benefit_state,
    )
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cache_path = _codegen_cache_dir() / f"value_kernel_{digest}.py"
    if not cache_path.exists():
        _atomic_write_text(cache_path, src)
    module_name = f"_fastcashflow_codegen_{digest}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, cache_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    kernel = module.kernel
    _VALUE_KERNEL_CODEGEN_CACHE[key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Semi-Markov codegen (Phase (c) prototype -- cancer reincidence)
# ---------------------------------------------------------------------------
#
# When the StateModel declares any state with ``duration_max > 0`` the engine
# tracks per-cohort occupancy in that state -- ``occ[state, tau]`` where
# ``tau`` is the sojourn time (months since entering the state). Transitions
# marked ``duration_dependent=True`` then carry per-cohort rates, the natural
# way to express recovery, reincidence and exclusion-period effects.
#
# The semi-Markov codegen extends the existing codegen pattern: every cohort
# of every state becomes a scalar local (``occ_<s>_<tau>``), every edge
# unrolls per-cohort (a residual stay advances the cohort with absorbing
# semantics at the last cohort, a transient transition enters the destination
# state's cohort 0). The generated source goes through the same disk-cached
# numba compile path as the Markov codegen -- compile-once per topology.
#
# Coverage-rule and diagnosis-coverage passes are emitted as no-op stubs when
# the StateModel is semi-Markov: this prototype targets the cancer-reincidence
# product, which has no rule-bearing or diagnosis coverages on the model
# points themselves -- the reincidence benefit rides the transition lump-sum.
# Full coverage-rule / diagnosis support for semi-Markov is a follow-up.


def _codegen_value_kernel_source_semi_markov(
    n_states, state_duration_max, edge_from, edge_to, edge_lump_sum,
    premium_state, benefit_state,
) -> str:
    """Generate the Python source of a semi-Markov-aware value kernel.

    Same structure as :func:`_codegen_value_kernel_source` but with per-state
    cohort scalars and per-cohort edge processing. State ``s`` with
    ``state_duration_max[s] = D`` produces ``occ_s_0, occ_s_1, ..., occ_s_{D-1}``
    local scalars; the last cohort absorbs everyone with sojourn time at
    least ``D - 1`` months.
    """
    n_edges = len(edge_from)
    edge_from = [int(x) for x in edge_from]
    edge_to = [int(x) for x in edge_to]
    edge_lump_sum = [bool(x) for x in edge_lump_sum]
    premium_state = [bool(x) for x in premium_state]
    benefit_state = [bool(x) for x in benefit_state]
    D = [int(x) for x in state_duration_max]

    def occ(s, tau):
        return f"occ_{s}_{tau}"

    def occ_next(s, tau):
        return f"occ_next_{s}_{tau}"

    # All cohort scalars across all states -- for sums and resets.
    all_occ = [occ(s, tau) for s in range(n_states) for tau in range(D[s])]
    all_next = [occ_next(s, tau) for s in range(n_states) for tau in range(D[s])]
    sum_all = " + ".join(all_occ)
    sum_prem = " + ".join(occ(s, tau) for s in range(n_states)
                          if premium_state[s]
                          for tau in range(D[s])) or "0.0"
    sum_ben = " + ".join(occ(s, tau) for s in range(n_states)
                         if benefit_state[s]
                         for tau in range(D[s])) or "0.0"

    L: list[str] = []

    def line(indent: int, text: str) -> None:
        L.append(" " * indent + text)

    def emit_init(indent: int) -> None:
        """Initialize all cohort scalars to 0, then seat ``cnt`` based on ss."""
        for s in range(n_states):
            for tau in range(D[s]):
                line(indent, f"{occ(s, tau)} = 0.0")
        # Seating: when entering a state via seating, always cohort 0.
        line(indent, "if ss == 0:")
        line(indent + 4, f"{occ(0, 0)} = cnt")
        for s in range(1, n_states):
            line(indent, f"elif ss == {s}:")
            line(indent + 4, f"{occ(s, 0)} = cnt")

    def emit_edge_step(indent: int, include_lump: bool = True,
                       scale: str = "") -> None:
        """Emit a per-timestep advance: reset next-buffer, process every edge
        across every source cohort, then copy back.

        Edge semantics:
          - residual (edge_from[e] == edge_to[e]): cohort tau advances to tau+1,
            with cohort D-1 absorbing (long-tail).
          - transient (edge_from[e] != edge_to[e]): enters destination state's
            cohort 0.
          - exits are not represented as edges; occupancy that doesn't go to
            any next-buffer slot simply leaves the in-force set.

        ``scale`` (e.g. ``" * undiag"``) multiplies every flow -- used by
        the diagnosis-coverage pass to deplete the not-yet-diagnosed
        in-force fraction along the state machine.
        """
        for s in range(n_states):
            for tau in range(D[s]):
                line(indent, f"{occ_next(s, tau)} = 0.0")
        for e in range(n_edges):
            s_from = edge_from[e]
            s_to = edge_to[e]
            is_residual = (s_from == s_to)
            ls = edge_lump_sum[e]
            for tau in range(D[s_from]):
                line(indent,
                     f"flow_{e}_{tau} = {occ(s_from, tau)}{scale} "
                     f"* edge_prob[sx, ridx, year, {e}, {tau}]")
                if is_residual:
                    # Cohort advance; cap at D-1 (absorbing).
                    next_tau = tau + 1 if tau + 1 < D[s_from] else D[s_from] - 1
                    line(indent,
                         f"{occ_next(s_from, next_tau)} += flow_{e}_{tau}")
                else:
                    # Transient: enter destination's cohort 0.
                    line(indent,
                         f"{occ_next(s_to, 0)} += flow_{e}_{tau}")
                if include_lump and ls:
                    line(indent,
                         f"pd += flow_{e}_{tau} "
                         f"* disability_benefit[mp] * dm")
        # Copy back.
        for s in range(n_states):
            for tau in range(D[s]):
                line(indent, f"{occ(s, tau)} = {occ_next(s, tau)}")

    # --- Prologue: imports + decorator ----------------------------------
    line(0, '"""Auto-generated by fastcashflow.engine.'
            '_codegen_value_kernel_source_semi_markov -- do not edit."""')
    line(0, "import numpy as np")
    line(0, "from numba import njit, prange")
    line(0, "")
    line(0, "")
    line(0, "@njit(parallel=True, cache=True)")
    line(0, "def kernel(edge_prob, start_state, issue_index, sex,")
    line(0, "           term_months, count, level_premium, single_premium,")
    line(0, "           premium_term_months, premium_frequency, "
            "annuity_frequency,")
    line(0, "           cov_kind, cov_amount, cov_offset, cov_rates, "
            "cov_risk,")
    line(0, "           cov_is_diagnosis, maturity_benefit, "
            "annuity_payment,")
    line(0, "           disability_income, disability_benefit,")
    line(0, "           expense_acquisition, maint_inflated_monthly,")
    line(0, "           discount_start, discount_mid, mortality_factor,")
    line(0, "           morbidity_factor, longevity_factor, "
            "disability_factor,")
    line(0, "           cov_waiting, cov_reduction_end, "
            "cov_reduction_factor):")
    line(4, "n_mp = issue_index.shape[0]")
    line(4, "bel = np.empty(n_mp)")
    line(4, "ra = np.empty(n_mp)")
    line(4, "csm = np.empty(n_mp)")
    line(4, "loss_component = np.empty(n_mp)")

    line(4, "for mp in prange(n_mp):")
    line(8, "term = term_months[mp]")
    line(8, "premium_term = premium_term_months[mp]")
    line(8, "prem_freq = premium_frequency[mp]")
    line(8, "ann_freq = annuity_frequency[mp]")
    line(8, "ridx = issue_index[mp]")
    line(8, "sx = sex[mp]")
    line(8, "cnt = count[mp]")
    line(8, "premium = level_premium[mp]")
    line(8, "annuity = annuity_payment[mp]")
    line(8, "c_start = cov_offset[mp]")
    line(8, "c_end = cov_offset[mp + 1]")
    line(8, "ss = start_state[mp]")
    emit_init(8)
    line(8, "pc = 0.0")
    line(8, "pcm = 0.0")
    line(8, "pd = 0.0")
    line(8, "pp = 0.0")
    line(8, "pe = 0.0")
    line(8, "pa = 0.0")
    line(8, "last_year = -1")
    line(8, "claim_rate = 0.0")
    line(8, "morb_rate = 0.0")
    line(8, "prem_due = 0")
    line(8, "ann_due = 0")
    line(8, "prem_left = premium_term")

    # --- Main t loop ---------------------------------------------------
    line(8, "for t in range(term):")
    line(12, "year = t // 12")
    line(12, "if year != last_year:")
    line(16, "claim_rate = 0.0")
    line(16, "morb_rate = 0.0")
    line(16, "for k in range(c_start, c_end):")
    line(20, "kind = cov_kind[k]")
    line(20, "if cov_is_diagnosis[kind]:")
    line(24, "continue          # diagnosis coverages run separately")
    line(20, "if cov_waiting[k] != 0 or cov_reduction_end[k] != 0:")
    line(24, "continue          # rule-bearing coverages run separately")
    line(20, "rate = cov_rates[kind, sx, ridx, year] * cov_amount[k]")
    line(20, "if cov_risk[kind] == 0:")
    line(24, "claim_rate += rate")
    line(20, "else:")
    line(24, "morb_rate += rate")
    line(16, "last_year = year")
    line(12, f"ift = {sum_all}")
    line(12, f"prem_occ = {sum_prem}")
    line(12, f"benefit_occ = {sum_ben}")
    line(12, "ds = discount_start[t]")
    line(12, "dm = discount_mid[t]")
    line(12, "single = prem_occ * single_premium[mp] if t == 0 else 0.0")
    line(12, "if prem_due == 0 and prem_left > 0:")
    line(16, "level = prem_occ * premium")
    line(16, "prem_due = prem_freq - 1")
    line(12, "else:")
    line(16, "level = 0.0")
    line(16, "prem_due -= 1")
    line(12, "prem_left -= 1")
    line(12, "pp += (level + single) * ds")
    line(12, "pc += ift * claim_rate * dm")
    line(12, "pcm += ift * morb_rate * dm")
    line(12, "if ann_due == 0:")
    line(16, "pa += ift * annuity * ds")
    line(16, "ann_due = ann_freq - 1")
    line(12, "else:")
    line(16, "ann_due -= 1")
    line(12, "pd += benefit_occ * disability_income[mp] * dm")
    line(12, "acquisition = cnt * expense_acquisition if t == 0 else 0.0")
    line(12, "pe += (acquisition + ift * maint_inflated_monthly[t]) * dm")
    emit_edge_step(12, include_lump=True)

    line(8, f"total = {sum_all}")
    line(8, "pm = total * maturity_benefit[mp] * discount_start[term]")

    # --- Coverage-rule pass --------------------------------------------
    # Rule-bearing non-diagnosis coverages: rerun the state machine on a
    # fresh cohort occupancy and apply the per-month benefit multiplier
    # (which can change partway through a year) to the total in-force.
    line(8, "for k in range(c_start, c_end):")
    line(12, "kind = cov_kind[k]")
    line(12, "if cov_is_diagnosis[kind]:")
    line(16, "continue")
    line(12, "wait = cov_waiting[k]")
    line(12, "red_end = cov_reduction_end[k]")
    line(12, "if wait == 0 and red_end == 0:")
    line(16, "continue          # rule-free -- already in the main pass")
    line(12, "benefit = cov_amount[k]")
    line(12, "red_factor = cov_reduction_factor[k]")
    line(12, "mortality_risk = cov_risk[kind] == 0")
    emit_init(12)
    line(12, "for t in range(term):")
    line(16, "year = t // 12")
    line(16, "if t >= wait:")
    line(20, "mult = red_factor if t < red_end else 1.0")
    line(20, f"inf = {sum_all}")
    line(20, "contrib = (inf * cov_rates[kind, sx, ridx, year]")
    line(20, "           * benefit * mult * discount_mid[t])")
    line(20, "if mortality_risk:")
    line(24, "pc += contrib")
    line(20, "else:")
    line(24, "pcm += contrib")
    emit_edge_step(16, include_lump=False)

    # --- Diagnosis-coverage pass ---------------------------------------
    # Diagnosis coverages pay once on first diagnosis. Each coverage runs
    # its claims off a depleting not-yet-diagnosed pool, implemented by
    # scaling every state-machine flow by (1 - d_rate) each timestep so
    # the occupancy itself tracks the joint depletion.
    line(8, "for k in range(c_start, c_end):")
    line(12, "kind = cov_kind[k]")
    line(12, "if not cov_is_diagnosis[kind]:")
    line(16, "continue")
    line(12, "benefit = cov_amount[k]")
    line(12, "wait = cov_waiting[k]")
    line(12, "red_end = cov_reduction_end[k]")
    line(12, "red_factor = cov_reduction_factor[k]")
    emit_init(12)
    line(12, "d_year = -1")
    line(12, "d_rate = 0.0")
    line(12, "for t in range(term):")
    line(16, "year = t // 12")
    line(16, "if year != d_year:")
    line(20, "d_rate = cov_rates[kind, sx, ridx, year]")
    line(20, "d_year = year")
    line(16, "if t >= wait:")
    line(20, "mult = red_factor if t < red_end else 1.0")
    line(20, f"healthy = {sum_all}")
    line(20, "pcm += healthy * d_rate * benefit * mult * discount_mid[t]")
    line(16, "undiag = 1.0 - d_rate")
    emit_edge_step(16, include_lump=False, scale=" * undiag")

    # --- Final output --------------------------------------------------
    line(8, "bel_mp = pc + pcm + pd + pm + pa + pe - pp")
    line(8, "ra_mp = (mortality_factor * pc + morbidity_factor * pcm")
    line(8, "         + disability_factor * pd "
            "+ longevity_factor * (pm + pa))")
    line(8, "fcf = bel_mp + ra_mp")
    line(8, "bel[mp] = bel_mp")
    line(8, "ra[mp] = ra_mp")
    line(8, "csm[mp] = max(0.0, -fcf)")
    line(8, "loss_component[mp] = max(0.0, fcf)")
    line(4, "return bel, ra, csm, loss_component")

    return "\n".join(L)


_VALUE_KERNEL_CODEGEN_SEMI_MARKOV_CACHE: dict = {}


def _get_value_kernel_codegen_semi_markov(
    n_states, state_duration_max, edge_from, edge_to, edge_lump_sum,
    premium_state, benefit_state,
):
    """Return a semi-Markov codegen-specialised kernel for the given
    topology + per-state cohort counts.

    Same two-level cache pattern as :func:`_get_value_kernel_codegen` --
    process-local dict for in-memory hits, content-addressed ``.py`` file
    on disk so ``@njit(cache=True)`` persists the compiled native code
    across processes.
    """
    key = (
        int(n_states),
        tuple(int(x) for x in state_duration_max),
        tuple(int(x) for x in edge_from),
        tuple(int(x) for x in edge_to),
        tuple(bool(x) for x in edge_lump_sum),
        tuple(bool(x) for x in premium_state),
        tuple(bool(x) for x in benefit_state),
    )
    cached = _VALUE_KERNEL_CODEGEN_SEMI_MARKOV_CACHE.get(key)
    if cached is not None:
        return cached

    src = _codegen_value_kernel_source_semi_markov(
        n_states, state_duration_max, edge_from, edge_to, edge_lump_sum,
        premium_state, benefit_state,
    )
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
    cache_path = _codegen_cache_dir() / f"value_kernel_sm_{digest}.py"
    if not cache_path.exists():
        _atomic_write_text(cache_path, src)
    module_name = f"_fastcashflow_codegen_sm_{digest}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, cache_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    kernel = module.kernel
    _VALUE_KERNEL_CODEGEN_SEMI_MARKOV_CACHE[key] = kernel
    return kernel


@njit(parallel=True, cache=True)
def _value_kernel_scalar(issue_index, sex, term_months, count, level_premium,
                         single_premium, premium_term_months, premium_frequency,
                         annuity_frequency, cov_kind, cov_amount, cov_offset,
                         cov_rates, cov_risk, cov_is_diagnosis,
                         maturity_benefit, annuity_payment, expense_acquisition,
                         maint_inflated_monthly, discount_start, discount_mid,
                         mortality_factor, morbidity_factor, longevity_factor,
                         cov_waiting, cov_reduction_end, cov_reduction_factor,
                         survival_monthly):
    """Scalar-inforce fast path of :func:`_value_kernel`.

    Used when the in-force projection collapses to a single survival track --
    no user-supplied StateModel, no waiver inception, every model point
    seated in the active state. The in-force is carried as a scalar; the
    monthly decay is one multiply against the precomputed
    ``survival_monthly[sex, age, year] = (1 - q_monthly) * (1 - l_monthly)``
    table. Numerically identical to ``_value_kernel`` for this configuration
    -- the disability income, disability lump-sum and benefit-state pieces
    of the general kernel evaluate to zero here -- and recovers the
    pre-Phase(b) speed (see ``docs/tutorial/13-why-fast.md``).
    """
    n_mp = issue_index.shape[0]
    bel = np.empty(n_mp)
    ra = np.empty(n_mp)
    csm = np.empty(n_mp)
    loss_component = np.empty(n_mp)

    for mp in prange(n_mp):
        term = term_months[mp]
        premium_term = premium_term_months[mp]
        prem_freq = premium_frequency[mp]
        ann_freq = annuity_frequency[mp]
        ridx = issue_index[mp]
        sx = sex[mp]
        cnt = count[mp]
        premium = level_premium[mp]
        annuity = annuity_payment[mp]
        c_start = cov_offset[mp]
        c_end = cov_offset[mp + 1]
        inforce = cnt
        pc = 0.0          # PV of death claims (mortality risk)
        pcm = 0.0         # PV of health claims (morbidity risk)
        pp = 0.0
        pe = 0.0
        pa = 0.0
        last_year = -1
        claim_rate = 0.0
        morb_rate = 0.0
        # Counters replace modulo / less-than checks in the inner loop --
        # ``prem_due`` ticks down to the next premium-paying month,
        # ``ann_due`` to the next annuity month, and ``prem_left`` to the
        # end of the premium-paying term. Profiling shows the modulo /
        # comparison form costs ~2/3 of the inner-loop time at large
        # portfolios -- the counter form lets the compiler keep the loop
        # branch-light and 1M MP runs in ~50 ms again.
        prem_due = 0
        ann_due = 0
        prem_left = premium_term
        for t in range(term):
            year = t // 12
            if year != last_year:
                claim_rate = 0.0
                morb_rate = 0.0
                for k in range(c_start, c_end):
                    kind = cov_kind[k]
                    if cov_is_diagnosis[kind]:
                        continue
                    if cov_waiting[k] != 0 or cov_reduction_end[k] != 0:
                        continue
                    rate = cov_rates[kind, sx, ridx, year] * cov_amount[k]
                    if cov_risk[kind] == 0:
                        claim_rate += rate
                    else:
                        morb_rate += rate
                last_year = year
            ds = discount_start[t]
            dm = discount_mid[t]
            single = inforce * single_premium[mp] if t == 0 else 0.0
            if prem_due == 0 and prem_left > 0:
                level = inforce * premium
                prem_due = prem_freq - 1
            else:
                level = 0.0
                prem_due -= 1
            prem_left -= 1
            pp += (level + single) * ds
            pc += inforce * claim_rate * dm
            pcm += inforce * morb_rate * dm
            if ann_due == 0:
                pa += inforce * annuity * ds
                ann_due = ann_freq - 1
            else:
                ann_due -= 1
            acquisition = cnt * expense_acquisition if t == 0 else 0.0
            pe += (acquisition + inforce * maint_inflated_monthly[t]) * dm
            inforce *= survival_monthly[sx, ridx, year]
        pm = inforce * maturity_benefit[mp] * discount_start[term]
        # Non-diagnosis coverages with a waiting or reduced-benefit rule:
        # rerun the survival on the same scalar track so the benefit
        # multiplier (which can change mid-year) applies cleanly.
        for k in range(c_start, c_end):
            kind = cov_kind[k]
            if cov_is_diagnosis[kind]:
                continue
            wait = cov_waiting[k]
            red_end = cov_reduction_end[k]
            if wait == 0 and red_end == 0:
                continue
            benefit = cov_amount[k]
            red_factor = cov_reduction_factor[k]
            mortality_risk = cov_risk[kind] == 0
            inf = cnt
            for t in range(term):
                year = t // 12
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    contrib = (inf * cov_rates[kind, sx, ridx, year]
                               * benefit * mult * discount_mid[t])
                    if mortality_risk:
                        pc += contrib
                    else:
                        pcm += contrib
                inf *= survival_monthly[sx, ridx, year]
        # Diagnosis coverages: claims run off a depleting "not yet diagnosed"
        # pool, which depletes both by survival and by the diagnosis rate.
        for k in range(c_start, c_end):
            kind = cov_kind[k]
            if not cov_is_diagnosis[kind]:
                continue
            benefit = cov_amount[k]
            wait = cov_waiting[k]
            red_end = cov_reduction_end[k]
            red_factor = cov_reduction_factor[k]
            healthy = cnt
            d_year = -1
            d_rate = 0.0
            for t in range(term):
                year = t // 12
                if year != d_year:
                    d_rate = cov_rates[kind, sx, ridx, year]
                    d_year = year
                if t >= wait:
                    mult = red_factor if t < red_end else 1.0
                    pcm += healthy * d_rate * benefit * mult * discount_mid[t]
                healthy *= survival_monthly[sx, ridx, year] * (1.0 - d_rate)
        bel_mp = pc + pcm + pm + pa + pe - pp
        ra_mp = (mortality_factor * pc + morbidity_factor * pcm
                 + longevity_factor * (pm + pa))
        fcf = bel_mp + ra_mp
        bel[mp] = bel_mp
        ra[mp] = ra_mp
        csm[mp] = max(0.0, -fcf)
        loss_component[mp] = max(0.0, fcf)

    return bel, ra, csm, loss_component


def value(
    model_points: ModelPoints,
    assumptions: Assumptions,
    *,
    backend: str = "cpu",
    discount_curve: FloatArray | None = None,
) -> Valuation:
    """Fast GMM valuation: BEL, RA and CSM per model point.

    One fused kernel; no per-month arrays are materialised. This is the
    memory-minimal, fastest path for large-scale valuation. For full cash
    flow / CSM trajectories use :func:`measure`.

    Parameters
    ----------
    backend :
        ``"cpu"`` (default) runs the numba parallel kernel across cores.
        ``"gpu"`` runs the CUDA kernel; it needs a CUDA device and is worth
        it only at large scale (kernel-launch and transfer cost is fixed).
    discount_curve :
        Optional ``(n_time,)`` array of annual discount rates -- one per
        projection month, a power-user override for the stochastic case
        where the rate must vary month by month. ``None`` (the default)
        uses the scalar or per-year curve on
        ``assumptions.discount_annual``; when supplied, it overrides that
        and bypasses the curves layer for the discount step.
    """
    if assumptions.ra_method != "confidence_level":
        raise ValueError(
            "value() computes the confidence-level RA only; use measure() "
            f"for ra_method={assumptions.ra_method!r}"
        )
    n_time = int(model_points.term_months.max())
    n_years = (n_time + 11) // 12

    # Mortality and lapse are evaluated on a dense sex x [min, max] issue-age
    # x duration grid. Using the age range rather than the exact distinct
    # ages avoids an O(n log n) sort (np.unique): min/max and the index
    # subtraction are O(n), and the few unused ages cost nothing -- the
    # assumption grid is tiny.
    min_age = int(model_points.issue_age.min())
    max_age = int(model_points.issue_age.max())
    durations = np.arange(n_years)
    sex_grid, issue_age_grid, duration_grid = np.meshgrid(
        np.array([0, 1]), np.arange(min_age, max_age + 1), durations,
        indexing="ij",
    )
    # Rates are supplied annual; the engine converts each to a monthly rate
    # on the constant-force basis (see assumptions.annual_to_monthly).
    mortality_annual_grid = assumptions.mortality_annual(
        sex_grid, issue_age_grid, duration_grid)
    mortality_grid = np.ascontiguousarray(annual_to_monthly(mortality_annual_grid))
    issue_index = (model_points.issue_age - min_age).astype(np.int64)
    lapse_grid = np.ascontiguousarray(annual_to_monthly(
        assumptions.lapse_annual(sex_grid, issue_age_grid, duration_grid)))
    # Fast path: when no waiver / paid-up mechanic is active and every model
    # point is seated in the active state, the in-force is a single survival
    # track. The scalar kernel carries it as one number and runs the
    # pre-Phase(b) speed path; the N-state kernel is reserved for products
    # that genuinely need an occupancy vector.
    fast_path = (backend == "cpu"
                 and assumptions.state_model is None
                 and assumptions.waiver_incidence_annual is None
                 and not np.any(model_points.state))
    if not fast_path:
        if assumptions.waiver_incidence_annual is None:
            waiver_grid = np.zeros_like(mortality_grid)
        else:
            waiver_grid = np.ascontiguousarray(annual_to_monthly(
                assumptions.waiver_incidence_annual(
                    sex_grid, issue_age_grid, duration_grid)))
        # In-force state machine -- the StateModel composes the transition
        # edges for the generic occupancy recursion (see
        # fastcashflow.statemodel). The rates are on the sex x age x duration
        # grid the kernel indexes.
        if assumptions.state_model is None:
            warnings.warn(
                "value() defaulting to WAIVER_MODEL because "
                "waiver_incidence_annual is set or model_points.state has "
                "non-zero entries. Set assumptions.state_model explicitly "
                "(e.g. STATE_MODELS['WAIVER']) -- the implicit fallback is "
                "deprecated and will be removed in a future major version.",
                DeprecationWarning, stacklevel=2,
            )
            state_model = WAIVER_MODEL
        else:
            state_model = assumptions.state_model
        semi_markov = is_semi_markov(state_model)
        if semi_markov:
            # Phase (c) path -- a state declared ``duration_max > 0`` tracks
            # per-cohort occupancy. The rate dict carries one entry per name
            # the model references; duration-dependent rates land here as
            # 4D arrays (sex, age, year, cohort), static rates as 3D.
            max_cohort = max(s.duration_max for s in state_model.states
                              if s.duration_max > 0)
            rate_dict = {"mortality": mortality_grid,
                          "lapse": lapse_grid}
            if assumptions.waiver_incidence_annual is not None:
                waiver_grid = np.ascontiguousarray(annual_to_monthly(
                    assumptions.waiver_incidence_annual(
                        sex_grid, issue_age_grid, duration_grid)))
                rate_dict["waiver_incidence"] = waiver_grid
            if assumptions.ci_incidence_annual is not None:
                ci_inc_grid = np.ascontiguousarray(annual_to_monthly(
                    assumptions.ci_incidence_annual(
                        sex_grid, issue_age_grid, duration_grid)))
                rate_dict["ci_incidence"] = ci_inc_grid
            if (assumptions.ci_reincidence_annual is not None
                    or assumptions.disability_recovery_annual is not None):
                # Build the (sex, age, year, cohort) grid by sweeping cohort
                # months 0..max_cohort-1. Duration-dependent rate callables
                # take the four-argument signature -- ``state_duration``
                # (the sojourn-time cohort index) lets a 면책 (exclusion)
                # window on reincidence or the duration-since-disablement
                # taper on a DI recovery rate drop straight in.
                cohort_idx = np.arange(max_cohort)
                sg4, ag4, dg4, cg4 = np.meshgrid(
                    np.array([0, 1]),
                    np.arange(min_age, max_age + 1),
                    durations,
                    cohort_idx,
                    indexing="ij",
                )
                if assumptions.ci_reincidence_annual is not None:
                    rate_dict["ci_reincidence"] = np.ascontiguousarray(
                        annual_to_monthly(
                            assumptions.ci_reincidence_annual(
                                sg4, ag4, dg4, cg4)))
                if assumptions.disability_recovery_annual is not None:
                    rate_dict["disability_recovery"] = np.ascontiguousarray(
                        annual_to_monthly(
                            assumptions.disability_recovery_annual(
                                sg4, ag4, dg4, cg4)))
            (edge_from, edge_to, edge_prob, edge_lump_sum, n_states,
             premium_state, benefit_state,
             state_duration_max) = compile_state_model_with_duration(
                state_model, rate_dict,
            )
            # compile_state_model_with_duration returns ``edge_prob`` shape
            # ``(n_edges, sex, age, year, max_D)``. Transpose to put the
            # (sex, age, year) lookup axes outermost and the edge / cohort
            # indices last, so a per-edge per-cohort access for one
            # (sex, age, year) stays in cache.
            edge_prob = np.ascontiguousarray(
                np.transpose(edge_prob, (1, 2, 3, 0, 4)))
        else:
            (edge_from, edge_to, edge_prob, edge_lump_sum, n_states,
             premium_state, benefit_state) = compile_state_model(
                state_model,
                {"mortality": mortality_grid,
                 "waiver_incidence": waiver_grid,
                 "lapse": lapse_grid},
            )
            # compile_state_model returns ``edge_prob`` with the edge axis
            # first -- (n_edges, sex, age, year). Transpose so the edge axis
            # is innermost: all edges for a given (sex, age, year) lookup
            # land in one cache line, ~25% faster on the multi-state hot
            # path.
            edge_prob = np.ascontiguousarray(
                np.transpose(edge_prob, (1, 2, 3, 0)))
            state_duration_max = None
        start_state = np.asarray(state_model.seating, np.int64)[model_points.state]
    cov_is_diagnosis, cov_risk = coverage_arrays(assumptions.riders)
    # coverage_rates stacks the annual mortality and rider rates; the whole
    # stack is converted to monthly. Slab 0 is the monthly mortality above.
    cov_rates = np.ascontiguousarray(annual_to_monthly(coverage_rates(
        mortality_annual_grid, [r.rate for r in assumptions.riders], sex_grid,
        issue_age_grid, duration_grid,
    )))

    maint_inflated_monthly = (maintenance_monthly_curve(assumptions, n_time)
                              * inflation_index(assumptions, n_time))
    if discount_curve is None:
        discount_start, discount_mid = discount_factors(assumptions, n_time)
    else:
        discount_curve = np.asarray(discount_curve, dtype=np.float64)
        if discount_curve.shape != (n_time,):
            raise ValueError(
                f"discount_curve must have shape ({n_time},) -- one annual "
                f"rate per projection month -- got {discount_curve.shape}"
            )
        monthly_curve = (1.0 + discount_curve) ** (1.0 / 12.0) - 1.0
        discount_start, discount_mid = discount_factors_from_curve(monthly_curve)
    z = _norm_ppf(assumptions.ra_confidence)
    mortality_factor = z * assumptions.mortality_cv
    morbidity_factor = z * assumptions.morbidity_cv
    longevity_factor = z * assumptions.longevity_cv
    disability_factor = z * assumptions.disability_cv

    # A claims settlement pattern discounts claims to their payment dates;
    # scaling the coverage amounts carries that into the fused kernel.
    cov_amount = model_points.cov_amount
    if assumptions.settlement_pattern is not None:
        cov_amount = cov_amount * _settlement_factor(
            assumptions.settlement_pattern, assumptions.discount_monthly
        )

    if fast_path:
        survival_monthly = np.ascontiguousarray(
            (1.0 - mortality_grid) * (1.0 - lapse_grid)
        )
        bel, ra, csm, loss_component = _value_kernel_scalar(
            issue_index,
            model_points.sex,
            model_points.term_months,
            model_points.count,
            model_points.level_premium,
            model_points.single_premium,
            model_points.premium_term_months,
            model_points.premium_frequency_months,
            model_points.annuity_frequency_months,
            model_points.cov_kind,
            cov_amount,
            model_points.cov_offset,
            cov_rates,
            cov_risk,
            cov_is_diagnosis,
            model_points.maturity_benefit,
            model_points.annuity_payment,
            assumptions.expense_acquisition,
            maint_inflated_monthly,
            discount_start,
            discount_mid,
            mortality_factor,
            morbidity_factor,
            longevity_factor,
            model_points.cov_waiting,
            model_points.cov_reduction_end,
            model_points.cov_reduction_factor,
            survival_monthly,
        )
        return Valuation(bel=bel, ra=ra, csm=csm, loss_component=loss_component)

    # The CPU kernel takes (n_states, n_edges) via Python closure -- they are
    # not in the args tuple. The GPU kernel still takes n_states explicitly
    # as a runtime arg, so the two paths build the call list separately.
    n_edges = int(edge_from.shape[0])
    common_args = (
        edge_from,
        edge_to,
        edge_prob,
        edge_lump_sum,
        premium_state,
        benefit_state,
        start_state,
        issue_index,
        model_points.sex,
        model_points.term_months,
        model_points.count,
        model_points.level_premium,
        model_points.single_premium,
        model_points.premium_term_months,
        model_points.premium_frequency_months,
        model_points.annuity_frequency_months,
        model_points.cov_kind,
        cov_amount,
        model_points.cov_offset,
        cov_rates,
        cov_risk,
        cov_is_diagnosis,
        model_points.maturity_benefit,
        model_points.annuity_payment,
        model_points.disability_income,
        model_points.disability_benefit,
        assumptions.expense_acquisition,
        maint_inflated_monthly,
        discount_start,
        discount_mid,
        mortality_factor,
        morbidity_factor,
        longevity_factor,
        disability_factor,
    )

    if backend == "cpu":
        if state_duration_max is not None:
            # Phase (c) semi-Markov path. The kernel takes a thinner arg
            # tuple than the Markov codegen -- the edge topology and the
            # per-state cohort counts are baked into the generated source
            # (one cache file per unique StateModel + duration shape).
            # Coverage-rule and diagnosis-coverage passes are emitted
            # alongside the main pass so contracts mixing semi-Markov
            # cohort tracking with rule-bearing or diagnosis riders work
            # in a single value() call.
            kernel = _get_value_kernel_codegen_semi_markov(
                n_states, state_duration_max, edge_from, edge_to,
                edge_lump_sum, premium_state, benefit_state,
            )
            bel, ra, csm, loss_component = kernel(
                edge_prob, start_state, issue_index,
                model_points.sex,
                model_points.term_months,
                model_points.count,
                model_points.level_premium,
                model_points.single_premium,
                model_points.premium_term_months,
                model_points.premium_frequency_months,
                model_points.annuity_frequency_months,
                model_points.cov_kind,
                cov_amount,
                model_points.cov_offset,
                cov_rates,
                cov_risk,
                cov_is_diagnosis,
                model_points.maturity_benefit,
                model_points.annuity_payment,
                model_points.disability_income,
                model_points.disability_benefit,
                assumptions.expense_acquisition,
                maint_inflated_monthly,
                discount_start,
                discount_mid,
                mortality_factor,
                morbidity_factor,
                longevity_factor,
                disability_factor,
                model_points.cov_waiting,
                model_points.cov_reduction_end,
                model_points.cov_reduction_factor,
            )
        else:
            # Markov path -- every multi-state model with no duration
            # tracking. The closure factory and the hand-unrolled
            # n_states=2 / n_states=3 kernels stay in the file as a
            # readable reference but are no longer on the default path.
            kernel = _get_value_kernel_codegen(
                n_states, edge_from, edge_to, edge_lump_sum,
                premium_state, benefit_state,
            )
            bel, ra, csm, loss_component = kernel(
                *common_args, model_points.cov_waiting,
                model_points.cov_reduction_end,
                model_points.cov_reduction_factor,
            )
    elif backend == "gpu":
        if state_duration_max is not None:
            raise NotImplementedError(
                "value(backend='gpu') does not support semi-Markov "
                "StateModels yet; use backend='cpu'"
            )
        if np.any(model_points.cov_waiting) or np.any(model_points.cov_reduction_end):
            raise ValueError(
                "value(backend='gpu') does not support coverage waiting / "
                "reduction periods yet; use backend='cpu'"
            )
        from fastcashflow._gpu import value_gpu
        bel, ra, csm, loss_component = value_gpu(
            common_args[0], common_args[1], common_args[2], common_args[3],
            n_states, *common_args[4:],
        )
    else:
        raise ValueError(f"backend must be 'cpu' or 'gpu', got {backend!r}")

    return Valuation(bel=bel, ra=ra, csm=csm, loss_component=loss_component)


def value_segmented(
    model_points: ModelPoints,
    basis: dict[tuple[str, str], Assumptions],
    **kwargs,
) -> Valuation:
    """Value a multi-segment portfolio: split, value each, concatenate.

    ``basis`` is the ``{(product, channel): Assumptions}`` dictionary
    returned by :func:`fastcashflow.read_assumptions`. ``model_points``
    must carry ``product`` and ``channel`` columns identifying each row's
    segment; for each unique (product, channel) the helper masks the
    matching rows, builds a sub-:class:`~fastcashflow.ModelPoints` via
    :meth:`~fastcashflow.ModelPoints.subset`, calls :func:`value` with the
    segment's ``Assumptions``, and writes the per-row results back to a
    single ``(n_mp,)`` :class:`Valuation`.

    Extra keyword arguments (``backend``, ``discount_curve``) flow through
    to :func:`value`. A single-segment ``basis`` is accepted as a
    convenience when ``product`` / ``channel`` is not set.
    """
    if model_points.product is None or model_points.channel is None:
        if len(basis) == 1:
            (assumptions,) = basis.values()
            return value(model_points, assumptions, **kwargs)
        raise ValueError(
            "model_points has no 'product'/'channel' set but the basis has "
            f"{len(basis)} segments; either set the columns or pass a "
            "single-segment basis"
        )

    product = model_points.product
    channel = model_points.channel
    n_mp = model_points.n_mp

    bel = np.empty(n_mp)
    ra = np.empty(n_mp)
    csm = np.empty(n_mp)
    loss_component = np.empty(n_mp)

    # Run each unique (product, channel) once -- order is stable on the
    # first-seen index, so debugging output reads top-to-bottom of the input.
    seen: set[tuple] = set()
    keys: list[tuple] = []
    for p, c in zip(product, channel):
        key = (str(p), str(c))
        if key not in seen:
            seen.add(key)
            keys.append(key)

    for key in keys:
        if key not in basis:
            raise ValueError(
                f"segment {key!r} appears in model_points but is not in the "
                f"basis (known segments: {sorted(basis)})"
            )
        mask = np.fromiter(
            ((str(p), str(c)) == key for p, c in zip(product, channel)),
            dtype=bool, count=n_mp,
        )
        idx = np.nonzero(mask)[0]
        sub = model_points.subset(idx)
        val = value(sub, basis[key], **kwargs)
        bel[idx] = val.bel
        ra[idx] = val.ra
        csm[idx] = val.csm
        loss_component[idx] = val.loss_component

    return Valuation(bel=bel, ra=ra, csm=csm, loss_component=loss_component)
