"""Shared test helpers -- single-DEATH boilerplate.

Most engine tests value a tiny term-life contract whose only claim is a flat
death benefit, so the same scaffold (the patterns dict, the monthly->annual
conversion, the Basis builder that wires mortality_annual into both the
in-force decrement and the DEATH coverage's rate) was duplicated across ~20
files. It is hoisted here. Tests that need multiple coverages or custom
patterns still keep a local builder.

A subtle point this helper enforces: with the engine slot 0 hardwire gone,
the DEATH coverage's rate is a separate field from ``mortality_annual``. If a
test overrides only one of them the two go silently out of sync. The builder
takes a single rate (scalar ``mortality_q`` or callable ``mortality_annual``)
and wires it into both, so the override path is the safe one by construction.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from fastcashflow import Basis, CalculationMethod, CoverageRate
from fastcashflow.modelpoints import STATE_LABELS


PATTERNS = {"DEATH": CalculationMethod.DEATH}


def annual_from_monthly(m: float) -> float:
    """Annual-equivalent of a flat monthly rate (engine converts back)."""
    return 1.0 - (1.0 - m) ** 12


def _flat_age_rate(monthly_q: float):
    """Per-policy flat rate keyed off ``issue_age.shape``."""
    annual = annual_from_monthly(monthly_q)
    return lambda sex, issue_age, duration: np.full(issue_age.shape, annual)


def _flat_dur_rate(monthly_q: float):
    """Per-policy flat rate keyed off ``duration.shape``."""
    annual = annual_from_monthly(monthly_q)
    return lambda sex, issue_age, duration: np.full(duration.shape, annual)


def make_death_assumptions(
    *,
    mortality_q: float | None = None,
    lapse_q: float | None = None,
    mortality_annual=None,
    lapse_annual=None,
    discount_annual: float = 0.0,
    ra_confidence: float = 0.75,
    mortality_cv: float = 0.10,
    expense_items: tuple = (),
    expense_inflation: float = 0.0,
    coverages=None,
    **other,
) -> Basis:
    """Build an Basis for a single-DEATH-coverage hand-calc test.

    Pass either ``mortality_q`` / ``lapse_q`` (flat monthly rates) or
    ``mortality_annual`` / ``lapse_annual`` (full callables). The DEATH
    coverage's rate is wired from the same callable as ``mortality_annual``;
    pass ``coverages=...`` to override that auto-wire (multi-coverage cases).
    Any extra keyword (waiver_incidence_annual, state_model, fund_fee,
    investment_return, ...) is forwarded to Basis.
    """
    if mortality_annual is None:
        if mortality_q is None:
            raise TypeError("make_death_assumptions: pass mortality_q or mortality_annual")
        mortality_annual = _flat_age_rate(mortality_q)
    if lapse_annual is None:
        lapse_annual = _flat_dur_rate(0.0 if lapse_q is None else lapse_q)
    if coverages is None:
        coverages = (CoverageRate("DEATH", mortality_annual),)
    return Basis(
        mortality_annual    = mortality_annual,
        lapse_annual        = lapse_annual,
        discount_annual     = discount_annual,
        ra_confidence       = ra_confidence,
        mortality_cv        = mortality_cv,
        expense_items       = expense_items,
        expense_inflation   = expense_inflation,
        coverages           = coverages,
        **other,
    )


# ---------------------------------------------------------------------------
# Frame renderers -- TEST-ONLY scaffold to exercise read_model_points.
#
# The engine ships no ModelPoints -> file exporter: an in-memory frame is a
# lossy projection (it cannot carry coverage waiting / reduction rules,
# issue_class, elapsed_months, or the VFA account fields), and a silently-lossy
# public export is a footgun. Only the reader (read_model_points)
# accept these as external input. These helpers render a *simple* test
# portfolio into those frames so the reader / file-format tests have input;
# they are valid only because the test portfolios carry none of the dropped
# fields.
# ---------------------------------------------------------------------------

def _coverage_label(mp, ctype, default):
    registry = mp.calculation_methods or {}
    for code, t in registry.items():
        if t == ctype:
            return code
    return default


def mp_to_frames(mp, basis):
    """Render a ModelPoints as a (policies, coverages) frame pair."""
    policies = pl.DataFrame({
        "mp_id":                    np.arange(mp.n_mp),
        "issue_age":                mp.issue_age,
        "sex":                      mp.sex,
        "term_months":              mp.term_months,
        "premium":                  mp.premium,
        "premium_term_months":      mp.premium_term_months,
        "premium_frequency_months": mp.premium_frequency_months,
        "annuity_frequency_months": mp.annuity_frequency_months,
        "disability_income":        mp.disability_income,
        "disability_benefit":       mp.disability_benefit,
        "count":                    mp.count,
        "state":                    np.array([STATE_LABELS[int(s)] for s in mp.state]),
    })
    label = {i: coverage.code for i, coverage in enumerate(basis.coverages)}
    mp_of_cov = np.repeat(np.arange(mp.n_mp), np.diff(mp.coverage_offset))
    mp_id = [int(m) for m in mp_of_cov]
    coverage_code = [label[int(k)] for k in mp.coverage_index]
    amount = [float(a) for a in mp.coverage_amount]
    for ctype, scalar in ((CalculationMethod.ANNUITY, mp.annuity_payment),
                          (CalculationMethod.MATURITY, mp.maturity_benefit)):
        code = _coverage_label(mp, ctype, str(ctype))
        for i in np.nonzero(scalar)[0]:
            mp_id.append(int(i))
            coverage_code.append(code)
            amount.append(float(scalar[i]))
    coverages = pl.DataFrame({
        "mp_id": mp_id, "coverage_code": coverage_code, "amount": amount,
    })
    return policies, coverages
