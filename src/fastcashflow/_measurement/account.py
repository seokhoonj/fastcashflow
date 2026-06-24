"""Universal-life account-chassis primitives shared across the measurement models.

:func:`_portfolio_has_account` detects an account-backed book from the
per-coverage flags; :func:`_account_roll_inputs` factors the account-roll inputs
a stochastic guarantee time-value pass re-rolls under return scenarios. Both are
model-neutral (the account roll is identical under GMM and VFA), so they live in
the shared measurement layer rather than the GMM engine.
"""
from __future__ import annotations

import numpy as np

from fastcashflow.basis import Basis, annual_to_monthly
from fastcashflow.coverage import (
    align_coverages, build_coverage_rates, coverage_arrays,
)
from fastcashflow.model_points import ModelPoints


def _portfolio_has_account(model_points: ModelPoints, basis: Basis) -> bool:
    """True when any coverage carries a universal-life account-chassis flag.

    Derived STRICTLY from the per-coverage ``funds_from_account`` /
    ``pays_account_balance`` flags read off the :class:`CoverageRate` objects
    (never ``account_value != 0``, which would wrongly flip the variable-annuity
    product onto the recursive roll). Deliberately does NOT resolve calculation
    methods -- the flags are independent of the method.
    """
    return any(getattr(r, "funds_from_account", False)
               or getattr(r, "pays_account_balance", False)
               for r in basis.coverages)


def _account_roll_inputs(model_points: ModelPoints, basis: Basis):
    """Per-policy universal-life account-roll inputs for a stochastic TVOG pass.

    Factors the same chain the measure path runs inline (coverage-rate grid ->
    expense gamma -> :func:`projection._account_kernel_args`) so a guarantee
    time-value pass can re-roll the account under return scenarios. Returns
    ``(account_value0, face, prem_to_av, coi_rate_m, admin_fee, account_charge,
    gmab, minimum_crediting_rate, surr_charge_rate)`` -- everything the scenario
    roll needs beyond the credit (which the scenario supplies).
    """
    from fastcashflow.projection import _expense_kernel_args, _account_kernel_args
    n_time = int(model_points.contract_boundary_months.max())
    n_years = (n_time + 11) // 12
    min_age = int(model_points.issue_age.min())
    max_age = int(model_points.issue_age.max())
    sex_grid, issue_age_grid, duration_grid = np.meshgrid(
        np.array([0, 1]), np.arange(min_age, max_age + 1), np.arange(n_years),
        indexing="ij")
    issue_class_grid = np.zeros_like(duration_grid)
    elapsed_grid = np.zeros_like(duration_grid)
    aligned = align_coverages(basis.coverages, model_points.coverage_codes)
    _cid, _crisk, cov_funds, cov_pays = coverage_arrays(
        aligned, model_points.calculation_methods)
    coverage_rates = np.ascontiguousarray(annual_to_monthly(build_coverage_rates(
        [r.rate for r in aligned], sex_grid, issue_age_grid, duration_grid,
        issue_class_grid, elapsed_grid, codes=[r.code for r in aligned])))
    issue_index = np.asarray(model_points.issue_age, np.int64) - min_age
    coverage_rates_per_mp = np.ascontiguousarray(           # (cov, mp, year)
        coverage_rates[:, np.asarray(model_points.sex, np.int64), issue_index, :])
    _a, _b, _c, gamma_fixed, _lae = _expense_kernel_args(basis, n_time)
    (_has, _mp_acc, account_value0, face, prem_to_av, coi_rate_m, admin_fee,
     _credit, account_charge, surr_charge_rate) = _account_kernel_args(
        model_points, basis, coverage_rates_per_mp, cov_funds, cov_pays,
        gamma_fixed, n_time, n_years)
    gmab = np.asarray(model_points.maturity_benefit, dtype=np.float64)
    return (account_value0, face, prem_to_av, coi_rate_m, admin_fee,
            account_charge, gmab,
            np.asarray(model_points.minimum_crediting_rate, dtype=np.float64),
            surr_charge_rate)
