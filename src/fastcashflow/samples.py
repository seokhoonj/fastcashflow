"""Bundled synthetic sample data -- ``fcf.samples.*``.

The single surface for the packaged demo data. Two uses, deliberately distinct:

* **load** -- get an assembled object to play with or feed straight to a
  measurement: :func:`basis`, :func:`model_points`, :func:`calculation_methods`,
  :func:`inforce_state`, :func:`return_scenarios` (toy fund returns for the VFA
  time-value-of-guarantees example) and :func:`rate_scenarios` (toy discount
  rates for the stochastic GMM valuation).
* **export** -- write a starter set of input *template files* to a directory
  (edit them, then read back with ``fcf.read_model_points`` / ``fcf.read_basis``).
  :func:`templates` lists the available template names.

``template="gmm"`` (default) is the protection portfolio; ``template="vfa"``
is the variable (account-value) contract set. The data is synthetic
(calibrated demo figures), never sourced from real portfolios.
"""
from pathlib import Path

import numpy as np

from fastcashflow import io as _io

#: Available sample templates -- see :func:`templates`.
_TEMPLATES = ("gmm", "vfa", "paa", "ul", "ul-annuity", "ul-cost-deduct",
              "ul-var-annuity", "annuity")

#: Fixed seed for :func:`scenarios` -- a reproducible toy path set, not a
#: calibration parameter.
_SCENARIO_SEED = 20260605

#: ``format=`` choices for :func:`export` -> the data-file extension. The
#: basis is always a multi-sheet ``.xlsx`` workbook regardless of this.
_FORMATS = {"csv": ".csv", "parquet": ".parquet",
            "feather": ".feather", "xlsx": ".xlsx"}


def templates() -> list[str]:
    """The available :func:`export` / load template names
    (``["gmm", "vfa", "paa", "ul", "ul-annuity", "ul-cost-deduct", "ul-var-annuity"]``)."""
    return list(_TEMPLATES)


def _ul_model_points():
    """A small synthetic universal-life portfolio.

    Three account-backed contracts that differ in face, account value at issue
    and premium. The face is carried on ``minimum_death_benefit`` and a DEATH
    coverage is registered (the account-backed death leg reads the account
    balance, topping up to the face); a uniform 2% ``minimum_crediting_rate``
    floor sits under the account. Pair with :func:`_ul_basis`; measure through
    ``gmm.measure`` (locked-in discount) or ``vfa.measure`` (underlying-items
    return). Synthetic demo figures, never sourced from a real portfolio.
    """
    from fastcashflow import CalculationMethod, ModelPoints

    face = np.array([100_000_000.0, 50_000_000.0, 80_000_000.0])
    return ModelPoints(
        sex=np.array([0, 1, 0]),
        issue_age=np.array([40.0, 50.0, 45.0]),
        term_months=np.array([120, 120, 60]),
        premium_term_months=np.array([120, 120, 60]),
        premium=np.array([500_000.0, 300_000.0, 600_000.0]),
        count=np.array([1.0, 1.0, 1.0]),
        account_value=np.array([0.0, 1_000_000.0, 0.0]),
        minimum_death_benefit=face,
        minimum_crediting_rate=np.array([0.02, 0.02, 0.02]),
        benefits={"DEATH": face},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
        product=np.array(["UL_A", "UL_A", "UL_A"]),
        channel=np.array(["FC", "FC", "FC"]),
    )


def _ul_basis():
    """The synthetic universal-life basis paired with :func:`_ul_model_points`.

    A flat COI (``coi_annual``) above the mortality, a 6% premium load, and a
    4% ``investment_return`` the account credits at. The DEATH coverage carries
    the account-chassis flags (``funds_from_account=True``,
    ``pays_account_balance=True``) so the shared projection routes its death
    leg through the account roll. A single :class:`~fastcashflow.Basis` (no
    per-segment router) -- both ``gmm.measure`` and ``vfa.measure`` take it.
    """
    from fastcashflow import Basis, CoverageRate

    coi = 0.0025
    return Basis(
        mortality_annual=0.002,
        lapse_annual=0.04,
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.1,
        investment_return=0.04,
        premium_load=0.06,
        coi_annual=coi,
        coverages=(CoverageRate("DEATH", coi, funds_from_account=True,
                                pays_account_balance=True),),
    )


def _ul_annuity_model_points():
    """A small synthetic universal-life *annuity* portfolio (2-phase).

    Two account-backed contracts that accumulate (phase 1, the ordinary UL roll)
    and then convert the balance to a guaranteed survival-contingent income
    (phase 2) at ``annuitization_months``: ``locked_annuity_payment =
    max(account, GMAB) * annuitization_rate``, paid annuity-due on the surviving
    in-force, with no further premium / COI / surrender and no maturity lump.
    Contract 0 is regular-premium (pays through the 15-year accumulation, then
    converts); contract 1 is single-premium (a seeded account, converts at year
    10). ``minimum_accumulation_benefit`` is the GMAB floor the conversion takes.
    Pair with :func:`_ul_annuity_basis`; measure through ``gmm.measure``.
    Synthetic demo figures, never sourced from a real portfolio.
    """
    from fastcashflow import CalculationMethod, ModelPoints

    face = np.array([50_000_000.0, 30_000_000.0])
    return ModelPoints(
        sex=np.array([0, 1]),
        issue_age=np.array([50.0, 55.0]),
        term_months=np.array([360, 300]),
        premium_term_months=np.array([180, 0]),       # <= annuitization_months
        premium=np.array([400_000.0, 0.0]),           # contract 1 = single-premium
        count=np.array([1.0, 1.0]),
        account_value=np.array([0.0, 30_000_000.0]),
        minimum_death_benefit=face,
        minimum_accumulation_benefit=np.array([40_000_000.0, 30_000_000.0]),
        minimum_crediting_rate=np.array([0.02, 0.02]),
        annuitization_months=np.array([180, 120]),    # convert at year 15 / 10
        annuitization_rate=np.array([0.004, 0.0045]), # monthly GAO rate
        benefits={"DEATH": face},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
        product=np.array(["UL_ANN", "UL_ANN"]),
        channel=np.array(["FC", "FC"]),
    )


def _ul_annuity_basis():
    """The synthetic universal-life-annuity basis paired with
    :func:`_ul_annuity_model_points`.

    The same account chassis as :func:`_ul_basis` (a flat COI, a premium load,
    an ``investment_return`` the account credits at) plus a ``longevity_cv`` --
    the payout phase is a survival-contingent income, so its risk adjustment is
    driven by longevity (annuitants living longer), not mortality. A single
    :class:`~fastcashflow.Basis`; measure it through ``gmm.measure``.
    """
    from fastcashflow import Basis, CoverageRate

    coi = 0.0025
    return Basis(
        mortality_annual=0.005,
        lapse_annual=0.03,
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.1,
        longevity_cv=0.15,
        investment_return=0.035,
        premium_load=0.05,
        coi_annual=coi,
        coverages=(CoverageRate("DEATH", coi, funds_from_account=True,
                                pays_account_balance=True),),
    )


def _ul_cost_deduct_model_points():
    """A small synthetic universal-life portfolio carrying a cost-deducting rider.

    Two account-backed contracts whose account funds BOTH the death-leg COI and a
    recurring-cancer rider (a fixed health benefit). The rider
    is declared ``funds_from_account=True, pays_account_balance=False`` on its
    :class:`~fastcashflow.basis.CoverageRate`: its monthly charge (``rate x
    amount``) is drawn from the account, but its benefit is the fixed CANCER sum,
    paid as a recurring morbidity claim -- never the account balance. Pair with
    :func:`_ul_cost_deduct_basis`; measure through ``gmm.measure``. Synthetic demo
    figures, never sourced from a real portfolio.
    """
    from fastcashflow import CalculationMethod, ModelPoints

    face = np.array([100_000_000.0, 50_000_000.0])
    cancer = np.array([30_000_000.0, 20_000_000.0])
    return ModelPoints(
        sex=np.array([0, 1]),
        issue_age=np.array([45.0, 50.0]),
        term_months=np.array([240, 240]),
        premium_term_months=np.array([240, 240]),
        premium=np.array([600_000.0, 400_000.0]),
        count=np.array([1.0, 1.0]),
        account_value=np.array([0.0, 1_000_000.0]),
        minimum_death_benefit=face,
        minimum_crediting_rate=np.array([0.02, 0.02]),
        benefits={"DEATH": face, "CANCER": cancer},
        calculation_methods={"DEATH": CalculationMethod.DEATH,
                             "CANCER": CalculationMethod.MORBIDITY},
        product=np.array(["UL_CD", "UL_CD"]),
        channel=np.array(["FC", "FC"]),
    )


def _ul_cost_deduct_basis():
    """The synthetic universal-life basis paired with
    :func:`_ul_cost_deduct_model_points`.

    The same account chassis as :func:`_ul_basis` plus a CANCER rider that funds
    its charge from the account (``funds_from_account=True``) but pays a fixed
    benefit (``pays_account_balance=False``). ``morbidity_cv`` prices the rider's
    health-benefit risk in the account-book risk adjustment. A single
    :class:`~fastcashflow.Basis`; measure through ``gmm.measure``.
    """
    from fastcashflow import Basis, CoverageRate

    coi = 0.004
    cancer_rate = 0.0024
    return Basis(
        mortality_annual=0.002,
        lapse_annual=0.04,
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.1,
        morbidity_cv=0.15,
        investment_return=0.04,
        premium_load=0.10,
        coi_annual=coi,
        coverages=(
            CoverageRate("DEATH", coi, funds_from_account=True,
                         pays_account_balance=True),
            CoverageRate("CANCER", cancer_rate, funds_from_account=True,
                         pays_account_balance=False),
        ),
    )


def _ul_var_annuity_model_points():
    """A small synthetic universal-life *variable-payout* annuity portfolio.

    Two account-backed contracts that accumulate then annuitize. Contract 0 takes
    a VARIABLE payout: a finite ``annuity_air_annual`` (the assumed
    interest rate, AIR) re-floats the phase-2 income each month by
    ``(1+fund)/(1+air)`` -- the annuity-unit method. Contract 1 keeps a FIXED GAO
    payout (``annuity_air_annual`` NaN). A variable payout is a direct-
    participation feature, so measure the book through ``vfa.measure``
    (``gmm.measure`` rejects a finite AIR). Synthetic demo figures, never sourced
    from a real portfolio.
    """
    from fastcashflow import CalculationMethod, ModelPoints

    face = np.array([50_000_000.0, 30_000_000.0])
    return ModelPoints(
        sex=np.array([0, 1]),
        issue_age=np.array([50.0, 55.0]),
        term_months=np.array([360, 300]),
        premium_term_months=np.array([180, 0]),
        premium=np.array([400_000.0, 0.0]),           # contract 1 = single-premium
        count=np.array([1.0, 1.0]),
        account_value=np.array([0.0, 30_000_000.0]),
        minimum_death_benefit=face,
        minimum_accumulation_benefit=np.array([40_000_000.0, 30_000_000.0]),
        minimum_crediting_rate=np.array([0.0, 0.0]),
        annuitization_months=np.array([180, 120]),    # convert at year 15 / 10
        annuitization_rate=np.array([0.004, 0.0045]), # initial monthly income rate
        annuity_air_annual=np.array([0.02, np.nan]),  # 0 = variable@2% AIR, 1 = fixed
        benefits={"DEATH": face},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
        product=np.array(["UL_VAR", "UL_VAR"]),
        channel=np.array(["FC", "FC"]),
    )


def _ul_var_annuity_basis():
    """The synthetic universal-life variable-payout-annuity basis paired with
    :func:`_ul_var_annuity_model_points`.

    The same account chassis as :func:`_ul_annuity_basis` plus a ``longevity_cv``
    (the payout bears longevity risk). Measure through ``vfa.measure``: the
    account-roll discount equals the ``investment_return``, so the fund cancels
    out of the variable payout and the BEL reduces to the AIR-reserve.
    """
    from fastcashflow import Basis, CoverageRate

    coi = 0.0025
    return Basis(
        mortality_annual=0.005,
        lapse_annual=0.03,
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.1,
        longevity_cv=0.15,
        investment_return=0.035,
        premium_load=0.05,
        coi_annual=coi,
        coverages=(CoverageRate("DEATH", coi, funds_from_account=True,
                                pays_account_balance=True),),
    )


def _annuity_model_points():
    """A small synthetic standalone (non-account) deferred-annuity portfolio.

    Two traditional annuity contracts that accumulate the reserve through a
    premium-paying deferral window, then pay a survival income on the new payout
    schedule (these forms route to the full projection kernel):

    * Contract 0 -- a deferred GUARANTEED-PERIOD life annuity: a 10-year deferral
      (``annuity_start_months=120``), then a life annuity whose first 20 years
      (``annuity_guarantee_months=240``) are paid regardless of survival.
    * Contract 1 -- a deferred TERM-CERTAIN annuity: a 5-year deferral, then a
      20-year (``annuity_term_months=240``) certain payout.

    Pair with :func:`_annuity_basis`; measure through ``gmm.measure``. Synthetic
    demo figures, never sourced from a real portfolio.
    """
    from fastcashflow import CalculationMethod, ModelPoints

    death = np.array([10_000_000.0, 10_000_000.0])
    return ModelPoints(
        sex=np.array([0, 1]),
        issue_age=np.array([50.0, 55.0]),
        term_months=np.array([600, 360]),
        premium_term_months=np.array([120, 60]),       # premium only in deferral
        premium=np.array([500_000.0, 300_000.0]),
        count=np.array([1.0, 1.0]),
        annuity_payment=np.array([250_000.0, 100_000.0]),
        annuity_start_months=np.array([120, 60]),      # deferral: income starts here
        annuity_guarantee_months=np.array([240, 0]),   # contract 0: 20y guaranteed
        annuity_term_months=np.array([0, 240]),        # contract 1: 20y term-certain
        benefits={"DEATH": death},
        calculation_methods={"DEATH": CalculationMethod.DEATH},
        product=np.array(["ANNUITY", "ANNUITY"]),
        channel=np.array(["FC", "FC"]),
    )


def _annuity_basis():
    """The synthetic standalone deferred-annuity basis paired with
    :func:`_annuity_model_points`.

    The payout is a survival-contingent income, so its risk adjustment is driven
    by ``longevity_cv`` (annuitants living longer); ``mortality_cv`` prices the
    death benefit during the deferral. A single :class:`~fastcashflow.Basis`;
    measure it through ``gmm.measure``.
    """
    from fastcashflow import Basis, CoverageRate

    q = 0.008
    return Basis(
        mortality_annual=q,
        lapse_annual=0.02,
        discount_annual=0.03,
        ra_confidence=0.75,
        mortality_cv=0.10,
        longevity_cv=0.15,
        coverages=(CoverageRate("DEATH", q),),
    )


def basis(template: str = "gmm"):
    """Bundled sample basis. ``template="gmm"`` (default) returns the per-segment
    :class:`~fastcashflow.BasisRouter` (a ``(product, channel)`` -> ``Basis``
    mapping); ``template="vfa"`` returns the single variable-contract
    :class:`~fastcashflow.Basis`; ``template="ul"`` returns the single
    account-backed universal-life :class:`~fastcashflow.Basis`;
    ``template="ul-annuity"`` returns the universal-life-annuity (2-phase
    accumulation -> income) :class:`~fastcashflow.Basis`."""
    if template == "vfa":
        return _io.load_sample_vfa_basis()
    if template == "paa":
        return _io.load_sample_paa_basis()
    if template == "ul":
        return _ul_basis()
    if template == "ul-annuity":
        return _ul_annuity_basis()
    if template == "ul-cost-deduct":
        return _ul_cost_deduct_basis()
    if template == "ul-var-annuity":
        return _ul_var_annuity_basis()
    if template == "annuity":
        return _annuity_basis()
    if template == "gmm":
        return _io.load_sample_basis()
    raise ValueError(f"template must be one of {_TEMPLATES}, got {template!r}")


def model_points(template: str = "gmm"):
    """Bundled sample model points (``template="gmm"`` default, ``"vfa"`` for the
    variable account-value contracts, ``"ul"`` for the account-backed
    universal-life contracts, ``"ul-annuity"`` for the universal-life-annuity
    2-phase accumulation -> income contracts)."""
    if template == "vfa":
        return _io.load_sample_vfa_model_points()
    if template == "paa":
        return _io.load_sample_paa_model_points()
    if template == "ul":
        return _ul_model_points()
    if template == "ul-annuity":
        return _ul_annuity_model_points()
    if template == "ul-cost-deduct":
        return _ul_cost_deduct_model_points()
    if template == "ul-var-annuity":
        return _ul_var_annuity_model_points()
    if template == "annuity":
        return _annuity_model_points()
    if template == "gmm":
        return _io.load_sample_model_points()
    raise ValueError(f"template must be one of {_TEMPLATES}, got {template!r}")


def calculation_methods():
    """Bundled sample coverage-code -> calculation-method taxonomy."""
    return _io.load_sample_calculation_methods()


def treaty(cession: float = 0.30):
    """Bundled sample reinsurance treaty -- a quota share ceding ``cession`` of
    the direct book (default 30%).

    A treaty is a parameter object, not a data file, so this is the one
    reinsurance-specific sample object: the underlying ceded contracts are the
    same :func:`model_points` / :func:`basis` portfolio. Pass it to
    :func:`~fastcashflow.reinsurance.measure` or
    :func:`~fastcashflow.reinsurance.settle` over a segment of the sample book
    (reinsurance is measured on a single :class:`~fastcashflow.Basis`)."""
    from fastcashflow.reinsurance import QuotaShare
    return QuotaShare(cession=cession)


def inforce_state():
    """Bundled sample in-force state (elapsed_months / count / prior_csm / ...)."""
    return _io.load_sample_inforce_state()


def return_scenarios(template: str = "vfa", n_scenarios: int = 1000):
    """Toy *fund-return* scenarios, shape ``(n_scenarios, n_time)``, for the
    variable (VFA) time-value-of-guarantees example -- the ``return_scenarios``
    input to :func:`~fastcashflow.vfa.measure`.

    Generated in memory (no bundled file): deterministic, modest-volatility
    monthly fund returns so the guarantee shows a believable time value (~3% of
    account value on the sample). This is NOT a calibrated economic scenario
    generator -- the engine *consumes* scenarios, it does not certify
    valuation-grade ones. For a real valuation supply your own set via
    :func:`~fastcashflow.read_scenarios`.

    Each cell is a one-month *fund return* (not an interest-rate path -- that is
    the separate ``scenarios`` input to :func:`~fastcashflow.gmm.stochastic`);
    ``n_time`` matches the bundled VFA sample's term and the fixed seed keeps the
    output stable.
    """
    if template != "vfa":
        raise ValueError(
            "return_scenarios are a variable-contract (VFA) input; template "
            f"must be 'vfa', got {template!r}"
        )
    mp = model_points("vfa")
    n_time = int(np.asarray(mp.term_months).max())
    rng = np.random.default_rng(_SCENARIO_SEED)
    central = (1.0 + 0.06) ** (1.0 / 12.0) - 1.0   # ~6% annual, monthly return
    vol = 0.005                                    # modest monthly sd -- a toy
    return central + vol * rng.standard_normal((n_scenarios, n_time))


def rate_scenarios(n_scenarios: int = 1000):
    """Toy *discount-rate* scenarios, shape ``(n_scenarios,)``, for the
    stochastic GMM valuation -- the ``scenarios`` input to
    :func:`~fastcashflow.gmm.stochastic`. The interest-rate counterpart to
    :func:`return_scenarios` (which is fund returns).

    Generated in memory: one flat annual discount rate per scenario, modest
    dispersion around ~3%, deterministic (fixed seed). This is NOT a calibrated
    economic scenario generator -- for a real valuation supply your own rate set
    (Hull-White / Vasicek / regulator-prescribed) via
    :func:`~fastcashflow.read_scenarios`. Flat (1-D) rates so the toy is
    portfolio-agnostic; a real run can pass a 2-D ``(n_scenarios, n_time)`` curve
    set instead.
    """
    rng = np.random.default_rng(_SCENARIO_SEED + 1)   # a stream distinct from returns
    rates = 0.03 + 0.01 * rng.standard_normal(n_scenarios)
    return np.maximum(rates, 1e-4)                     # keep the discount rate positive


def _export_tree(dest: Path, files: list[str]) -> str:
    """An ASCII tree of the files :func:`export` wrote, expanding the
    ``basis.xlsx`` workbook into its sheets -- a one-glance map of what landed
    in the directory and which assumption sheets the basis carries."""
    import openpyxl
    lines = [f"{dest}/"]
    for i, name in enumerate(files):
        last_file = i == len(files) - 1
        lines.append(f"{'`-- ' if last_file else '+-- '}{name}")
        if name.endswith(".xlsx"):
            wb = openpyxl.load_workbook(dest / name, read_only=True)
            sheets = wb.sheetnames
            wb.close()
            pad = "    " if last_file else "|   "
            for j, sheet in enumerate(sheets):
                last_sheet = j == len(sheets) - 1
                lines.append(f"{pad}{'`-- ' if last_sheet else '+-- '}{sheet}")
    return "\n".join(lines)


def export(output_dir, template: str = "gmm", format: str = "csv",
           *, quiet: bool = False) -> Path:
    """Write a starter set of input template files to ``output_dir``.

    ``template="gmm"`` writes ``basis.xlsx`` plus ``policies`` / ``coverages``
    / ``calculation_methods`` / ``inforce_state`` and the combined
    ``inforce_policies`` (the period-close one-file form); ``template="vfa"``
    writes the variable-contract ``basis.xlsx`` and ``policies``. Edit them and
    read back with :func:`~fastcashflow.read_model_points` /
    :func:`~fastcashflow.read_basis`.

    ``format`` picks the data-file extension -- ``"csv"`` (default),
    ``"parquet"``, ``"feather"`` or ``"xlsx"``. The basis is always a
    multi-sheet ``.xlsx`` workbook (it cannot be a flat table), so ``format``
    applies only to the policies / coverages / state files. Use ``"parquet"``
    for a portfolio large enough to stream with
    :func:`~fastcashflow.gmm.measure_stream`.

    Prints a tree of the files written -- expanding ``basis.xlsx`` into its
    sheets -- so it is clear what landed where. Pass ``quiet=True`` to suppress
    (e.g. in scripts). Returns the destination directory.
    """
    if template not in _TEMPLATES:
        raise ValueError(f"template must be one of {_TEMPLATES}, got {template!r}")
    if format not in _FORMATS:
        raise ValueError(
            f"format must be one of {tuple(_FORMATS)}, got {format!r}")
    ext = _FORMATS[format]
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if template == "gmm":
        _io._save_sample_basis(dest / "basis.xlsx")
        _io._save_sample_policies(dest / f"policies{ext}")
        _io._save_sample_coverages(dest / f"coverages{ext}")
        _io._save_sample_calculation_methods(dest / f"calculation_methods{ext}")
        _io._save_sample_inforce_state(dest / f"inforce_state{ext}")
        _io._save_sample_inforce_policies(dest / f"inforce_policies{ext}")
        files = ["basis.xlsx", f"policies{ext}", f"coverages{ext}",
                 f"calculation_methods{ext}", f"inforce_state{ext}",
                 f"inforce_policies{ext}"]
    elif template == "vfa":
        _io._drop_sample_table("sample_vfa_basis.xlsx", dest / "basis.xlsx")
        _io._drop_sample_table("sample_vfa_policies.csv", dest / f"policies{ext}")
        files = ["basis.xlsx", f"policies{ext}"]
    elif template == "paa":
        _io._drop_sample_table("sample_paa_basis.xlsx", dest / "basis.xlsx")
        _io._drop_sample_table("sample_paa_policies.csv", dest / f"policies{ext}")
        _io._drop_sample_table("sample_paa_coverages.csv", dest / f"coverages{ext}")
        files = ["basis.xlsx", f"policies{ext}", f"coverages{ext}"]
    else:  # ul / ul-annuity / ul-cost-deduct / ul-var-annuity -- load-only
        raise NotImplementedError(
            f"the {template!r} template is load-only -- build it in memory with "
            f"samples.model_points({template!r}) / samples.basis({template!r}); "
            "it has no exportable starter files")
    if not quiet:
        print(f"fastcashflow sample export -- template={template!r}, "
              f"{len(files)} files")
        print(_export_tree(dest, files))
    return dest


__all__ = ["templates", "basis", "model_points", "calculation_methods",
           "treaty", "inforce_state", "return_scenarios", "rate_scenarios",
           "export"]
