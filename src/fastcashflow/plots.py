"""Visualisation -- charts of the IFRS 17 figures the engine produces.

Turn a measurement, a reconciliation or a stochastic result into a chart.
The measurement and reconciliation charts dispatch on the result type --
GMM, PAA, VFA and reinsurance held each draw their own model's quantities
(a PAA result has an LRC and an LIC, not a BEL / RA / CSM split). Every
function draws onto a matplotlib Axes -- it creates one if none is given,
and returns it -- so the charts compose into larger figures and stay easy
to save or restyle.
"""
from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np

from fastcashflow.engine import _require_full
from fastcashflow._measurement_basis import _require_inception
from fastcashflow.movement import (
    PAAReconciliation,
    Reconciliation,
    ReinsuranceReconciliation,
    VFAReconciliation,
)
from fastcashflow.numerics import _norm_ppf
from fastcashflow._paa import _require_full_paa
from fastcashflow._vfa import _require_settlement_csm
import fastcashflow._gmm as _gmm
import fastcashflow._paa as _paa
import fastcashflow._vfa as _vfa
import fastcashflow._reinsurance as _reinsurance

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from fastcashflow.basis import Basis
    from fastcashflow.stochastic import StochasticResult

__all__ = [
    "plot_liability",
    "plot_cashflows",
    "plot_csm_runoff",
    "plot_risk_adjustment",
    "plot_analysis_of_change",
    "plot_stochastic",
]

# fastcashflow chart palette -- one colour per IFRS 17 quantity, kept
# consistent across every chart.
_COLOR = {
    "bel": "#3b6ea5",       # blue
    "ra": "#e0a458",        # amber
    "csm": "#2a9d8f",       # teal-green
    "loss": "#c1466b",      # rose
    "ink": "#1d2b35",       # near-black -- text and axes
    "grid": "#e6e8eb",      # light grid
    "up": "#2a9d8f",        # waterfall increase
    "down": "#e07a5f",      # waterfall decrease
    "total": "#52677a",     # waterfall opening / closing bars
}


def _plt():
    """Import matplotlib lazily, with a helpful error if it is missing."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:                           # pragma: no cover
        raise ImportError(
            "matplotlib is missing -- reinstall with 'pip install fastcashflow'"
        ) from exc
    return plt


def _compact(value: float, _pos: object = None) -> str:
    """Format a single monetary value compactly -- 1.4M, 320K, -184K, ..."""
    a = abs(value)
    if a >= 1e9:
        return f"{value / 1e9:,.1f}B"
    if a >= 1e6:
        return f"{value / 1e6:,.1f}M"
    if a >= 1e3:
        return f"{value / 1e3:,.0f}K"
    return f"{value:,.0f}"


def _gaussian_kde(data, grid):
    """A Gaussian kernel density estimate -- numpy only, no SciPy.

    The bandwidth follows Silverman's rule of thumb.
    """
    n = data.size
    std = data.std(ddof=1)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    spread = min(std, iqr / 1.349) if iqr > 0.0 else std
    bandwidth = 0.9 * spread * n ** (-0.2)
    z = (grid[:, None] - data[None, :]) / bandwidth
    kernel = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return kernel.sum(axis=1) / (n * bandwidth)


def _format_money_axis(ax, axis: str) -> None:
    """Format a whole axis as money in one consistent unit (K, M or B)."""
    from matplotlib.ticker import FuncFormatter

    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    peak = max(abs(lo), abs(hi))
    div, suffix = 1.0, ""
    if peak >= 1e9:
        div, suffix = 1e9, "B"
    elif peak >= 1e6:
        div, suffix = 1e6, "M"
    elif peak >= 1e3:
        div, suffix = 1e3, "K"

    def fmt(value, _pos=None):
        if value == 0:
            return "0"
        text = f"{value / div:,.2f}".rstrip("0").rstrip(".")
        return f"{text}{suffix}"

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(FuncFormatter(fmt))


def _axes(ax, figsize: tuple[float, float] = (9.0, 5.5)):
    """Return ``ax``, or a fresh Axes if it is ``None``.

    A freshly created figure uses constrained layout so the left-aligned
    title, the axis labels and the legend get their own space instead of
    crowding the plot. When the caller supplies ``ax`` (composing into their
    own figure) layout is their responsibility.
    """
    if ax is not None:
        return ax
    _, ax = _plt().subplots(figsize=figsize, dpi=120, constrained_layout=True)
    return ax


def _finish(ax, title, *, xlabel=None, ylabel=None, money_axis="y", title_pad=12):
    """Apply the fastcashflow house style to ``ax``."""
    ink = _COLOR["ink"]
    ax.set_title(title, fontsize=13, fontweight="bold", color=ink,
                 loc="left", pad=title_pad)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=ink)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=ink)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_COLOR["grid"])
    ax.tick_params(colors=ink, labelsize=9, length=0)
    ax.grid(axis="y", color=_COLOR["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    if money_axis in ("x", "y"):
        _format_money_axis(ax, money_axis)
    return ax


def _legend(ax) -> None:
    ax.legend(frameon=False, fontsize=9, labelcolor=_COLOR["ink"])


def _reject(entry: str, expected: str, obj: object) -> TypeError:
    """Build the unsupported-type error for a dispatching chart."""
    name = type(obj).__name__
    hint = ""
    if name.startswith("Portfolio"):
        hint = (" -- a portfolio container holds one result per model; pass "
                "one model slot's native result instead (e.g. the .gmm slot)")
    return TypeError(f"{entry} expects {expected}, got {name}{hint}")


# ---------------------------------------------------------------------------
# Liability components over time
# ---------------------------------------------------------------------------
@singledispatch
def plot_liability(measurement, *, ax: Axes | None = None,
                   title: str = "Liability components over time") -> Axes:
    """Plot the liability components over the contract's life.

    Dispatches on the measurement type: a GMM, VFA or reinsurance-held
    measurement draws the BEL, RA and CSM trajectories; a PAA measurement
    draws the LRC and the LIC (its liability has no BEL / RA / CSM split --
    and the LRC line excludes the loss component, whose run-off
    :func:`plot_analysis_of_change` shows with
    ``component="loss_component"``). Each line is the portfolio total of
    that component at each month. Needs the trajectories, so measure with
    ``full=True``.
    """
    raise _reject("plot_liability()",
                  "a GMM, PAA, VFA or reinsurance measurement", measurement)


def _component_lines(series, ax, title):
    """Draw portfolio-total component trajectories as labelled lines.

    ``series`` is ``(label, colour key, (n_mp, n_time+1) path)`` triples.
    """
    ax = _axes(ax)
    months = np.arange(series[0][2].shape[1])
    ax.axhline(0.0, color=_COLOR["ink"], linewidth=0.8)
    for label, color, path in series:
        ax.plot(months, path.sum(axis=0), color=_COLOR[color], linewidth=2.2,
                label=label)
    ax.set_xlim(0, max(int(months[-1]), 1))
    _finish(ax, title, xlabel="month", ylabel="amount")
    _legend(ax)
    return ax


def _bel_ra_csm_lines(measurement, ax, title):
    _require_full(measurement, "plot_liability()")
    return _component_lines(
        (("BEL", "bel", measurement.bel_path),
         ("RA", "ra", measurement.ra_path),
         ("CSM", "csm", measurement.csm_path)), ax, title)


@plot_liability.register
def _(measurement: _gmm.Measurement, *, ax=None,
      title="Liability components over time"):
    _require_inception(measurement, "plot_liability()")
    return _bel_ra_csm_lines(measurement, ax, title)


@plot_liability.register
def _(measurement: _vfa.Measurement, *, ax=None,
      title="Liability components over time"):
    _require_settlement_csm(measurement, "plot_liability()")
    return _bel_ra_csm_lines(measurement, ax, title)


@plot_liability.register
def _(measurement: _reinsurance.Measurement, *, ax=None,
      title="Reinsurance-held components over time"):
    _require_inception(measurement, "plot_liability()")
    return _bel_ra_csm_lines(measurement, ax, title)


@plot_liability.register
def _(measurement: _paa.Measurement, *, ax=None,
      title="Liability components over time"):
    # The LRC trajectory excludes the loss component (the paragraph-100
    # split); the label says so, and plot_analysis_of_change shows the loss
    # component's own run-off (component="loss_component").
    _require_inception(measurement, "plot_liability()")
    _require_full_paa(measurement, "plot_liability()")
    return _component_lines(
        (("LRC (excl. loss component)", "bel", measurement.lrc_path),
         ("LIC", "ra", measurement.lic_path)), ax, title)


# ---------------------------------------------------------------------------
# Projected cash flows
# ---------------------------------------------------------------------------
@singledispatch
def plot_cashflows(measurement, *, period_months: int = 12,
                   ax: Axes | None = None,
                   title: str = "Projected cash flows") -> Axes:
    """Plot the projected money in against the money out.

    Dispatches on the measurement type. A GMM, PAA or VFA measurement draws
    premium income against claim and expense outgo; a reinsurance-held
    measurement draws the ceded streams -- recoveries in against reinsurance
    premiums out. The monthly cash flows are aggregated into buckets of
    ``period_months`` months -- a policy year by default. Money in is drawn
    upward, money out downward, and the marked line is the net cash flow
    each period. Bucketing keeps a front-loaded month from dominating the
    chart while the cash-flow shape stays visible.
    """
    raise _reject("plot_cashflows()",
                  "a GMM, PAA, VFA or reinsurance measurement", measurement)


def _cashflow_bars(income, outgo, in_label, out_label, period_months, ax,
                   title):
    """Bucket two opposing monthly streams and draw the in / out / net bars."""
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    ax = _axes(ax)
    starts = np.arange(0, income.shape[0], period_months)
    income_b = np.add.reduceat(income, starts)
    outgo_b = np.add.reduceat(outgo, starts)
    x = np.arange(income_b.shape[0])

    ax.bar(x, income_b, width=0.62, color=_COLOR["csm"],
           label=in_label, zorder=3)
    ax.bar(x, -outgo_b, width=0.62, color=_COLOR["down"],
           label=out_label, zorder=3)
    ax.plot(x, income_b - outgo_b, color=_COLOR["ink"], linewidth=1.6,
            marker="o", markersize=4, label="net", zorder=4)
    ax.axhline(0.0, color=_COLOR["ink"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x])
    _finish(ax, title,
            xlabel="policy year" if period_months == 12 else "period",
            ylabel="amount")
    _legend(ax)
    return ax


def _direct_cashflow_chart(measurement, period_months, ax, title):
    cf = measurement.cashflows
    premium = cf.premium_cf.sum(axis=0)
    # Every monthly insurer outflow, so the "net" line is honest: claims,
    # morbidity, annuity, expenses, disability income/lump and surrender value.
    # (maturity_cf is a per-policy lump at each policy's term, not a monthly
    # series, so it is not placed on this period-binned timeline.)
    outgo = (cf.mortality_cf + cf.morbidity_cf + cf.annuity_cf + cf.expense_cf
             + cf.disability_cf + cf.surrender_cf).sum(axis=0)
    return _cashflow_bars(premium, outgo, "premiums in",
                          "claims & expenses out", period_months, ax, title)


@plot_cashflows.register
def _(measurement: _gmm.Measurement, *, period_months=12, ax=None,
      title="Projected cash flows"):
    _require_inception(measurement, "plot_cashflows()")
    _require_full(measurement, "plot_cashflows()")
    return _direct_cashflow_chart(measurement, period_months, ax, title)


@plot_cashflows.register
def _(measurement: _vfa.Measurement, *, period_months=12, ax=None,
      title="Projected cash flows"):
    _require_settlement_csm(measurement, "plot_cashflows()")
    _require_full(measurement, "plot_cashflows()")
    return _direct_cashflow_chart(measurement, period_months, ax, title)


@plot_cashflows.register
def _(measurement: _paa.Measurement, *, period_months=12, ax=None,
      title="Projected cash flows"):
    _require_inception(measurement, "plot_cashflows()")
    _require_full_paa(measurement, "plot_cashflows()")
    return _direct_cashflow_chart(measurement, period_months, ax, title)


@plot_cashflows.register
def _(measurement: _reinsurance.Measurement, *, period_months=12, ax=None,
      title="Projected ceded cash flows"):
    _require_inception(measurement, "plot_cashflows()")
    _require_full(measurement, "plot_cashflows()")
    return _cashflow_bars(
        measurement.recovery.sum(axis=0),
        measurement.reinsurance_premium.sum(axis=0),
        "recoveries in", "reinsurance premiums out", period_months, ax, title)


# ---------------------------------------------------------------------------
# CSM run-off
# ---------------------------------------------------------------------------
@singledispatch
def plot_csm_runoff(measurement, *, ax: Axes | None = None,
                    title: str = "CSM run-off") -> Axes:
    """Plot the contractual service margin running off to zero.

    Dispatches on the measurement type. A GMM or VFA measurement draws the
    unearned profit emerging into the income statement as service is
    provided; a reinsurance-held measurement draws its net cost or gain
    amortising -- that CSM may be negative, so its axis is not clamped at
    zero. A PAA measurement is rejected: the PAA carries no CSM.
    """
    raise _reject("plot_csm_runoff()",
                  "a GMM, VFA or reinsurance measurement", measurement)


def _csm_area(measurement, ax, title, *, clamp):
    _require_full(measurement, "plot_csm_runoff()")
    ax = _axes(ax)
    csm = measurement.csm_path.sum(axis=0)
    months = np.arange(csm.shape[0])
    ax.fill_between(months, csm, color=_COLOR["csm"], alpha=0.22)
    ax.plot(months, csm, color=_COLOR["csm"], linewidth=2.6)
    ax.set_xlim(0, max(int(months[-1]), 1))
    if clamp:
        ax.set_ylim(bottom=0.0)
    else:
        ax.axhline(0.0, color=_COLOR["ink"], linewidth=0.8)
    _finish(ax, title, xlabel="month", ylabel="CSM")
    return ax


@plot_csm_runoff.register
def _(measurement: _gmm.Measurement, *, ax=None, title="CSM run-off"):
    _require_inception(measurement, "plot_csm_runoff()")
    return _csm_area(measurement, ax, title, clamp=True)


@plot_csm_runoff.register
def _(measurement: _vfa.Measurement, *, ax=None, title="CSM run-off"):
    _require_settlement_csm(measurement, "plot_csm_runoff()")
    return _csm_area(measurement, ax, title, clamp=True)


@plot_csm_runoff.register
def _(measurement: _reinsurance.Measurement, *, ax=None,
      title="Reinsurance CSM run-off"):
    _require_inception(measurement, "plot_csm_runoff()")
    return _csm_area(measurement, ax, title, clamp=False)


@plot_csm_runoff.register
def _(measurement: _paa.Measurement, *, ax=None, title="CSM run-off"):
    raise TypeError(
        "plot_csm_runoff() does not apply to the PAA -- a PAA liability has "
        "no CSM (the LRC itself carries the unearned profit); "
        "plot_liability() shows the LRC running off")


# ---------------------------------------------------------------------------
# Risk adjustment as a confidence level
# ---------------------------------------------------------------------------
@singledispatch
def plot_risk_adjustment(measurement, basis: Basis,
                         *, bands: tuple[float, ...] = (0.75, 0.85),
                         ax: Axes | None = None,
                         title: str = "The risk adjustment as a confidence level",
                         ) -> Axes:
    """Plot the risk adjustment as a percentile of the liability distribution.

    The confidence-level method models the value arising from non-financial
    risk as a normal distribution centred on the best estimate; the risk
    adjustment is the margin from that mean out to a chosen percentile. This
    chart draws that normal distribution and shades the margin up to each
    confidence level in ``bands``. Dispatches on the measurement type: a GMM
    measurement requires a confidence-level basis; a VFA measurement's RA is
    always this construct (a confidence-level margin for expense risk), as
    is a reinsurance-held measurement's (the margin on the ceded claims --
    the risk transferred, which *reduces* the net cost, so its margin shades
    to the left of the best estimate). A PAA measurement is rejected -- the
    PAA carries no explicit risk adjustment.
    """
    raise _reject("plot_risk_adjustment()",
                  "a GMM, VFA or reinsurance measurement", measurement)


def _ra_fan(mu, ra, z_confidence, bands, ax, title, xlabel, side=1.0):
    """Draw the normal distribution and shade the RA margin per band.

    ``mu`` is the best estimate, ``ra`` the headline risk adjustment and
    ``z_confidence`` the z-score of the basis confidence level -- so
    ``sigma = ra / z_confidence`` recovers the implied distribution width.
    ``side`` is the direction the margin moves the fulfilment value:
    ``+1.0`` adds to a direct liability (FCF = BEL + RA), ``-1.0`` reduces a
    reinsurance-held net cost (FCF = BEL - RA, the risk transferred).
    """
    if ra <= 0.0:
        raise ValueError("the risk adjustment is zero -- nothing to plot")
    sigma = ra / z_confidence

    ax = _axes(ax)
    x = np.linspace(mu - 3.6 * sigma, mu + 3.6 * sigma, 400)
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    ax.plot(x, pdf, color=_COLOR["bel"], linewidth=2.0, zorder=4)
    ax.fill_between(x, pdf, color=_COLOR["bel"], alpha=0.10, zorder=1)

    ax.axvline(mu, color=_COLOR["ink"], linewidth=1.6, zorder=5,
               label="best estimate (BEL)")
    for band in sorted(bands):
        z_band = _norm_ppf(band)
        percentile = mu + side * z_band * sigma
        region = ((x >= mu) & (x <= percentile) if side >= 0.0
                  else (x >= percentile) & (x <= mu))
        ax.fill_between(x[region], pdf[region], color=_COLOR["ra"],
                        alpha=0.22, zorder=2)
        ax.axvline(percentile, color=_COLOR["ra"], linewidth=1.5,
                   linestyle="--", zorder=5,
                   label=f"{band:.0%} confidence -- RA {_compact(z_band * sigma)}")
    ax.set_ylim(bottom=0.0)
    _finish(ax, title, xlabel=xlabel, ylabel="density", money_axis="x")
    ax.set_yticks([])
    _legend(ax)
    return ax


@plot_risk_adjustment.register
def _(measurement: _gmm.Measurement, basis, *, bands=(0.75, 0.85), ax=None,
      title="The risk adjustment as a confidence level"):
    _require_inception(measurement, "plot_risk_adjustment()")
    if basis.ra_method != "confidence_level":
        raise ValueError(
            "plot_risk_adjustment shows the confidence-level risk "
            "adjustment; these basis use the cost-of-capital method"
        )
    return _ra_fan(float(measurement.bel.sum()), float(measurement.ra.sum()),
                   _norm_ppf(basis.ra_confidence), bands, ax, title,
                   "liability from non-financial risk")


@plot_risk_adjustment.register
def _(measurement: _vfa.Measurement, basis, *, bands=(0.75, 0.85), ax=None,
      title="The risk adjustment as a confidence level"):
    _require_settlement_csm(measurement, "plot_risk_adjustment()")
    # The VFA RA is always a confidence-level margin for expense risk
    # (z(ra_confidence) x expense_cv x PV(expenses)) -- no method check.
    return _ra_fan(float(measurement.bel.sum()), float(measurement.ra.sum()),
                   _norm_ppf(basis.ra_confidence), bands, ax, title,
                   "liability from non-financial risk")


@plot_risk_adjustment.register
def _(measurement: _reinsurance.Measurement, basis, *, bands=(0.75, 0.85),
      ax=None, title="The risk adjustment as a confidence level"):
    _require_inception(measurement, "plot_risk_adjustment()")
    # The reinsurance-held RA is always the confidence-level margin on the
    # ceded claims -- the risk transferred (paragraph 64) -- no method check.
    # It reduces the net cost (FCF = BEL - RA), so the margin shades to the
    # left of the best estimate.
    return _ra_fan(float(measurement.bel.sum()), float(measurement.ra.sum()),
                   _norm_ppf(basis.ra_confidence), bands, ax, title,
                   "reinsurance net cost from non-financial risk", side=-1.0)


@plot_risk_adjustment.register
def _(measurement: _paa.Measurement, basis, *, bands=(0.75, 0.85), ax=None,
      title="The risk adjustment as a confidence level"):
    raise TypeError(
        "plot_risk_adjustment() does not apply to the PAA -- a PAA liability "
        "carries no explicit risk adjustment (the LRC is an unearned-premium "
        "balance)")


# ---------------------------------------------------------------------------
# Analysis of change (waterfall)
# ---------------------------------------------------------------------------
@singledispatch
def plot_analysis_of_change(reconciliation, *, component: str = "csm",
                            ax: Axes | None = None,
                            title: str | None = None) -> Axes:
    """Plot one reporting period's analysis of change as a waterfall.

    Dispatches on the reconciliation type. A GMM reconciliation bridges
    ``component`` -- ``"bel"``, ``"ra"`` or ``"csm"`` -- from the opening
    balance to the closing balance through the future-service, finance and
    release drivers; a VFA or reinsurance reconciliation through finance and
    release. A PAA reconciliation selects one of its paragraph-100 blocks:
    ``component`` is ``"lrc"`` (the default there), ``"loss_component"`` or
    ``"lic_path"``.

    A *settlement* reconciliation (from ``gmm.settle`` / ``vfa.settle`` via
    ``reconcile``) has no waterfall arm in v1 and is rejected here; its
    ``str()`` form prints the full paragraph-44 / paragraph-45 table.
    """
    raise _reject("plot_analysis_of_change()",
                  "a GMM, PAA, VFA or reinsurance reconciliation",
                  reconciliation)


def _waterfall(steps, ax, title, ylabel):
    """Draw an opening -> drivers -> closing waterfall.

    ``steps`` is ``(label, value)`` pairs: the opening balance, each
    driver's signed contribution, and the closing balance.
    """
    labels = [label for label, _ in steps]
    values = [value for _, value in steps]
    opening, closing = values[0], values[-1]
    # Running level after the opening bar and after each driver; the last
    # one is pinned to the reported closing balance so the final driver bar
    # lands exactly on it.
    levels = [opening]
    for delta in values[1:-1]:
        levels.append(levels[-1] + delta)
    levels[-1] = closing
    spans = [(0.0, opening)]
    spans += [(levels[i], levels[i + 1]) for i in range(len(levels) - 1)]
    spans += [(0.0, closing)]

    ax = _axes(ax)
    for i, (lo, hi) in enumerate(spans):
        if i in (0, len(spans) - 1):
            color = _COLOR["total"]
        else:
            color = _COLOR["up"] if hi >= lo else _COLOR["down"]
        ax.bar(i, hi - lo, bottom=lo, width=0.62, color=color, zorder=3)
    for i, level in enumerate(levels):
        ax.plot([i + 0.31, i + 0.69], [level, level], color=_COLOR["ink"],
                linewidth=1.0, linestyle=(0, (4, 2)), zorder=2)
    for i, (lo, hi) in enumerate(spans):
        ax.annotate(_compact(values[i]), (i, max(lo, hi)),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=8.5, fontweight="bold", color=_COLOR["ink"])
    ax.axhline(0.0, color=_COLOR["ink"], linewidth=0.8)
    # Extra headroom: bold value labels above bars + space under the title.
    ax.margins(y=0.20)
    ax.set_xticks(range(len(spans)))
    ax.set_xticklabels(labels)
    # Waterfalls carry bold value labels above the bars, so the title needs
    # more clearance than the line charts (default pad=12).
    _finish(ax, title, ylabel=ylabel, title_pad=24)
    return ax


def _bel_ra_csm_component(component: str) -> str:
    component = component.lower()
    if component not in ("bel", "ra", "csm"):
        raise ValueError(
            f"component must be 'bel', 'ra' or 'csm', got {component!r}"
        )
    return component


@plot_analysis_of_change.register
def _(reconciliation: Reconciliation, *, component="csm", ax=None,
      title=None):
    r = reconciliation
    component = _bel_ra_csm_component(component)
    steps = (
        ("Opening", getattr(r, f"{component}_opening")),
        ("Future\nservice", getattr(r, f"{component}_future_service")),
        ("Finance", getattr(r, f"{component}_finance")),
        ("Release", getattr(r, f"{component}_release")),
        ("Closing", getattr(r, f"{component}_closing")),
    )
    if title is None:
        title = (f"{component.upper()} analysis of change "
                 f"-- months {r.month_start + 1}-{r.month_end}")
    return _waterfall(steps, ax, title, component.upper())


def _finance_release_waterfall(r, component, ax, title, kind):
    component = _bel_ra_csm_component(component)
    steps = (
        ("Opening", getattr(r, f"{component}_opening")),
        ("Finance", getattr(r, f"{component}_finance")),
        ("Release", getattr(r, f"{component}_release")),
        ("Closing", getattr(r, f"{component}_closing")),
    )
    if title is None:
        title = (f"{kind}{component.upper()} analysis of change "
                 f"-- months {r.month_start + 1}-{r.month_end}")
    return _waterfall(steps, ax, title, component.upper())


@plot_analysis_of_change.register
def _(reconciliation: VFAReconciliation, *, component="csm", ax=None,
      title=None):
    return _finance_release_waterfall(reconciliation, component, ax, title,
                                      "VFA ")


@plot_analysis_of_change.register
def _(reconciliation: ReinsuranceReconciliation, *, component="csm", ax=None,
      title=None):
    return _finance_release_waterfall(reconciliation, component, ax, title,
                                      "Reinsurance ")


@plot_analysis_of_change.register
def _(reconciliation: PAAReconciliation, *, component="lrc", ax=None,
      title=None):
    r = reconciliation
    blocks = {
        "lrc": ("LRC", (
            ("Opening", r.lrc_opening),
            ("Premiums", r.premiums),
            ("Revenue", r.revenue),
            ("Closing", r.lrc_closing),
        )),
        "loss_component": ("Loss component", (
            ("Opening", r.loss_component_opening),
            ("Released", r.loss_component_release),
            ("Closing", r.loss_component_closing),
        )),
        "lic_path": ("LIC", (
            ("Opening", r.lic_opening),
            ("Claims\nincurred", r.claims_incurred),
            ("Claims\npaid", r.claims_paid),
            ("Closing", r.lic_closing),
        )),
    }
    component = component.lower()
    if component not in blocks:
        raise ValueError(
            "component must be 'lrc', 'loss_component' or 'lic_path', got "
            f"{component!r}"
        )
    label, steps = blocks[component]
    if title is None:
        title = (f"{label} analysis of change "
                 f"-- months {r.month_start + 1}-{r.month_end}")
    return _waterfall(steps, ax, title, label)


# ---------------------------------------------------------------------------
# Stochastic distribution
# ---------------------------------------------------------------------------
def plot_stochastic(result: StochasticResult, *, line: str = "bel",
                    ax: Axes | None = None, bins: int = 30,
                    kde: bool = True, title: str | None = None) -> Axes:
    """Plot the distribution of a figure across the stochastic scenarios.

    ``line`` selects ``"bel"``, ``"ra"``, ``"csm"`` or ``"loss_component"``.
    A smooth Gaussian kernel density estimate is drawn over the histogram
    unless ``kde`` is ``False``; the dashed line marks the mean.
    """
    line = line.lower()
    valid = ("bel", "ra", "csm", "loss_component")
    if line not in valid:
        raise ValueError(f"line must be one of {valid}, got {line!r}")
    data = np.asarray(getattr(result, line), dtype=float)

    ax = _axes(ax)
    _counts, edges, _patches = ax.hist(
        data, bins=bins, color=_COLOR["bel"], alpha=0.6,
        edgecolor="white", linewidth=0.6, zorder=3,
    )
    if kde and data.size > 1 and data.std() > 0.0:
        grid = np.linspace(data.min(), data.max(), 256)
        density = _gaussian_kde(data, grid)
        ax.plot(grid, density * data.size * (edges[1] - edges[0]),
                color=_COLOR["ink"], linewidth=2.0, zorder=5)
    mean = float(data.mean())
    ax.axvline(mean, color=_COLOR["loss"], linewidth=1.8, linestyle="--",
               zorder=6, label=f"mean {_compact(mean)}")
    if title is None:
        title = f"{line.upper()} distribution over {data.size} scenarios"
    _finish(ax, title, xlabel=line.upper(), ylabel="scenarios",
            money_axis="x")
    _legend(ax)
    return ax
