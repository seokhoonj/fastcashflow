"""Visualisation -- charts of the IFRS 17 figures the engine produces.

Turn a measurement, a reconciliation or a stochastic result into a chart.
Every function draws onto a matplotlib Axes -- it creates one if none is
given, and returns it -- so the charts compose into larger figures and stay
easy to save or restyle.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fastcashflow.numerics import _norm_ppf

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from fastcashflow.assumptions import Assumptions
    from fastcashflow.engine import Measurement
    from fastcashflow.movement import Reconciliation
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


def _finish(ax, title, *, xlabel=None, ylabel=None, money_axis="y"):
    """Apply the fastcashflow house style to ``ax``."""
    ink = _COLOR["ink"]
    ax.set_title(title, fontsize=13, fontweight="bold", color=ink,
                 loc="left", pad=12)
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


def plot_liability(measurement: Measurement, *, ax: Axes | None = None,
                   title: str = "Liability components over time") -> Axes:
    """Plot the BEL, RA and CSM trajectories over the contract's life.

    Each line is the portfolio total of that component at each month.
    """
    ax = _axes(ax)
    bel = measurement.bel.sum(axis=0)
    ra = measurement.ra.sum(axis=0)
    csm = measurement.csm.sum(axis=0)
    months = np.arange(bel.shape[0])
    ax.axhline(0.0, color=_COLOR["ink"], linewidth=0.8)
    ax.plot(months, bel, color=_COLOR["bel"], linewidth=2.2, label="BEL")
    ax.plot(months, ra, color=_COLOR["ra"], linewidth=2.2, label="RA")
    ax.plot(months, csm, color=_COLOR["csm"], linewidth=2.2, label="CSM")
    ax.set_xlim(0, max(int(months[-1]), 1))
    _finish(ax, title, xlabel="month", ylabel="amount")
    _legend(ax)
    return ax


def plot_cashflows(measurement: Measurement, *, period_months: int = 12,
                   ax: Axes | None = None,
                   title: str = "Projected cash flows") -> Axes:
    """Plot projected premium income against claim and expense outgo.

    The monthly cash flows are aggregated into buckets of ``period_months``
    months -- a policy year by default. Premiums are drawn upward, claims
    and expenses downward, and the marked line is the net cash flow each
    period. Bucketing keeps the front-loaded acquisition expense from
    dominating the chart while the cash-flow shape stays visible.
    """
    if period_months < 1:
        raise ValueError(f"period_months must be >= 1, got {period_months}")
    ax = _axes(ax)
    cf = measurement.cashflows
    premium = cf.premium_cf.sum(axis=0)
    outgo = (cf.claim_cf + cf.morbidity_cf + cf.annuity_cf
             + cf.expense_cf).sum(axis=0)
    starts = np.arange(0, premium.shape[0], period_months)
    premium_b = np.add.reduceat(premium, starts)
    outgo_b = np.add.reduceat(outgo, starts)
    x = np.arange(premium_b.shape[0])

    ax.bar(x, premium_b, width=0.62, color=_COLOR["csm"],
           label="premiums in", zorder=3)
    ax.bar(x, -outgo_b, width=0.62, color=_COLOR["down"],
           label="claims & expenses out", zorder=3)
    ax.plot(x, premium_b - outgo_b, color=_COLOR["ink"], linewidth=1.6,
            marker="o", markersize=4, label="net", zorder=4)
    ax.axhline(0.0, color=_COLOR["ink"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x])
    _finish(ax, title,
            xlabel="policy year" if period_months == 12 else "period",
            ylabel="amount")
    _legend(ax)
    return ax


def plot_csm_runoff(measurement: Measurement, *, ax: Axes | None = None,
                    title: str = "CSM run-off") -> Axes:
    """Plot the contractual service margin running off to zero.

    The CSM is the unearned profit in the contract; its run-off is the
    profit emerging into the income statement as service is provided.
    """
    ax = _axes(ax)
    csm = measurement.csm.sum(axis=0)
    months = np.arange(csm.shape[0])
    ax.fill_between(months, csm, color=_COLOR["csm"], alpha=0.22)
    ax.plot(months, csm, color=_COLOR["csm"], linewidth=2.6)
    ax.set_xlim(0, max(int(months[-1]), 1))
    ax.set_ylim(bottom=0.0)
    _finish(ax, title, xlabel="month", ylabel="CSM")
    return ax


def plot_risk_adjustment(measurement: Measurement, assumptions: Assumptions,
                         *, bands: tuple[float, ...] = (0.75, 0.85),
                         ax: Axes | None = None,
                         title: str = "The risk adjustment as a confidence level",
                         ) -> Axes:
    """Plot the risk adjustment as a percentile of the liability distribution.

    The confidence-level method models the liability arising from
    non-financial risk as a normal distribution centred on the best
    estimate; the risk adjustment is the margin from that mean out to a
    chosen percentile. This chart draws that normal distribution and shades
    the margin up to each confidence level in ``bands``. It applies to the
    confidence-level method only.
    """
    if assumptions.ra_method != "confidence_level":
        raise ValueError(
            "plot_risk_adjustment shows the confidence-level risk "
            "adjustment; these assumptions use the cost-of-capital method"
        )
    mu = float(measurement.bel[:, 0].sum())
    ra = float(measurement.ra[:, 0].sum())
    if ra <= 0.0:
        raise ValueError("the risk adjustment is zero -- nothing to plot")
    sigma = ra / _norm_ppf(assumptions.ra_confidence)

    ax = _axes(ax)
    x = np.linspace(mu - 3.6 * sigma, mu + 3.6 * sigma, 400)
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    ax.plot(x, pdf, color=_COLOR["bel"], linewidth=2.0, zorder=4)
    ax.fill_between(x, pdf, color=_COLOR["bel"], alpha=0.10, zorder=1)

    ax.axvline(mu, color=_COLOR["ink"], linewidth=1.6, zorder=5,
               label="best estimate (BEL)")
    for band in sorted(bands):
        z_band = _norm_ppf(band)
        percentile = mu + z_band * sigma
        region = (x >= mu) & (x <= percentile)
        ax.fill_between(x[region], pdf[region], color=_COLOR["ra"],
                        alpha=0.22, zorder=2)
        ax.axvline(percentile, color=_COLOR["ra"], linewidth=1.5,
                   linestyle="--", zorder=5,
                   label=f"{band:.0%} confidence -- RA {_compact(z_band * sigma)}")
    ax.set_ylim(bottom=0.0)
    _finish(ax, title, xlabel="liability from non-financial risk",
            ylabel="density", money_axis="x")
    ax.set_yticks([])
    _legend(ax)
    return ax


def plot_analysis_of_change(reconciliation: Reconciliation, *,
                            component: str = "csm",
                            ax: Axes | None = None,
                            title: str | None = None) -> Axes:
    """Plot one reporting period's analysis of change as a waterfall.

    ``component`` selects ``"bel"``, ``"ra"`` or ``"csm"``. The waterfall
    bridges the opening balance to the closing balance through the
    future-service, finance and release drivers.
    """
    component = component.lower()
    if component not in ("bel", "ra", "csm"):
        raise ValueError(
            f"component must be 'bel', 'ra' or 'csm', got {component!r}"
        )
    r = reconciliation
    opening = getattr(r, f"{component}_opening")
    future = getattr(r, f"{component}_future_service")
    finance = getattr(r, f"{component}_finance")
    release = getattr(r, f"{component}_release")
    closing = getattr(r, f"{component}_closing")

    ax = _axes(ax)
    after_fs = opening + future
    after_fin = after_fs + finance
    spans = ((0.0, opening), (opening, after_fs), (after_fs, after_fin),
             (after_fin, closing), (0.0, closing))
    deltas = (opening, future, finance, release, closing)
    for i, (lo, hi) in enumerate(spans):
        if i in (0, 4):
            color = _COLOR["total"]
        else:
            color = _COLOR["up"] if hi >= lo else _COLOR["down"]
        ax.bar(i, hi - lo, bottom=lo, width=0.62, color=color, zorder=3)
    for i, level in enumerate((opening, after_fs, after_fin, closing)):
        ax.plot([i + 0.31, i + 0.69], [level, level], color=_COLOR["ink"],
                linewidth=1.0, linestyle=(0, (4, 2)), zorder=2)
    for i, (lo, hi) in enumerate(spans):
        ax.annotate(_compact(deltas[i]), (i, max(lo, hi)),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=8.5, fontweight="bold", color=_COLOR["ink"])
    ax.axhline(0.0, color=_COLOR["ink"], linewidth=0.8)
    # Extra headroom: bold value labels above bars + space under the title.
    ax.margins(y=0.20)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Opening", "Future\nservice", "Finance",
                        "Release", "Closing"])
    if title is None:
        title = (f"{component.upper()} analysis of change "
                 f"-- months {r.month_start + 1}–{r.month_end}")
    _finish(ax, title, ylabel=component.upper())
    return ax


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
