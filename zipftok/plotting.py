"""Matplotlib figures in the style of the paper.

* :func:`plot_rank_frequency` - Figure 1: log-log rank-frequency curves for several
  vocabulary sizes (ordered sizes use a single-hue light-to-dark ramp).
* :func:`plot_performance_and_r2` - Figure 2: downstream performance and Zipf R^2
  as a function of vocabulary size, drawn as two aligned panels (no dual y-axis).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
             "#104281", "#0d366b"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def _ramp(n: int) -> List[str]:
    if n <= 1:
        return [BLUE_RAMP[4]]
    idx = np.linspace(0, len(BLUE_RAMP) - 1, n).round().astype(int)
    return [BLUE_RAMP[i] for i in idx]


def _fmt_vocab(v: int) -> str:
    return f"{v / 1000:g}K" if v >= 1000 else str(v)


def plot_rank_frequency(curves: Dict[int, np.ndarray], out_path: str, title: Optional[str] = None,
                        max_points: int = 4000) -> None:
    """Figure 1. ``curves`` maps vocabulary size -> frequencies sorted in decreasing order."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted(curves)
    colors = _ramp(len(sizes))
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for v, c in zip(sizes, colors):
        f = np.asarray(curves[v], dtype=np.float64)
        f = f[f > 0]
        r = np.arange(1, f.size + 1)
        if f.size > max_points:  # thin out evenly in log-rank for a light figure
            keep = np.unique(np.round(np.logspace(0, np.log10(f.size), max_points)).astype(int) - 1)
            r, f = r[keep], f[keep]
        ax.plot(r, f, color=c, linewidth=1.4, label=_fmt_vocab(v))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank (log scale)", fontsize=9)
    ax.set_ylabel("Frequency (log scale)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    leg = ax.legend(title="Vocab size", fontsize=7, title_fontsize=7, frameon=False, ncol=2, loc="lower left")
    for t in leg.get_texts():
        t.set_color(TEXT_2)
    leg.get_title().set_color(TEXT_2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_performance_and_r2(
    panels: Sequence[dict],
    out_path: str,
    suptitle: Optional[str] = None,
) -> None:
    """Figure 2.

    ``panels`` is a list of dicts with keys ``title``, ``vocab_sizes``, ``series``
    (``{name: values}`` of downstream scores), ``r2`` (values) and optionally
    ``metric`` (y-label). Each panel becomes one column with performance on top
    and R^2 below, sharing the vocabulary-size axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 4.6), dpi=200, sharex="col",
                             gridspec_kw={"height_ratios": [1.4, 1.0]}, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, p in enumerate(panels):
        x = np.asarray(p["vocab_sizes"], dtype=float)
        top, bottom = axes[0, j], axes[1, j]
        _style(top)
        _style(bottom)
        names = list(p["series"])
        for k, name in enumerate(names):
            y = np.asarray(p["series"][name], dtype=float)
            color = SERIES[k % len(SERIES)]
            top.plot(x, y, color=color, linewidth=2, marker="o", markersize=3.5, label=name)
            b = int(np.nanargmax(y))
            top.annotate(f"{y[b]:.2f}", (x[b], y[b]), textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=6.5, color=TEXT_2)
        best_r2_x = None
        if len(names) == 1:
            best_r2_x = x[int(np.nanargmax(np.asarray(p["series"][names[0]], dtype=float)))]
        r2 = np.asarray(p["r2"], dtype=float)
        bottom.plot(x, r2, color=SERIES[0], linewidth=2, marker="o", markersize=3.5)
        if best_r2_x is not None:
            for ax in (top, bottom):
                ax.axvline(best_r2_x, color=TEXT_2, linewidth=0.8, linestyle="--", zorder=1)
        top.set_title(p["title"], fontsize=9, loc="left")
        top.set_ylabel(p.get("metric", "Score"), fontsize=8)
        bottom.set_ylabel("Zipf $R^2$", fontsize=8)
        bottom.set_xlabel("Vocabulary size", fontsize=8)
        bottom.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _fmt_vocab(int(v))))
        if len(names) > 1:
            leg = top.legend(fontsize=6.5, frameon=False, loc="lower right")
            for t in leg.get_texts():
                t.set_color(TEXT_2)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_selection(result, out_path: str) -> None:
    """Zipf score along the growth schedule, with ``V_opt`` and the stopping step marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    xs = [h.vocab_size for h in result.history]
    ys = [h.score for h in result.history]
    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=4)
    v = result.optimal_vocab_size
    ax.axvline(v, color=TEXT_2, linewidth=0.8, linestyle="--", zorder=1)
    ax.annotate(f"$V_{{opt}}$ = {_fmt_vocab(v)}", (v, result.best_score), textcoords="offset points",
                xytext=(6, -12), fontsize=7, color=TEXT_2)
    ax.set_xlabel("Vocabulary size", fontsize=8)
    ax.set_ylabel("Zipf $R^2$", fontsize=8)
    ax.set_title(f"Zipf-guided selection (epsilon={result.epsilon:g}, N={result.patience})", fontsize=9, loc="left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: _fmt_vocab(int(val))))
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
