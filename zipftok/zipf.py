"""Zipf's-law analysis of token rank-frequency distributions.

The paper quantifies how closely a tokenizer's token distribution follows a
power law ``f(r) ∝ r^{-k}`` by fitting a straight line to the log-log
rank-frequency curve with ordinary least squares and using the coefficient of
determination R^2 of that fit as the *Zipf alignment score* (Section 3.3 and
Section 5.1). This module implements that score and a few diagnostics around
it (two-regime segmented fit, fit against an ideal Zipf line with slope -1).

Everything here only depends on numpy so it can be used with any tokenizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]

__all__ = [
    "ZipfFit",
    "SegmentedFit",
    "rank_frequency",
    "fit_loglog",
    "zipf_score",
    "segmented_fit",
    "r_squared",
]


@dataclass
class ZipfFit:
    """Result of a least-squares line fit on the log-log rank-frequency curve.

    Attributes:
        r2: coefficient of determination of the fit (the Zipf alignment score).
        slope: fitted slope of log f vs. log r (negative for Zipf-like data).
        intercept: fitted intercept (in log10 units).
        exponent: Zipf exponent k = -slope.
        n_types: number of distinct tokens that actually occur (points on the curve).
        n_tokens: total number of token occurrences.
        reference: ``"fit"`` (free slope, as in the paper) or ``"zipf"`` (slope fixed to -1).
        weighting: ``"rank"`` (one point per rank) or ``"log_binned"``.
    """

    r2: float
    slope: float
    intercept: float
    exponent: float
    n_types: int
    n_tokens: int
    reference: str = "fit"
    weighting: str = "rank"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentedFit:
    """Two-regime (piecewise linear) fit of the log-log rank-frequency curve.

    Cancho & Solé (2001) observed two power-law regimes in word frequency
    distributions; Figure 1 of the paper shows the same dual-regime structure
    for small vocabularies. ``breakpoint_rank`` is the first rank of the second
    regime.
    """

    r2: float
    breakpoint_rank: int
    slope_head: float
    slope_tail: float
    intercept_head: float
    intercept_tail: float
    n_types: int

    def to_dict(self) -> dict:
        return asdict(self)


def rank_frequency(counts: ArrayLike, min_count: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Turn raw token counts into a rank-frequency distribution.

    Args:
        counts: occurrence count of every token type (order does not matter). Tokens
            with a count below ``min_count`` (in particular tokens that never occur)
            are dropped because log(0) is undefined.
        min_count: minimum count for a token to be placed on the curve.

    Returns:
        ``(ranks, freqs)`` where ``ranks = 1..n`` and ``freqs`` are the counts sorted
        in decreasing order.
    """
    c = np.asarray(counts, dtype=np.float64).ravel()
    if c.size and np.any(c < 0):
        raise ValueError("counts must be non-negative")
    c = c[c >= max(min_count, 1e-12)]
    freqs = np.sort(c)[::-1]
    ranks = np.arange(1, freqs.size + 1, dtype=np.float64)
    return ranks, freqs


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot``."""
    y = np.asarray(y, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        # A perfectly flat curve: a line explains it exactly.
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _log_binned(x: np.ndarray, y: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Average (x, y) points inside equally spaced bins of log-rank ``x``.

    With one point per rank, the tail (where most ranks live) dominates the
    least-squares objective. Log-binning gives every decade of ranks a similar
    weight. This is *not* what the paper uses by default; it is offered as an
    option for robustness checks.
    """
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    sums_x = np.bincount(idx, weights=x, minlength=n_bins)
    sums_y = np.bincount(idx, weights=y, minlength=n_bins)
    n = np.bincount(idx, minlength=n_bins)
    keep = n > 0
    return sums_x[keep] / n[keep], sums_y[keep] / n[keep]


def fit_loglog(
    ranks: ArrayLike,
    freqs: ArrayLike,
    *,
    reference: str = "fit",
    weighting: str = "rank",
    n_bins: int = 100,
    max_rank: Optional[int] = None,
) -> ZipfFit:
    """Least-squares line fit of ``log10(freq)`` against ``log10(rank)``.

    Args:
        ranks, freqs: output of :func:`rank_frequency`.
        reference: ``"fit"`` fits slope and intercept (the paper's choice: "we
            approximate each rank-frequency distribution with a least squares linear
            fit and adopt the coefficient of determination R^2"). ``"zipf"`` fixes the
            slope to -1 (classic Zipf) and only fits the intercept.
        weighting: ``"rank"`` uses every rank as one point (default, paper);
            ``"log_binned"`` averages points in ``n_bins`` log-spaced bins.
        max_rank: optionally restrict the fit to ranks ``<= max_rank``.
    """
    r = np.asarray(ranks, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    if r.shape != f.shape:
        raise ValueError("ranks and freqs must have the same shape")
    if max_rank is not None:
        keep = r <= max_rank
        r, f = r[keep], f[keep]
    if r.size < 2:
        raise ValueError("need at least two token types with non-zero frequency to fit a line")

    x = np.log10(r)
    y = np.log10(f)
    if weighting == "log_binned":
        x, y = _log_binned(x, y, n_bins)
    elif weighting != "rank":
        raise ValueError(f"unknown weighting {weighting!r}")

    if reference == "fit":
        slope, intercept = np.polyfit(x, y, deg=1)
    elif reference == "zipf":
        slope = -1.0
        intercept = float(np.mean(y - slope * x))
    else:
        raise ValueError(f"unknown reference {reference!r}")

    r2 = r_squared(y, slope * x + intercept)
    return ZipfFit(
        r2=float(r2),
        slope=float(slope),
        intercept=float(intercept),
        exponent=float(-slope),
        n_types=int(r.size),
        n_tokens=int(round(float(f.sum()))),
        reference=reference,
        weighting=weighting,
    )


def zipf_score(counts: ArrayLike, *, min_count: int = 1, **fit_kwargs) -> ZipfFit:
    """Zipf alignment score (R^2 of the log-log least-squares fit) from raw counts."""
    ranks, freqs = rank_frequency(counts, min_count=min_count)
    return fit_loglog(ranks, freqs, **fit_kwargs)


def segmented_fit(
    ranks: ArrayLike,
    freqs: ArrayLike,
    *,
    n_candidates: int = 200,
    min_points: int = 10,
) -> SegmentedFit:
    """Best two-segment piecewise-linear least-squares fit in log-log space.

    The breakpoint is searched over ``n_candidates`` log-spaced ranks; each side
    gets its own independent line. The returned R^2 is that of the piecewise
    model on the full curve.
    """
    r = np.asarray(ranks, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    n = r.size
    if n < 2 * min_points:
        raise ValueError(f"need at least {2 * min_points} points for a segmented fit")
    x = np.log10(r)
    y = np.log10(f)
    # Prefix sums let us evaluate the SSE of a line fit on any prefix/suffix in O(1).
    cx, cy = np.cumsum(x), np.cumsum(y)
    cxx, cxy, cyy = np.cumsum(x * x), np.cumsum(x * y), np.cumsum(y * y)

    def seg_stats(lo: int, hi: int):
        # statistics of points [lo, hi)
        def s(c):
            return c[hi - 1] - (c[lo - 1] if lo > 0 else 0.0)

        m = hi - lo
        sx, sy, sxx, sxy, syy = s(cx), s(cy), s(cxx), s(cxy), s(cyy)
        vxx = sxx - sx * sx / m
        vxy = sxy - sx * sy / m
        vyy = syy - sy * sy / m
        slope = vxy / vxx if vxx > 0 else 0.0
        intercept = (sy - slope * sx) / m
        sse = max(vyy - slope * vxy, 0.0)
        return sse, slope, intercept

    candidates = np.unique(
        np.clip(np.round(np.logspace(np.log10(min_points), np.log10(n - min_points), n_candidates)).astype(int),
                min_points, n - min_points)
    )
    best = None
    for b in candidates:
        sse_h, s_h, i_h = seg_stats(0, b)
        sse_t, s_t, i_t = seg_stats(b, n)
        sse = sse_h + sse_t
        if best is None or sse < best[0]:
            best = (sse, b, s_h, i_h, s_t, i_t)
    sse, b, s_h, i_h, s_t, i_t = best
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 1.0
    return SegmentedFit(
        r2=float(r2),
        breakpoint_rank=int(r[b]),
        slope_head=float(s_h),
        slope_tail=float(s_t),
        intercept_head=float(i_h),
        intercept_tail=float(i_t),
        n_types=int(n),
    )


def describe_counts(counts: Iterable[float]) -> dict:
    """Small summary of a token-count vector, handy for logging."""
    c = np.asarray(list(counts) if not isinstance(counts, np.ndarray) else counts, dtype=np.float64)
    used = c[c > 0]
    return {
        "n_vocab": int(c.size),
        "types_used": int(used.size),
        "unused_types": int(c.size - used.size),
        "n_tokens": int(round(float(used.sum()))),
        "hapax_types": int(np.sum(used == 1)),
    }
