"""Zipf-guided vocabulary size selection (Section 3.3 of the paper).

The procedure starts from a small vocabulary and grows it step by step with BPE.
After the t-th update the corpus is re-tokenized and the Zipf alignment score
``Zipf_t`` (R^2 of the log-log least-squares fit) is computed. The best score
``Zipf_max`` is tracked together with a *stagnation counter*: whenever
``Zipf_t`` fails to exceed ``Zipf_max`` by more than a small threshold
``epsilon`` the counter is incremented, and once it reaches ``N`` (``patience``)
consecutive steps the Zipfian fit is considered stable and the expansion stops.

The paper takes "the current vocabulary" at that point as ``V_opt``. By default
(``return_vocab="best"``) this is read as the vocabulary of the last meaningful
improvement, i.e. the one that set ``Zipf_max``: the steps after it only confirmed
that the fit had stabilised. ``return_vocab="stop"`` returns the vocabulary at
which the expansion stopped instead.

The search itself is independent of how a score is obtained, so
:func:`zipf_guided_selection` takes a callable ``score_fn(vocab_size) -> float``.
``scripts/select_vocab_size.py`` wires it to BPE training + re-tokenization.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

__all__ = [
    "SelectionStep",
    "SelectionResult",
    "vocab_schedule",
    "zipf_guided_selection",
    "select_from_curve",
]


@dataclass
class SelectionStep:
    step: int
    vocab_size: int
    score: float
    improved: bool
    best_score: float
    best_vocab_size: int
    stagnation: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SelectionResult:
    optimal_vocab_size: int
    best_score: float
    stopped_early: bool
    epsilon: float
    patience: int
    history: List[SelectionStep] = field(default_factory=list)
    best_vocab_size: int = 0
    return_vocab: str = "best"

    @property
    def last_vocab_size(self) -> int:
        return self.history[-1].vocab_size if self.history else self.optimal_vocab_size

    def to_dict(self) -> dict:
        d = asdict(self)
        d["last_vocab_size"] = self.last_vocab_size
        return d


def vocab_schedule(
    initial: int,
    maximum: int,
    step: Optional[int] = None,
    growth: Optional[float] = None,
    sizes: Optional[Iterable[int]] = None,
) -> Iterator[int]:
    """Yield the vocabulary sizes visited by the search.

    Exactly one of ``step`` (arithmetic growth), ``growth`` (geometric growth,
    e.g. 1.25) or ``sizes`` (an explicit grid) should be given.
    """
    given = sum(x is not None for x in (step, growth, sizes))
    if given != 1:
        raise ValueError("specify exactly one of step, growth or sizes")
    if sizes is not None:
        for v in sorted(set(int(s) for s in sizes)):
            if v > maximum:
                break
            if v >= initial:
                yield v
        return
    if initial <= 0 or maximum < initial:
        raise ValueError("need 0 < initial <= maximum")
    v = initial
    if step is not None:
        if step <= 0:
            raise ValueError("step must be positive")
        while v <= maximum:
            yield v
            v += step
    else:
        if growth is None or growth <= 1.0:
            raise ValueError("growth must be > 1")
        while v <= maximum:
            yield v
            v = max(v + 1, int(math.ceil(v * growth)))


def zipf_guided_selection(
    score_fn: Callable[[int], float],
    schedule: Iterable[int],
    *,
    epsilon: float = 1e-3,
    patience: int = 2,
    return_vocab: str = "best",
    callback: Optional[Callable[[SelectionStep], None]] = None,
) -> SelectionResult:
    """Grow the vocabulary until the Zipf score stops improving.

    Args:
        score_fn: maps a vocabulary size to its Zipf alignment score ``Zipf_t``.
        schedule: increasing vocabulary sizes to visit (see :func:`vocab_schedule`).
        epsilon: minimum improvement over ``Zipf_max`` that counts as progress.
        patience: ``N``; stop after this many consecutive non-improving steps.
        return_vocab: ``"best"`` (vocabulary that set ``Zipf_max``) or ``"stop"``
            (vocabulary at which the expansion stopped).
        callback: called with every :class:`SelectionStep` (e.g. for logging).

    Returns:
        A :class:`SelectionResult`; ``optimal_vocab_size`` is ``V_opt``.
    """
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if return_vocab not in ("best", "stop"):
        raise ValueError("return_vocab must be 'best' or 'stop'")
    best_score = -math.inf
    best_vocab = None
    stagnation = 0
    history: List[SelectionStep] = []
    stopped_early = False
    prev = None
    for t, vocab_size in enumerate(schedule):
        if prev is not None and vocab_size <= prev:
            raise ValueError("schedule must be strictly increasing")
        prev = vocab_size
        score = float(score_fn(vocab_size))
        improved = score > best_score + epsilon
        if improved:
            best_score, best_vocab, stagnation = score, vocab_size, 0
        else:
            stagnation += 1
        rec = SelectionStep(
            step=t,
            vocab_size=int(vocab_size),
            score=score,
            improved=improved,
            best_score=float(best_score),
            best_vocab_size=int(best_vocab),
            stagnation=stagnation,
        )
        history.append(rec)
        if callback is not None:
            callback(rec)
        if stagnation >= patience:
            stopped_early = True
            break
    if best_vocab is None:
        raise ValueError("empty schedule")
    chosen = best_vocab if return_vocab == "best" else history[-1].vocab_size
    return SelectionResult(
        optimal_vocab_size=int(chosen),
        best_score=float(best_score),
        stopped_early=stopped_early,
        epsilon=epsilon,
        patience=patience,
        history=history,
        best_vocab_size=int(best_vocab),
        return_vocab=return_vocab,
    )


def select_from_curve(
    vocab_sizes: Sequence[int],
    scores: Sequence[float],
    *,
    epsilon: float = 1e-3,
    patience: int = 2,
    return_vocab: str = "best",
) -> SelectionResult:
    """Run the stopping rule on a pre-computed (vocab size, Zipf score) curve."""
    if len(vocab_sizes) != len(scores):
        raise ValueError("vocab_sizes and scores must have the same length")
    order = sorted(range(len(vocab_sizes)), key=lambda i: vocab_sizes[i])
    lookup = {int(vocab_sizes[i]): float(scores[i]) for i in order}
    return zipf_guided_selection(
        lambda v: lookup[v],
        [int(vocab_sizes[i]) for i in order],
        epsilon=epsilon,
        patience=patience,
        return_vocab=return_vocab,
    )
