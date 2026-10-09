"""mBART multilingual denoising noise (Liu et al., 2020).

mBART corrupts each document with two noise functions:

1. *sentence permutation*: the order of the sentences in the document is shuffled;
2. *text infilling*: 35% of the words are masked by sampling spans whose lengths
   follow a Poisson distribution (lambda = 3.5); every span is replaced by a single
   ``<mask>`` token (a span of length 0 inserts a ``<mask>``).

The model is trained to reconstruct the original document. These functions
operate on plain Python lists of token ids so they can be unit-tested without
PyTorch; :class:`zipftok.collators.MBartDenoisingCollator` applies them per batch.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

__all__ = ["permute_sentences", "text_infilling", "word_starts_from_tokens"]


def permute_sentences(sentences: Sequence[Sequence[int]], rng: np.random.Generator) -> List[List[int]]:
    order = rng.permutation(len(sentences))
    return [list(sentences[i]) for i in order]


def word_starts_from_tokens(tokens: Sequence[str], marker: str = "Ġ") -> np.ndarray:
    """Boolean mask of tokens that start a new word.

    For byte-level BPE a token starting with ``Ġ`` (U+0120, the byte-level symbol
    for a space) begins a word. The first token always starts a word.
    """
    ws = np.array([t.startswith(marker) for t in tokens], dtype=bool)
    if ws.size:
        ws[0] = True
    return ws


def text_infilling(
    ids: Sequence[int],
    mask_id: int,
    rng: np.random.Generator,
    *,
    mask_ratio: float = 0.35,
    poisson_lambda: float = 3.5,
    word_starts: Optional[np.ndarray] = None,
) -> List[int]:
    """Mask ``mask_ratio`` of the words with Poisson-length spans, one ``<mask>`` per span.

    Args:
        ids: token ids of the (already permuted) document, without special tokens.
        mask_id: id of ``<mask>``.
        rng: numpy random generator.
        mask_ratio: fraction of words to mask (0.35 in mBART).
        poisson_lambda: mean span length in words (3.5 in mBART).
        word_starts: boolean array marking the first token of each word. When
            ``None`` every token is treated as a word (sub-word level masking).
    """
    ids = list(ids)
    n_tok = len(ids)
    if n_tok == 0 or mask_ratio <= 0:
        return ids
    if word_starts is None:
        word_starts = np.ones(n_tok, dtype=bool)
    word_starts = np.asarray(word_starts, dtype=bool)
    if word_starts.shape[0] != n_tok:
        raise ValueError("word_starts must have one entry per token")
    word_starts = word_starts.copy()
    word_starts[0] = True
    starts = np.flatnonzero(word_starts)  # token index where each word begins
    n_words = starts.size
    bounds = np.append(starts, n_tok)  # word w spans tokens [bounds[w], bounds[w+1])

    num_to_mask = int(np.ceil(n_words * mask_ratio))
    if num_to_mask == 0:
        return ids
    # Sample span lengths until they cover num_to_mask words, trimming the last one.
    lengths: List[int] = []
    total = 0
    while total < num_to_mask:
        l = int(rng.poisson(poisson_lambda))
        if total + l > num_to_mask:
            l = num_to_mask - total
        lengths.append(l)
        total += l
        if l == 0 and len(lengths) > 4 * num_to_mask + 10:
            break  # guard against pathological lambda ~ 0
    n_insert = sum(1 for l in lengths if l == 0)
    spans = [l for l in lengths if l > 0]

    masked_word = np.zeros(n_words, dtype=bool)
    if spans:
        span_starts = rng.choice(n_words, size=min(len(spans), n_words), replace=False)
        for s, l in zip(span_starts, spans):
            masked_word[s:s + l] = True

    out: List[int] = []
    w = 0
    while w < n_words:
        if masked_word[w]:
            out.append(mask_id)
            while w < n_words and masked_word[w]:
                w += 1
        else:
            out.extend(ids[bounds[w]:bounds[w + 1]])
            w += 1
    # Length-0 spans insert a mask token at random positions.
    for _ in range(n_insert):
        pos = int(rng.integers(0, len(out) + 1))
        out.insert(pos, mask_id)
    return out
