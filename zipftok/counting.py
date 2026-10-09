"""Token frequency counting.

Two equivalent strategies are provided:

``full``   encode every text with the complete tokenizer pipeline and count ids.
           Works for any HuggingFace tokenizer (BPE, WordPiece, Unigram, ...).

``words``  run the normalizer + pre-tokenizer once to obtain a table of distinct
           pre-tokenized words with their corpus counts, then encode each distinct
           word once with the bare BPE model and weight the result by the word
           count. Because pre-tokenization does not depend on the vocabulary size,
           the word table is shared by every vocabulary size that is analysed, which
           makes sweeping dozens of sizes (or the step-by-step growth of the
           selection algorithm) much cheaper. The counts are identical to ``full``
           for texts that do not literally contain special-token strings.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np

__all__ = ["WordTable", "aggregate_word_encodings", "count_tokens_full", "count_tokens_words", "count_tokens"]


def _batched(items: Iterable, n: int) -> Iterator[list]:
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass
class WordTable:
    """Distinct pre-tokenized words of a corpus and their counts."""

    words: List[str]
    counts: np.ndarray  # int64, aligned with ``words``
    n_texts: int

    @classmethod
    def from_counter(cls, counter: Counter, n_texts: int) -> "WordTable":
        items = counter.most_common()
        return cls([w for w, _ in items], np.asarray([c for _, c in items], dtype=np.int64), n_texts)

    @classmethod
    def from_texts(cls, tokenizer, texts: Iterable[str]) -> "WordTable":
        """Pre-tokenize ``texts`` with ``tokenizer``'s normalizer and pre-tokenizer."""
        normalizer = tokenizer.normalizer
        pre_tokenizer = tokenizer.pre_tokenizer
        counter: Counter = Counter()
        n = 0
        for text in texts:
            n += 1
            if normalizer is not None:
                text = normalizer.normalize_str(text)
            if pre_tokenizer is None:
                if text:
                    counter[text] += 1
                continue
            counter.update(w for w, _ in pre_tokenizer.pre_tokenize_str(text))
        return cls.from_counter(counter, n)

    def __len__(self) -> int:
        return len(self.words)

    @property
    def n_words(self) -> int:
        return int(self.counts.sum())

    def save(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump({"n_texts": self.n_texts, "words": self.words, "counts": self.counts.tolist()}, f,
                      ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "WordTable":
        import json

        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["words"], np.asarray(d["counts"], dtype=np.int64), int(d["n_texts"]))


def aggregate_word_encodings(
    ids_per_word: Sequence[Sequence[int]],
    word_counts: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """Token counts from the encoding of each distinct word and the word counts."""
    lengths = np.fromiter((len(x) for x in ids_per_word), dtype=np.int64, count=len(ids_per_word))
    if lengths.sum() == 0:
        return np.zeros(vocab_size, dtype=np.float64)
    flat = np.fromiter((i for x in ids_per_word for i in x), dtype=np.int64, count=int(lengths.sum()))
    weights = np.repeat(np.asarray(word_counts, dtype=np.float64), lengths)
    return np.bincount(flat, weights=weights, minlength=vocab_size).astype(np.float64)


def _vocab_size(tokenizer) -> int:
    return int(tokenizer.get_vocab_size(with_added_tokens=True))


def count_tokens_full(
    tokenizer,
    texts: Iterable[str],
    *,
    batch_size: int = 1000,
    exclude_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Token counts obtained by encoding every text with the full pipeline."""
    V = _vocab_size(tokenizer)
    counts = np.zeros(V, dtype=np.float64)
    for batch in _batched(texts, batch_size):
        encs = tokenizer.encode_batch(batch, add_special_tokens=False)
        ids = [i for e in encs for i in e.ids]
        if ids:
            counts += np.bincount(np.asarray(ids, dtype=np.int64), minlength=V)[:V]
    if exclude_ids:
        counts[list(exclude_ids)] = 0
    return counts


def count_tokens_words(
    tokenizer,
    table: WordTable,
    *,
    batch_size: int = 20000,
    exclude_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Token counts from a :class:`WordTable` (each distinct word encoded once)."""
    from .tokenization import bare_model_tokenizer

    bare = bare_model_tokenizer(tokenizer)
    V = _vocab_size(tokenizer)
    counts = np.zeros(V, dtype=np.float64)
    for start in range(0, len(table.words), batch_size):
        words = table.words[start:start + batch_size]
        encs = bare.encode_batch(words, add_special_tokens=False)
        counts += aggregate_word_encodings([e.ids for e in encs], table.counts[start:start + batch_size], V)
    if exclude_ids:
        counts[list(exclude_ids)] = 0
    return counts


def count_tokens(tokenizer, texts: Optional[Iterable[str]] = None, table: Optional[WordTable] = None,
                 *, method: str = "auto", exclude_special: bool = True, **kwargs) -> np.ndarray:
    """Dispatch to :func:`count_tokens_words` or :func:`count_tokens_full`.

    ``method="auto"`` uses the word table for BPE models when one is given (or can
    be built from ``texts``) and falls back to full encoding otherwise.
    """
    from .tokenization import special_token_ids

    exclude = special_token_ids(tokenizer) if exclude_special else None
    is_bpe = type(tokenizer.model).__name__ == "BPE"
    if method == "auto":
        method = "words" if is_bpe else "full"
    if method == "words":
        if not is_bpe:
            raise ValueError("method='words' requires a BPE tokenizer")
        if table is None:
            if texts is None:
                raise ValueError("need texts or a WordTable")
            table = WordTable.from_texts(tokenizer, texts)
        return count_tokens_words(tokenizer, table, exclude_ids=exclude, **kwargs)
    if method == "full":
        if texts is None:
            raise ValueError("method='full' needs the texts")
        return count_tokens_full(tokenizer, texts, exclude_ids=exclude, **kwargs)
    raise ValueError(f"unknown method {method!r}")
