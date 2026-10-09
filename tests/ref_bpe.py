"""A small pure-Python BPE trainer that mirrors HuggingFace's ``BpeTrainer``.

Used only by the tests (it is far too slow for real corpora). It follows the
same rules as ``tokenizers/src/models/bpe/trainer.rs``:

* ids are assigned in creation order: special tokens, then the alphabet sorted by
  code point, then one id per merge that creates a new string;
* at every step the most frequent pair is merged; ties go to the smallest pair of
  ids (``Ord for Merge`` compares ``other.pair`` with ``self.pair``);
* merging is applied left to right without overlaps (``Word::merge``);
* training stops as soon as the vocabulary has ``vocab_size`` entries, when no
  pair is left, or when the best pair is rarer than ``min_frequency``.

The output is a ``tokenizer.json``-style dict (BPE model with legacy merges).
"""

from collections import Counter
from typing import Dict, List, Sequence

import numpy as np


def _merge_word(word: List[int], a: int, b: int, new: int) -> List[int]:
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def train_reference_bpe(
    word_counts: Dict[str, int],
    vocab_size: int,
    special_tokens: Sequence[str] = ("[PAD]", "[UNK]"),
    min_frequency: int = 0,
    merge_format: str = "legacy",
) -> dict:
    w2id: Dict[str, int] = {}
    id2w: List[str] = []

    def add(tok: str) -> int:
        if tok not in w2id:
            w2id[tok] = len(id2w)
            id2w.append(tok)
        return w2id[tok]

    for t in special_tokens:
        add(t)
    for c in sorted({c for w in word_counts for c in w}, key=ord):
        add(c)
    items = sorted(word_counts.items())
    words = [[w2id[c] for c in w] for w, _ in items]
    counts = [c for _, c in items]

    merges = []
    while len(w2id) < vocab_size:
        pair_counts: Counter = Counter()
        for word, cnt in zip(words, counts):
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] += cnt
        if not pair_counts:
            break
        (a, b), best = max(pair_counts.items(), key=lambda kv: (kv[1], (-kv[0][0], -kv[0][1])))
        if best < 1 or min_frequency > best:
            break
        new_id = add(id2w[a] + id2w[b])
        merges.append((id2w[a], id2w[b]))
        words = [_merge_word(w, a, b, new_id) for w in words]

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": w2id[t], "content": t, "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False, "special": True}
            for t in special_tokens
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": dict(w2id),
            "merges": [f"{a} {b}" for a, b in merges] if merge_format == "legacy" else [[a, b] for a, b in merges],
        },
    }


def encode_word(tok_json: dict, word: str) -> List[int]:
    """Apply the merges of ``tok_json`` to one word by rank (what the BPE model does)."""
    model = tok_json["model"]
    vocab = model["vocab"]
    ranks = {}
    for i, m in enumerate(model["merges"]):
        a, b = m.split(" ") if isinstance(m, str) else m
        ranks.setdefault((a, b), i)
    symbols = list(word)
    while len(symbols) > 1:
        best = None
        for i in range(len(symbols) - 1):
            r = ranks.get((symbols[i], symbols[i + 1]))
            if r is not None and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        _, i = best
        symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2:]
    return [vocab[s] for s in symbols]


def synthetic_word_counts(seed=0, n_words=300, alphabet="acgtn"):
    rng = np.random.default_rng(seed)
    words = set()
    while len(words) < n_words:
        words.add("".join(rng.choice(list(alphabet), size=int(rng.integers(2, 9)))))
    words = sorted(words)
    # Zipf-like word frequencies
    counts = np.maximum(1, (2000 / np.arange(1, len(words) + 1) ** 1.1)).astype(int)
    order = rng.permutation(len(words))
    return {words[i]: int(c) for i, c in zip(order, counts)}
