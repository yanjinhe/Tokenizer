"""Dependency-free utilities that operate on HuggingFace ``tokenizer.json`` files.

BPE is a greedy procedure: the merge sequence learned when training up to a
large vocabulary size ``V_max`` starts with exactly the merges that training
up to any smaller size ``V`` would learn. In the HuggingFace ``tokenizers``
BPE trainer, token ids are also assigned in creation order (special tokens,
then the initial alphabet, then one new id per merge that creates a new
string), and training stops as soon as the vocabulary reaches ``vocab_size``.

Consequently a BPE tokenizer of size ``V`` can be obtained from one trained to
``V_max >= V`` by keeping the tokens with id ``< V`` and the merges up to the one
that created token ``V - 1``. This is what makes it cheap to sweep many
vocabulary sizes, and to "grow the vocabulary step by step" as in the Zipf-
guided selection procedure, without re-running the trainer at every step.

``tests/test_bpe_json.py`` checks this equivalence against a reference
re-implementation of the trainer.
"""

from __future__ import annotations

import copy
import json
from typing import Dict, List, Sequence, Tuple, Union

__all__ = [
    "load_tokenizer_json",
    "parse_merges",
    "format_merges",
    "merge_result",
    "bpe_layout",
    "truncate_bpe_json",
]

Pair = Tuple[str, str]


def load_tokenizer_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_merges(merges: Sequence[Union[str, Sequence[str]]]) -> Tuple[List[Pair], str]:
    """Parse the ``model.merges`` field.

    ``tokenizers < 0.20`` serialises merges as ``"a b"`` strings ("legacy"),
    newer versions as ``["a", "b"]`` pairs ("tuple"). Both are accepted and the
    detected format is returned so it can be written back unchanged.
    """
    pairs: List[Pair] = []
    fmt = "tuple"
    for m in merges:
        if isinstance(m, str):
            fmt = "legacy"
            parts = m.split(" ")
            if len(parts) != 2:
                raise ValueError(f"cannot parse legacy merge {m!r}")
            pairs.append((parts[0], parts[1]))
        else:
            if len(m) != 2:
                raise ValueError(f"cannot parse merge {m!r}")
            pairs.append((str(m[0]), str(m[1])))
    return pairs, fmt


def format_merges(pairs: Sequence[Pair], fmt: str) -> list:
    if fmt == "legacy":
        out = []
        for a, b in pairs:
            if " " in a or " " in b:
                raise ValueError("tokens containing spaces cannot be written in the legacy merge format")
            out.append(f"{a} {b}")
        return out
    return [[a, b] for a, b in pairs]


def merge_result(a: str, b: str, continuing_subword_prefix: str = "") -> str:
    """String produced by merging ``a`` and ``b`` (mirrors the trainer)."""
    if continuing_subword_prefix and b.startswith(continuing_subword_prefix):
        b = b[len(continuing_subword_prefix):]
    return a + b


def bpe_layout(tok_json: dict) -> Dict[str, object]:
    """Inspect a BPE ``tokenizer.json``.

    Returns a dict with ``vocab_size`` (model vocabulary size), ``n_initial``
    (special tokens + alphabet, i.e. tokens not produced by a merge),
    ``merges`` (parsed pairs), ``merge_ids`` (id of the token each merge
    produces), ``merge_format`` and ``creator`` (``id -> index of the merge
    that first created it``).
    """
    model = tok_json.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"expected a BPE model, got {model.get('type')!r}")
    vocab: Dict[str, int] = model["vocab"]
    n = len(vocab)
    ids = sorted(vocab.values())
    if ids != list(range(n)):
        raise ValueError("BPE vocabulary ids are not contiguous 0..n-1; cannot truncate safely")
    pairs, fmt = parse_merges(model.get("merges", []))
    prefix = model.get("continuing_subword_prefix") or ""
    merge_ids: List[int] = []
    creator: Dict[int, int] = {}
    for k, (a, b) in enumerate(pairs):
        for part in (a, b):
            if part not in vocab:
                raise ValueError(f"merge part {part!r} is not in the vocabulary")
        new = merge_result(a, b, prefix)
        if new not in vocab:
            raise ValueError(f"merge result {new!r} is not in the vocabulary")
        nid = vocab[new]
        merge_ids.append(nid)
        creator.setdefault(nid, k)
    # Tokens created by merges occupy the last ids [n_initial, n). (A merge can
    # also re-create a string that already existed, e.g. a special token that
    # appears literally in the corpus; such merges do not add a new id.)
    n_initial = n
    while n_initial > 0 and (n_initial - 1) in creator:
        n_initial -= 1
    order = [creator[i] for i in range(n_initial, n)]
    if any(b <= a for a, b in zip(order, order[1:])):
        raise ValueError(
            "token ids are not in merge-creation order; "
            "this tokenizer was not produced by the standard BPE trainer"
        )
    creator = {i: creator[i] for i in range(n_initial, n)}
    return {
        "vocab_size": n,
        "n_initial": n_initial,
        "merges": pairs,
        "merge_ids": merge_ids,
        "merge_format": fmt,
        "creator": creator,
    }


def truncate_bpe_json(tok_json: dict, vocab_size: int) -> dict:
    """Return a copy of ``tok_json`` truncated to ``vocab_size`` tokens.

    The result is identical (same vocabulary, same ids, same merges in the same
    order) to what the BPE trainer would produce when trained with
    ``vocab_size`` on the same data with the same settings.
    """
    layout = bpe_layout(tok_json)
    n = layout["vocab_size"]
    n_initial = layout["n_initial"]
    if vocab_size >= n:
        return copy.deepcopy(tok_json)
    if vocab_size < n_initial:
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than the initial vocabulary "
            f"(special tokens + alphabet = {n_initial})"
        )
    out = copy.deepcopy(tok_json)
    model = out["model"]
    pairs = layout["merges"]
    if vocab_size == n_initial:
        keep = 0
    else:
        # index of the merge that created the last kept token (id vocab_size - 1)
        keep = layout["creator"][vocab_size - 1] + 1
    kept_pairs = pairs[:keep]
    vocab = {t: i for t, i in model["vocab"].items() if i < vocab_size}
    for (a, b), nid in zip(kept_pairs, layout["merge_ids"][:keep]):
        if vocab.get(a, vocab_size) >= vocab_size or vocab.get(b, vocab_size) >= vocab_size or nid >= vocab_size:
            raise AssertionError("inconsistent merge order; truncation would reference dropped tokens")
    model["vocab"] = dict(sorted(vocab.items(), key=lambda kv: kv[1]))
    model["merges"] = format_merges(kept_pairs, layout["merge_format"])

    added = out.get("added_tokens") or []
    for tok in added:
        if tok.get("id", 0) >= vocab_size:
            raise ValueError(
                f"added token {tok.get('content')!r} has id {tok.get('id')} >= {vocab_size}; "
                "special tokens must be part of the trained vocabulary"
            )
    return out


def vocab_size_of(tok_json: dict) -> int:
    """Model vocabulary size plus added tokens that live outside of it."""
    vocab = tok_json["model"]["vocab"]
    n = len(vocab)
    extra = sum(1 for t in (tok_json.get("added_tokens") or []) if t["content"] not in vocab)
    return n + extra
