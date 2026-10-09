"""Corpus readers shared by tokenizer training, Zipf analysis and pre-training.

A corpus is given as a *spec* string:

``hf:NAME[:CONFIG[:SPLIT[:COLUMN]]]``
    a HuggingFace dataset, e.g. ``hf:bookcorpus::train:text``,
    ``hf:Skylion007/openwebtext::train:text`` or ``hf:wmt16:de-en:train:translation.de``
    (``COLUMN`` may be a dotted path into nested fields; empty fields use defaults).
anything else
    a local file, directory or glob. ``.txt`` files contain one text per line,
    ``.jsonl`` files one JSON object per line (``text_column`` selects the field),
    ``.csv``/``.tsv`` are read with ``text_column`` as header name, ``.smi`` files use
    the first whitespace-separated field, and FASTA files (``.fa``, ``.fasta``,
    ``.fna``) yield sequences. Any of these may be gzip-compressed (``.gz``).
"""

from __future__ import annotations

import csv
import glob
import gzip
import io
import itertools
import json
import os
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence

__all__ = [
    "CorpusSpec",
    "parse_spec",
    "iter_corpus",
    "iter_corpora",
    "read_fasta",
    "chunk_sequence",
    "load_hf_dataset",
]

FASTA_SUFFIXES = (".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")


@dataclass
class CorpusSpec:
    kind: str  # "hf" or "file"
    name: str
    config: Optional[str] = None
    split: str = "train"
    column: Optional[str] = None


def parse_spec(spec: str, default_column: str = "text") -> CorpusSpec:
    if spec.startswith("hf:"):
        parts = spec[3:].split(":")
        parts += [""] * (4 - len(parts))
        name, config, split, column = parts[:4]
        return CorpusSpec("hf", name, config or None, split or "train", column or default_column)
    return CorpusSpec("file", spec, column=default_column)


def _open(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _get_field(example: dict, dotted: str):
    cur = example
    for key in dotted.split("."):
        cur = cur[key]
    return cur


def load_hf_dataset(name: str, config: Optional[str] = None, split: Optional[str] = "train",
                    streaming: bool = False, **kwargs):
    """``datasets.load_dataset`` with ``trust_remote_code`` when the installed version supports it."""
    from datasets import load_dataset

    try:
        return load_dataset(name, config, split=split, streaming=streaming, trust_remote_code=True, **kwargs)
    except TypeError:
        return load_dataset(name, config, split=split, streaming=streaming, **kwargs)
    except ValueError as e:
        if "trust_remote_code" not in str(e):
            raise
        return load_dataset(name, config, split=split, streaming=streaming, **kwargs)


def read_fasta(path: str) -> Iterator[tuple]:
    """Yield ``(header, sequence)`` pairs from a (gzipped) FASTA file."""
    header, chunks = None, []
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def chunk_sequence(seq: str, chunk_length: int, min_length: int = 1, drop_n: bool = True) -> Iterator[str]:
    """Split a long sequence into non-overlapping chunks.

    With ``drop_n`` the sequence is first split at runs of ``N`` (unknown bases), as is
    usual when building DNA pre-training corpora.
    """
    seq = seq.upper()
    pieces = [p for p in seq.split("N") if p] if drop_n else [seq]
    for piece in pieces:
        for i in range(0, len(piece), chunk_length):
            c = piece[i:i + chunk_length]
            if len(c) >= min_length:
                yield c


def _expand_paths(path: str) -> List[str]:
    if os.path.isdir(path):
        out = []
        for root, _, files in os.walk(path):
            out.extend(os.path.join(root, f) for f in files)
        return sorted(out)
    matches = sorted(glob.glob(path))
    if not matches:
        raise FileNotFoundError(path)
    return matches


def _iter_file(path: str, column: Optional[str], fasta_chunk: Optional[int]) -> Iterator[str]:
    low = path.lower()
    base = low[:-3] if low.endswith(".gz") else low
    if low.endswith(FASTA_SUFFIXES):
        for _, seq in read_fasta(path):
            if fasta_chunk:
                yield from chunk_sequence(seq, fasta_chunk)
            else:
                yield seq.upper()
        return
    with _open(path) as f:
        if base.endswith(".jsonl") or base.endswith(".json"):
            for line in f:
                line = line.strip()
                if line:
                    yield str(_get_field(json.loads(line), column or "text"))
        elif base.endswith(".csv") or base.endswith(".tsv"):
            reader = csv.DictReader(f, delimiter="\t" if base.endswith(".tsv") else ",")
            for row in reader:
                yield row[column or "text"]
        elif base.endswith(".smi"):
            for line in f:
                parts = line.split()
                if parts and parts[0].lower() != "smiles":
                    yield parts[0]
        else:
            for line in f:
                line = line.rstrip("\n")
                if line.strip():
                    yield line


def iter_corpus(
    spec: str,
    *,
    text_column: str = "text",
    max_texts: Optional[int] = None,
    streaming: bool = True,
    fasta_chunk: Optional[int] = None,
    min_chars: int = 1,
) -> Iterator[str]:
    """Iterate over the texts of one corpus spec (see module docstring)."""
    s = parse_spec(spec, text_column)
    if s.kind == "hf":
        ds = load_hf_dataset(s.name, s.config, split=s.split, streaming=streaming)

        def gen():
            for ex in ds:
                v = _get_field(ex, s.column)
                if isinstance(v, str):
                    yield v

        it: Iterable[str] = gen()
    else:
        it = itertools.chain.from_iterable(_iter_file(p, s.column, fasta_chunk) for p in _expand_paths(s.name))
    it = (t for t in it if len(t.strip()) >= min_chars)
    if max_texts is not None:
        it = itertools.islice(it, max_texts)
    yield from it


def iter_corpora(
    specs: Sequence[str],
    *,
    max_texts_per_corpus: Optional[int] = None,
    **kwargs,
) -> Iterator[str]:
    """Chain several corpora, optionally taking at most ``max_texts_per_corpus`` from each."""
    for spec in specs:
        yield from iter_corpus(spec, max_texts=max_texts_per_corpus, **kwargs)


def reservoir_sample(items: Iterable[str], k: int, seed: int = 0) -> List[str]:
    """Uniform sample of ``k`` items from a stream of unknown length."""
    rng = random.Random(seed)
    sample: List[str] = []
    for i, x in enumerate(items):
        if i < k:
            sample.append(x)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = x
    return sample


def split_sentences_simple(text: str) -> List[str]:
    """Very small sentence splitter used for documents without line structure."""
    out, cur = [], []
    for ch in text:
        cur.append(ch)
        if ch in ".!?。！？":
            s = "".join(cur).strip()
            if s:
                out.append(s)
            cur = []
    s = "".join(cur).strip()
    if s:
        out.append(s)
    return out
