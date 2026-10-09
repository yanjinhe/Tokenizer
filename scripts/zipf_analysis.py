#!/usr/bin/env python
"""Zipf analysis of a set of tokenizers on a corpus (Hypothesis 1, Figure 1, R^2 columns).

For every tokenizer the corpus is tokenized, token frequencies are counted, the
rank-frequency distribution is fitted with a least-squares line in log-log space
and the coefficient of determination R^2 is reported (the Zipf alignment score).
Results go to ``zipf_scores.csv`` / ``.json`` and the log-log curves are drawn
to ``rank_frequency.png`` (Figure 1).

Examples
--------
Analyse tokenizers produced by train_tokenizer.py::

    python scripts/zipf_analysis.py --tokenizers outputs/text/tokenizers/vocab_* \
        --corpus hf:bookcorpus::train:text --max_texts 500000 --output_dir outputs/text/zipf

Or truncate one large BPE model on the fly::

    python scripts/zipf_analysis.py --base_tokenizer outputs/text/tokenizers/bpe_max \
        --vocab_sizes 2000,5000,10000,20000,30000,40000,50000 --corpus ... --output_dir ...
"""

import argparse
import csv
import json
import logging
import os
import re

import numpy as np

import _bootstrap  # noqa: F401

from zipftok.counting import WordTable, count_tokens
from zipftok.data import iter_corpora
from zipftok.plotting import plot_rank_frequency
from zipftok.tokenization import load_raw_tokenizer, truncate_tokenizer
from zipftok.utils import parse_int_list, setup_logging, write_json
from zipftok.zipf import describe_counts, fit_loglog, rank_frequency, segmented_fit

logger = logging.getLogger("zipf_analysis")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tokenizers", nargs="+", help="tokenizer directories or tokenizer.json files")
    g.add_argument("--base_tokenizer", help="a large BPE tokenizer to truncate to --vocab_sizes")
    p.add_argument("--vocab_sizes", type=parse_int_list, default=None)
    p.add_argument("--corpus", action="append", required=True, help="corpus spec (repeatable)")
    p.add_argument("--text_column", default="text")
    p.add_argument("--max_texts", type=int, default=None, help="max texts per corpus spec")
    p.add_argument("--fasta_chunk", type=int, default=None)
    p.add_argument("--method", choices=["auto", "words", "full"], default="auto",
                   help="'words' encodes every distinct pre-tokenized word once (fast, BPE only)")
    p.add_argument("--weighting", choices=["rank", "log_binned"], default="rank")
    p.add_argument("--reference", choices=["fit", "zipf"], default="fit",
                   help="'fit': free least-squares line (paper); 'zipf': slope fixed to -1")
    p.add_argument("--min_count", type=int, default=1)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_counts", action="store_true", help="also save raw token counts (counts_<V>.npy)")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--plot_title", default=None)
    return p.parse_args()


def vocab_from_path(path: str, tok) -> int:
    m = re.search(r"vocab_(\d+)", path)
    return int(m.group(1)) if m else tok.get_vocab_size()


def main():
    setup_logging()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    def corpus():
        return iter_corpora(args.corpus, text_column=args.text_column, max_texts_per_corpus=args.max_texts,
                            fasta_chunk=args.fasta_chunk)

    # (vocab_size, loader) pairs; tokenizers are materialised one at a time
    if args.base_tokenizer:
        if not args.vocab_sizes:
            raise SystemExit("--base_tokenizer needs --vocab_sizes")
        base = load_raw_tokenizer(args.base_tokenizer)
        jobs = [(v, (lambda v=v: truncate_tokenizer(base, v))) for v in sorted(args.vocab_sizes)]
    else:
        jobs = []
        for path in args.tokenizers:
            tok = load_raw_tokenizer(path)
            jobs.append((vocab_from_path(path, tok), (lambda path=path: load_raw_tokenizer(path))))
        jobs.sort(key=lambda x: x[0])

    table = None
    table_key = None
    rows, curves = [], {}
    for v, load in jobs:
        tok = load()
        is_bpe = type(tok.model).__name__ == "BPE"
        method = args.method if args.method != "auto" else ("words" if is_bpe else "full")
        if method == "words":
            # the word table only depends on normalizer + pre-tokenizer, so it is shared
            data = json.loads(tok.to_str())
            key = json.dumps([data.get("normalizer"), data.get("pre_tokenizer")], sort_keys=True)
            if table is None or key != table_key:
                logger.info("building word table ...")
                table = WordTable.from_texts(tok, corpus())
                table_key = key
                logger.info("%d texts, %d words, %d distinct words", table.n_texts, table.n_words, len(table))
            counts = count_tokens(tok, table=table, method="words")
        else:
            counts = count_tokens(tok, texts=corpus(), method="full")

        ranks, freqs = rank_frequency(counts, min_count=args.min_count)
        fit = fit_loglog(ranks, freqs, reference=args.reference, weighting=args.weighting)
        zipf1 = fit_loglog(ranks, freqs, reference="zipf", weighting=args.weighting)
        try:
            seg = segmented_fit(ranks, freqs)
        except ValueError:
            seg = None
        row = {
            "vocab_size": v,
            "r2": fit.r2,
            "slope": fit.slope,
            "exponent": fit.exponent,
            "intercept": fit.intercept,
            "r2_zipf_slope1": zipf1.r2,
            "segmented_r2": seg.r2 if seg else float("nan"),
            "breakpoint_rank": seg.breakpoint_rank if seg else -1,
            "slope_head": seg.slope_head if seg else float("nan"),
            "slope_tail": seg.slope_tail if seg else float("nan"),
            **describe_counts(counts),
        }
        rows.append(row)
        curves[v] = freqs
        if args.save_counts:
            np.save(os.path.join(args.output_dir, f"counts_{v}.npy"), counts)
        logger.info("vocab %6d | R2 %.4f | slope %.3f | used %d/%d types | %d tokens",
                    v, fit.r2, fit.slope, row["types_used"], row["n_vocab"], row["n_tokens"])

    csv_path = os.path.join(args.output_dir, "zipf_scores.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_json({"args": vars(args), "results": rows}, os.path.join(args.output_dir, "zipf_scores.json"))
    logger.info("wrote %s", csv_path)
    if not args.no_plot:
        out = os.path.join(args.output_dir, "rank_frequency.png")
        plot_rank_frequency(curves, out, title=args.plot_title)
        logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
