#!/usr/bin/env python
"""Zipf-guided vocabulary size selection (Section 3.3).

Starting from a small vocabulary, the vocabulary is grown step by step with BPE.
After each step the analysis corpus is re-tokenized and the Zipf score
``Zipf_t`` (R^2 of the least-squares fit of the log-log rank-frequency curve) is
computed. The best score ``Zipf_max`` is tracked; if ``Zipf_t`` fails to exceed
``Zipf_max`` by more than ``--epsilon`` for ``--patience`` (N) consecutive steps the
Zipfian fit is considered stable and the expansion stops. ``V_opt`` is the
vocabulary of the last meaningful improvement (``--return_vocab best``, default)
or the vocabulary at which the expansion stopped (``--return_vocab stop``).

Growing the vocabulary by BPE merges is implemented by training BPE once up to
``--max_vocab`` and taking prefixes of its merge list (identical to re-training at
each size, but much faster); ``--retrain`` re-trains at every step instead.

Examples
--------
::

    python scripts/select_vocab_size.py --preset smiles --corpus data/zinc20/zinc20_5M.txt \
        --initial_vocab 500 --step 500 --max_vocab 10000 --epsilon 1e-3 --patience 2 \
        --output_dir outputs/smiles/selection

    # offline: apply the stopping rule to a pre-computed curve (e.g. zipf_scores.csv)
    python scripts/select_vocab_size.py --scores_csv outputs/text/zipf/zipf_scores.csv \
        --epsilon 1e-3 --patience 2 --output_dir outputs/text/selection
"""

import argparse
import csv
import logging
import os

import _bootstrap  # noqa: F401

from zipftok.selection import select_from_curve, vocab_schedule, zipf_guided_selection
from zipftok.utils import parse_int_list, setup_logging, write_json

logger = logging.getLogger("select_vocab_size")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epsilon", type=float, default=1e-3, help="minimum improvement of Zipf_max")
    p.add_argument("--patience", type=int, default=2, help="N: consecutive non-improving steps before stopping")
    p.add_argument("--return_vocab", choices=["best", "stop"], default="best",
                   help="V_opt = vocabulary that set Zipf_max (best) or where the expansion stopped (stop)")
    # offline mode
    p.add_argument("--scores_csv", default=None, help="CSV with vocab_size and a score column (offline mode)")
    p.add_argument("--score_column", default="r2")
    # online mode
    p.add_argument("--preset", choices=["text", "translation", "dna", "smiles"])
    p.add_argument("--languages", nargs="*", default=[])
    p.add_argument("--corpus", action="append", help="BPE training corpus spec (repeatable)")
    p.add_argument("--text_column", default="text")
    p.add_argument("--max_texts_per_corpus", type=int, default=None)
    p.add_argument("--analysis_corpus", action="append", default=None,
                   help="corpus used to measure the Zipf fit (default: the training corpus)")
    p.add_argument("--analysis_max_texts", type=int, default=None, help="max texts per analysis corpus spec")
    p.add_argument("--fasta_chunk", type=int, default=None)
    p.add_argument("--initial_vocab", type=int, default=None)
    p.add_argument("--max_vocab", type=int, default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--step", type=int, default=None, help="arithmetic growth step")
    g.add_argument("--growth", type=float, default=None, help="geometric growth factor, e.g. 1.25")
    g.add_argument("--sizes", type=parse_int_list, default=None, help="explicit grid of vocabulary sizes")
    p.add_argument("--retrain", action="store_true", help="re-train BPE at every step instead of truncating")
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--weighting", choices=["rank", "log_binned"], default="rank")
    p.add_argument("--no_save_tokenizer", action="store_true")
    return p.parse_args()


def log_step(rec):
    logger.info("step %2d | vocab %7d | Zipf_t %.4f | Zipf_max %.4f (vocab %d) | %s | stagnation %d",
                rec.step, rec.vocab_size, rec.score, rec.best_score, rec.best_vocab_size,
                "improved" if rec.improved else "no gain ", rec.stagnation)


def write_history(result, out_dir):
    path = os.path.join(out_dir, "selection_history.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.history[0].to_dict().keys()))
        w.writeheader()
        for r in result.history:
            w.writerow(r.to_dict())
    write_json(result.to_dict(), os.path.join(out_dir, "selection.json"))
    try:
        from zipftok.plotting import plot_selection

        plot_selection(result, os.path.join(out_dir, "selection.png"))
    except ImportError:
        pass


def offline(args):
    sizes, scores = [], []
    with open(args.scores_csv) as f:
        for row in csv.DictReader(f):
            sizes.append(int(float(row["vocab_size"])))
            scores.append(float(row[args.score_column]))
    result = select_from_curve(sizes, scores, epsilon=args.epsilon, patience=args.patience,
                               return_vocab=args.return_vocab)
    for rec in result.history:
        log_step(rec)
    return result


def online(args):
    from zipftok.counting import WordTable, count_tokens
    from zipftok.data import iter_corpora
    from zipftok.tokenization import get_preset, save_tokenizer, train_bpe, truncate_tokenizer
    from zipftok.zipf import fit_loglog, rank_frequency

    for name in ("preset", "corpus", "initial_vocab", "max_vocab"):
        if getattr(args, name) is None:
            raise SystemExit(f"online mode needs --{name}")
    if args.step is None and args.growth is None and args.sizes is None:
        raise SystemExit("give one of --step, --growth or --sizes")
    preset = get_preset(args.preset, args.languages)

    def train_corpus():
        return iter_corpora(args.corpus, text_column=args.text_column,
                            max_texts_per_corpus=args.max_texts_per_corpus, fasta_chunk=args.fasta_chunk)

    def analysis_corpus():
        specs = args.analysis_corpus or args.corpus
        limit = args.analysis_max_texts if args.analysis_max_texts is not None else args.max_texts_per_corpus
        return iter_corpora(specs, text_column=args.text_column, max_texts_per_corpus=limit,
                            fasta_chunk=args.fasta_chunk)

    full = None
    if not args.retrain:
        logger.info("training BPE up to %d tokens", args.max_vocab)
        full = train_bpe(train_corpus(), args.max_vocab, preset, min_frequency=args.min_frequency)
        if full.get_vocab_size() < args.max_vocab:
            logger.warning("corpus supports only %d tokens", full.get_vocab_size())

    def make_tokenizer(v):
        if args.retrain:
            return train_bpe(train_corpus(), v, preset, min_frequency=args.min_frequency, show_progress=False)
        return truncate_tokenizer(full, v)

    probe = make_tokenizer(args.initial_vocab)
    logger.info("building word table of the analysis corpus ...")
    table = WordTable.from_texts(probe, analysis_corpus())
    logger.info("%d texts, %d words, %d distinct", table.n_texts, table.n_words, len(table))

    cache = {}

    def score_fn(v):
        tok = probe if v == args.initial_vocab else make_tokenizer(v)
        counts = count_tokens(tok, table=table, method="words")
        ranks, freqs = rank_frequency(counts)
        cache[v] = tok
        return fit_loglog(ranks, freqs, weighting=args.weighting).r2

    schedule = vocab_schedule(args.initial_vocab, args.max_vocab, step=args.step, growth=args.growth,
                              sizes=args.sizes)
    result = zipf_guided_selection(score_fn, schedule, epsilon=args.epsilon, patience=args.patience,
                                   return_vocab=args.return_vocab, callback=log_step)
    if not args.no_save_tokenizer:
        v = result.optimal_vocab_size
        out = os.path.join(args.output_dir, f"vocab_{v}")
        save_tokenizer(cache.get(v) or make_tokenizer(v), out, preset,
                       {"selected_by": "zipf_guided_selection", "epsilon": args.epsilon, "patience": args.patience,
                        "return_vocab": args.return_vocab})
        logger.info("saved V_opt tokenizer to %s", out)
    return result


def main():
    setup_logging()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    result = offline(args) if args.scores_csv else online(args)
    write_history(result, args.output_dir)
    logger.info("V_opt = %d (Zipf_max = %.4f; %s after %d steps)", result.optimal_vocab_size, result.best_score,
                "stopped" if result.stopped_early else "schedule exhausted", len(result.history))


if __name__ == "__main__":
    main()
