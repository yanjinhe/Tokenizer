#!/usr/bin/env python
"""Train BPE tokenizers for a grid of vocabulary sizes.

By default a single BPE model is trained up to the largest requested size and
every smaller size is obtained by truncating its merge list, which yields exactly
the tokenizer that training with that size would produce (see
``zipftok/bpe_json.py``). ``--independent`` re-trains the tokenizer from scratch
for every size instead.

Examples
--------
NLP (BERT; BookCorpus + OpenWebText)::

    python scripts/train_tokenizer.py --preset text \
        --corpus hf:bookcorpus::train:text --corpus hf:Skylion007/openwebtext::train:text \
        --max_texts_per_corpus 2000000 \
        --vocab_sizes 2000,5000,10000,20000,25000,27500,30000,32500,35000,37500,40000,50000 \
        --output_dir outputs/text/tokenizers

Translation (one shared tokenizer per language pair)::

    python scripts/train_tokenizer.py --preset translation --languages de en \
        --corpus hf:wmt16:de-en:train:translation.de --corpus hf:wmt16:de-en:train:translation.en \
        --vocab_sizes 2000,5000,...,140000 --output_dir outputs/translation_de-en/tokenizers

Genomics / chemistry::

    python scripts/train_tokenizer.py --preset dna    --corpus data/dna/train.txt   --vocab_sizes 500,...,10000
    python scripts/train_tokenizer.py --preset smiles --corpus data/zinc20/zinc20_5M.txt --vocab_sizes 500,...,8000
"""

import argparse
import logging
import os
import time

import _bootstrap  # noqa: F401

from zipftok.data import iter_corpora
from zipftok.tokenization import get_preset, save_tokenizer, train_bpe, truncate_tokenizer
from zipftok.utils import parse_int_list, setup_logging, write_json

logger = logging.getLogger("train_tokenizer")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", required=True, choices=["text", "translation", "dna", "smiles"])
    p.add_argument("--languages", nargs="*", default=[], help="language codes for the translation preset, e.g. de en")
    p.add_argument("--corpus", action="append", required=True, help="corpus spec (repeatable), see zipftok/data.py")
    p.add_argument("--text_column", default="text")
    p.add_argument("--max_texts_per_corpus", type=int, default=None)
    p.add_argument("--fasta_chunk", type=int, default=None, help="chunk length when reading FASTA files")
    p.add_argument("--vocab_sizes", required=True, type=parse_int_list, help="comma separated, e.g. 2000,5000,30k")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--independent", action="store_true", help="train every size from scratch instead of truncating")
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--limit_alphabet", type=int, default=None)
    p.add_argument("--max_token_length", type=int, default=None)
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()
    preset = get_preset(args.preset, args.languages)
    sizes = sorted(set(args.vocab_sizes))
    os.makedirs(args.output_dir, exist_ok=True)

    def corpus():
        return iter_corpora(args.corpus, text_column=args.text_column,
                            max_texts_per_corpus=args.max_texts_per_corpus, fasta_chunk=args.fasta_chunk)

    train_kwargs = dict(preset=preset, min_frequency=args.min_frequency, limit_alphabet=args.limit_alphabet,
                        max_token_length=args.max_token_length)
    summary = {"preset": preset.name, "corpus": args.corpus, "max_texts_per_corpus": args.max_texts_per_corpus,
               "mode": "independent" if args.independent else "truncate", "tokenizers": {}}

    if args.independent:
        for v in sizes:
            t0 = time.time()
            tok = train_bpe(corpus(), v, **train_kwargs)
            out = os.path.join(args.output_dir, f"vocab_{v}")
            save_tokenizer(tok, out, preset, {"requested_vocab_size": v, "mode": "independent"})
            summary["tokenizers"][v] = {"path": out, "actual_vocab_size": tok.get_vocab_size(), "seconds": time.time() - t0}
            logger.info("vocab %d -> %s (%.0fs)", v, out, time.time() - t0)
    else:
        v_max = sizes[-1]
        t0 = time.time()
        full = train_bpe(corpus(), v_max, **train_kwargs)
        logger.info("trained BPE with %d tokens in %.0fs", full.get_vocab_size(), time.time() - t0)
        if full.get_vocab_size() < v_max:
            logger.warning("the corpus only supports %d tokens (min_frequency=%d); larger sizes are capped",
                           full.get_vocab_size(), args.min_frequency)
        save_tokenizer(full, os.path.join(args.output_dir, "bpe_max"), preset, {"requested_vocab_size": v_max})
        for v in sizes:
            tok = truncate_tokenizer(full, v)
            out = os.path.join(args.output_dir, f"vocab_{v}")
            save_tokenizer(tok, out, preset, {"requested_vocab_size": v, "mode": "truncated", "source": "bpe_max"})
            summary["tokenizers"][v] = {"path": out, "actual_vocab_size": tok.get_vocab_size()}
            logger.info("vocab %d -> %s", v, out)
    write_json(summary, os.path.join(args.output_dir, "tokenizers.json"))


if __name__ == "__main__":
    main()
