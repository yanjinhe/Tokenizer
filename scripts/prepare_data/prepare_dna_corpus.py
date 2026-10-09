#!/usr/bin/env python
"""Build a DNA pre-training corpus (one sequence per line) from FASTA genomes.

DNABERT-2 pre-trains on the human genome plus a multi-species genome collection;
download the FASTA files used there and cut them into non-overlapping chunks
(runs of ``N`` are removed)::

    python scripts/prepare_data/prepare_dna_corpus.py --fasta genomes/*.fa.gz \
        --chunk_length 1000 --valid_fraction 0.005 --output_dir data/dna

writes ``data/dna/train.txt`` and ``data/dna/valid.txt``.
"""

import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from zipftok.data import chunk_sequence, read_fasta  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", nargs="+", required=True, help="FASTA files or globs (.fa/.fasta/.fna, optionally .gz)")
    p.add_argument("--chunk_length", type=int, default=1000)
    p.add_argument("--min_length", type=int, default=100)
    p.add_argument("--valid_fraction", type=float, default=0.005)
    p.add_argument("--max_chunks", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    files = sorted({f for pat in args.fasta for f in glob.glob(pat)})
    if not files:
        raise SystemExit("no FASTA files found")
    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    n_train = n_valid = 0
    with open(os.path.join(args.output_dir, "train.txt"), "w") as ftr, \
            open(os.path.join(args.output_dir, "valid.txt"), "w") as fva:
        for path in files:
            for header, seq in read_fasta(path):
                for chunk in chunk_sequence(seq, args.chunk_length, min_length=args.min_length):
                    if set(chunk) - set("ACGT"):
                        continue
                    if rng.random() < args.valid_fraction:
                        fva.write(chunk + "\n")
                        n_valid += 1
                    else:
                        ftr.write(chunk + "\n")
                        n_train += 1
                    if args.max_chunks and n_train + n_valid >= args.max_chunks:
                        break
                if args.max_chunks and n_train + n_valid >= args.max_chunks:
                    break
            print(f"{path}: {n_train} train / {n_valid} valid chunks so far")
            if args.max_chunks and n_train + n_valid >= args.max_chunks:
                break
    print(f"done: {n_train} train, {n_valid} valid chunks of <= {args.chunk_length} bp in {args.output_dir}")


if __name__ == "__main__":
    main()
