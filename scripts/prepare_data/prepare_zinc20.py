#!/usr/bin/env python
"""Extract the first N (default 5,000,000) SMILES of ZINC20 into a text file (one per line).

Input is either ZINC20 tranche files downloaded from https://zinc20.docking.org/
(``.smi`` / ``.txt``, optionally gzipped; the first whitespace-separated field is the
SMILES) or a corpus spec understood by ``zipftok.data`` (e.g. a Hugging Face mirror
``hf:<dataset>::train:smiles``)::

    python scripts/prepare_data/prepare_zinc20.py --input "zinc20/**/*.smi" \
        --n 5000000 --valid 10000 --output_dir data/zinc20
"""

import argparse
import glob
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from zipftok.data import iter_corpus  # noqa: E402


def iter_smiles(spec, column):
    if spec.startswith("hf:"):
        yield from iter_corpus(spec, text_column=column)
        return
    files = sorted(glob.glob(spec, recursive=True)) or [spec]
    for path in files:
        for s in iter_corpus(path, text_column=column):
            s = s.split()[0] if s.split() else ""
            if s and s.lower() != "smiles":
                yield s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--column", default="smiles")
    p.add_argument("--n", type=int, default=5_000_000, help="number of training SMILES")
    p.add_argument("--valid", type=int, default=10_000, help="number of extra SMILES held out for validation")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    it = iter_smiles(args.input, args.column)
    n_written = 0
    with open(os.path.join(args.output_dir, "train.txt"), "w") as f:
        for s in itertools.islice(it, args.n):
            f.write(s + "\n")
            n_written += 1
    with open(os.path.join(args.output_dir, "valid.txt"), "w") as f:
        n_valid = 0
        for s in itertools.islice(it, args.valid):
            f.write(s + "\n")
            n_valid += 1
    print(f"wrote {n_written} training and {n_valid} validation SMILES to {args.output_dir}")


if __name__ == "__main__":
    main()
