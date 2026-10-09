#!/usr/bin/env python
"""Figure 2: downstream performance and Zipf R^2 against vocabulary size.

``--paper`` redraws the figure from the numbers reported in the paper
(``results/paper/table*.csv``). ``--panel`` draws your own results, e.g. the CSV
written by scripts/aggregate_results.py::

    python scripts/plot_performance.py --out figure2.png \
        --panel "results/text_glue.csv|avg|r2|NLP: BERT on GLUE|GLUE avg"

Panel syntax: ``CSV|series[,series...]|r2_column|title|y-label`` (series may be
written as ``column=Legend name``).
"""

import argparse
import csv
import os

import _bootstrap  # noqa: F401

from zipftok.plotting import plot_performance_and_r2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "results", "paper")

PAPER_PANELS = [
    f"{PAPER}/table1_glue.csv|avg=GLUE avg|r2|(a) NLP: BERT on GLUE|GLUE score",
    f"{PAPER}/table2_gue.csv|avg=GUE avg|r2|(b) Genomics: BERT on GUE|Accuracy",
    f"{PAPER}/table3_moleculenet.csv|avg=MoleculeNet avg|r2|(c) Chemistry: BERT on MoleculeNet|ROC-AUC",
]
PAPER_TRANSLATION = [
    f"{PAPER}/table4_translation.csv|de_en=De-En,en_de=En-De|r2_de_en|(d) mBART: De-En / En-De|BLEU",
    f"{PAPER}/table4_translation.csv|fr_en=Fr-En,en_fr=En-Fr|r2_fr_en|(e) mBART: Fr-En / En-Fr|BLEU",
    f"{PAPER}/table4_translation.csv|zh_en=Zh-En,en_zh=En-Zh|r2_zh_en|(f) mBART: Zh-En / En-Zh|BLEU",
]


def read_panel(spec):
    path, series, r2col, title, metric = spec.split("|")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["vocab_size"]))
    out = {"title": title, "metric": metric, "vocab_sizes": [int(float(r["vocab_size"])) for r in rows],
           "r2": [float(r[r2col]) for r in rows], "series": {}}
    for s in series.split(","):
        col, _, name = s.partition("=")
        out["series"][name or col] = [float(r[col]) for r in rows]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper", action="store_true", help="redraw Figure 2 from the paper's tables")
    p.add_argument("--panel", action="append", default=[])
    p.add_argument("--out", required=True, help="output image (with --paper, a second *_translation image is written)")
    p.add_argument("--title", default=None)
    args = p.parse_args()
    if args.paper:
        plot_performance_and_r2([read_panel(s) for s in PAPER_PANELS], args.out, suptitle=args.title)
        root, ext = os.path.splitext(args.out)
        plot_performance_and_r2([read_panel(s) for s in PAPER_TRANSLATION], f"{root}_translation{ext}",
                                suptitle=args.title)
        print(f"wrote {args.out} and {root}_translation{ext}")
    if args.panel:
        plot_performance_and_r2([read_panel(s) for s in args.panel], args.out, suptitle=args.title)
        print(f"wrote {args.out}")
    if not args.paper and not args.panel:
        p.error("give --paper or at least one --panel")


if __name__ == "__main__":
    main()
