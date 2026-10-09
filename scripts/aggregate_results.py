#!/usr/bin/env python
"""Collect fine-tuning results into a table like Tables 1-4 of the paper.

Expected layout (produced by the run/*.sh pipelines)::

    <root>/vocab_<V>/finetune/<task>/seed_<s>/results.json   # contains "score"

Scores are averaged over seeds, one row per vocabulary size, one column per
task, plus ``avg`` and (optionally) the Zipf ``r2`` from zipf_analysis.py. The
table is written as CSV and Markdown, and Pearson / Spearman correlations
between R^2 and the average score are printed.

Example::

    python scripts/aggregate_results.py --root outputs/text \
        --tasks cola,sst2,mrpc,stsb,qqp,mnli,qnli,rte --zipf_csv outputs/text/zipf/zipf_scores.csv \
        --output results/text_glue.csv
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

import _bootstrap  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--tasks", default=None, help="comma separated task order (default: all found)")
    p.add_argument("--zipf_csv", default=None)
    p.add_argument("--output", required=True, help="output CSV path (a .md file is written next to it)")
    p.add_argument("--no_avg", action="store_true", help="do not add an avg column (e.g. translation)")
    return p.parse_args()


def main():
    args = parse_args()
    scores = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(args.root, "vocab_*", "finetune", "*", "seed_*", "results.json")):
        m = re.search(r"vocab_(\d+)[/\\]finetune[/\\]([^/\\]+)[/\\]seed_", path)
        if not m:
            continue
        with open(path) as f:
            scores[int(m.group(1))][m.group(2)].append(float(json.load(f)["score"]))
    if not scores:
        raise SystemExit(f"no results under {args.root}")
    tasks = args.tasks.split(",") if args.tasks else sorted({t for v in scores.values() for t in v})

    r2 = {}
    if args.zipf_csv:
        with open(args.zipf_csv) as f:
            for row in csv.DictReader(f):
                r2[int(float(row["vocab_size"]))] = float(row["r2"])

    rows = []
    for v in sorted(scores):
        row = {"vocab_size": v}
        for t in tasks:
            vals = scores[v].get(t, [])
            row[t] = float(np.mean(vals)) if vals else float("nan")
            row[f"{t}_n"] = len(vals)
        if not args.no_avg:
            row["avg"] = float(np.mean([row[t] for t in tasks]))
        if r2:
            row["r2"] = r2.get(v, float("nan"))
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fields = ["vocab_size"] + tasks + ([] if args.no_avg else ["avg"]) + (["r2"] if r2 else []) + \
             [f"{t}_n" for t in tasks]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})

    show = [c for c in fields if not c.endswith("_n")]
    md = ["| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
    best = {}
    for c in show[1:]:
        vals = [r[c] for r in rows if not np.isnan(r[c])]
        if c != "r2" and vals:
            best[c] = max(vals)
    for r in rows:
        cells = [f"{r['vocab_size']:,}"]
        for c in show[1:]:
            val = r[c]
            txt = "-" if np.isnan(val) else (f"{val:.4f}" if c == "r2" else f"{val:.2f}")
            if c in best and not np.isnan(val) and abs(val - best[c]) < 1e-9:
                txt = f"**{txt}**"
            cells.append(txt)
        md.append("| " + " | ".join(cells) + " |")
    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    if r2 and not args.no_avg:
        from scipy.stats import pearsonr, spearmanr

        x = np.array([r["r2"] for r in rows])
        y = np.array([r["avg"] for r in rows])
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() >= 3:
            print(f"\ncorrelation(R^2, avg): pearson {pearsonr(x[ok], y[ok])[0]:.3f}, "
                  f"spearman {spearmanr(x[ok], y[ok])[0]:.3f}")
    print(f"\nwrote {args.output} and {md_path}")


if __name__ == "__main__":
    main()
