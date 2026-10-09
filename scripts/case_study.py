#!/usr/bin/env python
"""Case study (Section 5.3, Figure 3): how one input is segmented at different vocabulary sizes.

Examples from the paper::

    python scripts/case_study.py --base_tokenizer outputs/text/tokenizers/bpe_max \
        --vocab_sizes 5000,30000,50000 --text "invisible footprints"

    python scripts/case_study.py --base_tokenizer outputs/smiles/tokenizers/bpe_max \
        --vocab_sizes 500,3000,8000 --text "CCCOc1ccc(cc1)c2cccc3c2nccn3"
"""

import argparse
import re

import _bootstrap  # noqa: F401

from zipftok.tokenization import display_tokens, load_raw_tokenizer, truncate_tokenizer
from zipftok.utils import parse_int_list


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tokenizers", nargs="+")
    g.add_argument("--base_tokenizer")
    p.add_argument("--vocab_sizes", type=parse_int_list, default=None)
    p.add_argument("--text", action="append", required=True)
    p.add_argument("--markdown", default=None, help="optionally write the table to this file")
    args = p.parse_args()

    toks = []
    if args.base_tokenizer:
        if not args.vocab_sizes:
            p.error("--base_tokenizer needs --vocab_sizes")
        base = load_raw_tokenizer(args.base_tokenizer)
        toks = [(v, truncate_tokenizer(base, v)) for v in sorted(args.vocab_sizes)]
    else:
        for path in args.tokenizers:
            m = re.search(r"vocab_(\d+)", path)
            t = load_raw_tokenizer(path)
            toks.append((int(m.group(1)) if m else t.get_vocab_size(), t))
        toks.sort(key=lambda x: x[0])

    lines = ["| input | vocab size | #tokens | segmentation |", "|---|---|---|---|"]
    for text in args.text:
        print(f"\n{text}")
        for v, tok in toks:
            pieces = display_tokens(tok, text)
            shown = " ".join("`" + x.replace("|", "\\|") + "`" for x in pieces)
            print(f"  {v:>7d}  ({len(pieces):2d} tokens)  {' / '.join(repr(x) for x in pieces)}")
            lines.append(f"| `{text}` | {v:,} | {len(pieces)} | {shown} |")
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
