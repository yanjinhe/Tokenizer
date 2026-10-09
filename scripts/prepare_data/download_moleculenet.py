#!/usr/bin/env python
"""Download the six MoleculeNet classification datasets used in Table 3 (DeepChem mirrors).

    python scripts/prepare_data/download_moleculenet.py --output_dir data/moleculenet
"""

import argparse
import os
import urllib.request

BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
FILES = ["BBBP.csv", "tox21.csv.gz", "sider.csv.gz", "clintox.csv.gz", "HIV.csv", "bace.csv"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for name in FILES:
        dst = os.path.join(args.output_dir, name)
        if os.path.exists(dst):
            print(f"{dst} exists, skipping")
            continue
        print(f"downloading {BASE + name}")
        urllib.request.urlretrieve(BASE + name, dst)
    print("done")


if __name__ == "__main__":
    main()
