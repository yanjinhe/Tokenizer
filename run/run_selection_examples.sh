#!/usr/bin/env bash
# Zipf-guided vocabulary size selection (Section 3.3) for each domain, without any model training.
# The pre-processed corpora are those of the other run/*.sh scripts.
set -euo pipefail
cd "$(dirname "$0")/.."
EPS=${EPS:-1e-3}
N=${N:-2}

python scripts/select_vocab_size.py --preset text \
  --corpus hf:bookcorpus::train:text --corpus hf:Skylion007/openwebtext::train:text \
  --max_texts_per_corpus "${TOK_TEXTS:-1000000}" --analysis_max_texts "${ZIPF_TEXTS:-200000}" \
  --initial_vocab 2000 --step 2500 --max_vocab 60000 --epsilon "$EPS" --patience "$N" \
  --output_dir outputs/selection/text

python scripts/select_vocab_size.py --preset dna --corpus data/dna/train.txt \
  --max_texts_per_corpus 1000000 --analysis_max_texts 200000 \
  --initial_vocab 500 --step 500 --max_vocab 12000 --epsilon "$EPS" --patience "$N" \
  --output_dir outputs/selection/dna

python scripts/select_vocab_size.py --preset smiles --corpus data/zinc20/train.txt \
  --initial_vocab 500 --step 500 --max_vocab 10000 --epsilon "$EPS" --patience "$N" \
  --output_dir outputs/selection/smiles
