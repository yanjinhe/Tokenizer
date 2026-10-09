#!/usr/bin/env bash
# Chemistry: BERT pre-trained with MLM on the first 5M SMILES of ZINC20, fine-tuned on
# MoleculeNet (Table 3, Figure 2, Figure 3).
#
# Prepare the data first:
#   python scripts/prepare_data/prepare_zinc20.py --input "zinc20/**/*.smi" --output_dir data/zinc20
#   python scripts/prepare_data/download_moleculenet.py --output_dir data/moleculenet
#   pip install rdkit   # for the scaffold split
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-outputs/smiles}
NGPU=${NGPU:-8}
SEEDS=${SEEDS:-"1 2 3"}
VOCABS=${VOCABS:-"500 1000 1500 2000 2500 3000 3500 4000 5000 8000"}
STAGES=${STAGES:-"tokenizers zipf pretrain finetune aggregate"}
TRAIN=${TRAIN:-data/zinc20/train.txt}
VALID=${VALID:-data/zinc20/valid.txt}
MOLNET=${MOLNET:-data/moleculenet}
STEPS=${STEPS:-100000}
SEQ_LEN=${SEQ_LEN:-128}             # pre-training length; also caps fine-tuning inputs
BATCH=${BATCH:-64}
ACCUM=${ACCUM:-1}
LR=${LR:-5e-4}
TASKS=${TASKS:-"bbbp tox21 sider clintox hiv bace"}
FT_GPU=${FT_GPU:-0}                # fine-tuning runs on one GPU (avoids DataParallel)
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has tokenizers; then
  python scripts/train_tokenizer.py --preset smiles --corpus "$TRAIN" \
    --vocab_sizes "${VOCABS// /,}" --output_dir "$OUT/tokenizers"
fi
if has zipf; then
  python scripts/zipf_analysis.py --tokenizers "$OUT"/tokenizers/vocab_* --corpus "$TRAIN" \
    --output_dir "$OUT/zipf" --plot_title "ZINC20 SMILES"
  python scripts/case_study.py --base_tokenizer "$OUT/tokenizers/bpe_max" --vocab_sizes 500,3000,8000 \
    --text "CCCOc1ccc(cc1)c2cccc3c2nccn3" --markdown "$OUT/zipf/case_study.md"
fi
for V in $VOCABS; do
  if has pretrain; then
    torchrun --nproc_per_node "$NGPU" scripts/pretrain_mlm.py \
      --tokenizer_dir "$OUT/tokenizers/vocab_$V" --model_config configs/models/bert_base.json \
      --train_corpus "$TRAIN" --validation_corpus "$VALID" --line_by_line --max_seq_length "$SEQ_LEN" \
      --per_device_train_batch_size "$BATCH" --gradient_accumulation_steps "$ACCUM" \
      --learning_rate "$LR" --max_steps "$STEPS" --fp16 \
      --output_dir "$OUT/vocab_$V/pretrain"
  fi
  if has finetune; then
    for T in $TASKS; do
      for S in $SEEDS; do
        CUDA_VISIBLE_DEVICES="$FT_GPU" python scripts/finetune_moleculenet.py --model_dir "$OUT/vocab_$V/pretrain" --data_dir "$MOLNET" \
          --task "$T" --seed "$S" --fp16 --output_dir "$OUT/vocab_$V/finetune/$T/seed_$S"
      done
    done
  fi
done
if has aggregate; then
  python scripts/aggregate_results.py --root "$OUT" --tasks "${TASKS// /,}" \
    --zipf_csv "$OUT/zipf/zipf_scores.csv" --output "$OUT/table_moleculenet.csv"
  python scripts/plot_performance.py --out "$OUT/figure_moleculenet.png" \
    --panel "$OUT/table_moleculenet.csv|avg=MoleculeNet avg|r2|Chemistry: BERT on MoleculeNet|ROC-AUC"
fi
