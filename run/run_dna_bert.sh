#!/usr/bin/env bash
# Genomics: DNABERT-2-style BERT pre-trained with MLM on DNA, fine-tuned on GUE (Table 2, Figure 2).
#
# Prepare the data first:
#   python scripts/prepare_data/prepare_dna_corpus.py --fasta genomes/*.fa.gz --output_dir data/dna
#   (download GUE following https://github.com/MAGICS-LAB/DNABERT_2 and extract it to data/GUE)
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-outputs/dna}
NGPU=${NGPU:-8}
SEEDS=${SEEDS:-"1 2 3"}
VOCABS=${VOCABS:-"500 1000 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 10000"}
STAGES=${STAGES:-"tokenizers zipf pretrain finetune aggregate"}
TRAIN=${TRAIN:-data/dna/train.txt}
VALID=${VALID:-data/dna/valid.txt}
GUE=${GUE:-data/GUE}
TOK_TEXTS=${TOK_TEXTS:-1000000}
ZIPF_TEXTS=${ZIPF_TEXTS:-200000}
STEPS=${STEPS:-100000}
SEQ_LEN=${SEQ_LEN:-256}             # pre-training length; also caps fine-tuning inputs
BATCH=${BATCH:-32}
ACCUM=${ACCUM:-1}
LR=${LR:-5e-4}
TASKS=${TASKS:-"cpd h_tfp1 h_tfp2 pd m_tfp1 m_tfp2 emp_h3 emp_h4"}
FT_GPU=${FT_GPU:-0}                # fine-tuning runs on one GPU (avoids DataParallel)
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has tokenizers; then
  python scripts/train_tokenizer.py --preset dna --corpus "$TRAIN" --max_texts_per_corpus "$TOK_TEXTS" \
    --vocab_sizes "${VOCABS// /,}" --output_dir "$OUT/tokenizers"
fi
if has zipf; then
  python scripts/zipf_analysis.py --tokenizers "$OUT"/tokenizers/vocab_* --corpus "$TRAIN" \
    --max_texts "$ZIPF_TEXTS" --output_dir "$OUT/zipf" --plot_title "DNA"
fi
for V in $VOCABS; do
  if has pretrain; then
    torchrun --nproc_per_node "$NGPU" scripts/pretrain_mlm.py \
      --tokenizer_dir "$OUT/tokenizers/vocab_$V" --model_config configs/models/bert_base.json \
      --train_corpus "$TRAIN" --validation_corpus "$VALID" --max_seq_length "$SEQ_LEN" \
      --per_device_train_batch_size "$BATCH" --gradient_accumulation_steps "$ACCUM" \
      --learning_rate "$LR" --max_steps "$STEPS" --fp16 \
      --output_dir "$OUT/vocab_$V/pretrain"
  fi
  if has finetune; then
    for T in $TASKS; do
      for S in $SEEDS; do
        CUDA_VISIBLE_DEVICES="$FT_GPU" python scripts/finetune_gue.py --model_dir "$OUT/vocab_$V/pretrain" --gue_dir "$GUE" --task "$T" \
          --seed "$S" --fp16 --output_dir "$OUT/vocab_$V/finetune/$T/seed_$S"
      done
    done
  fi
done
if has aggregate; then
  python scripts/aggregate_results.py --root "$OUT" --tasks "${TASKS// /,}" \
    --zipf_csv "$OUT/zipf/zipf_scores.csv" --output "$OUT/table_gue.csv"
  python scripts/plot_performance.py --out "$OUT/figure_gue.png" \
    --panel "$OUT/table_gue.csv|avg=GUE avg|r2|Genomics: BERT on GUE|Accuracy"
fi
