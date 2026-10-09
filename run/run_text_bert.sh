#!/usr/bin/env bash
# NLP: BERT pre-trained with MLM on OpenWebText + BookCorpus, fine-tuned on GLUE
# (Table 1, Figure 1, Figure 2a). Stages can be selected with STAGES.
#
#   NGPU=8 bash run/run_text_bert.sh
#   STAGES="finetune aggregate" VOCABS="30000" bash run/run_text_bert.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-outputs/text}
NGPU=${NGPU:-8}
SEEDS=${SEEDS:-"1 2 3"}
VOCABS=${VOCABS:-"2000 5000 10000 20000 25000 27500 30000 32500 35000 37500 40000 50000"}
STAGES=${STAGES:-"tokenizers zipf pretrain finetune aggregate"}
TOK_TEXTS=${TOK_TEXTS:-2000000}     # texts per corpus used to learn the BPE merges
ZIPF_TEXTS=${ZIPF_TEXTS:-500000}    # texts per corpus used to measure the Zipf fit
STEPS=${STEPS:-100000}
BATCH=${BATCH:-32}                   # per GPU; global batch = BATCH * NGPU * ACCUM
ACCUM=${ACCUM:-1}
LR=${LR:-5e-4}
TASKS=${TASKS:-"cola sst2 mrpc stsb qqp mnli qnli rte"}
CORPORA=(--corpus hf:bookcorpus::train:text --corpus hf:Skylion007/openwebtext::train:text)
FT_GPU=${FT_GPU:-0}                # fine-tuning runs on one GPU (avoids DataParallel)
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has tokenizers; then
  python scripts/train_tokenizer.py --preset text "${CORPORA[@]}" --max_texts_per_corpus "$TOK_TEXTS" \
    --vocab_sizes "${VOCABS// /,}" --output_dir "$OUT/tokenizers"
fi

if has zipf; then
  # R^2 on the pre-training corpus (Table 1) ...
  python scripts/zipf_analysis.py --tokenizers "$OUT"/tokenizers/vocab_* "${CORPORA[@]}" \
    --max_texts "$ZIPF_TEXTS" --output_dir "$OUT/zipf"
  # ... and the BookCorpus rank-frequency curves of Figure 1
  python scripts/zipf_analysis.py --tokenizers "$OUT"/tokenizers/vocab_* --corpus hf:bookcorpus::train:text \
    --max_texts "$ZIPF_TEXTS" --output_dir "$OUT/zipf_bookcorpus" --plot_title "BookCorpus"
fi

for V in $VOCABS; do
  if has pretrain; then
    torchrun --nproc_per_node "$NGPU" scripts/pretrain_mlm.py \
      --tokenizer_dir "$OUT/tokenizers/vocab_$V" --model_config configs/models/bert_base.json \
      "${CORPORA[@]/--corpus/--train_corpus}" --max_seq_length 128 \
      --per_device_train_batch_size "$BATCH" --gradient_accumulation_steps "$ACCUM" \
      --learning_rate "$LR" --max_steps "$STEPS" --fp16 \
      --output_dir "$OUT/vocab_$V/pretrain"
  fi
  if has finetune; then
    for T in $TASKS; do
      for S in $SEEDS; do
        CUDA_VISIBLE_DEVICES="$FT_GPU" python scripts/finetune_glue.py --model_dir "$OUT/vocab_$V/pretrain" --task "$T" --seed "$S" --fp16 \
          --output_dir "$OUT/vocab_$V/finetune/$T/seed_$S"
      done
    done
  fi
done

if has aggregate; then
  python scripts/aggregate_results.py --root "$OUT" --tasks "${TASKS// /,}" \
    --zipf_csv "$OUT/zipf/zipf_scores.csv" --output "$OUT/table_glue.csv"
  python scripts/plot_performance.py --out "$OUT/figure_glue.png" \
    --panel "$OUT/table_glue.csv|avg=GLUE avg|r2|NLP: BERT on GLUE|GLUE score"
fi
