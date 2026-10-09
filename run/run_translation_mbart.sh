#!/usr/bin/env bash
# Translation: mBART pre-trained with multilingual denoising on WMT, fine-tuned on IWSLT in both
# directions (Table 4, Figure 2). One language pair per invocation:
#
#   PAIR=de-en bash run/run_translation_mbart.sh   # WMT16 -> IWSLT14 (local files in data/iwslt14.de-en)
#   PAIR=fr-en bash run/run_translation_mbart.sh   # WMT14 -> IWSLT17 (Hugging Face hub)
#   PAIR=zh-en bash run/run_translation_mbart.sh   # WMT18 -> IWSLT15 (local files in data/iwslt15.zh-en)
set -euo pipefail
cd "$(dirname "$0")/.."

PAIR=${PAIR:-de-en}
L1=${PAIR%-*}
L2=${PAIR#*-}
OUT=${OUT:-outputs/translation_$PAIR}
NGPU=${NGPU:-8}
SEEDS=${SEEDS:-"1 2 3"}
VOCABS=${VOCABS:-"2000 5000 8000 10000 20000 30000 40000 50000 60000 70000 80000 100000 110000 120000 130000 140000"}
STAGES=${STAGES:-"tokenizers zipf pretrain finetune aggregate"}
TOK_TEXTS=${TOK_TEXTS:-2000000}    # sentences per language used to learn the merges
ZIPF_TEXTS=${ZIPF_TEXTS:-500000}
STEPS=${STEPS:-100000}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-2}
LR=${LR:-3e-4}

case "$PAIR" in
  de-en) WMT=wmt16; IWSLT_ARGS=(--data_dir "${IWSLT_DIR:-data/iwslt14.de-en}") ;;
  fr-en) WMT=wmt14; IWSLT_ARGS=(--dataset IWSLT/iwslt2017 --dataset_config iwslt2017-fr-en) ;;
  zh-en) WMT=wmt18; IWSLT_ARGS=(--data_dir "${IWSLT_DIR:-data/iwslt15.zh-en}") ;;
  *) echo "unknown PAIR $PAIR"; exit 1 ;;
esac
WMT=${WMT_DATASET:-$WMT}
CORPORA=(--corpus "hf:$WMT:$PAIR:train:translation.$L1" --corpus "hf:$WMT:$PAIR:train:translation.$L2")
FT_GPU=${FT_GPU:-0}                # fine-tuning runs on one GPU (avoids DataParallel)
has() { [[ " $STAGES " == *" $1 "* ]]; }

if has tokenizers; then
  python scripts/train_tokenizer.py --preset translation --languages "$L1" "$L2" "${CORPORA[@]}" \
    --max_texts_per_corpus "$TOK_TEXTS" --vocab_sizes "${VOCABS// /,}" --output_dir "$OUT/tokenizers"
fi
if has zipf; then
  python scripts/zipf_analysis.py --tokenizers "$OUT"/tokenizers/vocab_* "${CORPORA[@]}" \
    --max_texts "$ZIPF_TEXTS" --output_dir "$OUT/zipf" --plot_title "WMT $PAIR"
fi
for V in $VOCABS; do
  if has pretrain; then
    torchrun --nproc_per_node "$NGPU" scripts/pretrain_mbart.py \
      --tokenizer_dir "$OUT/tokenizers/vocab_$V" --model_config configs/models/mbart_6x6_1024.json \
      --dataset "$WMT" --dataset_config "$PAIR" --languages "$L1" "$L2" \
      --per_device_train_batch_size "$BATCH" --gradient_accumulation_steps "$ACCUM" \
      --learning_rate "$LR" --max_steps "$STEPS" --fp16 \
      --output_dir "$OUT/vocab_$V/pretrain"
  fi
  if has finetune; then
    for DIR in "$L1 $L2" "$L2 $L1"; do
      set -- $DIR
      for S in $SEEDS; do
        CUDA_VISIBLE_DEVICES="$FT_GPU" python scripts/finetune_translation.py --model_dir "$OUT/vocab_$V/pretrain" \
          --src_lang "$1" --tgt_lang "$2" "${IWSLT_ARGS[@]}" --seed "$S" --fp16 \
          --output_dir "$OUT/vocab_$V/finetune/$1-$2/seed_$S"
      done
    done
  fi
done
if has aggregate; then
  python scripts/aggregate_results.py --root "$OUT" --tasks "$L1-$L2,$L2-$L1" --no_avg \
    --zipf_csv "$OUT/zipf/zipf_scores.csv" --output "$OUT/table_bleu.csv"
  python scripts/plot_performance.py --out "$OUT/figure_bleu.png" \
    --panel "$OUT/table_bleu.csv|$L1-$L2,$L2-$L1|r2|mBART: $PAIR|BLEU"
fi
