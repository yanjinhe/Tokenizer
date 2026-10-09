#!/usr/bin/env python
"""Fine-tune a pre-trained mBART on IWSLT and report test BLEU (Section 4.3, Table 4).

Language pairs of the paper: IWSLT14 De-En, IWSLT17 Fr-En, IWSLT15 Zh-En, each in
both directions. Data can come from the Hugging Face hub (``--dataset`` with a
``translation`` field, e.g. ``IWSLT/iwslt2017`` / ``iwslt2017-fr-en``) or from
plain parallel text files ``{train,valid,test}.{lang}`` in ``--data_dir`` (e.g.
the output of fairseq's ``prepare-iwslt14.sh`` *before* BPE, or the IWSLT15
release converted to plain text).

Encoder input: ``x </s> <src_lang>``; target: ``y </s> <tgt_lang>``; generation
starts from ``<tgt_lang>`` and uses beam search.

Example::

    python scripts/finetune_translation.py --model_dir outputs/translation_de-en/vocab_60000/pretrain \
        --src_lang de --tgt_lang en --data_dir data/iwslt14.de-en --seed 1 \
        --output_dir outputs/translation_de-en/vocab_60000/finetune/de-en/seed_1
"""

import argparse
import logging
import os
import shutil

import _bootstrap  # noqa: F401

from zipftok.metrics import corpus_bleu
from zipftok.utils import set_seed, setup_logging, write_json

logger = logging.getLogger("finetune_translation")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--src_lang", required=True)
    p.add_argument("--tgt_lang", required=True)
    p.add_argument("--dataset", default=None, help="HF dataset id, e.g. IWSLT/iwslt2017")
    p.add_argument("--dataset_config", default=None, help="e.g. iwslt2017-fr-en")
    p.add_argument("--data_dir", default=None, help="directory with {train,valid,test}.{lang} text files")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_source_length", type=int, default=256)
    p.add_argument("--max_target_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--num_train_epochs", type=float, default=10)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=2500)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--generation_batch_size", type=int, default=64)
    p.add_argument("--bleu_backend", choices=["nltk", "sacrebleu"], default="nltk",
                   help="the paper computed BLEU with nltk; sacrebleu is provided for comparison")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--keep_checkpoint", action="store_true")
    p.add_argument("--report_to", default="none")
    return p.parse_args()


def read_parallel(args):
    """Return ``{split: (sources, targets)}`` for train / valid / test."""
    s, t = args.src_lang, args.tgt_lang
    out = {}
    if args.data_dir:
        for split, names in {"train": ["train"], "valid": ["valid", "validation", "dev"], "test": ["test"]}.items():
            for name in names:
                src_path = os.path.join(args.data_dir, f"{name}.{s}")
                if os.path.exists(src_path):
                    with open(src_path, encoding="utf-8") as f:
                        src = [l.rstrip("\n") for l in f]
                    with open(os.path.join(args.data_dir, f"{name}.{t}"), encoding="utf-8") as f:
                        tgt = [l.rstrip("\n") for l in f]
                    if len(src) != len(tgt):
                        raise ValueError(f"{name}: {len(src)} source vs {len(tgt)} target lines")
                    out[split] = (src, tgt)
                    break
            if split not in out:
                raise FileNotFoundError(f"no {split} files for {s}-{t} in {args.data_dir}")
        return out
    if not args.dataset:
        raise SystemExit("give --data_dir or --dataset")
    from zipftok.data import load_hf_dataset

    ds = load_hf_dataset(args.dataset, args.dataset_config, split=None)
    for split, name in {"train": "train", "valid": "validation", "test": "test"}.items():
        tr = ds[name]["translation"]
        out[split] = ([x[s] for x in tr], [x[t] for x in tr])
    return out


def main():
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    import torch
    from datasets import Dataset
    from transformers import AutoModelForSeq2SeqLM, Trainer

    from zipftok.collators import Seq2SeqCollator
    from zipftok.hf_compat import trainer_kwargs, training_arguments
    from zipftok.tokenization import LANGUAGE_CODES, load_hf_tokenizer

    tokenizer = load_hf_tokenizer(args.model_dir)
    src_id = tokenizer.convert_tokens_to_ids(LANGUAGE_CODES.get(args.src_lang, args.src_lang))
    tgt_id = tokenizer.convert_tokens_to_ids(LANGUAGE_CODES.get(args.tgt_lang, args.tgt_lang))
    for lang, lid in ((args.src_lang, src_id), (args.tgt_lang, tgt_id)):
        if lid is None or lid == tokenizer.unk_token_id:
            raise ValueError(f"language code for {lang!r} is not in the tokenizer vocabulary")
    eos = tokenizer.eos_token_id
    data = read_parallel(args)

    def encode(texts, lang_id, max_len):
        ids = tokenizer(texts, add_special_tokens=False, return_attention_mask=False,
                        return_token_type_ids=False)["input_ids"]
        return [x[: max_len - 2] + [eos, lang_id] for x in ids]

    def to_ds(src, tgt):
        return Dataset.from_dict({"input_ids": encode(src, src_id, args.max_source_length),
                                  "labels": encode(tgt, tgt_id, args.max_target_length)})

    train_ds, valid_ds = to_ds(*data["train"]), to_ds(*data["valid"])
    logger.info("%s->%s: %d train / %d valid / %d test pairs", args.src_lang, args.tgt_lang,
                len(train_ds), len(valid_ds), len(data["test"][0]))

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, dropout=args.dropout)
    # mBART decodes from the target language id (passed explicitly to generate() below as well)
    model.generation_config.decoder_start_token_id = tgt_id

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    targs = training_arguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        label_smoothing_factor=args.label_smoothing,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=100,
        fp16=args.fp16,
        seed=args.seed,
        remove_unused_columns=False,
        report_to=args.report_to,
    )
    trainer = Trainer(**trainer_kwargs(tokenizer, model=model, args=targs, train_dataset=train_ds,
                                       eval_dataset=valid_ds, data_collator=Seq2SeqCollator(tokenizer.pad_token_id)))
    trainer.train()

    # beam-search decoding of the test set
    model = trainer.model
    model.eval()
    device = next(model.parameters()).device
    src_test, ref_test = data["test"]
    enc = encode(src_test, src_id, args.max_source_length)
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))  # length-sorted batches
    decoded = [None] * len(enc)
    with torch.no_grad():
        for b in range(0, len(order), args.generation_batch_size):
            idx = order[b:b + args.generation_batch_size]
            batch = Seq2SeqCollator(tokenizer.pad_token_id)([{"input_ids": enc[i]} for i in idx])
            out = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                decoder_start_token_id=tgt_id,
                eos_token_id=eos,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=args.num_beams,
                max_new_tokens=args.max_target_length,
                early_stopping=True,
            )
            texts = tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, t in zip(idx, texts):
                decoded[i] = t.strip()
    hyps = decoded
    bleu = corpus_bleu(hyps, ref_test, target_lang=args.tgt_lang, backend=args.bleu_backend)
    results = {
        "task": f"{args.src_lang}-{args.tgt_lang}", "seed": args.seed, "bleu": bleu, "score": bleu,
        "bleu_backend": args.bleu_backend, "num_beams": args.num_beams, "n_test": len(hyps),
    }
    if trainer.is_world_process_zero():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "test_hypotheses.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(hyps) + "\n")
        if args.keep_checkpoint:
            trainer.save_model(os.path.join(args.output_dir, "model"))
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        write_json(results, os.path.join(args.output_dir, "results.json"))
        logger.info("%s seed %d: BLEU %.2f", results["task"], args.seed, bleu)


if __name__ == "__main__":
    main()
