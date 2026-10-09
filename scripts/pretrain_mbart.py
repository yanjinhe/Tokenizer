#!/usr/bin/env python
"""Multilingual denoising pre-training of mBART on WMT (Section 4.1).

For one language pair (De-En, Fr-En or Zh-En) both sides of the WMT training
data are used as monolingual text. Consecutive sentences of one language are
packed into documents of at most ``--max_length`` tokens. Every time a document
is drawn, its sentences are permuted and 35% of its words are masked with
Poisson(3.5)-length spans (one ``<mask>`` per span); the model reconstructs the
original document. Encoder input: ``noised </s> <lang>``; decoder:
``<lang> document </s>``.

Example::

    torchrun --nproc_per_node 8 scripts/pretrain_mbart.py \
        --tokenizer_dir outputs/translation_de-en/tokenizers/vocab_60000 \
        --model_config configs/models/mbart_6x6_1024.json \
        --dataset wmt16 --dataset_config de-en --languages de en \
        --output_dir outputs/translation_de-en/vocab_60000/pretrain --max_steps 100000 --fp16
"""

import argparse
import logging
import math
import os

import numpy as np

import _bootstrap  # noqa: F401

from zipftok.utils import set_seed, setup_logging, write_json

logger = logging.getLogger("pretrain_mbart")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--model_config", default="configs/models/mbart_6x6_1024.json")
    p.add_argument("--languages", nargs=2, required=True, help="e.g. de en")
    p.add_argument("--dataset", default="wmt16", help="HF dataset with a `translation` field")
    p.add_argument("--dataset_config", default=None, help="e.g. de-en (default: '<l1>-<l2>')")
    p.add_argument("--train_files", nargs="*", default=None,
                   help="instead of --dataset: plain-text files LANG=PATH, one sentence per line")
    p.add_argument("--max_sentences_per_language", type=int, default=None)
    p.add_argument("--n_validation_docs", type=int, default=2000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--mask_ratio", type=float, default=0.35)
    p.add_argument("--poisson_lambda", type=float, default=3.5)
    p.add_argument("--no_permute", action="store_true")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--preprocessing_num_workers", type=int, default=8)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=10000)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=5000)
    p.add_argument("--save_steps", type=int, default=10000)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=0,
                   help="keep 0 unless you want per-worker noise streams (the collator holds the RNG)")
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--ddp_timeout", type=int, default=21600,
                   help="seconds other ranks wait while rank 0 prepares the data (first run can be long)")
    p.add_argument("--report_to", default="none")
    return p.parse_args()


def line_generator(path, limit):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                yield {"text": line.strip()}


def load_sentences(args, lang):
    from datasets import Dataset

    from zipftok.data import load_hf_dataset

    if args.train_files:
        files = dict(x.split("=", 1) for x in args.train_files)
        return Dataset.from_generator(line_generator, cache_dir=args.cache_dir,
                                      gen_kwargs={"path": files[lang], "limit": args.max_sentences_per_language})
    cfg = args.dataset_config or f"{args.languages[0]}-{args.languages[1]}"
    ds = load_hf_dataset(args.dataset, cfg, split="train", cache_dir=args.cache_dir)
    if args.max_sentences_per_language:
        ds = ds.select(range(min(len(ds), args.max_sentences_per_language)))
    return ds.map(lambda b: {"text": [t[lang] for t in b["translation"]]}, batched=True,
                  remove_columns=ds.column_names, num_proc=args.preprocessing_num_workers)


def build_documents(args, tokenizer, lang):
    from zipftok.tokenization import LANGUAGE_CODES

    lang_id = tokenizer.convert_tokens_to_ids(LANGUAGE_CODES.get(lang, lang))
    if lang_id is None or lang_id == tokenizer.unk_token_id:
        raise ValueError(f"language code for {lang!r} is not in the tokenizer")
    body = args.max_length - 2
    sents = load_sentences(args, lang)

    def tok_fn(batch):
        ids = tokenizer(batch["text"], add_special_tokens=False, return_attention_mask=False,
                        return_token_type_ids=False)["input_ids"]
        return {"ids": [x[:body] for x in ids]}

    sents = sents.map(tok_fn, batched=True, remove_columns=["text"], num_proc=args.preprocessing_num_workers,
                      desc=f"tokenizing {lang}")

    def pack(batch):
        docs, lens = [], []
        cur, cur_lens = [], []
        for ids in batch["ids"]:
            if not ids:
                continue
            if cur and len(cur) + len(ids) > body:
                docs.append(cur)
                lens.append(cur_lens)
                cur, cur_lens = [], []
            cur = cur + ids
            cur_lens.append(len(ids))
        if cur:
            docs.append(cur)
            lens.append(cur_lens)
        return {"input_ids": docs, "sentence_lengths": lens, "lang_id": [lang_id] * len(docs)}

    return sents.map(pack, batched=True, batch_size=1000, remove_columns=["ids"],
                     num_proc=args.preprocessing_num_workers, desc=f"packing {lang} documents")


def main():
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    from datasets import concatenate_datasets
    from transformers import Trainer

    from zipftok.collators import MBartDenoisingCollator
    from zipftok.hf_compat import trainer_kwargs, training_arguments
    from zipftok.modeling import build_mbart, count_parameters
    from zipftok.tokenization import load_hf_tokenizer

    tokenizer = load_hf_tokenizer(args.tokenizer_dir)
    model = build_mbart(args.model_config, tokenizer)
    n_params = count_parameters(model)
    logger.info("vocab size %d, %.1fM parameters", len(tokenizer), n_params / 1e6)

    targs = training_arguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="polynomial",
        max_steps=args.max_steps,
        label_smoothing_factor=args.label_smoothing,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_timeout=args.ddp_timeout,
        report_to=args.report_to,
    )

    with targs.main_process_first(desc="building documents"):
        docs = concatenate_datasets([build_documents(args, tokenizer, l) for l in args.languages])
        docs = docs.shuffle(seed=args.seed)
        n_valid = min(args.n_validation_docs, max(1, len(docs) // 100))
        valid_ds = docs.select(range(n_valid))
        train_ds = docs.select(range(n_valid, len(docs)))
    logger.info("train documents: %d, validation documents: %d", len(train_ds), len(valid_ds))

    # whole-word masking: byte-level BPE tokens starting with "Ġ" begin a word
    vocab_tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    word_start = np.array([t is not None and t.startswith("Ġ") for t in vocab_tokens], dtype=bool)
    collator = MBartDenoisingCollator(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id,
        word_start=word_start,
        mask_ratio=args.mask_ratio,
        poisson_lambda=args.poisson_lambda,
        permute=not args.no_permute,
        max_length=args.max_length,
        seed=args.seed + targs.process_index,
    )
    trainer = Trainer(**trainer_kwargs(tokenizer, model=model, args=targs, train_dataset=train_ds,
                                       eval_dataset=valid_ds, data_collator=collator))
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)  # also saves the tokenizer (main process only)
    metrics = trainer.evaluate()
    if trainer.is_world_process_zero():
        out = {
            "vocab_size": len(tokenizer),
            "n_parameters": n_params,
            "train_loss": train_result.training_loss,
            "eval_loss": metrics.get("eval_loss"),
            "perplexity": math.exp(metrics["eval_loss"]) if metrics.get("eval_loss", 1e9) < 50 else float("inf"),
            "args": vars(args),
        }
        write_json(out, os.path.join(args.output_dir, "pretrain_results.json"))
        logger.info("done: %s", {k: v for k, v in out.items() if k != "args"})


if __name__ == "__main__":
    main()
