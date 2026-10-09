#!/usr/bin/env python
"""Pre-train a BERT model with masked language modelling (Section 4.1).

Used for all three encoder-only settings of the paper:

* NLP       - OpenWebText + BookCorpus, evaluated on GLUE;
* genomics  - DNA sequences following DNABERT-2, evaluated on GUE;
* chemistry - the first 5M SMILES of ZINC20, evaluated on MoleculeNet.

The architecture is fixed by ``--model_config`` and only the vocabulary changes
between runs. Texts are tokenized without special tokens, concatenated and cut
into blocks of ``max_seq_length - 2`` tokens wrapped in ``[CLS] ... [SEP]``
(``--line_by_line`` keeps one example per line instead, e.g. one molecule).

Example (8 GPUs)::

    torchrun --nproc_per_node 8 scripts/pretrain_mlm.py \
        --tokenizer_dir outputs/text/tokenizers/vocab_30000 \
        --model_config configs/models/bert_base.json \
        --train_corpus hf:bookcorpus::train:text --train_corpus hf:Skylion007/openwebtext::train:text \
        --output_dir outputs/text/vocab_30000/pretrain --max_seq_length 128 \
        --per_device_train_batch_size 32 --learning_rate 5e-4 --max_steps 100000 --fp16
"""

import argparse
import logging
import math
import os

import _bootstrap  # noqa: F401

from zipftok.utils import set_seed, setup_logging, write_json

logger = logging.getLogger("pretrain_mlm")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--model_config", default="configs/models/bert_base.json")
    p.add_argument("--train_corpus", action="append", required=True, help="corpus spec (repeatable)")
    p.add_argument("--validation_corpus", action="append", default=None)
    p.add_argument("--text_column", default="text")
    p.add_argument("--max_train_texts", type=int, default=None, help="max texts per training corpus spec")
    p.add_argument("--max_validation_texts", type=int, default=10000)
    p.add_argument("--validation_split_percentage", type=float, default=0.5,
                   help="held-out share of the training texts when no --validation_corpus is given")
    p.add_argument("--fasta_chunk", type=int, default=None)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--line_by_line", action="store_true")
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--preprocessing_num_workers", type=int, default=8)
    # optimisation
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta2", type=float, default=0.98)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--num_train_epochs", type=float, default=1.0, help="used only if --max_steps <= 0")
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=5000)
    p.add_argument("--save_steps", type=int, default=10000)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--ddp_timeout", type=int, default=21600,
                   help="seconds other ranks wait while rank 0 prepares the data (first run can be long)")
    p.add_argument("--report_to", default="none")
    return p.parse_args()


def corpus_generator(specs, limit, text_column, fasta_chunk):
    """Module-level generator so that the datasets cache key only depends on the corpus arguments
    (and the raw corpus is materialised once, not once per vocabulary size)."""
    from zipftok.data import iter_corpora

    for t in iter_corpora(specs, text_column=text_column, max_texts_per_corpus=limit, fasta_chunk=fasta_chunk):
        yield {"text": t}


def build_datasets(args, tokenizer):
    from datasets import Dataset

    def raw(specs, limit):
        return Dataset.from_generator(
            corpus_generator, cache_dir=args.cache_dir,
            gen_kwargs={"specs": list(specs), "limit": limit, "text_column": args.text_column,
                        "fasta_chunk": args.fasta_chunk})

    train = raw(args.train_corpus, args.max_train_texts)
    if args.validation_corpus:
        valid = raw(args.validation_corpus, args.max_validation_texts)
    else:
        n_valid = max(1, min(args.max_validation_texts, int(len(train) * args.validation_split_percentage / 100)))
        split = train.train_test_split(test_size=n_valid, seed=args.seed)
        train, valid = split["train"], split["test"]

    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    block = args.max_seq_length - 2

    if args.line_by_line:
        def tok_fn(batch):
            return tokenizer(batch["text"], truncation=True, max_length=args.max_seq_length,
                             return_special_tokens_mask=True)

        def prep(ds):
            return ds.map(tok_fn, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=["text"],
                          desc="tokenizing")
    else:
        def tok_fn(batch):
            return {"ids": tokenizer(batch["text"], add_special_tokens=False,
                                     return_attention_mask=False, return_token_type_ids=False)["input_ids"]}

        def group_fn(batch):
            flat = [i for ids in batch["ids"] for i in ids]
            n = (len(flat) // block) * block
            blocks = [flat[i:i + block] for i in range(0, n, block)]
            input_ids = [[cls_id] + b + [sep_id] for b in blocks]
            return {
                "input_ids": input_ids,
                "token_type_ids": [[0] * len(x) for x in input_ids],
                "attention_mask": [[1] * len(x) for x in input_ids],
                "special_tokens_mask": [[1] + [0] * (len(x) - 2) + [1] for x in input_ids],
            }

        def prep(ds):
            ds = ds.map(tok_fn, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=["text"],
                        desc="tokenizing")
            return ds.map(group_fn, batched=True, batch_size=1000, num_proc=args.preprocessing_num_workers,
                          remove_columns=["ids"], desc=f"grouping into blocks of {args.max_seq_length}")

    return prep(train), prep(valid)


def main():
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    from transformers import DataCollatorForLanguageModeling, Trainer

    from zipftok.hf_compat import trainer_kwargs, training_arguments
    from zipftok.modeling import build_bert_mlm, count_parameters
    from zipftok.tokenization import load_hf_tokenizer

    tokenizer = load_hf_tokenizer(args.tokenizer_dir)
    model = build_bert_mlm(args.model_config, tokenizer)
    n_params = count_parameters(model)
    logger.info("vocab size %d, %.1fM parameters", len(tokenizer), n_params / 1e6)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability,
                                               pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None)
    targs = training_arguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=args.adam_beta2,
        adam_epsilon=1e-6,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_train_epochs,
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
    with targs.main_process_first(desc="building datasets"):
        train_ds, valid_ds = build_datasets(args, tokenizer)
    logger.info("train examples: %d, validation examples: %d", len(train_ds), len(valid_ds))
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
