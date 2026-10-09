#!/usr/bin/env python
"""Fine-tune a pre-trained BERT on GLUE (WNLI excluded) and report the paper's metrics.

CoLA: Matthews correlation; MRPC / QQP: mean of accuracy and F1; STS-B: mean of
Pearson and Spearman; SST-2, MNLI (matched), QNLI, RTE: accuracy. Scores are
computed on the GLUE validation sets. One run = one task and one seed; the
paper averages three seeds (see scripts/aggregate_results.py).

Example::

    python scripts/finetune_glue.py --model_dir outputs/text/vocab_30000/pretrain --task rte --seed 1 \
        --output_dir outputs/text/vocab_30000/finetune/rte/seed_1
"""

import argparse
import logging
import os

import numpy as np

import _bootstrap  # noqa: F401

from zipftok.metrics import GLUE_ORDER, GLUE_TASKS, glue_task_metrics
from zipftok.utils import set_seed, setup_logging, write_json

logger = logging.getLogger("finetune_glue")

# Default epochs / learning rate per task (standard BERT fine-tuning settings).
DEFAULTS = {
    "cola": dict(epochs=3, lr=2e-5), "sst2": dict(epochs=3, lr=2e-5), "mrpc": dict(epochs=5, lr=2e-5),
    "stsb": dict(epochs=5, lr=2e-5), "qqp": dict(epochs=3, lr=2e-5), "mnli": dict(epochs=3, lr=2e-5),
    "qnli": dict(epochs=3, lr=2e-5), "rte": dict(epochs=5, lr=2e-5),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--task", required=True, choices=GLUE_ORDER)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--glue_dataset", default="nyu-mll/glue", help="HF id of GLUE ('glue' on old datasets versions)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--num_train_epochs", type=float, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--keep_checkpoint", action="store_true", help="save the fine-tuned model")
    p.add_argument("--report_to", default="none")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer

    from zipftok.data import load_hf_dataset
    from zipftok.hf_compat import trainer_kwargs, training_arguments
    from zipftok.tokenization import load_hf_tokenizer

    task = args.task
    spec = GLUE_TASKS[task]
    lr = args.learning_rate or DEFAULTS[task]["lr"]
    epochs = args.num_train_epochs or DEFAULTS[task]["epochs"]

    tokenizer = load_hf_tokenizer(args.model_dir)
    raw = load_hf_dataset(args.glue_dataset, task, split=None)
    k1, k2 = spec["keys"]

    def preprocess(batch):
        texts = (batch[k1],) if k2 is None else (batch[k1], batch[k2])
        enc = tokenizer(*texts, truncation=True, max_length=args.max_seq_length)
        enc["labels"] = [float(x) for x in batch["label"]] if task == "stsb" else batch["label"]
        return enc

    columns = raw["train"].column_names
    data = {split: raw[split].map(preprocess, batched=True, remove_columns=columns)
            for split in ["train"] + spec["eval_splits"]}

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=spec["num_labels"])

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        return {k: v for k, v in glue_task_metrics(task, preds, labels).items()}

    targs = training_arguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=args.fp16,
        seed=args.seed,
        report_to=args.report_to,
    )
    trainer = Trainer(**trainer_kwargs(
        tokenizer, model=model, args=targs, train_dataset=data["train"], eval_dataset=data[spec["eval_splits"][0]],
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics))
    trainer.train()

    results = {"task": task, "seed": args.seed, "learning_rate": lr, "epochs": epochs}
    for split in spec["eval_splits"]:
        m = trainer.evaluate(eval_dataset=data[split], metric_key_prefix=split)
        results[split] = {k[len(split) + 1:]: v for k, v in m.items() if k.startswith(split + "_")}
    # the paper reports MNLI-matched accuracy
    results["score"] = results[spec["eval_splits"][0]]["score"]
    epoch_log = [x for x in trainer.state.log_history if any(k.endswith("_score") for k in x)]
    results["per_epoch"] = epoch_log
    if args.keep_checkpoint:
        trainer.save_model(os.path.join(args.output_dir, "model"))
    if trainer.is_world_process_zero():
        write_json(results, os.path.join(args.output_dir, "results.json"))
        logger.info("%s seed %d: score %.2f", task, args.seed, results["score"])


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
