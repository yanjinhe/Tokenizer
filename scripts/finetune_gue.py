#!/usr/bin/env python
"""Fine-tune a DNA BERT on GUE tasks (DNABERT-2 benchmark) and report test accuracy.

Tasks of Table 2 and their GUE directories:

=========  ======================  =================================
name       GUE path                description
=========  ======================  =================================
cpd        prom/prom_core_all      core promoter detection (human)
h_tfp1     tf/0                    transcription factor prediction (human), set 0
h_tfp2     tf/1                    transcription factor prediction (human), set 1
pd         prom/prom_300_all       promoter detection (human)
m_tfp1     mouse/0                 transcription factor prediction (mouse), set 0
m_tfp2     mouse/1                 transcription factor prediction (mouse), set 1
emp_h3     EMP/H3                  epigenetic marks prediction (yeast), H3
emp_h4     EMP/H4                  epigenetic marks prediction (yeast), H4
=========  ======================  =================================

Each GUE directory holds ``train.csv``, ``dev.csv`` and ``test.csv`` with columns
``sequence,label``. The checkpoint with the best dev accuracy is evaluated on test.
Any other GUE directory can be passed with ``--task_path``.

Example::

    python scripts/finetune_gue.py --model_dir outputs/dna/vocab_4000/pretrain --gue_dir data/GUE \
        --task emp_h3 --seed 1 --output_dir outputs/dna/vocab_4000/finetune/emp_h3/seed_1
"""

import argparse
import logging
import os
import shutil

import numpy as np

import _bootstrap  # noqa: F401

from zipftok.metrics import classification_metrics
from zipftok.utils import set_seed, setup_logging, write_json

logger = logging.getLogger("finetune_gue")

GUE_TASKS = {
    "cpd": "prom/prom_core_all",
    "h_tfp1": "tf/0",
    "h_tfp2": "tf/1",
    "pd": "prom/prom_300_all",
    "m_tfp1": "mouse/0",
    "m_tfp2": "mouse/1",
    "emp_h3": "EMP/H3",
    "emp_h4": "EMP/H4",
}
# training epochs per task family (as in the DNABERT-2 fine-tuning scripts)
EPOCHS = {"prom": 4, "tf": 3, "mouse": 5, "EMP": 3}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--gue_dir", required=True, help="root of the extracted GUE benchmark")
    p.add_argument("--task", default=None, choices=sorted(GUE_TASKS))
    p.add_argument("--task_path", default=None, help="GUE sub-directory, e.g. EMP/H3K4me3 (overrides --task)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=None,
                   help="max tokens per sequence (default: 99th percentile of train lengths, capped at the "
                        "pre-training sequence length)")
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--num_train_epochs", type=float, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--keep_checkpoint", action="store_true")
    p.add_argument("--report_to", default="none")
    return p.parse_args()


def load_split(path):
    import pandas as pd

    df = pd.read_csv(path)
    seq_col = "sequence" if "sequence" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[-1]
    return [s.upper().strip() for s in df[seq_col].astype(str)], df[label_col].astype(int).tolist()


def main():
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer

    from zipftok.hf_compat import trainer_kwargs, training_arguments
    from zipftok.modeling import pretraining_max_length
    from zipftok.tokenization import load_hf_tokenizer

    if args.task_path is None and args.task is None:
        raise SystemExit("give --task or --task_path")
    rel = args.task_path or GUE_TASKS[args.task]
    name = args.task or rel.replace("/", "_")
    task_dir = os.path.join(args.gue_dir, rel)
    epochs = args.num_train_epochs or EPOCHS.get(rel.split("/")[0], 3)

    tokenizer = load_hf_tokenizer(args.model_dir)
    splits = {s: load_split(os.path.join(task_dir, f"{s}.csv")) for s in ("train", "dev", "test")}
    num_labels = len(set(splits["train"][1]))

    max_length = args.max_length
    if max_length is None:
        sample = splits["train"][0][:5000]
        lens = [len(x) for x in tokenizer(sample, add_special_tokens=True)["input_ids"]]
        max_length = int(min(pretraining_max_length(args.model_dir), np.percentile(lens, 99)))
    logger.info("task %s (%s): %d labels, max_length %d tokens, %d epochs", name, rel, num_labels, max_length, epochs)

    def to_ds(seqs, labels):
        enc = tokenizer(seqs, truncation=True, max_length=max_length)
        enc["labels"] = labels
        return Dataset.from_dict(dict(enc))

    data = {s: to_ds(*v) for s, v in splits.items()}
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=num_labels)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        return classification_metrics(preds, labels)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    targs = training_arguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=args.fp16,
        seed=args.seed,
        report_to=args.report_to,
    )
    trainer = Trainer(**trainer_kwargs(
        tokenizer, model=model, args=targs, train_dataset=data["train"], eval_dataset=data["dev"],
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics))
    trainer.train()
    dev = trainer.evaluate(eval_dataset=data["dev"], metric_key_prefix="dev")
    test = trainer.evaluate(eval_dataset=data["test"], metric_key_prefix="test")
    results = {
        "task": name, "gue_path": rel, "seed": args.seed, "max_length": max_length, "epochs": epochs,
        "dev": {k[4:]: v for k, v in dev.items() if k.startswith("dev_")},
        "test": {k[5:]: v for k, v in test.items() if k.startswith("test_")},
    }
    results["score"] = results["test"]["score"]  # accuracy x 100, as in Table 2
    if trainer.is_world_process_zero():
        if args.keep_checkpoint:
            trainer.save_model(os.path.join(args.output_dir, "model"))
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        write_json(results, os.path.join(args.output_dir, "results.json"))
        logger.info("%s seed %d: test accuracy %.2f", name, args.seed, results["score"])


if __name__ == "__main__":
    main()
