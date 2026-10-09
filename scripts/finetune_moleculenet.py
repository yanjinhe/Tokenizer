#!/usr/bin/env python
"""Fine-tune a SMILES BERT on MoleculeNet classification tasks and report ROC-AUC.

Datasets of Table 3 (CSV files from DeepChem, see scripts/prepare_data/download_moleculenet.py):

=======  ====================  ============  ==========================================
name     file                  SMILES col.   tasks
=======  ====================  ============  ==========================================
bbbp     BBBP.csv              smiles        p_np
tox21    tox21.csv.gz          smiles        12 nuclear-receptor / stress-response assays
sider    sider.csv.gz          smiles        27 side-effect categories
clintox  clintox.csv.gz        smiles        FDA_APPROVED, CT_TOX
hiv      HIV.csv               smiles        HIV_active
bace     bace.csv              mol           Class
=======  ====================  ============  ==========================================

Multi-task datasets are trained with a masked binary cross-entropy (missing labels
are ignored) and scored with the mean ROC-AUC over tasks. Molecules are split
80/10/10 by Bemis-Murcko scaffold (requires RDKit; ``--split random`` otherwise).
The checkpoint with the best validation ROC-AUC is evaluated on the test split.

Example::

    python scripts/finetune_moleculenet.py --model_dir outputs/smiles/vocab_3000/pretrain \
        --data_dir data/moleculenet --task bbbp --seed 1 \
        --output_dir outputs/smiles/vocab_3000/finetune/bbbp/seed_1
"""

import argparse
import logging
import os
import shutil
from collections import defaultdict

import numpy as np

import _bootstrap  # noqa: F401

from zipftok.metrics import multitask_roc_auc
from zipftok.utils import set_seed, setup_logging, write_json

logger = logging.getLogger("finetune_moleculenet")

TOX21_TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
               "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
DATASETS = {
    "bbbp": {"file": "BBBP.csv", "smiles": "smiles", "tasks": ["p_np"]},
    "tox21": {"file": "tox21.csv.gz", "smiles": "smiles", "tasks": TOX21_TASKS},
    "sider": {"file": "sider.csv.gz", "smiles": "smiles", "tasks": None},  # all non-SMILES columns
    "clintox": {"file": "clintox.csv.gz", "smiles": "smiles", "tasks": ["FDA_APPROVED", "CT_TOX"]},
    "hiv": {"file": "HIV.csv", "smiles": "smiles", "tasks": ["HIV_active"]},
    "bace": {"file": "bace.csv", "smiles": "mol", "tasks": ["Class"]},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--task", required=True, choices=sorted(DATASETS))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    p.add_argument("--split_seed", type=int, default=0, help="seed of the random split (scaffold split is deterministic)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=None, help="default: the pre-training sequence length")
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--num_train_epochs", type=float, default=10)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--keep_checkpoint", action="store_true")
    p.add_argument("--report_to", default="none")
    return p.parse_args()


def load_moleculenet(data_dir, name):
    import pandas as pd

    cfg = DATASETS[name]
    df = pd.read_csv(os.path.join(data_dir, cfg["file"]))
    tasks = cfg["tasks"] or [c for c in df.columns if c != cfg["smiles"]]
    smiles = df[cfg["smiles"]].astype(str).tolist()
    labels = df[tasks].astype(float).to_numpy()  # NaN = missing
    return smiles, labels, tasks


def scaffold_split(smiles, frac_train=0.8, frac_valid=0.1):
    """Deterministic Bemis-Murcko scaffold split (DeepChem ordering: largest scaffold sets first)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    groups = defaultdict(list)
    invalid = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            invalid.append(i)
            continue
        groups[MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)].append(i)
    sets = sorted(groups.values(), key=lambda x: (len(x), x[0]), reverse=True)
    n = len(smiles) - len(invalid)
    train_cutoff, valid_cutoff = frac_train * n, (frac_train + frac_valid) * n
    train, valid, test = [], [], []
    for g in sets:  # same rule as deepchem.splits.ScaffoldSplitter
        if len(train) + len(g) > train_cutoff:
            if len(train) + len(valid) + len(g) > valid_cutoff:
                test += g
            else:
                valid += g
        else:
            train += g
    return sorted(train), sorted(valid), sorted(test), invalid


def random_split(n, seed, frac_train=0.8, frac_valid=0.1):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(frac_train * n), int((frac_train + frac_valid) * n)
    return sorted(idx[:a].tolist()), sorted(idx[a:b].tolist()), sorted(idx[b:].tolist()), []


def main():
    setup_logging()
    args = parse_args()
    set_seed(args.seed)

    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, Trainer

    from zipftok.collators import MultiLabelCollator
    from zipftok.hf_compat import trainer_kwargs, training_arguments
    from zipftok.modeling import pretraining_max_length
    from zipftok.tokenization import load_hf_tokenizer

    class MaskedBCETrainer(Trainer):
        """Binary cross-entropy over the observed labels only (NaN = missing)."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            mask = ~torch.isnan(labels)
            if mask.any():
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[mask], labels[mask].to(logits.dtype), reduction="mean")
            else:  # every label of this batch is missing
                loss = logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

    smiles, labels, tasks = load_moleculenet(args.data_dir, args.task)
    if args.split == "scaffold":
        tr, va, te, invalid = scaffold_split(smiles)
    else:
        tr, va, te, invalid = random_split(len(smiles), args.split_seed)
    logger.info("%s: %d molecules, %d tasks, split %d/%d/%d (%d unparsable dropped)",
                args.task, len(smiles), len(tasks), len(tr), len(va), len(te), len(invalid))

    tokenizer = load_hf_tokenizer(args.model_dir)
    max_length = args.max_length or pretraining_max_length(args.model_dir)

    def to_ds(idx):
        enc = tokenizer([smiles[i] for i in idx], truncation=True, max_length=max_length)
        return Dataset.from_dict({"input_ids": enc["input_ids"], "labels": [labels[i].tolist() for i in idx]})

    data = {"train": to_ds(tr), "valid": to_ds(va), "test": to_ds(te)}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, num_labels=len(tasks), problem_type="multi_label_classification")

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        m = multitask_roc_auc(np.asarray(y, dtype=np.float64), probs)
        return {"roc_auc": m["roc_auc"], "score": m["score"], "n_valid_tasks": m["n_valid_tasks"]}

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    targs = training_arguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        logging_steps=20,
        fp16=args.fp16,
        seed=args.seed,
        remove_unused_columns=False,
        report_to=args.report_to,
    )
    trainer = MaskedBCETrainer(**trainer_kwargs(
        tokenizer, model=model, args=targs, train_dataset=data["train"], eval_dataset=data["valid"],
        data_collator=MultiLabelCollator(tokenizer.pad_token_id), compute_metrics=compute_metrics))
    trainer.train()
    valid = trainer.evaluate(eval_dataset=data["valid"], metric_key_prefix="valid")
    test_pred = trainer.predict(data["test"], metric_key_prefix="test")
    logits = test_pred.predictions[0] if isinstance(test_pred.predictions, tuple) else test_pred.predictions
    test = multitask_roc_auc(np.asarray(test_pred.label_ids, dtype=np.float64),
                             1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64))))
    results = {
        "task": args.task, "seed": args.seed, "split": args.split, "tasks": tasks, "max_length": max_length,
        "n_train": len(tr), "n_valid": len(va), "n_test": len(te),
        "valid": {k[6:]: v for k, v in valid.items() if k.startswith("valid_")},
        "test": test,
        "score": test["score"],  # ROC-AUC x 100, as in Table 3
    }
    if trainer.is_world_process_zero():
        if args.keep_checkpoint:
            trainer.save_model(os.path.join(args.output_dir, "model"))
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        write_json(results, os.path.join(args.output_dir, "results.json"))
        logger.info("%s seed %d: test ROC-AUC %.2f", args.task, args.seed, results["score"])


if __name__ == "__main__":
    main()
