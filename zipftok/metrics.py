"""Evaluation metrics used in the paper (Section 4.3).

* GLUE (WNLI excluded): CoLA -> Matthews correlation; MRPC and QQP -> mean of
  accuracy and F1; STS-B -> mean of Pearson and Spearman correlation; SST-2, MNLI,
  QNLI, RTE -> accuracy. The GLUE score is the average of the eight task scores.
* GUE (genomics): accuracy.
* MoleculeNet (chemistry): ROC-AUC (averaged over the sub-tasks of multi-task
  datasets such as Tox21, SIDER and ClinTox; missing labels are ignored).
* Translation: corpus BLEU.

All scores are reported on a 0-100 scale like the tables of the paper.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "GLUE_TASKS",
    "glue_task_metrics",
    "glue_score",
    "classification_metrics",
    "multitask_roc_auc",
    "corpus_bleu",
]

# task -> (text fields, number of labels, validation split(s))
GLUE_TASKS: Dict[str, dict] = {
    "cola": {"keys": ("sentence", None), "num_labels": 2, "eval_splits": ["validation"]},
    "sst2": {"keys": ("sentence", None), "num_labels": 2, "eval_splits": ["validation"]},
    "mrpc": {"keys": ("sentence1", "sentence2"), "num_labels": 2, "eval_splits": ["validation"]},
    "stsb": {"keys": ("sentence1", "sentence2"), "num_labels": 1, "eval_splits": ["validation"]},
    "qqp": {"keys": ("question1", "question2"), "num_labels": 2, "eval_splits": ["validation"]},
    "mnli": {"keys": ("premise", "hypothesis"), "num_labels": 3,
             "eval_splits": ["validation_matched", "validation_mismatched"]},
    "qnli": {"keys": ("question", "sentence"), "num_labels": 2, "eval_splits": ["validation"]},
    "rte": {"keys": ("sentence1", "sentence2"), "num_labels": 2, "eval_splits": ["validation"]},
}
GLUE_ORDER = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]


def _acc(preds, labels) -> float:
    preds, labels = np.asarray(preds), np.asarray(labels)
    return float(np.mean(preds == labels))


def glue_task_metrics(task: str, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """All metrics of a GLUE task plus ``score``, the paper's per-task number (0-100).

    ``predictions`` are logits (or regression outputs for STS-B).
    """
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import f1_score, matthews_corrcoef

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if task == "stsb":
        p = predictions.reshape(-1).astype(np.float64)
        pear = float(pearsonr(p, labels)[0])
        spear = float(spearmanr(p, labels)[0])
        return {"pearson": pear, "spearman": spear, "score": 100.0 * (pear + spear) / 2}
    preds = predictions.argmax(-1) if predictions.ndim > 1 else predictions
    acc = _acc(preds, labels)
    if task == "cola":
        mcc = float(matthews_corrcoef(labels, preds))
        return {"matthews_correlation": mcc, "accuracy": acc, "score": 100.0 * mcc}
    if task in ("mrpc", "qqp"):
        f1 = float(f1_score(labels, preds))
        return {"accuracy": acc, "f1": f1, "score": 100.0 * (acc + f1) / 2}
    return {"accuracy": acc, "score": 100.0 * acc}


def glue_score(task_scores: Dict[str, float], tasks: Sequence[str] = GLUE_ORDER) -> float:
    """Average of the per-task scores (the "Avg" column of Table 1)."""
    missing = [t for t in tasks if t not in task_scores]
    if missing:
        raise KeyError(f"missing GLUE tasks: {missing}")
    return float(np.mean([task_scores[t] for t in tasks]))


def classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Accuracy (the paper's GUE metric) plus macro-F1 and MCC for reference."""
    from sklearn.metrics import f1_score, matthews_corrcoef

    predictions = np.asarray(predictions)
    preds = predictions.argmax(-1) if predictions.ndim > 1 else predictions
    labels = np.asarray(labels)
    acc = _acc(preds, labels)
    return {
        "accuracy": acc,
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "matthews_correlation": float(matthews_corrcoef(labels, preds)),
        "score": 100.0 * acc,
    }


def multitask_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, object]:
    """Mean ROC-AUC over tasks; NaN labels are ignored, single-class tasks skipped."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_score = y_score.reshape(-1, 1)
    per_task: List[Optional[float]] = []
    for t in range(y_true.shape[1]):
        m = ~np.isnan(y_true[:, t])
        yt = y_true[m, t]
        if yt.size == 0 or np.unique(yt).size < 2:
            per_task.append(None)
            continue
        per_task.append(float(roc_auc_score(yt, y_score[m, t])))
    valid = [x for x in per_task if x is not None]
    mean = float(np.mean(valid)) if valid else float("nan")
    return {"roc_auc": mean, "per_task_roc_auc": per_task, "n_valid_tasks": len(valid), "score": 100.0 * mean}


def _nltk_tokenize(text: str, lang: str) -> List[str]:
    if lang.startswith("zh"):
        return [c for c in text if not c.isspace()]
    import re

    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def corpus_bleu(hypotheses: Sequence[str], references: Sequence[str], target_lang: str = "en",
                backend: str = "nltk") -> float:
    """Corpus BLEU (0-100).

    ``backend="sacrebleu"`` uses SacreBLEU's 13a tokenizer (``zh`` tokenizer for
    Chinese); ``backend="nltk"`` uses ``nltk.translate.bleu_score.corpus_bleu`` on a
    simple word/punctuation (characters for Chinese) tokenization.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    if backend == "sacrebleu":
        import sacrebleu

        tok = "zh" if target_lang.startswith("zh") else "13a"
        return float(sacrebleu.corpus_bleu(list(hypotheses), [list(references)], tokenize=tok).score)
    if backend == "nltk":
        from nltk.translate.bleu_score import SmoothingFunction
        from nltk.translate.bleu_score import corpus_bleu as nltk_bleu

        hyps = [_nltk_tokenize(h, target_lang) for h in hypotheses]
        refs = [[_nltk_tokenize(r, target_lang)] for r in references]
        return 100.0 * float(nltk_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1))
    raise ValueError(f"unknown BLEU backend {backend!r}")


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    v = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}
