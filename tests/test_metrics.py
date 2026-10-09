import csv
import os
import unittest

import numpy as np

try:  # works with `unittest discover -s tests`, `python -m unittest tests.x` and pytest
    from . import _common  # noqa: F401
except ImportError:
    import _common  # noqa: F401
ROOT = _common.ROOT
from zipftok.metrics import (GLUE_ORDER, classification_metrics, glue_score, glue_task_metrics,
                             multitask_roc_auc)


class TestGlue(unittest.TestCase):
    def test_mrpc_score_is_mean_of_acc_and_f1(self):
        labels = np.array([1, 0, 1, 1, 0, 1])
        logits = np.array([[0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], dtype=float)
        m = glue_task_metrics("mrpc", logits, labels)
        preds = logits.argmax(-1)
        acc = (preds == labels).mean()
        tp = ((preds == 1) & (labels == 1)).sum()
        f1 = 2 * tp / (2 * tp + ((preds == 1) & (labels == 0)).sum() + ((preds == 0) & (labels == 1)).sum())
        self.assertAlmostEqual(m["score"], 100 * (acc + f1) / 2)

    def test_stsb_and_cola(self):
        y = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
        m = glue_task_metrics("stsb", y[:, None] * 2 + 1, y)
        self.assertAlmostEqual(m["score"], 100.0)
        labels = np.array([0, 1, 0, 1])
        m = glue_task_metrics("cola", np.eye(2)[labels], labels)
        self.assertAlmostEqual(m["score"], 100.0)

    def test_glue_score_reproduces_table1(self):
        with open(os.path.join(ROOT, "results", "paper", "table1_glue.csv")) as f:
            for row in csv.DictReader(f):
                avg = glue_score({t: float(row[t]) for t in GLUE_ORDER})
                self.assertLess(abs(avg - float(row["avg"])), 0.006, row["vocab_size"])

    def test_glue_score_needs_all_tasks(self):
        with self.assertRaises(KeyError):
            glue_score({"cola": 1.0})


class TestOther(unittest.TestCase):
    def test_classification(self):
        m = classification_metrics(np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 3.0]]), np.array([0, 1, 0]))
        self.assertAlmostEqual(m["accuracy"], 2 / 3)
        self.assertAlmostEqual(m["score"], 200 / 3)

    def test_multitask_roc_auc_ignores_missing(self):
        y = np.array([[1, 0], [0, np.nan], [1, 1], [0, np.nan]], dtype=float)
        s = np.array([[0.9, 0.2], [0.1, 0.99], [0.8, 0.7], [0.3, 0.0]])
        m = multitask_roc_auc(y, s)
        self.assertEqual(m["n_valid_tasks"], 2)
        self.assertAlmostEqual(m["roc_auc"], 1.0)

    def test_single_class_task_skipped(self):
        y = np.array([[1, 1], [0, 1], [1, 1]], dtype=float)
        s = np.array([[0.9, 0.5], [0.2, 0.4], [0.7, 0.1]])
        m = multitask_roc_auc(y, s)
        self.assertEqual(m["n_valid_tasks"], 1)
        self.assertIsNone(m["per_task_roc_auc"][1])


class TestPaperTables(unittest.TestCase):
    def test_avg_columns_are_task_means(self):
        cases = {
            "table1_glue.csv": GLUE_ORDER,
            "table2_gue.csv": ["cpd", "h_tfp1", "h_tfp2", "pd", "m_tfp1", "m_tfp2", "emp_h3", "emp_h4"],
            "table3_moleculenet.csv": ["bbbp", "tox21", "sider", "clintox", "hiv", "bace"],
        }
        for name, cols in cases.items():
            with open(os.path.join(ROOT, "results", "paper", name)) as f:
                for row in csv.DictReader(f):
                    mean = np.mean([float(row[c]) for c in cols])
                    self.assertLess(abs(mean - float(row["avg"])), 0.006, (name, row["vocab_size"]))


if __name__ == "__main__":
    unittest.main()
