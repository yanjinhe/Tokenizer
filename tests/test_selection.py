import csv
import os
import unittest

try:  # works with `unittest discover -s tests`, `python -m unittest tests.x` and pytest
    from . import _common  # noqa: F401
except ImportError:
    import _common  # noqa: F401
ROOT = _common.ROOT
from zipftok.selection import select_from_curve, vocab_schedule, zipf_guided_selection


def paper_curve(name):
    with open(os.path.join(ROOT, "results", "paper", name)) as f:
        rows = list(csv.DictReader(f))
    return [int(r["vocab_size"]) for r in rows], [float(r["r2"]) for r in rows], [float(r["avg"]) for r in rows]


class TestSchedule(unittest.TestCase):
    def test_step(self):
        self.assertEqual(list(vocab_schedule(500, 2000, step=500)), [500, 1000, 1500, 2000])

    def test_growth(self):
        self.assertEqual(list(vocab_schedule(1000, 3000, growth=1.5)), [1000, 1500, 2250])

    def test_sizes(self):
        self.assertEqual(list(vocab_schedule(1000, 5000, sizes=[5000, 500, 2000, 9000])), [2000, 5000])

    def test_exactly_one(self):
        with self.assertRaises(ValueError):
            list(vocab_schedule(1, 10, step=1, growth=2.0))


class TestSelection(unittest.TestCase):
    def test_stops_after_patience(self):
        scores = {1: 0.50, 2: 0.60, 3: 0.70, 4: 0.705, 5: 0.708, 6: 0.99}
        calls = []

        def f(v):
            calls.append(v)
            return scores[v]

        res = zipf_guided_selection(f, [1, 2, 3, 4, 5, 6], epsilon=0.01, patience=2)
        self.assertEqual(res.optimal_vocab_size, 3)
        self.assertAlmostEqual(res.best_score, 0.70)
        self.assertTrue(res.stopped_early)
        self.assertEqual(calls, [1, 2, 3, 4, 5])  # never evaluates past the stopping point
        self.assertEqual([h.stagnation for h in res.history], [0, 0, 0, 1, 2])

    def test_counter_resets_on_improvement(self):
        res = select_from_curve([1, 2, 3, 4, 5], [0.5, 0.505, 0.6, 0.601, 0.602], epsilon=0.01, patience=2)
        self.assertEqual(res.optimal_vocab_size, 3)
        self.assertEqual([h.stagnation for h in res.history], [0, 1, 0, 1, 2])

    def test_small_gains_do_not_move_vopt(self):
        # monotone but saturating curve: gains below epsilon are not "meaningful"
        res = select_from_curve([1, 2, 3, 4, 5, 6], [0.5, 0.7, 0.8, 0.8004, 0.8008, 0.8009],
                                epsilon=0.001, patience=3)
        self.assertEqual(res.optimal_vocab_size, 3)

    def test_schedule_exhausted(self):
        res = select_from_curve([1, 2, 3], [0.1, 0.2, 0.3], epsilon=0.01, patience=2)
        self.assertFalse(res.stopped_early)
        self.assertEqual(res.optimal_vocab_size, 3)

    def test_return_stop_vocabulary(self):
        res = select_from_curve([1, 2, 3, 4, 5], [0.5, 0.6, 0.7, 0.701, 0.702], epsilon=0.01, patience=2,
                                return_vocab="stop")
        self.assertEqual(res.optimal_vocab_size, 5)
        self.assertEqual(res.best_vocab_size, 3)

    def test_unsorted_input_is_sorted(self):
        res = select_from_curve([3, 1, 2], [0.3, 0.1, 0.2], epsilon=0.01, patience=2)
        self.assertEqual([h.vocab_size for h in res.history], [1, 2, 3])

    def test_rejects_non_increasing_schedule(self):
        with self.assertRaises(ValueError):
            zipf_guided_selection(lambda v: 0.0, [2, 1])

    def test_paper_curves_with_default_settings(self):
        # The stopping rule with the script defaults (epsilon=1e-3, N=2) applied to the R^2 columns
        # reported in Tables 1-3. (Best average score in the paper: 30K, 4K and 3K.)
        expected = {"table1_glue.csv": 40000, "table2_gue.csv": 4000, "table3_moleculenet.csv": 3000}
        for name, v_opt in expected.items():
            sizes, r2, _ = paper_curve(name)
            res = select_from_curve(sizes, r2, epsilon=1e-3, patience=2)
            self.assertEqual(res.optimal_vocab_size, v_opt, name)


if __name__ == "__main__":
    unittest.main()
