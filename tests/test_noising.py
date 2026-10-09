import unittest

import numpy as np

try:  # works with `unittest discover -s tests`, `python -m unittest tests.x` and pytest
    from . import _common  # noqa: F401
except ImportError:
    import _common  # noqa: F401
from zipftok.noising import permute_sentences, text_infilling, word_starts_from_tokens

MASK = -1


class TestNoising(unittest.TestCase):
    def test_permutation(self):
        rng = np.random.default_rng(0)
        sents = [[1, 2], [3], [4, 5, 6], [7]]
        out = permute_sentences(sents, rng)
        self.assertEqual(sorted(map(tuple, out)), sorted(map(tuple, sents)))

    def test_mask_ratio_and_spans(self):
        rng = np.random.default_rng(0)
        ids = list(range(1000, 3000))
        kept = []
        for _ in range(20):
            out = text_infilling(ids, MASK, rng, mask_ratio=0.35, poisson_lambda=3.5)
            kept.append(sum(1 for t in out if t != MASK) / len(ids))
            # surviving tokens keep their original order
            surv = [t for t in out if t != MASK]
            self.assertEqual(surv, sorted(surv))
            # spans are collapsed to one mask token, so the output is much shorter
            self.assertLess(len(out), len(ids))
        # overlapping spans can mask slightly less than 35%
        self.assertGreater(np.mean(kept), 0.62)
        self.assertLess(np.mean(kept), 0.75)

    def test_whole_word_masking(self):
        rng = np.random.default_rng(1)
        # 300 words of 3 tokens each: word w = tokens (3w, 3w+1, 3w+2)
        ids = list(range(900))
        ws = np.array([i % 3 == 0 for i in range(900)])
        out = text_infilling(ids, MASK, rng, word_starts=ws)
        surv = [t for t in out if t != MASK]
        words = {}
        for t in surv:
            words.setdefault(t // 3, []).append(t)
        self.assertTrue(all(len(v) == 3 for v in words.values()))  # words are kept or masked as a whole

    def test_deterministic(self):
        a = text_infilling(list(range(100)), MASK, np.random.default_rng(5))
        b = text_infilling(list(range(100)), MASK, np.random.default_rng(5))
        self.assertEqual(a, b)

    def test_edge_cases(self):
        rng = np.random.default_rng(0)
        self.assertEqual(text_infilling([], MASK, rng), [])
        self.assertEqual(text_infilling([1, 2, 3], MASK, rng, mask_ratio=0.0), [1, 2, 3])
        out = text_infilling([1], MASK, rng)
        self.assertTrue(len(out) >= 1)

    def test_word_starts_from_tokens(self):
        ws = word_starts_from_tokens(["the", "Ġcat", "s", "Ġsat"])
        self.assertEqual(ws.tolist(), [True, True, False, True])


if __name__ == "__main__":
    unittest.main()
