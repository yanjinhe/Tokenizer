import copy
import unittest

try:  # works with `unittest discover -s tests`, `python -m unittest tests.x` and pytest
    from . import _common  # noqa: F401
except ImportError:
    import _common  # noqa: F401
from ref_bpe import encode_word, synthetic_word_counts, train_reference_bpe
from zipftok.bpe_json import bpe_layout, parse_merges, truncate_bpe_json


class TestTruncation(unittest.TestCase):
    def setUp(self):
        self.wc = synthetic_word_counts()
        self.full = train_reference_bpe(self.wc, 400)
        self.layout = bpe_layout(self.full)

    def test_layout(self):
        n_init = 2 + len({c for w in self.wc for c in w})
        self.assertEqual(self.layout["n_initial"], n_init)
        self.assertEqual(self.layout["vocab_size"], len(self.full["model"]["vocab"]))

    def test_truncation_equals_training_at_that_size(self):
        n_init = self.layout["n_initial"]
        for v in [n_init, n_init + 1, n_init + 7, 50, 120, 250, 399]:
            direct = train_reference_bpe(self.wc, v)
            trunc = truncate_bpe_json(self.full, v)
            self.assertEqual(trunc["model"]["vocab"], direct["model"]["vocab"], v)
            self.assertEqual(trunc["model"]["merges"], direct["model"]["merges"], v)
            self.assertEqual(len(trunc["model"]["vocab"]), v)
            # and the segmentation of every word is the same
            for w in list(self.wc)[:50]:
                self.assertEqual(encode_word(trunc, w), encode_word(direct, w))

    def test_tuple_merge_format_is_preserved(self):
        full = train_reference_bpe(self.wc, 200, merge_format="tuple")
        trunc = truncate_bpe_json(full, 100)
        self.assertTrue(all(isinstance(m, list) for m in trunc["model"]["merges"]))
        legacy = truncate_bpe_json(train_reference_bpe(self.wc, 200), 100)
        self.assertEqual(parse_merges(trunc["model"]["merges"])[0], parse_merges(legacy["model"]["merges"])[0])

    def test_larger_than_trained_returns_copy(self):
        out = truncate_bpe_json(self.full, 10 ** 6)
        self.assertEqual(out, self.full)
        self.assertIsNot(out, self.full)

    def test_too_small_raises(self):
        with self.assertRaises(ValueError):
            truncate_bpe_json(self.full, self.layout["n_initial"] - 1)

    def test_input_not_modified(self):
        before = copy.deepcopy(self.full)
        truncate_bpe_json(self.full, 100)
        self.assertEqual(before, self.full)

    def test_merge_recreating_an_existing_token(self):
        # A special token that also occurs literally in the corpus is re-created by a merge;
        # that merge adds no new id (the trainer looks the string up first).
        wc = {"abc": 10, "ab": 6, "bc": 5, "xbc": 1}
        full = train_reference_bpe(wc, 100, special_tokens=("ab",))
        layout = bpe_layout(full)
        self.assertGreater(len(layout["merges"]), layout["vocab_size"] - layout["n_initial"])
        self.assertEqual(layout["n_initial"], 1 + 4)
        for v in range(layout["n_initial"], layout["vocab_size"] + 1):
            direct = train_reference_bpe(wc, v, special_tokens=("ab",))
            trunc = truncate_bpe_json(full, v)
            self.assertEqual(trunc["model"]["vocab"], direct["model"]["vocab"])
            self.assertEqual(trunc["model"]["merges"], direct["model"]["merges"])

    def test_rejects_non_contiguous_ids(self):
        bad = copy.deepcopy(self.full)
        tok = next(iter(bad["model"]["vocab"]))
        bad["model"]["vocab"][tok] = 10 ** 6
        with self.assertRaises(ValueError):
            truncate_bpe_json(bad, 100)

    def test_rejects_non_bpe(self):
        with self.assertRaises(ValueError):
            truncate_bpe_json({"model": {"type": "WordPiece", "vocab": {}}}, 10)


if __name__ == "__main__":
    unittest.main()
