import unittest
from collections import Counter

import numpy as np

try:  # works with `unittest discover -s tests`, `python -m unittest tests.x` and pytest
    from . import _common  # noqa: F401
except ImportError:
    import _common  # noqa: F401
from ref_bpe import encode_word, synthetic_word_counts, train_reference_bpe
from zipftok.bpe_json import truncate_bpe_json
from zipftok.counting import WordTable, aggregate_word_encodings
from zipftok.zipf import zipf_score


class TestAggregation(unittest.TestCase):
    def test_word_table_counts_equal_corpus_counts(self):
        wc = synthetic_word_counts(seed=3)
        tok = train_reference_bpe(wc, 150)
        V = len(tok["model"]["vocab"])
        # "corpus": every word repeated by its count
        corpus = [w for w, c in wc.items() for _ in range(c)]
        direct = np.zeros(V)
        for w in corpus:
            for i in encode_word(tok, w):
                direct[i] += 1
        table = WordTable.from_counter(Counter(corpus), n_texts=len(corpus))
        agg = aggregate_word_encodings([encode_word(tok, w) for w in table.words], table.counts, V)
        np.testing.assert_array_equal(agg, direct)

    def test_r2_increases_with_vocabulary_on_zipfian_words(self):
        # Hypothesis 1 on toy data: a small (character-level) vocabulary bends the curve
        rng = np.random.default_rng(0)
        letters = list("abcdefghij")
        words = sorted({"".join(rng.choice(letters, size=int(rng.integers(3, 8)))) for _ in range(3000)})
        rng.shuffle(words)
        wc = {w: max(1, int(50000 / (i + 1))) for i, w in enumerate(words[:1500])}
        full = train_reference_bpe(wc, 700, special_tokens=())
        table = WordTable.from_counter(Counter(wc), n_texts=1)
        r2 = []
        for v in (40, 700):
            tok = truncate_bpe_json(full, v)
            counts = aggregate_word_encodings([encode_word(tok, w) for w in table.words], table.counts,
                                              len(tok["model"]["vocab"]))
            r2.append(zipf_score(counts).r2)
        self.assertGreater(r2[1], r2[0])


if __name__ == "__main__":
    unittest.main()
