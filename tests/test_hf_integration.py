"""End-to-end checks against the real HuggingFace ``tokenizers`` library (skipped if it is missing)."""

import tempfile
import unittest

import numpy as np

try:  # works with `unittest discover -s tests`, `python -m unittest tests.x` and pytest
    from . import _common  # noqa: F401
except ImportError:
    import _common  # noqa: F401

try:
    import tokenizers  # noqa: F401

    HAVE_TOKENIZERS = True
except ImportError:
    HAVE_TOKENIZERS = False


def toy_text(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    vocab = ["".join(rng.choice(list("abcdefghijklmnop"), size=int(rng.integers(2, 9)))) for _ in range(800)]
    p = 1.0 / np.arange(1, len(vocab) + 1)
    p /= p.sum()
    return [" ".join(rng.choice(vocab, size=int(rng.integers(5, 25)), p=p)) for _ in range(n)]


def toy_dna(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    motifs = ["".join(rng.choice(list("ACGT"), size=int(rng.integers(3, 9)))) for _ in range(60)]
    return ["".join(rng.choice(motifs, size=30)) for _ in range(n)]


@unittest.skipUnless(HAVE_TOKENIZERS, "tokenizers is not installed")
class TestWithTokenizers(unittest.TestCase):
    def test_truncation_matches_independent_training(self):
        from zipftok.tokenization import train_bpe, truncate_tokenizer

        for preset, data, sizes in [("text", toy_text(), (300, 500, 800)), ("dna", toy_dna(), (40, 100, 200))]:
            big = train_bpe(data, max(sizes) + 300, preset, show_progress=False)
            for v in sizes:
                direct = train_bpe(data, v, preset, show_progress=False)
                trunc = truncate_tokenizer(big, v)
                self.assertEqual(direct.get_vocab(), trunc.get_vocab(), (preset, v))
                for t in data[:50]:
                    self.assertEqual(direct.encode(t).ids, trunc.encode(t).ids)

    def test_word_counting_equals_full_encoding(self):
        from zipftok.counting import WordTable, count_tokens
        from zipftok.tokenization import train_bpe

        for preset, data in [("text", toy_text()), ("dna", toy_dna())]:
            tok = train_bpe(data, 400 if preset == "text" else 120, preset, show_progress=False)
            full = count_tokens(tok, texts=data, method="full")
            words = count_tokens(tok, table=WordTable.from_texts(tok, data), method="words")
            np.testing.assert_array_equal(full, words)
            self.assertGreater(full.sum(), 0)

    def test_save_and_reload(self):
        from zipftok.tokenization import get_preset, load_raw_tokenizer, read_meta, save_tokenizer, train_bpe

        preset = get_preset("translation", ["de", "en"])
        tok = train_bpe(toy_text(), 400, preset, show_progress=False)
        self.assertIsNotNone(tok.token_to_id("de_DE"))
        with tempfile.TemporaryDirectory() as d:
            save_tokenizer(tok, d, preset)
            again = load_raw_tokenizer(d)
            self.assertEqual(again.get_vocab(), tok.get_vocab())
            self.assertEqual(read_meta(d)["language_codes"], ["de_DE", "en_XX"])
            try:
                import transformers  # noqa: F401
            except ImportError:
                return
            from zipftok.tokenization import load_hf_tokenizer

            hf = load_hf_tokenizer(d)
            self.assertEqual(hf.pad_token, "<pad>")
            self.assertEqual(hf.convert_tokens_to_ids("en_XX"), tok.token_to_id("en_XX"))

    def test_bert_post_processor(self):
        from zipftok.tokenization import train_bpe

        tok = train_bpe(toy_text(), 300, "text", show_progress=False)
        enc = tok.encode("abc def", "ghi")
        self.assertEqual(enc.tokens[0], "[CLS]")
        self.assertEqual(enc.tokens.count("[SEP]"), 2)
        self.assertEqual(max(enc.type_ids), 1)


if __name__ == "__main__":
    unittest.main()
