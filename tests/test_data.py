import gzip
import json
import os
import tempfile
import unittest

try:  # works with `unittest discover -s tests`, `python -m unittest tests.x` and pytest
    from . import _common  # noqa: F401
except ImportError:
    import _common  # noqa: F401
from zipftok.data import chunk_sequence, iter_corpora, iter_corpus, parse_spec, read_fasta
from zipftok.utils import parse_int_list


class TestSpecs(unittest.TestCase):
    def test_parse_hf(self):
        s = parse_spec("hf:wmt16:de-en:train:translation.de")
        self.assertEqual((s.kind, s.name, s.config, s.split, s.column), ("hf", "wmt16", "de-en", "train", "translation.de"))
        s = parse_spec("hf:bookcorpus")
        self.assertEqual((s.config, s.split, s.column), (None, "train", "text"))
        s = parse_spec("hf:Skylion007/openwebtext::train:text")
        self.assertEqual((s.name, s.config), ("Skylion007/openwebtext", None))

    def test_parse_file(self):
        self.assertEqual(parse_spec("data/x.txt").kind, "file")

    def test_parse_int_list(self):
        self.assertEqual(parse_int_list("2000, 5k,27.5k 40000"), [2000, 5000, 27500, 40000])


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        with open(os.path.join(d, "a.txt"), "w") as f:
            f.write("hello world\n\nsecond line\n")
        with gzip.open(os.path.join(d, "b.jsonl.gz"), "wt") as f:
            f.write(json.dumps({"text": "from json", "meta": {"x": "nested"}}) + "\n")
        with open(os.path.join(d, "c.csv"), "w") as f:
            f.write("smiles,label\nCCO,1\nc1ccccc1,0\n")
        with open(os.path.join(d, "d.smi"), "w") as f:
            f.write("smiles zinc_id\nCCN ZINC1\nCCC ZINC2\n")
        with open(os.path.join(d, "e.fa"), "w") as f:
            f.write(">chr1\nACGTNNNNacgt\nAC\n>chr2\nGGGG\n")
        self.d = d

    def tearDown(self):
        self.tmp.cleanup()

    def p(self, name):
        return os.path.join(self.d, name)

    def test_txt(self):
        self.assertEqual(list(iter_corpus(self.p("a.txt"))), ["hello world", "second line"])

    def test_jsonl_gz_nested(self):
        self.assertEqual(list(iter_corpus(self.p("b.jsonl.gz"))), ["from json"])
        self.assertEqual(list(iter_corpus(self.p("b.jsonl.gz"), text_column="meta.x")), ["nested"])

    def test_csv_and_smi(self):
        self.assertEqual(list(iter_corpus(self.p("c.csv"), text_column="smiles")), ["CCO", "c1ccccc1"])
        self.assertEqual(list(iter_corpus(self.p("d.smi"))), ["CCN", "CCC"])

    def test_fasta(self):
        self.assertEqual(list(read_fasta(self.p("e.fa"))), [("chr1", "ACGTNNNNacgtAC"), ("chr2", "GGGG")])
        self.assertEqual(list(iter_corpus(self.p("e.fa"), fasta_chunk=3)), ["ACG", "T", "ACG", "TAC", "GGG", "G"])

    def test_limits_and_glob(self):
        self.assertEqual(len(list(iter_corpus(self.p("*.txt"), max_texts=1))), 1)
        out = list(iter_corpora([self.p("a.txt"), self.p("d.smi")], max_texts_per_corpus=1))
        self.assertEqual(out, ["hello world", "CCN"])

    def test_chunk_sequence(self):
        self.assertEqual(list(chunk_sequence("acgtnnacg", 2)), ["AC", "GT", "AC", "G"])
        self.assertEqual(list(chunk_sequence("ACGTACG", 3, min_length=3)), ["ACG", "TAC"])


if __name__ == "__main__":
    unittest.main()
