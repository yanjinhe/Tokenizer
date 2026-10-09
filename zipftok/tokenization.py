"""BPE tokenizer training, truncation and export (HuggingFace ``tokenizers``).

Four presets cover the domains studied in the paper:

* ``text``        English text for BERT (OpenWebText + BookCorpus); byte-level BPE.
* ``translation`` shared source/target vocabulary for mBART (WMT / IWSLT); byte-level
                  BPE, mBART special tokens and language codes.
* ``dna``         DNA sequences (DNABERT-2 style); character-level BPE without
                  pre-tokenization, so merges can span arbitrary k-mers.
* ``smiles``      SMILES strings (ZINC20); character-level BPE, so tokens can cover
                  functional groups and rings such as ``c1ccccc1`` or ``(=O)``.

The trainer is HuggingFace's ``BpeTrainer`` (the merge loop of Algorithm 1 in
Appendix A).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

from .bpe_json import truncate_bpe_json

BERT_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
MBART_SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
# mBART-style language codes
LANGUAGE_CODES = {
    "en": "en_XX",
    "de": "de_DE",
    "fr": "fr_XX",
    "zh": "zh_CN",
}
META_FILE = "zipftok_meta.json"


@dataclass
class Preset:
    name: str
    byte_level: bool
    special_tokens: List[str]
    unk_token: Optional[str]
    family: str  # "bert" or "mbart"
    normalizer: Optional[str] = None
    model_max_length: int = 512
    language_codes: List[str] = field(default_factory=list)


PRESETS: Dict[str, Preset] = {
    "text": Preset("text", True, BERT_SPECIAL_TOKENS, None, "bert", normalizer="nfc", model_max_length=512),
    "translation": Preset(
        "translation", True, MBART_SPECIAL_TOKENS, None, "mbart", normalizer="nfc", model_max_length=1024
    ),
    "dna": Preset("dna", False, BERT_SPECIAL_TOKENS, "[UNK]", "bert", normalizer=None, model_max_length=512),
    "smiles": Preset("smiles", False, BERT_SPECIAL_TOKENS, "[UNK]", "bert", normalizer=None, model_max_length=512),
}


def get_preset(name: str, languages: Sequence[str] = ()) -> Preset:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    p = PRESETS[name]
    codes = [LANGUAGE_CODES.get(l, l) for l in languages]
    if p.family == "mbart" and not codes:
        raise ValueError("the translation preset needs --languages (e.g. de en)")
    return Preset(
        name=p.name,
        byte_level=p.byte_level,
        special_tokens=list(p.special_tokens) + [c for c in codes if c not in p.special_tokens],
        unk_token=p.unk_token,
        family=p.family,
        normalizer=p.normalizer,
        model_max_length=p.model_max_length,
        language_codes=codes,
    )


def _require_tokenizers():
    try:
        import tokenizers  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("this function needs the `tokenizers` package (pip install tokenizers)") from e


def build_untrained_tokenizer(preset: Preset):
    """Tokenizer pipeline (normalizer, pre-tokenizer, decoder) with an empty BPE model."""
    _require_tokenizers()
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE

    tok = Tokenizer(BPE(unk_token=preset.unk_token) if preset.unk_token else BPE())
    if preset.normalizer == "nfc":
        tok.normalizer = normalizers.NFC()
    elif preset.normalizer == "nfkc":
        tok.normalizer = normalizers.NFKC()
    if preset.byte_level:
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True)
        tok.decoder = decoders.ByteLevel()
    else:
        # DNA / SMILES: every whitespace-free sequence is one "word", so BPE merges can
        # grow over the whole sequence (no k-mer or atom-level pre-segmentation).
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tok


def _batched(texts: Iterable[str], batch_size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for t in texts:
        batch.append(t)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def train_bpe(
    texts: Iterable[str],
    vocab_size: int,
    preset: Union[str, Preset] = "text",
    *,
    languages: Sequence[str] = (),
    min_frequency: int = 2,
    limit_alphabet: Optional[int] = None,
    max_token_length: Optional[int] = None,
    length: Optional[int] = None,
    show_progress: bool = True,
    batch_size: int = 1000,
):
    """Train a BPE tokenizer with ``vocab_size`` tokens (special tokens included)."""
    _require_tokenizers()
    from tokenizers import pre_tokenizers, trainers

    p = get_preset(preset, languages) if isinstance(preset, str) else preset
    tok = build_untrained_tokenizer(p)
    kwargs = dict(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        special_tokens=p.special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if p.byte_level else [],
    )
    if limit_alphabet is not None:
        kwargs["limit_alphabet"] = limit_alphabet
    if max_token_length is not None:
        kwargs["max_token_length"] = max_token_length
    trainer = trainers.BpeTrainer(**kwargs)
    n_batches = None if length is None else (length + batch_size - 1) // batch_size
    tok.train_from_iterator(_batched(texts, batch_size), trainer=trainer, length=n_batches)
    attach_post_processor(tok, p)
    return tok


def attach_post_processor(tok, preset: Preset) -> None:
    """``[CLS] A [SEP] (B [SEP])`` for BERT; mBART inputs are assembled by the data code."""
    from tokenizers import processors

    if preset.family == "bert":
        cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
        tok.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
        )
    # mBART: no post-processor (tokenizers<0.20 cannot unset one, and none is set by default)


def truncate_tokenizer(tok, vocab_size: int):
    """BPE tokenizer of size ``vocab_size`` obtained from a larger trained one.

    Equivalent to re-training with ``vocab_size`` on the same data (see
    :mod:`zipftok.bpe_json`).
    """
    from tokenizers import Tokenizer

    data = json.loads(tok.to_str())
    return Tokenizer.from_str(json.dumps(truncate_bpe_json(data, vocab_size), ensure_ascii=False))


def bare_model_tokenizer(tok):
    """Copy of ``tok`` without normalizer / pre-tokenizer / post-processor.

    Encoding an already pre-tokenized word with it runs only the BPE model, which
    lets us tokenize each distinct word once and weight it by its corpus count.
    """
    from tokenizers import Tokenizer

    data = json.loads(tok.to_str())
    for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder", "truncation", "padding"):
        data[key] = None
    return Tokenizer.from_str(json.dumps(data, ensure_ascii=False))


def special_token_ids(tok) -> List[int]:
    data = json.loads(tok.to_str())
    return sorted(t["id"] for t in (data.get("added_tokens") or []) if t.get("special", False))


def load_raw_tokenizer(path: str):
    """Load a ``tokenizers.Tokenizer`` from a ``tokenizer.json`` file or a directory containing one."""
    _require_tokenizers()
    from tokenizers import Tokenizer

    if os.path.isdir(path):
        path = os.path.join(path, "tokenizer.json")
    return Tokenizer.from_file(path)


def read_meta(path: str) -> dict:
    d = path if os.path.isdir(path) else os.path.dirname(path)
    meta_path = os.path.join(d, META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_tokenizer(tok, out_dir: str, preset: Preset, extra_meta: Optional[dict] = None) -> None:
    """Save ``tokenizer.json`` plus a ``transformers`` wrapper usable with ``AutoTokenizer``."""
    os.makedirs(out_dir, exist_ok=True)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    meta = {
        "preset": preset.name,
        "family": preset.family,
        "vocab_size": tok.get_vocab_size(with_added_tokens=True),
        "language_codes": preset.language_codes,
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(out_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    try:
        hf = to_transformers(tok, preset)
    except ImportError:
        return
    hf.save_pretrained(out_dir)


def to_transformers(tok, preset: Preset):
    """Wrap a ``tokenizers.Tokenizer`` into ``transformers.PreTrainedTokenizerFast``."""
    from transformers import PreTrainedTokenizerFast

    if preset.family == "bert":
        kwargs = dict(unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    else:
        kwargs = dict(
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            additional_special_tokens=list(preset.language_codes),
        )
    return PreTrainedTokenizerFast(tokenizer_object=tok, model_max_length=preset.model_max_length, **kwargs)


def load_hf_tokenizer(path: str):
    """Load the ``transformers`` tokenizer saved by :func:`save_tokenizer`."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(path)


def preset_from_dir(path: str) -> Preset:
    meta = read_meta(path)
    if not meta:
        raise ValueError(f"{path} has no {META_FILE}; was it created by scripts/train_tokenizer.py?")
    p = PRESETS[meta["preset"]]
    langs = meta.get("language_codes", [])
    return Preset(
        name=p.name,
        byte_level=p.byte_level,
        special_tokens=list(p.special_tokens) + [c for c in langs if c not in p.special_tokens],
        unk_token=p.unk_token,
        family=p.family,
        normalizer=p.normalizer,
        model_max_length=p.model_max_length,
        language_codes=list(langs),
    )


def display_tokens(tok, text: str) -> List[str]:
    """Human-readable tokens of ``text`` (byte-level symbols decoded back to text)."""
    enc = tok.encode(text, add_special_tokens=False)
    if tok.decoder is None:
        return list(enc.tokens)
    out = []
    for t in enc.tokens:
        out.append(tok.decoder.decode([t]))
    return out
