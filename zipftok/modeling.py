"""Model construction from the JSON configs in ``configs/models``.

Only the vocabulary-dependent fields (vocabulary size and special-token ids)
are taken from the tokenizer, so the same architecture is used for every
vocabulary size and differences in downstream performance can be attributed to
the tokenizer. The number of parameters therefore grows with the vocabulary
(about 88M -> 125M for BERT between 2K and 50K tokens; Appendix D reports 84M-124M).
"""

from __future__ import annotations

import json
from typing import Any, Dict


def load_config_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def build_bert_mlm(config_path: str, tokenizer):
    from transformers import BertConfig, BertForMaskedLM

    cfg = load_config_dict(config_path)
    cfg.update(vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id)
    config = BertConfig(**cfg)
    return BertForMaskedLM(config)


def build_mbart(config_path: str, tokenizer, decoder_start_token_id: int = None):
    from transformers import MBartConfig, MBartForConditionalGeneration

    cfg = load_config_dict(config_path)
    cfg.update(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=decoder_start_token_id if decoder_start_token_id is not None else tokenizer.eos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
    )
    config = MBartConfig(**cfg)
    return MBartForConditionalGeneration(config)


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def pretraining_max_length(model_dir: str, default: int = 512) -> int:
    """Sequence length used during pre-training (read from ``pretrain_results.json``).

    BERT uses learned absolute position embeddings, so positions beyond this length
    were never trained; fine-tuning inputs are truncated to it.
    """
    import os

    path = os.path.join(model_dir, "pretrain_results.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            value = (json.load(f).get("args") or {}).get("max_seq_length")
        if value:
            return int(value)
    return default
