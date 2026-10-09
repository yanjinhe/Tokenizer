"""PyTorch data collators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np

from .noising import permute_sentences, text_infilling

if TYPE_CHECKING:  # torch is imported lazily so that the package works without it
    import torch


def _pad(seqs: Sequence[Sequence[int]], value: int) -> "torch.Tensor":
    import torch

    n = max(len(s) for s in seqs)
    out = torch.full((len(seqs), n), value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(list(s), dtype=torch.long)
    return out


def mbart_shift_right(labels: "torch.Tensor", pad_token_id: int) -> "torch.Tensor":
    """Decoder inputs for mBART: move the last non-pad label (the language id) to the front.

    Same as ``transformers.models.mbart.modeling_mbart.shift_tokens_right``. Providing
    ``decoder_input_ids`` explicitly keeps training correct when the Trainer's label
    smoothing pops ``labels`` before calling the model.
    """
    import torch

    prev = labels.clone()
    prev.masked_fill_(prev == -100, pad_token_id)
    last = (prev.ne(pad_token_id).sum(dim=1) - 1).clamp(min=0).unsqueeze(-1)
    start = prev.gather(1, last).squeeze(-1)
    out = torch.empty_like(prev)
    out[:, 1:] = prev[:, :-1]
    out[:, 0] = start
    return out


@dataclass
class MBartDenoisingCollator:
    """Applies mBART noise on the fly (new noise every time an example is seen).

    Each feature has ``input_ids`` (concatenated sentences of one document, no
    special tokens), ``sentence_lengths`` and ``lang_id``. The encoder sees
    ``noise(document) </s> <lang>`` and the decoder reconstructs
    ``document </s> <lang>``; ``MBartForConditionalGeneration`` derives the decoder
    inputs ``<lang> document </s>`` from the labels.
    """

    pad_token_id: int
    eos_token_id: int
    mask_token_id: int
    word_start: Optional[np.ndarray] = None  # bool per vocabulary id
    mask_ratio: float = 0.35
    poisson_lambda: float = 3.5
    permute: bool = True
    max_length: int = 512
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def _split(self, ids: List[int], lengths: List[int]) -> List[List[int]]:
        out, i = [], 0
        for l in lengths:
            out.append(ids[i:i + l])
            i += l
        return [s for s in out if s]

    def __call__(self, features: List[Dict]) -> Dict[str, "torch.Tensor"]:
        sources, targets = [], []
        body_max = self.max_length - 2
        for f in features:
            ids = list(f["input_ids"])
            lengths = list(f.get("sentence_lengths") or [len(ids)])
            lang = int(f["lang_id"])
            sentences, budget = [], body_max
            for s in self._split(ids, lengths):
                if budget <= 0:
                    break
                sentences.append(s[:budget])
                budget -= len(sentences[-1])
            original = [t for s in sentences for t in s]
            doc = permute_sentences(sentences, self._rng) if self.permute and len(sentences) > 1 else sentences
            doc_ids = [t for s in doc for t in s]
            ws = self.word_start[np.asarray(doc_ids, dtype=np.int64)] if self.word_start is not None and doc_ids else None
            noised = text_infilling(doc_ids, self.mask_token_id, self._rng, mask_ratio=self.mask_ratio,
                                    poisson_lambda=self.poisson_lambda, word_starts=ws)[:body_max]
            sources.append(noised + [self.eos_token_id, lang])
            targets.append(original + [self.eos_token_id, lang])
        input_ids = _pad(sources, self.pad_token_id)
        labels = _pad(targets, -100)
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_token_id).long(),
            "labels": labels,
            "decoder_input_ids": mbart_shift_right(labels, self.pad_token_id),
        }


@dataclass
class Seq2SeqCollator:
    """Pads ``input_ids`` / ``labels`` (and builds mBART decoder inputs) for translation fine-tuning."""

    pad_token_id: int

    def __call__(self, features: List[Dict]) -> Dict[str, "torch.Tensor"]:
        input_ids = _pad([f["input_ids"] for f in features], self.pad_token_id)
        batch = {"input_ids": input_ids, "attention_mask": (input_ids != self.pad_token_id).long()}
        if "labels" in features[0]:
            batch["labels"] = _pad([f["labels"] for f in features], -100)
            batch["decoder_input_ids"] = mbart_shift_right(batch["labels"], self.pad_token_id)
        return batch


@dataclass
class MultiLabelCollator:
    """Pads inputs and stacks float multi-task labels that may contain NaN (missing)."""

    pad_token_id: int

    def __call__(self, features: List[Dict]) -> Dict[str, "torch.Tensor"]:
        import torch

        input_ids = _pad([f["input_ids"] for f in features], self.pad_token_id)
        batch = {"input_ids": input_ids, "attention_mask": (input_ids != self.pad_token_id).long()}
        if "labels" in features[0]:
            batch["labels"] = torch.tensor([list(map(float, f["labels"])) for f in features], dtype=torch.float32)
        return batch
