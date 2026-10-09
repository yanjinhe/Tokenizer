"""zipftok: Zipf-guided vocabulary size selection for BPE tokenizers.

Code for "Pre-trained Models Perform the Best When Token Distributions Follow
Zipf's Law" (He, Zeng and Jiang, EMNLP 2025).

The core analysis (``zipftok.zipf``, ``zipftok.selection``, ``zipftok.bpe_json``)
only needs numpy; tokenizer training uses HuggingFace ``tokenizers`` and the
pre-training / fine-tuning scripts use ``transformers`` and PyTorch.
"""

from .selection import SelectionResult, select_from_curve, vocab_schedule, zipf_guided_selection
from .zipf import ZipfFit, fit_loglog, rank_frequency, segmented_fit, zipf_score

__version__ = "1.0.0"

__all__ = [
    "ZipfFit",
    "fit_loglog",
    "rank_frequency",
    "segmented_fit",
    "zipf_score",
    "SelectionResult",
    "select_from_curve",
    "vocab_schedule",
    "zipf_guided_selection",
]
