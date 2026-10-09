"""Small helpers that keep the training scripts working across ``transformers`` versions.

The experiments in the paper used transformers v4.38 and PyTorch 2.0. Later
4.x releases renamed ``evaluation_strategy`` to ``eval_strategy`` and the
``tokenizer`` argument of ``Trainer`` to ``processing_class``; these helpers pick
whichever name the installed version understands.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def training_arguments(cls=None, **kwargs):
    """Build ``TrainingArguments`` (or ``Seq2SeqTrainingArguments``) from version-agnostic kwargs."""
    if cls is None:
        from transformers import TrainingArguments as cls  # noqa: N813
    params = inspect.signature(cls.__init__).parameters
    if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    unknown = [k for k in kwargs if k not in params]
    for k in unknown:
        logger.warning("TrainingArguments: ignoring unsupported argument %s=%r", k, kwargs.pop(k))
    return cls(**kwargs)


def trainer_kwargs(tokenizer, **kwargs) -> Dict[str, Any]:
    """Pass the tokenizer as ``processing_class`` (>= 4.46) or ``tokenizer`` (older)."""
    from transformers import Trainer

    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    return kwargs
