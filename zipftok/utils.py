"""Miscellaneous helpers (seeding, JSON I/O, logging)."""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(obj: Any, path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", level=level)


def parse_int_list(text: str):
    """``"2000,5000,10000"`` or ``"2000 5000"`` -> ``[2000, 5000, 10000]`` (``k`` suffix allowed)."""
    out = []
    for tok in text.replace(",", " ").split():
        tok = tok.strip().lower()
        mult = 1
        if tok.endswith("k"):
            mult, tok = 1000, tok[:-1]
        out.append(int(float(tok) * mult))
    return out
