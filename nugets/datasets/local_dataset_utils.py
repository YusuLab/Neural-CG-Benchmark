from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np

from .data_transforms import split_indices


def normalize_split(which: str) -> str:
    return "val" if which == "ood" else which


def split_file_list(
    files: Iterable[Path],
    *,
    which: str,
    split_seed: int,
    length: int | None = None,
) -> list[Path]:
    normalized = normalize_split(which)
    file_list = sorted(files)
    if normalized in ("train", "val") and len(file_list) >= 2:
        rng = np.random.default_rng(split_seed)
        split_map = split_indices(len(file_list), 0.9, 0.1, rng=rng)
        split_number = 0 if normalized == "train" else 1
        file_list = [path for path, split in zip(file_list, split_map) if split == split_number]
    if length is not None:
        file_list = file_list[:length]
    return file_list
