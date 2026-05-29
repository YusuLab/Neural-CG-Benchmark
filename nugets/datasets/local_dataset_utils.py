from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json
import pickle

import numpy as np
import torch

from .data_transforms import split_indices
from .datapoint_types import Graph_datapoint


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


def load_shortest_path_synthetic_graph(base_path: Path) -> Graph_datapoint:
    with base_path.with_name(base_path.name + "_graph.gpickle").open("rb") as handle:
        graph = pickle.load(handle)
    with base_path.with_name(base_path.name + "_features.json").open() as handle:
        features = json.load(handle)

    nodes = sorted(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    pointset = torch.tensor([[float(features[str(node)])] for node in nodes], dtype=torch.float32)

    directed_edges: list[tuple[int, int]] = []
    for src, dst in graph.edges():
        src_idx = node_to_index[src]
        dst_idx = node_to_index[dst]
        directed_edges.append((src_idx, dst_idx))
        directed_edges.append((dst_idx, src_idx))
    edges = torch.tensor(directed_edges, dtype=torch.int64).t().contiguous()
    return Graph_datapoint(pointset=pointset, edges=edges)
