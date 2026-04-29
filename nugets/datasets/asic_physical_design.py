from __future__ import annotations

from pathlib import Path

import torch
from ml_lib.datasets import Dataset

from nugets.datasets.datapoint_types import Graph_datapoint
from nugets.datasets.local_dataset_utils import graph_datapoint_from_pyg
from nugets.datasets.register import register


@register
class ChipDiffusionPygGraph(Dataset[Graph_datapoint]):
    datatype = Graph_datapoint

    def __init__(
        self,
        *,
        root: str = "/data/zhishang/chipdiffusion/datasets/graph/pyg_graph.pt",
        which: str = "train",
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"ChipDiffusion graph file does not exist: {self.root}")
        self.which = which
        data = torch.load(self.root, map_location="cpu")
        self.graph = graph_datapoint_from_pyg(data)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.graph

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "which": self.which,
        }
