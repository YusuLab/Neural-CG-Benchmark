from __future__ import annotations

from pathlib import Path

import torch
from ml_lib.datasets import Dataset

from nugets.datasets.datapoint_types import Graph_datapoint
from nugets.datasets.local_dataset_utils import graph_datapoint_from_pyg, load_shortest_path_synthetic_graph
from nugets.datasets.register import register


class SingleGraphDataset(Dataset[Graph_datapoint]):
    datatype = Graph_datapoint

    def __init__(self, *, which: str = "train"):
        self.which = which
        self._graph = self.load_graph()

    def load_graph(self) -> Graph_datapoint:
        raise NotImplementedError

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self._graph


class PygFileGraphDataset(SingleGraphDataset):
    graph_path: str = ""

    def __init__(self, *, root: str | None = None, which: str = "train"):
        self.root = Path(root) if root is not None else Path(self.graph_path)
        super().__init__(which=which)

    def load_graph(self) -> Graph_datapoint:
        if not self.root.exists():
            raise FileNotFoundError(
                f"Graph file does not exist: {self.root}\n"
                "Run: python download_research_datasets.py --entry synthetic-graph-benchmarks"
            )
        data = torch.load(self.root, map_location="cpu")
        return graph_datapoint_from_pyg(data)

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "which": self.which,
        }


@register
class BAShapes(PygFileGraphDataset):
    graph_path = "data/server-local/synthetic-graph-benchmarks/datasets/ba_shapes_300_80_pyg.pt"


@register
class BACommunity(PygFileGraphDataset):
    graph_path = "data/server-local/synthetic-graph-benchmarks/datasets/ba_community_350_100_pyg.pt"


@register
class TreeCycles(PygFileGraphDataset):
    graph_path = "data/server-local/synthetic-graph-benchmarks/datasets/tree_cycles_d8_c20_pyg.pt"


@register
class TreeGrid(PygFileGraphDataset):
    graph_path = "data/server-local/synthetic-graph-benchmarks/datasets/tree_grid_d8_g20_pyg.pt"


@register
class ShortestPathSynthetic(SingleGraphDataset):
    def __init__(
        self,
        *,
        root: str = "data/server-local/synthetic-graph-benchmarks/datasets/shortest_path_n300_p0",
        which: str = "train",
    ):
        self.root = Path(root)
        super().__init__(which=which)

    def load_graph(self) -> Graph_datapoint:
        return load_shortest_path_synthetic_graph(self.root)

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "which": self.which,
        }
