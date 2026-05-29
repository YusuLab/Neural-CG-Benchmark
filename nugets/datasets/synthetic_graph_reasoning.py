from __future__ import annotations

from pathlib import Path

import torch
from ml_lib.datasets import Dataset

from nugets.datasets.datapoint_types import Graph_datapoint
from nugets.datasets.local_dataset_utils import load_shortest_path_synthetic_graph
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
        return self._pyg_to_graph_datapoint(data)

    @staticmethod
    def _pyg_to_graph_datapoint(data) -> Graph_datapoint:
        if hasattr(data, "x") and data.x is not None:
            features = data.x
        elif hasattr(data, "feat") and data.feat is not None:
            features = data.feat
        else:
            features = torch.ones((int(getattr(data, "num_nodes")), 1), dtype=torch.float32)
        pointset = torch.as_tensor(features, dtype=torch.float32)
        if pointset.ndim == 1:
            pointset = pointset.unsqueeze(-1)
        edges = torch.as_tensor(data.edge_index, dtype=torch.int64)
        if edges.shape[0] != 2 and edges.shape[1] == 2:
            edges = edges.t().contiguous()
        return Graph_datapoint(pointset=pointset, edges=edges)

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "which": self.which,
        }


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
