from __future__ import annotations

from pathlib import Path

import torch
from ml_lib.datasets import Dataset

from nugets.datasets.datapoint_types import Graph_datapoint
from nugets.datasets.local_dataset_utils import (
    graph_datapoint_from_pyg,
    load_circuitnet_standardized_graph,
    load_dehnn_pyg_graph,
    load_superblue_processed_graph,
    split_file_list,
)
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


class _DEHNNDirectoryGraphDataset(Dataset[Graph_datapoint]):
    datatype = Graph_datapoint

    def __init__(
        self,
        *,
        root: str,
        which: str = "train",
        split_seed: int = 42,
        length: int | None = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"DE-HNN design root does not exist: {self.root}")
        self.which = which
        self.split_seed = split_seed
        self.length = length
        design_dirs = [path for path in self.root.iterdir() if path.is_dir() and (path / "pyg_data.pkl").exists()]
        self.paths = split_file_list(design_dirs, which=which, split_seed=split_seed, length=length)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        design_dir = self.paths[index]
        return load_dehnn_pyg_graph(design_dir / "pyg_data.pkl")

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "which": self.which,
            "split_seed": self.split_seed,
            "length": self.length,
        }


@register
class DEHNNMLCADNetlistGraphs(_DEHNNDirectoryGraphDataset):
    def __init__(
        self,
        *,
        root: str = "/data/zhishang/DEHNN/de_hnn_tx/data/mlcad/all_designs_netlist_data",
        **kwargs,
    ):
        super().__init__(root=root, **kwargs)


@register
class CircuitNetStandardizedGraphs(Dataset[Graph_datapoint]):
    datatype = Graph_datapoint

    def __init__(
        self,
        *,
        root: str = "/data/zhishang/CircuitNet/de_hnn_qm/new_data/processed_standardized",
        which: str = "train",
        split_seed: int = 42,
        length: int | None = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"CircuitNet standardized root does not exist: {self.root}")
        self.which = which
        self.split_seed = split_seed
        self.length = length
        graph_files = sorted(self.root.glob("*_standardized.pt"))
        self.paths = split_file_list(graph_files, which=which, split_seed=split_seed, length=length)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return load_circuitnet_standardized_graph(self.paths[index])

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "which": self.which,
            "split_seed": self.split_seed,
            "length": self.length,
        }


@register
class SuperblueProcessedGraphs(Dataset[Graph_datapoint]):
    datatype = Graph_datapoint

    def __init__(
        self,
        *,
        root: str = "/data/zhishang/data/superblue/2023-03-06_data",
        which: str = "train",
        split_seed: int = 42,
        length: int | None = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Superblue processed root does not exist: {self.root}")
        self.which = which
        self.split_seed = split_seed
        self.length = length
        graph_files = sorted(self.root.glob("*.node_features.pkl"))
        self.paths = split_file_list(graph_files, which=which, split_seed=split_seed, length=length)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return load_superblue_processed_graph(self.paths[index])

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "which": self.which,
            "split_seed": self.split_seed,
            "length": self.length,
        }
