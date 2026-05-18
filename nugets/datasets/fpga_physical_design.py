from __future__ import annotations

from pathlib import Path

from ml_lib.datasets import Dataset

from nugets.datasets.datapoint_types import Graph_datapoint
from nugets.datasets.local_dataset_utils import load_dehnn_pyg_graph, split_file_list
from nugets.datasets.register import register


@register
class DEHNNISPD16NetlistGraphs(Dataset[Graph_datapoint]):
    datatype = Graph_datapoint

    def __init__(
        self,
        *,
        root: str = "data/server-local/dehnn-netlist-dataset/ispd16_netlist_data",
        which: str = "train",
        split_seed: int = 42,
        length: int | None = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"DE-HNN FPGA design root does not exist: {self.root}\n"
                "Run: python download_research_datasets.py --entry dehnn-netlist-dataset"
            )
        self.which = which
        self.split_seed = split_seed
        self.length = length
        design_dirs = [path for path in self.root.iterdir() if path.is_dir() and (path / "pyg_data.pkl").exists()]
        self.paths = split_file_list(design_dirs, which=which, split_seed=split_seed, length=length)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return load_dehnn_pyg_graph(self.paths[index] / "pyg_data.pkl")

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "which": self.which,
            "split_seed": self.split_seed,
            "length": self.length,
        }
