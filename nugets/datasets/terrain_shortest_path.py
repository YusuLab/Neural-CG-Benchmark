from __future__ import annotations

from pathlib import Path

import numpy as np
from ml_lib.datasets import Dataset

from nugets.datasets.datapoint_types import Graph_datapoint
from nugets.datasets.local_dataset_utils import (
    ensure_2d_float_tensor,
    ensure_edge_index_tensor,
    split_file_list,
)
from nugets.datasets.register import register


class TerrainPatchDataset(Dataset[Graph_datapoint]):
    datatype = Graph_datapoint

    terrain_name: str = ""

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
            raise FileNotFoundError(
                f"Terrain dataset root does not exist: {self.root}\n"
                "Run: python download_research_datasets.py --entry shortest-paths-terrain-patches"
            )
        self.which = which
        self.split_seed = split_seed
        self.length = length
        self.paths = split_file_list(
            self.root.glob("*.npz"),
            which=which,
            split_seed=split_seed,
            length=length,
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path) as sample:
            pointset = ensure_2d_float_tensor(sample["node_features"])
            edges = ensure_edge_index_tensor(sample["edge_index"])
        return Graph_datapoint(pointset=pointset, edges=edges)

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "terrain_name": self.terrain_name,
            "which": self.which,
            "split_seed": self.split_seed,
            "length": self.length,
        }


@register
class NorwayTerrainPatches(TerrainPatchDataset):
    terrain_name = "norway"

    def __init__(
        self,
        *,
        root: str = "data/server-local/shortest-paths-terrain-patches/norway_patches",
        **kwargs,
    ):
        super().__init__(root=root, **kwargs)


@register
class HollandTerrainPatches(TerrainPatchDataset):
    terrain_name = "holland"

    def __init__(
        self,
        *,
        root: str = "data/server-local/shortest-paths-terrain-patches/holland_patches",
        **kwargs,
    ):
        super().__init__(root=root, **kwargs)


@register
class PhilTerrainPatches(TerrainPatchDataset):
    terrain_name = "phil"

    def __init__(
        self,
        *,
        root: str = "data/server-local/shortest-paths-terrain-patches/phil_patches",
        **kwargs,
    ):
        super().__init__(root=root, **kwargs)
