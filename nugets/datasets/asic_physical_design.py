from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from ml_lib.datasets import Dataset

from nugets.datasets.data_transforms import split_indices
from nugets.datasets.datapoint_types import (
    ChipDesignReference,
    CircuitNetDesignReference,
    SuperblueDesignReference,
)
from nugets.datasets.register import register


class _ChipDatasetBase:
    @staticmethod
    def _split_paths(
        files: Iterable[Path],
        *,
        which: str,
        split_seed: int,
        length: int | None = None,
    ) -> list[Path]:
        normalized = "val" if which == "ood" else which
        file_list = sorted(files)
        if normalized in ("train", "val") and len(file_list) >= 2:
            rng = np.random.default_rng(split_seed)
            split_map = split_indices(len(file_list), 0.9, 0.1, rng=rng)
            split_number = 0 if normalized == "train" else 1
            file_list = [p for p, s in zip(file_list, split_map) if s == split_number]
        if length is not None:
            file_list = file_list[:length]
        return file_list


@register
class ChipSyntheticDataset(_ChipDatasetBase, Dataset[ChipDesignReference]):
    datatype = ChipDesignReference
    hf_slug = "chipdiffusion-graph-dataset"
    default_root = "data/server-local/chipdiffusion-graph-dataset"

    def __init__(
        self,
        *,
        root: str = default_root,
        which: str = "train",
        split_seed: int = 42,
        length: int | None = None,
        auto_download: bool = True,
        download_root: str | None = None,
    ):
        self.root = Path(root)
        self.which = which
        self.split_seed = split_seed
        self.length = length
        self.auto_download = auto_download
        self.download_root = download_root
        if not self.root.exists() and auto_download:
            self._auto_download(download_root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"ChipDiffusion graph root does not exist: {self.root}\n"
                "Download from: https://huggingface.co/datasets/luckyjackluo/Neural-CG-Benchmark"
            )
        self.paths = self._split_paths(
            self._discover_design_paths(self.root),
            which=which,
            split_seed=split_seed,
            length=length,
        )
        if not self.paths:
            raise FileNotFoundError(f"No ChipDiffusion design references found under: {self.root}")

    def _auto_download(self, dest: str | Path | None = None) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for automatic dataset downloads. "
                "Install it with: uv add huggingface_hub"
            ) from exc

        _hf_repo_id = "luckyjackluo/Neural-CG-Benchmark"
        _hf_prefix = "server-local"

        if dest is not None:
            download_dest = Path(dest)
        else:
            parts = self.root.parts
            if _hf_prefix in parts:
                idx = parts.index(_hf_prefix)
                download_dest = Path(".") if idx == 0 else Path(*parts[:idx])
            else:
                download_dest = Path("data")

        snapshot_download(
            repo_id=_hf_repo_id,
            repo_type="dataset",
            allow_patterns=[f"{_hf_prefix}/{self.hf_slug}/**"],
            local_dir=str(download_dest),
            local_dir_use_symlinks=False,
        )

    def _discover_design_paths(self, root: Path) -> list[Path]:
        graph_files = self._discover_graph_files(root)
        if graph_files:
            return graph_files

        design_dirs = []
        for path in root.rglob("*"):
            if path.is_dir() and any(child.is_file() for child in path.iterdir()):
                design_dirs.append(path)
        return sorted(design_dirs)

    def _discover_graph_files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root]

        canonical_graph = root / "datasets" / "graph" / "pyg_graph.pt"
        if canonical_graph.exists():
            return [canonical_graph]

        return sorted(root.glob("**/*pyg*.pt"))

    def _make_reference(self, path: Path) -> ChipDesignReference:
        design_name = path.stem
        graph_path = path
        search_root = self.root
        if path.is_dir():
            design_name = path.name
            graph_path = self._find_related_path(path, ("pyg", "graph"))
            search_root = path
        if path.name == "pyg_graph.pt" and path.parent.name == "graph":
            design_name = path.parents[2].name if len(path.parents) >= 3 else path.stem

        return ChipDesignReference(
            root=self.root,
            design_name=design_name,
            graph_path=graph_path,
            placement_path=self._find_related_path(search_root, ("placement", "placements")),
            congestion_path=self._find_related_path(search_root, ("congestion", "overflow")),
            metadata_path=self._find_related_path(search_root, ("metadata", "config", "summary")),
        )

    def _find_related_path(self, anchor: Path, keywords: tuple[str, ...]) -> Path | None:
        if anchor.is_file():
            if any(keyword in anchor.name.lower() for keyword in keywords):
                return anchor
            return None
        if not anchor.exists():
            return None

        candidates = []
        for path in anchor.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if any(keyword in name for keyword in keywords):
                candidates.append(path)
            elif any(keyword in str(path.parent).lower() for keyword in keywords):
                candidates.append(path)
        return sorted(candidates)[0] if candidates else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self._make_reference(self.paths[index])

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "which": self.which,
            "split_seed": self.split_seed,
            "length": self.length,
            "auto_download": self.auto_download,
            "download_root": self.download_root,
        }


@register
class ChipCircuitNetDataset(_ChipDatasetBase, Dataset[CircuitNetDesignReference]):
    datatype = CircuitNetDesignReference

    def __init__(
        self,
        *,
        root: str = "data/server-local/circuitnet-design-graphs/processed_standardized",
        which: str = "train",
        split_seed: int = 42,
        length: int | None = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"CircuitNet congestion root does not exist: {self.root}\n"
                "Download from: https://huggingface.co/datasets/luckyjackluo/Neural-CG-Benchmark"
            )
        self.which = which
        self.split_seed = split_seed
        self.length = length
        graph_files = sorted(self.root.glob("*_standardized.pt"))
        self.paths = self._split_paths(graph_files, which=which, split_seed=split_seed, length=length)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        graph_path = self.paths[index]
        design_name = graph_path.name.removesuffix("_standardized.pt")
        return CircuitNetDesignReference(
            root=self.root,
            design_name=design_name,
            graph_path=graph_path,
        )

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "which": self.which,
            "split_seed": self.split_seed,
            "length": self.length,
        }


@register
class ChipASICDataset(_ChipDatasetBase, Dataset[SuperblueDesignReference]):
    datatype = SuperblueDesignReference

    def __init__(
        self,
        *,
        root: str = "data/server-local/superblue-processed-graph-features/2023-03-06_data",
        which: str = "train",
        split_seed: int = 42,
        length: int | None = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"Superblue congestion root does not exist: {self.root}\n"
                "Download from: https://huggingface.co/datasets/luckyjackluo/Neural-CG-Benchmark"
            )
        self.which = which
        self.split_seed = split_seed
        self.length = length
        self.paths = self._split_paths(
            self._discover_node_feature_files(),
            which=which,
            split_seed=split_seed,
            length=length,
        )
        if not self.paths:
            raise FileNotFoundError(f"No Superblue congestion samples found under: {self.root}")

    def _discover_node_feature_files(self) -> list[Path]:
        node_feature_files = []
        for path in self.root.glob("*.node_features.pkl"):
            sample_name = path.name.removesuffix(".node_features.pkl")
            if (
                (self.root / f"{sample_name}.net_features.pkl").exists()
                and (self.root / f"{sample_name}.bipartite.pkl").exists()
                and (self.root / f"{sample_name}.targets.pkl").exists()
            ):
                node_feature_files.append(path)
        return sorted(node_feature_files)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        node_feature_path = self.paths[index]
        sample_name = node_feature_path.name.removesuffix(".node_features.pkl")
        metadata_path = self.root / f"{sample_name}.global_information.pkl"
        return SuperblueDesignReference(
            root=self.root,
            sample_name=sample_name,
            node_feature_path=node_feature_path,
            net_feature_path=self.root / f"{sample_name}.net_features.pkl",
            bipartite_path=self.root / f"{sample_name}.bipartite.pkl",
            target_path=self.root / f"{sample_name}.targets.pkl",
            metadata_path=metadata_path if metadata_path.exists() else None,
        )

    def dataset_parameters(self):
        return {
            "root": str(self.root),
            "which": self.which,
            "split_seed": self.split_seed,
            "length": self.length,
        }
