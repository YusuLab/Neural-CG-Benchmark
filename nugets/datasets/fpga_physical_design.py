from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from ml_lib.datasets import Dataset

from nugets.datasets.data_transforms import split_indices
from nugets.datasets.datapoint_types import ChipFPGADesignReference
from nugets.datasets.register import register


@register
class ChipFPGADataset(Dataset[ChipFPGADesignReference]):
    datatype = ChipFPGADesignReference

    _MLCAD_DEFAULT_ROOT = "data/server-local/dehnn-netlist-dataset/all_designs_netlist_data"
    _ISPD16_DEFAULT_ROOT = "data/server-local/dehnn-netlist-dataset/ispd16_netlist_data"
    _HF_REPO_ID = "luckyjackluo/Neural-CG-Benchmark"
    _HF_PREFIX = "server-local"
    _HF_SLUG = "dehnn-netlist-dataset"

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

    def _auto_download(self, dest: str | Path | None = None) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for automatic dataset downloads. "
                "Install it with: pip install huggingface_hub"
            ) from exc

        if dest is not None:
            download_dest = Path(dest)
        else:
            parts = self.root.parts
            if self._HF_PREFIX in parts:
                idx = parts.index(self._HF_PREFIX)
                download_dest = Path(".") if idx == 0 else Path(*parts[:idx])
            else:
                download_dest = Path("data")

        snapshot_download(
            repo_id=self._HF_REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"{self._HF_PREFIX}/{self._HF_SLUG}/**"],
            local_dir=str(download_dest),
            local_dir_use_symlinks=False,
        )

    def __init__(
        self,
        *,
        source: Literal["mlcad", "ispd16"],
        root: str | None = None,
        target_root: str | None = None,
        which: str = "train",
        split_seed: int = 42,
        length: int | None = None,
        auto_download: bool = True,
        download_root: str | None = None,
    ):
        if source not in ("mlcad", "ispd16"):
            raise ValueError(f"source must be 'mlcad' or 'ispd16', got {source!r}")
        self.source = source
        self.auto_download = auto_download
        self.download_root = download_root

        if source == "mlcad":
            self.root = Path(root) if root is not None else Path(self._MLCAD_DEFAULT_ROOT)
            self.target_root = (
                Path(target_root) if target_root is not None
                else self.root.parent / "target_data"
            )
        else:
            self.root = Path(root) if root is not None else Path(self._ISPD16_DEFAULT_ROOT)
            self.target_root = (
                Path(target_root) if target_root is not None
                else self.root.parent / "ispd16_target_data_random"
            )

        if not self.root.exists() and auto_download:
            self._auto_download(download_root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"DE-HNN {source.upper()} netlist root does not exist: {self.root}\n"
                "Download from: https://huggingface.co/datasets/luckyjackluo/Neural-CG-Benchmark"
            )
        if not self.target_root.exists():
            raise FileNotFoundError(
                f"DE-HNN {source.upper()} target root does not exist: {self.target_root}"
            )
        self.which = which
        self.split_seed = split_seed
        self.length = length
        self.paths = self._split_paths(
            self._discover_target_dirs(),
            which=which,
            split_seed=split_seed,
            length=length,
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No DE-HNN {source.upper()} targets found under: {self.target_root}"
            )

    def _discover_target_dirs(self) -> list[Path]:
        target_file = (
            "route_utilization_map.pkl" if self.source == "mlcad" else "target_site_util.pkl"
        )
        target_dirs = []
        for path in self.target_root.iterdir():
            if not path.is_dir():
                continue
            if not (path / target_file).exists():
                continue
            if self._graph_path_for_variant(path.name).exists():
                target_dirs.append(path)
        return sorted(target_dirs)

    def _design_name_from_variant(self, variant_name: str) -> str:
        if self.source == "mlcad":
            match = re.match(r"^(Design_\d+)_", variant_name)
            if match is None:
                raise ValueError(f"Cannot infer DE-HNN MLCAD design name from variant: {variant_name}")
            return match.group(1)
        else:
            if "_seed_" not in variant_name:
                raise ValueError(f"Cannot infer DE-HNN ISPD16 design name from variant: {variant_name}")
            return variant_name.split("_seed_", maxsplit=1)[0]

    def _graph_path_for_variant(self, variant_name: str) -> Path:
        design_name = self._design_name_from_variant(variant_name)
        return self.root / f"{design_name}_netlist_data" / "pyg_data.pkl"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        target_dir = self.paths[index]
        design_name = self._design_name_from_variant(target_dir.name)
        if self.source == "mlcad":
            return ChipFPGADesignReference(
                root=self.root,
                target_root=self.target_root,
                design_name=design_name,
                variant_name=target_dir.name,
                graph_path=self._graph_path_for_variant(target_dir.name),
                target_dir=target_dir,
                source="mlcad",
                node_loc_x_path=target_dir / "node_loc_x.pkl",
                node_loc_y_path=target_dir / "node_loc_y.pkl",
                route_utilization_map_path=target_dir / "route_utilization_map.pkl",
                target_node_utilization_path=target_dir / "target_node_utilization.pkl",
                target_net_hpwl_path=target_dir / "target_net_hpwl.pkl",
            )
        else:
            return ChipFPGADesignReference(
                root=self.root,
                target_root=self.target_root,
                design_name=design_name,
                variant_name=target_dir.name,
                graph_path=self._graph_path_for_variant(target_dir.name),
                target_dir=target_dir,
                source="ispd16",
                node_loc_x_path=target_dir / "node_loc_x_LG.pkl",
                node_loc_y_path=target_dir / "node_loc_y_LG.pkl",
                target_site_util_path=target_dir / "target_site_util.pkl",
                rudy_map_path=target_dir / "RUDY_map_LG.pkl",
                pin_density_map_path=target_dir / "pin_density_map_LG.pkl",
                target_hpwl_path=target_dir / "target_hpwl_LG.pkl",
            )

    def dataset_parameters(self):
        return {
            "source": self.source,
            "root": str(self.root),
            "target_root": str(self.target_root),
            "which": self.which,
            "split_seed": self.split_seed,
            "length": self.length,
            "auto_download": self.auto_download,
            "download_root": self.download_root,
        }
