from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from ml_lib.datasets import Dataset

from nugets.datasets.datapoint_types import ChipFPGADesignReference
from nugets.datasets.local_dataset_utils import split_file_list
from nugets.datasets.register import register


@register
class ChipFPGADataset(Dataset[ChipFPGADesignReference]):
    datatype = ChipFPGADesignReference

    _MLCAD_DEFAULT_ROOT = "data/server-local/dehnn-netlist-dataset/all_designs_netlist_data"
    _ISPD16_DEFAULT_ROOT = "data/server-local/dehnn-netlist-dataset/ispd16_netlist_data"

    def __init__(
        self,
        *,
        source: Literal["mlcad", "ispd16"],
        root: str | None = None,
        target_root: str | None = None,
        which: str = "train",
        split_seed: int = 42,
        length: int | None = None,
    ):
        if source not in ("mlcad", "ispd16"):
            raise ValueError(f"source must be 'mlcad' or 'ispd16', got {source!r}")
        self.source = source

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

        if not self.root.exists():
            raise FileNotFoundError(
                f"DE-HNN {source.upper()} netlist root does not exist: {self.root}\n"
                "Run: python download_research_datasets.py --entry dehnn-netlist-dataset"
            )
        if not self.target_root.exists():
            raise FileNotFoundError(
                f"DE-HNN {source.upper()} target root does not exist: {self.target_root}"
            )
        self.which = which
        self.split_seed = split_seed
        self.length = length
        self.paths = split_file_list(
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
        }
