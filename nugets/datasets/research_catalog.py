from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class DatasetCatalogEntry:
    name: str
    category: str
    domain_module: str
    description: str
    local_paths: tuple[str, ...]
    formats: tuple[str, ...]
    suggested_datapoint_type: str
    status: str

    def slug(self) -> str:
        return slugify(self.name)

    def existing_local_paths(self) -> tuple[str, ...]:
        return tuple(path for path in self.local_paths if Path(path).exists())


LOCAL_DATASET_CATALOG: tuple[DatasetCatalogEntry, ...] = (
    DatasetCatalogEntry(
        name="Superblue Raw Circuit Data",
        category="asic_physical_design",
        domain_module="asic_physical_design",
        description=(
            "Raw integrated-circuit design metadata and source files for the "
            "Superblue family, including compressed cell metadata, settings, "
            "instance lists, and original design documentation."
        ),
        local_paths=(
            "/data/zhishang/data/superblue/raw_data",
            "/data/zhishang/data/superblue",
        ),
        formats=("json.gz", "csv", "pdf", "json", "pt"),
        suggested_datapoint_type="Graph_datapoint",
        status="cataloged_only",
    ),
    DatasetCatalogEntry(
        name="Superblue Processed Graph Features",
        category="asic_physical_design",
        domain_module="asic_physical_design",
        description=(
            "Processed circuit graph artifacts for Superblue designs, including "
            "bipartite graphs, hypergraph star expansions, node and net features, "
            "targets, partitions, and global metadata."
        ),
        local_paths=(
            "/data/zhishang/data/superblue/2023-03-06_data",
        ),
        formats=("pkl", "pkl.gz"),
        suggested_datapoint_type="Graph_datapoint",
        status="cataloged_only",
    ),
    DatasetCatalogEntry(
        name="DEHNN Netlist Dataset",
        category="asic_physical_design",
        domain_module="asic_physical_design",
        description=(
            "Netlist and hypergraph data used by DE-HNN, derived from RocketTile "
            "and Superblue-style flows. Includes raw-to-processed graph builders, "
            "PyG datasets, embeddings, and netlist-level tensors."
        ),
        local_paths=(
            "/data/zhishang/DEHNN/de_hnn/data",
            "/data/zhishang/DEHNN/de_hnn_tx/data",
            "/data/zhishang/DEHNN/de_hnn_tx",
        ),
        formats=("pt", "pkl", "npy", "csv"),
        suggested_datapoint_type="Graph_datapoint",
        status="cataloged_only",
    ),
    DatasetCatalogEntry(
        name="CircuitNet Design Graphs",
        category="asic_physical_design",
        domain_module="asic_physical_design",
        description=(
            "CircuitNet-derived design representations, verilog metadata, macro "
            "indices, feature tensors, explanation artifacts, and design-level "
            "training/evaluation inputs for congestion and related tasks."
        ),
        local_paths=(
            "/data/zhishang/CircuitNet/de_hnn_qm",
            "/data/zhishang/CircuitNet/de_hnn_qm/verilog",
        ),
        formats=("pt", "npy", "json"),
        suggested_datapoint_type="Graph_datapoint",
        status="cataloged_only",
    ),
    DatasetCatalogEntry(
        name="DREAMPlace ASIC Benchmarks",
        category="asic_physical_design",
        domain_module="asic_physical_design",
        description=(
            "Benchmark inputs for ASIC placement experiments used by DREAMPlace, "
            "including benchmark definitions and placement-oriented design data."
        ),
        local_paths=(
            "/data/zhishang/DREAMPlace/benchmarks",
        ),
        formats=("py", "txt", "tcl", "json"),
        suggested_datapoint_type="Graph_datapoint",
        status="cataloged_only",
    ),
    DatasetCatalogEntry(
        name="DREAMPlaceFPGA Benchmarks",
        category="fpga_physical_design",
        domain_module="fpga_physical_design",
        description=(
            "FPGA benchmark netlists and associated placement inputs used by "
            "DREAMPlaceFPGA, including IF netlists and sample ISPD benchmarks."
        ),
        local_paths=(
            "/data/zhishang/DREAMPlaceFPGA/benchmarks",
        ),
        formats=("netlist", "txt", "json"),
        suggested_datapoint_type="Graph_datapoint",
        status="cataloged_only",
    ),
    DatasetCatalogEntry(
        name="ChipDiffusion Graph Dataset",
        category="asic_physical_design",
        domain_module="asic_physical_design",
        description=(
            "Graph-format circuit dataset and placement-related assets used by "
            "ChipDiffusion, including a PyG graph object, parsing configs, "
            "placement outputs, and evaluation bundles."
        ),
        local_paths=(
            "/data/zhishang/chipdiffusion/datasets",
            "/data/zhishang/chipdiffusion/placements",
            "/data/zhishang/chipdiffusion/custom_evaluation_results",
        ),
        formats=("pt", "yaml", "json", "csv", "pkl"),
        suggested_datapoint_type="Graph_datapoint",
        status="loader_supported",
    ),
    DatasetCatalogEntry(
        name="UnifiedLearning ChipGen Logs",
        category="experiment_logs",
        domain_module="",
        description=(
            "Merged ChipGen regression and ablation outputs tracked under the "
            "UnifiedLearning workspace. These are primarily experiment logs and "
            "not yet normalized into a reusable dataset loader."
        ),
        local_paths=(
            "/data/zhishang/UnifiedLearning/data/chipgen/test_logs",
            "/data/zhishang/UnifiedLearning/data",
        ),
        formats=("log", "csv"),
        suggested_datapoint_type="unmapped",
        status="logs_only",
    ),
    DatasetCatalogEntry(
        name="Shortest Paths Terrain Patches",
        category="terrain_shortest_path",
        domain_module="terrain_shortest_path",
        description=(
            "Large terrain patch collections for shortest-path distance "
            "approximation, including Norway, Holland, and Phil terrain subsets "
            "stored as compressed NumPy archives."
        ),
        local_paths=(
            "/data/zhishang/shortest-paths-nn/norway_patches",
            "/data/zhishang/shortest-paths-nn/holland_patches",
            "/data/zhishang/shortest-paths-nn/phil_patches",
            "/data/zhishang/shortest-paths-nn/dataset",
        ),
        formats=("npz", "py"),
        suggested_datapoint_type="Graph_datapoint",
        status="loader_supported",
    ),
    DatasetCatalogEntry(
        name="Synthetic Graph Benchmarks",
        category="synthetic_graph_reasoning",
        domain_module="synthetic_graph_reasoning",
        description=(
            "Small graph-learning benchmarks with stored labels, features, "
            "NetworkX graph objects, and PyG tensors. These include BA shapes, "
            "tree-grid, tree-cycles, and shortest-path synthetic graphs."
        ),
        local_paths=(
            "/data/zhishang/to_merge/gnn_agop/data/datasets",
            "/data/zhishang/to_merge/graph-learning-merged/data",
        ),
        formats=("pt", "json", "gpickle", "png"),
        suggested_datapoint_type="Graph_datapoint",
        status="loader_supported",
    ),
    DatasetCatalogEntry(
        name="Mirrored GitHub Research Datasets",
        category="mirrors",
        domain_module="",
        description=(
            "Mirror copies of active research repositories, useful for provenance "
            "and synchronization but not treated as canonical storage for loader "
            "development."
        ),
        local_paths=(
            "/data/zhishang/github/DEHNN",
            "/data/zhishang/github/chipdiffusion",
            "/data/zhishang/github/UnifiedLearning",
            "/data/zhishang/github/Algorithmic-Intelligence-Ecosystem",
        ),
        formats=("mixed",),
        suggested_datapoint_type="unmapped",
        status="mirror_only",
    ),
)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def get_catalog_map() -> dict[str, DatasetCatalogEntry]:
    return {entry.slug(): entry for entry in LOCAL_DATASET_CATALOG}


def get_uploadable_entries(*, include_logs: bool = False, include_mirrors: bool = False) -> tuple[DatasetCatalogEntry, ...]:
    entries = []
    for entry in LOCAL_DATASET_CATALOG:
        if entry.status == "logs_only" and not include_logs:
            continue
        if entry.status == "mirror_only" and not include_mirrors:
            continue
        if entry.existing_local_paths():
            entries.append(entry)
    return tuple(entries)
