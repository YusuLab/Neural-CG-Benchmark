from types import SimpleNamespace
import pickle

import torch

from nugets.datasets import get_dataset_register
from nugets.datasets.datapoint_types import (
    ChipDesignReference,
    ChipFPGADesignReference,
    CircuitNetDesignReference,
    GraphTensorLabelDatapoint,
    Graph_datapoint,
    SuperblueDesignReference,
)
from nugets.tasks.chip_tasks import (
    ChipPlacementTask,
    ChipCircuitNetCongestionTransform,
    ChipFPGASiteUtilizationTransform,
    ChipFPGARouteUtilizationTransform,
)


def test_chip_synthetic_constructor_is_lazy(tmp_path, monkeypatch):
    graph_path = tmp_path / "chipdiffusion-graph-dataset" / "datasets" / "graph" / "pyg_graph.pt"
    graph_path.parent.mkdir(parents=True)
    graph_path.touch()

    def fail_load(*args, **kwargs):
        raise AssertionError("ChipSyntheticDataset should not torch.load during construction")

    monkeypatch.setattr("torch.load", fail_load)
    register = get_dataset_register()
    dataset = register["ChipSyntheticDataset"](root=str(tmp_path / "chipdiffusion-graph-dataset"))

    assert dataset.paths == [graph_path]
    assert len(dataset) == 1


def test_chip_synthetic_dataset_returns_file_reference(tmp_path):
    graph_path = tmp_path / "chipdiffusion-graph-dataset" / "datasets" / "graph" / "pyg_graph.pt"
    graph_path.parent.mkdir(parents=True)
    graph_path.touch()
    congestion_path = (
        tmp_path / "chipdiffusion-graph-dataset" / "datasets" / "graph" / "congestion.csv"
    )
    congestion_path.touch()

    register = get_dataset_register()
    dataset = register["ChipSyntheticDataset"](
        root=str(tmp_path / "chipdiffusion-graph-dataset")
    )
    sample = dataset[0]

    assert isinstance(sample, ChipDesignReference)
    assert sample.graph_path == graph_path
    assert sample.congestion_path == congestion_path


def test_chip_synthetic_dataset_can_reference_design_directories(tmp_path):
    design_dir = tmp_path / "chipdiffusion-raw" / "design_a"
    design_dir.mkdir(parents=True)
    congestion_path = design_dir / "congestion.csv"
    congestion_path.touch()

    register = get_dataset_register()
    dataset = register["ChipSyntheticDataset"](root=str(tmp_path / "chipdiffusion-raw"))
    sample = dataset[0]

    assert sample.design_name == "design_a"
    assert sample.graph_path is None
    assert sample.congestion_path == congestion_path


def test_chip_placement_task_infers_feature_dim(tmp_path, monkeypatch):
    graph_path = tmp_path / "chipdiffusion-graph-dataset" / "datasets" / "graph" / "pyg_graph.pt"
    graph_path.parent.mkdir(parents=True)
    graph_path.touch()
    graph_payload = SimpleNamespace(
        x=torch.ones((4, 7), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64),
    )

    def load_graph(path, *args, **kwargs):
        assert path == graph_path
        return graph_payload

    class DummyBackbone:
        def get_input_dim(self):
            return 3

        def get_output_dim(self):
            return 5

    monkeypatch.setattr("nugets.tasks.chip_tasks.torch.load", load_graph)
    task = ChipPlacementTask(
        "ChipSyntheticDataset",
        {"root": str(tmp_path / "chipdiffusion-graph-dataset")},
        ood_dataset_name="ChipSyntheticDataset",
        ood_dataset_parameters={"root": str(tmp_path / "chipdiffusion-graph-dataset")},
    )

    encoder_decoder = task.get_encoder_decoder(DummyBackbone())

    assert encoder_decoder.in_proj.in_features == 7
    assert encoder_decoder.in_proj.out_features == 3


def test_dehnn_mlcad_congestion_dataset_returns_variant_reference(tmp_path):
    root = tmp_path / "all_designs_netlist_data"
    target_root = tmp_path / "target_data"
    graph_path = root / "Design_105_netlist_data" / "pyg_data.pkl"
    graph_path.parent.mkdir(parents=True)
    graph_path.touch()
    target_dir = target_root / "Design_105_5000_EarlyBlockPlacement"
    target_dir.mkdir(parents=True)
    for name in (
        "node_loc_x.pkl",
        "node_loc_y.pkl",
        "route_utilization_map.pkl",
        "target_node_utilization.pkl",
        "target_net_hpwl.pkl",
    ):
        (target_dir / name).touch()

    register = get_dataset_register()
    dataset = register["ChipFPGADataset"](
        source="mlcad",
        root=str(root),
        target_root=str(target_root),
    )
    sample = dataset[0]

    assert isinstance(sample, ChipFPGADesignReference)
    assert sample.source == "mlcad"
    assert sample.design_name == "Design_105"
    assert sample.variant_name == "Design_105_5000_EarlyBlockPlacement"
    assert sample.graph_path == graph_path
    assert sample.route_utilization_map_path == target_dir / "route_utilization_map.pkl"


def test_dehnn_mlcad_congestion_graph_uses_netlist_placement_and_route_map(tmp_path, monkeypatch):
    graph_path = tmp_path / "pyg_data.pkl"
    graph_path.touch()
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    payloads = {
        "node_loc_x.pkl": torch.tensor([10.0, 20.0]).numpy(),
        "node_loc_y.pkl": torch.tensor([1.0, 2.0]).numpy(),
        "route_utilization_map.pkl": torch.ones((3, 4)).numpy(),
    }
    for name, value in payloads.items():
        with (target_dir / name).open("wb") as handle:
            pickle.dump(value, handle)
    graph_payload = SimpleNamespace(
        node_features=torch.ones((2, 2), dtype=torch.float32),
        net_features=2 * torch.ones((1, 3), dtype=torch.float32),
        edge_index_source_to_net=torch.tensor([[0], [0]], dtype=torch.int64),
        edge_index_sink_to_net=torch.tensor([[1], [0]], dtype=torch.int64),
    )

    def load_graph(path, *args, **kwargs):
        assert path == graph_path
        return graph_payload

    monkeypatch.setattr("nugets.tasks.chip_tasks.torch.load", load_graph)
    reference = ChipFPGADesignReference(
        root=tmp_path,
        target_root=target_dir,
        design_name="Design_105",
        variant_name="Design_105_5000_EarlyBlockPlacement",
        graph_path=graph_path,
        target_dir=target_dir,
        source="mlcad",
        node_loc_x_path=target_dir / "node_loc_x.pkl",
        node_loc_y_path=target_dir / "node_loc_y.pkl",
        route_utilization_map_path=target_dir / "route_utilization_map.pkl",
    )

    sample = ChipFPGARouteUtilizationTransform()._load(reference)

    assert isinstance(sample, GraphTensorLabelDatapoint)
    assert sample.pointset.shape == (3, 5)
    assert sample.edges.shape == (2, 4)
    assert sample.label.shape == (3, 4)
    assert torch.equal(sample.pointset[:2, 2:4], torch.tensor([[10.0, 1.0], [20.0, 2.0]]))


def test_dehnn_ispd16_site_utilization_dataset_returns_variant_reference(tmp_path):
    root = tmp_path / "ispd16_netlist_data"
    target_root = tmp_path / "ispd16_target_data_random"
    graph_path = root / "FPGA01_netlist_data" / "pyg_data.pkl"
    graph_path.parent.mkdir(parents=True)
    graph_path.touch()
    target_dir = target_root / "FPGA01_seed_0_filler_0.7"
    target_dir.mkdir(parents=True)
    for name in (
        "node_loc_x_LG.pkl",
        "node_loc_y_LG.pkl",
        "target_site_util.pkl",
        "target_hpwl_LG.pkl",
        "RUDY_map_LG.pkl",
        "pin_density_map_LG.pkl",
    ):
        (target_dir / name).touch()

    register = get_dataset_register()
    dataset = register["ChipFPGADataset"](
        source="ispd16",
        root=str(root),
        target_root=str(target_root),
    )
    sample = dataset[0]

    assert isinstance(sample, ChipFPGADesignReference)
    assert sample.source == "ispd16"
    assert sample.design_name == "FPGA01"
    assert sample.variant_name == "FPGA01_seed_0_filler_0.7"
    assert sample.graph_path == graph_path
    assert sample.target_site_util_path == target_dir / "target_site_util.pkl"


def test_dehnn_ispd16_site_utilization_graph_uses_site_util_target(tmp_path, monkeypatch):
    graph_path = tmp_path / "pyg_data.pkl"
    graph_path.touch()
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    payloads = {
        "node_loc_x_LG.pkl": torch.tensor([10.0, 20.0]).numpy(),
        "node_loc_y_LG.pkl": torch.tensor([1.0, 2.0]).numpy(),
        "target_site_util.pkl": torch.ones((3, 4)).numpy(),
    }
    for name, value in payloads.items():
        with (target_dir / name).open("wb") as handle:
            pickle.dump(value, handle)
    graph_payload = SimpleNamespace(
        node_features=torch.ones((2, 2), dtype=torch.float32),
        net_features=2 * torch.ones((1, 3), dtype=torch.float32),
        edge_index_source_to_net=torch.tensor([[0], [0]], dtype=torch.int64),
        edge_index_sink_to_net=torch.tensor([[1], [0]], dtype=torch.int64),
    )

    def load_graph(path, *args, **kwargs):
        assert path == graph_path
        return graph_payload

    monkeypatch.setattr("nugets.tasks.chip_tasks.torch.load", load_graph)
    reference = ChipFPGADesignReference(
        root=tmp_path,
        target_root=target_dir,
        design_name="FPGA01",
        variant_name="FPGA01_seed_0_filler_0.7",
        graph_path=graph_path,
        target_dir=target_dir,
        source="ispd16",
        node_loc_x_path=target_dir / "node_loc_x_LG.pkl",
        node_loc_y_path=target_dir / "node_loc_y_LG.pkl",
        target_site_util_path=target_dir / "target_site_util.pkl",
    )

    sample = ChipFPGASiteUtilizationTransform()._load(reference)

    assert isinstance(sample, GraphTensorLabelDatapoint)
    assert sample.pointset.shape == (3, 5)
    assert sample.edges.shape == (2, 4)
    assert sample.label.shape == (3, 4)
    assert torch.equal(sample.pointset[:2, 2:4], torch.tensor([[10.0, 1.0], [20.0, 2.0]]))


def test_circuitnet_congestion_dataset_returns_file_reference(tmp_path):
    graph_path = tmp_path / "processed_standardized" / "design_a_standardized.pt"
    graph_path.parent.mkdir(parents=True)
    graph_path.touch()

    register = get_dataset_register()
    dataset = register["ChipCircuitNetDataset"](root=str(graph_path.parent))
    sample = dataset[0]

    assert isinstance(sample, CircuitNetDesignReference)
    assert sample.design_name == "design_a"
    assert sample.graph_path == graph_path


def test_circuitnet_instance_congestion_graph_uses_avg_target_inst(tmp_path, monkeypatch):
    graph_path = tmp_path / "design_a_standardized.pt"
    graph_path.touch()

    class CircuitPayload:
        avg_target_inst = torch.tensor([0.25, 0.5], dtype=torch.float32)

        def __getitem__(self, key):
            return {
                "inst": SimpleNamespace(x=torch.ones((2, 2), dtype=torch.float32)),
                "net": SimpleNamespace(x=2 * torch.ones((1, 3), dtype=torch.float32)),
                ("inst", "to", "net"): SimpleNamespace(
                    edge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.int64)
                ),
                ("net", "to", "inst"): SimpleNamespace(
                    edge_index=torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
                ),
            }[key]

    def load_graph(path, *args, **kwargs):
        assert path == graph_path
        return CircuitPayload()

    monkeypatch.setattr("nugets.tasks.chip_tasks.torch.load", load_graph)
    reference = CircuitNetDesignReference(
        root=tmp_path,
        design_name="design_a",
        graph_path=graph_path,
    )

    sample = ChipCircuitNetCongestionTransform()._load(reference)

    assert isinstance(sample, GraphTensorLabelDatapoint)
    assert sample.pointset.shape == (3, 4)
    assert sample.edges.shape == (2, 4)
    assert torch.equal(sample.label, torch.tensor([[0.25], [0.5]]))


def test_local_dataset_registry_contains_new_entries():
    register = get_dataset_register()
    for name in (
        "ShortestPathSynthetic",
        "ChipSyntheticDataset",
        "ChipFPGADataset",
        "ChipCircuitNetDataset",
        "ChipASICDataset",
    ):
        assert name in register



def test_superblue_loader_returns_references_and_splits(tmp_path):
    root = tmp_path / "superblue"
    root.mkdir()
    for i in range(12):
        name = f"sb{i:02d}"
        for suffix in (".node_features.pkl", ".net_features.pkl", ".bipartite.pkl", ".targets.pkl"):
            (root / f"{name}{suffix}").touch()

    register = get_dataset_register()
    dataset_type = register["ChipASICDataset"]
    train = dataset_type(root=str(root), which="train", split_seed=7)
    val = dataset_type(root=str(root), which="val", split_seed=7)

    assert len(train) > 0
    assert len(val) > 0
    sample = train[0]
    assert isinstance(sample, SuperblueDesignReference)

    train_names = {path.name for path in train.paths}
    val_names = {path.name for path in val.paths}
    assert train_names.isdisjoint(val_names)
