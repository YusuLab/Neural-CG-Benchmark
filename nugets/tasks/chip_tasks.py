from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from ml_lib.datasets import Transform
from torch import Tensor

from nugets.datasets.datapoint_types import (
    ChipDesignReference,
    ChipFPGADesignReference,
    CircuitNetDesignReference,
    Graph_datapoint,
    GraphTensorLabelBatch,
    GraphTensorLabelDatapoint,
    LabeledGraphBatch,
    LabeledGraphDatapoint,
    MaskedGraphBatch,
    MaskedGraphDatapoint,
    SuperblueDesignReference,
)
from nugets.models.backbone import BackBone
from nugets.models.encoder_decoders.chip_encoders import (
    GraphIndexedNodeRegressionEncoderDecoder,
    GraphMapRegressionEncoderDecoder,
    GraphMaskedNodeRegressionEncoderDecoder,
    GraphNodeRegressionEncoderDecoder,
)

from .register import register
from .task import Task


def _ensure_2d_float_tensor(array) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    return tensor


def _ensure_edge_index_tensor(array) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=torch.int64)
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D edge tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] != 2 and tensor.shape[1] == 2:
        tensor = tensor.t().contiguous()
    if tensor.shape[0] != 2:
        raise ValueError(f"Expected edge tensor shape (2, E), got {tuple(tensor.shape)}")
    return tensor


def _load_pickle(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _graph_datapoint_from_pyg(data) -> Graph_datapoint:
    if hasattr(data, "x") and data.x is not None:
        features = data.x
    elif hasattr(data, "feat") and data.feat is not None:
        features = data.feat
    else:
        num_nodes = int(getattr(data, "num_nodes"))
        features = torch.ones((num_nodes, 1), dtype=torch.float32)
    return Graph_datapoint(
        pointset=_ensure_2d_float_tensor(features),
        edges=_ensure_edge_index_tensor(data.edge_index),
    )


class ChipPlacementTransform(Transform):
    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        reference: ChipDesignReference = self.inner[idx]
        if reference.graph_path is None:
            raise FileNotFoundError(
                f"Chip design {reference.design_name!r} does not have a graph file"
            )
        graph_payload = torch.load(reference.graph_path, map_location="cpu")
        graph = _graph_datapoint_from_pyg(graph_payload)
        label = self._load_placement_label(reference, graph_payload)
        if label.shape[0] != graph.pointset.shape[0]:
            raise ValueError(
                f"Placement label for {reference.design_name!r} has {label.shape[0]} "
                f"nodes, but graph has {graph.pointset.shape[0]} nodes"
            )
        return LabeledGraphDatapoint(
            pointset=graph.pointset,
            edges=graph.edges,
            label=label,
        )

    def _load_placement_label(self, reference: ChipDesignReference, graph_payload) -> Tensor:
        if reference.placement_path is not None:
            return self._load_placement_tensor(reference.placement_path)
        return self._placement_tensor_from_payload(graph_payload)

    def _load_placement_tensor(self, path: Path) -> Tensor:
        return self._placement_tensor_from_payload(self._load_tensor_payload(path))

    def _load_tensor_payload(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".pt":
            return torch.load(path, map_location="cpu")
        if suffix == ".npy":
            return np.load(path)
        if suffix == ".npz":
            payload = np.load(path)
            for key in ("placement", "pos", "positions", "label", "labels", "y"):
                if key in payload:
                    return payload[key]
            keys = list(payload.keys())
            if len(keys) == 1:
                return payload[keys[0]]
            raise KeyError(f"No placement-like key found in {path}. Available keys: {keys}")
        if suffix == ".json":
            with path.open() as handle:
                return json.load(handle)
        if suffix == ".csv":
            return np.genfromtxt(path, delimiter=",")
        raise ValueError(f"Unsupported tensor payload file type: {path}")

    def _placement_tensor_from_payload(self, payload) -> Tensor:
        if isinstance(payload, torch.Tensor):
            return _ensure_2d_float_tensor(payload)
        if isinstance(payload, dict):
            for key in ("placement", "pos", "positions", "label", "labels", "y"):
                if key in payload:
                    return self._placement_tensor_from_payload(payload[key])
            if all(key in payload for key in ("x", "y")):
                return _ensure_2d_float_tensor(np.column_stack([payload["x"], payload["y"]]))
            raise KeyError(f"No placement-like key found in payload. Available keys: {list(payload)}")
        if hasattr(payload, "pos") and payload.pos is not None:
            return _ensure_2d_float_tensor(payload.pos)
        if hasattr(payload, "y") and payload.y is not None:
            return _ensure_2d_float_tensor(payload.y)
        if hasattr(payload, "placement") and payload.placement is not None:
            return _ensure_2d_float_tensor(payload.placement)
        return _ensure_2d_float_tensor(payload)



class ChipFPGARouteUtilizationTransform(Transform):
    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self._load(self.inner[idx])

    def _load(self, reference: ChipFPGADesignReference) -> GraphTensorLabelDatapoint:
        graph_payload = torch.load(reference.graph_path, map_location="cpu")
        node_features = _ensure_2d_float_tensor(graph_payload.node_features)
        net_features = _ensure_2d_float_tensor(graph_payload.net_features)
        node_loc_x = _ensure_2d_float_tensor(_load_pickle(reference.node_loc_x_path))
        node_loc_y = _ensure_2d_float_tensor(_load_pickle(reference.node_loc_y_path))
        node_features = torch.cat([node_features, node_loc_x, node_loc_y], dim=1)

        feature_dim = max(node_features.shape[1], net_features.shape[1]) + 1
        pointset = torch.zeros(
            (node_features.shape[0] + net_features.shape[0], feature_dim),
            dtype=torch.float32,
        )
        pointset[: node_features.shape[0], : node_features.shape[1]] = node_features
        pointset[: node_features.shape[0], -1] = 0.0
        pointset[node_features.shape[0] :, : net_features.shape[1]] = net_features
        pointset[node_features.shape[0] :, -1] = 1.0

        source_to_net = _ensure_edge_index_tensor(graph_payload.edge_index_source_to_net).clone()
        sink_to_net = _ensure_edge_index_tensor(graph_payload.edge_index_sink_to_net).clone()
        source_to_net[1] += node_features.shape[0]
        sink_to_net[1] += node_features.shape[0]
        edges = torch.cat(
            [source_to_net, sink_to_net, source_to_net.flip(0), sink_to_net.flip(0)],
            dim=1,
        )

        label = torch.as_tensor(
            _load_pickle(reference.route_utilization_map_path),
            dtype=torch.float32,
        )
        return GraphTensorLabelDatapoint(pointset=pointset, edges=edges, label=label)


class ChipFPGASiteUtilizationTransform(Transform):
    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self._load(self.inner[idx])

    def _load(self, reference: ChipFPGADesignReference) -> GraphTensorLabelDatapoint:
        graph_payload = torch.load(reference.graph_path, map_location="cpu")
        node_features = _ensure_2d_float_tensor(graph_payload.node_features)
        net_features = _ensure_2d_float_tensor(graph_payload.net_features)
        node_loc_x = _ensure_2d_float_tensor(_load_pickle(reference.node_loc_x_path))
        node_loc_y = _ensure_2d_float_tensor(_load_pickle(reference.node_loc_y_path))
        node_features = torch.cat([node_features, node_loc_x, node_loc_y], dim=1)

        feature_dim = max(node_features.shape[1], net_features.shape[1]) + 1
        pointset = torch.zeros(
            (node_features.shape[0] + net_features.shape[0], feature_dim),
            dtype=torch.float32,
        )
        pointset[: node_features.shape[0], : node_features.shape[1]] = node_features
        pointset[: node_features.shape[0], -1] = 0.0
        pointset[node_features.shape[0] :, : net_features.shape[1]] = net_features
        pointset[node_features.shape[0] :, -1] = 1.0

        source_to_net = _ensure_edge_index_tensor(graph_payload.edge_index_source_to_net).clone()
        sink_to_net = _ensure_edge_index_tensor(graph_payload.edge_index_sink_to_net).clone()
        source_to_net[1] += node_features.shape[0]
        sink_to_net[1] += node_features.shape[0]
        edges = torch.cat(
            [source_to_net, sink_to_net, source_to_net.flip(0), sink_to_net.flip(0)],
            dim=1,
        )

        label = torch.as_tensor(
            _load_pickle(reference.target_site_util_path),
            dtype=torch.float32,
        )
        return GraphTensorLabelDatapoint(pointset=pointset, edges=edges, label=label)


class ChipCircuitNetCongestionTransform(Transform):
    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self._load(self.inner[idx])

    def _load(self, reference: CircuitNetDesignReference) -> GraphTensorLabelDatapoint:
        graph_payload = torch.load(reference.graph_path, map_location="cpu")
        inst_features = _ensure_2d_float_tensor(graph_payload["inst"].x)
        net_features = _ensure_2d_float_tensor(graph_payload["net"].x)

        feature_dim = max(inst_features.shape[1], net_features.shape[1]) + 1
        pointset = torch.zeros(
            (inst_features.shape[0] + net_features.shape[0], feature_dim),
            dtype=torch.float32,
        )
        pointset[: inst_features.shape[0], : inst_features.shape[1]] = inst_features
        pointset[: inst_features.shape[0], -1] = 0.0
        pointset[inst_features.shape[0] :, : net_features.shape[1]] = net_features
        pointset[inst_features.shape[0] :, -1] = 1.0

        inst_to_net = _ensure_edge_index_tensor(graph_payload["inst", "to", "net"].edge_index).clone()
        net_to_inst = _ensure_edge_index_tensor(graph_payload["net", "to", "inst"].edge_index).clone()
        inst_to_net[1] += inst_features.shape[0]
        net_to_inst[0] += inst_features.shape[0]
        edges = torch.cat([inst_to_net, net_to_inst], dim=1)

        label = _ensure_2d_float_tensor(graph_payload.avg_target_inst)
        return GraphTensorLabelDatapoint(pointset=pointset, edges=edges, label=label)


class ChipASICCongestionTransform(Transform):
    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        return self._load(self.inner[idx])

    def _load(
        self, reference: SuperblueDesignReference, *, target_key: str = "classify"
    ) -> MaskedGraphDatapoint:
        node_payload = _load_pickle(reference.node_feature_path)
        net_payload = _load_pickle(reference.net_feature_path)
        bipartite_payload = _load_pickle(reference.bipartite_path)
        target_payload = _load_pickle(reference.target_path)

        inst_features = _ensure_2d_float_tensor(node_payload["instance_features"])
        net_features = _ensure_2d_float_tensor(net_payload["instance_features"])

        feature_dim = max(inst_features.shape[1], net_features.shape[1]) + 1
        pointset = torch.zeros(
            (inst_features.shape[0] + net_features.shape[0], feature_dim),
            dtype=torch.float32,
        )
        pointset[: inst_features.shape[0], : inst_features.shape[1]] = inst_features
        pointset[: inst_features.shape[0], -1] = 0.0
        pointset[inst_features.shape[0] :, : net_features.shape[1]] = net_features
        pointset[inst_features.shape[0] :, -1] = 1.0

        net_to_inst = _ensure_edge_index_tensor(bipartite_payload["edge_index"]).clone()
        net_to_inst[0] += inst_features.shape[0]
        inst_to_net = net_to_inst.flip(0)
        edges = torch.cat([net_to_inst, inst_to_net], dim=1)

        label = _ensure_2d_float_tensor(target_payload[target_key])
        if label.shape[0] != inst_features.shape[0]:
            raise ValueError(
                f"Superblue target {reference.sample_name!r} has {label.shape[0]} instance labels, "
                f"but node features contain {inst_features.shape[0]} instances"
            )
        return MaskedGraphDatapoint(
            pointset=pointset,
            edges=edges,
            label=label,
            labeled_nodes=torch.arange(inst_features.shape[0], dtype=torch.int64),
        )



@register
class ChipPlacementTask(Task):
    """Predict per-node placement coordinates from chip graph topology."""

    def process_dataset(self, dataset):
        transform = ChipPlacementTransform()
        return transform(dataset)

    def datapoint_type(self):
        return LabeledGraphDatapoint

    def get_encoder_decoder(
        self,
        backbone: BackBone,
        loss_function: str = "mse_loss",
        placement_dim: int = 2,
        **kwargs,
    ):
        del kwargs
        dataset_info = self.dataset_info()
        input_dim = dataset_info.get("feature_dim")
        if input_dim is None:
            input_dim = dataset_info.get("dim")
        if input_dim is None:
            input_dim = self._infer_feature_dim()
        return GraphNodeRegressionEncoderDecoder(
            input_dim=input_dim,
            backbone_input_dim=backbone.get_input_dim(),
            backbone_output_dim=backbone.get_output_dim(),
            output_dim=placement_dim,
            loss_function=loss_function,
        )

    def _infer_feature_dim(self) -> int:
        dataset = self.get_inner_dataset("train")
        reference = dataset[0]
        if not isinstance(reference, ChipDesignReference):
            if hasattr(reference, "pointset"):
                return int(reference.pointset.shape[1])
            raise TypeError(
                "ChipPlacementTask expected ChipDesignReference samples or graph-like "
                f"samples with pointset, got {type(reference).__name__}"
            )
        if reference.graph_path is None:
            raise FileNotFoundError(
                f"Chip design {reference.design_name!r} does not have a graph file, "
                "so ChipPlacementTask cannot infer the node feature dimension"
            )
        graph_payload = torch.load(reference.graph_path, map_location="cpu")
        graph = _graph_datapoint_from_pyg(graph_payload)
        return int(graph.pointset.shape[1])

    def compute_metrics(self, datapoint: LabeledGraphBatch, results):
        prediction = getattr(results, "data", results)
        target = datapoint.label.data
        error = torch.linalg.norm(prediction - target, dim=1)
        return {
            "placement_rmse": torch.sqrt(torch.mean((prediction - target) ** 2)),
            "placement_mean_l2": error.mean(),
        }


@register
class ChipFPGARouteUtilizationTask(Task):
    """Predict a route utilization map from MLCAD netlist and placement variant data."""

    def process_dataset(self, dataset):
        transform = ChipFPGARouteUtilizationTransform()
        return transform(dataset)

    def datapoint_type(self):
        return GraphTensorLabelDatapoint

    def get_encoder_decoder(
        self,
        backbone: BackBone,
        loss_function: str = "mse_loss",
        output_shape: tuple[int, int] | None = None,
        **kwargs,
    ):
        del kwargs
        sample = self.process_dataset(self.get_inner_dataset("train"))[0]
        input_dim = int(sample.pointset.shape[1])
        if output_shape is None:
            output_shape = tuple(sample.label.shape)
        return GraphMapRegressionEncoderDecoder(
            input_dim=input_dim,
            backbone_input_dim=backbone.get_input_dim(),
            backbone_output_dim=backbone.get_output_dim(),
            output_shape=output_shape,
            loss_function=loss_function,
        )

    def compute_metrics(self, datapoint: GraphTensorLabelBatch, results):
        error = results - datapoint.label
        return {
            "route_utilization_rmse": torch.sqrt(torch.mean(error ** 2)),
            "route_utilization_mae": torch.mean(torch.abs(error)),
        }


@register
class ChipFPGASiteUtilizationTask(Task):
    """Predict an ISPD16 target site-utilization map from netlist and placement data."""

    def process_dataset(self, dataset):
        transform = ChipFPGASiteUtilizationTransform()
        return transform(dataset)

    def datapoint_type(self):
        return GraphTensorLabelDatapoint

    def get_encoder_decoder(
        self,
        backbone: BackBone,
        loss_function: str = "mse_loss",
        output_shape: tuple[int, int] | None = None,
        **kwargs,
    ):
        del kwargs
        sample = self.process_dataset(self.get_inner_dataset("train"))[0]
        input_dim = int(sample.pointset.shape[1])
        if output_shape is None:
            output_shape = tuple(sample.label.shape)
        return GraphMapRegressionEncoderDecoder(
            input_dim=input_dim,
            backbone_input_dim=backbone.get_input_dim(),
            backbone_output_dim=backbone.get_output_dim(),
            output_shape=output_shape,
            loss_function=loss_function,
        )

    def compute_metrics(self, datapoint: GraphTensorLabelBatch, results):
        error = results - datapoint.label
        return {
            "site_utilization_rmse": torch.sqrt(torch.mean(error ** 2)),
            "site_utilization_mae": torch.mean(torch.abs(error)),
        }


@register
class ChipCircuitNetCongestionTask(Task):
    """Predict average congestion per CircuitNet instance."""

    def process_dataset(self, dataset):
        transform = ChipCircuitNetCongestionTransform()
        return transform(dataset)

    def datapoint_type(self):
        return GraphTensorLabelDatapoint

    def get_encoder_decoder(
        self,
        backbone: BackBone,
        loss_function: str = "mse_loss",
        **kwargs,
    ):
        del kwargs
        sample = self.process_dataset(self.get_inner_dataset("train"))[0]
        input_dim = int(sample.pointset.shape[1])
        output_dim = int(sample.label.shape[1]) if sample.label.ndim > 1 else 1
        return GraphMaskedNodeRegressionEncoderDecoder(
            input_dim=input_dim,
            backbone_input_dim=backbone.get_input_dim(),
            backbone_output_dim=backbone.get_output_dim(),
            output_dim=output_dim,
            loss_function=loss_function,
        )

    def compute_metrics(self, datapoint: GraphTensorLabelBatch, results):
        prediction = getattr(results, "data", results)
        target = datapoint.label.reshape(-1, datapoint.label.shape[-1])
        error = prediction[: target.shape[0]] - target
        return {
            "instance_congestion_rmse": torch.sqrt(torch.mean(error ** 2)),
            "instance_congestion_mae": torch.mean(torch.abs(error)),
        }


@register
class ChipASICCongestionTask(Task):
    """Predict per-instance Superblue congestion labels from circuit topology."""

    def process_dataset(self, dataset):
        transform = ChipASICCongestionTransform()
        return transform(dataset)

    def datapoint_type(self):
        return MaskedGraphDatapoint

    def get_encoder_decoder(
        self,
        backbone: BackBone,
        loss_function: str = "mse_loss",
        **kwargs,
    ):
        del kwargs
        sample = self.process_dataset(self.get_inner_dataset("train"))[0]
        input_dim = int(sample.pointset.shape[1])
        output_dim = int(sample.label.shape[1]) if sample.label.ndim > 1 else 1
        return GraphIndexedNodeRegressionEncoderDecoder(
            input_dim=input_dim,
            backbone_input_dim=backbone.get_input_dim(),
            backbone_output_dim=backbone.get_output_dim(),
            output_dim=output_dim,
            loss_function=loss_function,
        )

    def compute_metrics(self, datapoint: MaskedGraphBatch, results):
        prediction = getattr(results, "data", results)[datapoint.labeled_nodes]
        target = datapoint.label.data
        error = prediction - target
        metrics = {
            "instance_congestion_rmse": torch.sqrt(torch.mean(error ** 2)),
            "instance_congestion_mae": torch.mean(torch.abs(error)),
        }
        if target.shape[-1] == 1:
            metrics["instance_congestion_accuracy"] = (
                (prediction >= 0.5).to(dtype=target.dtype) == target
            ).to(dtype=torch.float32).mean()
        return metrics
