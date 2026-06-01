from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import torch
from torch import Tensor
import numpy as np

from ml_lib.datasets import Datapoint, register as dataset_register
from ml_lib.datasets.datasets.randomly_generated_dataset import GeneratedDataset
from torch_heterogeneous_batching.batch import Batch

@dataclass
class Set_datapoint(Datapoint):
    pointset: torch.Tensor
    @classmethod
    def collate(cls, points):
        input_sets = [p.pointset for p in points]
        set_batch = Batch.from_list(input_sets, order=1)
        return Set_batch(set_batch)

@dataclass
class Set_batch(Datapoint):
    pointset: Batch
    # rpe: Batch | None = None

@dataclass
class Graph_datapoint(Set_datapoint):
    pointset: torch.Tensor
    edges: torch.Tensor

    @classmethod
    def collate(cls, points):
        input_sets = [p.pointset for p in points]
        set_batch = Batch.from_list(input_sets, order=1)
        set_ptr = set_batch.ptr
        edges = [p.edges + offset for p, offset in zip(points, set_ptr)]
        edges_batch = torch.cat(edges, dim=1)
        return Graph_batch(set_batch, edges_batch)

@dataclass
class Graph_batch(Set_batch):
    pointset: Batch
    edges: torch.Tensor # edges are in data coordinates (pointset.data indices)


@dataclass
class LabeledGraphDatapoint(Graph_datapoint):
    label: torch.Tensor

    @classmethod
    def collate(cls, points):
        input_sets = [p.pointset for p in points]
        labelsets = [p.label for p in points]
        set_batch = Batch.from_list(input_sets, order=1)
        label_batch = Batch.from_list(labelsets, order=1)
        set_ptr = set_batch.ptr
        edges = [p.edges + offset for p, offset in zip(points, set_ptr)]
        edges_batch = torch.cat(edges, dim=1)
        return LabeledGraphBatch(set_batch, edges_batch, label_batch)


@dataclass
class LabeledGraphBatch(Graph_batch):
    label: Batch


@dataclass
class GraphTensorLabelDatapoint(Graph_datapoint):
    label: torch.Tensor

    @classmethod
    def collate(cls, points):
        input_sets = [p.pointset for p in points]
        set_batch = Batch.from_list(input_sets, order=1)
        set_ptr = set_batch.ptr
        edges = [p.edges + offset for p, offset in zip(points, set_ptr)]
        edges_batch = torch.cat(edges, dim=1)
        labels = torch.stack([p.label for p in points])
        return GraphTensorLabelBatch(set_batch, edges_batch, labels)


@dataclass
class GraphTensorLabelBatch(Graph_batch):
    label: torch.Tensor


@dataclass
class Point_datapoint(Datapoint):
    point: torch.Tensor

    @classmethod
    def collate(cls, points):
        input_points = [p.point for p in points]
        return cls(torch.stack(input_points, dim=0))


@dataclass
class DistanceDatapoint(Datapoint):
    set1: Tensor
    set2: Tensor
    distance: Tensor
    # rpe_set1: None | torch.Tensor = None
    # rpe_set2: None | torch.Tensor = None

    @staticmethod
    def collate(batch):
        set1 = Batch.from_list([x.set1 for x in batch], order=1)
        set2 = Batch.from_list([x.set2 for x in batch], order=1)
        distance = torch.stack([x.distance for x in batch])
        return DistanceBatch(set1=set1, set2=set2, distance=distance)

@dataclass
class DistanceBatch(Datapoint):
    set1: Batch
    set2: Batch
    distance: Tensor
    # rpe_set1: Batch | None = None
    # rpe_set2: Batch | None = None

@dataclass
class LabeledSetBatch(Datapoint):
   pointset: Batch
   label: Tensor

@dataclass
class LabeledSetDatapoint(Datapoint):
    pointset: Tensor
    label: Tensor

    @staticmethod
    def collate(batch):
        pointset = Batch.from_list([p.pointset for p in batch], order=1)
        label = torch.stack([p.label for p in batch])
        return LabeledSetBatch(pointset=pointset, label=label)

@dataclass
class SetToLabelSetBatch(Datapoint):
    pointset: Batch
    labelset: Batch

@dataclass
class SetToLabelSetDatapoint(Datapoint):
    pointset: Tensor
    labelset: Tensor

    @staticmethod
    def collate(batch):
        pointset = Batch.from_list([x.pointset for x in batch], order=1)
        labelset = Batch.from_list([x.labelset for x in batch], order=1)
        return SetToLabelSetBatch(pointset=pointset, labelset=labelset)

@dataclass
class QueryDatapoint(Datapoint):
    pointset: Tensor
    query: Tensor
    label: Tensor

    @staticmethod
    def collate(batch):
        pointset = Batch.from_list([x.pointset for x in batch], order=1)
        label = torch.stack([x.label for x in batch])
        queryset = torch.stack([x.query for x in batch])
        return QueryBatch(pointset=pointset, label=label, queryset=queryset)

@dataclass
class QueryBatch(Datapoint):
    pointset: Batch
    queryset: Tensor
    label: Batch|Tensor


@dataclass
class ChipDesignReference(Datapoint):
    root: Path
    design_name: str
    graph_path: Path | None = None
    placement_path: Path | None = None
    congestion_path: Path | None = None
    metadata_path: Path | None = None

    @staticmethod
    def collate(batch):
        return list(batch)


@dataclass
class ChipFPGADesignReference(Datapoint):
    root: Path
    target_root: Path
    design_name: str
    variant_name: str
    graph_path: Path
    target_dir: Path
    source: str  # "mlcad" or "ispd16"
    node_loc_x_path: Path | None = None
    node_loc_y_path: Path | None = None
    # mlcad fields
    route_utilization_map_path: Path | None = None
    target_node_utilization_path: Path | None = None
    target_net_hpwl_path: Path | None = None
    # ispd16 fields
    target_site_util_path: Path | None = None
    rudy_map_path: Path | None = None
    pin_density_map_path: Path | None = None
    target_hpwl_path: Path | None = None

    @staticmethod
    def collate(batch):
        return list(batch)


@dataclass
class CircuitNetDesignReference(Datapoint):
    root: Path
    design_name: str
    graph_path: Path

    @staticmethod
    def collate(batch):
        return list(batch)


@dataclass
class SuperblueDesignReference(Datapoint):
    root: Path
    sample_name: str
    node_feature_path: Path
    net_feature_path: Path
    bipartite_path: Path
    target_path: Path
    metadata_path: Path | None = None

    @staticmethod
    def collate(batch):
        return list(batch)


@dataclass
class MaskedGraphDatapoint(Graph_datapoint):
    label: torch.Tensor
    labeled_nodes: torch.Tensor

    @classmethod
    def collate(cls, points):
        input_sets = [p.pointset for p in points]
        labelsets = [p.label for p in points]
        set_batch = Batch.from_list(input_sets, order=1)
        label_batch = Batch.from_list(labelsets, order=1)
        set_ptr = set_batch.ptr
        edges = [p.edges + offset for p, offset in zip(points, set_ptr)]
        labeled_nodes = [
            p.labeled_nodes.to(dtype=torch.int64) + offset
            for p, offset in zip(points, set_ptr)
        ]
        return MaskedGraphBatch(
            pointset=set_batch,
            edges=torch.cat(edges, dim=1),
            label=label_batch,
            labeled_nodes=torch.cat(labeled_nodes, dim=0),
        )


@dataclass
class MaskedGraphBatch(Graph_batch):
    label: Batch
    labeled_nodes: torch.Tensor
