from __future__ import annotations

import torch
from torch import Tensor

from nugets.datasets.datapoint_types import (
    GraphTensorLabelBatch,
    LabeledGraphBatch,
    MaskedGraphBatch,
)
from nugets.models.model import EncoderDecoder


class GraphNodeRegressionEncoderDecoder(EncoderDecoder):
    def __init__(
        self,
        *,
        input_dim: int,
        backbone_input_dim: int,
        backbone_output_dim: int,
        output_dim: int,
        loss_function: str = "mse_loss",
        **kwargs,
    ):
        super().__init__()
        self.in_proj = torch.nn.Linear(input_dim, backbone_input_dim)
        self.out_proj = torch.nn.Linear(backbone_output_dim, output_dim)
        if loss_function != "mse_loss":
            raise ValueError("GraphNodeRegressionEncoderDecoder supports only mse_loss")
        self.loss_fn = torch.nn.functional.mse_loss

    def encode(self, batch: LabeledGraphBatch):
        pointset = self.in_proj(batch.pointset)
        return LabeledGraphBatch(pointset=pointset, edges=batch.edges, label=batch.label), None

    def decode(self, backbone_result):
        return self.out_proj(backbone_result)

    def compute_loss(self, batch: LabeledGraphBatch, backbone_result, encoder_info):
        del encoder_info
        prediction = self.decode(backbone_result)
        prediction_data = getattr(prediction, "data", prediction)
        return self.loss_fn(prediction_data, batch.label.data)


class GraphMapRegressionEncoderDecoder(EncoderDecoder):
    def __init__(
        self,
        *,
        input_dim: int,
        backbone_input_dim: int,
        backbone_output_dim: int,
        output_shape: tuple[int, int],
        loss_function: str = "mse_loss",
        **kwargs,
    ):
        super().__init__()
        self.in_proj = torch.nn.Linear(input_dim, backbone_input_dim)
        self.out_proj = torch.nn.Linear(backbone_output_dim, output_shape[0] * output_shape[1])
        self.output_shape = output_shape
        if loss_function != "mse_loss":
            raise ValueError("GraphMapRegressionEncoderDecoder supports only mse_loss")
        self.loss_fn = torch.nn.functional.mse_loss

    def encode(self, batch: GraphTensorLabelBatch):
        pointset = self.in_proj(batch.pointset)
        return GraphTensorLabelBatch(pointset=pointset, edges=batch.edges, label=batch.label), None

    def decode(self, backbone_result):
        pooled = backbone_result.mean()
        return self.out_proj(pooled).reshape((-1, *self.output_shape))

    def compute_loss(self, batch: GraphTensorLabelBatch, backbone_result, encoder_info):
        del encoder_info
        prediction = self.decode(backbone_result)
        return self.loss_fn(prediction, batch.label)


class GraphMaskedNodeRegressionEncoderDecoder(EncoderDecoder):
    def __init__(
        self,
        *,
        input_dim: int,
        backbone_input_dim: int,
        backbone_output_dim: int,
        output_dim: int = 1,
        loss_function: str = "mse_loss",
        **kwargs,
    ):
        super().__init__()
        self.in_proj = torch.nn.Linear(input_dim, backbone_input_dim)
        self.out_proj = torch.nn.Linear(backbone_output_dim, output_dim)
        if loss_function != "mse_loss":
            raise ValueError("GraphMaskedNodeRegressionEncoderDecoder supports only mse_loss")
        self.loss_fn = torch.nn.functional.mse_loss

    def encode(self, batch: GraphTensorLabelBatch):
        pointset = self.in_proj(batch.pointset)
        return GraphTensorLabelBatch(pointset=pointset, edges=batch.edges, label=batch.label), None

    def decode(self, backbone_result):
        return self.out_proj(backbone_result)

    def compute_loss(self, batch: GraphTensorLabelBatch, backbone_result, encoder_info):
        del encoder_info
        prediction = self.decode(backbone_result).data
        target = batch.label.reshape(-1, batch.label.shape[-1])
        return self.loss_fn(prediction[: target.shape[0]], target)


class GraphIndexedNodeRegressionEncoderDecoder(EncoderDecoder):
    def __init__(
        self,
        *,
        input_dim: int,
        backbone_input_dim: int,
        backbone_output_dim: int,
        output_dim: int = 1,
        loss_function: str = "mse_loss",
        **kwargs,
    ):
        super().__init__()
        self.in_proj = torch.nn.Linear(input_dim, backbone_input_dim)
        self.out_proj = torch.nn.Linear(backbone_output_dim, output_dim)
        if loss_function != "mse_loss":
            raise ValueError("GraphIndexedNodeRegressionEncoderDecoder supports only mse_loss")
        self.loss_fn = torch.nn.functional.mse_loss

    def encode(self, batch: MaskedGraphBatch):
        pointset = self.in_proj(batch.pointset)
        return (
            MaskedGraphBatch(
                pointset=pointset,
                edges=batch.edges,
                label=batch.label,
                labeled_nodes=batch.labeled_nodes,
            ),
            None,
        )

    def decode(self, backbone_result):
        return self.out_proj(backbone_result)

    def compute_loss(self, batch: MaskedGraphBatch, backbone_result, encoder_info):
        del encoder_info
        prediction = self.decode(backbone_result).data[batch.labeled_nodes]
        return self.loss_fn(prediction, batch.label.data)
