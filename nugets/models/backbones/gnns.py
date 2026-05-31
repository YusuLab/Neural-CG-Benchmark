from typing import Callable
import torch
from torch import nn
from torch_heterogeneous_batching.batch import Batch
from torch_geometric.nn import GAT as GAT_nn

from nugets.datasets.datapoint_types import Graph_batch
from nugets.models.backbone import (BackBone, int_hyperparameter, bool_hyperparameter, 
                model_attribute, hyperparameter,  other_backbone_hyperparameter, InnerBackbone)
from nugets.models.backbones.register import register
import nugets.losses.losses as Losses

@register
class GAT(BackBone):
    """
    
    Graph attention network backbone

    """
    n_heads: int = int_hyperparameter(description="number of attention heads for GAT")
    n_layers: int = int_hyperparameter(description="number of layers")
    input_dim: int = int_hyperparameter(description="input dimension")
    output_dim: int = int_hyperparameter(description = "output dimension")

    feed_forward_hidden_dim: int=int_hyperparameter(description="number of hidden dimensions")

    def __setup__(self):
        self.gat = GAT_nn(in_channels=self.input_dim,
                          out_channels=self.output_dim,
                          num_layers=self.n_layers,
                          heads=self.n_heads,
                          hidden_channels=self.feed_forward_hidden_dim)
    
    def forward(self, graph_batch: Graph_batch, return_reg_loss=False):
        del return_reg_loss
        node_embeddings = self.gat(graph_batch.pointset.data, graph_batch.edges)
        return Batch(data=node_embeddings, ptr=graph_batch.pointset.ptr), None

    def get_input_dim(self): return self.input_dim
    def get_output_dim(self): return self.output_dim
