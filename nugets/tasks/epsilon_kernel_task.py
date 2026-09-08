from typing import Callable

from math import sqrt
from dataclasses import dataclass
from re import L

from ml_lib.datasets import Transform, Datapoint
# import numpy as np
import ot
import torch
from torch import Tensor
from torch_heterogeneous_batching import Batch
from nugets.datasets.datapoint_types import Set_batch, Set_datapoint
from nugets.models.backbone import BackBone
import warnings


from .task import Task
from .register import register
from .transforms import SetLabelTransform

@register
class EpsilonKernelTask(Task):
    def process_dataset(self, dataset):
        return dataset
    
    def datapoint_type(self):
        return Set_datapoint

    def get_encoder_decoder(self, backbone:BackBone, loss_function: str="direction_width_loss", **kwargs):

        from nugets.models.encoder_decoders.epsilon_kernel import EpsilonKernelIdentityEncoderDecoder
        dataset_info = self.dataset_info()
        backbone_input_dim = dataset_info["dim"]
        backbone_output_dim = dataset_info["dim"]
        if loss_function != "epsilon_kernel_projection_loss":
            warnings.warn("Only direction_width_loss compatible with this task.")
        return EpsilonKernelIdentityEncoderDecoder(input_dim=dataset_info["dim"],
                                                   backbone_input_dim=backbone_input_dim,
                                                   backbone_output_dim=backbone_output_dim,
                                                   output_dim = dataset_info["dim"],
                                                   loss_function="direction_width_loss")