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

from .task import Task
from .register import register
from .transforms import SetLabelTransform

@register
class EpsilonKernelTask(Task):
    def process_dataset(self, dataset):
        return dataset
    
    def datapoint_type(self):
        return Set_datapoint

    def get_encoder_decoder(self, backbone:BackBone, loss_function: str, **kwargs):
        raise NotImplementedError("Not done yet")