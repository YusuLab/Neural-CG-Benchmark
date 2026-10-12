from typing import Literal
import numpy as np
import torch
from pathlib import Path

from ml_lib.datasets.datasets.randomly_generated_dataset import GeneratedDataset
from ml_lib.datasets import Dataset
from .data_transforms import SplitTransform

from nugets.datasets.register import register as dataset_register
from .datapoint_types import Point_datapoint, Set_datapoint


@dataset_register 
class CustomPointCloudDataset(Dataset[Set_datapoint]):
    """

    Custom point cloud dataset. 
    Input: .npy file with point cloud data
           Optional parameters available for length, size of each point cloud, and dimension
           If none are given, these attributes will be automatically read off the .npy file

    """
    datatype = Set_datapoint
    length: int | None = None

    def __init__(self, length=None, normalization=False, split_seed=42, which="train", **kwargs):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, i):
        dp = self.inner[i]
        raise NotImplementedError
    
    def dataset_parameters(self):
        raise NotImplementedError
