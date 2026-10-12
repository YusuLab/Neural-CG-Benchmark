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
           We assume that the point cloud data is given in the shape (M, N, d) where M is the number of point clouds, N is the number of points per point cloud, and d is the dimension of the point cloud. 
           Optional parameters available for length, size of each point cloud, and dimension
           If none are given, these attributes will be automatically read off the .npy file

    """
    datatype = Set_datapoint
    length: int | None = None

    def __init__(self, filepath, length=None, normalization=False, split_seed=42, which="train", **kwargs):
        inner = np.load(filepath)
        self.filepath = filepath
        self.dim = inner.shape[2]
        if normalization:
            inner = inner - np.expand_dims(np.mean(inner, axis=1), 1) # center dataset
            mins = inner.min(axis=1, keepdims=True)      
            maxs = inner.max(axis=1, keepdims=True)     
            sizes = maxs - mins                         

            scale = sizes.max(axis=2, keepdims=True) / 2
            scale = np.where(scale == 0, 1, scale)

            inner = inner / scale
                
            np.linalg.norm(inner, axis=1).max() # scale dataset
        # If this dataset is requested as part of an OOD dataset, return its validation set
        if which == "ood": 
            which = "val"
        is_train_or_val = which in ("train", "val") 
        if is_train_or_val:
            split_transform: SplitTransform = SplitTransform(
                    which=which, seed=split_seed, 
                    splits=["train", "val"], percents=[.9, .1])
            inner = split_transform(inner)
        if length is not None:
            rng = np.random.default_rng(seed=split_seed)
            indices = rng.choice(len(inner), size=length, replace=False)
            inner = inner[indices]
            self.length = length
        else:
            self.length = self.inner.shape[0]
        self.inner = torch.tensor(inner, dtype=torch.float32)
    
    def __len__(self):
        return len(self.inner)
    
    def __getitem__(self, i):
        dp = self.inner[i]
        return Set_datapoint(dp)
    
    def dataset_parameters(self):
        return{'dim': self.dim, 'length': self.length, 'normalization': True, 'filepath': self.filepath}
