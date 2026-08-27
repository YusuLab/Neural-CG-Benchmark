from typing import Literal
from pathlib import Path

from torch_geometric.datasets import ShapeNet as Pyg_ShapeNet
from ml_lib.datasets import Dataset
from ml_lib.datasets.splitting import SplitTransform

from .dataset_utils import download_from_url, extract_data_from_zip, extract_nested_zips
from nugets.datasets.datapoint_types import Set_datapoint
from nugets.datasets.register import register as dataset_register
import numpy as np
import torch


@dataset_register
class RNASeqPointCloud(Dataset[Set_datapoint]):
    """
    High dimensional RNAseq data. 
    """
    datatype: Set_datapoint
    seed: int = 42 # random seed set for sampling point clouds. 
    length: int | None = None
    split_seed: int = 42

    def __init__(self, length=100, size=100, which="train", seed=42, **kwargs):
        #TODO: Change this to pull from huggingface 
        raw_data = np.load('/data/sam/rna-2k/data/rna.npy')

        rng = np.random.default_rng(seed=42)
        self.length = length
        self.size = size

        selected_indices = rng.choice(len(inner), size = (self.length, self.size))
        self.inner = raw_data[selected_indices]
        self.size = size
        self.dimension = raw_data.shape[1]

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        dp = self.inner[i]
        return Set_datapoint(pointset=dp)

    def dataset_parameters(self):
        return {'dim': self.dimension, 'size': self.size, 'length':self.length}