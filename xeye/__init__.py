# from .dataset import Dataset
from .exp import Dataset
from .fast_dataset import FastDataset
from .manual_dataset import ManualDataset
from .build_dataset import BuildDataset

#version
__version__ = "1.1"

__all__ = [
    "Dataset",
    "FastDataset",
    "ManualDataset",
    "BuildDataset"
]