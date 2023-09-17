from typing import Any
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset, random_split
from torch import nn as nn, Tensor
import numpy as np
from utils import debug


class Data:
    def __init__(self, dataset: Dataset = None):
        self.dataset = dataset

    def load(
        self,
        batchSize: int = 1,
        deleteDataset: bool = True,
    ):
        if not hasattr(self, "loader") or batchSize != self.loader.batch_size:
            self.loader = DataLoader(
                self.dataset,
                batch_size=batchSize,
            )
            self.batchSize = self.loader.batch_size
            self.numOfBatches = int(np.math.ceil(len(self.dataset) / self.batchSize))
        if deleteDataset:
            del self.dataset

    def trainTestSplit(self, testSize: int | float = 0.2):
        if isinstance(testSize, int):
            lengths = [len(self.dataset) - testSize, testSize]
        elif isinstance(testSize, float):
            lengths = [1 - testSize, testSize]
        else:
            raise debug.TypeError(testSize=testSize)
        traindataset, testdataset = random_split(self.dataset, lengths=lengths)
        return (
            Data(traindataset),
            Data(testdataset),
        )

    def trainTestSplitByIndices(
        self, trainIndices: list[int], testIndices: list[int]
    ) -> tuple:
        traindataset = Subset(self.dataset, trainIndices)
        testdataset = Subset(self.dataset, testIndices)
        return (
            Data(traindataset),
            Data(testdataset),
        )

    def __len__(self) -> int:
        if hasattr(self, "dataset"):
            return len(self.dataset)
        if hasattr(self, "loader"):
            return self.numOfBatches * self.batchSize

    def __getitem__(self, offset: int):
        if hasattr(self, "dataset"):
            return self.dataset[offset]
        raise Exception(
            "Cannot index non-existing dataset. Method load is called with deleteDataset=True"
        )

    def __iter__(self):
        if hasattr(self, "loader"):
            return iter(self.loader)
        if hasattr(self, "dataset"):
            return iter(self.dataset)

    # def __setattr__(self, __name: str, __value: Any) -> None:
    #     if __name in (
    #         "batchSize",
    #         "numOfBatches",
    #     ):
    #         raise ValueError(f"The attribute {__name} cannot be set")
    #     super.__setattr__(__name, __value)
