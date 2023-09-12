from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset, random_split
from torch import nn as nn, Tensor
import numpy as np
import debug


class Data:
    def __init__(
        self,
        data: Tensor = None,
        labels: Tensor = None,
        dataset: Dataset = None,
    ):
        if isinstance(data, Tensor) and isinstance(labels, Tensor):
            dataset = TensorDataset(data, labels)
        elif not isinstance(dataset, Dataset):
            raise debug.TypeError(data=data, labels=labels, dataset=dataset)
        self.dataset = dataset

    def load(
        self,
        batchSize: int = 1,
        deleteDataset: bool = False,
    ):
        self.numOfBatches = int(np.math.ceil(len(self.dataset) / batchSize))
        self.loader = DataLoader(
            self.dataset,
            batch_size=batchSize,
        )
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
            Data(dataset=traindataset),
            Data(dataset=testdataset),
        )

    def trainTestSplitByIndices(
        self, trainIndices: list[int], testIndices: list[int]
    ) -> tuple:
        traindataset = Subset(self.dataset, trainIndices)
        testdataset = Subset(self.dataset, testIndices)
        return (
            Data(dataset=traindataset),
            Data(dataset=testdataset),
        )

    def __len__(self) -> int:
        if hasattr(self, "dataset"):
            return len(self.dataset)
        if hasattr(self, "loader"):
            return len(self.loader) * self.numOfBatches

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
