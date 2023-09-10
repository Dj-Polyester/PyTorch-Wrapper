# generic
import itertools

# number crunching
import torch
from torch import nn as nn, Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset, random_split
from torchvision.datasets import VisionDataset
from torch.optim.optimizer import Optimizer
import numpy as np

# hyperparameter tuning and cross validation
from sklearn import model_selection

# Data
VALIDATION_SIZE = "Validation size"

# Cross validation
HOLDOUT = "Holdout"
KFOLD = "KFold"
NUMBER_OF_SPLITS = "Number of splits"
NUMBER_OF_REPEATS = "Number of repeats"

# Nontunable params
# model
CLASSIFICATION = "Classification"
REGRESSION = "Regression"
# Tunable params
# data
BATCH_SIZE = "batchSize"
TRAIN_BATCH_SIZE = "trainBatchSize"
VALIDATION_BATCH_SIZE = "validationBatchSize"
# model
LEARNING_RATE = "learningRate"
OPTIMIZER = "optimizer"


class Debug:
    printEnabled = True

    @staticmethod
    def TypeError(**kwargs):
        kwargsValues = kwargs.values()
        types = ", ".join(map(lambda x: x.__class__.__name__, kwargsValues))
        names = ", ".join(map(str, kwargs.keys()))
        return TypeError(f"Variable(s) {names} have invalid type(s) {types}")

    @staticmethod
    def ValueError(**kwargs):
        values = ", ".join(map(str, kwargs.values()))
        names = ", ".join(map(str, kwargs.keys()))
        return ValueError(f"Variable(s) {names} have invalid value(s) {values}")

    @staticmethod
    def Print(**kwargs):
        if Debug.printEnabled:
            print(", ".join([f"{k}: {v}" for k, v in kwargs.items()]))


class Data:
    def __init__(
        self,
        data: Tensor = None,
        labels: Tensor = None,
        dataset: TensorDataset | VisionDataset | Subset = None,
    ):
        if isinstance(data, Tensor) and isinstance(labels, Tensor):
            dataset = TensorDataset(data, labels)
        elif (
            not isinstance(dataset, TensorDataset)
            and not isinstance(dataset, VisionDataset)
            and not isinstance(dataset, Subset)
        ):
            raise Debug.TypeError(data=data, labels=labels, dataset=dataset)
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
            raise Debug.TypeError(testSize=testSize)
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


class Metrics:
    @staticmethod
    def Accuracy(labelfun, preds: Tensor, labels: Tensor):
        return 100 * (labelfun(preds) == labels).float().mean()


class Model:
    def __init__(
        self,
        data: Data,
        _in: int = None,
        out: int = 1,
        netType: str = CLASSIFICATION,
    ):
        self.data = data
        inputSize = data[0][0].numel()
        if _in == None:
            if inputSize == None:
                raise Debug.ValueError(_in=_in, dataSize=inputSize)
            _in = inputSize[1]
        self._in = _in
        self.out = out
        # loss and labeling functions
        if netType == CLASSIFICATION:
            if out == 1:
                self.lossfun = nn.BCEWithLogitsLoss()
                self.labelfun = lambda preds: preds > 0
                # self.lossfun = nn.BCELoss()
                # self.labelfun = lambda preds: preds > .5
            else:
                self.lossfun = nn.CrossEntropyLoss()
                self.labelfun = lambda preds: torch.argmax(preds, axis=1)
            self.accfun = Metrics.Accuracy
        elif netType == REGRESSION:
            self.lossfun = nn.MSELoss()
            self.accfun = lambda _, __: 0
        else:
            raise Debug.ValueError(netType=netType)

    def train(self, trainData: Data):
        batchAccAccumulated = 0
        batchLossAccumulated = 0
        # enable regularization and batch normalization
        self.net.train()
        for X, y in trainData:
            X, y = X.to(device=self.device, non_blocking=True), y.to(
                device=self.device, non_blocking=True
            )
            # forward pass
            yHat = self.net(X)

            # compute loss
            loss = self.lossfun(yHat, y)
            batchAccAccumulated += self.accfun(self.labelfun, yHat, y)
            batchLossAccumulated += loss

            # backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return (
            batchAccAccumulated / trainData.numOfBatches,
            batchLossAccumulated / trainData.numOfBatches,
        )

    def eval(self, testData: Data):
        batchAccAccumulated = 0
        batchLossAccumulated = 0
        # disable regularization and batch normalization
        self.net.eval()
        with torch.inference_mode():
            for X, y in testData:
                X, y = X.to(device=self.device, non_blocking=True), y.to(
                    device=self.device, non_blocking=True
                )
                # forward pass
                yHat = self.net(X)

                # compute loss
                batchAccAccumulated += self.accfun(self.labelfun, yHat, y)
                batchLossAccumulated += self.lossfun(yHat, y)

        return (
            batchAccAccumulated / testData.numOfBatches,
            batchLossAccumulated / testData.numOfBatches,
        )

    def setupNet(self, net: nn.Module, device="cuda", **kwargs):
        _optim = kwargs.get(OPTIMIZER, "SGD")
        _lr = kwargs.get(LEARNING_RATE, 0.01)
        self.device = device

        optimHyperParamsKeys = [
            OPTIMIZER,
            LEARNING_RATE,
        ]
        for k in optimHyperParamsKeys:
            if k in kwargs:
                del kwargs[k]

        self.net: nn.Module = net(_in=self._in, out=self.out, **kwargs).to(
            device=self.device
        )

        # optimizer
        self.optimizer: Optimizer = getattr(torch.optim, _optim)(
            self.net.parameters(), lr=_lr
        )


class CrossValidation:
    def __init__(
        self,
        # data parameters
        model: Model,
        testSize: int = 0.2,
        # experiment parameters
        cvType: str = HOLDOUT,
        numberOfEpochs: int = 100,
    ):
        # data
        self.trainData, self.testData = model.data.trainTestSplit(testSize)
        # model
        self.model = model
        # cv
        self.numberOfEpochs = numberOfEpochs
        self.crossvalidate = getattr(self, cvType.lower())
        Debug.Print(numberOfEpochs=numberOfEpochs, testSize=testSize)

    def iter4epochs(self, trainData: Data, validationData: Data, **kwargs):
        batchSize = kwargs.get(BATCH_SIZE, None)
        trainBatchSize = kwargs.get(TRAIN_BATCH_SIZE, 1)
        validationBatchSize = kwargs.get(VALIDATION_BATCH_SIZE, len(validationData))

        if isinstance(batchSize, int):
            trainBatchSize = validationBatchSize = batchSize

        trainData.load(trainBatchSize)
        validationData.load(validationBatchSize)
        for i in range(self.numberOfEpochs):
            meanTrainAcc, meanTrainLoss = self.model.train(trainData)
            meanValidationAcc, meanValidationLoss = self.model.eval(validationData)
            Debug.Print(
                epoch=i,
                meanTrainAcc=meanTrainAcc,
                meanTrainLoss=meanTrainLoss,
                meanValidationAcc=meanValidationAcc,
                meanValidationLoss=meanValidationLoss,
            )

    def holdout(self, **kwargs):
        """Trains on train set, tests on validation set"""
        validationSize = kwargs.get(VALIDATION_SIZE, 0.2)
        trainData, validationData = self.trainData.trainTestSplit(validationSize)
        self.iter4epochs(trainData, validationData, **kwargs)

    def kfold(self, **kwargs):
        """Trains on train set, tests on the fold"""
        nsplits = kwargs.get(NUMBER_OF_SPLITS, 5)
        nrepeats = kwargs.get(NUMBER_OF_REPEATS, 10)
        rskf = model_selection.RepeatedStratifiedKFold(
            n_splits=nsplits, n_repeats=nrepeats
        )
        for i, (trainIndices, testIndices) in enumerate(
            rskf.split(*self.trainData.dataset.tensors)
        ):
            Debug.Print(split=i)
            trainData, validationData = self.trainData.trainTestSplitByIndices(
                trainIndices, testIndices
            )
            self.iter4epochs(trainData, validationData, **kwargs)

    def fit(self, **kwargs):
        # model architecture
        self.model.net.resetParameters()
        self.crossvalidate(**kwargs)


class Tuner:
    def __init__(
        self,
        net: nn.Module,
        cv: CrossValidation,
        cvParams: dict,
        tunableParams: dict,
        condition=lambda **_: True,
        device: str = "cuda",
    ) -> None:
        self.cv = cv
        self.tunableParams = tunableParams
        self.cvParams = cvParams
        self.condition = condition
        self.device = device
        self.net = net
        Debug.Print(device=device, cvType=self.cv.crossvalidate, cvParams=cvParams)

    def fit(self):
        for i, hyperParams in enumerate(
            self.hyperparamGen(self.condition, **self.tunableParams)
        ):
            dataHyperParamsKeys = [
                BATCH_SIZE,
                TRAIN_BATCH_SIZE,
                VALIDATION_BATCH_SIZE,
            ]
            dataHyperParams = {
                k: hyperParams[k] for k in dataHyperParamsKeys if k in hyperParams
            }
            for k in dataHyperParamsKeys:
                if k in hyperParams:
                    del hyperParams[k]
            modelHyperParams = hyperParams

            Debug.Print(
                param=i,
                dataHyperParams=dataHyperParams,
                modelHyperParams=modelHyperParams,
            )
            self.cv.model.setupNet(self.net, self.device, **modelHyperParams)
            self.cv.fit(**self.cvParams, **dataHyperParams)

    # https://stackoverflow.com/a/5228294/10713877

    def hyperparamGen(self, condition=lambda **_: True, **kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            config = dict(zip(keys, instance))
            if condition(**config):
                yield config

    def printHyperparamConfigs(self):
        for i, hyperParams in enumerate(
            self.hyperparamGen(self.condition, **self.tunableParams)
        ):
            Debug.Print(param=i, params=hyperParams)
