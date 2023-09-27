import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer

from .data import *
from .metrics import Metric, Metrics
from .literals import OPTIMIZER, LEARNING_RATE


class TorchEstimator:
    impossibleMetrics = []

    def __init__(
        self,
        data: Data,
        _in: int = None,
        out: int = 1,
        _metrics: str = Metric.Accuracy,
    ):
        self.data = data
        inputSize = data[0][0].numel()
        if _in == None:
            if inputSize == None:
                raise Debug.ValueError(_in=_in, dataSize=inputSize)
            _in = inputSize[1]
        self._in = _in
        self.out = out
        self.metrics = Metrics(_metrics, self.impossibleMetrics)

    def label(self, preds):
        return preds

    def train(self, trainData: Data) -> Tensor:
        batchAccumulatedScores = self.metrics.zerosFrom().to(device=self.device)

        # enable regularization and batch normalization
        self.net.train()
        for X, y in trainData:
            X, y = X.to(device=self.device, non_blocking=True), y.to(
                device=self.device, non_blocking=True
            )
            # forward pass
            yHat = self.net(X)

            # compute scores
            loss = self.metrics[Metric.LOSS](yHat, y)
            batchAccumulatedScores += self.metrics.scoresFrom(
                self.label,
                yHat,
                y,
                loss,
            ).to(device=self.device)

            # backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return batchAccumulatedScores / trainData.numOfBatches

    def eval(self, testData: Data) -> Tensor:
        batchAccumulatedScores = self.metrics.zerosFrom().to(device=self.device)

        # disable regularization and batch normalization
        self.net.eval()
        with torch.inference_mode():
            for X, y in testData:
                X, y = X.to(device=self.device, non_blocking=True), y.to(
                    device=self.device, non_blocking=True
                )
                # forward pass
                yHat = self.net(X)

                # compute scores
                batchAccumulatedScores += self.metrics.scoresFrom(
                    self.label,
                    yHat,
                    y,
                ).to(device=self.device)

        return batchAccumulatedScores / testData.numOfBatches

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


class TorchClassifier(TorchEstimator):
    impossibleMetrics = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.out == 1:
            self.metrics[Metric.LOSS] = nn.BCEWithLogitsLoss()
            self.label = lambda preds: preds > 0
            # self.metrics[Metric.LOSS] = nn.BCELoss()
            # self.labelfun = lambda preds: preds > .5
        else:
            self.metrics[Metric.LOSS] = nn.CrossEntropyLoss()
            self.label = lambda preds: torch.argmax(preds, axis=1)


class TorchRegressor(TorchEstimator):
    impossibleMetrics = [Metric.Accuracy]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics[Metric.LOSS] = nn.MSELoss()
