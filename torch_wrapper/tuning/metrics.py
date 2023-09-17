from types import FunctionType
import torch
from torch import Tensor
from ..utils import Debug


class Metric:
    @staticmethod
    def Accuracy(labelfun: FunctionType, preds: Tensor, labels: Tensor):
        return 100 * (labelfun(preds) == labels).float().mean()

    ACCURACY = Accuracy.__name__
    LOSS = "Loss"


class Metrics:
    @staticmethod
    def dictFromCallables(*metricsCallables) -> dict:
        return {metric.__name__: metric for metric in metricsCallables}

    str2callable = dictFromCallables(
        Metric.Accuracy,
    )

    def __init__(
        self,
        _metrics: str | FunctionType | list,
        impossibleMetrics: list = [],
        onlyCalcKeys=False,
    ):
        self.onlyCalcNames = onlyCalcKeys
        if onlyCalcKeys:
            self._container = []
            self._createAndAddIfPossible = self._createAndAddNameIfPossible
            self.keys = lambda: self._container
        else:
            self._container = {}
            self._createAndAddIfPossible = self._createAndAddNameValPairIfPossible
            self.keys = self._container.keys
            self.values = self._container.values
            self.items = self._container.items
        self.impossibleMetrics = impossibleMetrics
        if isinstance(_metrics, list):
            for metric in _metrics:
                self._createAndAddIfPossible(metric)
        elif isinstance(_metrics, str) or callable(_metrics):
            self._createAndAddIfPossible(_metrics)
        else:
            Debug.TypeError(_metrics=_metrics)

    def _createAndAddNameValPairIfPossible(self, metric: str | FunctionType):
        metricName, metricVal = self._createNameValPair(metric)
        if (
            metricName not in self.impossibleMetrics
            and metricVal not in self.impossibleMetrics
        ):
            self._container[metricName] = metricVal

    def _createAndAddNameIfPossible(self, metric: str | FunctionType):
        metricName = self._createName(metric)
        if metricName not in self.impossibleMetrics:
            self._container.append(metricName)

    def _createNameValPair(self, metric: str | FunctionType):
        if isinstance(metric, str):
            if metric in Metrics.str2callable.keys():
                return metric, Metrics.str2callable[metric]
            else:
                raise KeyError(f"The variable metric {metric} is invalid")
        elif callable(metric):
            return metric.__name__, metric
        else:
            Debug.TypeError(metric=metric)

    def _createName(self, metric: str | FunctionType):
        if isinstance(metric, str):
            if metric in Metrics.str2callable.keys() or metric == Metric.LOSS:
                return metric
            else:
                raise KeyError(f"The variable metric {metric} is invalid")
        elif callable(metric):
            return metric.__name__
        else:
            Debug.TypeError(metric=metric)

    def __len__(self):
        return len(self._container)

    def __str__(self) -> str:
        str(self._container)

    def __getitem__(self, key):
        return self._container[key]

    def __setitem__(self, key, newvalue):
        self._container[key] = newvalue

    def zerosFrom(self):
        return torch.zeros(len(self))

    def zerosFrom3d(self, numberOfEpochs):
        return torch.zeros(2, numberOfEpochs, len(self))

    def zerosFrom4d(self, numberOfHyperparams, numberOfEpochs):
        return torch.zeros(numberOfHyperparams, 2, numberOfEpochs, len(self))

    def scoresFrom(self, labelfun, yHat, y, lossCached: Tensor = None):
        if not isinstance(self._container, dict):
            raise Exception("The self._container attribute is not a dictionary")
        loss = (
            self._container[Metric.LOSS](yHat, y) if lossCached == None else lossCached
        )

        return Tensor(
            [
                loss.item() if k == Metric.LOSS else v(labelfun, yHat, y)
                for k, v in self._container.items()
            ]
        )

    def scores2Dict(self, tensor: Tensor):
        return {k: tensor[i].item() for i, k in enumerate(self.keys())}
