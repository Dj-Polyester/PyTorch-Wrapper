from types import FunctionType
import torch
from torch import Tensor
import debug


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
        filename: str = None,
    ):
        self.filename = filename
        self.map = {}
        self.impossibleMetrics = impossibleMetrics
        if _metrics != None:
            if isinstance(_metrics, list):
                for metric in _metrics:
                    self.addIfPossible(*self._create(metric))
            elif isinstance(_metrics, str):
                self.addIfPossible(*self._create(_metrics))
            elif callable(_metrics):
                self.metricsNames = [_metrics.__name__]
                self.addIfPossible(*self._create(_metrics))
            else:
                debug.TypeError(_metrics=_metrics)

    def addIfPossible(self, metricName, metricVal):
        if (
            metricName not in self.impossibleMetrics
            and metricVal not in self.impossibleMetrics
        ):
            self.map[metricName] = metricVal

    def _create(self, metric: str | FunctionType):
        if isinstance(metric, str):
            if metric in Metrics.str2callable.keys():
                return metric, Metrics.str2callable[metric]
            else:
                raise KeyError(f"The variable metric {metric} is invalid")
        elif callable(metric):
            return metric.__name__, metric
        else:
            debug.TypeError(metric=metric)

    def __len__(self):
        return len(self.map)

    def keys(self):
        return self.map.keys()

    def values(self):
        return self.map.values()

    def items(self):
        return self.map.items()

    def __getitem__(self, key):
        return self.map[key]

    def __setitem__(self, key, newvalue):
        self.map[key] = newvalue

    def __str__(self) -> str:
        return str(self.map)

    def zerosFrom(self):
        return torch.zeros(len(self.map))

    def zerosFrom3d(self, numberOfEpochs):
        return torch.zeros(2, numberOfEpochs, len(self.map))

    def scoresFrom(self, labelfun, yHat, y, lossCached: Tensor = None):
        loss = self.map[Metric.LOSS](yHat, y) if lossCached == None else lossCached

        return Tensor(
            [
                loss.item() if k == Metric.LOSS else v(labelfun, yHat, y)
                for k, v in self.map.items()
            ]
        )

    def scores2Dict(self, tensor: Tensor):
        return {k: tensor[i].item() for i, k in enumerate(self.map.keys())}
