from utils import Debug, Str
from ..tuning import Tuner
from ..tuning.metrics import Metrics
import matplotlib.pyplot as plt
from types import FunctionType


class Plotter:
    def plot2d(
        self,
        metrics: str | FunctionType | list | Metrics,
        tuner: Tuner = None,
        filename: str = None,
    ):
        # obtain metrics
        if tuner == None and filename == None:
            Debug.ValueError(tuner=tuner, filename=filename)

        if (
            isinstance(metrics, str)
            or isinstance(metrics, FunctionType)
            or isinstance(metrics, list)
        ):
            self.metrics = Metrics(metrics)
        elif tuner == None:
            Debug.ValueError(tuner=tuner, metrics=metrics)
        else:
            self.metrics = tuner.cv.model.metrics
        if filename != None:
            allMetrics = None
            scores = None
            with open(filename, "r") as f:
                allMetrics, scores = f.readlines()
            allMetrics = list(allMetrics)
            scores = Str(scores)
            pass
        elif tuner != None:
            pass
