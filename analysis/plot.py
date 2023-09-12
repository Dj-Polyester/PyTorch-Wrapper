import debug
from ..tuning import Tuner
from ..tuning.metrics import Metrics
import matplotlib.pyplot as plt


class Plot:
    def plot2d(tuner: Tuner = None, metrics: Metrics | list = None):
        if tuner == None and metrics == None:
            debug.ValueError(tuner=tuner, metrics=metrics)

        if metrics != None:
            filename = metrics.filename
            if isinstance(metrics, list):
                metrics = Metrics(metrics)

    def plot():
        pass
