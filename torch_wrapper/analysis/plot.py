from ..tuning import Tuner
from ..tuning.metrics import Metrics
from ..utils import Debug, Str
import matplotlib.pyplot as plt
from types import FunctionType
from torch import Tensor


# The scores tensor will be of size (len(metrics), 2, numberOfHyperparams, numberOfEpochs)
class Plotter:
    def plot2d(
        self,
        metrics: str | FunctionType | list | Metrics,
        tuner: Tuner = None,
        filename: str = None,
        memEfficient=True,
    ):
        # obtain metrics
        if tuner == None and filename == None:
            raise Debug.ValueError(tuner=tuner, filename=filename)

        if (
            isinstance(metrics, str)
            or isinstance(metrics, FunctionType)
            or isinstance(metrics, list)
        ):
            self.metrics = Metrics(metrics, onlyCalcKeys=True)
        elif tuner == None:
            Debug.ValueError(tuner=tuner, metrics=metrics)
        else:
            self.metrics = tuner.cv.model.metrics
        if filename != None:
            with open(filename, "r") as f:
                allMetrics = f.readline()
                astr = Str(f) if memEfficient else Str(f.read())
                scores = Tensor(astr.tolist())
            allMetrics = list(allMetrics)
            pass
        elif tuner != None:
            pass
