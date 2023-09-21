from ..tuning.literals import NUMBER_OF_EPOCHS, NUMBER_OF_HIDDEN_PARAMETERS
from ..tuning import Tuner
from ..tuning.metrics import Metrics
from ..utils import Debug
from ..utils.path import fromRaw, mkdirIfNotExists, Path
from types import FunctionType
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from .analyze import Analyzer
import numpy as np


class Plotter(Analyzer):
    def plot2d(
        self,
        tuner: Tuner = None,
        filePath: str | Path = None,
        metricsToShow: str | FunctionType | list | Metrics = None,
        memEfficient: bool = True,
        trainVal: bool = True,
    ):
        self._plot(
            tuner=tuner,
            filePath=filePath,
            metricsToShow=metricsToShow,
            memEfficient=memEfficient,
            trainVal=trainVal,
            plotFn=self._plotSingle2d,
        )

    def _plot(
        self,
        **kwargs,
    ):
        tuner: Tuner = kwargs.get("tuner", None)
        self.filePath: str | Path = kwargs.get("filePath", None)
        metricsToShow: str | FunctionType | list | Metrics = kwargs.get(
            "metricsToShow", None
        )
        memEfficient: bool = kwargs.get("memEfficient", True)
        trainVal: bool = kwargs.get("trainVal", True)
        plotFn = kwargs.get("plotFn", None)

        self._checkTunerAndFilename(tuner)
        if self.filePath != None:
            (
                scoresForConfigs,
                metrics,
                numberOfHyperParams,
                numberOfEpochs,
            ) = self._calcMetricsAndScoresForConfigsFromFile(memEfficient)
        elif tuner != None:
            (
                scoresForConfigs,
                metrics,
                numberOfHyperParams,
                numberOfEpochs,
            ) = self._calcMetricsAndScoresForConfigsFromTuner(tuner)

        self.metricsToShow = self._constructMetricsToShow(metricsToShow, metrics)

        self.colLabels = ["Train", "Validation"] if trainVal else ["Test"]
        # The scores tensor will be of size (len(metrics), 1 or 2, numberOfEpochs, numberOfHyperparams)

        if self.metricsToShow == metrics:
            self.scoresForConfigsSelected = scoresForConfigs
        else:
            metricsArray = np.array(self.metrics.keys())
            indices2Show = np.isin(metricsArray, metricsToShow.keys())
            self.metricsToShow = metricsArray[indices2Show]
            self.scoresForConfigsSelected = scoresForConfigs[indices2Show]

        fig, axs = plt.subplots(len(self.metricsToShow), len(self.colLabels))

        for metricIndex, axTuple in enumerate(axs):
            for colLabelIndex, ax in enumerate(axTuple):
                plotFn(fig, ax, metricIndex, colLabelIndex)
                ax.set_xlabel(NUMBER_OF_EPOCHS)
                ax.set_ylabel(NUMBER_OF_HIDDEN_PARAMETERS)
        plt.show()

    def _plotSingle2d(
        self,
        fig: Figure,
        ax: Axes,
        metricIndex: int,
        colLabelIndex: int,
    ):
        im = ax.imshow(
            self.scoresForConfigsSelected[metricIndex, colLabelIndex].T.detach()
        )
        ax.set_title(
            f"{self.colLabels[colLabelIndex]} {self.metricsToShow[metricIndex]}"
        )
        if colLabelIndex == len(self.colLabels) - 1:
            fig.colorbar(im)

    def save(self, filePath: str | Path):
        saveLoc = fromRaw(filePath)
        mkdirIfNotExists(saveLoc)
        plt.savefig(saveLoc)
