from ..tuning.metrics import Metrics
from ..utils import Debug, Str
from ..utils.path import fromRaw, Path
from ..tuning import Tuner
from types import FunctionType
from _io import TextIOWrapper
from torch import Tensor


class Analyzer:
    def _checkTunerAndFilename(self, tuner: Tuner):
        if self.filePath == None:
            if tuner == None:
                raise Debug.ValueError(tuner=tuner, filename=self.filePath)
            elif tuner != None:
                self.filePath = tuner.filePath
        else:
            self.filePath = fromRaw(self.filePath)

    def _constructMetricsToShow(
        self,
        metricsToShow: str | FunctionType | list | Metrics,
        metrics: Metrics,
    ):
        if (
            isinstance(metricsToShow, str)
            or isinstance(metricsToShow, FunctionType)
            or isinstance(metricsToShow, list)
        ):
            return Metrics(metricsToShow, onlyCalcKeys=True)
        elif metricsToShow == None:
            return metrics
        else:
            Debug.TypeError(metricsToShow=metricsToShow)

    def _calcMetricsAndScoresForConfigsFromFile(self, memEfficient: bool):
        self.filePath = Path(self.filePath)
        dirname = self.filePath.parents[0]
        if not dirname.exists():
            raise Exception("The directory for the")

        numberOfHyperParams = None
        numberOfEpochs = None
        with open(self.filePath, "r") as f:
            listProcessed = eval(f.readline())
            metricsList = self._readLineAsContainer(f, memEfficient)
            metrics = Metrics(metricsList, onlyCalcKeys=True)

            if listProcessed:
                scoresList = self._readAsContainer(f, memEfficient)
                scoresForConfigs = Tensor(scoresList)
            else:
                numberOfHyperParams, numberOfEpochs = eval(f.readline())
                scoresForConfigs = metrics.zerosFrom4d(
                    numberOfHyperParams, numberOfEpochs
                )
                for i in range(numberOfHyperParams):
                    lineList = self._readLineAsContainer(f, memEfficient)
                    scoresForConfigs[i] = Tensor(lineList)
                scoresForConfigs = scoresForConfigs.permute(3, 1, 2, 0)
        return (
            scoresForConfigs,
            metrics,
            numberOfHyperParams,
            numberOfEpochs,
        )

    def _calcMetricsAndScoresForConfigsFromTuner(self, tuner: Tuner):
        if not hasattr(tuner, "scoresForConfigs"):
            raise Debug.AttributeError(tuner="scoresForConfigs")
        return (
            tuner.scoresForConfigs,
            tuner.cv.model.metrics,
            tuner.numberOfHyperParams,
            tuner.cv.numberOfEpochs,
        )

    def _readAsContainer(self, f: TextIOWrapper, memEfficient: bool, _type=list):
        return Str(f).tocontainer(_type=_type) if memEfficient else eval(f.read())

    def _readLineAsContainer(self, f: TextIOWrapper, memEfficient: bool, _type=list):
        return (
            Str(f, limiter="\n").tocontainer(_type=_type)
            if memEfficient
            else eval(f.readline())
        )
