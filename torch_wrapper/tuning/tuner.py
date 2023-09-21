import itertools
from .cv import *
from ..utils.path import fromRaw, mkdirIfNotExists, Path
from .literals import BATCH_SIZE, TRAIN_BATCH_SIZE, VALIDATION_BATCH_SIZE


class Tuner:
    def __init__(
        self,
        net: nn.Module,
        cv: CrossValidation,
        tunableParams: dict,
        condition=lambda **_: True,
        device: str = "cuda",
        filePath: str | Path = Path("results") / Path("result.txt"),
        processList=False,
    ) -> None:
        self.processList = processList
        self.filePath = filePath
        self.cv = cv
        self.tunableParams = tunableParams
        self.condition = condition
        self.device = device
        self.net = net
        self.numberOfHyperParams = self._calcNumberOfHyperParams()
        Debug.Print(device=device, cvType=self.cv.__class__.__name__)

    def fit(self):
        file = None
        if self.filePath != None:
            self.filePath = fromRaw(self.filePath)
            mkdirIfNotExists(self.filePath)
            file = open(self.filePath, "a+")
            file.truncate(0)
            file.write(
                f"{self.processList}\n" + f"{list(self.cv.model.metrics.keys())}\n"
            )

        if self.processList:
            self.scoresForConfigs = self.cv.model.metrics.zerosFrom4d(
                self.numberOfHyperParams, self.cv.numberOfEpochs
            )
        elif file != None:
            print([self.numberOfHyperParams, self.cv.numberOfEpochs], file=file)

        for i, hyperParams in enumerate(
            self.hyperparamGen(self.condition, **self.tunableParams)
        ):
            scoresForConfig = self.setupAndFitCV(i, hyperParams)
            if self.processList:
                self.scoresForConfigs[i] = scoresForConfig
            elif file != None:
                print(scoresForConfig.tolist(), file=file)

        if file != None:
            if self.processList:
                self.scoresForConfigs = self.scoresForConfigs.permute(3, 1, 2, 0)
                file.write(str(self.scoresForConfigs.tolist()))
            file.close()

    def setupAndFitCV(self, i, hyperParams):
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
        return self.cv.fit(**dataHyperParams)

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

    def _calcNumberOfHyperParams(self):
        numberOfHyperParams = 1
        for tunableParamVal in self.tunableParams.values():
            numberOfHyperParams *= len(tunableParamVal)
        return numberOfHyperParams
