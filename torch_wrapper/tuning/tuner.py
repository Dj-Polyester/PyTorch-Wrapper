import itertools
from .cv import *


class Tuner:
    def __init__(
        self,
        net: nn.Module,
        cv: CrossValidation,
        tunableParams: dict,
        condition=lambda **_: True,
        device: str = "cuda",
        filename: str = None,
    ) -> None:
        self.filename = filename
        self.cv = cv
        self.tunableParams = tunableParams
        self.condition = condition
        self.device = device
        self.net = net
        Debug.Print(device=device, cvType=self.cv.__class__.__name__)

    def fit(self):
        file = None
        filename = self.filename
        if filename != None:
            file = open(filename, "a+")
            file.truncate(0)

        for i, hyperParams in enumerate(
            self.hyperparamGen(self.condition, **self.tunableParams)
        ):
            scoresForConfig = self.setupAndFit(i, hyperParams)
            if i == 0:
                self.scoresForConfigs = self.cv.model.metrics.zerosFrom4d(
                    self.numberOfHyperParams(),
                    self.cv.numberOfEpochs,
                )
                if file != None:
                    print(
                        list(self.cv.model.metrics.keys()),
                        file=file,
                    )
            self.scoresForConfigs[i] = scoresForConfig

        if file != None:
            self.scoresForConfigs = self.scoresForConfigs.permute(3, 1, 0, 2)
            file.write(str(self.scoresForConfigs.tolist()))
            file.close()

    def setupAndFit(self, i, hyperParams):
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

    def numberOfHyperParams(self):
        if not hasattr(self, "_numberOfHyperParams"):
            numberOfHyperParamsTmp = 1
            for tunableParamVal in self.tunableParams.values():
                numberOfHyperParamsTmp *= len(tunableParamVal)
            self._numberOfHyperParams = numberOfHyperParamsTmp
        return self._numberOfHyperParams
