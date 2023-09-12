import itertools
from .cv import *


class Tuner:
    def __init__(
        self,
        net: nn.Module,
        cv: CrossValidation,
        cvParams: dict,
        tunableParams: dict,
        condition=lambda **_: True,
        device: str = "cuda",
    ) -> None:
        self.cv = cv
        self.tunableParams = tunableParams
        self.cvParams = cvParams
        self.condition = condition
        self.device = device
        self.net = net
        debug.Print(
            device=device, cvType=self.cv.crossValidate.__name__, cvParams=cvParams
        )

    def fit(self):
        file = None
        filename = self.cv.model.metrics.filename
        if filename != None:
            file = open(filename, "a+")
            file.truncate(0)
        for i, hyperParams in enumerate(
            self.hyperparamGen(self.condition, **self.tunableParams)
        ):
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

            debug.Print(
                param=i,
                dataHyperParams=dataHyperParams,
                modelHyperParams=modelHyperParams,
            )
            self.cv.model.setupNet(self.net, self.device, **modelHyperParams)
            scoresForConfig = self.cv.fit(**self.cvParams, **dataHyperParams)
            if file != None:
                print(scoresForConfig.tolist(), file=file)
        if file != None:
            file.close()

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
            debug.Print(param=i, params=hyperParams)
