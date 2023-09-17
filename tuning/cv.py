# hyperparameter tuning and cross validation
from sklearn import model_selection

from tuning.estimator import TorchEstimator
from .estimator import *

# Data
BATCH_SIZE = "batchSize"
TRAIN_BATCH_SIZE = "trainBatchSize"
VALIDATION_BATCH_SIZE = "validationBatchSize"
# Cross validation
VALIDATION_SIZE = "validationSize"
NUMBER_OF_SPLITS = "nSplits"
NUMBER_OF_REPEATS = "nRepeats"


class CrossValidation:
    def __init__(
        self,
        # data parameters
        model: TorchEstimator,
        testSize: int = 0.2,
        # experiment parameters
        numberOfEpochs: int = 100,
    ):
        # data
        self.trainData, self.testData = model.data.trainTestSplit(testSize)
        # model
        self.model = model
        # cv
        self.numberOfEpochs = numberOfEpochs

    def iter4epochs(self, trainData: Data, validationData: Data, **kwargs):
        batchSize = kwargs.get(BATCH_SIZE, None)
        trainBatchSize = kwargs.get(TRAIN_BATCH_SIZE, 1)
        validationBatchSize = kwargs.get(VALIDATION_BATCH_SIZE, len(validationData))

        if isinstance(batchSize, int):
            trainBatchSize = validationBatchSize = batchSize

        trainData.load(trainBatchSize)
        validationData.load(validationBatchSize)
        scores = self.model.metrics.zerosFrom3d(self.numberOfEpochs)
        for i in range(self.numberOfEpochs):
            scores[0, i] = self.model.train(trainData)
            scores[1, i] = self.model.eval(validationData)

            if Debug.printEnabled:
                Debug.Print(
                    epoch=i + 1,
                    trainScores=self.model.metrics.scores2Dict(scores[0, i]),
                    validationScores=self.model.metrics.scores2Dict(scores[1, i]),
                )
        return scores

    def fit(self, **kwargs) -> Tensor:
        raise NotImplementedError()


class HoldoutCrossValidation(CrossValidation):
    def __init__(
        self,
        model: TorchEstimator,
        testSize: int = 0.2,
        numberOfEpochs: int = 100,
        validationSize=0.2,
    ):
        super().__init__(model, testSize, numberOfEpochs)
        self.validationSize = validationSize
        Debug.Print(
            numberOfEpochs=numberOfEpochs,
            testSize=testSize,
            validationSize=validationSize,
        )

    def fit(self, **kwargs) -> Tensor:
        raise NotImplementedError()

        trainData.load(trainBatchSize)
        validationData.load(validationBatchSize)
        scores = self.model.metrics.zerosFrom3d(self.numberOfEpochs).to(
            device=self.model.device
        )
        for i in range(self.numberOfEpochs):
            scores[0, i] = self.model.train(trainData)
            scores[1, i] = self.model.eval(validationData)

            if Debug.printEnabled:
                Debug.Print(
                    epoch=i + 1,
                    trainScores=self.model.metrics.scores2Dict(scores[0, i]),
                    validationScores=self.model.metrics.scores2Dict(scores[1, i]),
                )
        return scores

    def fit(self, **kwargs):
        """Trains on train set, tests on validation set"""
        self.model.net.resetParameters()
        trainData, validationData = self.trainData.trainTestSplit(self.validationSize)
        return self.iter4epochs(trainData, validationData, **kwargs)


class KFoldCrossValidation(CrossValidation):
    def __init__(
        self,
        model: TorchEstimator,
        testSize: int = 0.2,
        numberOfEpochs: int = 100,
        nSplits: int = 5,
        nRepeats: int = 10,
    ):
        super().__init__(model, testSize, numberOfEpochs)
        self.nSplits = nSplits
        self.nRepeats = nRepeats
        Debug.Print(
            numberOfEpochs=numberOfEpochs,
            testSize=testSize,
            nSplits=nSplits,
            nRepeats=nRepeats,
        )

    def fit(self, **kwargs):
        """Trains on train set, tests on the fold"""
        rskf = model_selection.RepeatedStratifiedKFold(
            n_splits=self.nsplits, n_repeats=self.nrepeats
        )
        sumScores = self.model.metrics.zerosFrom3d(self.numberOfEpochs)
        for i, (trainIndices, testIndices) in enumerate(
            rskf.split(*self.trainData.dataset.tensors)
        ):
            Debug.Print(split=i)
            self.model.net.resetParameters()
            trainData, validationData = self.trainData.trainTestSplitByIndices(
                trainIndices, testIndices
            )
            sumScores += self.iter4epochs(
                trainData,
                validationData,
                False,
                **kwargs,
            )

        numberOfSplits = rskf.get_n_splits()
        return sumScores / numberOfSplits

    def _splitKfold():
        pass
