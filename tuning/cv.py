# hyperparameter tuning and cross validation
from sklearn import model_selection
from .estimator import *

# Data
BATCH_SIZE = "batchSize"
TRAIN_BATCH_SIZE = "trainBatchSize"
VALIDATION_BATCH_SIZE = "validationBatchSize"
# Cross validation
HOLDOUT = "Holdout"
VALIDATION_SIZE = "Validation size"
KFOLD = "KFold"
NUMBER_OF_SPLITS = "Number of splits"
NUMBER_OF_REPEATS = "Number of repeats"


class CrossValidation:
    def __init__(
        self,
        # data parameters
        model: TorchEstimator,
        testSize: int = 0.2,
        # experiment parameters
        cvType: str = HOLDOUT,
        numberOfEpochs: int = 100,
    ):
        # data
        self.trainData, self.testData = model.data.trainTestSplit(testSize)
        # model
        self.model = model
        # cv
        self.numberOfEpochs = numberOfEpochs
        self.crossValidate = getattr(self, cvType.lower())
        debug.Print(numberOfEpochs=numberOfEpochs, testSize=testSize)

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

            if debug.printEnabled:
                debug.Print(
                    epoch=i + 1,
                    trainScores=self.model.metrics.scores2Dict(scores[0, i]),
                    validationScores=self.model.metrics.scores2Dict(scores[1, i]),
                )
        return scores

    def holdout(self, **kwargs):
        """Trains on train set, tests on validation set"""
        self.model.net.resetParameters()
        validationSize = kwargs.get(VALIDATION_SIZE, 0.2)
        trainData, validationData = self.trainData.trainTestSplit(validationSize)
        return self.iter4epochs(trainData, validationData, **kwargs)

    def kfold(self, **kwargs):
        """Trains on train set, tests on the fold"""
        nsplits = kwargs.get(NUMBER_OF_SPLITS, 5)
        nrepeats = kwargs.get(NUMBER_OF_REPEATS, 10)
        rskf = model_selection.RepeatedStratifiedKFold(
            n_splits=nsplits, n_repeats=nrepeats
        )
        meanScores = self.model.metrics.zerosFrom3d(self.numberOfEpochs)
        for i, (trainIndices, testIndices) in enumerate(
            rskf.split(*self.trainData.dataset.tensors)
        ):
            debug.Print(split=i)
            self.model.net.resetParameters()
            trainData, validationData = self.trainData.trainTestSplitByIndices(
                trainIndices, testIndices
            )
            meanScores += self.iter4epochs(trainData, validationData, **kwargs)

        numberOfSplits = rskf.get_n_splits()
        return meanScores / numberOfSplits

    def fit(self, **kwargs) -> Tensor:
        return self.crossValidate(**kwargs)
