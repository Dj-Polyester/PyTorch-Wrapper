# %%
from tuning import *
from models.FFN import *
import matplotlib.pyplot as plt
import datasets.qwerties as qwerties

# %%
data, labels = qwerties.generate(A=[1, 1], B=[2, 3])


# %%
zeroCoo = data[torch.where(labels == 0)[0]]
oneCoo = data[torch.where(labels == 1)[0]]
zeroCoo[:, 0].shape


# %%

plt.plot(zeroCoo[:, 0], zeroCoo[:, 1], "go")
plt.plot(oneCoo[:, 0], oneCoo[:, 1], "bo")
plt.show()


# %%
dataset = Data(data, labels)

# %%
DEVICE = "cpu"


tuner = Tuner(
    net=FFN,
    cv=CrossValidation(
        TorchClassifier(data=dataset, _in=2, filename="results.txt"),
    ),
    cvParams={VALIDATION_SIZE: 0.2},
    tunableParams={
        # data
        BATCH_SIZE: [20, 40],
        # model
        LEARNING_RATE: [0.01, 0.1],
        OPTIMIZER: ["SGD"],
        DEPTH: [4, 8],
        NUMBER_OF_HIDDEN_PARAMS: [32, 128],
        DROPOUT_RATE: [0.2, 0.5, 0.7],
        ENABLE_BATCHNORM: [True],
        ACTIVATION_FUNCTION: ["ReLU"],
    },
    device=DEVICE,
)

# %%
tuner.fit()
