# %%
import torch
from torch_wrapper.tuning import *
from torch_wrapper.models.nets.FFN import *
import torch_wrapper.datasets.qwerties as qwerties
import matplotlib.pyplot as plt


# %%
dataset, numclasses = qwerties.generate(
    [[1, 1], [2, 3], [0, 2]], [25, 50, 100], [0.3, 1, 2]
)


# %%
coos = [0] * numclasses
colors = [
    "b",
    "g",
    "r",
    "c",
    "m",
    "y",
    "k",
    "w",
]
shapes = [
    ".",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
    "_",
]


# %%
data = dataset.tensors[0]
labels = dataset.tensors[1]
for cls in range(numclasses):
    coo = data[torch.where(labels == cls)[0]]
    colorIndex = cls % len(colors)
    shapeIndex = cls % len(shapes)
    marker = f"{colors[colorIndex]}{shapes[shapeIndex]}"
    plt.plot(coo[:, 0], coo[:, 1], marker)
plt.show()


# %%
compoundData = Data(dataset)


# %%
DEVICE = "cpu"


tuner = Tuner(
    net=FFN,
    cv=HoldoutCrossValidation(
        TorchClassifier(data=compoundData, _in=2, out=numclasses),
    ),
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

# %%
from torch_wrapper.analysis import Plotter

# %%
plotter = Plotter()

# %%
plotter.plot2d(tuner, tuner.filePath)

# %%
plotter.plot2d(tuner, tuner.filePath, memEfficient=False)
