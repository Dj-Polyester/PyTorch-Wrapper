# generic
import operator
from functools import reduce

# number crunching
from torch import Tensor
import torch.nn as nn

DEPTH = "depth"
NUMBER_OF_HIDDEN_PARAMS = "numOfHiddenParams"
DROPOUT_RATE = "dropoutRate"
ENABLE_BATCHNORM = "enableBatchNorm"
ACTIVATION_FUNCTION = "activationFunction"


class FFN(nn.Module):
    def __init__(
        self,
        _in: int,
        out: int,
        depth: int,
        numOfHiddenParams: int,
        dropoutRate: float,
        enableBatchNorm: bool,
        activationFunction: str,
    ):
        super().__init__()

        self._in = _in
        self.out = out
        self.depth = depth
        self.hiddenParams = numOfHiddenParams
        self.dr = dropoutRate
        self.enableBatchNorm = enableBatchNorm
        self.actFunc = activationFunction

        if self.depth < 1:
            raise ValueError(f"The depth value {self.depth} is less than 1")
        self.layers = nn.ModuleList(
            [nn.Linear(self._in, self.out)]
            if self.depth == 1
            else self.hiddenLayer(_in=self._in)
            + self.flatten(
                [
                    self.hiddenLayer(_batchNorm=self.enableBatchNorm)
                    for _ in range(self.depth - 2)
                ]
            )
            + [nn.Linear(self.hiddenParams, self.out)]
        )

    # https://stackoverflow.com/a/952943/10713877

    def hiddenLayer(self, _in: int = None, _batchNorm: bool = False):
        if _in == None:
            _in = self.hiddenParams
        return (
            [
                nn.BatchNorm1d(self.hiddenParams),
            ]
            if _batchNorm
            else []
        ) + [
            nn.Linear(_in, self.hiddenParams),
            getattr(nn, self.actFunc)(),
            nn.Dropout(p=self.dr),
        ]

    def flatten(self, alist: list):
        return [] if alist == [] else reduce(operator.concat, alist)

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def resetParameters(self):
        for param in self.parameters():
            if hasattr(param, "reset_parameters"):
                param.reset_parameters()
