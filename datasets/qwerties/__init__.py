import torch
from torch import Tensor
from pathlib import Path
import json


def generate(A, B, root=Path("datasets") / Path("qwerties"), nperclust=100, blur=1):
    qwertiesFilePath = root / Path("qwerties.json")
    if Path.exists(qwertiesFilePath):
        with open(qwertiesFilePath, "r") as f:
            dataLabels = json.load(f)
            data = Tensor(dataLabels["data"])
            labels = Tensor(dataLabels["labels"])
        return data, labels

    a = [A[0] + torch.randn(nperclust) * blur, A[1] + torch.randn(nperclust) * blur]
    b = [B[0] + torch.randn(nperclust) * blur, B[1] + torch.randn(nperclust) * blur]

    data = torch.hstack((torch.vstack(a), torch.vstack(b))).T
    labels = torch.vstack(
        (torch.zeros(len(data) // 2, 1), torch.ones(len(data) // 2, 1))
    )
    with open(qwertiesFilePath, "w") as f:
        json.dump(
            {
                "data": data.tolist(),
                "labels": labels.tolist(),
            },
            f,
        )
    return data, labels
