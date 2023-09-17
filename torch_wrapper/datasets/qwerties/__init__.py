import torch
from torch import Tensor
from torch.utils.data import TensorDataset
from pathlib import Path
from ... import LOCATION
import json


def generate(
    centroids: list,
    clusterSizes: list[int],
    blurs: list,
    root=LOCATION / Path("datasets") / Path("qwerties"),
):
    if len(centroids) != len(clusterSizes):
        raise Exception("lengths of centroids and clusterSizes are not the same")
    if len(blurs) != len(clusterSizes):
        raise Exception("lengths of blurs and clusterSizes are not the same")
    if len(centroids) != len(blurs):
        raise Exception("lengths of centroids and blurs are not the same")

    qwertiesFilePath = root / Path("qwerties.json")
    if Path.exists(qwertiesFilePath):
        with open(qwertiesFilePath, "r") as f:
            dataLabels = json.load(f)
            _centroids = dataLabels["centroids"]
            _clusterSizes = dataLabels["clusterSizes"]
            _blurs = dataLabels["blurs"]

            if (
                _centroids != centroids
                or _clusterSizes != clusterSizes
                or _blurs != blurs
            ):
                return _genFromScratch(
                    centroids,
                    clusterSizes,
                    blurs,
                    qwertiesFilePath,
                )
            data = Tensor(dataLabels["data"])
            labels = Tensor(dataLabels["labels"]).long()
        return TensorDataset(data, labels), len(centroids)

    return _genFromScratch(
        centroids,
        clusterSizes,
        blurs,
        qwertiesFilePath,
    )


def _genFromScratch(
    centroids: list,
    clusterSizes: list[int],
    blurs: list,
    qwertiesFilePath,
):
    data = torch.hstack(
        [
            torch.vstack(
                (
                    centroid[0] + torch.randn(clusterSize) * blur,
                    centroid[1] + torch.randn(clusterSize) * blur,
                )
            )
            for centroid, clusterSize, blur in zip(centroids, clusterSizes, blurs)
        ]
    ).T
    labels = torch.hstack(
        [
            cls * torch.ones(clusterSize).long()
            for cls, clusterSize in enumerate(clusterSizes)
        ]
    )

    with open(qwertiesFilePath, "w") as f:
        json.dump(
            {
                "data": data.tolist(),
                "labels": labels.tolist(),
                "centroids": centroids,
                "clusterSizes": clusterSizes,
                "blurs": blurs,
            },
            f,
        )
    return TensorDataset(data, labels), len(centroids)
