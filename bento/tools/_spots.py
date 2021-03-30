import bento
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint


def detect_spots(data, imagedir, device="auto", copy=False):
    """
    Detect and label localization patterns.
    TODO change data to be iterable compatible with skorch.predict_proba

    Parameters
    ----------
    data : [type]
        [description]
    imagedir : str
        Folder for bento rasterized images.
    Returns
    -------
    [type]
        [description]
    """
    adata = data.copy() if copy else data

    # Default to gpu if possible. Otherwise respect specified parameter
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = "/".join(bento.__file__.split("/")[:-1]) + "/models/spots"

    model_params = pickle.load(open(f"{model_dir}/params.p", "rb"))

    # Load ensemble model
    # Reload model with temperature and softmax
    net = NeuralNetClassifier(module=SpotsModule(**model_params), device=device)

    # Load model
    net.initialize()
    net.load_params(checkpoint=Checkpoint(dirname=f"{model_dir}"))
    # TODO install cudatoolkit. pip?

    dataset = datasets.ImageFolder(
        imagedir,
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]),
    )

    pred_prob = net.predict_proba(dataset)

    pred = pred_prob >= 0.5
    pred = pred.astype(int)

    # Class names
    classes = ["cell2D", "cellext", "foci", "nuc2D", "polarized", "random"]
    classes = [f"spots_{c}" for c in classes]

    spots_pred_long = [
        str(path).split("/")[-1].split(".")[0].split("_") for path, _ in dataset.imgs
    ]
    spots_pred_long = pd.DataFrame(spots_pred_long, columns=["cell", "gene"])
    spots_pred_long[classes] = pred_prob
    # spots_pred_long[[f'{c}_prob' for c in classes]] = pred

    for c in classes:
        df = (
            spots_pred_long[["cell", "gene", c]]
            .pivot(index="cell", columns="gene", values=c)
            .fillna(-1)
            .reindex(columns=adata.var_names, fill_value=-1)
            .astype(np.float64)
        )

        adata.layers[f"{c}_prob"] = df

        # Convert prob to label
        df[df >= 0.5] = 1
        df[(df < 0.5) & (df >= 0)] = 0
        df = df.astype(np.int8)
        adata.layers[c] = df

    return adata if copy else None


def get_conv_dim(in_size, padding, dilation, kernel_size, stride):
    outsize = 1 + (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
    return int(outsize)


class SpotsModule(nn.Module):
    def __init__(
        self,
        n_conv_layers,
        in_dim,
        out_channels,
        kernel_size,
        f_units_l0,
        f_units_l1,
    ) -> None:
        super().__init__()
        conv_layers = []

        in_channels = 1
        in_dim = in_dim

        # Stack (convolutions + batchnorm + activation) + maxpool
        for i in range(n_conv_layers):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
            )
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())

            # Compute convolved output dimensions
            in_dim = get_conv_dim(
                in_dim, padding=0, dilation=1, kernel_size=kernel_size, stride=1
            )

            in_channels = out_channels
            out_channels *= 2

        out_channels = int(out_channels / 2)

        conv_layers.append(nn.MaxPool2d(2, 2))
        in_dim = int(in_dim / 2)

        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        fc_layers = [nn.Flatten()]

        # Compute flatten size
        in_features = out_channels * in_dim * in_dim
        for i in [f_units_l0, f_units_l1]:
            out_features = i
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.BatchNorm1d(out_features))
            fc_layers.append(nn.ReLU())

            in_features = out_features

        fc_layers.append(nn.Linear(in_features, 6))

        self.model = nn.Sequential(*[*conv_layers, *fc_layers])

    def forward(self, x):
        x = self.model(x)

        x = F.softmax(x, dim=-1)

        return x