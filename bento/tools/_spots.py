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

from sklearn.preprocessing import OneHotEncoder


def detect_spots(data, imagedir, device="auto", model="pattern", copy=False):
    """
    Detect and label localization patterns.
    TODO change data to be iterable compatible with skorch.predict_proba

    Parameters
    ----------
    data : [type]
        [description]
    imagedir : str
        Folder for rasterized images.
    Returns
    -------
    [type]
        [description]
    """
    adata = data.copy() if copy else data

    # Default to gpu if possible. Otherwise respect specified parameter
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = "/".join(bento.__file__.split("/")[:-1]) + "/models/spots/" + model

    model_params = pickle.load(open(f"{model_dir}/params.p", "rb"))

    # Load model
    modules = dict(pattern=SpotsModule, five_pattern=FiveSpotsModule)
    module = modules[model](**model_params)

    net = NeuralNetClassifier(module=module, device=device)
    net.initialize()
    net.load_params(checkpoint=Checkpoint(dirname=f"{model_dir}"))

    dataset = datasets.ImageFolder(
        imagedir,
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]),
    )

    # Sample by n_classes
    pred_prob = net.predict_proba(dataset)

    if isinstance(module, FiveSpotsModule):
        label_names = [
            "cell edge",
            "nuclear edge",
            "perinuclear",
            "protrusions",
            "random",
        ]
    else:
        label_names = [
            "cell edge",
            "foci",
            "nuclear edge",
            "perinuclear",
            "protrusions",
            "random",
        ]

    encoder = OneHotEncoder(handle_unknown="ignore").fit(
        np.array(label_names).reshape(-1, 1)
    )

    # Cell gene names
    sample_names = [
        str(path).split("/")[-1].split(".")[0].split("_") for path, _ in dataset.imgs
    ]
    spots_pred_long = pd.DataFrame(sample_names, columns=["cell", "gene"])
    spots_pred_long["label"] = encoder.inverse_transform(pred_prob >= 0.5)

    pattern_labels = (
        spots_pred_long.pivot(index="cell", columns="gene", values="label")
        .fillna("none")
        .reindex(index=adata.obs_names, columns=adata.var_names, fill_value="none")
    )

    adata.layers[model] = pattern_labels

    # Annotate points with pattern labels
    plabels_long = pattern_labels.reset_index().melt(id_vars="cell")
    plabels_long = plabels_long.rename({"value": model}, axis=1)

    # Overwrite existing values
    if model in adata.uns["points"].columns:
        adata.uns["points"].drop([model], axis=1, inplace=True)

    # Annotate points
    adata.uns["points"] = adata.uns["points"].merge(
        plabels_long, how="left", on=["cell", "gene"]
    )

    # Save pattern values as categorical to save memory
    adata.uns["points"][model] = adata.uns["points"][model].astype("category")

    # Save to adata.var
    distr_to_var(adata, model)

    return adata if copy else None


def distr_to_var(data, layer, copy=False):
    adata = data.copy() if copy else data

    summary = (adata.to_df(layer).apply(lambda g: g.value_counts()) / adata.shape[0]).T
    adata.var[summary.columns] = summary

    return adata if copy else None

def get_conv_dim(in_size, padding, dilation, kernel_size, stride):
    outsize = 1 + (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
    return int(outsize)


class DataFlatten:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.reshape(X.shape[0], -1)
        return X


class DataReshape:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.reshape(X.shape[0], 1, 64, 64)
        return X


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


class FiveSpotsModule(nn.Module):
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

        fc_layers.append(nn.Linear(in_features, 5))

        self.model = nn.Sequential(*[*conv_layers, *fc_layers])

    def forward(self, x):
        x = self.model(x)

        x = F.softmax(x, dim=-1)

        return x