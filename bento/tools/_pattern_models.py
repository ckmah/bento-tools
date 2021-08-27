import pickle

OneHotEncoder = None
LabelBinarizer = None
skorch = None
import bento
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

from ._pattern_stats import pattern_stats

from ..utils import track

PATTERN_NAMES = [
    "cell_edge",
    "foci",
    "nuclear_edge",
    "perinuclear",
    "protrusions",
    "random",
]


@track
def predict_patterns(
    data, imagedir, batch_size=1024, model="multiclass", device="auto", copy=False
):
    """Predict transcript localization patterns with binary classifiers.
    Patterns include:
    1. cell edge
    2. foci
    3. nuclear edge
    4. perinuclear
    5. protrusions
    6. random

    Parameters
    ----------
    data : spatial formatted AnnData

    imagedir : str
        Path to rasterized sample images.
    batch_size : int, optional
        Number of samples to evaluate at once, by default 1024.
    device : str, optional
        "cuda" for GPU, "cpu" for processor, by default "auto"
    copy : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """

    global skorch, LabelBinarizer, OneHotEncoder

    if skorch is None:
        import skorch

    if LabelBinarizer is None:
        from sklearn.preprocessing import LabelBinarizer

    if OneHotEncoder is None:
        from sklearn.preprocessing import OneHotEncoder

    adata = data.copy() if copy else data

    dataset = datasets.ImageFolder(
        imagedir,
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]),
    )

    # Default to gpu if possible. Otherwise respect specified parameter
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model == "multiclass":
        model_dir = "/".join(bento.__file__.split("/")[:-1]) + "/models/multiclass"
        model_params = pickle.load(open(f"{model_dir}/params.p", "rb"))

        # Load model
        module = SpotsModule(**model_params)
        net = skorch.NeuralNetClassifier(
            module=module, batch_size=batch_size, device=device
        )
        net.initialize()
        net.load_params(checkpoint=skorch.callbacks.Checkpoint(dirname=f"{model_dir}"))

        # 2d array, sample by n_classes
        pred_prob = net.predict_proba(dataset)

        encoder = OneHotEncoder(handle_unknown="ignore").fit([PATTERN_NAMES])
        classes = list(np.array(encoder.categories_).flatten())

    elif model == "multilabel":
        model_dir = "/".join(bento.__file__.split("/")[:-1]) + "/models/multilabel"
        model_params = pickle.load(open(f"{model_dir}/params.p", "rb"))

        nets = []
        pred_prob = []
        for c in PATTERN_NAMES:
            module = SpotsBinaryModule(**model_params)
            net = skorch.NeuralNetClassifier(module=module, device=device)
            net.initialize()
            net.load_params(
                checkpoint=skorch.callbacks.Checkpoint(
                    dirname=f"{model_dir}", fn_prefix=f"{c}_"
                )
            )
            nets.append(net)

            # 2d array, sample by 2
            pred_prob.append(net.predict_proba(dataset))

        # Reshape to sample by n_classes
        pred_prob = np.array(pred_prob)[:, :, 1].T

        encoder = LabelBinarizer().fit(PATTERN_NAMES)
        classes = encoder.classes_.tolist()

    else:
        raise ValueError("No such model.")

    # Cell gene names
    sample_names = [
        str(path).split("/")[-1].split(".")[0].rsplit("_", 1)
        for path, _ in dataset.imgs
    ]
    spots_pred_long = pd.DataFrame(sample_names, columns=["cell", "gene"])

    # Pair probabilities to samples
    spots_pred_long[classes] = pred_prob > 0.5

    # Flatten p(patterns) for all genes in each cell
    loc_embed = (
        spots_pred_long.pivot(index="cell", columns="gene", values=classes)
        .fillna(0)
        .replace({True: 1, False: 0})
        .reindex(adata.obs_names)
        .values
    )

    # Save flattened probabilities
    adata.obsm[f"pattern_embed"] = loc_embed

    # Save each class to adata
    for c in classes:
        indicator_df = (
            spots_pred_long.pivot(index="cell", columns="gene", values=c)
            .replace({True: 1, False: 0})
            .reindex(index=adata.obs_names, columns=adata.var_names, fill_value=np.nan).astype(float)
        )

        indicator_df.columns.name = "gene"

        # Save to adata.layers
        adata.layers[c] = indicator_df

        # Construct [cell, gene, value]
        labels_long = indicator_df.reset_index().melt(id_vars="cell")
        labels_long = labels_long.rename({"value": c}, axis=1)

        # Overwrite existing values
        if c in adata.uns["points"].columns:
            adata.uns["points"].drop(c, axis=1, inplace=True)

        # Save pattern label to adata.uns.points
        adata.uns["points"] = adata.uns["points"].merge(
            labels_long, how="left", on=["cell", "gene"]
        )

        # Save pattern values as categorical to save memory
        adata.uns["points"][c] = adata.uns["points"][c].astype("category")

    # Save to adata.var
    pattern_stats(adata, c)

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


class SpotsModule(torch.nn.Module):
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
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
            )
            conv_layers.append(torch.nn.BatchNorm2d(out_channels))
            conv_layers.append(torch.nn.ReLU())

            # Compute convolved output dimensions
            in_dim = get_conv_dim(
                in_dim, padding=0, dilation=1, kernel_size=kernel_size, stride=1
            )

            in_channels = out_channels
            out_channels *= 2

        out_channels = int(out_channels / 2)

        conv_layers.append(torch.nn.MaxPool2d(2, 2))
        in_dim = int(in_dim / 2)

        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        fc_layers = [torch.nn.Flatten()]

        # Compute flatten size
        in_features = out_channels * in_dim * in_dim
        for i in [f_units_l0, f_units_l1]:
            out_features = i
            fc_layers.append(torch.nn.Linear(in_features, out_features))
            fc_layers.append(torch.nn.BatchNorm1d(out_features))
            fc_layers.append(torch.nn.ReLU())

            in_features = out_features

        fc_layers.append(torch.nn.Linear(in_features, 6))

        self.model = torch.nn.Sequential(*[*conv_layers, *fc_layers])

    def forward(self, x):
        x = self.model(x)

        x = torch.nn.functional.softmax(x, dim=-1)

        return x


class SpotsBinaryModule(torch.nn.Module):
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
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
            )
            conv_layers.append(torch.nn.BatchNorm2d(out_channels))
            conv_layers.append(torch.nn.ReLU())

            # Compute convolved output dimensions
            in_dim = get_conv_dim(
                in_dim, padding=0, dilation=1, kernel_size=kernel_size, stride=1
            )

            in_channels = out_channels
            out_channels *= 2

        out_channels = int(out_channels / 2)

        conv_layers.append(torch.nn.MaxPool2d(2, 2))
        in_dim = int(in_dim / 2)

        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        fc_layers = [torch.nn.Flatten()]

        # Compute flatten size
        in_features = out_channels * in_dim * in_dim
        for i in [f_units_l0, f_units_l1]:
            out_features = i
            fc_layers.append(torch.nn.Linear(in_features, out_features))
            fc_layers.append(torch.nn.BatchNorm1d(out_features))
            fc_layers.append(torch.nn.ReLU())

            in_features = out_features

        fc_layers.append(torch.nn.Linear(in_features, 2))

        self.model = torch.nn.Sequential(*[*conv_layers, *fc_layers])

    def forward(self, x):
        x = self.model(x)

        x = torch.nn.functional.softmax(x, dim=-1)

        return x


class FiveSpotsModule(torch.nn.Module):
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
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                )
            )
            conv_layers.append(torch.nn.BatchNorm2d(out_channels))
            conv_layers.append(torch.nn.ReLU())

            # Compute convolved output dimensions
            in_dim = get_conv_dim(
                in_dim, padding=0, dilation=1, kernel_size=kernel_size, stride=1
            )

            in_channels = out_channels
            out_channels *= 2

        out_channels = int(out_channels / 2)

        conv_layers.append(torch.nn.MaxPool2d(2, 2))
        in_dim = int(in_dim / 2)

        # We optimize the number of layers, hidden units and dropout ratio in each layer.
        fc_layers = [torch.nn.Flatten()]

        # Compute flatten size
        in_features = out_channels * in_dim * in_dim
        for i in [f_units_l0, f_units_l1]:
            out_features = i
            fc_layers.append(torch.nn.Linear(in_features, out_features))
            fc_layers.append(torch.nn.BatchNorm1d(out_features))
            fc_layers.append(torch.nn.ReLU())

            in_features = out_features

        fc_layers.append(torch.nn.Linear(in_features, 5))

        self.model = torch.nn.Sequential(*[*conv_layers, *fc_layers])

    def forward(self, x):
        x = self.model(x)

        x = torch.nn.functional.softmax(x, dim=-1)

        return x
