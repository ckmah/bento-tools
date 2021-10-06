import pickle
import warnings

OneHotEncoder = None
LabelBinarizer = None
skorch = None
import bento
import numpy as np
import pandas as pd
import statsmodels.formula.api as sfm
import torch
from joblib import Parallel, delayed
from patsy import PatsyError
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError
from torchvision import datasets, transforms
from tqdm.auto import tqdm

warnings.simplefilter("ignore", ConvergenceWarning)


PATTERN_NAMES = [
    "cell_edge",
    "foci",
    "nuclear_edge",
    "perinuclear",
    "protrusions",
    "random",
]


# def predict_patterns(data, imagedir, batch_size=1024, device="auto", copy=False):
#     """Predict transcript localization patterns with multiclass classifier.
#     Patterns include:
#     1. cell edge
#     2. foci
#     3. nuclear edge
#     4. perinuclear
#     5. protrusions
#     6. random

#     Parameters
#     ----------
#     data : spatial formatted AnnData

#     imagedir : str
#         Path to rasterized sample images.
#     batch_size : int, optional
#         Number of samples to evaluate at once, by default 1024.
#     device : str, optional
#         "cuda" for GPU, "cpu" for processor, by default "auto"
#     copy : bool, optional
#         [description], by default False

#     Returns
#     -------
#     [type]
#         [description]
#     """

#     global skorch, OneHotEncoder

#     if skorch is None:
#         import skorch

#     if OneHotEncoder is None:
#         from sklearn.preprocessing import OneHotEncoder

#     adata = data.copy() if copy else data

#     dataset = datasets.ImageFolder(
#         imagedir,
#         transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]),
#     )

#     # Default to gpu if possible. Otherwise respect specified parameter
#     if device == "auto":
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     model_dir = "/".join(bento.__file__.split("/")[:-1]) + "/models/multiclass"
#     model_params = pickle.load(open(f"{model_dir}/params.p", "rb"))

#     # Load model
#     module = SpotsModule(**model_params)
#     net = skorch.NeuralNetClassifier(
#         module=module, batch_size=batch_size, device=device
#     )
#     net.initialize()
#     net.load_params(checkpoint=skorch.callbacks.Checkpoint(dirname=f"{model_dir}"))

#     # 2d array, sample by n_classes
#     pred_prob = net.predict_proba(dataset)

#     label_names = PATTERN_NAMES
#     encoder = OneHotEncoder(handle_unknown="ignore").fit(
#         np.array(label_names).reshape(-1, 1)
#     )

#     # Cell gene names
#     sample_names = [
#         str(path).split("/")[-1].split(".")[0].rsplit("_", 1)
#         for path, _ in dataset.imgs
#     ]
#     spots_pred_long = pd.DataFrame(sample_names, columns=["cell", "gene"])

#     # Pair probabilities to samples
#     spots_pred_long[label_names] = pred_prob

#     # Flatten p(patterns) for all genes in each cell
#     loc_embed = (
#         spots_pred_long.pivot(index="cell", columns="gene", values=label_names)
#         .fillna(0)
#         .reindex(adata.obs_names)
#         .values
#     )

#     # Save flattened probabilities
#     adata.obsm[f"multiclass_embed"] = loc_embed

#     # TODO use spares matrices to avoid slow/big df pivots
#     # https://stackoverflow.com/questions/55404617/faster-alternatives-to-pandas-pivot-table
#     # Build "pattern" genexcell layer, where values are pattern labels
#     spots_pred_long["label"] = encoder.inverse_transform(pred_prob >= 0.5)

#     pattern_labels = (
#         spots_pred_long.pivot(index="cell", columns="gene", values="label")
#         .fillna("none")
#         .reindex(index=adata.obs_names, columns=adata.var_names, fill_value="none")
#     )

#     pattern_labels.columns.name = "gene"

#     adata.layers[model] = pattern_labels

#     # Annotate points with pattern labels
#     plabels_long = pattern_labels.reset_index().melt(id_vars="cell")
#     plabels_long = plabels_long.rename({"value": "multiclass"}, axis=1)

#     # Overwrite existing values
#     if "multiclass" in adata.uns["points"].columns:
#         adata.uns["points"].drop(["multiclass"], axis=1, inplace=True)

#     # Annotate points
#     adata.uns["points"] = adata.uns["points"].merge(
#         plabels_long, how="left", on=["cell", "gene"]
#     )

#     # Save pattern values as categorical to save memory
#     adata.uns["points"]["multiclass"] = adata.uns["points"]["multiclass"].astype(
#         "category"
#     )

#     # Save to adata.var
#     distr_to_var(adata, "multiclass")

#     return adata if copy else None


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
        .reindex(adata.obs_names)
        .values
    )

    # Save flattened probabilities
    adata.obsm[f"pattern_embed"] = loc_embed

    # Save each class to adata
    for c in classes:
        indicator_df = (
            spots_pred_long.pivot(index="cell", columns="gene", values=c)
            .fillna("none")
            .reindex(index=adata.obs_names, columns=adata.var_names, fill_value="none")
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
        distr_to_var(adata, c)

    return adata if copy else None


def distr_to_var(data, layer, copy=False):
    """Computes frequencies of input layer values across cells and across genes.
    Assumes layer values are categorical.

    Parameters
    ----------
    data : [type]
        [description]
    layer : [type]
        [description]
    copy : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    adata = data.copy() if copy else data

    # Save frequencies across genes to adata.var
    gene_summary = adata.to_df(layer).replace({'none': 0}).sum(axis=0)
    adata.var[layer] = gene_summary

    # Save frequencies across cells to adata.obs
    cell_summary = adata.to_df(layer).replace({'none': 0}).sum(axis=1)
    adata.obs[layer] = cell_summary

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


def spots_diff(
    data, phenotype=None, continuous=False, combined=False, n_cores=1, copy=False
):
    """Gene-wise test for differential localization across phenotype of interest.

    Parameters
    ----------
    data : AnnData
        Anndata formatted spatial data.
    phenotype : str
        Variable grouping cells for differential analysis. Must be in data.obs_names.
    continuous : bool
        Whether the phenotype is continuous or categorical. By default False.
    n_cores : int, optional
        cores used for multiprocessing, by default 1
    copy : bool, optional
        Return view of AnnData if False, return copy if True. By default False.
    """
    adata = data.copy() if copy else data

    # Parallelize on chunks
    patterns = adata.layers["pattern"].T
    phenotype_vector = adata.obs[phenotype].tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diff_output = Parallel(n_jobs=n_cores)(
            delayed(_spots_diff_gene)(
                gene, gp, phenotype, phenotype_vector, continuous, combined
            )
            for gene, gp in tqdm(zip(adata.var_names, patterns), total=len(patterns))
        )

    # Format pattern column
    diff_output = pd.concat(diff_output)

    # FDR correction
    results_adj = []
    for _, df in diff_output.groupby("pattern"):
        df["padj"] = multipletests(df["pvalue"], method="hs")[1]
        results_adj.append(df)

    results_adj = pd.concat(results_adj).dropna()

    # -log10pvalue, padj
    results_adj["-log10p"] = -np.log10(results_adj["pvalue"].astype(np.float32))
    results_adj["-log10padj"] = -np.log10(results_adj["padj"].astype(np.float32))

    # Cap significance values
    results_adj.loc[results_adj["-log10p"] > 20, "-log10p"] = 20
    results_adj.loc[results_adj["-log10padj"] > 12, "-log10padj"] = 12

    # Sort results
    results_adj = results_adj.sort_values("pvalue")

    # Save back to AnnData
    adata.uns[f"diff_{phenotype}"] = results_adj

    return adata if copy else None


def _spots_diff_gene(gene, patterns, phenotype, phenotype_vector, combined):
    """Perform pairwise comparison between groupby and every class.

    Parameters
    ----------
    chunk : tuple

    Returns
    -------
    DataFrame
        Differential localization test results. [# of patterns, ]
    """
    results = []
    # Series denoting pattern frequencies
    pattern_dummies = pd.get_dummies(patterns)
    pattern_dummies = pattern_dummies.drop("none", axis=1)
    pattern_names = pattern_dummies.columns.tolist()

    # One hot encode categories
    group_dummies = pd.get_dummies(pd.Series(phenotype_vector))
    group_dummies.columns = [f"{phenotype}_{g}" for g in group_dummies.columns]
    group_names = group_dummies.columns.tolist()
    group_data = pd.concat([pattern_dummies, group_dummies], axis=1)
    group_data.columns = group_data.columns.astype(str)

    # Perform one group vs rest logistic regression
    for g in group_names:
        try:
            res = sfm.logit(
                formula=f"{g} ~ {' + '.join(pattern_names)}", data=group_data
            ).fit(disp=0)

            # Look at marginal effect of each pattern coefficient
            r = res.get_margeff(dummy=True).summary_frame()
            r["gene"] = gene
            r["phenotype"] = g
            # r["pattern"] = p

            r.columns = [
                "dy/dx",
                "std_err",
                "z",
                "pvalue",
                "ci_low",
                "ci_high",
                "gene",
                "phenotype",
                # "pattern",
            ]
            # r.reset_index(drop=True, inplace=True)
            r = r.reset_index().rename({"index": "pattern"}, axis=1)

            results.append(r)
        except (np.linalg.LinAlgError, PerfectSeparationError, PatsyError):
            continue

    return pd.concat(results) if len(results) > 0 else None