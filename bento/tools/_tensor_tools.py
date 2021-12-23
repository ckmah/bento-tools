import cell2cell as c2c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .._utils import track, PATTERN_NAMES, TENSOR_DIM_NAMES


@track
def to_tensor(data, layers, mask=False, copy=False):
    """
    Generate tensor from data where dimensions are (layers, cells, genes).

    Parameters
    ----------
    layers : list of str
        keys in data.layers
        whether to only use highly variably expressed genes as defined by scanpy, default False
    mask : bool
        whether to place nans with 0, default False
    copy : bool

    Attributes
    ----------
    AnnData, None
        Returns copy of AnnData if copy=True, otherwise modifies data in place and returns None

        `data.uns['tensor']` : np.ndarray
            3D numpy array of shape (len(layers), adata.n_obs, adata.n_vars)

        `data.uns['tensor_labels'] : dict
            Element labels across each dimension. Keys are dimension names (layers, cells, genes), values are lists of str

    """
    adata = data.copy() if copy else data

    cells = data.obs_names.tolist()
    genes = adata.var_names.tolist()

    # Build tensor from specified layers
    tensor = []
    for l in layers:
        tensor.append(adata.to_df(l).values)
    tensor = np.array(tensor)

    # Replace nans with 0 if mask == True
    if mask:
        tensor_mask = ~np.isnan(tensor)
        tensor[~tensor_mask] = 0

    # Save tensor values
    adata.uns["tensor"] = tensor

    # Save tensor dimension indexes
    adata.uns["tensor_labels"] = dict(zip(TENSOR_DIM_NAMES, [layers, cells, genes]))

    return adata


def select_tensor_rank(data, upper_rank=10, runs=5, device="auto", random_state=888):
    """
    Parameters
    ----------
    upper_rank : int
        Maximum rank to perform decomposition.
    runs : int
        Number of times to run decomposition for calculating the confidence interval.
    """
    to_tensor(data, layers=PATTERN_NAMES, mask=True)
    tensor_c2c = init_c2c_tensor(data, device=device)

    fig, error = tensor_c2c.elbow_rank_selection(
        upper_rank=upper_rank,
        runs=runs,
        init="random",
        automatic_elbow=True,
        random_state=random_state,
    )

    plt.tight_layout()
    return fig, error


@track
def decompose_tensor(data, rank, device="auto", random_state=888, copy=False):
    adata = data.copy() if copy else data

    to_tensor(data, layers=PATTERN_NAMES, mask=True)
    tensor_c2c = init_c2c_tensor(data, device=device)

    tensor_c2c.compute_tensor_factorization(
        rank=rank,
        init="random",
        random_state=random_state,
    )

    adata.uns["tensor_loadings"] = tensor_c2c.factors

    _assign_factors(data)

    return adata


def _assign_factors(data, n_clusters=None, copy=False):
    adata = data.copy() if copy else data

    # Get tensor dimension names
    dim_names = list(adata.uns["tensor_labels"].keys())
    cell_load = adata.uns["tensor_loadings"][dim_names[1]]
    gene_load = adata.uns["tensor_loadings"][dim_names[2]]

    # If 1 component decomposition, don't cluster later
    cluster_factors = True
    if cell_load.shape[1] == 1:
        cluster_factors = False

    # Zscale for clustering
    cell_load = pd.DataFrame(
        StandardScaler().fit_transform(cell_load),
        index=cell_load.index,
        columns=cell_load.columns,
    )
    gene_load = pd.DataFrame(
        StandardScaler().fit_transform(gene_load),
        index=gene_load.index,
        columns=gene_load.columns,
    )

    # Get sorted cell order from clustermap

    iorder = sns.clustermap(
        cell_load.T, row_cluster=cluster_factors, cmap="RdBu_r", center=0
    ).dendrogram_col.reordered_ind
    plt.close()

    # Reorder cell names
    iorder = pd.Series(
        range(len(cell_load)), index=cell_load.index[iorder], name="td_cluster"
    )
    cell_to_factor = cell_load.join(iorder)["td_cluster"].tolist()

    # Save associated tensor decomposition factor to adata.obs
    adata.obs["td_cluster"] = cell_to_factor

    # Get sorted cell order from clustermap
    iorder = sns.clustermap(
        gene_load.T, row_cluster=cluster_factors, cmap="RdBu_r", center=0
    ).dendrogram_col.reordered_ind
    plt.close()

    # Reorder cell names
    iorder = pd.Series(
        range(len(gene_load)), index=gene_load.index[iorder], name="td_cluster"
    )
    gene_to_factor = gene_load.join(iorder)["td_cluster"].tolist()

    # Save associated tensor decomposition factor to adata.var
    adata.var["td_cluster"] = gene_to_factor
    return adata


def init_c2c_tensor(data, device="auto"):

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = None

    print(f"Device: {device}")

    order_labels = list(data.uns["tensor_labels"].keys())
    order_names = list(data.uns["tensor_labels"].values())

    tensor_c2c = c2c.tensor.PreBuiltTensor(
        data.uns["tensor"],
        order_names=order_names,
        order_labels=order_labels,
        device=device,
    )

    return tensor_c2c
