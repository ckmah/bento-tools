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
    data : AnnData
    layers : list of str
        Keys in data.layers to build tensor.
    mask : bool
        Whether to replace nans with 0, default False.
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `uns['tensor']` : np.ndarray
            3D numpy array of shape (len(layers), adata.n_obs, adata.n_vars)

        `uns['tensor_labels'] : dict
            Element labels across each dimension. Keys are dimension names (layers, cells, genes), 
            values are lists of str

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


def select_tensor_rank(data, layers, upper_rank=5, runs=3, device="auto", random_state=888, copy=False):
    """Perform `bento.tl.decompose_tensor()` up to rank `upper_rank` repeating each decomposition 
    `runs` times to compute a 95% confidence interval, plotting reconstruction error for each rank.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData.
    layers : list of str
        Keys in data.layers to build tensor.
    upper_rank : int
        Maximum rank to perform decomposition, by default 10.
    runs : int
        Number of times to run decomposition for calculating the confidence interval, by default 5.
    device : str, optional
        Type of device to use, valid options include `cpu`, `gpu`, and `auto`, by default `auto`.
        Option `auto` prefers `gpu` over `cpu` if available.
    random_state : int, optional
        Random seed used for reproducibility, by default 888
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    adata : anndata.AnnData
        Returns `adata` if `copy=True`, otherwise adds fields to `data`:

        `uns['tensor']` : np.ndarray
            3D numpy array of shape (len(layers), adata.n_obs, adata.n_vars)

        `uns['tensor_labels'] : dict
            Element labels across each dimension. Keys are dimension names (layers, cells, genes), 
            values are lists of str
    """
    adata = data.copy() if copy else data

    to_tensor(data, layers=layers, mask=True)
    tensor_c2c = init_c2c_tensor(data, device=device)

    fig, error = tensor_c2c.elbow_rank_selection(
        upper_rank=upper_rank,
        runs=runs,
        init="random",
        automatic_elbow=True,
        random_state=random_state,
    );

    plt.tight_layout()
    

def lp_signatures(data, rank, device="auto", random_state=888, copy=False):
    """Calculate localization signatures by performing tensor decomposition on the dataset tensor. 
        Wrapper for `bento.tl.decompose_tensor()`.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData.
    rank : int
        Rank to perform decomposition.
    device : str, optional
        Type of device to use, valid options include `cpu`, `gpu`, and `auto`, by default `auto`.
        Option `auto` prefers `gpu` over `cpu` if available.
    random_state : int, optional
        Random seed used for reproducibility, by default 888.
    copy : bool
        Return a copy of `data` instead of writing to data, by default False.

    Returns
    -------
    _type_
        _description_
    """
    return decompose_tensor(data, PATTERN_NAMES, rank, device=device, random_state=random_state, copy=copy)


@track
def decompose_tensor(data, layers, rank, device="auto", random_state=888, copy=False):
    """Perform tensor decomposition on the 3-dimensional tensor built from `[cell, gene, layers]`.

    Parameters
    ----------
    data : _type_
        _description_
    layers : _type_
        _description_
    rank : _type_
        _description_
    device : str, optional
        _description_, by default "auto"
    random_state : int, optional
        _description_, by default 888
    copy : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    adata = data.copy() if copy else data

    to_tensor(data, layers=layers, mask=True)
    tensor_c2c = init_c2c_tensor(data, device=device)

    tensor_c2c.compute_tensor_factorization(
        rank=rank,
        init="random",
        random_state=random_state,
    )

    tensor_loadings = {}
    for layer,df in tensor_c2c.factors.items():
        df.columns = df.columns.str.replace('Factor', 'Signature')
        tensor_loadings[layer] = df

    adata.uns["tensor_loadings"] = tensor_loadings
    
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
