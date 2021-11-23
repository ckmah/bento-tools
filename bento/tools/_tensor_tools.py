import cell2cell as c2c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .._utils import track


@track
def to_tensor(
    data, layers, use_highly_variable_genes=False, scale=None, mask=False, copy=False
):
    """
    Generate tensor from data where dimensions are (layers, cells, genes).

    Parameters
    ----------
    layers : list of str
        keys in data.layers
    use_highly_variable_genes : bool
        whether to only use highly variably expressed genes as defined by scanpy, default False
        looks in data.var['highly_variable']
    scale : str
        valid values include z_score or unit, default None
    mask : bool
        whether to place nans with 0, default False
    copy : bool



    """
    adata = data.copy() if copy else data

    n_features = len(layers)
    cells = data.obs_names.tolist()
    n_cells = len(cells)
    tensor = []

    # TODO calculating highly variable genes not integrated, relies on scanpy

    for l in layers:
        if use_highly_variable_genes:
            n_genes = adata.var["highly_variable"].sum()
            genes = adata.var_names[adata.var["highly_variable"]].tolist()
            tensor.append(adata.to_df(l).loc[:, adata.var["highly_variable"]].values)
        else:
            n_genes = adata.n_vars
            genes = adata.var_names.tolist()
            tensor.append(adata.to_df(l).values)
    tensor = np.array(tensor)

    if scale == "z_score":
        tensor = (
            StandardScaler()
            .fit_transform(tensor.reshape(n_features, -1).T)
            .reshape(n_features, n_cells, n_genes)
        )
    elif scale == "unit":
        tensor = (
            MinMaxScaler()
            .fit_transform(tensor.reshape(n_features, -1).T)
            .reshape(n_features, n_cells, n_genes)
        )

    if mask:
        tensor_mask = ~np.isnan(tensor)
        tensor[~tensor_mask] = 0

    adata.uns["tensor"] = tensor
    adata.uns["tensor_labels"] = dict(layers=layers, cells=cells, genes=genes)

    return adata


def select_tensor_rank(data, upper_rank=10, runs=5, device="auto", random_state=888):
    '''
    Parameters
    ----------
    upper_rank : int
        Maximum rank to perform decomposition.
    runs : int
        Number of times to run decomposition for calculating the confidence interval.
    '''
    tensor_c2c, meta_tf = init_tensor(data, device=device)

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

    tensor_c2c, meta_tf = init_tensor(data, device=device)

    tensor_c2c.compute_tensor_factorization(
        rank=rank,
        init="random",
        random_state=random_state,
    )

    adata.uns["tensor_loadings"] = tensor_c2c.factors

    return adata


@track
def assign_factors(data, n_clusters=None, copy=False):
    adata = data.copy() if copy else data

    feature_load = adata.uns["tensor_loadings"]["Features"]
    cell_load = adata.uns["tensor_loadings"]["Cells"]
    gene_load = adata.uns["tensor_loadings"]["Genes"]

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
        cell_load.T, cmap="RdBu_r", center=0
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
        gene_load.T, cmap="RdBu_r", center=0
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


def init_tensor(data, device="auto"):

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tensor_c2c = c2c.tensor.PreBuiltTensor(
        data.uns["tensor"],
        order_names=data.uns["tensor_labels"].values(),
        order_labels=["Features", "Cells", "Genes"],
        device=device,
    )

    meta_tf = c2c.tensor.generate_tensor_metadata(
        interaction_tensor=tensor_c2c,
        metadata_dicts=[None, None, None],
        fill_with_order_elements=True,
    )

    return tensor_c2c, meta_tf
