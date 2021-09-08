import cell2cell as c2c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .._utils import track


@track
def to_tensor(data, layers, scale=None, mask=False, copy=False):
    adata = data.copy() if copy else data

    n_features = len(layers)
    tensor = []
    for l in layers:
        tensor.append(data.to_df(l).values)
    tensor = np.array(tensor)

    if scale == "z_score":
        tensor = (
            StandardScaler()
            .fit_transform(tensor.reshape(n_features, -1).T)
            .reshape(n_features, data.n_obs, data.n_vars)
        )
    elif scale == "unit":
        tensor = (
            MinMaxScaler()
            .fit_transform(tensor.reshape(n_features, -1).T)
            .reshape(n_features, data.n_obs, data.n_vars)
        )

    if mask:
        tensor_mask = ~np.isnan(tensor)
        tensor[~tensor_mask] = 0

    adata.uns["tensor"] = tensor
    adata.uns["tensor_labels"] = [
        layers,
        data.obs_names.tolist(),
        data.var_names.tolist(),
    ]

    return adata


def select_tensor_rank(data, upper_rank=10, runs=5, device="auto", random_state=888):
    tensor_c2c, meta_tf = init_tensor(data, device=device)

    fig, error = tensor_c2c.elbow_rank_selection(
        upper_rank=upper_rank,
        runs=runs,
        init="random",
        automatic_elbow=True,
        random_state=random_state,
    )

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

    # Standard score loadings
    cell_load = pd.DataFrame(
        zscore(cell_load, axis=1), index=cell_load.index, columns=cell_load.columns
    )
    gene_load = pd.DataFrame(
        zscore(gene_load, axis=1), index=gene_load.index, columns=gene_load.columns
    )

    # Cluster cells by loadings
    if n_clusters is None:
        n_clusters = feature_load.shape[1]

    cell_clusters = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="average"
    ).fit_predict(cell_load)

    # Assign cluster with largest average loading to factor
    cell_to_factor = cell_load.apply(
        lambda col: col.groupby(cell_clusters).agg(np.mean).idxmax(), axis=0
    )

    # Cell cluster to factor lookup
    cell_to_factor = dict(zip(cell_to_factor.values.flatten(), cell_to_factor.index))

    # Save associated tensor decomposition factor to adata.obs
    adata.obs.loc[cell_load.index, "td_cluster"] = [
        cell_to_factor[i] for i in cell_clusters
    ]

    # Repeat the same assignment procedure for genes...

    # Cluster genes by loadings
    gene_clusters = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="average"
    ).fit_predict(gene_load)

    # Assign cluster with largest average loading to factor
    gene_to_factor = gene_load.apply(
        lambda col: col.groupby(gene_clusters).agg(np.mean).idxmax(), axis=0
    )
    # Cell cluster to factor lookup
    gene_to_factor = dict(zip(gene_to_factor.values.flatten(), gene_to_factor.index))

    # Save associated tensor decomposition factor to adata.obs
    adata.var.loc[gene_load.index, "td_cluster"] = [
        gene_to_factor[i] for i in gene_clusters
    ]
    return adata


def init_tensor(data, device="auto"):

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tensor_c2c = c2c.tensor.PreBuiltTensor(
        data.uns["tensor"],
        order_names=data.uns["tensor_labels"],
        order_labels=["Features", "Cells", "Genes"],
        device=device,
    )

    meta_tf = c2c.tensor.generate_tensor_metadata(
        interaction_tensor=tensor_c2c,
        metadata_dicts=[None, None, None],
        fill_with_order_elements=True,
    )

    return tensor_c2c, meta_tf
