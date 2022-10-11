import numpy as np
import pandas as pd
from dask import dataframe as dd
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback

from ..preprocessing import get_points
from ._embeddings import _count_neighbors


def coloc_quotient(data, n_neighbors=20, min_count=20, chunksize=64, copy=False):
    """Calculate pairwise gene colocalization quotient in each cell.

    Parameters
    ----------
    adata : AnnData
        Anndata formatted spatial data.
    n_neighbors : int
        Number of nearest neighbors to consider, default 25
    min_count : int
        Minimum number of points for a given gene in a cell to be considered, default 20
    chunksize : int
        Number of cells per processing chunk. Default 64.
    Returns
    -------
    adata : AnnData
        .uns['clq']: Pairwise gene colocalization similarity within each cell formatted as a long dataframe.
    """

    adata = data.copy() if copy else data

    points = (
        get_points(adata, asgeo=False)
        .sort_values("cell")[["cell", "gene", "x", "y"]]
        .reset_index(drop=True)
    )

    meta = _cell_clq(
        points.loc[points["cell"] == points["cell"].values[0]], adata.n_vars, n_neighbors, min_count=1
    )
    
    if chunksize:
        chunksize = min(chunksize, points['cell'].nunique())

    ddf = dd.from_pandas(points, npartitions=1)

    # Partition so {chunksize} cells per partition
    _, group_loc = np.unique(
        points["cell"].astype(str),
        return_index=True,
    )

    if chunksize < len(group_loc):
        divisions = [group_loc[loc] for loc in range(0, len(group_loc), chunksize)]
        divisions.append(len(points) - 1)
        ddf = ddf.repartition(divisions=divisions)

    with TqdmCallback(desc="", tqdm_class=tqdm):
        cell_metrics = (
            ddf.groupby("cell")
            .apply(
                lambda df: _cell_clq(df, adata.n_vars, n_neighbors, min_count),
                meta=meta,
            )
            .compute()
            .reset_index()
            .drop("level_1", axis=1)
        )

    cell_metrics[["cell", "gene", "neighbor"]] = (
        cell_metrics[["cell", "gene", "neighbor"]].astype(str).astype("category")
    )

    # Save to uns['clq'] as adjacency list
    adata.uns["clq"] = cell_metrics

    return adata if copy else None


def _cell_clq(cell_points, n_genes, n_neighbors, min_count):

    # Count number of points for each gene
    counts = cell_points["gene"].value_counts()

    # Only keep genes >= min_count
    counts = counts[counts >= min_count]
    valid_genes = counts.sort_index().index.tolist()
    counts = counts[valid_genes]

    if len(valid_genes) < 2:
        return pd.DataFrame()

    # Get points
    valid_points = cell_points[cell_points["gene"].isin(valid_genes)]

    # Cleanup gene categories
    valid_points["gene"] = valid_points["gene"].cat.remove_unused_categories()

    neighbor_counts = _count_neighbors(valid_points, n_genes, n_neighbors, agg=True)

    clq_df = _clq_statistic(neighbor_counts, counts)

    return clq_df



def _clq_statistic(neighbor_counts, counts):
    """
    Compute the colocation quotient for each gene pair.

    Parameters
    ----------
    neighbor_counts : pd.DataFrame
        Dataframe with columns "gene", "neighbor", and "count".
    counts : pd.Series
        Series of raw gene counts.
    """
    clq_df = neighbor_counts.copy()
    clq_df["clq"] = (clq_df["count"] / counts.loc[clq_df["gene"]].values) / (
        counts.loc[clq_df["neighbor"]].values / counts.sum()
    )
    return clq_df.drop("count", axis=1)


def global_clq(neighbor_counts, counts):
    gclq_df = neighbor_counts.copy()
    gclq_df = gclq_df[gclq_df["gene"] == gclq_df["neighbor"]]
    global_counts = counts.loc[gclq_df["gene"]].values
    total_count = global_counts.sum()
    gclq_df["gclq"] = (
        gclq_df["gclq"].sum()
        / (global_counts * ((global_counts - 1) / (total_count - 1))).sum()
    )

    return global_clq


def local_clq(adata, gene_a, gene_b):
    """
    Compute local colocation quotients for every point between two genes across all cells. Note that this is not a symmetric function.
    Parameters
    ----------
    gene_a : str
        Gene name
    gene_b : str
        Gene name
    Returns
    -------
    clq : float
        Local colocation quotient for each point in gene_b
    """
    # Get points for cell
    points = get_points(adata, asgeo=False)

    # Get counts for each gene
    counts = points["gene"].value_counts()

    return
    # nai->b / nb (N - 1)
