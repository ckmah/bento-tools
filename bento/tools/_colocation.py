import numpy as np
import pandas as pd
from numpy.random import default_rng
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from ..preprocessing import get_points
from sklearn.neighbors import NearestNeighbors

# Step 1: count_neighbors()
# Step 2: get_clq()


def count_neighbors(points_df, n_neighbors, agg=True):
    """Build nearest neighbor index for points.

    Parameters
    ----------
    points_df : pd.DataFrame
        Points dataframe. Must have columns "x", "y", and "gene".
    n_neighbors : int
        Number of nearest neighbors to consider.
    agg : bool
        Whether to aggregate nearest neighbors at the gene-level. Default True.
    Returns
    -------
    list or dict of dicts
        If agg='point', returns a list of dicts, one for each point. Dict keys are gene names, values are counts.
        If agg='gene', returns a DataFrame with columns "gene", "neighbor", and "count".
    """
    # Build knn index
    neighbor_index = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(points_df[["x", "y"]].values)
        .kneighbors(points_df[["x", "y"]].values, return_distance=False)
    )
    point_labels = points_df["gene"].values

    # Get gene-level neighbor counts for each gene
    if agg is True:
        source_genes, source_indices = np.unique(point_labels, return_index=True)

        gene_index = []

        for g, gi in zip(source_genes, source_indices):
            # First get all points for this gene
            g_neighbors = np.unique(neighbor_index[gi].flatten())
              # get unique neighbor points
            g_neighbors = point_labels[g_neighbors]  # Get point gene names
            neighbor_names, neighbor_counts = np.unique(
                g_neighbors, return_counts=True
            )  # aggregate neighbor gene counts

            for neighbor, count in zip(neighbor_names, neighbor_counts):
                gene_index.append([g, neighbor, count])

        gene_index = pd.DataFrame(gene_index, columns=["gene", "neighbor", "count"])
        return gene_index
    # Get gene-level neighbor counts for each point
    else:
        neighborhood_shape = neighbor_index.shape
        gene_labels = point_labels[neighbor_index.flatten()].reshape(neighborhood_shape)
        point_index = []
        for row in gene_labels[:, 1:]:
            point_index.append(dict(zip(np.unique(row, return_counts=True))))
        return point_index


def coloc_quotient(data, n_neighbors=20, min_count=20, chunksize=64, copy=False):
    """Calculate pairwise gene colocalization quotient in each cell.

    Parameters
    ----------
    adata : AnnData
        Anndata formatted spatial data.
    n_neighbors : int
        Number of nearest neighbors to consider, default 25
    min_count : int
        Minimum points needed to be eligible for analysis. default 20
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
        points.loc[points["cell"] == adata.obs_names[0]], n_neighbors, min_count
    )

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

    with ProgressBar():
        cell_metrics = (
            ddf.groupby("cell")
            .apply(
                lambda df: _cell_clq(df, n_neighbors, min_count),
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


def _cell_clq(cell_points, n_neighbors, min_count):

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

    neighbor_counts = count_neighbors(valid_points, n_neighbors, agg=True)

    clq_df = _clq(neighbor_counts, counts)

    return clq_df


def _clq(neighbor_counts, counts):
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

    return clq_df.drop('count', axis=1)


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

    return
    # nai->b / nb (N - 1)
