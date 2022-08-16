import numpy as np
import pandas as pd
from numpy.random import default_rng
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.neighbors import NearestNeighbors

from ..preprocessing import get_points


def coloc_quotient(
    data, n_neighbors=25, radius=None, min_count=20, batchsize=64, copy=False
):
    """Calculate pairwise gene colocalization quotient in each cell. Specify either n_neighbors or radius, for knn neighbors or radius neighbors.

    Parameters
    ----------
    adata : AnnData
        Anndata formatted spatial data.
    n_neighbors : int
        Number of nearest neighbors to consider, default 25
    radius : int
        Max radius to search for neighboring points, default None
    min_count : int
        Minimum points needed to be eligible for analysis. default 5
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
        points.loc[points["cell"] == adata.obs_names[0]], n_neighbors, radius, min_count
    )


    ddf = dd.from_pandas(points, npartitions=1)

    # Partition so {batchsize} cells per partition
    _, group_loc = np.unique(
        points["cell"].astype(str),
        return_index=True,
    )

    if batchsize < len(group_loc):
        divisions = [group_loc[loc] for loc in range(0, len(group_loc), batchsize)]
        divisions.append(len(points) - 1)
        ddf = ddf.repartition(divisions=divisions)
        
    with ProgressBar():
        cell_metrics = (
            ddf.groupby("cell")
            .apply(
                lambda df: _cell_clq(df, n_neighbors, radius, min_count),
                meta=meta,
            )
            .compute()
            .reset_index()
            .drop("level_1", axis=1)
        )

    cell_metrics[["cell", "gene", "neighbor"]] = cell_metrics[["cell", "gene", "neighbor"]].astype(str).astype("category")
    
    
    # Save to uns['clq'] as adjacency list
    adata.uns["clq"] = cell_metrics

    return adata if copy else None


def _cell_clq(cell_points, n_neighbors, radius, min_count):

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
    n_points = valid_points.shape[0]

    # Cleanup gene categories
    valid_points["gene"] = valid_points["gene"].cat.remove_unused_categories()

    # Get neighbors within fixed outer_radius for every point
    if n_neighbors:
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(valid_points[["x", "y"]])
        point_index = nn.kneighbors(valid_points[["x", "y"]], return_distance=False)
    elif radius:
        nn = NearestNeighbors(radius=radius).fit(valid_points[["x", "y"]])
        point_index = nn.radius_neighbors(
            valid_points[["x", "y"]], return_distance=False
        )

    # Flatten adjacency list to pairs
    source_index = []
    neighbor_index = []
    for source, neighbors in zip(range(valid_points.shape[0]), point_index):
        source_index.extend([source] * len(neighbors))
        neighbor_index.extend(neighbors)

    source_index = np.array(source_index)
    neighbor_index = np.array(neighbor_index)

    # Remove self neighbors
    is_self = source_index == neighbor_index
    source_index = source_index[~is_self]
    neighbor_index = neighbor_index[~is_self]

    # Remove duplicate neighbors
    _, is_uniq = np.unique(neighbor_index, return_index=True)
    source_index = source_index[is_uniq]
    neighbor_index = neighbor_index[is_uniq]

    # Index to gene mapping; dict for fast lookup
    index2gene = valid_points["gene"].reset_index(drop=True).to_dict()

    # Map to genes
    source2neighbor = np.array(
        [[index2gene[i], index2gene[j]] for i, j in zip(source_index, neighbor_index)]
    )
    
    # For each gene, count neighbors
    obs_genes, obs_count = np.unique(source2neighbor, axis=0, return_counts=True)
    
    # TODO move to gene gene clq function
    clq_df = pd.DataFrame(obs_genes, columns=["gene", "neighbor"])
    clq_df["clq"] = (obs_count / counts.loc[clq_df["gene"]].values) / (
        counts.loc[clq_df["neighbor"]].values / n_points
    )

    return clq_df