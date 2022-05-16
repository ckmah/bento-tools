import numpy as np
import pandas as pd
from numpy.random import default_rng
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.neighbors import NearestNeighbors


def coloc_quotient(
    data, n_neighbors=25, radius=None, min_count=5, permutations=10, copy=False
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
    permutations : int
        Number of permutations to compute significance TODO account for multiple testing
    Returns
    -------
    adata : AnnData
        .uns['coloc_quotient']: Pairwise gene colocalization similarity within each cell formatted as a long dataframe.
    """

    adata = data.copy() if copy else data

    points = data.uns["points"][["x", "y", "gene", "cell"]]
    points = points[
        points["cell"].isin(data.obs_names) & points["gene"].isin(data.var_names)
    ]

    ngroups = points.groupby(["cell"]).ngroups
    if ngroups > 0:
        npartitions = min(100, ngroups)

        with ProgressBar():
            cell_metrics = (
                dd.from_pandas(points.set_index("cell"), npartitions=npartitions)
                .groupby(["cell"])
                .apply(
                    lambda df: _cell_clq(
                        df, n_neighbors, radius, min_count, permutations
                    ),
                    meta=object,
                )
                .compute()
            )

    colnames = {
        "neighbor": str,
        "neighbor_count": int,
        "neighbor_fraction": float,
        "quotient": float,
        "pvalue": float,
        "gene": str,
    }
    # cell_metrics = cell_metrics.drop('level_1', axis=1).astype(meta)
    cell_labels = [
        [cell] * len(v) for cell, v in zip(cell_metrics.index, cell_metrics.values)
    ]
    cell_labels = np.concatenate(cell_labels)

    cell_metrics = pd.DataFrame(np.concatenate(cell_metrics.values), columns=list(colnames.keys()))
    cell_metrics = cell_metrics.astype(colnames)
    cell_metrics["cell"] = cell_labels
    

    adata.uns["coloc_quotient"] = cell_metrics

    return adata if copy else None


def _cell_clq(cell_points, n_neighbors, radius, min_count, permutations):

    # Count number of points for each gene
    counts = cell_points["gene"].value_counts()

    # Only keep genes >= min_count
    counts = counts[counts >= min_count]
    valid_genes = counts.sort_index().index.tolist()
    counts = counts[valid_genes]

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
    source_genes = np.array([index2gene[i] for i in source_index])
    neighbor_genes = np.array([index2gene[i] for i in neighbor_index])

    # Preshuffle neighbors for permutations
    perm_neighbors = []
    if permutations > 0:
        # Permute neighbors
        rng = default_rng()
        for i in range(permutations):
            perm_neighbors.append(rng.permutation(neighbor_genes))

    neighbor_space = {g: 0 for g in valid_genes}

    # Iterate across genes
    stats_list = []

    for cur_gene, cur_total in zip(valid_genes, counts[valid_genes]):

        # Select pairs where source = gene of interest
        cur_neighbor_genes = neighbor_genes[source_genes == cur_gene]

        # Count neighbors
        obs_genes, obs_count = np.unique(cur_neighbor_genes, return_counts=True)

        # Save counts and order with dict
        obs_space = neighbor_space.copy()
        obs_space.update(zip(obs_genes, obs_count))
        obs_count = np.array(list(obs_space.values()))

        # Calculate colocation quotient for all neighboring genes
        # print(obs_count, counts)
        obs_quotient = (obs_count / cur_total) / ((counts - 1) / (n_points - 1))
        obs_quotient = np.expand_dims(obs_quotient, 0)

        obs_fraction = obs_count / counts

        # Perform permutations for significance
        if permutations > 0:
            perm_counts = []
            for i in range(permutations):
                # Count neighbors
                perm_genes, perm_count = np.unique(
                    perm_neighbors[i], return_counts=True
                )

                # Save counts
                perm_space = neighbor_space.copy()
                perm_space.update(dict(zip(perm_genes, perm_count)))
                perm_counts.append(np.array(list(perm_space.values())))

            # (permutations, len(valid_genes)) array
            perm_counts = np.array(perm_counts)

            # Calculate colocation quotient
            perm_quotients = (perm_counts / cur_total) / (
                (counts.values - 1) / (n_points - 1)
            )

            # Fraction of times statistic is greater than permutations
            pvalue = (
                2
                * np.array(
                    [
                        np.greater_equal(obs_quotient, perm_quotients).sum(axis=0),
                        np.less_equal(obs_quotient, perm_quotients).sum(axis=0),
                    ]
                ).min(axis=0)
                / permutations
            )

            stats_list.append(
                np.array(
                    [
                        obs_fraction.index,
                        obs_count,
                        obs_fraction.values,
                        obs_quotient[0],
                        pvalue,
                        [cur_gene] * len(obs_count),
                    ]
                )
            )

        else:
            stats_list.append(
                np.array(
                    [
                        obs_fraction.index,
                        obs_count,
                        obs_fraction.values,
                        obs_quotient[0],
                        [1] * len(obs_count),
                        [cur_gene] * len(obs_count),
                    ]
                )
            )

    #     stats_df = pd.DataFrame(
    #         np.concatenate(stats_list, axis=1).T,
    #         index=["neighbor", "neighbor_count", "neighbor_fraction", "quotient", "pvalue", "gene"],
    #     ).T
    return np.concatenate(stats_list, axis=1).T
