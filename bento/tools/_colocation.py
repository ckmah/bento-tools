import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.neighbors import NearestNeighbors


def coloc_quotient(data, radius=2, min_count=5, permutations=10, copy=False):
    """Calculate pairwise gene colocalization quotient in each cell.

    Parameters
    ----------
    adata : AnnData
        Anndata formatted spatial data.
    radius : int
        Max radius to search for neighboring points, default 2
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

        if permutations > 0:
            meta = {
                "neighbor_count": int,
                "neighbor_fraction": float,
                "quotient": float,
                "pvalue": float,
                "gene": str,
            }
        else:
            meta = {
                "neighbor_count": int,
                "neighbor_fraction": float,
                "quotient": float,
                "gene": str,
            }
        with ProgressBar():
            cell_metrics = (
                dd.from_pandas(points.set_index("cell"), npartitions=npartitions)
                .groupby(["cell"])
                .apply(lambda df: cell_clq(df, min_count, radius, permutations), meta=meta)
                .reset_index()
                .compute()
            )

    # cell_metrics = pd.concat(cell_metrics).reset_index()
    cell_metrics = cell_metrics.rename(columns={"level_1": "neighbor"})

    adata.uns["coloc_quotient"] = cell_metrics

    return adata if copy else None

def cell_clq(cell_points, min_count, radius, permutations):

    # Count number of points for each gene
    counts = cell_points["gene"].value_counts()

    # Only keep genes >= min_count
    counts = counts[counts >= min_count]
    valid_genes = counts.index.tolist()

    # Get points
    valid_points = cell_points[cell_points["gene"].isin(valid_genes)]
    n_points = valid_points.shape[0]

    # Cleanup gene categories
    valid_points["gene"] = valid_points["gene"].cat.remove_unused_categories()

    # Get neighbors within fixed outer_radius for every point
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

    # Remove intra-gene neighbors
    is_intragene = source_genes == neighbor_genes
    source_index = source_index[~is_intragene]
    neighbor_index = neighbor_index[~is_intragene]
    source_genes = source_genes[~is_intragene]
    neighbor_genes = neighbor_genes[~is_intragene]

    # Iterate across genes
    stats_list = []
    for cur_gene in valid_genes:

        # Select pairs where source = gene of interest
        cur_select = source_genes == cur_gene
        cur_neighbor_genes = neighbor_genes[cur_select]

        # Colocation statistics
        uniq_neighbors, obs_count = np.unique(
            cur_neighbor_genes, return_counts=True
        )
        obs_total = counts[uniq_neighbors].values
        obs_fraction = obs_count / obs_total
        obs_quotient = (obs_count / counts[cur_gene]) / (obs_total / (n_points - 1))

        # Perform permutations for significance
        perm_counts = []
        if permutations > 0:
            for i in range(permutations):
                # Draw from empirical frequencies without replacement
                rand_neighbor_genes = np.random.choice(
                    neighbor_genes, size=len(cur_neighbor_genes), replace=False
                )
                perm_genes, perm_count = np.unique(
                    rand_neighbor_genes, return_counts=True
                )
                perm_count = pd.Series(perm_count, index=perm_genes)
                perm_counts.append(perm_count)

            # Permutation statistics
            perm_counts = pd.concat(perm_counts, axis=1).T

            perm_quotients = (perm_counts / counts[cur_gene]) / (
                counts.loc[perm_counts.columns] / (n_points - 1)
            )
            perm_quotients = perm_quotients.T.reindex(uniq_neighbors, fill_value=0)

            # Fraction of times statistic is greater than permutations
            pvalue = (np.expand_dims(obs_quotient, 1) > perm_quotients).sum(
                axis=1
            ) / permutations

            # Format statistics as dataframe
            stat_df = pd.DataFrame(
                [obs_count, obs_fraction, obs_quotient, pvalue],
                index=["neighbor_count", "neighbor_fraction", "quotient", "pvalue"],
                columns=uniq_neighbors,
            ).T
            stat_df["gene"] = cur_gene
            stats_list.append(stat_df)
        else:
            # Format statistics as dataframe
            stat_df = pd.DataFrame(
                [obs_count, obs_fraction, obs_quotient],
                index=["neighbor_count", "neighbor_fraction", "quotient"],
                columns=uniq_neighbors,
            ).T
            stat_df["gene"] = cur_gene
            stats_list.append(stat_df)

    stats_df = pd.concat(stats_list)
    # stats_df["cell"] = cell
    return stats_df
