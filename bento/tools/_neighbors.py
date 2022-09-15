import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors

from ..preprocessing import get_points


def local_point_embedding(
    data,
    n_neighbors=None,
    radius=None,
    relative=False,
    copy=False,
):
    adata = data.copy() if copy else data

    points = (
        get_points(adata).sort_values("cell")[["cell", "gene", "x", "y"]].reset_index()
    )

    # Process points of each cell separately
    cells, start_loc = np.unique(
        points["cell"],
        return_index=True,
    )
    end_loc = np.append(start_loc[1:], points.shape[0])

    cell_metrics = []
    for start, end in tqdm(zip(start_loc, end_loc), total=len(cells)):
        cell_points = points.iloc[start:end]
        cell_metrics.append(
            _count_neighbors(
                cell_points,
                n_neighbors=n_neighbors,
                radius=radius,
                relative=relative,
                agg=False,
            )
        )

    cell_metrics = pd.concat(cell_metrics, axis=0)

    adata.uns["local_point_embed"] = cell_metrics

    return adata if copy else None


def _count_neighbors(
    points_df, n_neighbors=None, radius=None, relative=False, agg=True
):
    """Build nearest neighbor index for points.

    Parameters
    ----------
    points_df : pd.DataFrame
        Points dataframe. Must have columns "x", "y", and "gene".
    n_neighbors : int
        Number of nearest neighbors to consider per gene.
    agg : bool
        Whether to aggregate nearest neighbors counts at the gene-level or for each point. Default True.
    Returns
    -------
    DataFrame or dict of dicts
        If agg is True, returns a DataFrame with columns "gene", "neighbor", and "count".
        If agg is False, returns a list of dicts, one for each point. Dict keys are gene names, values are counts.

    """
    if n_neighbors and radius:
        raise ValueError("Only specify one of n_neighbors or radius, not both.")
    if not n_neighbors and not radius:
        raise ValueError("Neither n_neighbors or radius is specified, one required.")

    if points_df.shape[0] < 1:
        return pd.DataFrame(
            [0] * points_df["gene"].nunique(),
            columns=points_df["gene"].cat.categories,
        )

    # Build knn index
    if n_neighbors:
        # Can't find more neighbors than total points
        n_neighbors = min(n_neighbors, points_df.shape[0])
        neighbor_index = (
            NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
            .fit(points_df[["x", "y"]])
            .kneighbors(points_df[["x", "y"]], return_distance=False)
        )
    elif radius:
        neighbor_index = (
            NearestNeighbors(radius=radius, n_jobs=-1)
            .fit(points_df[["x", "y"]])
            .radius_neighbors(points_df[["x", "y"]], return_distance=False)
        )

    gene_names = points_df["gene"].cat.categories.values
    gene_codes = points_df["gene"].cat.codes.values

    # Get gene-level neighbor counts for each gene
    if agg is True:
        source_genes, source_indices = np.unique(gene_codes, return_index=True)

        gene_index = []

        for g, gi in zip(source_genes, source_indices):
            # First get all points for this gene
            g_neighbors = np.unique(neighbor_index[gi].flatten())
            # get unique neighbor points
            g_neighbors = gene_codes[g_neighbors]  # Get point gene names
            neighbor_names, neighbor_counts = np.unique(
                g_neighbors, return_counts=True
            )  # aggregate neighbor gene counts

            for neighbor, count in zip(neighbor_names, neighbor_counts):
                gene_index.append([g, neighbor, count])

        gene_index = pd.DataFrame(gene_index, columns=["gene", "neighbor", "count"])
        gene_index["gene"] = points_df["gene"].cat.categories[gene_index["gene"]]
        return gene_index

    # Get gene-level neighbor counts for each point
    else:
        neighborhood_sizes = [len(n) for n in neighbor_index]
        flat_index = np.concatenate(neighbor_index).ravel()

        # Count number of times each gene is a neighbor of a given point
        point_neighbor_counts = []
        label_query = gene_codes[flat_index]
        cur_pos = 0

        for s in neighborhood_sizes:
            cur_neighbors = label_query[cur_pos : cur_pos + s]
            cur_labels, cur_counts = np.unique(cur_neighbors, return_counts=True)
            cur_point_counts = dict(zip(cur_labels, cur_counts))
            point_neighbor_counts.append(cur_point_counts)
            cur_pos = cur_pos + s

        # [Points x gene]
        point_neighbor_counts = pd.DataFrame(
            point_neighbor_counts,
            index=points_df["index"],
        )
        point_neighbor_counts.columns = gene_names[
            point_neighbor_counts.columns
        ].tolist()

        if relative:
            point_neighbor_counts = point_neighbor_counts.div(
                point_neighbor_counts.sum(axis=1), axis=0
            )

        return point_neighbor_counts
