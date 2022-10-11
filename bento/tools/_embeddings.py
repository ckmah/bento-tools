import numpy as np
import pandas as pd
from dask import dataframe as dd
from tqdm.dask import TqdmCallback
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix, vstack

from sklearn.neighbors import NearestNeighbors

from ..preprocessing import get_points


def nn_embed(
    data,
    n_neighbors=None,
    radius=None,
    agg=False,
    relative=False,
    copy=False,
):
    """
    Generate local gene neighborhood embeddings within each cell. Specify either n_neighbors or radius, not both.

    Parameters
    ----------
    data : AnnData
         Anndata formatted spatial data.
    n_neighbors : int
        Number of nearest neighbors to consider per point.
    radius : float
        Radius to consider for nearest neighbors.
    agg : bool
        Whether to aggregate nearest neighbors counts at the gene-level or for each point. Default False.
    relative : bool, optional
        Whether to normalize counts by number of neighbors. Default False.
    copy : bool, optional
        Return a copy of `data` instead of writing to data, by default False.


    """
    adata = data.copy() if copy else data

    adata.uns["points"] = get_points(adata).sort_values("cell")

    points = get_points(adata)[["cell", "gene", "x", "y"]]

    # Extract gene names and codes
    gene_names = points["gene"].cat.categories.tolist()
    gene_codes = points["gene"].cat.codes
    n_genes = len(gene_names)

    # Factorize for more efficient computation
    points["gene"] = gene_codes.values

    ddf = dd.from_pandas(points, npartitions=1)

    # Process points of each cell separately
    cells, group_loc = np.unique(
        points["cell"],
        return_index=True,
    )

    end_loc = np.append(group_loc[1:], points.shape[0])

    cell_metrics = []
    for start, end in tqdm(zip(group_loc, end_loc), total=len(cells)):
        cell_points = points.iloc[start:end]
        cell_metrics.append(
            _count_neighbors(
                cell_points,
                n_genes,
                n_neighbors=n_neighbors,
                radius=radius,
                relative=relative,
                agg=agg,
            )
        )
    cell_metrics = vstack(cell_metrics)

    adata.uns["nn_genes"] = gene_names
    adata.uns["nn_embed"] = cell_metrics

    return adata if copy else None


def _count_neighbors(
    points_df, n_genes, n_neighbors=None, radius=None, relative=False, agg=True
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

    # Get gene-level neighbor counts for each gene
    if agg:
        gene_code = points_df["gene"].values
        source_genes, source_indices = np.unique(gene_code, return_index=True)

        gene_index = []

        for g, gi in zip(source_genes, source_indices):
            # First get all points for this gene
            g_neighbors = np.unique(neighbor_index[gi].flatten())
            # get unique neighbor points
            g_neighbors = gene_code[g_neighbors]  # Get point gene names
            neighbor_names, neighbor_counts = np.unique(
                g_neighbors, return_counts=True
            )  # aggregate neighbor gene counts

            for neighbor, count in zip(neighbor_names, neighbor_counts):
                gene_index.append([g, neighbor, count])

        gene_index = pd.DataFrame(gene_index, columns=["gene", "neighbor", "count"])

        return gene_index

    # Get gene-level neighbor counts for each point
    else:
        gene_code = points_df["gene"].values
        neighborhood_sizes = np.array([len(n) for n in neighbor_index])
        flat_nindex = np.concatenate(neighbor_index)

        # Count number of times each gene is a neighbor of a given point
        flat_ncodes = gene_code[flat_nindex]
        point_ncounts = []
        cur_pos = 0
        # np.bincount only works on ints but much faster than np.unique
        # https://stackoverflow.com/questions/66037744/2d-vectorization-of-unique-values-per-row-with-condition
        for s in neighborhood_sizes:
            cur_codes = flat_ncodes[cur_pos : cur_pos + s]
            point_ncounts.append(np.bincount(cur_codes, minlength=n_genes))
            cur_pos = cur_pos + s

        point_ncounts = np.array(point_ncounts)

        # Normalize by # neighbors
        if relative:
            point_ncounts = point_ncounts / neighborhood_sizes.reshape(-1, 1)

        point_ncounts = csr_matrix(point_ncounts)

        return point_ncounts
