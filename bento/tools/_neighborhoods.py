import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def _count_neighbors(
    points, n_genes, query_points=None, n_neighbors=None, radius=None, agg="gene"
):
    """Build nearest neighbor index for points.

    Parameters
    ----------
    points : pd.DataFrame
        Points dataframe. Must have columns "x", "y", and "gene".
    n_genes : int
        Number of genes in overall dataset. Used to initialize unique gene counts.
    query_points : pd.DataFrame, optional
        Points to query. If None, use points_df. Default None.
    n_neighbors : int
        Number of nearest neighbors to consider per gene.
    agg : "gene", "binary", None
        Whether to aggregate nearest neighbors counts. "Gene" aggregates counts by gene, whereas "binary" counts neighbors only once per point. If None, return neighbor counts for each point.
         Default "gene".
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

    if query_points is None:
        query_points = points

    # Build knn index
    if n_neighbors:
        # Can't find more neighbors than total points
        try:
            n_neighbors = min(n_neighbors, points.shape[0])
            neighbor_index = (
                NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
                .fit(points[["x", "y"]])
                .kneighbors(query_points[["x", "y"]], return_distance=False)
            )
        except ValueError as e:
            raise ValueError(e)
    elif radius:
        try:
            neighbor_index = (
                NearestNeighbors(radius=radius, n_jobs=-1)
                .fit(points[["x", "y"]])
                .radius_neighbors(query_points[["x", "y"]], return_distance=False)
            )
        except ValueError:
            print(points.shape, query_points.shape)

    # Get gene-level neighbor counts for each gene
    if agg == "gene":
        gene_code = points["gene"].values
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

    else:
        # Get gene-level neighbor counts for each point
        gene_codes = points["gene"].cat.codes.values
        neighborhood_sizes = np.array([len(n) for n in neighbor_index])
        flat_nindex = np.concatenate(neighbor_index)
        # Get gene name for each neighbor
        flat_ncodes = gene_codes[flat_nindex]

        point_ncounts = []
        cur_pos = 0
        # np.bincount only works on ints but much faster than np.unique
        # https://stackoverflow.com/questions/66037744/2d-vectorization-of-unique-values-per-row-with-condition
        for s in neighborhood_sizes:
            cur_codes = flat_ncodes[cur_pos : cur_pos + s]
            point_neighbor_counts = np.bincount(cur_codes, minlength=n_genes)
            # Count number of times each gene is a neighbor of a given point
            if agg == "binary":
                n_indicator = (point_neighbor_counts > 0).astype(int)
                point_ncounts.append(n_indicator)

            # Quantify abundance of each gene as a neighbor of a given point
            elif agg is None:
                point_ncounts.append(point_neighbor_counts)

            cur_pos += s

        point_ncounts = np.array(point_ncounts)
        point_ncounts = csr_matrix(point_ncounts)

        return point_ncounts
