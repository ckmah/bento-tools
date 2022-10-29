import decoupler as dc
import numpy as np
import pandas as pd
import pkg_resources
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix, vstack
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from .._utils import track
from ..preprocessing import get_points
from ..tools._shape_features import analyze_shapes


@track
def flow(
    data,
    mode="point",
    n_neighbors=None,
    radius=None,
    normalization=None,
    reduce=True,
    copy=False,
):
    """
    RNAFlow: Method for embedding spatial data with local gene expression neighborhoods.

    """
    adata = data.copy() if copy else data

    adata.uns["points"] = get_points(adata).sort_values("cell")

    points = get_points(adata)[["cell", "gene", "x", "y"]]

    if mode == "point":
        query_points = zip(points["cell"].unique(), [None] * points["cell"].nunique())
    elif mode == "cell":
        analyze_shapes(adata, "cell_shape", "raster")
        rasters = adata.obs["cell_raster"]
        query_points = dict()
        # Flatten to long dataframe
        for c, pt_vals in rasters.items():
            query_points[c] = pd.DataFrame(pt_vals, columns=["x", "y"])

    # Extract gene names and codes
    gene_names = points["gene"].cat.categories.tolist()
    gene_codes = points["gene"].cat.codes
    n_genes = len(gene_names)

    # Factorize for more efficient computation
    points["gene"] = gene_codes.values

    # Process points of each cell separately
    cells, group_loc = np.unique(
        points["cell"],
        return_index=True,
    )

    end_loc = np.append(group_loc[1:], points.shape[0])

    # Embed each cell neighborhood independently
    cell_flows = []
    for cell, start, end in tqdm(zip(cells, group_loc, end_loc), total=len(cells)):
        cell_points = points.iloc[start:end]
        fl = _count_neighbors(
            cell_points,
            n_genes,
            query_points[cell],
            radius=radius,
            n_neighbors=n_neighbors,
            agg=False,
        )
        cell_flows.append(fl)

    cell_flows = vstack(cell_flows)

    # Use scanpy for post-processing
    flow_data = AnnData(cell_flows)

    if normalization:
        print("Normalizing embedding...")

    if normalization == "log":
        sc.pp.log1p(flow_data, base=2)
    elif normalization == "total":
        sc.pp.normalize_total(adata, target_sum=1)

    # TODO: subsample fit transform to reduce memory usage
    if reduce:
        print("Reducing dimensionality...")
        sc.pp.pca(flow_data)

    # Save flow embeddings
    adata.uns["flow_genes"] = gene_names
    adata.uns["flow"] = cell_flows

    if reduce:
        adata.uns[f"flow_pca"] = flow_data.obsm["X_pca"]

    # Save downsampled flow point coordinates
    if mode == "cell":
        flow_points = []
        # Format as long dataframe
        for c, pt_vals in query_points.items():
            pt_vals["cell"] = c
            flow_points.append(pt_vals)
        adata.uns["flow_points"] = pd.concat(flow_points)

    return adata if copy else None


def _count_neighbors(
    points, n_genes, query_points=None, n_neighbors=None, radius=None, agg=True
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
        neighbor_index = (
            NearestNeighbors(radius=radius, n_jobs=-1)
            .fit(points[["x", "y"]])
            .radius_neighbors(query_points[["x", "y"]], return_distance=False)
        )

    # Get gene-level neighbor counts for each gene
    if agg:
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

    # Get gene-level neighbor counts for each point
    gene_code = points["gene"].values
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
    point_ncounts = csr_matrix(point_ncounts)

    return point_ncounts


def fazal2019_loc_scores(data, batch_size=10000, min_n=5, copy=False):
    """Compute enrichment scores from subcellular compartment gene sets from Fazal et al. 2019 (APEX-seq).
    Wrapper for `bento.tl.spatial_enrichment`.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    batch_size : int
        Number of points to process in each batch. Default 10000.
    min_n : int
        Minimum number of points required to compute enrichment score. Default 5.

    Returns
    -------
    DataFrame
        Enrichment scores for each gene set.
    """
    adata = data.copy() if copy else data

    stream = pkg_resources.resource_stream(__name__, "gene_sets/fazal2019.csv")
    gene_sets = pd.read_csv(stream)

    # Compute enrichment scores
    fe(adata, gene_sets, batch_size=batch_size, min_n=min_n)

    return adata if copy else None


@track
def fe(
    data,
    net,
    source="source",
    target="target",
    weight="weight",
    batch_size=10000,
    min_n=5,
    copy=False,
):
    """
    Perform functional enrichment on point embeddings.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    net : DataFrame
        DataFrame with columns "source", "target", and "weight". See decoupler API for more details.

    """

    # 1. embed points with bento.tl.pt_embed without normalization
    # 2. Run decoupler on embedding given net (see dc.run_* methods)
    # 3. Save results to data.uns['pt_enrichment']
    #     - columns are various pathways in net

    adata = data.copy() if copy else data

    # Make sure embedding is run first
    if "pt_embed" not in data.uns:
        print("Run bento.tl.pt_embed first.")
        return

    mat = adata.uns["pt_embed"]  # sparse matrix in csr format
    samples = adata.uns["points"].index
    features = adata.uns["pt_genes"]

    enrichment = dc.run_wsum(
        mat=[mat, samples, features],
        net=net,
        source=source,
        target=target,
        weight=weight,
        batch_size=batch_size,
        min_n=5,
        verbose=True,
    )

    scores = enrichment[1]

    adata.uns["fe"] = scores
    # adata.uns["fe_genes"] = common_genes
    _fe_stats(adata, net, source=source, target=target)

    return adata if copy else None


def _fe_stats(data, net, source="source", target="target", copy=False):

    adata = data.copy() if copy else data

    # rows = cells, columns = pathways, values = count of genes in pathway
    expr_binary = adata.to_df() >= 5
    # {cell : present gene list}
    expr_genes = expr_binary.apply(lambda row: adata.var_names[row], axis=1)

    # Count number of genes present in each pathway
    net_ngenes = net.groupby(source).size()

    sources = []
    # common_genes = {}  # list of [cells: gene set overlaps]
    common_ngenes = []  # list of [cells: overlap sizes]
    for source, group in net.groupby(source):
        sources.append(source)
        common = expr_genes.apply(lambda genes: set(genes).intersection(group[target]))
        # common_genes[sources] = np.array(common)
        common_ngenes.append(common.apply(len))

    fe_stats = pd.concat(common_ngenes, axis=1)
    fe_stats.columns = sources

    adata.uns["fe_stats"] = fe_stats
    adata.uns["fe_size"] = net_ngenes

    return adata if copy else None
