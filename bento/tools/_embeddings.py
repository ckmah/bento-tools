import decoupler as dc
import numpy as np
import pandas as pd
import pkg_resources
import scanpy as sc
from anndata import AnnData
from matplotlib.colors import to_hex
from scipy.sparse import csr_matrix, vstack
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import quantile_transform
from tqdm.auto import tqdm

from .._utils import track
from ..preprocessing import get_points


@track
def pt_embed(
    data,
    n_neighbors=None,
    radius=None,
    normalization=None,
    reduce=None,
    copy=False,
):
    """
    Generate local gene neighborhood embeddings within each cell. Specify either n_neighbors or radius, not both.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    n_neighbors : int
        Number of nearest neighbors to consider per point.
    radius : float
        Radius to consider for nearest neighbors.
    normalization : str, optional
        Whether to normalize embeddings. Options include "log" and "total". Default False.
    reduce : str
        Dimensionality reduction method to use. Options include "pca" and "umap". Default None.
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

    # Process points of each cell separately
    cells, group_loc = np.unique(
        points["cell"],
        return_index=True,
    )

    end_loc = np.append(group_loc[1:], points.shape[0])

    # Embed each cell neighborhood independently
    cell_metrics = []
    for start, end in tqdm(zip(group_loc, end_loc), total=len(cells)):
        cell_points = points.iloc[start:end]
        cell_metrics.append(
            _count_neighbors(
                cell_points, n_genes, n_neighbors=n_neighbors, radius=radius, agg=False
            )
        )
    cell_metrics = vstack(cell_metrics)

    # Use scanpy for post-processing
    ndata = AnnData(cell_metrics)

    if normalization:
        print("Normalizing embedding...")
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

    if normalization == "log":
        sc.pp.log1p(ndata, base=2)
    elif normalization == "total":
        sc.pp.normalize_total(adata, target_sum=1)

    if reduce:
        print("Reducing dimensionality...")

        if reduce == "pca":
            sc.pp.pca(ndata)
        elif reduce == "umap":
            sc.pp.neighbors(ndata, n_neighbors=30)
            sc.tl.umap(ndata, n_components=3)

        colors_rgb = quantile_transform(
            ndata.obsm[f"X_{reduce}"][:, :3]
        )  # TODO: move this to plotting
        colors = [to_hex(c) for c in colors_rgb]

    # Save embeddings
    adata.uns["pt_genes"] = gene_names
    adata.uns["pt_embed"] = cell_metrics

    if reduce:
        adata.uns[f"pt_{reduce}"] = ndata.obsm[f"X_{reduce}"]
        adata.uns["points"][
            ["red_channel", "green_channel", "blue_channel"]
        ] = colors_rgb
        adata.uns["points"]["embed_color"] = colors

    return adata if copy else None


def _count_neighbors(points_df, n_genes, n_neighbors=None, radius=None, agg=True):
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
        point_ncounts = csr_matrix(point_ncounts)

        return point_ncounts


def fazal2019_loc_scores(data, batch_size=10000, min_n=5, copy=False):
    """Compute enrichment scores from subcellular compartment gene sets from Fazal et al. 2019 (APEX-seq). Wrapper for `bento.tl.spatial_enrichment`.

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
        common = expr_genes.apply(
            lambda genes: set(genes).intersection(group[target])
        )
        # common_genes[sources] = np.array(common)
        common_ngenes.append(common.apply(len))

    fe_stats = pd.concat(common_ngenes, axis=1)
    fe_stats.columns = sources

    adata.uns["fe_stats"] = fe_stats
    adata.uns["fe_size"] = net_ngenes

    return adata if copy else None