import decoupler as dc
import numpy as np
import pandas as pd
import pkg_resources
import scanpy as sc
import anndata
from anndata import AnnData
from scipy.sparse import vstack
from tqdm.auto import tqdm
from sklearn.preprocessing import quantile_transform


from .._utils import track
from ..geometry import get_points
from ._neighborhoods import _count_neighbors
from ._shape_features import analyze_shapes


@track
def flow(
    data,
    mode="point",
    n_neighbors=None,
    radius=None,
    render_resolution=0.01,
    copy=False,
):
    """
    RNAFlow: Method for embedding spatial data with local gene expression neighborhoods.
    Must specify one of `n_neighbors` or `radius`.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    mode : str
        Mode to compute embeddings. "Point" will embed every point while "cell" performs
        embedding for set of points on a uniform grid across the entire cell. Default "point".
    n_neighbors : int
        Number of neighbors to use for local neighborhood.
    radius : float
        Radius to use for local neighborhood.
    render_resolution : float
        Resolution to use for rendering embedding (for mode=cell). Default 0.01.
    copy : bool
        Whether to return a copy the AnnData object. Default False.
    """
    adata = data.copy() if copy else data

    adata.uns["points"] = get_points(adata).sort_values("cell")

    points = get_points(adata)[["cell", "gene", "x", "y"]]

    # "point" mode embeds every point while "cell" mode embeds points on a uniform grid
    if mode == "point":
        step = None
        query_points = dict(
            zip(points["cell"].unique(), [None] * points["cell"].nunique())
        )
    elif mode == "cell":
        print(f"Embedding at {100*render_resolution}% resolution...")
        step = 1 / render_resolution
        # Get grid rasters
        analyze_shapes(
            adata,
            "cell_shape",
            "raster",
            feature_kws=dict(raster={"step": step}, progress=False),
        )
        rasters = adata.obs["cell_raster"]

        # Cell to raster mapping for easy lookup
        query_points = dict()
        for c, pt_vals in rasters.items():
            query_points[c] = pd.DataFrame(pt_vals, columns=["x", "y"])

    # Extract gene names and codes
    gene_names = points["gene"].cat.categories.tolist()
    gene_codes = points["gene"].cat.codes
    n_genes = len(gene_names)

    # Factorize for more efficient computation
    points["gene"] = gene_codes.values

    # Process points of each cell separately
    cells_ordered, group_loc = np.unique(
        points["cell"],
        return_index=True,
    )
    end_loc = np.append(group_loc[1:], points.shape[0])

    # Embed each cell neighborhood independently
    cell_flows = []
    for cell, start, end in tqdm(
        zip(cells_ordered, group_loc, end_loc), total=len(cells_ordered)
    ):
        cell_points = points.iloc[start:end]
        fl = _count_neighbors(
            cell_points,
            n_genes,
            query_points[cell],
            radius=radius,
            n_neighbors=n_neighbors,
            agg=False,
        )

        # log2 fold change wrt cell
        fl = np.log2((fl.todense() + 1) / (adata[cell].X.toarray() + 1))

        cell_flows.append(fl)

    cell_flows = vstack(cell_flows).todense()
    cell_flows = np.nan_to_num(cell_flows)
    flow_data = AnnData(cell_flows)

    print("Reducing dimensionality...")
    sc.tl.pca(flow_data)

    embed = flow_data.obsm["X_pca"]

    print("Quantile transforming...")
    embed = quantile_transform(embed)

    # Save flow embeddings; if "point" mode, size of cell_flows is same as points. if "cell" mode, size of cell_flows is same as rasters
    adata.uns["flow_genes"] = gene_names
    adata.uns["flow"] = flow_data.X
    adata.uns["flow_embed"] = embed
    adata.uns["flow_variance_ratio"] = flow_data.uns["pca"]["variance_ratio"]

    # Save downsampled flow point coordinates
    if mode == "cell":
        flow_points = []
        # Format as long dataframe
        for c in cells_ordered:
            pt_vals = query_points[c].copy()
            pt_vals["cell"] = c
            flow_points.append(pt_vals)
        adata.uns["flow_points"] = pd.concat(flow_points)
    elif mode == "point" and "flow_points" in adata.uns:
        del adata.uns["flow_points"]

    print("Done.")

    return adata if copy else None


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

    adata = data.copy() if copy else data

    # Make sure embedding is run first
    if "flow" not in data.uns:
        print("Run bento.tl.flow first.")
        return

    mat = adata.uns["flow"]  # sparse matrix in csr format

    if "flow_points" in adata.uns:
        samples = adata.uns["flow_points"].index
    else:
        samples = adata.uns["points"].index
    features = adata.uns["flow_genes"]

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
    _fe_stats(adata, net, source=source, target=target, copy=copy)

    return adata if copy else None


def _fe_stats(data, net, source="source", target="target", copy=False):

    adata = data.copy() if copy else data

    # rows = cells, columns = pathways, values = count of genes in pathway
    expr_binary = adata.to_df() >= 5
    # {cell : present gene list}
    expr_genes = expr_binary.apply(lambda row: adata.var_names[row], axis=1)

    # Count number of genes present in each pathway
    net_ngenes = net.groupby(source).size().to_frame().T.rename(index={0: "n_genes"})

    sources = []
    # common_genes = {}  # list of [cells: gene set overlaps]
    common_ngenes = []  # list of [cells: overlap sizes]
    for source, group in net.groupby(source):
        sources.append(source)
        common = expr_genes.apply(lambda genes: set(genes).intersection(group[target]))
        # common_genes[source] = np.array(common)
        common_ngenes.append(common.apply(len))

    fe_stats = pd.concat(common_ngenes, axis=1)
    fe_stats.columns = sources

    adata.uns["fe_stats"] = fe_stats
    # adata.uns["fe_genes"] = common_genes
    adata.uns["fe_ngenes"] = net_ngenes

    return adata if copy else None
