from typing import Iterable, Literal, Optional, Union

import decoupler as dc
import emoji
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import rasterio
import shapely
from anndata import AnnData
from kneed import KneeLocator
from minisom import MiniSom
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, minmax_scale, quantile_transform
from sklearn.utils import resample
from tqdm.auto import tqdm

from .._utils import register_points, track
from ..geometry import get_points, sindex_points
from ._neighborhoods import _count_neighbors
from ._shape_features import analyze_shapes


@track
@register_points("cell_raster", ["flow", "flow_embed", "flow_vis"])
def flow(
    data: AnnData,
    method: Literal["knn", "radius"] = "radius",
    n_neighbors: Optional[int] = None,
    radius: Optional[int] = None,
    render_resolution: int = 0.1,
    copy: bool = False,
):
    """
    RNAflow: Embedding each pixel as normalized local composition normalized by cell composition.
    For k-nearest neighborhoods or "knn", method, specify n_neighbors. For radius neighborhoods, specify radius.
    The default method is "radius" with radius=50. RNAflow requires a minimum of 4 genes per cell to compute all embeddings properly.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    method: str
        Method to use for local neighborhood. Either 'knn' or 'radius'.
    n_neighbors : int
        Number of neighbors to use for local neighborhood.
    radius : float
        Radius to use for local neighborhood.
    render_resolution : float
        Resolution to use for rendering embedding. Default 0.1 samples at 10% original resolution (10 units between pixels)
    copy : bool
        Whether to return a copy the AnnData object. Default False.

    Returns
    -------
    adata : AnnData
        .uns["flow"] : scipy.csr_matrix
            [pixels x genes] sparse matrix of normalized local composition.
        .uns["flow_embed"] : np.ndarray
            [pixels x components] array of embedded flow values.
        .uns["flow_vis"] : np.ndarray
            [pixels x 3] array of RGB values for visualization.
        .uns["flow_genes"] : list
            List of genes used for embedding.
        .uns["flow_variance_ratio"] : np.ndarray
            [components] array of explained variance ratio for each component.
    """
    if n_neighbors is None and radius is None:
        radius = 50

    adata = data.copy() if copy else data

    adata.uns["points"] = get_points(adata).sort_values("cell")

    points = get_points(adata)[["cell", "gene", "x", "y"]]

    # embeds points on a uniform grid
    pbar = tqdm(total=3)
    pbar.set_description(emoji.emojize("Embedding"))
    step = 1 / render_resolution
    # Get grid rasters
    analyze_shapes(
        adata,
        "cell_shape",
        "raster",
        progress=False,
        feature_kws=dict(raster={"step": step}),
    )
    # Long dataframe of raster points
    adata.uns["cell_raster"] = adata.uns["cell_raster"].sort_values("cell")
    raster_points = adata.uns["cell_raster"]

    # Extract gene names and codes
    gene_names = points["gene"].cat.categories.tolist()
    gene_codes = points["gene"].cat.codes
    n_genes = len(gene_names)

    # Factorize for more efficient computation
    # points["gene"] = gene_codes.values

    points_grouped = points.groupby("cell")
    rpoints_grouped = raster_points.groupby("cell")
    cells = list(points_grouped.groups.keys())

    cell_composition = adata[cells, gene_names].X.toarray()

    # Compute cell composition
    cell_composition = cell_composition / (cell_composition.sum(axis=1).reshape(-1, 1))
    cell_composition = np.nan_to_num(cell_composition)

    # Embed each cell neighborhood independently
    cell_flows = []
    for i, cell in enumerate(tqdm(cells, leave=False)):
        cell_points = points_grouped.get_group(cell)
        rpoints = rpoints_grouped.get_group(cell)
        if method == "knn":
            gene_count = _count_neighbors(
                cell_points,
                n_genes,
                rpoints,
                n_neighbors=n_neighbors,
                agg=None,
            )
        elif method == "radius":
            gene_count = _count_neighbors(
                cell_points,
                n_genes,
                rpoints,
                radius=radius,
                agg=None,
            )
        gene_count = gene_count.toarray()
        # embedding: distance neighborhood composition and cell composition
        # Compute composition of neighborhood
        flow_composition = gene_count / (gene_count.sum(axis=1).reshape(-1, 1))
        cflow = flow_composition - cell_composition[i]
        cflow = StandardScaler(with_mean=False).fit_transform(cflow)

        # Convert back to sparse matrix
        cflow = csr_matrix(cflow)

        cell_flows.append(cflow)

    cell_flows = vstack(cell_flows) if len(cell_flows) > 1 else cell_flows[0]
    cell_flows.data = np.nan_to_num(cell_flows.data)
    pbar.update()

    pbar.set_description(emoji.emojize("Reducing"))
    n_components = min(n_genes - 1, 10)
    pca_model = TruncatedSVD(n_components=n_components, algorithm="arpack").fit(
        cell_flows
    )
    flow_embed = pca_model.transform(cell_flows)
    variance_ratio = pca_model.explained_variance_ratio_

    # For color visualization of flow embeddings
    flow_vis = quantile_transform(flow_embed[:, :3])
    flow_vis = minmax_scale(flow_vis, feature_range=(0.1, 0.9))
    pbar.update()

    pbar.set_description(emoji.emojize("Saving"))
    adata.uns["flow"] = cell_flows  # sparse gene embedding
    adata.uns["flow_genes"] = gene_names  # gene names
    adata.uns["flow_embed"] = flow_embed
    adata.uns["flow_variance_ratio"] = variance_ratio
    adata.uns["flow_vis"] = flow_vis

    pbar.set_description(emoji.emojize("Done. :bento_box:"))
    pbar.update()
    pbar.close()

    return adata if copy else None


@track
def flowmap(
    data: AnnData,
    n_clusters: Union[Iterable[int], int] = range(2, 9),
    num_iterations: int = 1000,
    train_size: float = 0.2,
    render_resolution: float = 0.1,
    random_state: int = 11,
    plot_error: bool = True,
    copy: bool = False,
):
    """Cluster flow embeddings using self-organizing maps (SOMs) and vectorize clusters as Polygon shapes.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    n_clusters : int or list
        Number of clusters to use. If list, will pick best number of clusters
        using the elbow heuristic evaluated on the quantization error.
    num_iterations : int
        Number of iterations to use for SOM training.
    train_size : float
        Fraction of cells to use for SOM training. Default 0.2.
    render_resolution : float
        Resolution used for rendering embedding. Default 0.1.
    random_state : int
        Random state to use for SOM training. Default 11.
    plot_error : bool
        Whether to plot quantization error. Default True.
    copy : bool
        Whether to return a copy the AnnData object. Default False.

    Returns
    -------
    data : AnnData
        .uns["cell_raster"] : DataFrame
            Adds "flowmap" column denoting cluster membership.
        .uns["points"] : DataFrame
            Adds "flowmap#" columns for each cluster.
        .obs : GeoSeries
            Adds "flowmap#_shape" columns for each cluster rendered as (Multi)Polygon shapes.
    """
    adata = data.copy() if copy else data

    # Check if flow embedding has been computed
    if "flow_embed" not in adata.uns:
        raise ValueError(
            "Flow embedding has not been computed. Run `bento.tl.flow()` first."
        )

    flow_embed = adata.uns["flow_embed"]
    raster_points = adata.uns["cell_raster"]

    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    if isinstance(n_clusters, range):
        n_clusters = list(n_clusters)

    # Subsample flow embeddings for faster training
    if train_size > 1:
        raise ValueError("train_size must be less than 1.")
    if train_size == 1:
        flow_train = flow_embed
    if train_size < 1:

        flow_train = resample(
            flow_embed,
            n_samples=int(train_size * flow_embed.shape[0]),
            random_state=random_state,
        )

    # Perform SOM clustering over n_clusters range and pick best number of clusters using elbow heuristic
    pbar = tqdm(total=4)
    pbar.set_description(emoji.emojize(f"Optimizing # of clusters"))
    som_models = {}
    quantization_errors = []
    for k in tqdm(n_clusters, leave=False):
        som = MiniSom(1, k, flow_train.shape[1], random_seed=random_state)
        som.random_weights_init(flow_train)
        som.train(flow_train, num_iterations, random_order=False, verbose=False)
        som_models[k] = som
        quantization_errors.append(som.quantization_error(flow_embed))

    # Use kneed to find elbow
    if len(n_clusters) > 1:
        kl = KneeLocator(
            n_clusters, quantization_errors, curve="convex", direction="decreasing"
        )
        best_k = kl.elbow

        if plot_error:
            kl.plot_knee()
            plt.show()
    else:
        best_k = n_clusters[0]
    pbar.update()

    # Use best k to assign each sample to a cluster
    pbar.set_description(f"Assigning to {best_k} clusters")
    som = som_models[best_k]
    winner_coordinates = np.array([som.winner(x) for x in flow_embed]).T

    # Indices start at 0, so add 1
    qnt_index = np.ravel_multi_index(winner_coordinates, (1, best_k)) + 1
    raster_points["flowmap"] = qnt_index
    adata.uns["cell_raster"] = raster_points.copy()

    pbar.update()

    # Vectorize polygons in each cell
    pbar.set_description(emoji.emojize("Vectorizing domains"))
    cells = raster_points["cell"].unique().tolist()
    # Scale down to render resolution
    # raster_points[["x", "y"]] = raster_points[["x", "y"]] * render_resolution

    # Cast to int
    raster_points[["x", "y", "flowmap"]] = raster_points[["x", "y", "flowmap"]].astype(
        int
    )

    rpoints_grouped = raster_points.groupby("cell")
    flowmap_df = dict()
    for cell in tqdm(cells, leave=False):
        rpoints = rpoints_grouped.get_group(cell)

        # Fill in image at each point xy with flowmap value by casting to dense matrix
        image = (
            csr_matrix(
                (
                    rpoints["flowmap"],
                    (
                        (rpoints["y"] * render_resolution).astype(int),
                        (rpoints["x"] * render_resolution).astype(int),
                    ),
                )
            )
            .todense()
            .astype("int16")
        )

        # Find all the contours
        contours = rasterio.features.shapes(image)
        polygons = np.array([(shapely.geometry.shape(p), v) for p, v in contours])
        shapes = gpd.GeoDataFrame(
            polygons[:, 1],
            geometry=gpd.GeoSeries(polygons[:, 0]).T,
            columns=["flowmap"],
        )

        # Remove background shape
        shapes["flowmap"] = shapes["flowmap"].astype(int)
        shapes = shapes[shapes["flowmap"] != 0]

        # Group same fields as MultiPolygons
        shapes = shapes.dissolve("flowmap")["geometry"]

        flowmap_df[cell] = shapes

    flowmap_df = pd.DataFrame.from_dict(flowmap_df).T
    flowmap_df.columns = "flowmap" + flowmap_df.columns.astype(str) + "_shape"

    # Upscale to match original resolution
    flowmap_df = flowmap_df.apply(
        lambda col: gpd.GeoSeries(col).scale(
            xfact=1 / render_resolution, yfact=1 / render_resolution, origin=(0, 0)
        )
    )
    pbar.update()

    pbar.set_description("Saving")
    old_cols = adata.obs.columns[adata.obs.columns.str.startswith("flowmap")]
    adata.obs = adata.obs.drop(old_cols, axis=1, errors="ignore")

    adata.obs[flowmap_df.columns] = flowmap_df.reindex(adata.obs_names)

    old_cols = adata.uns["points"].columns[
        adata.uns["points"].columns.str.startswith("flowmap")
    ]
    adata.uns["points"] = adata.uns["points"].drop(old_cols, axis=1)

    # TODO SLOW
    sindex_points(adata, "points", flowmap_df.columns.tolist())
    pbar.update()
    pbar.set_description("Done")
    pbar.close()

    return adata if copy else None


def fe_fazal2019(
    data: AnnData, batch_size: int = 10000, min_n: int = 5, copy: bool = False
) -> Optional[AnnData]:
    """Compute enrichment scores from subcellular compartment gene sets from Fazal et al. 2019 (APEX-seq).
    Wrapper for `bento.tl.fe`.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    batch_size : int
        Number of points to process in each batch. Default 10000.
    min_n : int
        Minimum number of targets per source. If less, sources are removed.
    copy : bool
        Return a copy instead of writing to `adata`. Default False.
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
@register_points("cell_raster", ["flow_fe"])
def fe(
    data: AnnData,
    net: pd.DataFrame,
    groupby: Optional[str] = None,
    source: Optional[str] = None,
    target: Optional[str] = None,
    weight: Optional[str] = None,
    batch_size: int = 10000,
    min_n: int = 0,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Perform functional enrichment on point embeddings. Wrapper for decoupler wsum function.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    net : DataFrame
        DataFrame with columns "source", "target", and "weight". See decoupler API for more details.
    groupby : str, optional
        Column in `adata.uns["cell_raster"]` to group by. Default None.
    source : str, optional
        Column name for source nodes in `net`. Default "source".
    target : str, optional
        Column name for target nodes in `net`. Default "target".
    weight : str, optional
        Column name for weights in `net`. Default "weight".
    batch_size : int
        Number of points to process in each batch. Default 10000.
    min_n : int
        Minimum number of targets per source. If less, sources are removed.
    copy : bool
        Return a copy instead of writing to `adata`. Default False.

    Returns
    -------
    adata : AnnData
        uns["flow_fe"] : DataFrame
            Enrichment scores for each gene set.
    """

    adata = data.copy() if copy else data

    # Make sure embedding is run first
    if "flow" not in data.uns:
        print("Run bento.tl.flow first.")
        return

    mat = adata.uns["flow"]  # sparse matrix in csr format
    zero_rows = mat.getnnz(1) == 0

    samples = adata.uns["cell_raster"].index.astype(str)
    features = adata.uns["flow_genes"]

    enrichment = dc.run_wsum(
        mat=[mat, samples, features],
        net=net,
        source=source,
        target=target,
        weight=weight,
        batch_size=batch_size,
        min_n=min_n,
        verbose=True,
    )

    scores = enrichment[1].reindex(index=samples)

    if groupby:
        scores = scores.groupby(
            adata.uns["cell_raster"][groupby].reset_index(drop=True)
        ).mean()
        scores = adata.uns["cell_raster"].merge(
            scores, left_on="flow", right_index=True, how="left"
        )[scores.columns]

    adata.uns["flow_fe"] = scores
    _fe_stats(adata, net, source=source, target=target, copy=copy)

    return adata if copy else None


def _fe_stats(
    data: AnnData,
    net: pd.DataFrame,
    source: str = "source",
    target: str = "target",
    copy: bool = False,
):

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
