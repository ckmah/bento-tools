import decoupler as dc
import numpy as np
import pandas as pd
import pkg_resources
from scipy.sparse import vstack, csr_matrix
from scipy.stats import mstats
from tqdm.auto import tqdm
from sklearn.preprocessing import quantile_transform, minmax_scale, StandardScaler
from sklearn.decomposition import TruncatedSVD
from minisom import MiniSom
from skimage.transform import rescale
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
import emoji

from .._utils import track
from ..geometry import get_points
from ._neighborhoods import _count_neighbors
from ._shape_features import analyze_shapes
from ..geometry import sindex_points


def robust_clr(mtx):
    """Robust CLR transform on 2d numpy array. Use geometric mean of nonzero values"""
    mtx_ma = np.ma.masked_equal(mtx, 0)
    gmeans = mstats.gmean(mtx_ma, axis=1).reshape(-1, 1)
    # CLR transform
    rclr = np.log(mtx_ma / gmeans)
    rclr = np.ma.asarray(rclr)
    return mtx


@track
def flow(
    data,
    n_neighbors=None,
    radius=50,
    render_resolution=0.01,
    n_clusters=5,
    num_iterations=1000,
    random_state=11,
    copy=False,
):
    """
    RNAFlow: Method for embedding spatial data with local gene expression neighborhoods.
    Must specify one of `n_neighbors` or `radius`.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    norm : bool
        Whether to normalize embedding by cell specific expression. Default True.
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

    # embeds points on a uniform grid
    pbar = tqdm(total=5)
    pbar.set_description(emoji.emojize(f"Embedding"))
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
    points["gene"] = gene_codes.values

    points_grouped = points.groupby("cell")
    rpoints_grouped = raster_points.groupby("cell")
    cells_ordered = list(points_grouped.groups.keys())

    # Compute cell composition
    cell_composition = adata[cells_ordered, gene_names].X.toarray()
    cell_composition = cell_composition / (cell_composition.sum(axis=1).reshape(-1, 1))
    cell_composition = robust_clr(cell_composition)
    cell_composition = np.nan_to_num(cell_composition)

    # Embed each cell neighborhood independently
    cell_flows = []
    for i, cell in enumerate(cells_ordered):
        cell_points = points_grouped.get_group(cell)
        rpoints = rpoints_grouped.get_group(cell)
        gene_count = _count_neighbors(
            cell_points,
            n_genes,
            rpoints,
            radius=radius,
            n_neighbors=n_neighbors,
            agg=False,
        )
        gene_count = gene_count.toarray()

        # Compute flow: aitchison distance between cell and neighborhood composition
        fl_composition = gene_count / (gene_count.sum(axis=1).reshape(-1, 1))
        fl_composition = robust_clr(fl_composition)
        cflow = fl_composition - cell_composition[i]

        # Convert back to sparse matrix
        cflow = csr_matrix(cflow)

        # Normalize within cell
        cflow = StandardScaler(with_mean=False).fit_transform(cflow)

        cell_flows.append(cflow)

    cell_flows = vstack(cell_flows) if len(cell_flows) > 1 else cell_flows[0]
    cell_flows.data = np.nan_to_num(cell_flows.data)
    pbar.update()

    pbar.set_description(emoji.emojize("Reducing"))
    pca_model = TruncatedSVD(n_components=10).fit(cell_flows)
    flow_embed = pca_model.transform(cell_flows)
    variance_ratio = pca_model.explained_variance_ratio_

    # For color visualization of flow embeddings
    flow_vis = quantile_transform(flow_embed)
    flow_vis = minmax_scale(flow_vis, feature_range=(0.1, 0.9))
    flow_vis = pd.DataFrame(flow_vis[:, :3], columns=["c1", "c2", "c3"])
    pbar.update()

    pbar.set_description(emoji.emojize("Fitting SOM"))
    som = MiniSom(1, n_clusters, flow_embed.shape[1], random_seed=random_state)
    som.random_weights_init(flow_embed)
    som.train(flow_embed, num_iterations, random_order=True, verbose=False)
    winner_coordinates = np.array([som.winner(x) for x in flow_embed]).T
    qnt_index = (
        np.ravel_multi_index(winner_coordinates, (1, n_clusters)) + 1
    )  # start clusters at 1
    raster_points["flow"] = qnt_index
    pbar.update()

    pbar.set_description(emoji.emojize("Clustering"))
    flow_field_df = dict()
    for cell in cells_ordered:
        rpoints = rpoints_grouped.get_group(cell)
        # Render to scaled image
        rpoints[["x", "y"]] = rpoints[["x", "y"]] * render_resolution
        max_x = int(rpoints["x"].max())
        max_y = int(rpoints["y"].max())
        image = np.zeros((max_y + 1, max_x + 1))
        for x, y, val in rpoints[["x", "y", "flow"]].values.astype(int):
            image[y, x] = val

        # Scale back up to original
        image = rescale(image, 1 / render_resolution, order=0)
        image = image.astype("int32")

        # Find all the contours
        contours = rasterio.features.shapes(image)
        polygons = np.array([(Polygon(p["coordinates"][0]), v) for p, v in contours])
        shapes = gpd.GeoDataFrame(
            polygons[:, 1],
            geometry=gpd.GeoSeries(polygons[:, 0]).T,
            columns=["flow"],
        )

        # Cast as int and remove background shape
        shapes["flow"] = shapes["flow"].astype(int)
        shapes = shapes[shapes["flow"] != 0]

        # Group same fields as MultiPolygons
        shapes = shapes.dissolve("flow")["geometry"]
        shapes.index = "flow" + shapes.index.astype(str) + "_shape"

        flow_field_df[cell] = shapes

    flow_field_df = pd.DataFrame(flow_field_df).T
    pbar.update()

    pbar.set_description(emoji.emojize(":bento_box: Saving"))
    adata.uns["flow"] = cell_flows  # sparse gene rclr
    adata.uns["flow_genes"] = gene_names  # gene names
    adata.uns["flow_embed"] = flow_embed
    adata.uns["flow_variance_ratio"] = variance_ratio
    adata.uns["flow_vis"] = flow_vis
    adata.uns["cell_raster"] = raster_points

    adata.obs = adata.obs.drop(flow_field_df.columns, axis=1, errors="ignore")
    adata.obs = adata.obs.join(flow_field_df)

    # TODO test this
    sindex_points(adata, "points", flow_field_df.columns.tolist())

    pbar.set_description(emoji.emojize(":bento_box: Done"))
    pbar.update()
    pbar.close()

    return adata if copy else None


def fe_fazal2019(data, batch_size=10000, min_n=5, copy=False):
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
    groupby=None,
    source="source",
    target="target",
    weight="weight",
    batch_size=10000,
    min_n=0,
    copy=False,
):
    """
    Perform functional enrichment on point embeddings. Wrapper for decoupler wsum function.

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
