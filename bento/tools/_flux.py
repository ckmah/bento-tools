from typing import Iterable, Literal, Optional, Union

import decoupler as dc
import emoji
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pkg_resources
import rasterio
import rasterio.features
import shapely
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from kneed import KneeLocator
from minisom import MiniSom
from shapely import Polygon
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, minmax_scale, quantile_transform
from sklearn.utils import resample
from tqdm.auto import tqdm

from ..geometry import get_points, sjoin_points, set_points_metadata
from ..tools._neighborhoods import _count_neighbors
from ..tools._shape_features import analyze_shapes


def flux(
    sdata: SpatialData,
    points_key: str = "transcripts",
    instance_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    method: Literal["knn", "radius"] = "radius",
    n_neighbors: Optional[int] = None,
    radius: Optional[int] = 50,
    res: int = 0.1,
    random_state: int = 11,
    recompute: bool = False,
):
    """
    RNAflux: Embedding each pixel as normalized local composition normalized by cell composition.
    For k-nearest neighborhoods or "knn", method, specify n_neighbors. For radius neighborhoods, specify radius.
    The default method is "radius" with radius=50. RNAflux requires a minimum of 4 genes per cell to compute all embeddings properly.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    points_key : str
        key for points element that holds transcript coordinates
    instance_key : str
        Key for cell_boundaries instances
    feature_key : str
        Key for gene instances
    method: str
        Method to use for local neighborhood. Either 'knn' or 'radius'.
    n_neighbors : int
        Number of neighbors to use for local neighborhood.
    radius : float
        Radius to use for local neighborhood.
    res : float
        Resolution to use for rendering embedding. Default 0.05 samples at 5% original resolution (5 units between pixels)

    Returns
    -------
    sdata : SpatialData
        .points["{instance_key}_raster"]: pd.DataFrame
            Length pixels DataFrame containing all computed flux values, embeddings, and colors as columns in a single DataFrame.
            flux values: <gene_name> for each gene used in embedding.
            embeddings: flux_embed_<i> for each component of the embedding.
            colors: hex color codes for each pixel.
        .table.uns["flux_genes"] : list
            List of genes used for embedding.
        .table.uns["flux_variance_ratio"] : np.ndarray
            [components] array of explained variance ratio for each component.
    """

    if (
        f"{instance_key}_raster" in sdata.points
        and len(sdata.points[f"{instance_key}_raster"].columns) > 3
        and not recompute
    ):
        return

    if n_neighbors is None and radius is None:
        radius = 50

    points = get_points(sdata, points_key=points_key, astype="pandas", sync=True)
    points = points[[instance_key, feature_key, "x", "y"]].sort_values(instance_key)

    # embeds points on a uniform grid
    pbar = tqdm(total=3)
    pbar.set_description(emoji.emojize("Embedding"))
    step = 1 / res
    # Get grid rasters
    analyze_shapes(
        sdata,
        instance_key,
        "raster",
        progress=False,
        feature_kws=dict(raster={"step": step}),
    )

    raster_points = get_points(
        sdata, points_key=f"{instance_key}_raster", astype="pandas", sync=True
    ).sort_values(instance_key)

    # Extract gene names and codes
    gene_names = points[feature_key].cat.categories.tolist()
    n_genes = len(gene_names)

    points_grouped = points.groupby(instance_key)
    rpoints_grouped = raster_points.groupby(instance_key)
    cells = list(points_grouped.groups.keys())

    cell_composition = sdata.table[cells, gene_names].X.toarray()

    # Compute cell composition
    cell_composition = cell_composition / (cell_composition.sum(axis=1).reshape(-1, 1))
    cell_composition = np.nan_to_num(cell_composition)

    # Embed each cell neighborhood independently
    cell_fluxs = []
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
        flux_composition = gene_count / (gene_count.sum(axis=1).reshape(-1, 1))
        cflux = flux_composition - cell_composition[i]
        cflux = StandardScaler(with_mean=False).fit_transform(cflux)

        # Convert back to sparse matrix
        cflux = csr_matrix(cflux)

        cell_fluxs.append(cflux)

    # Stack all cells
    cell_fluxs = vstack(cell_fluxs) if len(cell_fluxs) > 1 else cell_fluxs[0]
    cell_fluxs.data = np.nan_to_num(cell_fluxs.data)
    pbar.update()

    # todo: Slow step, try algorithm="randomized" may be faster
    pbar.set_description(emoji.emojize("Reducing"))
    n_components = min(n_genes - 1, 10)
    pca_model = TruncatedSVD(
        n_components=n_components, algorithm="randomized", random_state=random_state
    ).fit(cell_fluxs)
    flux_embed = pca_model.transform(cell_fluxs)
    variance_ratio = pca_model.explained_variance_ratio_

    # For color visualization of flux embeddings
    flux_color = vec2color(flux_embed, fmt="hex", vmin=0.1, vmax=0.9)
    pbar.update()
    pbar.set_description(emoji.emojize("Saving"))

    flux_df = pd.DataFrame(cell_fluxs.todense().tolist(), columns=gene_names)
    flux_embed_df = pd.DataFrame(
        flux_embed.tolist(),
        columns=[f"flux_embed_{i}" for i in range(len(flux_embed.tolist()[0]))],
    )
    raster_points = pd.concat(
        [raster_points, flux_df, flux_embed_df],
        axis=1,
        join="outer",
        ignore_index=False,
    )

    raster_points["flux_color"] = flux_color
    flux_df = raster_points.drop(columns=["x", "y", instance_key])
    set_points_metadata(sdata, points_key=f"{instance_key}_raster", metadata=flux_df)

    sdata.table.uns["flux_variance_ratio"] = variance_ratio
    sdata.table.uns["flux_genes"] = gene_names  # gene names

    pbar.set_description(emoji.emojize("Done. :bento_box:"))
    pbar.update()
    pbar.close()


def vec2color(
    vec: np.ndarray,
    alpha_vec: Optional[np.ndarray] = None,
    fmt: Literal[
        "rgb",
        "hex",
    ] = "hex",
    vmin: float = 0,
    vmax: float = 1,
):
    """Convert vector to color."""

    # Grab the first 3 channels
    color = vec[:, :3]
    color = quantile_transform(color[:,:3])
    color = minmax_scale(color, feature_range=(vmin, vmax))

    # If vec has fewer than 3 channels, fill empty channels with 0
    if color.shape[1] < 3:
        color = np.pad(color, ((0, 0), (0, 3 - color.shape[1])), constant_values=0)

    
    # Add alpha channel
    if alpha_vec is not None:
        alpha = alpha_vec.reshape(-1, 1)
        # alpha = quantile_transform(alpha)
        alpha = alpha / alpha.max()
        color = np.c_[color, alpha]

    if fmt == "rgb":
        pass
    elif fmt == "hex":
        color = np.apply_along_axis(mpl.colors.to_hex, 1, color, keep_alpha=True)
    return color


def fluxmap(
    sdata: SpatialData,
    points_key: str = "transcripts",
    instance_key: str = "cell_boundaries",
    n_clusters: Union[Iterable[int], int] = range(2, 9),
    num_iterations: int = 1000,
    train_size: float = 0.2,
    res: float = 0.1,
    random_state: int = 11,
    plot_error: bool = True,
):
    """Cluster flux embeddings using self-organizing maps (SOMs) and vectorize clusters as Polygon shapes.

    Parameters
    ----------
    sdata : SpatialData
        Spatial formatted SpatialData object.
    points_key : str
        key for points element that holds transcript coordinates
    instance_key : str
        Key for cell_boundaries instances
    n_clusters : int or list
        Number of clusters to use. If list, will pick best number of clusters
        using the elbow heuristic evaluated on the quantization error.
    num_iterations : int
        Number of iterations to use for SOM training.
    train_size : float
        Fraction of cells to use for SOM training. Default 0.2.
    res : float
        Resolution used for rendering embedding. Default 0.05.
    random_state : int
        Random state to use for SOM training. Default 11.
    plot_error : bool
        Whether to plot quantization error. Default True.

    Returns
    -------
    sdata : SpatialData
        .points["points"] : DataFrame
            Adds "fluxmap" column denoting cluster membership.
        .shapes["fluxmap#_shape"] : GeoSeries
            Adds "fluxmap#_shape" columns for each cluster rendered as (Multi)Polygon shapes.
    """

    raster_points = get_points(
        sdata, points_key=f"{instance_key}_raster", astype="pandas", sync=True
    )

    # Check if flux embedding has been computed
    if "flux_embed_0" not in raster_points.columns:
        raise ValueError(
            "Flux embedding has not been computed. Run `bento.tl.flux()` first."
        )

    flux_embed = raster_points.filter(like="flux_embed_")
    sorted_column_names = sorted(
        flux_embed.columns.tolist(), key=lambda x: int(x.split("_")[-1])
    )
    flux_embed = flux_embed[sorted_column_names].to_numpy()

    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    if isinstance(n_clusters, range):
        n_clusters = list(n_clusters)

    # Subsample flux embeddings for faster training
    if train_size > 1:
        raise ValueError("train_size must be less than 1.")
    if train_size == 1:
        flux_train = flux_embed
    if train_size < 1:
        flux_train = resample(
            flux_embed,
            n_samples=int(train_size * flux_embed.shape[0]),
            random_state=random_state,
        )

    # Perform SOM clustering over n_clusters range and pick best number of clusters using elbow heuristic
    pbar = tqdm(total=4)
    pbar.set_description(emoji.emojize(f"Optimizing # of clusters"))
    som_models = {}
    quantization_errors = []
    for k in tqdm(n_clusters, leave=False):
        som = MiniSom(1, k, flux_train.shape[1], random_seed=random_state)
        som.random_weights_init(flux_train)
        som.train(flux_train, num_iterations, random_order=False, verbose=False)
        som_models[k] = som
        quantization_errors.append(som.quantization_error(flux_embed))

    # Use kneed to find elbow
    if len(n_clusters) > 1:
        kl = KneeLocator(
            n_clusters, quantization_errors, curve="convex", direction="decreasing"
        )
        best_k = kl.elbow

        if plot_error:
            kl.plot_knee()
            plt.show()

        if best_k is None:
            print("No elbow found. Rerun with a fixed k or a different range.")
            return

    else:
        best_k = n_clusters[0]
    pbar.update()

    # Use best k to assign each sample to a cluster
    pbar.set_description(f"Assigning to {best_k} clusters")
    som = som_models[best_k]
    winner_coordinates = np.array([som.winner(x) for x in flux_embed]).T

    # Indices start at 0, so add 1
    qnt_index = np.ravel_multi_index(winner_coordinates, (1, best_k)) + 1
    raster_points["fluxmap"] = qnt_index
    set_points_metadata(
        sdata,
        points_key=f"{instance_key}_raster",
        metadata=list(qnt_index),
        column_names="fluxmap",
    )

    pbar.update()

    # Vectorize polygons in each cell
    pbar.set_description(emoji.emojize("Vectorizing domains"))
    cells = raster_points[instance_key].unique().tolist()

    # Cast to int
    raster_points[["x", "y", "fluxmap"]] = raster_points[["x", "y", "fluxmap"]].astype(
        int
    )

    rpoints_grouped = raster_points.groupby(instance_key)
    fluxmap_df = dict()
    for cell in tqdm(cells, leave=False):
        rpoints = rpoints_grouped.get_group(cell)

        # Fill in image at each point xy with fluxmap value by casting to dense matrix
        image = (
            csr_matrix(
                (
                    rpoints["fluxmap"],
                    (
                        (rpoints["y"] * res).astype(int),
                        (rpoints["x"] * res).astype(int),
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
            columns=["fluxmap"],
        )

        # Remove background shape
        shapes["fluxmap"] = shapes["fluxmap"].astype(int)
        shapes = shapes[shapes["fluxmap"] != 0]

        # Group same fields as MultiPolygons
        shapes = shapes.dissolve("fluxmap")["geometry"]
        fluxmap_df[cell] = shapes

    fluxmap_df = pd.DataFrame.from_dict(fluxmap_df).T
    fluxmap_df.columns = "fluxmap" + fluxmap_df.columns.astype(str) + "_boundaries"

    # Upscale to match original resolution
    fluxmap_df = fluxmap_df.apply(
        lambda col: gpd.GeoSeries(col).scale(
            xfact=1 / res, yfact=1 / res, origin=(0, 0)
        )
    )
    pbar.update()

    pbar.set_description("Saving")

    old_shapes = [k for k in sdata.shapes.keys() if k.startswith("fluxmap")]
    for key in old_shapes:
        del sdata.shapes[key]

    transform = sdata.shapes[instance_key].attrs
    fluxmap_df = fluxmap_df.reindex(sdata.table.obs_names).where(
        fluxmap_df.notna(), other=Polygon()
    )
    for fluxmap in fluxmap_df.columns:
        sdata.shapes[fluxmap] = ShapesModel.parse(
            gpd.GeoDataFrame(geometry=fluxmap_df[fluxmap])
        )
        sdata.shapes[fluxmap].attrs = transform

    old_cols = sdata.points[points_key].columns[
        sdata.points[points_key].columns.str.startswith("fluxmap")
    ]
    sdata.points[points_key] = sdata.points[points_key].drop(old_cols, axis=1)

    # TODO SLOW
    sjoin_points(
        sdata=sdata, shape_keys=fluxmap_df.columns.tolist(), points_key=points_key
    )
    pbar.update()
    pbar.set_description("Done")
    pbar.close()
