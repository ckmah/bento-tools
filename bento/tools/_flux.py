from typing import Iterable, Literal, Optional, Union

import dask
import dask.delayed
import emoji
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import shapely
from kneed import KneeLocator
from minisom import MiniSom
from scipy.sparse import csr_matrix, vstack
from shapely import Polygon
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, minmax_scale, quantile_transform
from sklearn.utils import resample
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import ShapesModel
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback

from .._utils import (
    get_points,
    get_shape_metadata,
    set_points_metadata,
    get_points_metadata,
)
from ..io._index import _sjoin_points, _sjoin_shapes
from ..tools._neighborhoods import _count_neighbors
from ..tools._shape_features import analyze_shapes


def flux(
    sdata: SpatialData,
    points_key: str = "transcripts",
    instance_key: str = "cell_boundaries",
    feature_key: str = "feature_name",
    method: Literal["knn", "radius"] = "radius",
    n_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
    res: Optional[float] = 1,
    train_size: Optional[float] = 1,
    random_state: int = 11,
    recompute: bool = False,
    num_workers=1,
):
    """
    Compute RNAflux embeddings of each pixel as local composition normalized by cell composition.
    For k-nearest neighborhoods or "knn", method, specify n_neighbors. For radius neighborhoods, specify radius.
    The default method is "radius" with radius = 1/4 of cell radius. RNAflux requires a minimum of 4 genes per cell to compute all embeddings properly.

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
        Fraction of mean cell radius to use for local neighborhood.
    res : float
        Resolution to use for rendering embedding.

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

    if method == "radius":
        analyze_shapes(sdata, instance_key, "radius", progress=False, recompute=True)
        mean_radius = (
            get_shape_metadata(
                sdata, shape_key=instance_key, metadata_keys=f"{instance_key}_radius"
            )
            .mean()
            .values[0]
        )
        # Default radius = 33% of average cell radius
        if radius is None:
            radius = mean_radius / 3
        # If radius is a fraction, use that fraction of average cell radius
        elif radius <= 1:
            radius = radius * mean_radius
        # If radius is an integer, use that as the radius

    # Grab molecules
    points = get_points(sdata, points_key=points_key, astype="pandas", sync=True)
    points = points[[instance_key, feature_key, "x", "y"]]

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
        recompute=recompute,
        feature_kws=dict(raster={"step": step}),
    )

    # Grab raster points
    raster_points = get_points(
        sdata, points_key=f"{instance_key}_raster", astype="pandas", sync=False
    )

    # Extract gene names and codes
    gene_names = points[feature_key].cat.categories.tolist()
    n_genes = len(gene_names)

    points_grouped = points.groupby(instance_key)
    rpoints_grouped = raster_points.groupby(instance_key)
    cells = list(points_grouped.groups.keys())
    cells.sort()

    # points_grouped = dask.delayed(points_grouped)
    # rpoints_grouped = dask.delayed(rpoints_grouped)

    cell_composition = sdata.table[cells, gene_names].X.toarray()

    # Compute cell composition
    cell_composition = cell_composition / (cell_composition.sum(axis=1).reshape(-1, 1))
    cell_composition[np.isnan(cell_composition)] = 0

    # Define a function that contains the operations to be performed in the for loop
    def process_cell(bag):
        cpoints, rpoints, n_genes, method, n_neighbors, radius, cell_composition = bag
        rpoint_index = rpoints.index.tolist()
        if method == "knn":
            gene_count = _count_neighbors(
                cpoints,
                n_genes,
                rpoints,
                n_neighbors=n_neighbors,
                agg=None,
            )
        elif method == "radius":
            gene_count = _count_neighbors(
                cpoints,
                n_genes,
                rpoints,
                radius=radius,
                agg=None,
            )
        gene_count = gene_count.toarray()

        # Count points in each neighborhood
        total_count = gene_count.sum(axis=1)

        # embedding: distance neighborhood composition and cell composition
        # Compute composition of neighborhood
        flux_composition = gene_count / (gene_count.sum(axis=1).reshape(-1, 1))
        cflux = flux_composition - cell_composition
        cflux = StandardScaler(with_mean=False).fit_transform(cflux)

        # Convert back to sparse matrix
        cflux = csr_matrix(cflux)

        return cflux, total_count, rpoint_index

    # Use dask.delayed in the for loop
    import dask.bag as db

    # Create a sequence of tuples containing the arguments for the process_cell function
    args = [
        (
            points_grouped.get_group(cell),
            rpoints_grouped.get_group(cell),
            n_genes,
            method,
            n_neighbors,
            radius,
            cell_composition[i],
        )
        for i, cell in enumerate(cells)
    ]

    # Create a Dask Bag from the sequence
    bags = db.from_sequence(args)

    # Use the map method of the Dask Bag to apply the process_cell function to each tuple in the sequence
    results = bags.map(process_cell)

    # Use dask.compute to execute the operations in parallel
    with TqdmCallback(desc="Batches"), dask.config.set(num_workers=num_workers):
        results = results.compute()

    # Stack all cells
    # cell_fluxs = vstack(cell_fluxs) if len(cell_fluxs) > 1 else cell_fluxs[0]
    cell_fluxs = vstack([cflux for cflux, _, _ in results])
    cell_fluxs.data = np.nan_to_num(cell_fluxs.data, copy=False)
    rpoints_counts = np.hstack([counts for _, counts, _ in results])
    rpoint_index = np.hstack([index for _, _, index in results])
    # rpoints_counts = np.concatenate(rpoint_counts)
    pbar.update()

    # todo: Slow step, try algorithm="randomized" may be faster
    pbar.set_description(emoji.emojize("Reducing"))
    n_components = min(n_genes, 10)

    train_n = int(train_size * len(cells))
    train_x = resample(
        cell_fluxs,
        replace=False,
        n_samples=train_n,
        random_state=random_state,
    )
    model = TruncatedSVD(
        n_components=n_components, algorithm="randomized", random_state=random_state
    ).fit(train_x)
    flux_embed = model.transform(cell_fluxs)
    variance_ratio = model.explained_variance_ratio_

    pbar.update()
    pbar.set_description(emoji.emojize("Saving"))

    embed_names = [f"flux_embed_{i}" for i in range(flux_embed.shape[1])]
    flux_color = vec2color(flux_embed, alpha_vec=rpoints_counts)

    # Save flux embeddings and colors after reindexing to raster points
    metadata = pd.DataFrame.sparse.from_spmatrix(
        cell_fluxs, index=rpoint_index, columns=gene_names
    )
    metadata[embed_names] = flux_embed
    metadata["flux_color"] = flux_color
    metadata["flux_counts"] = rpoints_counts

    # Compute index order once and apply to all
    if not metadata.index.equals(raster_points.index):
        _, indexer = metadata.index.reindex(
            raster_points.index.astype(metadata.index.dtype)
        )
        metadata = metadata.iloc[indexer]

    set_points_metadata(
        sdata,
        points_key=f"{instance_key}_raster",
        metadata=metadata,
        columns=metadata.columns,
    )

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
    color = quantile_transform(color[:, :3])
    color = minmax_scale(color, feature_range=(vmin, vmax))

    # If vec has fewer than 3 channels, fill empty channels with 0
    if color.shape[1] < 3:
        color = np.pad(color, ((0, 0), (0, 3 - color.shape[1])), constant_values=0)

    # Replace NaNs with 0
    color[np.isnan(color)] = 0

    # Add alpha channel
    if alpha_vec is not None:
        alpha = alpha_vec.reshape(-1, 1)
        # alpha = quantile_transform(alpha)
        alpha = alpha / alpha.max()
        alpha[np.isnan(alpha)] = 0
        color = np.c_[color, alpha]

    if fmt == "rgb":
        pass
    elif fmt == "hex":
        color = np.apply_along_axis(mpl.colors.to_hex, 1, color, keep_alpha=True)

    ["#00000000" if c is np.nan else c for c in color]

    return color


def fluxmap(
    sdata: SpatialData,
    points_key: str = "transcripts",
    instance_key: str = "cell_boundaries",
    n_clusters: Union[Iterable[int], int] = range(2, 9),
    num_iterations: int = 1000,
    min_count: int = 50,
    train_size: float = 1,
    res: float = 1,
    random_state: int = 11,
    plot_error: bool = False,
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
        .shapes["fluxmap#"] : GeoSeries
            Adds "fluxmap#" columns for each cluster rendered as (Multi)Polygon shapes.
    """

    raster_points = get_points(
        sdata, points_key=f"{instance_key}_raster", astype="pandas", sync=False
    )
    rpoints_index = raster_points.index

    # Check if flux embedding has been computed
    if "flux_embed_0" not in raster_points.columns:
        raise ValueError(
            "Flux embedding has not been computed. Run `bento.tl.flux()` first."
        )

    flux_embed = raster_points.filter(like="flux_embed_")

    # Keep only points with minimum neighborhood count
    flux_counts = raster_points["flux_counts"]
    valid_points = flux_counts >= min_count
    flux_embed = flux_embed[valid_points]
    embed_index = flux_embed.index

    # Sort columns
    flux_embed = flux_embed[sorted(flux_embed.columns.tolist())].to_numpy()

    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    if isinstance(n_clusters, range):
        n_clusters = list(n_clusters)

    # Subsample flux embeddings for faster training
    if train_size > 1:
        raise ValueError("train_size must be equal to or less than 1.")
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
    pbar.set_description(emoji.emojize("Optimizing # of clusters"))
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

    # Indices start at 0, so add 1; we will treat 0 as background
    qnt_index = np.ravel_multi_index(winner_coordinates, (1, best_k)) + 1
    qnt_index = pd.Series(qnt_index, index=embed_index).reindex(
        rpoints_index, fill_value=0
    )
    raster_points["fluxmap"] = qnt_index
    set_points_metadata(
        sdata,
        points_key=f"{instance_key}_raster",
        metadata=list(qnt_index),
        columns="fluxmap",
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

        # Translate so all points are positive and save offsets
        x_offset = rpoints["x"].min()
        y_offset = rpoints["y"].min()
        rpoints["x"] = rpoints["x"] - x_offset
        rpoints["y"] = rpoints["y"] - y_offset

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

        # Add back offsets
        shapes["geometry"] = shapes["geometry"].translate(x_offset, y_offset)

        # Remove background shape
        shapes["fluxmap"] = shapes["fluxmap"].astype(int)
        shapes = shapes[shapes["fluxmap"] != 0]

        # Group same fields as MultiPolygons
        shapes = shapes.dissolve("fluxmap")["geometry"]
        fluxmap_df[cell] = shapes

    fluxmap_df = pd.DataFrame.from_dict(fluxmap_df).T
    fluxmap_df.columns = "fluxmap" + fluxmap_df.columns.astype(str)

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

    sd_attrs = sdata.shapes[instance_key].attrs
    fluxmap_df = fluxmap_df.reindex(sdata.table.obs_names).where(
        fluxmap_df.notna(), other=Polygon()
    )
    fluxmap_names = fluxmap_df.columns.tolist()
    for fluxmap in fluxmap_names:
        sdata.shapes[fluxmap] = ShapesModel.parse(
            gpd.GeoDataFrame(geometry=fluxmap_df[fluxmap])
        )
        sdata.shapes[fluxmap].attrs = sd_attrs

    old_cols = sdata.points[points_key].columns[
        sdata.points[points_key].columns.str.startswith("fluxmap")
    ]
    sdata.points[points_key] = sdata.points[points_key].drop(old_cols, axis=1)

    _sjoin_points(sdata=sdata, shape_keys=fluxmap_names, points_key=points_key)
    _sjoin_shapes(sdata=sdata, instance_key=instance_key, shape_keys=fluxmap_names)

    pbar.update()
    pbar.set_description("Done")
    pbar.close()
