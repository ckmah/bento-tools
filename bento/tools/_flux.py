from typing import Iterable, Literal, Optional, Union

import emoji
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from anndata import AnnData
from kneed import KneeLocator
from minisom import MiniSom
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from sklearn.preprocessing import StandardScaler, minmax_scale, quantile_transform
from sklearn.utils import resample
from tqdm.auto import tqdm
from rich.progress import Progress

from bento._settings import settings
from bento._utils import register_points, track
from bento.geometry import get_points, sindex_points
from bento.tools._neighborhoods import _count_neighbors
from bento.tools._shape_features import analyze_shapes


@track
@register_points("cell_raster", ["flux", "flux_embed", "flux_counts"])
def flux(
    data: AnnData,
    method: Literal["knn", "radius"] = "radius",
    n_neighbors: Optional[int] = None,
    radius: Optional[int] = 0.5,
    n_components = 100,
    res: float = 1,
    train_size: float = 1,
    use_highly_variable: bool = False,
    random_state: int = 11,
    copy: bool = False,
):
    """
    RNAflux: Embedding each pixel as normalized local composition normalized by cell composition.
    For k-nearest neighborhoods or "knn", method, specify n_neighbors. For radius neighborhoods, specify radius.
    The default method is "radius" with radius=50. RNAflux requires a minimum of 4 genes per cell to compute all embeddings properly.

    Parameters
    ----------
    data : AnnData
        Spatial formatted AnnData object.
    method: str
        Method to use for local neighborhood. Either 'knn' or 'radius'.
    n_neighbors : int
        Number of neighbors to use for local neighborhood.
    radius : float
        Radius to use for local neighborhood. Uses cell radius / 2 if None.
    res : float
        Resolution to use for rendering embedding. Default 0.05 samples at 5% original resolution (5 units between pixels)
    copy : bool
        Whether to return a copy the AnnData object. Default False.

    Returns
    -------
    adata : AnnData
        .uns["flux"] : scipy.csr_matrix
            [pixels x genes] sparse matrix of normalized local composition.
        .uns["flux_embed"] : np.ndarray
            [pixels x components] array of embedded flux values.
        .uns["flux_color"] : np.ndarray
            [pixels x 3] array of RGB values for visualization.
        .uns["flux_genes"] : list
            List of genes used for embedding.
        .uns["flux_variance_ratio"] : np.ndarray
            [components] array of explained variance ratio for each component.
    """

    adata = data.copy() if copy else data

    settings.log.start("Running flux().")

    # Get points
    adata.uns["points"] = get_points(adata).sort_values("cell")
    points = get_points(adata)[["cell", "gene", "x", "y"]]

    # Only use highly variable genes
    if use_highly_variable:
        high_var = adata.var["highly_variable"]
        high_var = high_var[high_var].index.tolist()
        points = points[points["gene"].isin(high_var)]

    # Extract gene names and codes
    points["gene"] = points["gene"].cat.remove_unused_categories()
    gene_names = points["gene"].cat.categories.tolist()
    n_genes = len(gene_names)


    # By default use 50% of average cell radius
    if method == "radius":
        analyze_shapes(adata, "cell_shape", "radius", progress=False, recompute=True)

        # Default radius = 50% of average cell radius
        if radius is None:
            radius = adata.obs["cell_radius"].mean() / 2
        # If radius is a fraction, use that fraction of average cell radius
        elif radius <= 1:
            radius = radius * (adata.obs["cell_radius"].mean())
        # If radius is an integer, use that as the radius

        settings.log.info(f"radius = {radius}")

    # embeds points on a uniform grid
    settings.log.step("Embedding")
    step = 1 / res
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


    points_grouped = points.groupby("cell")
    rpoints_grouped = raster_points.groupby("cell")
    cells = list(points_grouped.groups.keys())
    # Compute cell composition for each cell
    cell_composition = adata[cells, gene_names].X.toarray()
    cell_composition = cell_composition / (cell_composition.sum(axis=1).reshape(-1, 1))
    cell_composition = np.nan_to_num(cell_composition)

    # Embed each cell neighborhood independently
    cell_fluxs = []
    rpoint_counts = []
    for i, cell in tqdm(enumerate(cells), total=len(cells)):
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

        # Count points in each neighborhood
        total_count = gene_count.sum(axis=1)
        
        # Get max neighborhood size
        max_count = total_count.max()

        # Compute gene composition of neighborhood
        flux_composition = gene_count / total_count.reshape(-1, 1)
        
        # Formula: distance between cell composition and neighborhood composition * relative neighborhood size
        cflux = (flux_composition - cell_composition[i])
        # * (total_count.reshape(-1, 1) / max_count)

        # TODO do i need this
        cflux = StandardScaler(with_mean=False).fit_transform(cflux)

        # Convert back to sparse matrix
        cflux = csr_matrix(cflux)

        cell_fluxs.append(cflux)
        rpoint_counts.append(total_count)

    # Stack all cells
    cell_fluxs = vstack(cell_fluxs) if len(cell_fluxs) > 1 else cell_fluxs[0]
    cell_fluxs.data = np.nan_to_num(cell_fluxs.data)
    rpoints_counts = np.concatenate(rpoint_counts)

    settings.log.step("SVD")
    train_n_samples = max(int(train_size * cell_fluxs.shape[0]), 10000)
    train_n_samples = min(train_n_samples, cell_fluxs.shape[0])
    train_x = resample(
        cell_fluxs,
        replace=False,
        n_samples=train_n_samples,
        random_state=random_state,
    )
    settings.log.step(f"Train size: {train_n_samples}")

    svd_model = IncrementalPCA(
        batch_size=None, n_components=n_components
    ).fit(train_x)
    flux_embed = svd_model.transform(cell_fluxs)
    flux_sv = svd_model.components_
    variance_ratio = svd_model.explained_variance_ratio_

    # Use the elbow method to determine the number of components to keep
    kl = KneeLocator(
        range(len(variance_ratio)), variance_ratio, curve="convex", direction="decreasing"
    )
    if kl.elbow is not None:
        n_components = kl.elbow
    else:
        n_components = len(variance_ratio)

    settings.log.step("Saving results")
    adata.uns["flux"] = cell_fluxs  # sparse gene embedding
    adata.uns["flux_genes"] = gene_names  # gene names
    adata.uns["flux_embed"] = flux_embed
    adata.uns["flux_sv"] = flux_sv
    adata.uns["flux_n_components"] = n_components
    adata.uns["flux_counts"] = rpoints_counts
    adata.uns["flux_variance_ratio"] = variance_ratio

    settings.log.end("Done.")

    return adata if copy else None


@track
def fluxmap(
    data: AnnData,
    n_clusters: Union[Iterable[int], int] = range(2, 9),
    n_components: Optional[int] = None,
    num_iterations: int = 1000,
    min_points: int = 50,
    train_size: float = 0.2,
    res: float = 0.1,
    random_state: int = 11,
    plot_error: bool = True,
    copy: bool = False,
):
    """Cluster flux embeddings using self-organizing maps (SOMs) and vectorize clusters as Polygon shapes.

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
    res : float
        Resolution used for rendering embedding. Default 0.05.
    random_state : int
        Random state to use for SOM training. Default 11.
    plot_error : bool
        Whether to plot quantization error. Default True.
    copy : bool
        Whether to return a copy the AnnData object. Default False.

    Returns
    -------
    adata : AnnData
        .uns["cell_raster"] : DataFrame
            Adds "fluxmap" column denoting cluster membership.
        .uns["points"] : DataFrame
            Adds "fluxmap#" columns for each cluster.
        .obs : GeoSeries
            Adds "fluxmap#_shape" columns for each cluster rendered as (Multi)Polygon shapes.
    """
    adata = data.copy() if copy else data

    # Check if flux embedding has been computed
    if "flux_embed" not in adata.uns:
        raise ValueError(
            "Flux embedding has not been computed. Run `bento.tl.flux()` first."
        )

    if n_components is None:
        n_components = adata.uns["flux_n_components"]
    flux_embed = adata.uns["flux_embed"][:, :n_components]
    raster_points = adata.uns["cell_raster"]
    flux_counts = adata.uns["flux_counts"]

    # Exclude points with low counts
    valid_points = flux_counts > min_points

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
            flux_embed[valid_points],
            n_samples=int(train_size * flux_embed[valid_points].shape[0]),
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
    qnt_index[~valid_points] = 0
    raster_points["fluxmap"] = qnt_index
    adata.uns["cell_raster"] = raster_points.copy()

    pbar.update()

    # Vectorize polygons in each cell
    pbar.set_description(emoji.emojize("Vectorizing domains"))
    cells = raster_points["cell"].unique().tolist()
    # Scale down to render resolution
    # raster_points[["x", "y"]] = raster_points[["x", "y"]] * res

    # Cast to int
    raster_points[["x", "y", "fluxmap"]] = raster_points[["x", "y", "fluxmap"]].astype(
        int
    )

    rpoints_grouped = raster_points.groupby("cell")
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
    fluxmap_df.columns = "fluxmap" + fluxmap_df.columns.astype(str) + "_shape"

    # Upscale to match original resolution
    fluxmap_df = fluxmap_df.apply(
        lambda col: gpd.GeoSeries(col).scale(
            xfact=1 / res, yfact=1 / res, origin=(0, 0)
        )
    )
    pbar.update()

    pbar.set_description("Saving")
    old_cols = adata.obs.columns[adata.obs.columns.str.startswith("fluxmap")]
    adata.obs = adata.obs.drop(old_cols, axis=1, errors="ignore")

    adata.obs[fluxmap_df.columns] = fluxmap_df.reindex(adata.obs_names)

    old_cols = adata.uns["points"].columns[
        adata.uns["points"].columns.str.startswith("fluxmap")
    ]
    adata.uns["points"] = adata.uns["points"].drop(old_cols, axis=1)

    # TODO SLOW
    sindex_points(adata, "points", fluxmap_df.columns.tolist())
    pbar.update()
    pbar.set_description("Done")
    pbar.close()

    return adata if copy else None
