import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os

import geopandas
import matplotlib.path as mplPath
import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision
from astropy.stats import RipleysKEstimator
from joblib import Parallel, delayed
from rasterio import features
from scipy.spatial import distance, distance_matrix
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from ..io import get_points


def coloc_sim(data, radius=3, min_count=5, n_cores=1, copy=False):
    """Calculate pairwise gene colocalization similarity using a KNN approach.

    Parameters
    ----------
    adata : AnnData
        Anndata formatted spatial data.
    outer_radius : int, optional
        Number of pixels to search for neighbors, by default 3
    Returns
    -------
    adata : AnnData
        .uns['coloc_sim']: Pairwise gene colocalization similarity within each cell.
    """
    adata = data.copy() if copy else data

    # Filter points and counts by min_count
    counts = adata.to_df()

    # Helper function to apply per cell
    def cell_coloc_sim(p, g_density, name):

        # Get xy coordinates
        xy = p[["x", "y"]].values

        # Get neighbors within fixed outer_radius for every point
        nn = NearestNeighbors(radius=radius).fit(xy)
        distances, point_index = nn.radius_neighbors(xy, return_distance=True)

        # Enumerate point-wise gene labels
        gene_index = p["gene"].reset_index(drop=True)

        # Convert to adjacency list of points, no double counting
        neighbor_pairs = []
        for g1, neighbors, n_dists in zip(gene_index.values, point_index, distances):
            for g2, d in zip(neighbors, n_dists):
                neighbor_pairs.append([g1, g2, d])

        # Calculate pair-wise gene similarity
        neighbor_pairs = pd.DataFrame(neighbor_pairs, columns=["g1", "g2", "p_dist"])

        # Keep minimum distance to g2 point
        neighbor_pairs = neighbor_pairs.groupby(["g1", "g2"]).agg("min").reset_index()
        neighbor_pairs.columns = ["g1", "g2", "point_dist"]

        # Map to gene index
        neighbor_pairs["g2"] = neighbor_pairs["g2"].map(gene_index)

        # Count number of points within distance of increasing radius
        r_step = 0.5
        expected_counts = [
            lambda dists: (dists <= r).sum()
            for r in np.arange(r_step, radius + r_step, r_step)
        ]
        metrics = (
            neighbor_pairs.groupby(["g1", "g2"])
            .agg({"point_dist": expected_counts})
            .reset_index()
        )

        metrics["g1"] = metrics["g1"].map(adata.uns["point_gene_index"])
        metrics["g2"] = metrics["g2"].map(adata.uns["point_gene_index"])

        # Colocalization metric: max of L_ij(r) for r <= radius
        g2_density = g_density.loc[metrics["g2"].tolist()].values
        metrics["coloc_sim"] = (
            (metrics["point_dist"].divide(g2_density * np.pi, axis=0))
            .pow(0.5)
            .max(axis=1)
        )
        metrics["cell"] = name

        # Ignore self colocalization
        metrics = metrics.loc[metrics["g1"] != metrics["g2"]]

        return metrics[["cell", "g1", "g2", "coloc_sim"]]

    # Only keep genes >= min_count in each cell
    gene_densities = []
    counts.apply(lambda row: gene_densities.append(row[row >= min_count]), axis=1)
    # Calculate point density per gene per cell
    gene_densities /= adata.obs["cell_area"]
    gene_densities = gene_densities.values

    cell_metrics = Parallel(n_jobs=n_cores)(
        delayed(cell_coloc_sim)(
            get_points(adata, cells=g_density.name, genes=g_density.index.tolist()),
            g_density,
            g_density.name,
        )
        for g_density in tqdm(gene_densities)
    )

    cell_metrics = pd.concat(cell_metrics)

    # Save coloc similarity
    adata.uns["coloc_sim"] = cell_metrics

    return adata if copy else None


# TODO need physical unit size of coordinate system to standardize rendering resolution
def rasterize_cells(
    data,
    imgdir,
    label_layer=None,
    scale_factor=15,
    out_dim=64,
    n_cores=1,
    overwrite=True,
    copy=False,
):
    """Rasterize points and cell masks to grayscale image. Writes directly to file.

    Parameters
    ----------
    data : AnnData
        AnnData formatted spatial data.
    imgdir : str
        Directory where images will be stored.
    """
    adata = data.copy() if copy else data

    os.makedirs(f"{imgdir}", exist_ok=True)

    def write_img(s, n, p, cell_name):

        # Get bounds and size of cell in raw coordinate space
        bounds = s.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        # Define top left corner for centering/scaling transform
        west = bounds[0] + width / 2 - (out_dim / 2 * scale_factor)
        north = bounds[3] - height / 2 + (out_dim / 2 * scale_factor)

        # Define transform
        tf_origin = rasterio.transform.from_origin(
            west, north, scale_factor, scale_factor
        )

        # Rasterize cell
        base_raster = features.rasterize(
            [s],
            fill=0,
            default_value=20,
            out_shape=(out_dim, out_dim),
            transform=tf_origin,
        )

        # Rasterize nucleus
        if n is not None:
            features.rasterize(
                [n], default_value=40, transform=tf_origin, out=base_raster
            )

        warnings.filterwarnings(
            action="ignore", category=rasterio.errors.NotGeoreferencedWarning
        )

        # Rasterize and write points
        genes = p["gene"].unique().tolist()

        if label_layer:
            labels = dict(
                zip(genes, list(adata[cell_name, genes].layers[label_layer].flatten()))
            )
        else:
            labels = dict(zip(genes, ["foo"] * len(genes)))

        p = geopandas.GeoDataFrame(p, geometry=geopandas.points_from_xy(p["x"], p["y"]))

        for gene in genes:
            label = labels[gene]
            gene_name = adata.uns["point_gene_index"][gene]

            os.makedirs(f"{imgdir}/{label}", exist_ok=True)

            # TODO implement overwrite param
            if not overwrite and os.path.exists(
                f"{imgdir}/{label}/{cell_name}_{gene_name}.tif"
            ):
                return

            cg_points = p.loc[p["gene"] == gene]

            gene_raster = base_raster.copy()

            # Set base as 40
            gene_raster = features.rasterize(
                shapes=cg_points.geometry,
                default_value=40,
                transform=tf_origin,
                out=gene_raster,
            )

            # Plus 20 per point
            features.rasterize(
                shapes=cg_points.geometry,
                default_value=20,
                transform=tf_origin,
                merge_alg=rasterio.enums.MergeAlg("ADD"),
                out=gene_raster,
            )

            # Convert to tensor
            gene_raster = torch.from_numpy(gene_raster.astype(np.float32) / 255)

            torchvision.utils.save_image(
                gene_raster, f"{imgdir}/{label}/{cell_name}_{gene_name}.tif"
            )

    # Parallelize points
    Parallel(n_jobs=n_cores)(
        delayed(write_img)(
            adata.obs.loc[cell_name, "cell_shape"],
            adata.obs.loc[cell_name, "nucleus_shape"],
            get_points(adata, cells=cell_name),
            cell_name,
        )
        for cell_name in tqdm(adata.obs_names.tolist())
    )

    # TODO write filepaths to adata

    return adata if copy else None


def prepare_features(data, features=[], copy=False):
    """Prepare features from raw data. This is the entry point to constructing uns.sample_data.

    Parameters
    ----------
    adata : AnnData
        AnnData formatted spatial data.
    features : list of str
        List of feature names to compute, by default empty list
    copy : bool
        Return copy of AnnData with new fields if True. Otherwise modifies in-place.
    Returns
    -------
    adata : AnnData
        Updated Anndata formatted spatial data. Unstructured data will have new entries, sample_index and sample_data.
    """
    adata = data.copy() if copy else data

    # Cast to list if single str
    if type(features) == str:
        features = [features]

    # Get cells
    cells = pd.Series(adata.obs["cell"].unique())
    if (cells == "-1").any():
        print("Extracellular points detected. These will be ignored.")
        cells = cells.loc[cells != "-1"]

    # Prepare features for each cell
    f = []
    for cell in tqdm(cells.tolist(), desc=f"Processing {len(cells)} cells"):
        cell_data = _prepare_cell_features(adata, features, cell)
        cell_data = pd.DataFrame(cell_data).T
        cell_data.columns = features
        cell_data.reset_index(inplace=True)
        cell_data["cell"] = cell
        f.append(cell_data)

    # Stack as single dataframe
    f = pd.concat(f, axis=0)
    f = f.set_index(["cell", "gene"])

    # TODO when to overwrite/update?
    # Save sample -> (cell, gene) mapping
    sample_index = f.index.to_frame(index=False)
    sample_index.columns = ["cell", "gene"]
    sample_index.index = sample_index.index.astype(str)
    adata.uns["sample_index"] = sample_index

    # Save obs mapping
    obs = adata.obs.merge(
        adata.uns["sample_index"].reset_index(), how="left", on=["cell", "gene"]
    )

    if "sample_id" in obs.keys():
        obs.drop("sample_id", axis=1, inplace=True)
    obs.rename({"index": "sample_id"}, axis=1, inplace=True)
    obs["sample_id"].fillna("-1", inplace=True)
    obs.index = obs.index.astype(str)
    adata.obs = obs

    # Save each feature as {feature : np.array} in adata.uns
    feature_dict = f.to_dict(orient="list")
    feature_dict = {key: np.array(values) for key, values in feature_dict.items()}

    # Save each feature to `sample_data` dictionary.
    # Each feature follows {sample_id: feature_value} format, where sample_id corresponds to uns['sample_index'].index.
    if "sample_data" not in adata.uns:
        adata.uns["sample_data"] = dict()
    adata.uns["sample_data"] = feature_dict

    return adata if copy else None


def _prepare_cell_features(data, features, cell, **kwargs):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    features : [type]
        [description]
    cell : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # Compute each feature independently
    computed_features = []
    for feature in features:
        # Lookup function
        fn = feature_set[feature]["function"]

        # All functions should take two args: AnnData and index for cell
        # Execute function that returns DataFrame [genes x 1+] float
        cell_data = data[data.obs["cell"] == cell].copy()
        computed_features.append(fn(cell_data, cell, **kwargs))

    return computed_features


def _calc_ripley_features(cell_data, cell):
    """
    Compute 5 features extracted from Ripley L-function.

    Returns
    -------
    dict : keyed feature values
        max_l : float
        max_gradient : float
        min_gradient : float
        monotony : float
        l_4 : float
    """

    cell_mask = cell_data.uns["masks"]["cell"].loc[cell, "geometry"]

    # L/4 where length of the cell L is max dist between 2 points on polygon defining cell border
    cell_mask_points = np.array(cell_mask.exterior.coords.xy).T
    l_4_dist = distance_matrix(cell_mask_points, cell_mask_points).max() / 4

    def _calc(gene_data, cell):
        # Compute ripley function for r=(1, cell diameter / 2)
        ripley_norm, radii = _ripley(
            gene_data.X, cell_mask, radii=np.linspace(1, l_4_dist * 2, 100)
        )

        # Max value of the L-function
        max_l = ripley_norm.max()

        # Max and min value of the gradient of L
        # Rolling number determines min points needed per gene
        ripley_smooth = pd.Series(ripley_norm).rolling(5).mean()
        ripley_smooth.dropna(inplace=True)
        # if len(ripley_smooth) < 2:
        #     print('Ripley:')
        #     print(ripley_norm)
        #     print(ripley_smooth)
        ripley_gradient = np.gradient(ripley_smooth)
        max_gradient = ripley_gradient.max()
        min_gradient = ripley_gradient.min()

        # Monotony of L-function in the interval
        l_corr = spearmanr(radii, ripley_norm)[0]

        # L-function at L/4 where length of the cell L is max dist between 2 points on polygon defining cell border
        l_4 = _ripley(gene_data.X, cell_mask, [l_4_dist])[0][0]

        return np.array([max_l, max_gradient, min_gradient, l_corr, l_4])

    return cell_data.obs.groupby("gene").apply(
        lambda obs: _calc(cell_data[obs.index, :], cell)
    )


def _calc_norm_distance_quantile_features(cell_data, cell):
    """Return 5th, 10th, 20th, and 50th quantiles for distance to cell and nucleus centroids.

    Parameters
    ----------
    data : AnnData
        [description]
    cell : [type]
        [description]

    Returns
    -------
    pd.Series
    """

    # Select cell mask
    cell_mask = cell_data.uns["masks"]["cell"].loc[cell, "geometry"]

    # Select nucleus mask
    nucleus_index = cell_data.uns["mask_index"]["nucleus"]
    n_index = nucleus_index[cell_data.uns["mask_index"]["nucleus"] == cell].index[0]
    nucleus_mask = cell_data.uns["masks"]["nucleus"].loc[n_index, "geometry"]

    def _calc(gene_data, cell):

        # Select point coordinates
        points = np.array(gene_data.X)

        # Calculate normalized distances from points to centroids
        cell_centroid_dist = _calc_norm_dist_to_centroid(points, cell_mask)
        nucleus_centroid_dist = _calc_norm_dist_to_centroid(points, nucleus_mask)

        # Calculate normalized distances from points to mask; distance = point to closest point on mask
        cell_mask_dist = _calc_norm_dist_to_mask(points, cell_mask)
        nucleus_mask_dist = _calc_norm_dist_to_mask(points, nucleus_mask)

        features_raw = [
            cell_centroid_dist,
            nucleus_centroid_dist,
            cell_mask_dist,
            nucleus_mask_dist,
        ]
        features_label_prefix = [
            "cell_centroid_distance",
            "nucleus_centroid_distance",
            "cell_mask_distance",
            "nucleus_mask_distance",
        ]

        # Get quantiles of each distance distribution
        features = []
        features_labels = []
        for feature, label in zip(features_raw, features_label_prefix):
            quantiles = [5, 10, 20, 50]
            features.extend(np.percentile(feature, quantiles))
            features_labels.extend([f"{label}_q{q}" for q in quantiles])

        return np.array(features)

    return cell_data.obs.groupby("gene").apply(
        lambda obs: _calc(cell_data[obs.index, :], cell)
    )


def _calc_norm_dist_to_centroid(points, mask):
    """Calculate normalized distance between every point in points to centroid of mask.

    Parameters
    ----------
    points : 2d array
        Array of 2D coordinates.
    mask : Polygon
        [description]
    """
    mask_points = np.array(mask.exterior.coords.xy).T
    centroid = np.array(mask.centroid).reshape(1, 2)

    expected_distance = abs(distance.cdist(mask_points, centroid)).mean()
    observed_distance = abs(distance.cdist(points, centroid))
    return observed_distance / expected_distance


def _calc_norm_dist_to_mask(points, mask):
    """Calculate normalized distance between every point in points to mask.

    Parameters
    ----------
    points : 2d array
        Array of 2D coordinates.
    mask : Polygon
        [description]
    """
    mask_points = np.array(mask.exterior.coords.xy).T
    centroid = np.array(mask.centroid).reshape(1, 2)

    expected_distance = abs(distance.cdist(mask_points, centroid)).mean()
    observed_distance = abs(distance.cdist(points, mask_points)).min(axis=0)
    return observed_distance / expected_distance


def _calc_morph_enrichment(cell_data, cell):
    # Select cell mask
    cell_mask = cell_data.uns["masks"]["cell"].loc[cell, "geometry"]

    # Morphological openings on cell mask
    # Mean cell radius size
    mean_cell_radius = distance.cdist(
        np.array(cell_mask.centroid).reshape(1, 2), np.array(cell_mask.exterior.xy).T
    ).mean()

    # Define proportion of radius to perform morph opening
    proportions = [0.1, 0.25]
    morph_masks = {}
    for p in proportions:
        # Define proportion of radius
        d = mean_cell_radius * p

        # Compute morph opening
        morph_masks[p] = cell_mask.buffer(distance=-d).buffer(distance=d)

    def _calc(gene_data, cell):

        enrichment = []

        # Create GeoDataFrame from points
        points = pd.DataFrame(gene_data.X, columns=["x", "y"])
        points = geopandas.GeoDataFrame(
            geometry=geopandas.points_from_xy(points.x, points.y)
        )

        # Count fraction of points in mask
        for p in proportions:
            n_points = len(geopandas.clip(points, morph_masks[p]))
            enrichment.append(np.float(n_points) / len(points))

        return np.array(enrichment)

    return cell_data.obs.groupby("gene").apply(
        lambda obs: _calc(cell_data[obs.index, :], cell)
    )


def _calc_nuclear_fraction(cell_data, cell):
    def _calc(gene_data, cell):
        nucleus_index = cell_data.uns["mask_index"]["nucleus"]
        n_index = nucleus_index[nucleus_index["cell"] == cell].index[0]
        nuclear_count = sum(gene_data.obs["nucleus"] == n_index)
        ratio = float(nuclear_count) / len(gene_data)
        return ratio

    return cell_data.obs.groupby("gene").apply(
        lambda obs: _calc(cell_data[obs.index, :], cell)
    )


def _calc_indexes(cell_data, cell):
    """Calculate polarization index, dispersion index, and peripheral index. Requires both cell and nucleus."""

    # Calculate of cell-dependent only values
    cell_mask = cell_data.uns["masks"]["cell"].loc[cell, "geometry"]
    cell_centroid = np.array(cell_mask.centroid).reshape(1, 2)
    grid_points = _poly2grid(cell_mask)
    cell_centroid_array = np.tile(cell_centroid, len(grid_points))
    cell_centroid_array = cell_centroid_array.reshape(-1, 2)
    radius_of_gyration = mean_squared_error(grid_points, cell_centroid_array)

    nucleus_index = cell_data.uns["mask_index"]["nucleus"]
    n_index = nucleus_index[cell_data.uns["mask_index"]["nucleus"] == cell].index[0]
    nucleus_centroid = np.array(
        cell_data.uns["masks"]["nucleus"].loc[n_index, "geometry"].centroid
    ).reshape(1, 2)

    # Calculate second moment of points with centroid as reference
    def _calc_second_moment(centroid, pts):
        """
        centroid : [1 x 2] float
        pts : [n x 2] float
        """
        radii = distance.cdist(centroid, pts)
        second_moment = np.sum(radii * radii / len(pts))
        return second_moment

    nuclear_moment = _calc_second_moment(nucleus_centroid, grid_points)

    def _calc(gene_data, cell):

        # Calculate distance between point and cell centroids
        points = np.array(gene_data.X)
        point_centroid = points.mean(axis=0).reshape(1, 2)
        offset = distance.cdist(point_centroid, cell_centroid)[0][0]

        # Calculate polarization index
        polarization_index = offset / radius_of_gyration

        # Calculate dispersion index; d-index = 2nd moment of spots, norm. by 2nd moment of uniform spots; moments relative to spot centroid
        dispersion_index = _calc_second_moment(
            point_centroid, points
        ) / _calc_second_moment(point_centroid, grid_points)

        # Calculate peripheral distribution index; d-index = 2nd moment of spots, norm. by 2nd moment of uniform spots; moments relative to nucleus centroid
        peripheral_distribution_index = (
            _calc_second_moment(nucleus_centroid, points) / nuclear_moment
        )

        return np.array(
            [polarization_index, dispersion_index, peripheral_distribution_index]
        )

    return cell_data.obs.groupby("gene").apply(
        lambda obs: _calc(cell_data[obs.index, :], cell)
    )


def _poly2grid(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    x, y = np.meshgrid(
        np.arange(minx, maxx, step=np.float(20)),
        np.arange(miny, maxy, step=np.float(20)),
    )
    x = x.flatten()
    y = y.flatten()
    xy = np.array([x, y]).T
    polygon_path = mplPath.Path(np.array(polygon.exterior.xy).T)
    polygon_cell_mask = polygon_path.contains_points(xy)
    xy = xy[polygon_cell_mask]

    return xy


def _ripley(points, mask, radii=None):
    """
    Calculate estimation for Ripley H function (0-centered L function).

    Parameters
    ----------
    points : 2d array
    mask : shapely.Polygon
    radii : float list

    Returns
    -------
    list
        ripley l function
    radii
    """
    estimator = RipleysKEstimator(
        area=mask.area,
        x_min=float(points[:, 0].min()),
        x_max=float(points[:, 0].max()),
        y_min=float(points[:, 1].min()),
        y_max=float(points[:, 1].max()),
    )

    # https://docs.astropy.org/en/stable/api/astropy.stats.RipleysKEstimator.html#astropy.stats.RipleysKEstimator
    # if radii is None:
    #     radii = np.linspace(1, np.sqrt(mask.area / 2) ** 0.5, 200)
    ripley = estimator.Hfunction(data=points, radii=radii, mode="none")

    return ripley, radii


# Store feature names, descriptions, and respective functions.
feature_set = dict(
    {
        "ripley": {"description": "ripley features", "function": _calc_ripley_features},
        "distance": {
            "description": "distance features",
            "function": _calc_norm_distance_quantile_features,
        },
        "morphology": {
            "description": "morphology features",
            "function": _calc_morph_enrichment,
        },
        "mask_fraction": {
            "description": "ripley features",
            "function": _calc_nuclear_fraction,
        },
        "indexes": {"description": "index features", "function": _calc_indexes},
    }
)


def get_feature(data, feature):
    """Convenience function for retrieving prepared sample feature values.

    Parameters
    ----------
    data : AnnData
        AnnData formatted spatial data.
    feature : str
        Name of feature. See `bento.tl.list_features` for list of implemented features.

    Returns
    -------
    np.array or None
        [description]
    """
    try:
        return data.uns["sample_data"][feature]
    except KeyError:
        return None


def list_features():
    """Return table of features.

    Returns
    -------
    dict
        Feature names and descriptions.
    """
    return {f'{k}: {feature_set[k]["description"]}' for k, v in feature_set.items()}
