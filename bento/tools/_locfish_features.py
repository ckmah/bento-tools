from ast import Try
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os

import dask.dataframe as dd
import geopandas as gpd
import matplotlib.path as mplPath
import numpy as np
import pandas as pd
from astropy.stats import RipleysKEstimator
from joblib import Parallel, delayed
from scipy.spatial import distance, distance_matrix
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

from ..preprocessing import get_points

def ripley_features(data, n_cores=1, copy=False):
    """
    Compute 5 features extracted from Ripley L-function.

    https://docs.astropy.org/en/stable/api/astropy.stats.RipleysKEstimator.html#astropy.stats.RipleysKEstimator

    Returns
    -------
    dict : keyed feature values
        max_l : float
        max_gradient : float
        min_gradient : float
        monotony : float
        l_4 : float
    """

    adata = data.copy() if copy else data

    def ripley_per_cell(c, p, cell_name):
        # L/4 where length of the cell L is max dist between 2 points on polygon defining cell border
        c_coords = np.array(c.exterior.coords.xy).T
        l_4_dist = int(distance_matrix(c_coords, c_coords).max() / 4)

        features = []
        for gene, gene_pts in p.groupby("gene"):
            estimator = RipleysKEstimator(
                area=c.area,
                x_min=float(gene_pts["x"].min()),
                x_max=float(gene_pts["x"].max()),
                y_min=float(gene_pts["y"].min()),
                y_max=float(gene_pts["y"].max()),
            )

            # Compute ripley function for r=(1, cell diameter / 2), step size = 1 pixel
            radii = np.linspace(1, l_4_dist * 2, num=l_4_dist * 2)
            stats = estimator.Hfunction(
                data=gene_pts[["x", "y"]].values, radii=radii, mode="none"
            )

            # Max value of the L-function
            max_l = max(stats)

            # Max and min value of the gradient of L
            ripley_smooth = pd.Series(stats).rolling(5).mean()
            ripley_smooth.dropna(inplace=True)

            ripley_gradient = np.gradient(ripley_smooth)
            max_gradient = ripley_gradient.max()
            min_gradient = ripley_gradient.min()

            # Monotony of L-function in the interval
            l_corr = spearmanr(radii, stats)[0]

            # L-function at L/4 where length of the cell L is max dist between 2 points on polygon defining cell border
            l_4 = estimator.Hfunction(
                data=gene_pts[["x", "y"]].values, radii=[l_4_dist], mode="none"
            )[0]

            features.append(
                [cell_name, gene, max_l, max_gradient, min_gradient, l_corr, l_4]
            )

        return features

    # Parallelize points
    cell_features = Parallel(n_jobs=n_cores)(
        delayed(ripley_per_cell)(
            adata.obs.loc[cell_name, "cell_shape"],
            get_points(adata, cells=cell_name),
            cell_name,
        )
        for cell_name in tqdm(adata.obs_names.tolist())
    )

    cell_features = np.array(cell_features).reshape(-1, 7)

    colnames = [
        "cell",
        "gene",
        "max_l",
        "max_l_gradient",
        "min_l_gradient",
        "l_corr",
        "l_at_l4",
    ]
    cell_features = pd.DataFrame(cell_features, columns=colnames)

    for f in colnames[2:]:
        adata.layers[f] = (
            cell_features.pivot(index="cell", columns="gene", values=f)
            .reindex(index=adata.obs_names, columns=adata.var_names)
            .astype(float)
        )

    return adata if copy else None


def distance_features(data, n_cores=1, copy=False):
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
    adata = data.copy() if copy else data

    def distance_per_cell(c, n, p, cell_name):

        features = []
        for gene, gene_pts in p.groupby("gene"):
            # Calculate normalized distances from points to centroids
            xy = gene_pts[["x", "y"]].values
            cell_centroid_dist = norm_dist_to_centroid(xy, c)
            nucleus_centroid_dist = norm_dist_to_centroid(xy, n)

            # Calculate normalized distances from points to mask; distance = point to closest point on mask
            cell_mask_dist = norm_dist_to_mask(xy, c)
            nucleus_mask_dist = norm_dist_to_mask(xy, n)

            features_raw = [
                cell_centroid_dist,
                nucleus_centroid_dist,
                cell_mask_dist,
                nucleus_mask_dist,
            ]

            # Get quantiles of each distance distribution
            stats = [cell_name, gene]
            for f in features_raw:
                quantiles = [5, 10, 20, 50]
                stats.extend(np.percentile(f, quantiles))

            # There should be 16 features (18 with cell/gene names)
            features.append(np.array(stats))

        return features

    # Parallelize points
    cell_features = Parallel(n_jobs=n_cores)(
        delayed(distance_per_cell)(
            adata.obs.loc[cell_name, "cell_shape"],
            adata.obs.loc[cell_name, "nucleus_shape"],
            get_points(adata, cells=cell_name),
            cell_name,
        )
        for cell_name in tqdm(adata.obs_names.tolist())
    )

    cell_features = np.array(cell_features).reshape(-1, 18)

    # Enuemerate feature names
    feature_labels = ["cell", "gene"]
    features_label_prefix = [
        "cell_centroid_dist",
        "nucleus_centroid_dist",
        "cell_mask_dist",
        "nucleus_mask_dist",
    ]

    for label in features_label_prefix:
        quantiles = [5, 10, 20, 50]
        feature_labels.extend([f"{label}_q{q}" for q in quantiles])

    cell_features = pd.DataFrame(cell_features, columns=feature_labels)

    for f in feature_labels[2:]:
        adata.layers[f] = (
            cell_features.pivot(index="cell", columns="gene", values=f)
            .reindex(index=adata.obs_names, columns=adata.var_names)
            .astype(float)
        )

    return adata if copy else None


def norm_dist_to_centroid(points, mask):
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


def norm_dist_to_mask(points, mask):
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


def morph_enrichment(data, n_cores=1, copy=False):
    adata = data.copy() if copy else data

    proportions = [0.05, 0.1]

    def enrichment_per_cell(c, p, cell_name):

        # Mean cell radius size
        mean_cell_radius = distance.cdist(
            np.array(c.centroid).reshape(1, 2), np.array(c.exterior.xy).T
        ).mean()

        # Define proportion of radius to perform morph opening
        morph_masks = {}
        for prop in proportions:
            # Define proportion of radius; compute morph opening
            d = mean_cell_radius * prop
            morph_masks[prop] = c.buffer(distance=-d).buffer(distance=d)

        features = []

        for gene, gene_pts in p.groupby("gene"):

            # Create GeoDataFrame from points
            pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gene_pts.x, gene_pts.y))

            # Count fraction of points in mask
            enrichment = [cell_name, gene]
            for prop in proportions:
                n_points = len(gpd.clip(pts, morph_masks[prop]))
                enrichment.append(np.float(n_points) / len(pts))

            features.append(np.array(enrichment))
        return features

    # Parallelize points
    cell_features = Parallel(n_jobs=n_cores)(
        delayed(enrichment_per_cell)(
            adata.obs.loc[cell_name, "cell_shape"],
            get_points(adata, cells=cell_name),
            cell_name,
        )
        for cell_name in tqdm(adata.obs_names.tolist())
    )

    cell_features = np.array(cell_features).reshape(-1, 4)

    # Enuemerate feature labels
    feature_labels = ["cell", "gene"]
    [feature_labels.append(f"morph_{prop}") for prop in proportions]

    cell_features = pd.DataFrame(cell_features, columns=feature_labels)

    for f in feature_labels[2:]:
        adata.layers[f] = (
            cell_features.pivot(index="cell", columns="gene", values=f)
            .reindex(index=adata.obs_names, columns=adata.var_names)
            .astype(float)
        )

    return adata if copy else None


def nuclear_fraction(data, n_cores=1, copy=False):
    """
    Parameters
    ----------
    n_cores : int
        Parameter kept for uniformity. Increasing number of cores will not do anything.
    """
    adata = data.copy() if copy else data

    adata.layers["nuclear_fraction"] = adata.layers["unspliced"] / adata.X

    return adata if copy else None


def moment_stats(data, n_cores=1, copy=False):
    """Calculate polarization index, dispersion index, and peripheral index. Requires both cell and nucleus."""
    adata = data.copy() if copy else data

    def moments_per_cell(c, n, p, cell_name):
        # Calculate of cell-dependent only values
        cell_centroid = np.array(c.centroid).reshape(1, 2)
        grid_points = _poly2grid(c)
        cell_centroid_array = np.tile(cell_centroid, len(grid_points))
        cell_centroid_array = cell_centroid_array.reshape(-1, 2)
        radius_of_gyration = mean_squared_error(grid_points, cell_centroid_array)

        nucleus_centroid = np.array(n.centroid).reshape(1, 2)

        nuclear_moment = _second_moment(nucleus_centroid, grid_points)

        features = []

        for gene, gene_pts in p.groupby("gene"):

            # Calculate distance between point and cell centroids
            pts = gene_pts[["x", "y"]].values
            point_centroid = pts.mean(axis=0).reshape(1, 2)
            offset = distance.cdist(point_centroid, cell_centroid)[0][0]

            # Calculate polarization index
            polarization_index = offset / radius_of_gyration

            # Calculate dispersion index; d-index = 2nd moment of spots, norm. by 2nd moment of uniform spots; moments relative to spot centroid
            dispersion_index = _second_moment(point_centroid, pts) / _second_moment(
                point_centroid, grid_points
            )

            # Calculate peripheral distribution index; d-index = 2nd moment of spots, norm. by 2nd moment of uniform spots; moments relative to nucleus centroid
            peripheral_distribution_index = (
                _second_moment(nucleus_centroid, pts) / nuclear_moment
            )

            features.append(
                np.array(
                    [
                        cell_name,
                        gene,
                        polarization_index,
                        dispersion_index,
                        peripheral_distribution_index,
                    ]
                )
            )

        return features

    # Parallelize points
    cell_features = Parallel(n_jobs=n_cores)(
        delayed(moments_per_cell)(
            adata.obs.loc[cell_name, "cell_shape"],
            adata.obs.loc[cell_name, "nucleus_shape"],
            get_points(adata, cells=cell_name),
            cell_name,
        )
        for cell_name in tqdm(adata.obs_names.tolist())
    )

    cell_features = np.array(cell_features).reshape(-1, 5)

    # Enuemerate feature labels
    feature_labels = [
        "cell",
        "gene",
        "polar_moment",
        "dispersion_moment",
        "peripheral_moment",
    ]

    cell_features = pd.DataFrame(cell_features, columns=feature_labels)

    for f in feature_labels[2:]:
        adata.layers[f] = (
            cell_features.pivot(index="cell", columns="gene", values=f)
            .reindex(index=adata.obs_names, columns=adata.var_names)
            .astype(float)
        )

    return adata if copy else None


# Calculate second moment of points with centroid as reference
def _second_moment(centroid, pts):
    """
    centroid : [1 x 2] float
    pts : [n x 2] float
    """
    radii = distance.cdist(centroid, pts)
    second_moment = np.sum(radii * radii / len(pts))
    return second_moment


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
