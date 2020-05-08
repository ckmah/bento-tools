import sys

import geopandas
import numpy as np
import pandas as pd
from astropy.stats import RipleysKEstimator
import matplotlib.path as mplPath
from scipy.spatial import distance, distance_matrix
from scipy.stats import spearmanr
from scipy.stats.mstats import zscore
from sklearn.metrics import mean_squared_error
from pandarallel import pandarallel
from tqdm import tqdm

pandarallel.initialize(nb_workers=4)

# Global computations for memoization
_progress_bar = None
_cell_morph_masks = {}

def prepare_features(data):
    """
    """
    global _progress_bar
    _progress_bar = tqdm(
        zip(data['cell_id'], data['cell'], data['nucleus']), total=len(data['cell']))
    _progress_bar.set_description('Preparing features...')

    features = []
    for id, cell_mask, nucleus_mask in _progress_bar:
        # Select points in cell
        cell_points = geopandas.clip(data['points'], cell_mask)
        _cell_morph_masks = {}

        # Prepare features for each gene
        cell_features = cell_points.groupby('gene').parallel_apply(
            lambda gene_points: _prepare_features(gene_points, cell_mask, nucleus_mask))

        cell_features['cell_id'] = id
        features.append(cell_features)


    features = pd.concat(features)

    # Z-score normalize features
    features = features.apply(zscore)

    # TODO add more features
    data['features'] = features
    return data


def _prepare_features(points, cell_mask, nucleus_mask):
    cell_features = _calc_ripley_features(points, cell_mask)
    cell_features = cell_features.append(
        _calc_norm_distance_quantile_features(points, cell_mask, nucleus_mask))
    cell_features = cell_features.append(
        _calc_morph_enrichment(points, cell_mask))
    cell_features = cell_features.append(
        _calc_nuclear_fraction(points, cell_mask, nucleus_mask))
    cell_features = cell_features.append(_calc_indexes(points, cell_mask))

    return cell_features


def _calc_ripley_features(points, mask):
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
    global _progress_bar
    _progress_bar.set_description('Computing Ripley features...')

    ripley_norm, radii = _ripley(points, mask)

    # Max value of the L-function
    max_l = ripley_norm.max()

    # Max and min value of the gradient of L
    ripley_smooth = pd.Series(ripley_norm).rolling(5).mean()
    ripley_smooth.dropna(inplace=True)
    ripley_gradient = np.gradient(ripley_smooth)
    max_gradient = ripley_gradient.max()
    min_gradient = ripley_gradient.min()

    # Monotony of L-function in the interval
    l_corr = spearmanr(radii, ripley_norm)[0]

    # L-function at L/4 where length of the cell L is max dist between 2 points on polygon defining cell border
    mask_points = np.array(mask.exterior.coords.xy).T
    l_4_dist = distance_matrix(mask_points, mask_points).max() / 4
    l_4 = _ripley(points, mask, [l_4_dist])[0][0]

    return pd.Series([max_l, max_gradient, min_gradient, l_corr, l_4],
                     index=['max_l', 'max_gradient', 'min_gradient', 'l_corr', 'l_4'])


def _ripley(points, mask, radii=None):
    """
    Calculate estimation for Ripley H function.

    Parameters
    ----------
    points : GeoDataFrame
    mask : shapely.Polygon
    radii : float list

    Returns
    -------
    list
        ripley l function
    radii
    """
    estimator = RipleysKEstimator(area=mask.area,
                                  x_min=points.x.min(), x_max=points.x.max(),
                                  y_min=points.y.min(), y_max=points.y.max())

    # https://docs.astropy.org/en/stable/api/astropy.stats.RipleysKEstimator.html#astropy.stats.RipleysKEstimator
    if radii is None:
        radii = np.linspace(1, np.sqrt(mask.area / 2) ** 0.5, 200)
    ripley = estimator.Hfunction(
        data=points[['x', 'y']], radii=radii, mode='none')

    return ripley, radii


def _calc_norm_distance_quantile_features(points, cell_mask, nucleus_mask):
    """Return 5th, 10th, 20th, and 50th quantiles for distance to cell and nucleus centroids.

    Parameters
    ----------
    points : [type]
        [description]
    cell_mask : [type]
        [description]
    nucleus_mask : [type]
        [description]

    Returns
    -------
    pd.Series
    """
    global _progress_bar
    _progress_bar.set_description('Computing distance features...')

    # Calculate normalized distances from points to centroids
    cell_centroid_dist = _calc_norm_dist_to_centroid(points, cell_mask)
    nucleus_centroid_dist = _calc_norm_dist_to_centroid(points, nucleus_mask)

    # Calculate normalized distances from points to mask; distance = point to closest point on mask
    cell_mask_dist = _calc_norm_dist_to_mask(points, cell_mask)
    nucleus_mask_dist = _calc_norm_dist_to_mask(points, nucleus_mask)

    features_raw = [cell_centroid_dist, nucleus_centroid_dist,
                    cell_mask_dist, nucleus_mask_dist]
    features_label_prefix = ['cell_centroid_distance', 'nucleus_centroid_distance',
                             'cell_mask_distance', 'nucleus_mask_distance']

    # Get quantiles of each distance distribution
    features = []
    features_labels = []
    for feature, label in zip(features_raw, features_label_prefix):
        quantiles = [5, 10, 20, 50]
        features.extend(np.percentile(feature, quantiles))
        features_labels.extend([f'{label}_q{q}' for q in quantiles])

    return pd.Series(features, index=features_labels)


def _calc_norm_dist_to_centroid(points, mask):
    """Calculate distance between every value in points to point.

    Parameters
    ----------
    points : [type]
        [description]
    point : [type]
        [description]
    """
    values = points[['x', 'y']]
    mask_values = np.array(mask.exterior.coords.xy).T
    centroid = np.array(mask.centroid).reshape(1, 2)

    expected_distance = abs(distance.cdist(mask_values, centroid)).mean()
    observed_distance = abs(distance.cdist(values, centroid))
    return observed_distance / expected_distance


def _calc_norm_dist_to_mask(points, mask):
    values = points[['x', 'y']]
    mask_values = np.array(mask.exterior.coords.xy).T
    centroid = np.array(mask.centroid).reshape(1, 2)

    expected_distance = abs(distance.cdist(mask_values, centroid)).mean()
    observed_distance = abs(distance.cdist(values, mask_values)).min(axis=0)
    return observed_distance / expected_distance


def _calc_nuclear_fraction(points, cell_mask, nucleus_mask):
    total_count = len(points)
    nuclear_count = len(geopandas.clip(points, nucleus_mask))
    ratio = float(nuclear_count) / total_count
    return pd.Series(ratio, index=['nuclear_fraction'])


def _calc_indexes(points, mask):
    """Calculate polarization index, dispersion index, and peripheral index.
    """
    global _progress_bar
    _progress_bar.set_description('Computing index features...')

    point_centroid = [points[['x', 'y']].mean().values]
    cell_centroid = [np.array(mask.centroid)]
    offset = distance.cdist(point_centroid, cell_centroid)[0][0]

    mask_grid = _poly2grid(mask)
    cell_centroid_array = np.tile(cell_centroid, len(mask_grid)).reshape(-1, 2)

    radius_of_gyration = mean_squared_error(mask_grid, cell_centroid_array)
    polarization_index = offset / radius_of_gyration

    return pd.Series([polarization_index], index=['polarization_index'])


def _poly2grid(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    x, y = np.meshgrid(np.linspace(minx, maxx, num=100),
                       np.linspace(miny, maxy, num=100))
    x = x.flatten()
    y = y.flatten()
    xy = np.array([x, y]).T
    polygon_path = mplPath.Path(np.array(polygon.exterior.xy).T)
    polygon_mask = polygon_path.contains_points(xy)
    xy = xy[polygon_mask]

    return xy


def _calc_morph_enrichment(points, mask):
    global _progress_bar
    _progress_bar.set_description('Computing morphological features...')

    # Mean radius size
    mean_radius = distance.cdist(np.array(mask.centroid).reshape(1, 2),
                                 np.array(mask.exterior.xy).T)
    mean_radius = abs(mean_radius).mean()
    total_points = len(points)

    global _cell_morph_masks

    enrichment = {}
    proportions = [.1, .25]
    for p in proportions:
        d = mean_radius * p

        p_str = str(p)
        if p_str not in _cell_morph_masks:
            _cell_morph_masks[p_str] = mask.buffer(distance=-d).buffer(distance=d)

        n_points = len(geopandas.clip(points, _cell_morph_masks[p_str]))

        stat = np.float(n_points) / total_points
        enrichment[f'morph_enrichment_{p}'] = stat

    return pd.Series(enrichment)
