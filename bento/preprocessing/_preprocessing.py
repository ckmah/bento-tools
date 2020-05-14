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

from .._settings import settings

pandarallel.initialize(nb_workers=settings.n_cores, verbose=0)

_progress_bar = None
# Global computations for memoization
_cell_cache = {}

def prepare_features(data,smFISH=False):
    """
    """
    global _progress_bar
    _progress_bar = tqdm(data['cell_id'])
    _progress_bar.set_description('Preparing features...')

    points = data['points']
    cell = data['cell']
    nucleus = data['nucleus']

    features = []
    # Prepare features for each cell independently
    for cell_id in _progress_bar:
        # Select points in cell
        cell_points = points.loc[points['cell_id'] == cell_id]
        cell_mask = cell.loc[cell['cell_id'] == cell_id, 'geometry'].values[0]
        nucleus_mask = nucleus.loc[cell['cell_id'] == cell_id, 'geometry'].values[0]

        _prepare_cell_cache(cell_points, cell_mask, nucleus_mask, cell_id)

        # Prepare features for each gene in cell
        cell_features = cell_points.groupby('gene').parallel_apply(lambda gene_points: _prepare_features(
            gene_points,
            cell_mask,
            nucleus_mask,
            cell_id))

        cell_features['cell_id'] = cell_id
        features.append(cell_features)


    features = pd.concat(features)
    if smFISH == False:
        # Z-score normalize features
        features = features.apply(zscore)
    else:
        pass

    # TODO add more features
    data['features'] = features
    return data


def _prepare_cell_cache(points, cell_mask, nucleus_mask, cell_id):
    """Precompute cell-specific features.

    Parameters
    ----------
    cell_mask : Geo
        [description]
    nucleus_mask : [type]
        [description]
    """
    cache = {}

    # Uniform grid of points across cell
    cache['grid_points'] = _poly2grid(cell_mask)

    # Get all centroids
    cache['grid_centroid'] = [cache['grid_points'].mean(axis=1)]
    cache['cell_centroid'] = [np.array(cell_mask.centroid)]
    cache['nucleus_centroid'] = [np.array(nucleus_mask.centroid)]

    # Calculate cell's radius of gyration
    # TODO memoize this
    cell_centroid_array = np.tile(
        cache['cell_centroid'], len(cache['grid_points'])).reshape(-1, 2)
    cache['radius_of_gyration'] = mean_squared_error(cache['grid_points'], cell_centroid_array)

    # Mean cell radius size
    cell_radius = distance.cdist(cache['cell_centroid'],
                                 np.array(cell_mask.exterior.xy).T)
    cache['mean_cell_radius'] = abs(cell_radius).mean()

    proportions = [.1, .25]
    for p in proportions:
        d = cache['mean_cell_radius'] * p

        morph_p = f'morph_{p}_mask'
        cache[morph_p] = cell_mask.buffer(distance=-d).buffer(distance=d)

    global _cell_cache
    _cell_cache[cell_id] = cache


def _poly2grid(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    x, y = np.meshgrid(np.arange(minx, maxx, step=np.float(20)),
                       np.arange(miny, maxy, step=np.float(20)))
    x = x.flatten()
    y = y.flatten()
    xy = np.array([x, y]).T
    polygon_path = mplPath.Path(np.array(polygon.exterior.xy).T)
    polygon_cell_mask = polygon_path.contains_points(xy)
    xy = xy[polygon_cell_mask]

    return xy

def _prepare_features(points, cell_mask, nucleus_mask, cell_id):
    cell_features = _calc_ripley_features(points, cell_mask)
    cell_features = cell_features.append(
        _calc_norm_distance_quantile_features(points, cell_mask, nucleus_mask))
    cell_features = cell_features.append(
        _calc_morph_enrichment(points, cell_mask, cell_id))
    cell_features = cell_features.append(
        _calc_nuclear_fraction(points, cell_mask, nucleus_mask))
    cell_features = cell_features.append(_calc_indexes(points, cell_id))

    return cell_features


def _calc_ripley_features(points, cell_mask):
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

    ripley_norm, radii = _ripley(points, cell_mask)

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
    cell_mask_points = np.array(cell_mask.exterior.coords.xy).T
    l_4_dist = distance_matrix(cell_mask_points, cell_mask_points).max() / 4
    l_4 = _ripley(points, cell_mask, [l_4_dist])[0][0]

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
    # TODO use loc instead of clip
    nuclear_count = len(geopandas.clip(points, nucleus_mask))
    ratio = float(nuclear_count) / total_count
    return pd.Series(ratio, index=['nuclear_fraction'])


def _calc_indexes(points, cell_id):
    """Calculate polarization index, dispersion index, and peripheral index.
    """
    global _cell_cache
    cache = _cell_cache[cell_id]

    # Calculate distance between point and cell centroids
    point_centroid = [points[['x', 'y']].mean().values]
    offset = distance.cdist(point_centroid, _cell_cache[cell_id]['cell_centroid'])[0][0]

    # Calculate polarization index
    polarization_index = offset / cache['radius_of_gyration']

    # Calculate second moment of points with centroid as reference
    def _calc_second_moment(centroid, pts):
        '''
        centroid : [1 x 2] float
        pts : [n x 2] float
        '''
        radii = distance.cdist(centroid, pts)
        second_moment = np.sum(radii * radii / len(pts))
        return second_moment

    # Calculate dispersion index; d-index = 2nd moment of spots, norm. by 2nd moment of uniform spots; moments relative to spot centroid
    dispersion_index = _calc_second_moment(point_centroid, points[['x', 'y']].values) / \
                       _calc_second_moment(point_centroid, cache['grid_points'])

    # Calculate peripheral distribution index; d-index = 2nd moment of spots, norm. by 2nd moment of uniform spots; moments relative to nucleus centroid
    peripheral_distribution_index = _calc_second_moment(cache['nucleus_centroid'], points[['x', 'y']].values) / \
                                    _calc_second_moment(cache['nucleus_centroid'], cache['grid_points'])


    feature_labels = ['polarization_index', 'dispersion_index', 'peripheral_distribution_index']
    return pd.Series([polarization_index, dispersion_index, peripheral_distribution_index], index=feature_labels)


def _calc_morph_enrichment(points, cell_mask, cell_id):
    global _cell_cache
    cache = _cell_cache[cell_id]
    # Mean cell radius size
    cell_radius = distance.cdist(cache['cell_centroid'],
                                 np.array(cell_mask.exterior.xy).T)
    mean_cell_radius = abs(cell_radius).mean()

    enrichment = {}
    proportions = [.1, .25]
    # Count proportion of points in mask
    for p in proportions:
        # Count points in morph mask
        n_points = len(geopandas.clip(points, cache[f'morph_{p}_mask']))
        enrichment[f'morph_enrichment_{p}'] = np.float(n_points) / len(points)

    return pd.Series(enrichment)
