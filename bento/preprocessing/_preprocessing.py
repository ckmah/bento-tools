import numpy as np
import pandas as pd
import geopandas
from astropy.stats import RipleysKEstimator
from scipy.stats import spearmanr
from scipy.stats.mstats import zscore
from scipy.spatial import distance, distance_matrix
from loguru import logger
from tqdm import tqdm


def prepare_features(data):
    """
    """
    num_genes = len(data['points']['gene'].unique())
    logger.info(f'Preparing features for {num_genes} genes...')

    features = []
    for id, cell_mask, nucleus_mask in tqdm(zip(data['cell_id'], data['cell'], data['nucleus']), total=len(data['cell'])):
        # Select points in cell
        cell_points = geopandas.clip(data['points'], cell_mask)

        # Prepare features for each gene
        cell_features = cell_points.groupby('gene').apply(
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
    Calculate estimation for Ripley K function
    """
    estimator_params = dict(area=mask.area,
                            x_min=points.x.min(), x_max=points.x.max(),
                            y_min=points.y.min(), y_max=points.y.max())
    estimator = RipleysKEstimator(**estimator_params)

    # https://docs.astropy.org/en/stable/api/astropy.stats.RipleysKEstimator.html#astropy.stats.RipleysKEstimator
    if radii is None:
        radii = np.linspace(1, np.sqrt(
            estimator_params['area'] / 2) ** 0.5, 200)
    ripley = estimator(data=points[['x', 'y']], radii=radii, mode='none')

    # L-function normalization (see supp info: https://www.nature.com/articles/s41467-018-06868-w)
    ripley_norm = np.sqrt(ripley / np.pi) - radii

    return ripley_norm, radii


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
    # TODO remove later when this method is called in prepare_features
    points = geopandas.clip(points, cell_mask)

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


def _calc_polarization_index():
    pass


def _calc_dispersion_index():
    pass


def _calc_morph_enrichment():
    pass
