
import warnings
from collections import defaultdict

import cv2
import geopandas
import matplotlib.path as mplPath
import numpy as np
import pandas as pd
from astropy.stats import RipleysKEstimator
from scipy.spatial import distance, distance_matrix
from scipy.stats import spearmanr
from scipy.stats.mstats import zscore
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm

from .._settings import pandarallel, settings

# Parallelize by gene if many genes, otherwise by cell
gene_parallel_threshold = 1000

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
    cells = pd.Series(adata.obs['cell'].unique())
    if (cells == -1).any():
        print('Extracellular points detected. These will be ignored.')
        cells = cells.loc[cells != -1]

    # Prepare features for each cell separately
    tqdm.pandas(desc='Processing cells')
    f = cells.progress_apply(lambda cell: _prepare_cell_features(adata[adata.obs['cell'] == cell].copy(), features, cell))

    # Stack as single dataframe
    f = pd.concat(f.tolist(), keys=cells.tolist(), axis=0)
    
    # Save sample -> (cell, gene) mapping
    adata.uns['sample_index'] = f.index.to_frame(index=False)
    adata.uns['sample_index'].columns = ['cell', 'gene']    

    # Save obs mapping 
    adata.obs = adata.obs.merge(adata.uns['sample_index'].reset_index(), how='left', on=['cell', 'gene'])
    adata.obs.rename({'index': 'sample_id'}, axis=1, inplace=True)
    adata.obs['sample_id'].fillna(-1, inplace=True)
    adata.obs['sample_id'] = adata.obs['sample_id'].astype(int)

    # Save each feature as {feature : np.array} in adata.uns
    feature_dict = f.reset_index(drop=True).to_dict(orient='list')
    feature_dict = {key: np.array(values) for key, values in feature_dict.items()}

    # Save each feature to `sample_data` dictionary.
    # Each feature follows {sample_id: feature_value} format, where sample_id corresponds to uns['sample_index'].index.
    if 'sample_data' not in adata.uns:
        adata.uns['sample_data'] = dict()
    adata.uns['sample_data'].update(feature_dict)

    return adata if copy else None


def _prepare_cell_features(cell_data, features, cell):
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
        fn = feature_set[feature]['function']

        # All functions should take two args: AnnData and index for cell
        # Execute function that returns DataFrame [genes x 1+] float
        computed_features.append(fn(cell_data, cell))

    # print(computed_features)
    return pd.concat(computed_features, axis=1)


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

    cell_mask = cell_data.uns['masks']['cell'].loc[cell, 'geometry']

    # L/4 where length of the cell L is max dist between 2 points on polygon defining cell border
    cell_mask_points = np.array(cell_mask.exterior.coords.xy).T
    l_4_dist = distance_matrix(
        cell_mask_points, cell_mask_points).max() / 4

    def _calc(gene_data, cell):
        # Compute ripley function for r=(1, cell diameter / 2)
        ripley_norm, radii = _ripley(gene_data.X, cell_mask, radii=np.linspace(1, l_4_dist*2, 100))

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

    return cell_data.obs.groupby('gene').apply(lambda obs: _calc(cell_data[obs.index, :], cell))


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
    cell_mask = cell_data.uns['masks']['cell'].loc[cell, 'geometry']

    # Select nucleus mask
    nucleus_index = cell_data.uns['mask_index']['nucleus']
    n_index = nucleus_index[cell_data.uns['mask_index']['nucleus'] == cell].index[0]
    nucleus_mask = cell_data.uns['masks']['nucleus'].loc[n_index, 'geometry']

    def _calc(gene_data, cell):

        # Select point coordinates
        points = np.array(gene_data.X)

        # Calculate normalized distances from points to centroids
        cell_centroid_dist = _calc_norm_dist_to_centroid(points, cell_mask)
        nucleus_centroid_dist = _calc_norm_dist_to_centroid(
            points, nucleus_mask)

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

        return np.array(features)

    return cell_data.obs.groupby('gene').apply(lambda obs: _calc(cell_data[obs.index, :], cell))


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
    cell_mask = cell_data.uns['masks']['cell'].loc[cell, 'geometry']

    # Morphological openings on cell mask
    # Mean cell radius size
    mean_cell_radius = distance.cdist(
        np.array(cell_mask.centroid).reshape(1, 2), np.array(cell_mask.exterior.xy).T).mean()

    # Define proportion of radius to perform morph opening
    proportions = [.1, .25]
    morph_masks = {}
    for p in proportions:
        # Define proportion of radius
        d = mean_cell_radius * p

        # Compute morph opening
        morph_masks[p] = cell_mask.buffer(
            distance=-d).buffer(distance=d)

    def _calc(gene_data, cell):

        enrichment = []

        # Create GeoDataFrame from points
        points = pd.DataFrame(gene_data.X, columns=['x', 'y'])
        points = geopandas.GeoDataFrame(
            geometry=geopandas.points_from_xy(points.x, points.y))

        # Count fraction of points in mask
        for p in proportions:
            n_points = len(geopandas.clip(points, morph_masks[p]))
            enrichment.append(np.float(n_points) / len(points))

        return np.array(enrichment)

    return cell_data.obs.groupby('gene').apply(lambda obs: _calc(cell_data[obs.index, :], cell))


def _calc_nuclear_fraction(cell_data, cell):
    def _calc(gene_data, cell):
        nucleus_index = cell_data.uns['mask_index']['nucleus']
        n_index = nucleus_index[nucleus_index['cell'] == cell].index[0]
        nuclear_count = sum(gene_data.obs['nucleus'] == n_index)
        ratio = float(nuclear_count) / len(gene_data)
        return ratio

    return cell_data.obs.groupby('gene').apply(lambda obs: _calc(cell_data[obs.index, :], cell))


def _calc_indexes(cell_data, cell):
    """Calculate polarization index, dispersion index, and peripheral index. Requires both cell and nucleus.
    """

    # Calculate of cell-dependent only values
    cell_mask = cell_data.uns['masks']['cell'].loc[cell, 'geometry']
    cell_centroid = np.array(cell_mask.centroid).reshape(1, 2)
    grid_points = _poly2grid(cell_mask)
    cell_centroid_array = np.tile(cell_centroid, len(grid_points))
    cell_centroid_array = cell_centroid_array.reshape(-1, 2)
    radius_of_gyration = mean_squared_error(grid_points, cell_centroid_array)

    nucleus_index = cell_data.uns['mask_index']['nucleus']
    n_index = nucleus_index[cell_data.uns['mask_index']['nucleus'] == cell].index[0]
    nucleus_centroid = np.array(
        cell_data.uns['masks']['nucleus'].loc[n_index, 'geometry'].centroid).reshape(1, 2)

    # Calculate second moment of points with centroid as reference
    def _calc_second_moment(centroid, pts):
        '''
        centroid : [1 x 2] float
        pts : [n x 2] float
        '''
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
        dispersion_index = _calc_second_moment(point_centroid, points) / \
            _calc_second_moment(point_centroid, grid_points)

        # Calculate peripheral distribution index; d-index = 2nd moment of spots, norm. by 2nd moment of uniform spots; moments relative to nucleus centroid
        peripheral_distribution_index = _calc_second_moment(nucleus_centroid, points) / \
            nuclear_moment

        return np.array([polarization_index, dispersion_index, peripheral_distribution_index])

    return cell_data.obs.groupby('gene').apply(lambda obs: _calc(cell_data[obs.index, :], cell))


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
    estimator = RipleysKEstimator(area=mask.area,
                                  x_min=float(points[:, 0].min()), x_max=float(points[:, 0].max()),
                                  y_min=float(points[:, 1].min()), y_max=float(points[:, 1].max()))

    # https://docs.astropy.org/en/stable/api/astropy.stats.RipleysKEstimator.html#astropy.stats.RipleysKEstimator
    # if radii is None:
    #     radii = np.linspace(1, np.sqrt(mask.area / 2) ** 0.5, 200)
    ripley = estimator.Hfunction(data=points, radii=radii, mode='none')

    return ripley, radii



def _rasterize(cell_data, cell):
    # TODO parameterize?
    output_size = 64
    
    # Initialize base image
    base_img = np.zeros((output_size, output_size))
    
    ##### Cell mask
    cell_xy = np.array(cell_data.uns['masks']['cell'].loc[cell, 'geometry'].exterior.xy).reshape(2, -1).T

    # shift to 0
    offset = cell_xy.min(axis=0)
    cell_xy = cell_xy - offset

    # scale to res
    scale_factor = (output_size*0.99) / cell_xy.max()
    cell_xy = cell_xy * scale_factor

    # Center
    center_offset = (output_size/2) - cell_xy.max(axis=0) / 2
    cell_xy = cell_xy + center_offset

    # Rasterize
    cell_xy = np.floor(cell_xy).astype(int)

    # Save to base image
    base_img = cv2.fillPoly(base_img, [cell_xy], 1)

    ##### Nuclear mask
    mask_index = cell_data.uns['mask_index']
    
    nucleus_i = mask_index['nucleus'].loc[mask_index['nucleus']['cell'] == cell].index[0]
    nucleus = cell_data.uns['masks']['nucleus'].loc[nucleus_i, 'geometry']
    nucleus_xy = np.array(nucleus.exterior.xy).reshape(2, -1).T
    nucleus_xy = (nucleus_xy - offset) * scale_factor + center_offset
    nucleus_xy = np.floor(nucleus_xy).astype(int)

    # Save to base image
    base_img = cv2.fillPoly(base_img, [nucleus_xy], 2)

    def _calc(gene_data, cell):
        ##### Points
        points = gene_data.X
        points = (points - offset) * scale_factor + center_offset
        points = np.floor(points).astype(int)
        points = np.clip(points, 0, output_size-1)
        
        # To dense image; points values start at 3+
        pts_img = np.zeros((output_size, output_size))
        for coo in points:
            if pts_img[coo[1], coo[0]] == 0:
                pts_img[coo[1], coo[0]] = 3
            else:
                pts_img[coo[1], coo[0]] += 1

        gene_img = base_img.copy()
        gene_img = np.where(pts_img > 0, pts_img, base_img).astype(np.float32)
        # image = torch.from_numpy(image) # convert to Tensor
        return pd.Series([gene_img], index=['raster'])

    if len(cell_data.obs['gene']) > gene_parallel_threshold:
        return cell_data.obs.groupby('gene').parallel_apply(lambda obs: _calc(cell_data[obs.index], cell))
    else:
        return cell_data.obs.groupby('gene').apply(lambda obs: _calc(cell_data[obs.index], cell))

# Store feature names, descriptions, and respective functions.
feature_set = dict({'ripley': {'description': 'ripley features', 'function': _calc_ripley_features},
             'distance': {'description': 'distance features', 'function': _calc_norm_distance_quantile_features},
             'morphology': {'description': 'morphology features', 'function': _calc_morph_enrichment},
             'mask_fraction': {'description': 'ripley features', 'function': _calc_nuclear_fraction},
             'indexes': {'description': 'index features', 'function': _calc_indexes},
             'raster': {'description': 'rasterize to image', 'function': _rasterize}})


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
        return data.uns['sample_data'][feature]
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

