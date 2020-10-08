import sys

import geopandas
import pandas as pd
import numpy as np
from collections import defaultdict
import cv2

from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance, distance_matrix
from scipy.stats import spearmanr
from scipy.stats.mstats import zscore

from .._utils import _poly2grid, _ripley, _quantify_variable

from .._settings import settings
from .._settings import pandarallel

from collections import defaultdict

# TODO
def filter_cells(data, min_points=5):
    if not min_points:
        print('`min_points` can not be None')
        return data
    else:
        npoints_cell = data.obs['cell'].value_counts()
        filt_cells = npoints_cell[npoints_cell >= min_points].index
        filt_points = data.obs['cell'].isin(filt_cells)
        return data[filt_points, :]



def filter_genes(data, min_points):

    def _filter_genes(obs_gene_df, min_points):
        """
        Return
        """
        gene_expr = obs_gene_df.groupby('gene').apply(len)
        genes_keep = gene_expr[gene_expr >= min_points].index
        obs_gene_df_filt = obs_gene_df.loc[obs_gene_df['gene'].isin(genes_keep)]
        return obs_gene_df_filt

    # For each cell, select genes that pass min. threshold
    gene_by_cell = data.obs.groupby('cell')[['gene']]
    obs_filt = gene_by_cell.apply(lambda obs_gene_df: _filter_genes(obs_gene_df, min_points))
    obs_keep = obs_filt.index.get_level_values(1)

    return data[obs_keep, :]


def prepare_features(data, features=[]):
    """Prepare features from raw data. TODO move features to preprocessing

    Parameters
    ----------
    data : AnnData
        AnnData formatted spatial data.
    features : list of str
        List of feature names to compute, by default empty list

    Returns
    -------
    [type]
        [description]
    """
    adata = data.copy()
    # Initialize features dict
    if 'features' not in adata.uns.keys():
        adata.uns['features'] = pd.DataFrame()

    # Prepare features for each cell separately
    cells = pd.Series(adata.obs['cell'].unique())
    cells = cells[cells != -1] # Ignore transcripts outside cells
    f = cells.progress_apply(lambda cell: _prepare_cell_features(adata[adata.obs['cell'] == cell], features, cell))

    f = pd.concat(f.tolist(), keys=cells.tolist(), axis=0)
    adata.uns['features'] = f
    adata.uns['features'].index = pd.MultiIndex.from_tuples(f.index, names=('cell', 'gene'))
    return adata


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

        return pd.Series([max_l, max_gradient, min_gradient, l_corr, l_4],
                         index=['max_l', 'max_gradient', 'min_gradient', 'l_corr', 'l_4'])

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

        return pd.Series(features, index=features_labels)

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

        enrichment = {}

        # Create GeoDataFrame from points
        points = pd.DataFrame(gene_data.X, columns=['x', 'y'])
        points = geopandas.GeoDataFrame(
            geometry=geopandas.points_from_xy(points.x, points.y))

        # Count fraction of points in mask
        for p in proportions:
            n_points = len(geopandas.clip(points, morph_masks[p]))
            enrichment[f'morph_enrichment_{p}'] = np.float(
                n_points) / len(points)

        return pd.Series(enrichment)

    return cell_data.obs.groupby('gene').apply(lambda obs: _calc(cell_data[obs.index, :], cell))


def _calc_nuclear_fraction(cell_data, cell):
    def _calc(gene_data, cell):
        nucleus_index = cell_data.uns['mask_index']['nucleus']
        n_index = nucleus_index[nucleus_index['cell'] == cell].index[0]
        nuclear_count = sum(gene_data.obs['nucleus'] == n_index)
        ratio = float(nuclear_count) / len(gene_data)
        return pd.Series(ratio, index=['nuclear_fraction'])

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

        feature_labels = ['polarization_index',
                          'dispersion_index', 'peripheral_distribution_index']
        return pd.Series([polarization_index, dispersion_index, peripheral_distribution_index], index=feature_labels)

    return cell_data.obs.groupby('gene').apply(lambda obs: _calc(cell_data[obs.index, :], cell))


def _rasterize(cell_data, cell):
    output_size = 32
    
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

    return cell_data.obs.groupby('gene').apply(lambda obs: _calc(cell_data[obs.index, :], cell))


    

# Store feature names, descriptions, and respective functions.
# Format:
# { 'feature_name': {
#     'description': ''.
#     'function': fn,
#   }
# }
feature_set = dict({'ripley': {'description': 'ripley features', 'function': _calc_ripley_features},
             'distance': {'description': 'distance features', 'function': _calc_norm_distance_quantile_features},
             'morphology': {'description': 'morphology features', 'function': _calc_morph_enrichment},
             'mask_fraction': {'description': 'ripley features', 'function': _calc_nuclear_fraction},
             'indexes': {'description': 'index features', 'function': _calc_indexes},
             'raster': {'description': 'rasterize to image', 'function': _rasterize}})


def get_features():
    """Return table of features.

    Returns
    -------
    DataFrame
        Formatted list of features.
    """
    feature_table = pd.DataFrame(feature_set)
    return feature_table
