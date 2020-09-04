import geopandas
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap as u
from collections import defaultdict
import cv2

from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance, distance_matrix
from scipy.stats import spearmanr

from .._utils import _poly2grid, _ripley
from .._settings import settings
from .._settings import pandarallel

tqdm.pandas()

def subsample(data, fraction):
    """Subsample data compartmentalized by cell.

    Parameters
    ----------
    data : AnnData
        AnnData formatted spatial transcriptomics data.
    fraction : float
        Float between (0, 1] to subsample data.

    Returns
    -------
    AnnData
        Returns subsampled view of original AnnData object.
    """
    keep = data.obs.groupby('cell').apply(
        lambda df: df.sample(frac=fraction)).index.droplevel(0)
    return data[keep, :]


def pca(data, n_components=2):
    """
    """
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data.uns['features'])
    columns = [str(c) for c in range(0, pca_components.shape[1])]
    pca_components = pd.DataFrame(pca_components,
                         index=data.uns['features'].index,
                         columns=columns)
    data.uns['pca_components'] = pca_components
    return data


def umap(data, n_components=2, n_neighbors=15, **kwargs):
    """
    """
    fit = u.UMAP(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
    umap_components = fit.fit_transform(data.uns['features'])
    columns = [str(c) for c in range(0, umap_components.shape[1])]
    umap_components = pd.DataFrame(umap_components,
                         index=data.uns['features'].index,
                         columns=columns)
    data.uns['umap_components'] = umap_components
    return data


def classify_genes(data, model, labels=None):
    # model fit predict return x y
    if labels:
        pred_labels = model.fit_predict(data['features'], labels)
    else:
        pred_labels = model.fit_predict(data['features'])
    return model, pred_labels


def prepare_features(data, features=[]):
    """Prepare features from raw data.

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
    features = cells.progress_apply(lambda cell: _prepare_cell_features(adata[adata.obs['cell'] == cell], features, cell))

    features = pd.concat(features.tolist(), keys=cells.tolist(), axis=0)
    adata.uns['features'] = features
    adata.uns['features'].index = pd.MultiIndex.from_tuples(features.index, names=('cell', 'gene'))
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
        fn = menu[feature]['function']

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

    # To dense image
    cell_img = np.zeros((output_size, output_size))
    cell_img = cv2.fillPoly(cell_img, [cell_xy], 1)

    ##### Nuclear mask
    nucleus_index = cell_data.uns['mask_index']['nucleus']
    n_index = nucleus_index[cell_data.uns['mask_index']['nucleus'] == cell].index[0]
    nucleus_xy = np.array(cell_data.uns['masks']['nucleus'].loc[n_index, 'geometry'].exterior.xy).reshape(2, -1).T
    nucleus_xy = (nucleus_xy - offset) * scale_factor + center_offset
    nucleus_xy = np.floor(nucleus_xy).astype(int)

    nuc_img = np.zeros((output_size, output_size))
    nuc_img = cv2.fillPoly(nuc_img, [nucleus_xy], 1)

    def _calc(gene_data, cell):
    ##### Points
        points = gene_data.X
        points = (points - offset) * scale_factor + center_offset
        points = np.floor(points).astype(int)

        pts_img = np.zeros((output_size, output_size))
        for coo in points:
            pts_img[coo[1], coo[0]] += 1

        image = np.stack([cell_img, nuc_img, pts_img]).astype(np.float32)
        # image = torch.from_numpy(image) # convert to Tensor
        return pd.Series(image, index=['raster'])

    return cell_data.obs.groupby('gene').apply(lambda obs: _calc(cell_data[obs.index, :], cell))


    

# Store feature names, descriptions, and respective functions.
# Format:
# { 'feature_name': {
#     'description': ''.
#     'function': fn,
#   }
# }
menu = dict({'ripley': {'description': 'ripley features', 'function': _calc_ripley_features},
             'distance': {'description': 'distance features', 'function': _calc_norm_distance_quantile_features},
             'morphology': {'description': 'morphology features', 'function': _calc_morph_enrichment},
             'mask_fraction': {'description': 'ripley features', 'function': _calc_nuclear_fraction},
             'indexes': {'description': 'index features', 'function': _calc_indexes},
             'raster': {'description': 'rasterize to image', 'function': _rasterize}})


def get_menu():
    """Return features on the menu.

    Returns
    -------
    DataFrame
        Formatted list of features.
    """
    menu_table = pd.DataFrame(menu)
    return menu_table
